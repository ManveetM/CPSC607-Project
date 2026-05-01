using System;
using Unity.Collections;
using UnityEngine;
using UnityEngine.InputSystem;

public class GridSimulation : MonoBehaviour
{
    private enum SimulationBackend
    {
        HandAuthoredRule,
        LearnedNcaCpu,
        LearnedNcaCompute,
    }

    [Header("Backend")]
    [SerializeField] private SimulationBackend simulationBackend = SimulationBackend.LearnedNcaCompute;
    [SerializeField] private TextAsset learnedModelJson;
    [SerializeField] private ComputeShader learnedNcaComputeShader;
    [SerializeField] private bool useModelRecommendedSize = true;
    [SerializeField] private bool autoSeedOnStart = true;
    [SerializeField] private bool stochasticUpdates = true;
    [SerializeField] private int randomSeed = 0;
    [SerializeField] private float learnedStepSize = 1.0f;

    [Header("Grid")]
    [SerializeField] private int width = 64;
    [SerializeField] private int height = 64;
    [SerializeField] private int channelCount = 4;
    [SerializeField] private float pixelsPerUnit = 10.0f;

    [Header("Simulation")]
    [SerializeField] private float updateInterval = 0.05f;
    [SerializeField] private bool runState = false;

    [Header("Hand-Authored Rule")]
    [SerializeField] private float diffusionRate = 0.20f;
    [SerializeField] private float decayRate = 0.05f;
    [SerializeField] private float growthRate = 0.15f;
    [SerializeField] private bool useHiddenChannelDriver = true;

    [Header("Painting")]
    [SerializeField] private float paintValue = 1.0f;
    [SerializeField] private bool allowPaintingWhileRunning = false;
    [SerializeField] private int paintRadius = 1;
    [SerializeField] private int eraseRadius = 3;

    private const int ComputeThreadGroupSize = 8;

    private Texture2D cpuTexture;
    private Texture2D spriteProxyTexture;
    private RenderTexture gpuDisplayTexture;
    private Material runtimeSpriteMaterial;
    private SpriteRenderer spriteRenderer;
    private float timer;
    private System.Random random;
    private int stepCounter;

    private float[,,] currentState;
    private float[,,] nextState;
    private bool[,] preLifeMask;
    private bool[,] postLifeMask;
    private float[] perceptionBuffer;
    private float[] hiddenBuffer;

    private ComputeBuffer gpuCurrentStateBuffer;
    private ComputeBuffer gpuNextStateBuffer;
    private ComputeBuffer gpuMaskedStateBuffer;
    private ComputeBuffer gpuConv1WeightBuffer;
    private ComputeBuffer gpuConv1BiasBuffer;
    private ComputeBuffer gpuConv2WeightBuffer;
    private ComputeBuffer gpuConv2BiasBuffer;
    private ComputeBuffer gpuDxKernelBuffer;
    private ComputeBuffer gpuDyKernelBuffer;

    private int stepKernel;
    private int applyLifeMaskKernel;
    private int renderKernel;
    private int clearKernel;
    private int paintKernel;

    private NcaUnityModel loadedModel;

    private void Start()
    {
        InitializeBackend();
        InitializeSimulation();

        if (autoSeedOnStart)
        {
            SeedCenter();
        }
        else
        {
            RenderState();
        }
    }

    private void Update()
    {
        if (!IsDisplayReady())
        {
            return;
        }

        HandleInput();

        if (runState)
        {
            TickSimulation();
        }
    }

    private void OnDestroy()
    {
        ReleaseGpuResources();

        if (runtimeSpriteMaterial != null)
        {
            Destroy(runtimeSpriteMaterial);
        }

        if (cpuTexture != null)
        {
            Destroy(cpuTexture);
        }

        if (spriteProxyTexture != null)
        {
            Destroy(spriteProxyTexture);
        }

        if (gpuDisplayTexture != null)
        {
            gpuDisplayTexture.Release();
            Destroy(gpuDisplayTexture);
        }
    }

    private void InitializeBackend()
    {
        random = randomSeed == 0 ? new System.Random() : new System.Random(randomSeed);
        stepCounter = 0;

        if (!UsesLearnedModel())
        {
            return;
        }

        if (learnedModelJson == null)
        {
            Debug.LogWarning("No learned model assigned. Falling back to hand-authored rule.");
            simulationBackend = SimulationBackend.HandAuthoredRule;
            return;
        }

        loadedModel = JsonUtility.FromJson<NcaUnityModel>(learnedModelJson.text);
        if (!IsValidModel(loadedModel))
        {
            Debug.LogWarning("Failed to parse learned NCA model. Falling back to hand-authored rule.");
            loadedModel = null;
            simulationBackend = SimulationBackend.HandAuthoredRule;
            return;
        }

        channelCount = loadedModel.channelN;
        if (useModelRecommendedSize && loadedModel.recommendedStateSize > 0)
        {
            width = loadedModel.recommendedStateSize;
            height = loadedModel.recommendedStateSize;
        }

        if (simulationBackend == SimulationBackend.LearnedNcaCompute)
        {
            if (!SystemInfo.supportsComputeShaders || learnedNcaComputeShader == null)
            {
                Debug.LogWarning("Compute shaders unavailable or not assigned. Falling back to CPU learned NCA.");
                simulationBackend = SimulationBackend.LearnedNcaCpu;
            }
        }
    }

    private void InitializeSimulation()
    {
        spriteRenderer = GetComponent<SpriteRenderer>();
        if (spriteRenderer == null)
        {
            throw new InvalidOperationException("GridSimulation requires a SpriteRenderer on the same GameObject.");
        }

        if (simulationBackend == SimulationBackend.LearnedNcaCompute && loadedModel != null)
        {
            InitializeComputeLearnedNca();
            return;
        }

        InitializeCpuSimulation();
    }

    private void InitializeCpuSimulation()
    {
        currentState = new float[width, height, channelCount];
        nextState = new float[width, height, channelCount];
        preLifeMask = new bool[width, height];
        postLifeMask = new bool[width, height];

        if (simulationBackend == SimulationBackend.LearnedNcaCpu && loadedModel != null)
        {
            perceptionBuffer = new float[loadedModel.perceptionFeatureCount];
            hiddenBuffer = new float[loadedModel.hiddenN];
        }

        cpuTexture = new Texture2D(width, height, TextureFormat.RGBA32, false)
        {
            filterMode = FilterMode.Point,
            wrapMode = TextureWrapMode.Clamp
        };

        ClearState();
        AssignSpriteTexture(cpuTexture);
    }

    private void InitializeComputeLearnedNca()
    {
        gpuCurrentStateBuffer = new ComputeBuffer(width * height * channelCount, sizeof(float));
        gpuNextStateBuffer = new ComputeBuffer(width * height * channelCount, sizeof(float));
        gpuMaskedStateBuffer = new ComputeBuffer(width * height * channelCount, sizeof(float));
        gpuConv1WeightBuffer = new ComputeBuffer(loadedModel.conv1Weight.Length, sizeof(float));
        gpuConv1BiasBuffer = new ComputeBuffer(loadedModel.conv1Bias.Length, sizeof(float));
        gpuConv2WeightBuffer = new ComputeBuffer(loadedModel.conv2Weight.Length, sizeof(float));
        gpuConv2BiasBuffer = new ComputeBuffer(loadedModel.conv2Bias.Length, sizeof(float));
        gpuDxKernelBuffer = new ComputeBuffer(loadedModel.dxKernel.Length, sizeof(float));
        gpuDyKernelBuffer = new ComputeBuffer(loadedModel.dyKernel.Length, sizeof(float));

        gpuConv1WeightBuffer.SetData(loadedModel.conv1Weight);
        gpuConv1BiasBuffer.SetData(loadedModel.conv1Bias);
        gpuConv2WeightBuffer.SetData(loadedModel.conv2Weight);
        gpuConv2BiasBuffer.SetData(loadedModel.conv2Bias);
        gpuDxKernelBuffer.SetData(loadedModel.dxKernel);
        gpuDyKernelBuffer.SetData(loadedModel.dyKernel);

        stepKernel = learnedNcaComputeShader.FindKernel("StepNca");
        applyLifeMaskKernel = learnedNcaComputeShader.FindKernel("ApplyLifeMask");
        renderKernel = learnedNcaComputeShader.FindKernel("RenderState");
        clearKernel = learnedNcaComputeShader.FindKernel("ClearState");
        paintKernel = learnedNcaComputeShader.FindKernel("PaintBrush");

        gpuDisplayTexture = new RenderTexture(width, height, 0, RenderTextureFormat.ARGB32)
        {
            enableRandomWrite = true,
            filterMode = FilterMode.Point,
            wrapMode = TextureWrapMode.Clamp
        };
        gpuDisplayTexture.Create();

        spriteProxyTexture = new Texture2D(width, height, TextureFormat.RGBA32, false);
        Color32[] pixels = new Color32[width * height];
        for (int i = 0; i < pixels.Length; i++)
        {
            pixels[i] = new Color32(255, 255, 255, 255);
        }

        spriteProxyTexture.SetPixels32(pixels);
        spriteProxyTexture.Apply();

        Shader displayShader = Shader.Find("Unlit/NcaDisplay");
        if (displayShader == null)
        {
            Debug.LogWarning("Unlit/NcaDisplay shader not found. Falling back to CPU learned NCA.");
            simulationBackend = SimulationBackend.LearnedNcaCpu;
            ReleaseGpuResources();
            if (gpuDisplayTexture != null)
            {
                gpuDisplayTexture.Release();
                Destroy(gpuDisplayTexture);
                gpuDisplayTexture = null;
            }
            if (spriteProxyTexture != null)
            {
                Destroy(spriteProxyTexture);
                spriteProxyTexture = null;
            }
            InitializeCpuSimulation();
            return;
        }

        runtimeSpriteMaterial = new Material(displayShader);
        runtimeSpriteMaterial.SetTexture("_SimulationTex", gpuDisplayTexture);

        Sprite sprite = Sprite.Create(
            spriteProxyTexture,
            new Rect(0, 0, width, height),
            new Vector2(0.5f, 0.5f),
            pixelsPerUnit
        );

        spriteRenderer.sprite = sprite;
        spriteRenderer.material = runtimeSpriteMaterial;

        ConfigureSharedComputeParameters();
        ClearState();
    }

    private void ConfigureSharedComputeParameters()
    {
        learnedNcaComputeShader.SetInt("_Width", width);
        learnedNcaComputeShader.SetInt("_Height", height);
        learnedNcaComputeShader.SetInt("_ChannelCount", channelCount);
        learnedNcaComputeShader.SetInt("_HiddenN", loadedModel.hiddenN);
        learnedNcaComputeShader.SetInt("_PerceptionFeatureCount", loadedModel.perceptionFeatureCount);
        learnedNcaComputeShader.SetFloat("_FireRate", loadedModel.fireRate);
        learnedNcaComputeShader.SetInt("_AlphaChannel", loadedModel.alphaChannel);
        learnedNcaComputeShader.SetFloat("_LivingThreshold", loadedModel.livingThreshold);
        learnedNcaComputeShader.SetFloat("_LearnedStepSize", learnedStepSize);
        learnedNcaComputeShader.SetInt("_UseStochasticUpdates", stochasticUpdates ? 1 : 0);
        learnedNcaComputeShader.SetBuffer(stepKernel, "_Conv1Weight", gpuConv1WeightBuffer);
        learnedNcaComputeShader.SetBuffer(stepKernel, "_Conv1Bias", gpuConv1BiasBuffer);
        learnedNcaComputeShader.SetBuffer(stepKernel, "_Conv2Weight", gpuConv2WeightBuffer);
        learnedNcaComputeShader.SetBuffer(stepKernel, "_Conv2Bias", gpuConv2BiasBuffer);
        learnedNcaComputeShader.SetBuffer(stepKernel, "_DxKernel", gpuDxKernelBuffer);
        learnedNcaComputeShader.SetBuffer(stepKernel, "_DyKernel", gpuDyKernelBuffer);
        learnedNcaComputeShader.SetBuffer(applyLifeMaskKernel, "_CurrentState", gpuCurrentStateBuffer);
        learnedNcaComputeShader.SetBuffer(applyLifeMaskKernel, "_NextState", gpuNextStateBuffer);
        learnedNcaComputeShader.SetBuffer(applyLifeMaskKernel, "_MaskedState", gpuMaskedStateBuffer);
        learnedNcaComputeShader.SetBuffer(renderKernel, "_CurrentState", gpuCurrentStateBuffer);
        learnedNcaComputeShader.SetTexture(renderKernel, "_Result", gpuDisplayTexture);
        learnedNcaComputeShader.SetBuffer(clearKernel, "_State", gpuCurrentStateBuffer);
        learnedNcaComputeShader.SetBuffer(paintKernel, "_State", gpuCurrentStateBuffer);
    }

    private void ReleaseGpuResources()
    {
        ReleaseBuffer(ref gpuCurrentStateBuffer);
        ReleaseBuffer(ref gpuNextStateBuffer);
        ReleaseBuffer(ref gpuMaskedStateBuffer);
        ReleaseBuffer(ref gpuConv1WeightBuffer);
        ReleaseBuffer(ref gpuConv1BiasBuffer);
        ReleaseBuffer(ref gpuConv2WeightBuffer);
        ReleaseBuffer(ref gpuConv2BiasBuffer);
        ReleaseBuffer(ref gpuDxKernelBuffer);
        ReleaseBuffer(ref gpuDyKernelBuffer);
    }

    private void ReleaseBuffer(ref ComputeBuffer buffer)
    {
        if (buffer == null)
        {
            return;
        }

        buffer.Release();
        buffer = null;
    }

    private bool IsDisplayReady()
    {
        return simulationBackend == SimulationBackend.LearnedNcaCompute ? gpuDisplayTexture != null : cpuTexture != null;
    }

    private bool UsesLearnedModel()
    {
        return simulationBackend == SimulationBackend.LearnedNcaCpu || simulationBackend == SimulationBackend.LearnedNcaCompute;
    }

    private void AssignSpriteTexture(Texture2D texture)
    {
        Sprite sprite = Sprite.Create(
            texture,
            new Rect(0, 0, width, height),
            new Vector2(0.5f, 0.5f),
            pixelsPerUnit
        );

        spriteRenderer.sprite = sprite;
    }

    private void HandleInput()
    {
        if (runState && !allowPaintingWhileRunning)
        {
            return;
        }

        Mouse mouse = Mouse.current;
        if (mouse == null)
        {
            return;
        }

        bool leftPressed = mouse.leftButton.isPressed;
        bool rightPressed = mouse.rightButton.isPressed;

        if (!leftPressed && !rightPressed)
        {
            return;
        }

        if (!TryGetCellFromMouse(out int x, out int y))
        {
            return;
        }

        if (leftPressed)
        {
            PaintCellsOn(x, y, Mathf.Max(0, paintRadius));
            RenderState();
        }
        else if (rightPressed)
        {
            PaintCellsOff(x, y, Mathf.Max(0, eraseRadius));
            RenderState();
        }
    }

    private void TickSimulation()
    {
        timer += Time.deltaTime;

        while (timer >= updateInterval)
        {
            timer -= updateInterval;
            StepAutomaton();
            RenderState();
        }
    }

    private void StepAutomaton()
    {
        switch (simulationBackend)
        {
            case SimulationBackend.LearnedNcaCompute:
                StepLearnedNcaCompute();
                break;
            case SimulationBackend.LearnedNcaCpu:
                StepLearnedNcaCpu();
                break;
            default:
                StepHandAuthoredRule();
                break;
        }
    }

    private void StepHandAuthoredRule()
    {
        for (int x = 0; x < width; x++)
        {
            for (int y = 0; y < height; y++)
            {
                UpdateHandAuthoredCell(x, y);
            }
        }

        SwapBuffers();
        ClearNextState();
    }

    private void UpdateHandAuthoredCell(int x, int y)
    {
        if (useHiddenChannelDriver && channelCount >= 4)
        {
            float selfEnergy = currentState[x, y, 3];
            float avgEnergy = GetNeighborAverage(currentState, x, y, 3);
            float diffusion = diffusionRate * (avgEnergy - selfEnergy);
            float growth = growthRate * avgEnergy;
            float decay = decayRate * selfEnergy;
            float nextEnergy = Mathf.Clamp01(selfEnergy + diffusion + growth - decay);

            nextState[x, y, 3] = nextEnergy;
            nextState[x, y, 0] = nextEnergy;
            nextState[x, y, 1] = nextEnergy * 0.7f;
            nextState[x, y, 2] = 1.0f - nextEnergy * 0.4f;
            return;
        }

        for (int c = 0; c < channelCount; c++)
        {
            float self = currentState[x, y, c];
            float avg = GetNeighborAverage(currentState, x, y, c);
            float diffusion = diffusionRate * (avg - self);
            float growth = growthRate * avg;
            float decay = decayRate * self;
            nextState[x, y, c] = Mathf.Clamp01(self + diffusion + growth - decay);
        }
    }

    private void StepLearnedNcaCpu()
    {
        ComputeLivingMask(currentState, preLifeMask);

        for (int x = 0; x < width; x++)
        {
            for (int y = 0; y < height; y++)
            {
                ComputePerception(x, y);

                for (int h = 0; h < loadedModel.hiddenN; h++)
                {
                    int weightOffset = h * loadedModel.perceptionFeatureCount;
                    float hiddenValue = loadedModel.conv1Bias[h];
                    for (int f = 0; f < loadedModel.perceptionFeatureCount; f++)
                    {
                        hiddenValue += loadedModel.conv1Weight[weightOffset + f] * perceptionBuffer[f];
                    }

                    hiddenBuffer[h] = Mathf.Max(0f, hiddenValue);
                }

                bool applyUpdate = !stochasticUpdates || random.NextDouble() <= loadedModel.fireRate;
                for (int c = 0; c < channelCount; c++)
                {
                    float delta = 0f;
                    if (applyUpdate)
                    {
                        int weightOffset = c * loadedModel.hiddenN;
                        delta = loadedModel.conv2Bias[c];
                        for (int h = 0; h < loadedModel.hiddenN; h++)
                        {
                            delta += loadedModel.conv2Weight[weightOffset + h] * hiddenBuffer[h];
                        }
                    }

                    nextState[x, y, c] = currentState[x, y, c] + (delta * learnedStepSize);
                }
            }
        }

        ComputeLivingMask(nextState, postLifeMask);
        ApplyLifeMaskCpu();
        SwapBuffers();
        ClearNextState();
        stepCounter++;
    }

    private void StepLearnedNcaCompute()
    {
        learnedNcaComputeShader.SetFloat("_LearnedStepSize", learnedStepSize);
        learnedNcaComputeShader.SetInt("_UseStochasticUpdates", stochasticUpdates ? 1 : 0);
        learnedNcaComputeShader.SetInt("_StepIndex", stepCounter);
        learnedNcaComputeShader.SetBuffer(stepKernel, "_CurrentState", gpuCurrentStateBuffer);
        learnedNcaComputeShader.SetBuffer(stepKernel, "_NextState", gpuNextStateBuffer);
        Dispatch2D(stepKernel);

        Dispatch2D(applyLifeMaskKernel);

        ComputeBuffer temp = gpuCurrentStateBuffer;
        gpuCurrentStateBuffer = gpuMaskedStateBuffer;
        gpuMaskedStateBuffer = temp;

        learnedNcaComputeShader.SetBuffer(applyLifeMaskKernel, "_CurrentState", gpuCurrentStateBuffer);
        learnedNcaComputeShader.SetBuffer(applyLifeMaskKernel, "_NextState", gpuNextStateBuffer);
        learnedNcaComputeShader.SetBuffer(applyLifeMaskKernel, "_MaskedState", gpuMaskedStateBuffer);
        learnedNcaComputeShader.SetBuffer(renderKernel, "_CurrentState", gpuCurrentStateBuffer);

        stepCounter++;
    }

    private void ApplyLifeMaskCpu()
    {
        for (int x = 0; x < width; x++)
        {
            for (int y = 0; y < height; y++)
            {
                if (preLifeMask[x, y] && postLifeMask[x, y])
                {
                    continue;
                }

                for (int c = 0; c < channelCount; c++)
                {
                    nextState[x, y, c] = 0f;
                }
            }
        }
    }

    private void ComputePerception(int x, int y)
    {
        for (int c = 0; c < channelCount; c++)
        {
            float identity = SampleState(currentState, x, y, c);
            float dx = 0f;
            float dy = 0f;

            for (int ky = 0; ky < 3; ky++)
            {
                for (int kx = 0; kx < 3; kx++)
                {
                    int kernelIndex = (ky * 3) + kx;
                    float sample = SampleState(currentState, x + kx - 1, y + ky - 1, c);
                    dx += sample * loadedModel.dxKernel[kernelIndex];
                    dy += sample * loadedModel.dyKernel[kernelIndex];
                }
            }

            int featureOffset = c * 3;
            perceptionBuffer[featureOffset] = identity;
            perceptionBuffer[featureOffset + 1] = dx;
            perceptionBuffer[featureOffset + 2] = dy;
        }
    }

    private void ComputeLivingMask(float[,,] state, bool[,] mask)
    {
        int alphaChannel = GetAlphaChannel();
        float threshold = GetLivingThreshold();

        for (int x = 0; x < width; x++)
        {
            for (int y = 0; y < height; y++)
            {
                float maxAlpha = 0f;
                for (int ox = -1; ox <= 1; ox++)
                {
                    for (int oy = -1; oy <= 1; oy++)
                    {
                        float alpha = SampleState(state, x + ox, y + oy, alphaChannel);
                        if (alpha > maxAlpha)
                        {
                            maxAlpha = alpha;
                        }
                    }
                }

                mask[x, y] = maxAlpha > threshold;
            }
        }
    }

    private float GetNeighborAverage(float[,,] state, int x, int y, int channel)
    {
        float sum = 0f;
        int count = 0;

        for (int ox = -1; ox <= 1; ox++)
        {
            for (int oy = -1; oy <= 1; oy++)
            {
                int nx = x + ox;
                int ny = y + oy;
                if (nx < 0 || nx >= width || ny < 0 || ny >= height)
                {
                    continue;
                }

                sum += state[nx, ny, channel];
                count++;
            }
        }

        return count > 0 ? sum / count : 0f;
    }

    private float SampleState(float[,,] state, int x, int y, int channel)
    {
        if (x < 0 || x >= width || y < 0 || y >= height || channel < 0 || channel >= channelCount)
        {
            return 0f;
        }

        return state[x, y, channel];
    }

    private void SwapBuffers()
    {
        float[,,] temp = currentState;
        currentState = nextState;
        nextState = temp;
    }

    private void RenderState()
    {
        if (simulationBackend == SimulationBackend.LearnedNcaCompute)
        {
            Dispatch2D(renderKernel);
            return;
        }

        NativeArray<Color32> pixelData = cpuTexture.GetRawTextureData<Color32>();
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                int index = ((height - 1 - y) * width) + x;
                pixelData[index] = UsesLearnedModel() ? GetLearnedModelColorCpu(x, y) : GetHandAuthoredColor(x, y);
            }
        }

        cpuTexture.Apply();
    }

    private void Dispatch2D(int kernel)
    {
        int groupsX = Mathf.CeilToInt(width / (float)ComputeThreadGroupSize);
        int groupsY = Mathf.CeilToInt(height / (float)ComputeThreadGroupSize);
        learnedNcaComputeShader.Dispatch(kernel, groupsX, groupsY, 1);
    }

    private Color32 GetHandAuthoredColor(int x, int y)
    {
        byte r = FloatToByte(SampleState(currentState, x, y, 0));
        byte g = FloatToByte(SampleState(currentState, x, y, 1));
        byte b = FloatToByte(SampleState(currentState, x, y, 2));
        return new Color32(r, g, b, 255);
    }

    private Color32 GetLearnedModelColorCpu(int x, int y)
    {
        float alpha = Mathf.Clamp01(SampleState(currentState, x, y, loadedModel.alphaChannel));
        float r = Mathf.Clamp01(1f - alpha + SampleState(currentState, x, y, 0));
        float g = Mathf.Clamp01(1f - alpha + SampleState(currentState, x, y, 1));
        float b = Mathf.Clamp01(1f - alpha + SampleState(currentState, x, y, 2));
        return new Color32(FloatToByte(r), FloatToByte(g), FloatToByte(b), 255);
    }

    private byte FloatToByte(float value)
    {
        return (byte)Mathf.RoundToInt(Mathf.Clamp01(value) * 255f);
    }

    private bool TryGetCellFromMouse(out int x, out int y)
    {
        x = -1;
        y = -1;

        Mouse mouse = Mouse.current;
        if (mouse == null || Camera.main == null)
        {
            return false;
        }

        Vector2 screenPos = mouse.position.ReadValue();
        Vector3 worldPos = Camera.main.ScreenToWorldPoint(new Vector3(screenPos.x, screenPos.y, 10f));
        Vector3 localPos = transform.InverseTransformPoint(worldPos);

        x = Mathf.FloorToInt(localPos.x * pixelsPerUnit + (width / 2f));
        int displayY = Mathf.FloorToInt(localPos.y * pixelsPerUnit + (height / 2f));
        y = height - 1 - displayY;

        return x >= 0 && x < width && y >= 0 && y < height;
    }

    private void PaintCellsOn(int centerX, int centerY, int radius)
    {
        ApplyBrush(centerX, centerY, radius, true);
    }

    private void PaintCellsOff(int centerX, int centerY, int radius)
    {
        ApplyBrush(centerX, centerY, radius, false);
    }

    private void ApplyBrush(int centerX, int centerY, int radius, bool paintOn)
    {
        if (simulationBackend == SimulationBackend.LearnedNcaCompute)
        {
            learnedNcaComputeShader.SetBuffer(paintKernel, "_State", gpuCurrentStateBuffer);
            learnedNcaComputeShader.SetInt("_BrushCenterX", centerX);
            learnedNcaComputeShader.SetInt("_BrushCenterY", centerY);
            learnedNcaComputeShader.SetInt("_BrushRadius", radius);
            learnedNcaComputeShader.SetInt("_BrushMode", paintOn ? 1 : 0);
            learnedNcaComputeShader.SetFloat("_PaintValue", paintValue);
            Dispatch2D(paintKernel);
            return;
        }

        for (int x = centerX - radius; x <= centerX + radius; x++)
        {
            for (int y = centerY - radius; y <= centerY + radius; y++)
            {
                if (x < 0 || x >= width || y < 0 || y >= height)
                {
                    continue;
                }

                int dx = x - centerX;
                int dy = y - centerY;
                if ((dx * dx) + (dy * dy) > radius * radius)
                {
                    continue;
                }

                if (paintOn)
                {
                    PaintCellOnCpu(x, y);
                }
                else
                {
                    PaintCellOffCpu(x, y);
                }
            }
        }
    }

    private void PaintCellOnCpu(int x, int y)
    {
        if (UsesLearnedModel())
        {
            for (int c = 0; c < channelCount; c++)
            {
                currentState[x, y, c] = 0f;
            }

            for (int c = loadedModel.alphaChannel; c < channelCount; c++)
            {
                currentState[x, y, c] = paintValue;
            }

            return;
        }

        if (channelCount >= 4)
        {
            currentState[x, y, 0] = paintValue;
            currentState[x, y, 1] = paintValue;
            currentState[x, y, 2] = paintValue;
            currentState[x, y, 3] = paintValue;
            return;
        }

        for (int c = 0; c < channelCount; c++)
        {
            currentState[x, y, c] = paintValue;
        }
    }

    private void PaintCellOffCpu(int x, int y)
    {
        for (int c = 0; c < channelCount; c++)
        {
            currentState[x, y, c] = 0f;
        }
    }

    [ContextMenu("Clear State")]
    public void ClearState()
    {
        stepCounter = 0;

        if (simulationBackend == SimulationBackend.LearnedNcaCompute)
        {
            learnedNcaComputeShader.SetBuffer(clearKernel, "_State", gpuCurrentStateBuffer);
            Dispatch2D(clearKernel);
            learnedNcaComputeShader.SetBuffer(clearKernel, "_State", gpuNextStateBuffer);
            Dispatch2D(clearKernel);
            learnedNcaComputeShader.SetBuffer(clearKernel, "_State", gpuMaskedStateBuffer);
            Dispatch2D(clearKernel);
            RenderState();
            return;
        }

        if (currentState == null)
        {
            return;
        }

        Array.Clear(currentState, 0, currentState.Length);
        if (nextState != null)
        {
            Array.Clear(nextState, 0, nextState.Length);
        }

        RenderState();
    }

    private void ClearNextState()
    {
        if (nextState == null)
        {
            return;
        }

        Array.Clear(nextState, 0, nextState.Length);
    }

    [ContextMenu("Seed Center")]
    public void SeedCenter()
    {
        int cx = width / 2;
        int cy = height / 2;
        PaintCellsOn(cx, cy, 0);
        RenderState();
    }

    [ContextMenu("Seed Small Cross")]
    public void SeedSmallCross()
    {
        int cx = width / 2;
        int cy = height / 2;

        PaintCellsOn(cx, cy, 0);
        if (cx > 0) PaintCellsOn(cx - 1, cy, 0);
        if (cx < width - 1) PaintCellsOn(cx + 1, cy, 0);
        if (cy > 0) PaintCellsOn(cx, cy - 1, 0);
        if (cy < height - 1) PaintCellsOn(cx, cy + 1, 0);

        RenderState();
    }

    public void ToggleRunState()
    {
        runState = !runState;
    }

    private int GetAlphaChannel()
    {
        return UsesLearnedModel() ? loadedModel.alphaChannel : Mathf.Min(3, channelCount - 1);
    }

    private float GetLivingThreshold()
    {
        return UsesLearnedModel() ? loadedModel.livingThreshold : 0.1f;
    }

    private static bool IsValidModel(NcaUnityModel model)
    {
        if (model == null)
        {
            return false;
        }

        if (model.channelN <= 0 || model.hiddenN <= 0 || model.perceptionFeatureCount != model.channelN * 3)
        {
            return false;
        }

        return model.dxKernel != null
            && model.dxKernel.Length == 9
            && model.dyKernel != null
            && model.dyKernel.Length == 9
            && model.conv1Bias != null
            && model.conv1Bias.Length == model.hiddenN
            && model.conv2Bias != null
            && model.conv2Bias.Length == model.channelN
            && model.conv1Weight != null
            && model.conv1Weight.Length == model.hiddenN * model.perceptionFeatureCount
            && model.conv2Weight != null
            && model.conv2Weight.Length == model.channelN * model.hiddenN;
    }
}

[Serializable]
public class NcaUnityModel
{
    public string runName;
    public int targetSize;
    public int targetPadding;
    public int recommendedStateSize;
    public int channelN;
    public int hiddenN;
    public int perceptionFeatureCount;
    public float fireRate;
    public int alphaChannel;
    public float livingThreshold;
    public float[] identityKernel;
    public float[] dxKernel;
    public float[] dyKernel;
    public float[] conv1Weight;
    public float[] conv1Bias;
    public float[] conv2Weight;
    public float[] conv2Bias;
}
