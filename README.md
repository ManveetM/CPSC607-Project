# Growing Neural Cellular Automata (2D)
  
Author: Manveet Mandal  
Project for CPSC 607. First the README will explain the project meta details and how to set things up.
Next there will be some embedded [visualizations/showcases](#results) of the project.

## What this project includes

- Final project [report](Report.pdf)
- 2D GNCA training in PyTorch
- growth-only and regeneration training
- two damage modes for regeneration:
  - `circle`
  - `dropout`
- fixed-horizon evaluation and `best_eval` checkpointing
- Unity export for interactive visualization
- CPU and compute-shader inference paths in Unity

## Main files

- [train_nca.py](./src/python/train_nca.py): training, evaluation, export
- [nca_model.py](./src/python/nca_model.py): GNCA model definition
- [unity_export.py](./src/python/unity_export.py): Unity-friendly export helper
- [export_unity_model.py](./src/python/export_unity_model.py): convert existing runs to Unity JSON
- [GridSimulation.cs](./src/unity/GridSimulation.cs): Unity runtime script
- [NcaCompute.compute](./src/unity/NcaCompute.compute): Unity compute-shader backend
- [NcaDisplay.shader](./src/unity/NcaDisplay.shader): Unity display shader

## Setup

Create a virtual environment and install dependencies:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Targets can be:
- SVG
- PNG

Put them anywhere accessible and pass them with `--target-image`.

## Run training

### Growth-only example

```bash
.venv/bin/python train_nca.py \
  --train-steps 10000 \
  --target-image images/lizard_emoji.svg \
  --target-size 64 \
  --pool-size 1024 \
  --batch-size 8 \
  --min-iter 64 \
  --max-iter 96 \
  --rollout-steps 96 \
  --learning-rate 0.002 \
  --lr-decay-step 3000 \
  --lr-decay-factor 0.1 \
  --output-dir outputs/lizard_growth_64
```

### Regeneration with circle damage

```bash
.venv/bin/python train_nca.py \
  --train-steps 10000 \
  --target-image images/lizard_emoji.svg \
  --target-size 64 \
  --pool-size 1024 \
  --batch-size 8 \
  --min-iter 64 \
  --max-iter 96 \
  --damage-n 3 \
  --damage-mode circle \
  --eval-every 250 \
  --eval-horizons 96 200 400 \
  --eval-rollouts 4 \
  --learning-rate 0.002 \
  --lr-decay-step 3000 \
  --lr-decay-factor 0.1 \
  --output-dir outputs/lizard_regen_64
```

### Regeneration with dropout damage

```bash
.venv/bin/python train_nca.py \
  --train-steps 10000 \
  --target-image images/lizard_emoji.svg \
  --target-size 64 \
  --pool-size 1024 \
  --batch-size 8 \
  --min-iter 64 \
  --max-iter 96 \
  --damage-n 3 \
  --damage-mode dropout \
  --dropout-p-min 0.05 \
  --dropout-p-max 0.30 \
  --eval-every 250 \
  --eval-horizons 96 200 400 \
  --eval-rollouts 4 \
  --learning-rate 0.002 \
  --lr-decay-step 3000 \
  --lr-decay-factor 0.1 \
  --output-dir outputs/lizard_regen_dropout_64
```

## Output structure

Each run directory in `outputs/...` contains:

- `config.json`: run settings
- `loss.csv`: training loss log
- `eval.csv`: fixed-horizon evaluation log
- `best_eval_metrics.json`: best evaluation summary
- `target.png`: resized target
- `final_growth.gif`: rollout from seed
- `weights.json`: raw model weights
- `unity_model.json`: Unity-friendly flattened export
- `checkpoints/`: periodic checkpoints and `best_eval.pt`
- `previews/`: sampled growth previews

## Unity usage

1. Import these files into your Unity project:
   - `GridSimulation.cs`
   - `NcaCompute.compute`
   - `NcaDisplay.shader`
   - one `unity_model.json` file from `outputs/...`
2. Add a `SpriteRenderer` to a GameObject.
3. Attach `GridSimulation`.
4. Assign:
   - `Learned Model Json` = chosen `unity_model.json`
   - `Learned Nca Compute Shader` = `NcaCompute.compute`
5. Set backend to:
   - `LearnedNcaCompute` for GPU
   - `LearnedNcaCpu` for fallback/debugging
6. Press Play.

Notes:
- `Use Model Recommended Size` will override width/height from the export.
- Right click erases cells; left click seeds/paints.
- If live editing while running is blocked, enable `Allow Painting While Running`.

## Existing runs

There are already multiple runs in `outputs/`, including growth, circle regeneration, and dropout regeneration experiments for several targets.

## Visualizations

### Some Examples of Growth

****Lizard****  
![](outputs/lizard_regen_dropout_64/final_growth.gif)

****Hands****  
![](outputs/hands_regen_dropout_64/final_growth.gif)

****Tiger****  
![](outputs/tiger_regen_dropout_64/final_growth.gif)

****Satellite****  
![](outputs/satellite_regen_dropout_64/final_growth.gif)

****Flag****  
![](outputs/flag_regen_dropout_64/final_growth.gif)

****Vista****  
![](outputs/vista_regen_dropout_64/final_growth.gif)

### Examples of Regeneration

****Lizard****  
https://github.com/user-attachments/assets/aaaa8286-64f6-4e64-8a46-2dc19b69c153

****Vista****  
https://github.com/user-attachments/assets/83519927-6ff7-497a-b26b-0d9e46af1c77

****Tiger****  
https://github.com/user-attachments/assets/2c6536df-8c93-43c4-8674-0951652fe38f
