from __future__ import annotations

import shutil
import subprocess
import tempfile
from io import BytesIO
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch import nn
from torch.nn import functional as F

try:
    import cairosvg
except ImportError:
    cairosvg = None


TargetShape = Literal["disk", "ring", "diamond", "plus", "square"]


@dataclass
class NCAConfig:
    channel_n: int = 16
    hidden_n: int = 128
    fire_rate: float = 0.5
    target_size: int = 40
    target_padding: int = 16
    alpha_channel: int = 3
    living_threshold: float = 0.1

    @property
    def state_size(self) -> int:
        return self.target_size + (2 * self.target_padding)


def create_procedural_target(
    size: int,
    shape: TargetShape,
    fill_rgba: tuple[int, int, int, int] = (80, 210, 120, 255),
) -> torch.Tensor:
    image = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    inset = max(3, size // 6)
    center = size // 2

    if shape == "disk":
        draw.ellipse((inset, inset, size - inset, size - inset), fill=fill_rgba)
    elif shape == "ring":
        draw.ellipse((inset, inset, size - inset, size - inset), fill=fill_rgba)
        inner = inset + max(3, size // 8)
        draw.ellipse((inner, inner, size - inner, size - inner), fill=(0, 0, 0, 0))
    elif shape == "diamond":
        draw.polygon(
            [(center, inset), (size - inset, center), (center, size - inset), (inset, center)],
            fill=fill_rgba,
        )
    elif shape == "plus":
        arm = max(3, size // 8)
        draw.rectangle((center - arm, inset, center + arm, size - inset), fill=fill_rgba)
        draw.rectangle((inset, center - arm, size - inset, center + arm), fill=fill_rgba)
    elif shape == "square":
        draw.rectangle((inset, inset, size - inset, size - inset), fill=fill_rgba)
    else:
        raise ValueError(f"Unsupported target shape: {shape}")

    return pil_rgba_to_tensor(image)


def load_target_image(image_path: str | Path, size: int) -> torch.Tensor:
    image_path = Path(image_path)
    image = load_image_for_target(image_path)
    image.thumbnail((size, size), Image.Resampling.LANCZOS)

    canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    offset_x = (size - image.width) // 2
    offset_y = (size - image.height) // 2
    canvas.paste(image, (offset_x, offset_y), image)
    return pil_rgba_to_tensor(canvas)


def load_image_for_target(image_path: Path) -> Image.Image:
    if image_path.suffix.lower() != ".svg":
        return Image.open(image_path).convert("RGBA")

    if cairosvg is not None:
        png_bytes = cairosvg.svg2png(url=str(image_path))
        return Image.open(BytesIO(png_bytes)).convert("RGBA")

    qlmanage_path = shutil.which("qlmanage")
    if qlmanage_path is None:
        raise RuntimeError("SVG input requires qlmanage on this machine or a pre-rasterized PNG target.")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        command = [
            qlmanage_path,
            "-t",
            "-s",
            "1024",
            "-o",
            str(temp_path),
            str(image_path),
        ]
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise RuntimeError(
                f"Failed to rasterize SVG target with qlmanage.\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
            )

        png_candidates = sorted(temp_path.glob("*.png"))
        if not png_candidates:
            raise RuntimeError("qlmanage completed but did not produce a PNG for the SVG target.")

        return Image.open(png_candidates[0]).convert("RGBA")


def pil_rgba_to_tensor(image: Image.Image) -> torch.Tensor:
    arr = np.asarray(image, dtype=np.float32) / 255.0
    arr[..., :3] *= arr[..., 3:4]
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def rgba_to_rgb(x: torch.Tensor) -> torch.Tensor:
    rgb = x[:, :3]
    alpha = torch.clamp(x[:, 3:4], 0.0, 1.0)
    return torch.clamp(1.0 - alpha + rgb, 0.0, 1.0)


def pad_target(target_rgba: torch.Tensor, padding: int) -> torch.Tensor:
    return F.pad(target_rgba, (padding, padding, padding, padding))


def make_seed(batch_size: int, config: NCAConfig, device: torch.device) -> torch.Tensor:
    size = config.state_size
    seed = torch.zeros((batch_size, config.channel_n, size, size), device=device)
    center = size // 2
    seed[:, config.alpha_channel :, center, center] = 1.0
    return seed


class NeuralCellularAutomata(nn.Module):
    def __init__(self, config: NCAConfig):
        super().__init__()
        self.config = config

        self.update_net = nn.Sequential(
            nn.Conv2d(config.channel_n * 3, config.hidden_n, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(config.hidden_n, config.channel_n, kernel_size=1),
        )

        nn.init.zeros_(self.update_net[2].weight)
        nn.init.zeros_(self.update_net[2].bias)

        self.register_buffer("identity_kernel", torch.tensor([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]))
        self.register_buffer("dx_kernel", torch.tensor([[-0.125, 0.0, 0.125], [-0.25, 0.0, 0.25], [-0.125, 0.0, 0.125]]))
        self.register_buffer("dy_kernel", torch.tensor([[-0.125, -0.25, -0.125], [0.0, 0.0, 0.0], [0.125, 0.25, 0.125]]))

    def living_mask(self, x: torch.Tensor) -> torch.Tensor:
        alpha = x[:, self.config.alpha_channel : self.config.alpha_channel + 1]
        return F.max_pool2d(alpha, kernel_size=3, stride=1, padding=1) > self.config.living_threshold

    def perceive(self, x: torch.Tensor) -> torch.Tensor:
        base = torch.stack([self.identity_kernel, self.dx_kernel, self.dy_kernel], dim=0).to(x)
        kernels = base.unsqueeze(1).repeat(self.config.channel_n, 1, 1, 1)
        return F.conv2d(x, kernels, padding=1, groups=self.config.channel_n)

    def forward(self, x: torch.Tensor, step_size: float = 1.0) -> torch.Tensor:
        pre_life_mask = self.living_mask(x)
        perception = self.perceive(x)
        delta = self.update_net(perception) * step_size
        update_mask = (torch.rand_like(x[:, :1]) <= self.config.fire_rate).to(x.dtype)
        x = x + (delta * update_mask)
        post_life_mask = self.living_mask(x)
        life_mask = (pre_life_mask & post_life_mask).to(x.dtype)
        return x * life_mask
