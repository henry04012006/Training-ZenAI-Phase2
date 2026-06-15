"""Utility functions for the manual Stable Diffusion assignment."""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image, ImageDraw


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_torch_dtype(dtype_name: str, device: str | torch.device) -> torch.dtype:
    device_type = torch.device(device).type
    normalized_name = dtype_name.lower()

    if device_type == "cpu" and normalized_name in {"float16", "fp16", "bfloat16", "bf16"}:
        return torch.float32
    if normalized_name in {"float16", "fp16"}:
        return torch.float16
    if normalized_name in {"float32", "fp32"}:
        return torch.float32
    if normalized_name in {"bfloat16", "bf16"}:
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def create_generator(seed: int, device: str | torch.device) -> torch.Generator:
    device_type = torch.device(device).type
    generator_device = "cpu" if device_type == "mps" else device_type
    return torch.Generator(device=generator_device).manual_seed(seed)


def ensure_output_dir(output_dir: str | Path) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def save_image(image: Image.Image, path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    return output_path


def make_image_grid(
    images: Sequence[Image.Image],
    labels: Sequence[str],
    save_path: str | Path,
    columns: int = 3,
) -> Path:
    if not images:
        raise ValueError("At least one image is required to make a grid.")
    if len(images) != len(labels):
        raise ValueError("The number of images and labels must match.")

    padding = 16
    label_height = 32
    columns = min(columns, len(images))
    rows = math.ceil(len(images) / columns)
    cell_width = max(image.width for image in images)
    cell_height = max(image.height for image in images) + label_height

    grid_width = columns * cell_width + (columns + 1) * padding
    grid_height = rows * cell_height + (rows + 1) * padding
    grid = Image.new("RGB", (grid_width, grid_height), "white")
    draw = ImageDraw.Draw(grid)

    for index, (image, label) in enumerate(zip(images, labels)):
        row, column = divmod(index, columns)
        cell_x = padding + column * (cell_width + padding)
        cell_y = padding + row * (cell_height + padding)
        image_x = cell_x + (cell_width - image.width) // 2
        grid.paste(image.convert("RGB"), (image_x, cell_y + label_height))
        draw.text((cell_x, cell_y), label, fill="black")

    return save_image(grid, save_path)


def guidance_scale_to_filename(scale: float) -> str:
    return str(scale).replace(".", "_")


def clear_device_cache(device: str | torch.device) -> None:
    device_type = torch.device(device).type
    if device_type == "cuda":
        torch.cuda.empty_cache()
    elif device_type == "mps" and hasattr(torch, "mps"):
        torch.mps.empty_cache()
