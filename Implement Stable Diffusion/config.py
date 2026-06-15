"""Experiment configuration for manual Stable Diffusion inference."""

from __future__ import annotations

import os

import torch


def detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


MODEL_ID = os.getenv("SD_MODEL_ID", "runwayml/stable-diffusion-v1-5")

PROMPT = os.getenv(
    "SD_PROMPT",
    "A cute corgi wearing sunglasses on the beach, high quality, detailed",
)
NEGATIVE_PROMPT = os.getenv("SD_NEGATIVE_PROMPT", "")

GUIDANCE_SCALES = [1, 3, 5, 7.5, 12, 20]
COMPARE_GUIDANCE_SCALE = 7.5

SEED = int(os.getenv("SD_SEED", "42"))
HEIGHT = int(os.getenv("SD_HEIGHT", "512"))
WIDTH = int(os.getenv("SD_WIDTH", "512"))
NUM_INFERENCE_STEPS = int(os.getenv("SD_NUM_INFERENCE_STEPS", "25"))
BATCH_SIZE = 1

SCHEDULER_TYPE = "ddim"
ETA = 0.0

DEVICE = os.getenv("SD_DEVICE", detect_device())
DTYPE = os.getenv("SD_DTYPE", "float16" if DEVICE in {"cuda", "mps"} else "float32")

OUTPUT_DIR = "outputs"
USE_SAFETENSORS = True
