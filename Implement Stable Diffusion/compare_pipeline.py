"""Compare manual Stable Diffusion inference with the official Diffusers pipeline."""

from __future__ import annotations

import torch
from diffusers import DDIMScheduler, StableDiffusionPipeline

import config
from manual_sd import ManualStableDiffusion
from utils import (
    create_generator,
    ensure_output_dir,
    get_torch_dtype,
    make_image_grid,
    save_image,
    set_seed,
)


def load_diffusers_pipeline() -> StableDiffusionPipeline:
    dtype = get_torch_dtype(config.DTYPE, config.DEVICE)
    pipe = StableDiffusionPipeline.from_pretrained(
        config.MODEL_ID,
        torch_dtype=dtype,
        use_safetensors=config.USE_SAFETENSORS,
    )
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(config.DEVICE)
    pipe.set_progress_bar_config(disable=False)
    return pipe


def main() -> None:
    set_seed(config.SEED)
    output_dir = ensure_output_dir(config.OUTPUT_DIR)

    manual_pipeline = ManualStableDiffusion(
        model_id=config.MODEL_ID,
        device=config.DEVICE,
        dtype=config.DTYPE,
        scheduler_type=config.SCHEDULER_TYPE,
        use_safetensors=config.USE_SAFETENSORS,
    )
    manual_image = manual_pipeline.generate(
        prompt=config.PROMPT,
        negative_prompt=config.NEGATIVE_PROMPT,
        guidance_scale=config.COMPARE_GUIDANCE_SCALE,
        seed=config.SEED,
        height=config.HEIGHT,
        width=config.WIDTH,
        num_inference_steps=config.NUM_INFERENCE_STEPS,
        eta=config.ETA,
        batch_size=config.BATCH_SIZE,
    )
    manual_path = save_image(manual_image, output_dir / "manual_pipeline.png")
    print(f"Saved {manual_path}")

    diffusers_pipeline = load_diffusers_pipeline()
    generator = create_generator(config.SEED, config.DEVICE)
    with torch.inference_mode():
        diffusers_image = diffusers_pipeline(
            prompt=config.PROMPT,
            negative_prompt=config.NEGATIVE_PROMPT,
            guidance_scale=config.COMPARE_GUIDANCE_SCALE,
            generator=generator,
            height=config.HEIGHT,
            width=config.WIDTH,
            num_inference_steps=config.NUM_INFERENCE_STEPS,
            eta=config.ETA,
        ).images[0]

    diffusers_path = save_image(diffusers_image, output_dir / "diffusers_pipeline.png")
    print(f"Saved {diffusers_path}")

    grid_path = make_image_grid(
        [manual_image, diffusers_image],
        ["manual pipeline", "diffusers pipeline"],
        output_dir / "comparison_grid.png",
        columns=2,
    )
    print(f"Saved {grid_path}")


if __name__ == "__main__":
    main()
