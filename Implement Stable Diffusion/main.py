"""Run the guidance scale experiment with the manual Stable Diffusion pipeline."""

from __future__ import annotations

import config
from manual_sd import ManualStableDiffusion
from utils import (
    clear_device_cache,
    ensure_output_dir,
    guidance_scale_to_filename,
    make_image_grid,
    save_image,
    set_seed,
)


def main() -> None:
    set_seed(config.SEED)
    output_dir = ensure_output_dir(config.OUTPUT_DIR)
    pipeline = ManualStableDiffusion(
        model_id=config.MODEL_ID,
        device=config.DEVICE,
        dtype=config.DTYPE,
        scheduler_type=config.SCHEDULER_TYPE,
        use_safetensors=config.USE_SAFETENSORS,
    )

    images = []
    labels = []
    for guidance_scale in config.GUIDANCE_SCALES:
        print(f"Generating image with guidance_scale={guidance_scale}...")
        image = pipeline.generate(
            prompt=config.PROMPT,
            negative_prompt=config.NEGATIVE_PROMPT,
            guidance_scale=guidance_scale,
            seed=config.SEED,
            height=config.HEIGHT,
            width=config.WIDTH,
            num_inference_steps=config.NUM_INFERENCE_STEPS,
            eta=config.ETA,
            batch_size=config.BATCH_SIZE,
        )
        filename = f"guidance_scale_{guidance_scale_to_filename(guidance_scale)}.png"
        save_path = save_image(image, output_dir / filename)
        print(f"Saved {save_path}")

        images.append(image)
        labels.append(f"guidance scale = {guidance_scale}")
        clear_device_cache(config.DEVICE)

    grid_path = make_image_grid(images, labels, output_dir / "guidance_scale_grid.png")
    print(f"Saved {grid_path}")


if __name__ == "__main__":
    main()
