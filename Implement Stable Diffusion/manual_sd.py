"""Manual Stable Diffusion inference built from Diffusers components."""

from __future__ import annotations

import inspect

import numpy as np
import torch
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
from PIL import Image
from transformers import CLIPTextModel, CLIPTokenizer

from utils import create_generator, get_torch_dtype


class ManualStableDiffusion:
    def __init__(
        self,
        model_id: str,
        device: str,
        dtype: str,
        scheduler_type: str = "ddim",
        use_safetensors: bool = True,
    ) -> None:
        if scheduler_type.lower() != "ddim":
            raise ValueError("This assignment implementation supports only DDIMScheduler.")

        self.model_id = model_id
        self.device = torch.device(device)
        self.dtype = get_torch_dtype(dtype, self.device)

        self.tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(
            model_id,
            subfolder="text_encoder",
            torch_dtype=self.dtype,
            use_safetensors=use_safetensors,
        ).to(self.device)
        self.unet = UNet2DConditionModel.from_pretrained(
            model_id,
            subfolder="unet",
            torch_dtype=self.dtype,
            use_safetensors=use_safetensors,
        ).to(self.device)
        self.vae = AutoencoderKL.from_pretrained(
            model_id,
            subfolder="vae",
            torch_dtype=self.dtype,
            use_safetensors=use_safetensors,
        ).to(self.device)
        self.scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")

        self.text_encoder.eval()
        self.unet.eval()
        self.vae.eval()

    def generate(
        self,
        prompt: str,
        negative_prompt: str,
        guidance_scale: float,
        seed: int,
        height: int,
        width: int,
        num_inference_steps: int,
        eta: float = 0.0,
        batch_size: int = 1,
    ) -> Image.Image:
        self._validate_inputs(height, width, batch_size)

        with torch.inference_mode():
            cond_embeddings = self._encode_prompt(prompt)
            uncond_embeddings = self._encode_prompt(negative_prompt)
            latents = self._prepare_latents(seed, height, width, batch_size)

            self.scheduler.set_timesteps(num_inference_steps, device=self.device)
            for timestep in self.scheduler.timesteps:
                latent_model_input = self.scheduler.scale_model_input(latents, timestep)
                cond_noise = self._predict_noise(latent_model_input, timestep, cond_embeddings)
                uncond_noise = self._predict_noise(latent_model_input, timestep, uncond_embeddings)
                noise_pred = uncond_noise + guidance_scale * (cond_noise - uncond_noise)
                latents = self._step_scheduler(noise_pred, timestep, latents, eta)

            return self._decode_latents(latents)

    @staticmethod
    def _validate_inputs(height: int, width: int, batch_size: int) -> None:
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError("height and width must be divisible by 8")
        if batch_size != 1:
            raise ValueError("This assignment implementation supports BATCH_SIZE = 1 only.")

    def _encode_prompt(self, prompt: str) -> torch.Tensor:
        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = text_input.input_ids.to(self.device)
        return self.text_encoder(input_ids)[0]

    def _prepare_latents(
        self,
        seed: int,
        height: int,
        width: int,
        batch_size: int,
    ) -> torch.Tensor:
        generator = create_generator(seed, self.device)
        latent_shape = (
            batch_size,
            self.unet.config.in_channels,
            height // 8,
            width // 8,
        )

        if self.device.type == "mps":
            latents = torch.randn(latent_shape, generator=generator, dtype=torch.float32)
            latents = latents.to(device=self.device, dtype=self.dtype)
        else:
            latents = torch.randn(
                latent_shape,
                generator=generator,
                device=self.device,
                dtype=self.dtype,
            )
        return latents * self.scheduler.init_noise_sigma

    def _predict_noise(
        self,
        latent_model_input: torch.Tensor,
        timestep: torch.Tensor,
        embeddings: torch.Tensor,
    ) -> torch.Tensor:
        return self.unet(
            latent_model_input,
            timestep,
            encoder_hidden_states=embeddings,
        ).sample

    def _step_scheduler(
        self,
        noise_pred: torch.Tensor,
        timestep: torch.Tensor,
        latents: torch.Tensor,
        eta: float,
    ) -> torch.Tensor:
        step_kwargs = {}
        if "eta" in inspect.signature(self.scheduler.step).parameters:
            step_kwargs["eta"] = eta
        return self.scheduler.step(noise_pred, timestep, latents, **step_kwargs).prev_sample

    def _decode_latents(self, latents: torch.Tensor) -> Image.Image:
        scaling_factor = getattr(self.vae.config, "scaling_factor", 0.18215)
        image = self.vae.decode(latents / scaling_factor).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.detach().float().cpu().permute(0, 2, 3, 1).numpy()
        image = (image[0] * 255).round().astype(np.uint8)
        return Image.fromarray(image)
