# Assignment 2.4 — Manual Stable Diffusion Inference

## Project Goal

This project implements the inference process of Stable Diffusion manually using Hugging Face Diffusers components.

The main purpose is to understand the internal workflow of Stable Diffusion instead of only calling a high-level pipeline. The project manually connects the core components of Stable Diffusion, including:

```text
CLIP Tokenizer
CLIP Text Encoder
UNet
DDIM Scheduler
VAE Decoder
```

This project focuses only on **Assignment 2.4: Stable Diffusion inference**.


---

## Project Scope and Non-goals

This is an **inference-only** project. The model components are loaded from pretrained checkpoints and used to generate images.

The project does **not** include:

```text
training
fine-tuning
LoRA training or loading
updating model weights
batch generation for all guidance scales at once
multi-prompt generation
API server
web UI
production optimization
```

The goal is correctness, readability, and understanding the Stable Diffusion inference workflow. Speed and memory optimization are secondary.

---

## Main Requirement

Do **not** use the high-level Diffusers pipeline as the main implementation.

The following code is not allowed for the main manual inference part:

```python
pipe = StableDiffusionPipeline.from_pretrained(...)
image = pipe(prompt)
```

However, the official `StableDiffusionPipeline` may be used later only for comparison with the manually implemented pipeline.

The core implementation must manually perform:

```text
prompt encoding
unconditional prompt encoding
latent noise initialization
denoising loop
classifier-free guidance
DDIM scheduler step
VAE decoding
image postprocessing
```

---

## Allowed Components

The project may load pretrained components from Hugging Face Diffusers and Transformers.

Allowed examples:

```python
from transformers import CLIPTokenizer, CLIPTextModel
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler

tokenizer = CLIPTokenizer.from_pretrained(...)
text_encoder = CLIPTextModel.from_pretrained(...)
unet = UNet2DConditionModel.from_pretrained(...)
vae = AutoencoderKL.from_pretrained(...)
scheduler = DDIMScheduler.from_pretrained(...)
```

The important rule is that these components must be manually connected together. The project should build its own custom Stable Diffusion inference pipeline.

---

## Model Access Note

Some Hugging Face checkpoints, such as:

```python
MODEL_ID = "runwayml/stable-diffusion-v1-5"
```

may require accepting the model license or logging in with a Hugging Face token before they can be downloaded.

If model loading fails because of access permissions, log in with the Hugging Face CLI or provide a valid token according to the Hugging Face documentation.

The model weights are only loaded for inference. They must not be trained, fine-tuned, or modified by this project.

---

## Suggested Project Structure

```text
assignment_2_4/
│
├── README.md
├── requirements.txt
├── config.py
├── main.py
├── manual_sd.py
├── compare_pipeline.py
├── utils.py
│
├── outputs/
│   ├── guidance_scale_1.png
│   ├── guidance_scale_3.png
│   ├── guidance_scale_5.png
│   ├── guidance_scale_7_5.png
│   ├── guidance_scale_12.png
│   ├── guidance_scale_20.png
│   ├── guidance_scale_grid.png
│   ├── manual_pipeline.png
│   ├── diffusers_pipeline.png
│   └── comparison_grid.png
│
└── report.md
```

---

## File Responsibilities

### `README.md`

This file explains the assignment requirements, project scope, implementation rules, and expected outputs.

It is mainly used so that both the developer and coding agent understand what needs to be implemented.

---

### `requirements.txt`

This file stores the required Python libraries.

Suggested dependencies:

```text
torch
diffusers
transformers
accelerate
safetensors
Pillow
matplotlib
numpy
```

Install dependencies with:

```bash
pip install -r requirements.txt
```

---

### `config.py`

This file stores all experiment settings.

Do not hard-code prompt, model id, image size, seed, guidance scales, or inference steps inside `main.py` or `manual_sd.py`.

When changing the experiment setup, edit `config.py`.

Example:

```python
MODEL_ID = "runwayml/stable-diffusion-v1-5"

PROMPT = "A cute corgi wearing sunglasses on the beach, high quality, detailed"
NEGATIVE_PROMPT = ""

GUIDANCE_SCALES = [1, 3, 5, 7.5, 12, 20]

SEED = 42
HEIGHT = 512
WIDTH = 512
NUM_INFERENCE_STEPS = 25

SCHEDULER_TYPE = "ddim"
ETA = 0.0

DEVICE = "mps"
DTYPE = "float16"
BATCH_SIZE = 1

OUTPUT_DIR = "outputs"
```

The purpose of `config.py` is to make the project easy to modify without touching the core pipeline code.

For this project, the default device should be compatible with the local machine. Prefer automatic device selection in code:

```python
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"
```

On the target MacBook Pro M4 Pro, the expected device is usually:

```python
DEVICE = "mps"
```

Recommended default settings for this machine:

```python
HEIGHT = 512
WIDTH = 512
NUM_INFERENCE_STEPS = 25
BATCH_SIZE = 1
DTYPE = "float16"
```

If MPS has errors, unstable output, or black images, switch to:

```python
DTYPE = "float32"
```

Do not hard-code these values inside `main.py` or `manual_sd.py`.

---

### `manual_sd.py`

This is the most important file in the project.

It should contain the custom manual Stable Diffusion inference implementation.

This file should handle:

```text
load CLIP tokenizer
load CLIP text encoder
load UNet
load VAE
load DDIM scheduler
encode text prompt
encode null prompt
initialize latent noise
run manual denoising loop
apply Classifier-Free Guidance manually
perform scheduler step manually
decode final latent using VAE decoder
postprocess image tensor into PIL image
```

Suggested class structure:

```python
class ManualStableDiffusion:
    def __init__(self, model_id, device, dtype):
        ...

    def generate(
        self,
        prompt,
        negative_prompt,
        guidance_scale,
        seed,
        height,
        width,
        num_inference_steps,
    ):
        ...
```

The main inference logic must be inside this file.

---

### `main.py`

This file runs the guidance scale experiment.

It should:

```text
read settings from config.py
create the manual Stable Diffusion pipeline
loop through guidance scales [1, 3, 5, 7.5, 12, 20]
generate one image for each guidance scale
save each generated image
create one grid image comparing all guidance scales
```

This file should not contain the full Stable Diffusion logic. It should only control the experiment.

---

### `compare_pipeline.py`

This file compares the manual pipeline with the official Diffusers pipeline.

The official high-level pipeline is allowed only in this file for comparison.

It should generate:

```text
one image using the manual pipeline
one image using StableDiffusionPipeline.from_pretrained(...)
one comparison grid
```

The comparison should use the same:

```text
checkpoint
prompt
negative prompt
guidance scale
random seed
image size
number of inference steps
scheduler type if possible
```

Suggested comparison guidance scale:

```python
guidance_scale = 7.5
```

Expected outputs:

```text
outputs/manual_pipeline.png
outputs/diffusers_pipeline.png
outputs/comparison_grid.png
```

---

### `utils.py`

This file contains helper functions.

Suggested helper functions:

```python
def set_seed(seed):
    ...

def get_torch_dtype(dtype_name):
    ...

def save_image(image, path):
    ...

def make_image_grid(images, labels, save_path):
    ...

def guidance_scale_to_filename(scale):
    ...
```

The purpose of `utils.py` is to keep `main.py` and `manual_sd.py` clean and readable.

---

### `outputs/`

This folder stores all generated images.

Expected outputs from the guidance scale experiment:

```text
outputs/guidance_scale_1.png
outputs/guidance_scale_3.png
outputs/guidance_scale_5.png
outputs/guidance_scale_7_5.png
outputs/guidance_scale_12.png
outputs/guidance_scale_20.png
outputs/guidance_scale_grid.png
```

Expected outputs from pipeline comparison:

```text
outputs/manual_pipeline.png
outputs/diffusers_pipeline.png
outputs/comparison_grid.png
```

---

### `report.md`

This file contains a short written analysis of the experiment.

It should include:

```text
checkpoint used
prompt used
negative prompt used
image size
seed
number of inference steps
guidance scales tested
observation for each guidance scale
identified sweet spot
comparison between manual pipeline and Diffusers pipeline
possible causes of differences
```

The report does not need to be very long. It should clearly show that the guidance scale experiment was performed and understood.

---

## Required Manual Inference Flow

The manual Stable Diffusion inference process must follow this flow:

```text
1. Encode the text prompt using CLIP tokenizer and CLIP text encoder
2. Encode the null prompt "" or negative prompt to obtain unconditional embeddings
3. Initialize random latent noise
4. Set scheduler timesteps
5. Run the denoising loop over scheduler timesteps
6. At each timestep:
   - Predict conditional noise using prompt embeddings
   - Predict unconditional noise using null prompt embeddings
   - Apply Classifier-Free Guidance
   - Perform one DDIM scheduler step
7. Decode the final latent using the VAE decoder
8. Convert the decoded tensor into a normal image
```

---

## Stable Diffusion Concept Summary

Stable Diffusion does not denoise directly in pixel space. It performs denoising in latent space.

A normal RGB image may have shape:

```python
(3, 512, 512)
```

After VAE compression, it becomes a smaller latent tensor:

```python
(4, 64, 64)
```

For batch size 1, the latent shape is usually:

```python
(1, 4, 64, 64)
```

This is because the VAE downsamples the spatial dimensions by a factor of 8.

So for a 512 x 512 image:

```text
512 / 8 = 64
```

The latent tensor is not a normal image that humans can directly view. It is a compressed representation of an image.

The UNet denoises this latent tensor step by step. At the end, the VAE decoder converts the final latent tensor back into an RGB image.

This is why Stable Diffusion is faster than diffusion models that work directly on full-resolution pixel images.

---

## Classifier-Free Guidance

Classifier-Free Guidance, or CFG, controls how strongly the generated image follows the text prompt.

At each denoising step, the UNet predicts two noise tensors:

```text
cond_noise:
noise prediction using the actual text prompt

uncond_noise:
noise prediction using the null prompt "" or negative prompt
```

Then the final guided noise prediction is computed as:

```python
noise_pred = uncond_noise + guidance_scale * (cond_noise - uncond_noise)
```

Meaning of `guidance_scale`:

```text
low guidance scale:
more random, more diverse, less prompt-aligned

medium guidance scale:
good balance between image quality and prompt alignment

very high guidance scale:
over-guided image, possible artifacts, over-saturation, distorted details
```

---

## Guidance Scale Experiment

The project must generate images using the same prompt but different guidance scale values:

```python
GUIDANCE_SCALES = [1, 3, 5, 7.5, 12, 20]
```

To make the comparison fair, all other settings must remain fixed:

```text
same checkpoint
same prompt
same negative prompt
same random seed
same image size
same number of inference steps
same scheduler
```

Only `guidance_scale` should change.

The final output should include a grid image comparing all guidance scales side by side.

Expected observation:

```text
Guidance scale = 1:
The image is usually more random and may not follow the prompt well.

Guidance scale = 3:
The image starts to follow the prompt better but may still be weak in detail.

Guidance scale = 5:
The image usually becomes clearer and more aligned with the prompt.

Guidance scale = 7.5:
This is often a good balance between image quality and prompt fidelity.

Guidance scale = 12:
The image may start to become over-guided. Colors may become stronger and details may look less natural.

Guidance scale = 20:
The image may show clear degradation, such as artifacts, distorted structure, over-saturation, or unnatural details.
```

The expected sweet spot is usually around:

```text
5 to 8
```

However, the final conclusion must be based on the actual generated images.

---

## Important Details from the Official Diffusers Pipeline

The official Diffusers Stable Diffusion pipeline should be used only as a reference.

Do not copy the entire pipeline. It contains many advanced features that are not necessary for this assignment, such as:

```text
safety checker
IP-Adapter
LoRA loading
textual inversion
callbacks
CPU offload
extra attention processors
multiple prompt formats
```

For this assignment, only the core inference logic is needed.

Important implementation details to follow:

---

### 0. Inference Runtime Rules

The manual pipeline should run in inference mode.

After loading pretrained modules, call `.eval()` on:

```text
text_encoder
unet
vae
```

Image generation should run inside `torch.no_grad()` or `torch.inference_mode()` so gradients are not tracked.

The manual implementation only needs `BATCH_SIZE = 1`. Generate guidance scale outputs one by one instead of batching all guidance scales together. This keeps the memory requirement lower on MPS.

`height` and `width` must be validated before creating latents.

---

### 1. Validate Image Size

`height` and `width` must be divisible by 8.

This is required because the VAE downsamples the image by a factor of 8.

Example check:

```python
if height % 8 != 0 or width % 8 != 0:
    raise ValueError("height and width must be divisible by 8")
```

---

### 2. Set Scheduler Timesteps

Before the denoising loop, call:

```python
scheduler.set_timesteps(num_inference_steps, device=device)
timesteps = scheduler.timesteps
```

This ensures that the scheduler uses the correct number of inference steps.

---

### 3. Prepare Latent Noise with Correct Shape

For Stable Diffusion v1.x, the latent tensor usually has 4 channels.

Example:

```python
vae_scale_factor = 8

latent_shape = (
    batch_size,
    unet.config.in_channels,
    height // vae_scale_factor,
    width // vae_scale_factor,
)

latents = torch.randn(
    latent_shape,
    generator=generator,
    device=device,
    dtype=dtype,
)
```

For a 512 x 512 output image, the latent shape is:

```python
(1, 4, 64, 64)
```

---

### 4. Scale Initial Latent Noise

The initial latent noise should be multiplied by the scheduler's initial noise sigma:

```python
latents = latents * scheduler.init_noise_sigma
```

This makes the manual pipeline closer to the official Diffusers behavior.

---

### 5. Scale Model Input Before UNet

Before passing latents into the UNet, use:

```python
latent_model_input = scheduler.scale_model_input(latents, t)
```

Some schedulers require this scaling depending on the timestep.

---

### 6. Use Two UNet Forward Passes for This Assignment

The assignment explicitly requires two forward passes:

```text
one forward pass for conditional noise
one forward pass for unconditional noise
```

Required implementation style:

```python
cond_noise = unet(
    latent_model_input,
    t,
    encoder_hidden_states=cond_embeddings,
).sample

uncond_noise = unet(
    latent_model_input,
    t,
    encoder_hidden_states=uncond_embeddings,
).sample

noise_pred = uncond_noise + guidance_scale * (cond_noise - uncond_noise)
```

The official Diffusers pipeline often optimizes this by concatenating unconditional and conditional embeddings into one batch and calling the UNet once. That optimization is useful, but the main implementation for this assignment should use two separate UNet calls to match the assignment requirement.

An optional optimized version may be added later, but it is not required.

---

### 7. Update Latents with the Scheduler

After calculating the guided noise prediction, update the latents:

```python
latents = scheduler.step(
    noise_pred,
    t,
    latents,
).prev_sample
```

This performs one reverse diffusion step from the current noisy latent to a slightly less noisy latent.

---

### 8. Decode Latents with VAE Scaling Factor

Before decoding, divide by the VAE scaling factor:

```python
latents_for_decode = latents / vae.config.scaling_factor
image = vae.decode(latents_for_decode).sample
```

This is an important detail. Forgetting this step can make the output image incorrect.

---

### 9. Postprocess Image Tensor

The VAE decoder output is usually in the range `[-1, 1]`.

Convert it to `[0, 1]`:

```python
image = (image / 2 + 0.5).clamp(0, 1)
```

Then convert the tensor to a PIL image and save it.

---

## Manual Denoising Loop Pseudocode

The core loop in `manual_sd.py` should look like this:

```python
scheduler.set_timesteps(num_inference_steps, device=device)

for t in scheduler.timesteps:
    latent_model_input = scheduler.scale_model_input(latents, t)

    cond_noise = unet(
        latent_model_input,
        t,
        encoder_hidden_states=cond_embeddings,
    ).sample

    uncond_noise = unet(
        latent_model_input,
        t,
        encoder_hidden_states=uncond_embeddings,
    ).sample

    noise_pred = uncond_noise + guidance_scale * (cond_noise - uncond_noise)

    latents = scheduler.step(
        noise_pred,
        t,
        latents,
    ).prev_sample
```

This is the required core logic of Assignment 2.4.

---

## Prompt Encoding Pseudocode

The prompt should be encoded using CLIP tokenizer and CLIP text encoder.

Example:

```python
text_input = tokenizer(
    prompt,
    padding="max_length",
    max_length=tokenizer.model_max_length,
    truncation=True,
    return_tensors="pt",
)

cond_embeddings = text_encoder(
    text_input.input_ids.to(device)
)[0]
```

The unconditional or negative prompt should be encoded similarly:

```python
uncond_input = tokenizer(
    negative_prompt,
    padding="max_length",
    max_length=tokenizer.model_max_length,
    truncation=True,
    return_tensors="pt",
)

uncond_embeddings = text_encoder(
    uncond_input.input_ids.to(device)
)[0]
```

If `NEGATIVE_PROMPT = ""`, then this is equivalent to using a null prompt.

---

## Image Decoding Pseudocode

After the denoising loop ends:

```python
latents_for_decode = latents / vae.config.scaling_factor

image = vae.decode(latents_for_decode).sample

image = (image / 2 + 0.5).clamp(0, 1)
image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
image = (image * 255).round().astype("uint8")
```

Then convert the NumPy array into a PIL image.

---

## Comparison with Official Diffusers Pipeline

After the manual pipeline works, compare its output with the official Diffusers pipeline.

This comparison is allowed only as an evaluation step.

Example:

```python
from diffusers import StableDiffusionPipeline
```

The comparison should use the same:

```text
checkpoint
prompt
negative prompt
guidance scale
seed
height
width
number of inference steps
scheduler if possible
dtype
device
```

Suggested comparison value:

```python
guidance_scale = 7.5
```

The generated images may not be perfectly identical, but they should be similar if the manual implementation is correct.

Expected comparison behavior:

```text
Manual pipeline and official Diffusers pipeline outputs do not need to be pixel-identical.

They should be visually similar when checkpoint, prompt, negative prompt, seed, scheduler,
guidance scale, image size, dtype, and device are matched as closely as possible.

Small differences may come from scheduler configuration, dtype precision, MPS behavior,
or implementation details inside the official Diffusers pipeline.
```

---

## Hardware Note

This project is developed on:

```text
MacBook Pro M4 Pro
RAM: 24GB unified memory
SSD: 512GB
Backend: Apple Silicon GPU via PyTorch MPS
```

The code should not assume CUDA is available. Device selection should support `cuda`, `mps`, and `cpu`.

Recommended device selection:

```python
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"
```

For this machine, the expected device is usually:

```python
DEVICE = "mps"
```

Recommended default settings:

```python
HEIGHT = 512
WIDTH = 512
NUM_INFERENCE_STEPS = 25
BATCH_SIZE = 1
DTYPE = "float16"
```

Generate images one by one instead of batching all guidance scales together. If MPS has errors, unstable output, or black images, switch to:

```python
DTYPE = "float32"
```

The code should prioritize correctness and readability first, then optimize speed and memory usage.

---

## Acceptance Checklist

Before considering the assignment complete, verify that:

```text
main.py runs successfully
all guidance scale images are generated
outputs/guidance_scale_grid.png is generated
compare_pipeline.py runs successfully
outputs/manual_pipeline.png is generated
outputs/diffusers_pipeline.png is generated
outputs/comparison_grid.png is generated
report.md records the checkpoint, prompt, negative prompt, image size, seed, steps, and guidance scales
report.md includes observations for each guidance scale
report.md identifies a guidance scale sweet spot based on the generated images
report.md compares the manual pipeline result with the official Diffusers pipeline result
StableDiffusionPipeline is not used inside the main manual inference implementation
```




