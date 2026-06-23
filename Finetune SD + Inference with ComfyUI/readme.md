# Fine-tune Stable Diffusion with LoRA and Inference using ComfyUI

## Overview

Assignment 2.5 focuses on fine-tuning Stable Diffusion for a specific subject using LoRA, then performing inference with the trained model using both Diffusers and ComfyUI.

The main goal of this assignment is to understand how to adapt Stable Diffusion to a concrete use case through lightweight fine-tuning, and to become familiar with using ComfyUI as a node-based inference workflow tool.

---

## Main Objectives

The objectives of this assignment are:

```text
1. Fine-tune Stable Diffusion on a small personal image dataset using LoRA.
2. Use the trained LoRA model to generate images of the target subject.
3. Evaluate the generated results using CLIP-based metrics.
4. Perform inference with the trained model using Diffusers.
5. Design an inference workflow for the trained model using ComfyUI.
```

---

## Project Folder Structure

The project should be notebook-first so training is easy to follow and run step
by step. The `Itay/` folder is the original dataset and must not be modified
directly. Generated files should be written to `outputs/`.

```text
Finetune SD + Inference with ComfyUI/
├── Itay/
├── train_itay_lora.ipynb
├── workflows/
│   └── itayperson_comfyui.json
├── outputs/
│   ├── prepared/
│   ├── lora/
│   ├── samples/
│   └── metrics/
├── config.env
├── requirements.txt
├── readme.md
└── .gitignore
```

### Folder Meaning

```text
Itay/                  Original Itay dataset.
train_itay_lora.ipynb  Main notebook for data preparation, LoRA training, inference, and CLIP evaluation.
workflows/             ComfyUI workflow files.
outputs/prepared/      Processed training images and captions.
outputs/lora/          Trained LoRA weights.
outputs/samples/       Generated images for inspection and evaluation.
outputs/metrics/       CLIP-I and CLIP-T metric outputs.
config.env             Shared model id, paths, and baseline hyperparameters used by the notebook.
```

The base model does not need to live inside this project. The notebook should
load it through `SD_MODEL_ID`, with this default:

```python
SD_MODEL_ID = "runwayml/stable-diffusion-v1-5"
```

If the model is already cached by Hugging Face, Diffusers will reuse the local
cache automatically. A local snapshot path can also be used directly:

```text
/Users/tun/.cache/huggingface/hub/models--runwayml--stable-diffusion-v1-5/snapshots/451f4fe16113bff5a5d2269ed5ad43b0592e9a14
```

An optional `models/` folder may be added later only if a local checkpoint needs
to be copied into the project.

---

## Project Pipeline

The project is organized into three main stages:

```text
1. Prepare data and fine-tune LoRA
   - Open train_itay_lora.ipynb.
   - Read images from Itay/.
   - Fix EXIF orientation and strip metadata.
   - Crop/resize training images.
   - Create captions using the trigger token itayperson.
   - Train a low-rank LoRA on top of runwayml/stable-diffusion-v1-5 or another SD 1.5 Diffusers model passed through SD_MODEL_ID.
   - Save the trained LoRA to outputs/lora/.

2. Evaluate generated results
   - Use the same notebook to generate test images with the trained LoRA.
   - Save generated images to outputs/samples/.
   - Compute CLIP-I for subject fidelity.
   - Compute CLIP-T for prompt fidelity.
   - Save metric results to outputs/metrics/.

3. Run inference with Diffusers and ComfyUI
   - Use the notebook's Diffusers inference section to load the base model and LoRA.
   - Use a ComfyUI workflow to load the checkpoint, apply the LoRA, sample, decode, and save images.
   - The ComfyUI workflow should include at least one helper node or improvement path, such as an upscaler, Face Detailer, ControlNet, LCM sampler, or another quality/speed/flexibility node.
```

The current implementation provides Stage 1 and Stage 2 in
`train_itay_lora.ipynb`. Stage 3 ComfyUI workflow design is planned after the
LoRA baseline has been trained and evaluated.

---

## Stage 1 Training Context from Diffusers

The Hugging Face Diffusers advanced diffusion training reference focuses on
DreamBooth LoRA for SDXL, but several ideas are useful for this project. This
project adapts those ideas to a simpler SD 1.5, local, notebook-first workflow.

### Techniques to Apply

```text
1. DreamBooth-style subject training
   - Treat Itay as the target subject.
   - Use a unique trigger token: itayperson.
   - Captions and prompts should consistently include this token.

2. LoRA instead of full fine-tuning
   - Keep the base Stable Diffusion model frozen.
   - Train only low-rank adapter weights.
   - Save the result as a portable LoRA file in outputs/lora/.

3. Custom captions
   - Create a caption for each processed image.
   - Baseline caption can be simple: photo of itayperson.
   - If an image has a very specific background, outfit, or pose, add a short description so the model does not confuse identity with context.

4. Notebook-friendly Accelerate setup
   - Use accelerate configuration inside the notebook instead of relying on an interactive terminal setup.
   - Keep the notebook runnable step by step on the local machine.

5. Validation during training
   - Define a small fixed validation prompt set before training.
   - Generate sample images during or immediately after training.
   - Use the same prompts later for visual comparison and CLIP evaluation.

6. Conservative baseline hyperparameters
   - Start with rank 8.
   - Use batch size 1.
   - Use 512x512 resolution for SD 1.5.
   - Start with around 1000 training steps, then tune only if the first result is underfit or overfit.
```

### Techniques to Keep Optional

```text
1. Text encoder training
   - Useful when the trigger token is weak or identity is not learned well.
   - Not recommended for the first baseline because it increases memory use and overfitting risk.

2. Pivotal tuning / textual inversion
   - More advanced subject-token learning.
   - Not needed for the first pass.

3. Prodigy optimizer
   - Can reduce learning-rate tuning.
   - Optional; the first pass can use AdamW for simplicity.

4. Min-SNR gamma
   - Can improve diffusion training stability.
   - Optional after the baseline works.

5. DoRA / B-LoRA / targeted U-Net blocks
   - Interesting advanced methods.
   - Out of scope for the first LoRA baseline.

6. W&B logging, push to Hugging Face Hub, cloud upload
   - Not needed for this local/private project.
```

### Required Before Training

```text
1. Install notebook dependencies:
   pip install -r requirements.txt

2. Choose the base model:
   SD_MODEL_ID=runwayml/stable-diffusion-v1-5

3. Confirm the base model is in Diffusers format or loadable by Diffusers:
   - unet/
   - vae/
   - text_encoder/
   - tokenizer/
   - scheduler/
   - model_index.json

   The already cached model can be reused; it does not need to be downloaded again
   if Diffusers can load it from the Hugging Face cache.

4. Run train_itay_lora.ipynb from top to bottom:
   - environment and dependency check
   - path/config setup
   - dataset preparation
   - caption creation/review
   - LoRA training
   - validation image generation
   - CLIP-I and CLIP-T evaluation

5. Keep all generated files under outputs/.
```

---

## Dataset
For this project, the dataset in the Itay folder will be used.
---

## Fine-tuning Requirement

Stable Diffusion must be fine-tuned using LoRA.
The model should be trained on approximately:
```text
23 personal images
```

The purpose of LoRA fine-tuning is to allow the model to learn the visual identity or appearance of the target subject while keeping training lightweight.

The training process should aim to make the generated images as similar as possible to the subject in the Itay dataset.

---

## Reference Materials

The following resources are suggested as references:

```text
Kohya Scripts
Diffusers
```

The assignment also provides a reference comparison page:

```text
[Comparison] Astria, Kohya, Headshot, PhotoAI
```

This reference should be used to understand the expected quality of the generated images, especially in terms of subject similarity, image realism, and prompt controllability.

---

## Expected Training Goal

The trained LoRA model should be able to generate images that preserve the identity or appearance of the target subject from the Itay dataset.

The generated images should satisfy two important criteria:

```text
1. Subject fidelity:
   The generated image should look similar to the target subject.

2. Prompt fidelity:
   The generated image should follow the content and style described in the text prompt.
```

The goal is not only to generate visually pleasing images, but also to ensure that the fine-tuned model can preserve the subject while still responding to different prompts.

---

## Evaluation Metrics

The trained model must be evaluated using the following metrics:

```text
CLIP-I
CLIP-T
```

### CLIP-I

CLIP-I measures subject fidelity.

It evaluates how visually similar the generated images are to the target subject images.

A higher CLIP-I score indicates that the generated image better preserves the identity or appearance of the target subject.

### CLIP-T

CLIP-T measures prompt fidelity.

It evaluates how well the generated images match the text prompt.

A higher CLIP-T score indicates that the generated image follows the prompt more accurately.

---

## Inference Requirement

After fine-tuning, the trained LoRA model must be used for inference in two ways:

```text
1. Inference using Diffusers
2. Inference using ComfyUI
```

### Diffusers Inference

The trained LoRA model should be loaded and used with Hugging Face Diffusers to generate images from text prompts.

This step verifies that the trained model can be used programmatically in a Python-based inference pipeline.

### ComfyUI Inference

The trained LoRA model should also be used inside ComfyUI.

A ComfyUI workflow must be designed for inference using the trained model.

The workflow should be able to load the base Stable Diffusion model, apply the trained LoRA, encode prompts, generate images, decode outputs, and save final images.

---

## ComfyUI Requirement

The assignment requires using ComfyUI to design an inference workflow for the trained model.

The purpose of this part is to become familiar with node-based Stable Diffusion inference.

The workflow should demonstrate that the trained LoRA model can be integrated and used inside ComfyUI.

Additional ComfyUI nodes may be used to improve:

```text
image quality
inference speed
workflow flexibility
```

---

## Bonus Requirement

As a bonus, additional adapter-based methods may be explored and compared with the LoRA training result.

The purpose of this bonus part is to understand how LoRA compares with other adapter-based approaches for adapting Stable Diffusion to a specific subject.

This part is optional and should be treated as an extension beyond the main LoRA requirement.

---

## Final Expected Outcome

The final result of Assignment 2.5 should demonstrate that:

```text
1. Stable Diffusion has been fine-tuned on the Itay dataset using LoRA.
2. The trained LoRA can generate images that resemble the target subject.
3. The generated images can still follow different text prompts.
4. The result is evaluated using CLIP-I and CLIP-T.
5. The trained model can be used for inference with Diffusers.
6. The trained model can also be used in a ComfyUI inference workflow.
```

This assignment focuses on applying Stable Diffusion to a specific subject through LoRA fine-tuning and using the trained model in practical inference workflows.
