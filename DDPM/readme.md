# Assignment 2.2: Implement DDPM from Scratch

## 📌 Project Overview
The objective of this assignment is to gain a deep understanding of how Diffusion Models work by implementing a **Denoising Diffusion Probabilistic Model (DDPM)** entirely from scratch using PyTorch. 

**Important:** The use of the `diffusers` library or any pre-built diffusion pipelines is strictly prohibited. Everything must be built from the ground up.

## 🎯 Detailed Requirements
The project includes the implementation of the following core components:
1. **Noise Scheduler:** Implement linear (and optionally cosine) variance schedules to compute $\beta_t$, $\alpha_t$, and $\bar{\alpha}_t$.
2. **Forward Process:** Implement `q_sample(x0, t, noise)` to progressively add Gaussian noise to the images.
3. **U-Net Architecture:** Build a minimal U-Net using ResNet blocks and Sinusoidal Time Embeddings.
4. **Training Loop:** Implement the PyTorch training loop to sample $t$, add noise, predict the noise using the U-Net, and compute the MSE loss.
5. **Sampling/Inference Loop:** Implement the reverse denoising process starting from random pure noise $x_T$ to generate images over $T$ steps.

**Dataset:** MNIST (or CIFAR-10)
**Framework:** PyTorch Only

## 📊 Expected Results
To fulfill the final assignment requirements, running the notebook will output the following:
1. **Training Results:** A **Loss Curve chart** (and logs) demonstrating the convergence of the MSE loss over the training steps/epochs.
2. **Generated Images:** A visualization grid of **images generated randomly from pure latent noise**, showcasing the model's ability to denoise and construct digits from scratch.

### 🌟 Bonus (Advanced)
- **Class Conditioning:** Train a conditional diffusion model based on digit labels (0–9) using Classifier-Free Guidance (CFG).
- **Visualization:** Visualize class-conditional generation results.

## 📂 Directory Tree
The project follows a modular structure to separate core model logic from data processing and training loops:

```text
ddpm_mnist/
│
├── models/                 # Core AI architecture and network modules
│   ├── blocks.py           # Neural network components (Time Embeddings, ResNet blocks)
│   ├── unet.py             # Minimal U-Net assembly and structural definition
│   └── diffusion.py        # DDPM logic (Noise scheduler, Forward & Reverse processes)
│
├── utils/                  # Helper functions and visualization utilities
│   └── tools.py            # Functions for plotting loss curves and visualizing image grids
│
├── configs/                # Project configurations (Optional)
│   └── config.py           # Hyperparameters (T, beta schedules, learning rate, batch size)
│
├── main.ipynb              # Main Jupyter Notebook (Data loading, Training, and Inference)
├── README.md               # High-level project overview and usage guide for users
├── context.md              # Deep technical blueprint, mathematical formulas, and knowledge base
└── rules.md                # Strict development constraints and coding rules for AI Agents