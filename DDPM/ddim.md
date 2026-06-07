# DDIM Implementation Guide

This document outlines the theoretical background, mathematical equations, and implementation details for replacing the stochastic DDPM sampling loop with Denoising Diffusion Implicit Models (DDIM) sampling.

---

## Part 1: DDIM Core Theory & Implementation Specification

### 1. Key Concepts
* **Weight Reuse**: DDIM shares the same forward noising process ($q(x_t|x_0)$) and training objective (MSE loss) as DDPM. The pre-trained U-Net weights from Assignment 2 (`checkpoints/ddpm_mnist_epoch_30.pth`) are fully compatible. **No retraining is required.**
* **Probability Flow ODE**: Setting the variance parameter $\sigma_t = 0$ (or stochasticity parameter $\eta = 0$) transforms the reverse sampling process into a fully deterministic mapping. This permits strided sampling (jumping over timesteps) without catastrophic quality degradation.

### 2. Mathematical Formulation
To transition from a noisy state $x_{\tau_i}$ to $x_{\tau_{i-1}}$ (where $\tau_{i-1} < \tau_i$), we perform the following steps:

1. **Predict Clean Image ($\hat{x}_0$)**:
   $$\hat{x}_0 = \frac{x_{\tau_i} - \sqrt{1 - \bar{\alpha}_{\tau_i}} \cdot \epsilon_{\text{guided}}}{\sqrt{\bar{\alpha}_{\tau_i}}}$$
   *Where $\epsilon_{\text{guided}}$ is computed using Classifier-Free Guidance (CFG).*

2. **Compute Stochastic Coefficient ($\sigma_{\tau_i}$)**:
   $$\sigma_{\tau_i} = \eta \cdot \sqrt{\frac{1 - \bar{\alpha}_{\tau_{i-1}}}{1 - \bar{\alpha}_{\tau_i}} \cdot \left(1 - \frac{\bar{\alpha}_{\tau_i}}{\bar{\alpha}_{\tau_{i-1}}}\right)}$$
   *For deterministic sampling, we set $\eta = 0.0$, which implies $\sigma_{\tau_i} = 0.0$.*

3. **Compute Next State ($x_{\tau_{i-1}}$)**:
   $$x_{\tau_{i-1}} = \sqrt{\bar{\alpha}_{\tau_{i-1}}} \cdot \hat{x}_0 + \sqrt{1 - \bar{\alpha}_{\tau_{i-1}} - \sigma_{\tau_i}^2} \cdot \epsilon_{\text{guided}} + \sigma_{\tau_i} \cdot z$$
   *Where $z \sim \mathcal{N}(0, I)$ is random Gaussian noise (if $\tau_{i-1} \ge 0$, else $z = 0$).*
   *Edge Case Handling:* For $\tau_0 = -1$, we set $\bar{\alpha}_{-1} = 1.0$, which simplifies the step to $x_{-1} = \hat{x}_0$.

### 3. Implementation Details in `models/diffusion.py`
The method `ddim_sample` should be added to the `Diffusion` class. It must:
* Reuse `self._extract` to obtain vectorized batch-compatible $\bar{\alpha}$ values.
* Reuse `self.alphas_cumprod` for cumulative variance coefficients.
* Support class conditioning and CFG scale $w$.
* Use a robust `torch.linspace` scheduler to divide the range $[T-1, 0]$ into exactly $S$ steps.

#### PyTorch Pseudo-code
```python
@torch.no_grad()
def ddim_sample(
    self,
    model: nn.Module,
    num_samples: int,
    channels: int,
    img_size: int,
    device: torch.device,
    c: torch.Tensor,
    steps: int = 50,
    eta: float = 0.0,
    cfg_scale: float = 3.0
) -> torch.Tensor:
    """
    Generates images using Denoising Diffusion Implicit Models (DDIM) sampling.
    Can be run in deterministic mode (eta = 0.0) and supports accelerated inference steps.
    """
    model.eval()
    
    # 1. Initialize latent noise x_T
    x = torch.randn((num_samples, channels, img_size, img_size), device=device)
    
    # 2. Create strided schedule from T-1 down to 0
    # For example, if steps=50, timesteps will contain 50 elements from 999 down to 0
    timesteps = torch.linspace(self.timesteps - 1, 0, steps, dtype=torch.long, device=device)
    
    # 3. Iterative reverse process
    for i in range(steps):
        t = timesteps[i]
        t_prev = timesteps[i + 1] if i + 1 < steps else torch.tensor(-1, device=device)
        
        # Batch timestep tensors
        vec_t = torch.full((num_samples,), t, dtype=torch.long, device=device)
        vec_t_prev = torch.full((num_samples,), t_prev, dtype=torch.long, device=device) if t_prev >= 0 else None
        
        # Extract cumulative alphas
        alpha_bar_t = self._extract(self.alphas_cumprod, vec_t, x.shape)
        alpha_bar_s = self._extract(self.alphas_cumprod, vec_t_prev, x.shape) if t_prev >= 0 else torch.ones_like(alpha_bar_t)
        
        # Predict noise using Classifier-Free Guidance (CFG)
        eps_cond = model(x, vec_t, c)
        eps_uncond = model(x, vec_t, torch.full_like(c, 10))
        eps = (1.0 + cfg_scale) * eps_cond - cfg_scale * eps_uncond
        
        # Estimate clean image x_0
        pred_x0 = (x - (1.0 - alpha_bar_t).sqrt() * eps) / alpha_bar_t.sqrt()
        
        # Compute sigma (stochasticity parameter)
        if eta > 0.0 and t_prev >= 0:
            sigma = eta * torch.sqrt(
                ((1.0 - alpha_bar_s) / (1.0 - alpha_bar_t)) * (1.0 - alpha_bar_t / alpha_bar_s)
            )
        else:
            sigma = torch.zeros_like(alpha_bar_t)
            
        # Direction pointing to x_s
        direction = torch.sqrt(1.0 - alpha_bar_s - sigma**2) * eps
        
        # Compute x_{prev}
        if t_prev >= 0:
            noise = torch.randn_like(x)
            x = alpha_bar_s.sqrt() * pred_x0 + direction + sigma * noise
        else:
            x = pred_x0  # Return predicted x_0 on final step
            
    model.train()
    return x
```

---

## Part 2: Creation of `ddim_inference.ipynb`

This section specifies the cell-by-cell structure of `ddim_inference.ipynb`. To ensure readability and maintainability (conforming to `rules.md`), the code is divided into logical, bite-sized cells (under 50 lines each). Visualization grids are plotted cleanly using dedicated individual subplots.

### Cell 1: Library Imports
* **Purpose:** Dedicate the first cell exclusively to libraries.
* **Content:**
  ```python
  import time
  import torch
  import torchvision
  import numpy as np
  import matplotlib.pyplot as plt
  
  from configs.config import DiffusionConfig
  from models.unet import UNet
  from models.diffusion import Diffusion
  from utils.tools import show_images
  ```

### Cell 2: SLERP Helper Definition
* **Purpose:** Implement Spherical Linear Interpolation (slerp) for semantic interpolation.
* **Content:**
  ```python
  def slerp(v0: torch.Tensor, v1: torch.Tensor, t: float) -> torch.Tensor:
      """Spherical linear interpolation between two normalized vectors."""
      v0_norm = v0 / torch.norm(v0, dim=-1, keepdim=True)
      v1_norm = v1 / torch.norm(v1, dim=-1, keepdim=True)
      dot = torch.sum(v0_norm * v1_norm, dim=-1, keepdim=True)
      omega = torch.acos(torch.clamp(dot, -1.0, 1.0))
      sin_omega = torch.sin(omega)
      
      # Fallback to linear interpolation (lerp) if vectors are nearly parallel
      res = torch.where(
          sin_omega < 1e-5,
          (1.0 - t) * v0 + t * v1,
          (torch.sin((1.0 - t) * omega) / sin_omega) * v0 + (torch.sin(t * omega) / sin_omega) * v1
      )
      return res
  ```

### Cell 3: Model and Checkpoint Initialization
* **Purpose:** Load the pre-trained DDPM checkpoint and restore EMA weights.
* **Content:**
  ```python
  device = torch.device(DiffusionConfig.DEVICE)
  print(f"Using device: {device}")
  
  model = UNet(
      in_channels=DiffusionConfig.CHANNELS,
      out_channels=DiffusionConfig.CHANNELS,
      base_channels=96,
      num_classes=DiffusionConfig.NUM_CLASSES
  ).to(device)
  
  diffusion = Diffusion(
      timesteps=DiffusionConfig.TIMESTEPS,
      beta_start=DiffusionConfig.BETA_START,
      beta_end=DiffusionConfig.BETA_END,
      schedule_type="cosine"
  ).to(device)
  
  # Load the trained checkpoint and map to the active device
  checkpoint = torch.load("checkpoints/ddpm_mnist_epoch_30.pth", map_location=device)
  model.load_state_dict(checkpoint['ema_shadow'])
  print("Checkpoint EMA shadow weights successfully loaded.")
  ```

### Cell 4: Experiment 1 - Speed vs. Quality (Sampling Execution)
* **Purpose:** Generate images using DDPM (1000 steps) vs DDIM (50 steps and 20 steps) using a fixed seed noise and measure execution speed.
* **Content:**
  - Instantiate a single fixed noise tensor `seed_noise` and label class sequence `c_test` (digits 0-9).
  - Sample with standard DDPM `p_sample` loop and record execution time.
  - Sample with `ddim_sample` for 50 steps and 20 steps using the same `seed_noise`, and record execution times.

### Cell 5: Experiment 1 - Speed vs. Quality (Visualization)
* **Purpose:** Plot the 3 grids (DDPM-1000, DDIM-50, DDIM-20) side-by-side with clear speedup labels in titles.

### Cell 6: Experiment 2 - Determinism vs. Stochasticity (Execution)
* **Purpose:** Sample 3 times from the same fixed latent vector using DDPM and DDIM ($\eta = 0.0$).
* **Content:**
  - Instantiate `fixed_noise` vector representing a conditional digit.
  - Run standard DDPM sampling 3 times.
  - Run DDIM sampling ($\eta = 0.0$) 3 times.

### Cell 7: Experiment 2 - Determinism vs. Stochasticity (Visualization)
* **Purpose:** Plot the 3 DDPM outputs in Row 1 (showing variations) and 3 DDIM outputs in Row 2 (showing identical pixels).

### Cell 8: Experiment 3 - Semantic Interpolation (Execution)
* **Purpose:** Blend two different latent noise vectors using `slerp` and decode using DDIM.
* **Content:**
  - Generate latent vector A ($x_T^{(A)}$) for a start digit (e.g. '3') and latent vector B ($x_T^{(B)}$) for an end digit (e.g. '8').
  - Compute 10 interpolated latents using the `slerp` helper with a linear $t$ space from 0.0 to 1.0.
  - Denoise all 10 latents using DDIM ($\eta = 0.0$) over 50 steps.

### Cell 9: Experiment 3 - Semantic Interpolation (Visualization)
* **Purpose:** Plot the 10 decoded images in a single row to show a smooth semantic morphing from digit A to digit B.