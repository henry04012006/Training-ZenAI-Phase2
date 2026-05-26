import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def linear_beta_schedule(timesteps: int, beta_start: float = 1e-4, beta_end: float = 0.02) -> torch.Tensor:
    """Linearly interpolates beta coefficients between beta_start and beta_end."""
    return torch.linspace(beta_start, beta_end, timesteps)

def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """Generates a cosine variance schedule as proposed in Improved DDPM."""
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    # Cosine cumulative product schedule formula
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    # Derive betas from alphas_cumprod
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    # Clip betas to prevent numerical issues at the end of the schedule
    return torch.clamp(betas, 0.0001, 0.9999)

class Diffusion(nn.Module):
    """Core mathematical engine for DDPM forward corruption and reverse sampling."""
    def __init__(
        self,
        timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        schedule_type: str = "linear",
    ):
        super().__init__()
        self.timesteps = timesteps
        self.schedule_type = schedule_type
        
        # Initialize selected variance schedule
        if schedule_type == "linear":
            betas = linear_beta_schedule(timesteps, beta_start, beta_end)
        elif schedule_type == "cosine":
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f"Unsupported schedule type: {schedule_type}")
            
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        
        # Register buffers so they are automatically tracked and moved to the correct device
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))

    def _extract(self, a: torch.Tensor, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
        """Helper to extract schedule coefficients for the batch at timestep t."""
        batch_size = t.shape[0]
        out = a.to(device=t.device).gather(0, t)
        # Reshape to match dimensions of input image tensors [B, 1, 1, 1]
        return out.view(batch_size, *((1,) * (len(x_shape) - 1)))

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Forward process: corruption mapping that adds noise to clean image x0 at step t."""
        sqrt_alpha_bar = self._extract(self.sqrt_alphas_cumprod, t, x0.shape)
        sqrt_one_minus_alpha_bar = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x0.shape)
        return sqrt_alpha_bar * x0 + sqrt_one_minus_alpha_bar * noise

    def get_loss(
        self,
        model: nn.Module,
        x0: torch.Tensor,
        c: torch.Tensor,
        t: torch.Tensor,
        p_uncond: float = 0.1,
    ) -> torch.Tensor:
        """Computes the simple MSE loss for conditional DDPM training.
        
        Randomly drops out class conditioning to train both conditional and
        unconditional representations for Classifier-Free Guidance (CFG).
        """
        # 1. Sample standard normal noise
        noise = torch.randn_like(x0)
        
        # 2. Corrupt data x0 to xt at step t
        xt = self.q_sample(x0, t, noise)
        
        # 3. Apply Classifier-Free Guidance random dropout to class labels
        # Mask class label to index 10 (unconditional token) with probability p_uncond
        mask = torch.rand_like(c, dtype=torch.float) < p_uncond
        c_masked = torch.where(mask, torch.tensor(10, device=c.device), c)
        
        # 4. Predict added noise vector using U-Net
        noise_pred = model(xt, t, c_masked)
        
        # 5. Compute mean squared error
        return F.mse_loss(noise, noise_pred)

    def p_sample(
        self,
        model: nn.Module,
        x: torch.Tensor,
        t: torch.Tensor,
        c: torch.Tensor,
        cfg_scale: float = 3.0,
    ) -> torch.Tensor:
        """Single reverse Markov chain step: sample x_{t-1} given x_t and class c.
        
        Combines conditional and unconditional noise predictions using CFG.
        """
        # Extract scheduler coefficients for the batch
        alpha = self._extract(self.alphas, t, x.shape)
        beta = self._extract(self.betas, t, x.shape)
        alpha_bar = self._extract(self.alphas_cumprod, t, x.shape)
        
        # Apply numerical stability precaution to denominator to prevent NaN division
        denom_safe = torch.clamp(1.0 - alpha_bar, min=1e-8).sqrt()
        
        # Predict noise: 1) conditional on label c, 2) unconditional (null label 10)
        eps_cond = model(x, t, c)
        uncond_c = torch.full_like(c, 10)
        eps_uncond = model(x, t, uncond_c)
        
        # Perform Classifier-Free Guidance extrapolation
        eps_guided = (1.0 + cfg_scale) * eps_cond - cfg_scale * eps_uncond
        
        # Reverse Markov chain mean formula
        sqrt_recip_alpha = 1.0 / alpha.sqrt()
        mean = sqrt_recip_alpha * (x - (1.0 - alpha) / denom_safe * eps_guided)
        
        # Stochastic sample injection
        if t[0] == 0:
            # Deterministic last step (z = 0)
            return mean
        else:
            # Langevin dynamic step: inject standard Gaussian noise scaled by sigma_t
            z = torch.randn_like(x)
            sigma = beta.sqrt()  # sigma_t = sqrt(beta_t)
            return mean + sigma * z

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        num_samples: int,
        channels: int,
        img_size: int,
        device: torch.device,
        c: torch.Tensor,
        cfg_scale: float = 3.0,
    ) -> torch.Tensor:
        """Generates images by iteratively denoising pure Gaussian noise backward."""
        model.eval()
        # Initialize with standard Gaussian noise
        x = torch.randn((num_samples, channels, img_size, img_size), device=device)
        
        # Iterate backwards from T-1 down to 0
        for t in reversed(range(self.timesteps)):
            t_tensor = torch.full((num_samples,), t, dtype=torch.long, device=device)
            x = self.p_sample(model, x, t_tensor, c, cfg_scale=cfg_scale)
            
        model.train()
        return x
