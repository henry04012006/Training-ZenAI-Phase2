import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal position embeddings for time step conditioning.
    
    Maps scalar timesteps to dense embedding vectors of a specified dimension,
    using alternating sine and cosine functions of varying frequencies.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        # time: [B] tensor of timesteps
        device = time.device
        half_dim = self.dim // 2
        # Compute frequency scaling factors
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        # Outer product of time and frequencies
        embeddings = time[:, None] * embeddings[None, :]
        # Concatenate sine and cosine components
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class ResNetBlock(nn.Module):
    """Residual block with integrated time and class conditioning.
    
    Applies Group Normalization, SiLU activation, and Convolution.
    Injects a projected conditioning vector (time + class) element-wise
    into the feature maps before the second convolution.
    """
    def __init__(self, in_channels: int, out_channels: int, cond_dim: int, num_groups: int = 32):
        super().__init__()
        self.norm1 = nn.GroupNorm(num_groups, in_channels)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        
        # Project conditioning vector to match out_channels
        self.cond_proj = nn.Linear(cond_dim, out_channels)
        
        self.norm2 = nn.GroupNorm(num_groups, out_channels)
        self.act2 = nn.SiLU()
        self.dropout = nn.Dropout(p=0.1)  # Added dropout layer for regularisation
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
        # Shortcut mapping if input and output channel dimensions differ
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor, cond_emb: torch.Tensor) -> torch.Tensor:
        # x: [B, in_channels, H, W]
        # cond_emb: [B, cond_dim]
        
        # First conv block
        h = self.conv1(self.act1(self.norm1(x)))
        
        # Project and inject conditioning vector [B, out_channels, 1, 1]
        cond_feature = self.cond_proj(cond_emb)[:, :, None, None]
        h = h + cond_feature
        
        # Second conv block with dropout regularization
        h = self.conv2(self.dropout(self.act2(self.norm2(h))))
        
        # Add residual connection
        return h + self.shortcut(x)

class SelfAttention(nn.Module):
    """Multi-Head Self-Attention block with a residual connection.
    
    Applies Group Normalization, flattens spatial dimensions to sequence tokens,
    projects to Q, K, V queries, computes self-attention using PyTorch's native
    scaled dot product attention (FlashAttention), projects the output back,
    and reshapes back to the spatial layout.
    """
    def __init__(self, channels: int, num_heads: int = 4, num_groups: int = 32):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        
        self.norm = nn.GroupNorm(num_groups, channels)
        # Linear projection for combined Q, K, V queries
        self.qkv_proj = nn.Linear(channels, channels * 3)
        # Output projection back to feature channels
        self.out_proj = nn.Linear(channels, channels)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        S = H * W
        
        # Pre-normalize the inputs
        h = self.norm(x)
        
        # Reshape to [B, H*W, C] (Sequence length = H*W, Feature dimension = C)
        h = h.view(B, C, S).transpose(1, 2)
        
        # Project inputs to Q, K, V combined representation: [B, S, 3 * C]
        qkv = self.qkv_proj(h)
        
        # Reshape and split into Q, K, V of shape [B, num_heads, S, head_dim]
        # First shape: [B, S, 3, num_heads, head_dim]
        qkv = qkv.view(B, S, 3, self.num_heads, self.head_dim)
        # Permute to: [3, B, num_heads, S, head_dim]
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Compute FlashAttention (scaled dot product attention)
        # Output shape: [B, num_heads, S, head_dim]
        out = F.scaled_dot_product_attention(q, k, v)
        
        # Transpose and reshape back to [B, S, C]
        out = out.transpose(1, 2).contiguous().view(B, S, C)
        
        # Project output back to original channel size
        out = self.out_proj(out)
        
        # Reshape back to spatial layout [B, C, H, W]
        out = out.transpose(1, 2).view(B, C, H, W)
        
        # Residual connection
        return x + out


class EMA:
    """Exponential Moving Average (EMA) helper class for tracking model parameters.
    
    Tracks a running average of the network parameters and provides methods
    to update, apply/swap, and restore the model weights during training and inference.
    """
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        # Keep a copy of trainable parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone().detach()

    @torch.no_grad()
    def update(self, model: nn.Module):
        """Update shadow weights: shadow = decay * shadow + (1 - decay) * online."""
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                # Perform in-place update on the shadow parameter tensor
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name].to(param.device)
                self.shadow[name].copy_(new_average)

    @torch.no_grad()
    def apply_shadow(self, model: nn.Module):
        """Swap online parameters with the shadow parameters."""
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                # Backup online weight before overwriting
                self.backup[name] = param.data.clone().detach()
                param.data.copy_(self.shadow[name].to(param.device))

    @torch.no_grad()
    def restore(self, model: nn.Module):
        """Restore online model parameters from backup."""
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data.copy_(self.backup[name])
        self.backup = {}

