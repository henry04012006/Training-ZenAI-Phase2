import math
import torch
import torch.nn as nn

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
        
        # Second conv block
        h = self.conv2(self.act2(self.norm2(h)))
        
        # Add residual connection
        return h + self.shortcut(x)

class SelfAttention(nn.Module):
    """Multi-Head Self-Attention block with a residual connection.
    
    Applies Group Normalization, flattens spatial dimensions to sequence tokens,
    computes self-attention, and reshapes back to the spatial layout.
    """
    def __init__(self, channels: int, num_heads: int = 4, num_groups: int = 32):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups, channels)
        self.mha = nn.MultiheadAttention(embed_dim=channels, num_heads=num_heads, batch_first=True)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        # Pre-normalize
        h = self.norm(x)
        # Reshape to [B, H*W, C] (Sequence length = H*W, Feature dimension = C)
        h = h.view(B, C, H * W).transpose(1, 2)
        # Self-attention output
        attn_out, _ = self.mha(h, h, h)
        # Reshape back to spatial tensor [B, C, H, W]
        attn_out = attn_out.transpose(1, 2).view(B, C, H, W)
        # Residual connection
        return x + attn_out
