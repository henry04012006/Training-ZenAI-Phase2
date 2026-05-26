import torch
import torch.nn as nn
from models.blocks import SinusoidalPositionEmbeddings, ResNetBlock, SelfAttention

class UNet(nn.Module):
    """Symmetrical U-Net architecture with time and class conditioning.
    
    Accepts noisy images of shape [B, 1, 32, 32], timesteps of shape [B], and
    class labels of shape [B] to predict the added noise tensor of shape [B, 1, 32, 32].
    
    Spatial dimensions transition: 32 -> 16 -> 8 -> 4 -> 8 -> 16 -> 32.
    """
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 64,
        class_dim: int = 128,
        cond_dim: int = 256,
        num_classes: int = 10,  # 0-9 digits, class index 10 represents unconditional token
    ):
        super().__init__()
        
        # 1. Conditioning Projection Layers
        self.time_embed = SinusoidalPositionEmbeddings(base_channels)
        self.time_mlp = nn.Sequential(
            nn.Linear(base_channels, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )
        
        # nn.Embedding has size num_classes + 1 (11 classes) to hold the unconditional null token at index 10
        self.class_emb = nn.Embedding(num_classes + 1, class_dim)
        self.class_proj = nn.Sequential(
            nn.Linear(class_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )
        
        # 2. Initial input projection
        self.init_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        
        # 3. Down-path (Contracting Stream)
        # Scale 1: 32x32
        self.down_block1 = ResNetBlock(base_channels, base_channels, cond_dim)
        self.downsample1 = nn.Conv2d(base_channels, base_channels, kernel_size=4, stride=2, padding=1)
        
        # Scale 2: 16x16 (with Self-Attention)
        self.down_block2 = ResNetBlock(base_channels, base_channels * 2, cond_dim)
        self.down_attn2 = SelfAttention(base_channels * 2)
        self.downsample2 = nn.Conv2d(base_channels * 2, base_channels * 2, kernel_size=4, stride=2, padding=1)
        
        # Scale 3: 8x8
        self.down_block3 = ResNetBlock(base_channels * 2, base_channels * 4, cond_dim)
        self.downsample3 = nn.Conv2d(base_channels * 4, base_channels * 4, kernel_size=4, stride=2, padding=1)
        
        # 4. Bottleneck (Scale 4: 4x4)
        self.mid_block1 = ResNetBlock(base_channels * 4, base_channels * 8, cond_dim)
        self.mid_attn = SelfAttention(base_channels * 8)
        self.mid_block2 = ResNetBlock(base_channels * 8, base_channels * 8, cond_dim)
        
        # 5. Up-path (Expanding Stream)
        self.upsample1 = nn.Upsample(scale_factor=2, mode="nearest")
        # Concat channels: (Bottleneck Out) base_channels * 8 + (Down Scale 3 Out) base_channels * 4 = base_channels * 12
        self.up_block1 = ResNetBlock(base_channels * 12, base_channels * 4, cond_dim)
        
        self.upsample2 = nn.Upsample(scale_factor=2, mode="nearest")
        # Concat channels: base_channels * 4 + base_channels * 2 = base_channels * 6
        self.up_block2 = ResNetBlock(base_channels * 6, base_channels * 2, cond_dim)
        self.up_attn2 = SelfAttention(base_channels * 2)
        
        self.upsample3 = nn.Upsample(scale_factor=2, mode="nearest")
        # Concat channels: base_channels * 2 + base_channels = base_channels * 3
        self.up_block3 = ResNetBlock(base_channels * 3, base_channels, cond_dim)
        
        # 6. Final output projection
        self.final_conv = nn.Conv2d(base_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Forward pass of the U-Net.
        
        Args:
            x: Noisy image tensor of shape [B, 1, 32, 32].
            t: Timestep tensor of shape [B].
            c: Class label tensor of shape [B] (elements in [0, 10]).
            
        Returns:
            Predicted noise tensor of shape [B, 1, 32, 32].
        """
        # 1. Compute and project conditioning embeddings
        time_emb = self.time_mlp(self.time_embed(t))      # [B, cond_dim]
        class_emb = self.class_proj(self.class_emb(c))    # [B, cond_dim]
        cond_emb = time_emb + class_emb                   # [B, cond_dim]
        
        # 2. Initial input projection
        h = self.init_conv(x)
        
        # 3. Down-path
        # Scale 1 (32x32)
        s1 = self.down_block1(h, cond_emb)
        h = self.downsample1(s1)
        
        # Scale 2 (16x16)
        s2 = self.down_block2(h, cond_emb)
        s2 = self.down_attn2(s2)
        h = self.downsample2(s2)
        
        # Scale 3 (8x8)
        s3 = self.down_block3(h, cond_emb)
        h = self.downsample3(s3)
        
        # 4. Bottleneck (4x4)
        h = self.mid_block1(h, cond_emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, cond_emb)
        
        # 5. Up-path
        # Up 1 (4x4 -> 8x8)
        h = self.upsample1(h)
        h = torch.cat([h, s3], dim=1)  # Skip connection concatenation
        h = self.up_block1(h, cond_emb)
        
        # Up 2 (8x8 -> 16x16)
        h = self.upsample2(h)
        h = torch.cat([h, s2], dim=1)  # Skip connection concatenation
        h = self.up_block2(h, cond_emb)
        h = self.up_attn2(h)
        
        # Up 3 (16x16 -> 32x32)
        h = self.upsample3(h)
        h = torch.cat([h, s1], dim=1)  # Skip connection concatenation
        h = self.up_block3(h, cond_emb)
        
        # 6. Final projection
        return self.final_conv(h)
