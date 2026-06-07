import torch
import torch.nn as nn
from models.blocks import SinusoidalPositionEmbeddings, ResNetBlock, SelfAttention

class UNet(nn.Module):
    """Symmetrical U-Net architecture with time and class conditioning.
    
    Accepts noisy images of shape [B, 1, 32, 32], timesteps of shape [B], and
    class labels of shape [B] to predict the added noise tensor of shape [B, 1, 32, 32].
    
    Supports dynamic stacking of multiple ResNet blocks per spatial scale
    via the layers_per_block parameter. Uses learnable ConvTranspose2d layers
    for spatial upsampling.
    
    Spatial dimensions transition: 32 -> 16 -> 8 -> 4 -> 8 -> 16 -> 32.
    """
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 64,
        class_dim: int = 128,
        cond_dim: int = 256,
        num_classes: int = 10,       # 0-9 digits, class index 10 represents unconditional token
        layers_per_block: int = 2,   # Configurable number of ResNet layers per scale stage
    ):
        super().__init__()
        
        self.layers_per_block = layers_per_block
        
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
        self.down_blocks1 = nn.ModuleList()
        in_c = base_channels
        for _ in range(layers_per_block):
            self.down_blocks1.append(ResNetBlock(in_c, base_channels, cond_dim))
            in_c = base_channels
        self.downsample1 = nn.Conv2d(base_channels, base_channels, kernel_size=4, stride=2, padding=1)
        
        # Scale 2: 16x16 (with Self-Attention)
        self.down_blocks2 = nn.ModuleList()
        in_c = base_channels
        for i in range(layers_per_block):
            out_c = base_channels * 2
            self.down_blocks2.append(ResNetBlock(in_c, out_c, cond_dim))
            in_c = out_c
        self.down_attn2 = SelfAttention(base_channels * 2)
        self.downsample2 = nn.Conv2d(base_channels * 2, base_channels * 2, kernel_size=4, stride=2, padding=1)
        
        # Scale 3: 8x8
        self.down_blocks3 = nn.ModuleList()
        in_c = base_channels * 2
        for i in range(layers_per_block):
            out_c = base_channels * 4
            self.down_blocks3.append(ResNetBlock(in_c, out_c, cond_dim))
            in_c = out_c
        self.downsample3 = nn.Conv2d(base_channels * 4, base_channels * 4, kernel_size=4, stride=2, padding=1)
        
        # 4. Bottleneck (Scale 4: 4x4)
        self.mid_block1 = ResNetBlock(base_channels * 4, base_channels * 8, cond_dim)
        self.mid_attn = SelfAttention(base_channels * 8)
        self.mid_block2 = ResNetBlock(base_channels * 8, base_channels * 8, cond_dim)
        
        # 5. Up-path (Expanding Stream)
        # Up 1: 4x4 -> 8x8 (learnable upsampler)
        self.upsample1 = nn.ConvTranspose2d(base_channels * 8, base_channels * 8, kernel_size=4, stride=2, padding=1)
        self.up_blocks1 = nn.ModuleList()
        # Concat channels: (Upsampled Out) base_channels * 8 + (Down Scale 3 Out) base_channels * 4 = base_channels * 12
        in_c = base_channels * 12
        for i in range(layers_per_block):
            out_c = base_channels * 4
            self.up_blocks1.append(ResNetBlock(in_c, out_c, cond_dim))
            in_c = out_c
            
        # Up 2: 8x8 -> 16x16 (learnable upsampler)
        self.upsample2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 4, kernel_size=4, stride=2, padding=1)
        self.up_blocks2 = nn.ModuleList()
        # Concat channels: base_channels * 4 + base_channels * 2 = base_channels * 6
        in_c = base_channels * 6
        for i in range(layers_per_block):
            out_c = base_channels * 2
            self.up_blocks2.append(ResNetBlock(in_c, out_c, cond_dim))
            in_c = out_c
        self.up_attn2 = SelfAttention(base_channels * 2)
        
        # Up 3: 16x16 -> 32x32 (learnable upsampler)
        self.upsample3 = nn.ConvTranspose2d(base_channels * 2, base_channels * 2, kernel_size=4, stride=2, padding=1)
        self.up_blocks3 = nn.ModuleList()
        # Concat channels: base_channels * 2 + base_channels = base_channels * 3
        in_c = base_channels * 3
        for i in range(layers_per_block):
            out_c = base_channels
            self.up_blocks3.append(ResNetBlock(in_c, out_c, cond_dim))
            in_c = out_c
        
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
        for block in self.down_blocks1:
            h = block(h, cond_emb)
        s1 = h
        h = self.downsample1(h)
        
        # Scale 2 (16x16)
        for block in self.down_blocks2:
            h = block(h, cond_emb)
        s2 = self.down_attn2(h)
        h = self.downsample2(s2)
        
        # Scale 3 (8x8)
        for block in self.down_blocks3:
            h = block(h, cond_emb)
        s3 = h
        h = self.downsample3(h)
        
        # 4. Bottleneck (4x4)
        h = self.mid_block1(h, cond_emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, cond_emb)
        
        # 5. Up-path
        # Up 1 (4x4 -> 8x8)
        h = self.upsample1(h)
        h = torch.cat([h, s3], dim=1)  # Skip connection concatenation
        for block in self.up_blocks1:
            h = block(h, cond_emb)
        
        # Up 2 (8x8 -> 16x16)
        h = self.upsample2(h)
        h = torch.cat([h, s2], dim=1)  # Skip connection concatenation
        for block in self.up_blocks2:
            h = block(h, cond_emb)
        h = self.up_attn2(h)
        
        # Up 3 (16x16 -> 32x32)
        h = self.upsample3(h)
        h = torch.cat([h, s1], dim=1)  # Skip connection concatenation
        for block in self.up_blocks3:
            h = block(h, cond_emb)
        
        # 6. Final projection
        return self.final_conv(h)
