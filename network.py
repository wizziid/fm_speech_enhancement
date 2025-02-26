import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

class Network(nn.Module):
    """
    U-Net style network with residual blocks, attention, and time embedding.
    """
    def __init__(self, input_shape, base_channels=128, embedding_dim=512, n_residual_blocks=6, n_att_blocks=0, device="cpu"):
        super().__init__()
        self.device = torch.device(device)
        self.input_shape = input_shape
        self.base_channels = base_channels
        self.embedding_dim = embedding_dim
        self.time_layer = nn.Linear(self.embedding_dim, self.embedding_dim).to(self.device)
        self.initial_conv = nn.Conv2d(2, base_channels, kernel_size=5, padding=2).to(self.device)

        enc_blocks, norm_layers_enc, attention_enc, downsample_layers = [], [], [], []
        channels = base_channels
        for i in range(n_residual_blocks):
            next_channels = channels * 2 if i % 2 == 1 else channels
            enc_blocks.append(ResidualBlock(channels, next_channels).to(self.device))
            norm_layers_enc.append(nn.BatchNorm2d(next_channels).to(self.device))
            attention_enc.append(SelfAttention(next_channels).to(self.device) if i >= n_residual_blocks - n_att_blocks else None)
            downsample_layers.append(nn.Conv2d(next_channels, next_channels, kernel_size=4, stride=2, padding=1).to(self.device))
            channels = next_channels  
        
        self.enc_blocks = nn.ModuleList(enc_blocks)
        self.norm_layers_enc = nn.ModuleList(norm_layers_enc)
        self.attention_enc = nn.ModuleList(attention_enc)
        self.downsample = nn.ModuleList(downsample_layers)
        
        self.bottleneck = ResidualBlock(channels, channels).to(self.device)
        self.norm_bottleneck = nn.BatchNorm2d(channels).to(self.device)
        
        dec_blocks, norm_layers_dec, attention_dec, upsample_layers = [], [], [], []
        for i in range(n_residual_blocks):
            next_channels = channels // 2 if i % 2 == 0 else channels
            upsample_layers.append(nn.ConvTranspose2d(channels, channels, kernel_size=4, stride=2, padding=1).to(self.device))
            dec_blocks.append(ResidualBlock(channels * 2, next_channels).to(self.device))
            norm_layers_dec.append(nn.BatchNorm2d(next_channels).to(self.device))
            attention_dec.append(SelfAttention(next_channels).to(self.device) if i < n_att_blocks else None)
            channels = next_channels 
        
        self.upsample = nn.ModuleList(upsample_layers)
        self.dec_blocks = nn.ModuleList(dec_blocks)
        self.norm_layers_dec = nn.ModuleList(norm_layers_dec)
        self.attention_dec = nn.ModuleList(attention_dec)
        self.final_conv = nn.Conv2d(base_channels, 2, kernel_size=5, padding=2).to(self.device)
    
    def forward(self, x, t):
        x, t = x.to(self.device), t.to(self.device).reshape(-1, 1)
        embedding = self.time_embedding(t)
        embedding = F.relu(self.time_layer(embedding))
        skips = []
        x = self.initial_conv(x)
        for block, norm, attn, down in zip(self.enc_blocks, self.norm_layers_enc, self.attention_enc, self.downsample):
            x = block(x)  
            x = norm(x)
            if attn: x = attn(x)
            skips.append(x)
            x = down(x)
        
        x = checkpoint.checkpoint(self.bottleneck, x, use_reentrant=False)
        x = self.norm_bottleneck(x)
        
        for block, norm, attn, up, skip in zip(self.dec_blocks, self.norm_layers_dec, self.attention_dec, self.upsample, reversed(skips)):
            x = up(x)
            x = torch.cat([x, skip], dim=1)
            x = checkpoint.checkpoint(block, x, use_reentrant=False)
            x = norm(x)
            if attn: x = attn(x)
        
        return self.final_conv(x)

    def time_embedding(self, t):
        batch_size = t.shape[0]
        i = torch.arange(0, self.embedding_dim // 2, device=self.device).float()
        time_steps = t.reshape(batch_size, 1)
        sin_emb = torch.sin(2 * torch.pi * time_steps / (10000 ** (2 * i / self.embedding_dim)))
        cos_emb = torch.cos(2 * torch.pi * time_steps / (10000 ** (2 * i / self.embedding_dim)))
        return torch.cat((sin_emb, cos_emb), dim=1)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=5, padding=2)
        self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        self.relu = nn.ReLU()
    
    def forward(self, x):
        residual = self.skip(x)
        x = self.relu(self.conv1(x))
        x = self.conv2(x)
        return x + residual

class SelfAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x):
        batch, channels, height, width = x.shape
        qkv = self.qkv(x).reshape(batch, 3, channels, height * width)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        attn = self.softmax(q @ k.transpose(-2, -1) / (channels ** 0.5))
        return (attn @ v).reshape(batch, channels, height, width) + x
