import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

class Network(nn.Module):
    """
    U-Net style network with residual blocks, attention, and time embedding.
    """
    def __init__(self, input_shape, input_channels=3, base_channels=128, embedding_dim=256, n_residual_blocks=5, n_att_blocks=0, down_samples=5, channel_increases=5, device="cpu"):
        super().__init__()

        assert n_residual_blocks % down_samples == 0, "The down_samples should be a factor of n_residual_blocks"
        assert n_residual_blocks % channel_increases == 0, "The down_samples should be a factor of n_residual_blocks"
        assert torch.log2(torch.tensor(min(input_shape[-1], input_shape[-2]))) > down_samples, "Down_samples should be less than log_2(min(Height, Width))"
        assert n_att_blocks <= n_residual_blocks, "Should have n_att_blocks < n_residual_blocks"

        self.device = torch.device(device)
        self.input_shape = input_shape
        self.base_channels = base_channels
        self.n_residual_blocks = n_residual_blocks
        self.n_att_blocks = n_att_blocks
        self.embedding_dim = embedding_dim
        self.down_samples = down_samples
        self.channel_increases = channel_increases

        # Shared parameters
        self.activation = nn.SiLU()
        self.kernel_size = 3
        self.padding = 1

        self.time_layer = nn.Linear(self.embedding_dim, self.embedding_dim).to(self.device)
        self.initial_conv = nn.Conv2d(input_channels, base_channels, kernel_size=3, padding=1).to(self.device)

        enc_blocks, norm_layers_enc, attention_enc, downsample_layers = [], [], [], []
        channels = base_channels
        for i in range(n_residual_blocks):
            
            down_sample = (i+1) % self.down_samples == 0  
            increase_channels = (i+1) % self.channel_increases == 0
            add_attention = i >= (self.n_residual_blocks - self.n_att_blocks)

            next_channels = channels * 2 if increase_channels else channels
            enc_blocks.append(ResidualBlock(channels, next_channels, self.activation, self.kernel_size, self.padding, self.embedding_dim).to(self.device))
            norm_layers_enc.append(nn.GroupNorm(32, next_channels).to(self.device))
            attention_enc.append(SelfAttention(next_channels, self.activation).to(self.device) if add_attention else None)
            downsample_layers.append(nn.Conv2d(next_channels, next_channels, kernel_size=4, stride=2, padding=1).to(self.device) if down_sample else None)
            channels = next_channels  
        
        self.enc_blocks = nn.ModuleList(enc_blocks)
        self.norm_layers_enc = nn.ModuleList(norm_layers_enc)
        self.attention_enc = nn.ModuleList(attention_enc)
        self.downsample = nn.ModuleList(downsample_layers)
        
        self.bottleneck = ResidualBlock(channels, channels, self.activation, self.kernel_size, self.padding, self.embedding_dim).to(self.device)
        self.norm_bottleneck = nn.GroupNorm(32, channels).to(self.device)
        self.bottleneck_att = SelfAttention(channels, self.activation).to(self.device) if self.n_att_blocks != 0 else None
        
        dec_blocks, norm_layers_dec, attention_dec, upsample_layers = [], [], [], []
        for i in range(n_residual_blocks):

            up_sample = (i) % self.down_samples == 0  
            decrease_channels = (i) % self.channel_increases == 0
            add_attention = i < self.n_att_blocks

            next_channels = channels // 2 if decrease_channels else channels
            upsample_layers.append(nn.ConvTranspose2d(channels, channels, kernel_size=4, stride=2, padding=1).to(self.device) if up_sample else None)
            dec_blocks.append(ResidualBlock(channels * 2, next_channels, self.activation, self.kernel_size, self.padding, self.embedding_dim).to(self.device))
            norm_layers_dec.append(nn.GroupNorm(32, next_channels).to(self.device))
            attention_dec.append(SelfAttention(next_channels, self.activation).to(self.device) if add_attention else None)
            channels = next_channels 
        
        self.upsample = nn.ModuleList(upsample_layers)
        self.dec_blocks = nn.ModuleList(dec_blocks)
        self.norm_layers_dec = nn.ModuleList(norm_layers_dec)
        self.attention_dec = nn.ModuleList(attention_dec)
        self.final_conv = nn.Conv2d(base_channels, input_channels, kernel_size=3, padding=1).to(self.device)

    
    def forward(self, x, t):
        x, t = x.to(self.device), t.to(self.device).reshape(-1, 1)

        embedding = self.time_embedding(t)  # [batch, embedding_dim]
        embedding = self.activation(self.time_layer(embedding))  # [batch, embedding_dim]

        skips = []
        x = self.initial_conv(x)

        # Encoder
        for block, norm, attn, down in zip(self.enc_blocks, self.norm_layers_enc, self.attention_enc, self.downsample):
            x = block(x, embedding)  # Pass time embedding directly
            x = norm(x)
            if attn: x = attn(x)
            skips.append(x)
            if down: x = down(x)

        # Bottleneck
        x = self.bottleneck(x, embedding)  # Pass time embedding
        x = self.norm_bottleneck(x)

        # Decoder
        for block, norm, attn, up, skip in zip(self.dec_blocks, self.norm_layers_dec, self.attention_dec, self.upsample, reversed(skips)):
            if up: x = up(x)
            x = torch.cat([x, skip], dim=1)  # Skip connection remains
            x = block(x, embedding)  # Pass time embedding
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
    def __init__(self, in_channels, out_channels, activation, kernel_size, padding, embedding_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        self.activation = activation

        # Linear layer to transform time embedding to match `in_channels`
        self.time_proj = nn.Linear(embedding_dim, in_channels)

    def forward(self, x, t):
        t_proj = self.time_proj(t)  
        t_proj = t_proj.view(x.shape[0], -1, 1, 1)  
        x = x + t_proj
        residual = self.skip(x)
        x = self.activation(self.conv1(x))
        x = self.conv2(x)
        return x + residual

class SelfAttention(nn.Module):
    def __init__(self, channels, activation):
        super().__init__()
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)
        self.activation = activation
    
    def forward(self, x):
        batch, channels, height, width = x.shape
        qkv = self.qkv(x).reshape(batch, 3, channels, height * width)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        attn = self.softmax(q @ k.transpose(-2, -1) / (channels ** 0.5))
        return (attn @ v).reshape(batch, channels, height, width) + x
