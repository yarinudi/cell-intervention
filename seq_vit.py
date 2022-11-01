import os
import torch
import torch.nn as nn

import math
import numpy as np
from einops import rearrange
from einops.layers.torch import Rearrange

from mpp import MPPLoss


class PositionalEncoding(nn.Module):
    def __init__(self, dim_model, max_len):
        super().__init__()
        # Modified version from: https://pytorch.org/tutorials/beginner/transformer_tutorial.html

        # Encoding - From formula
        pos_encoding = torch.zeros(max_len, dim_model)
        positions_list = torch.arange(0, max_len, dtype=torch.float).view(-1, 1) # 0, 1, 2, 3, 4, 5
        division_term = torch.exp(torch.arange(0, dim_model, 2).float() * (-math.log(10000.0)) / dim_model) # 1000^(2i/dim_model)
        
        # PE(pos, 2i) = sin(pos/1000^(2i/dim_model))
        pos_encoding[:, 0::2] = torch.sin(positions_list * division_term)
        
        # PE(pos, 2i + 1) = cos(pos/1000^(2i/dim_model))
        pos_encoding[:, 1::2] = torch.cos(positions_list * division_term)
        
        # Saving buffer (same as parameter without gradients needed)
        pos_encoding = pos_encoding.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pos_encoding",pos_encoding)
        
    def forward(self, token_embedding: torch.tensor) -> torch.tensor:
        return self.pos_encoding[:token_embedding.size(0), :]


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head),
                FeedForward(dim, mlp_dim)
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class SeqViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, device, max_pixel_val, channels=1, output_channel_bits=3, dim_head=64, dropout_p=0.4, max_len=4, mean=None, std=None):
        super().__init__()
        """
            Modified version from the great work of SimpleViT in vit-pytorch:
            https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/simple_vit.py

            image_height - corresponds to number of frames.
            image_width - corresponds to number of features.
                e.g: 12x256 image, 12 frames with 256 features each.
            
            patch_height - corresponds to number of frames to path together.
            patch_width - corresponds to number of features.

        """
        image_height, image_width = image_size[0], image_size[1]
        patch_height, patch_width = patch_size[0], patch_size[1]

        assert image_height % patch_height == 0 or (image_height % patch_height) == image_height and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        assert image_height % max_len == 0, 'Max sub-sequence length must be divisible by the image height.'

        patch_dim = channels * patch_height * patch_width

        self.image_height = image_height
        self.max_len = max_len
        self.dropout = nn.Dropout(dropout_p)
        self.device = device

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b h w (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, dim)
        )

        self.positional_encoder = PositionalEncoding(
            dim_model=dim, max_len=max_len
        )

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim)

        self.to_latent = nn.Identity()
        self.linear_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

        self.soft_max = nn.Softmax(dim=1)

        self.to_bits = nn.Linear(dim, 2**(output_channel_bits * channels))
        self.loss = MPPLoss(patch_size, channels, output_channel_bits, max_pixel_val, mean, std)

    def forward(self, masked_img, orig_img, mask):
        # input img shape = (batch size, channels, height, width)
        
        x = self.to_patch_embedding(masked_img)
        # patches shape = (batch size, height, width, patch_dim)

        x = rearrange(x, 'b ... d -> b (...) d')
        # rearrange shape = (batch size, transformer dim)
        
        x = x + self.positional_encoder(x) # residual connection + pos encoding
        x = self.dropout(x)
        
        x_transformer = self.transformer(x)
        # transformer blocks shape = (batch size, height, transformer dim)

        x = x_transformer.mean(dim=1)
        # mean shape = (batch size, transformer dim)

        x = self.to_latent(x)
        x = self.linear_head(x)
        
        out = self.soft_max(x)
        # mean shape = (batch size, output dim)

        # regression logits for reg loss
        reg_logits = self.to_bits(x_transformer)
        mpp_loss = self.loss(reg_logits, orig_img, mask)

        return out, mpp_loss
