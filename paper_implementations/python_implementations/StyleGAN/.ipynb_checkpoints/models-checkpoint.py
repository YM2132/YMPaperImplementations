import torch
import torch.nn as nn
import torch.nn.functional as F

import random

from conv_blocks import *
from layers import *

class MappingNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.norm = PixelNorm()
        
        self.layers = nn.Sequential(
            EqualLRLinear(256, 256),
            nn.LeakyReLU(0.2),
            EqualLRLinear(256, 256),
            nn.LeakyReLU(0.2),
            EqualLRLinear(256, 256),
            nn.LeakyReLU(0.2),
            EqualLRLinear(256, 256),
            nn.LeakyReLU(0.2),
            EqualLRLinear(256, 256),
            nn.LeakyReLU(0.2),
            EqualLRLinear(256, 256),
            nn.LeakyReLU(0.2),
            EqualLRLinear(256, 256),
            nn.LeakyReLU(0.2),
            EqualLRLinear(256, 256),
            nn.LeakyReLU(0.2),
        )
    
    def forward(self, x):
        x = self.norm(x)

        out = self.layers(x)
        
        return out

class Discriminator(nn.Module):
    def __init__(self, out_c=256):
        super().__init__()
        
        self.blocks = nn.ModuleList([
            D_ConvBlock(out_c//4, out_c//4, 3, 1),
            D_ConvBlock(out_c//4, out_c//2, 3, 1),
            D_ConvBlock(out_c//2, out_c, 3, 1),
            D_ConvBlock(out_c, out_c, 3, 1),
            D_ConvBlock(out_c, out_c, 3, 1),
            D_ConvBlock(out_c, out_c, 3, 1), 
            D_ConvBlock(out_c+1, out_c, 3, 1, 4, 0, mbatch=True),
        ])
        
        self.from_rgb = nn.ModuleList([
            EqualLRConv2d(3, out_c//4, 1),
            EqualLRConv2d(3, out_c//4, 1),
            EqualLRConv2d(3, out_c//2, 1),
            EqualLRConv2d(3, out_c, 1),
            EqualLRConv2d(3, out_c, 1),
            EqualLRConv2d(3, out_c, 1),
            EqualLRConv2d(3, out_c, 1),
        ])
        self.num_layers = len(self.blocks)
        
        self.linear = EqualLRLinear(out_c, 1)
    
    def forward(self, x, layer_num, alpha):
        for i in reversed(range(layer_num)):
            idx = self.num_layers - i - 1
            if i+1 == layer_num:
                out = self.from_rgb[idx](x)
            out = self.blocks[idx](out)
            if i > 0:
                out = F.interpolate(out, scale_factor=0.5, mode='bilinear')
                
                if i+1 == layer_num and 0 <= alpha < 1:
                    skip = F.interpolate(x, scale_factor=0.5, mode='bilinear')
                    skip = self.from_rgb[idx + 1](skip)
                    out = ((1 - alpha) * skip) + (alpha * out)
        
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        
        return out

class Generator(nn.Module):
    def __init__(self, in_c=256):
        super().__init__()

        self.g_mapping = MappingNetwork()

        self.block_4x4 = G_ConvBlock(in_c, in_c, 3, 1, upsample=False)
        self.block_8x8 = G_ConvBlock(in_c, in_c, 3, 1)
        self.block_16x16 = G_ConvBlock(in_c, in_c, 3, 1)
        self.block_32x32 = G_ConvBlock(in_c, in_c, 3, 1)
        self.block_64x64 = G_ConvBlock(in_c, in_c//2, 3, 1)
        self.block_128x128 = G_ConvBlock(in_c//2, in_c//4, 3, 1)
        self.block_256x256 = G_ConvBlock(in_c//4, in_c//4, 3, 1)

        self.to_rgb_4 = EqualLRConv2d(in_c, 3, 1)
        self.to_rgb_8 = EqualLRConv2d(in_c, 3, 1)
        self.to_rgb_16 = EqualLRConv2d(in_c, 3, 1)
        self.to_rgb_32 = EqualLRConv2d(in_c, 3, 1)
        self.to_rgb_64 = EqualLRConv2d(in_c//2, 3, 1)
        self.to_rgb_128 = EqualLRConv2d(in_c//4, 3, 1)
        self.to_rgb_256 = EqualLRConv2d(in_c//4, 3, 1)

        self.tanh = nn.Tanh()

    def forward(self, z, layer_num, alpha):   
        w = self.g_mapping(z)

        if torch.rand(1).item() < 0.9:
            z2 = torch.randn_like(z)
            w2 = self.g_mapping(z2)

            crossover_point = random.randint(1, layer_num)
        else:
            crossover_point = None
        
        if crossover_point and 1 >= crossover_point:
            w = w2
        out_4 = self.block_4x4(w)
        if layer_num == 1:
            out = self.to_rgb_4(out_4)
            out = self.tanh(out)
            return out

        if crossover_point and 2 >= crossover_point:
            w = w2
        out_8 = self.block_8x8(w, out_4)
        if layer_num == 2:
            skip = self.to_rgb_4(out_4)
            skip = F.interpolate(skip, scale_factor=2, mode='bilinear')
            out = self.to_rgb_8(out_8)
            
            out = ((1-alpha) * skip) + (alpha * out)
            out = self.tanh(out)
            return out

        if crossover_point and 3 >= crossover_point:
            w = w2
        out_16 = self.block_16x16(w, out_8)
        if layer_num == 3:
            skip = self.to_rgb_8(out_8)
            skip = F.interpolate(skip, scale_factor=2, mode='bilinear')
            out = self.to_rgb_16(out_16)
            
            out = ((1-alpha) * skip) + (alpha * out)
            out = self.tanh(out)
            return out

        if crossover_point and 4 >= crossover_point:
            w = w2
        out_32 = self.block_32x32(w, out_16)
        if layer_num == 4:
            skip = self.to_rgb_16(out_16)
            skip = F.interpolate(skip, scale_factor=2, mode='bilinear')
            out = self.to_rgb_32(out_32)
            
            out = ((1-alpha) * skip) + (alpha * out)
            out = self.tanh(out)
            return out

        if crossover_point and 5 >= crossover_point:
            w = w2
        out_64 = self.block_64x64(w, out_32)
        if layer_num == 5:
            skip = self.to_rgb_32(out_32)
            skip = F.interpolate(skip, scale_factor=2, mode='bilinear')
            out = self.to_rgb_64(out_64)
            
            out = ((1-alpha) * skip) + (alpha * out)
            out = self.tanh(out)
            return out

        if crossover_point and 6 >= crossover_point:
            w = w2
        out_128 = self.block_128x128(w, out_64)
        if layer_num == 6:
            skip = self.to_rgb_64(out_64)
            skip = F.interpolate(skip, scale_factor=2, mode='bilinear')
            out = self.to_rgb_128(out_128)
            
            out = ((1-alpha) * skip) + (alpha * out)
            out = self.tanh(out)
            return out

        if crossover_point and 7 >= crossover_point:
            w = w2
        out_256 = self.block_256x256(w, out_128)
        if layer_num == 7:
            skip = self.to_rgb_128(out_128)
            skip = F.interpolate(skip, scale_factor=2, mode='bilinear')
            out = self.to_rgb_256(out_256)
            
            out = ((1-alpha) * skip) + (alpha * out)
            out = self.tanh(out)
            return out