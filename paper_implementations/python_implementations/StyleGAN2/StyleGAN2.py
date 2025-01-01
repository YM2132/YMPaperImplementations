# Thanks for checking out my code and hopefully my blog @ ym2132.github.io!
# A note on this code, it is not the cleanest piece of code but I want to retain some of my mistakes for future reference and so others 
# who embark on a similar journey may learn from my mistakes. Anything which is commented out is something that I did wrong and changed,
# the file will run and the model will train irrespective of these mistakes. Have fun training your StyleGAN2!!!


# StyleGAN training is heavily dependant on FFHQ, see https://gwern.net/face#compute and https://github.com/l4rz/practical-aspects-of-stylegan2-training?tab=readme-ov-file
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import torch
import torchvision

import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image
from torchvision import datasets, transforms, utils

from torch.utils.data import DataLoader

from torchmetrics.image.fid import FrechetInceptionDistance

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from datetime import datetime
from math import sqrt
import math
import sys
import random


# We can make use of a GPU if you have one on your computer. This works for Nvidia and M series GPU's
if torch.backends.mps.is_available():
    device = torch.device("mps")
    # These 2 lines assign some data on the memory of the device and output it. The output confirms
    # if we have set the intended device
    x = torch.ones(1, device=device)
    print (x)
elif torch.backends.cuda.is_built():
    device = torch.device("cuda")
    x = torch.ones(1, device=device)
    print (x)
else:
    device = ("cpu")
    x = torch.ones(1, device=device)
    print (x)
    
# I also define a function we use to examine the outputs of the Generator
def show_images(images, num_images=16, figsize=(10,10)):
    # Ensure the input is on CPU
    images = images.cpu().detach()
    
    # Normalize images from [-1, 1] to [0, 1]
    images = (images + 1) / 2
    
    # Clamp values to [0, 1] range
    images = torch.clamp(images, 0, 1)
    
    # Make a grid of images
    grid = torchvision.utils.make_grid(images[:num_images], nrow=4)
    
    # Convert to numpy and transpose
    grid = grid.numpy().transpose((1, 2, 0))
    
    # Display the grid
    plt.figure(figsize=figsize)
    plt.imshow(grid)
    plt.axis('off')
    plt.show()

# Credit: https://github.com/rosinality/stylegan2-pytorch/blob/master/model.py#L94
class EqualLRConv2d(nn.Module):
    def __init__(
        self, in_channel, out_channel, kernel_size, stride=1, padding=0, bias=True
    ):
        super().__init__()

        self.weight = nn.Parameter(
            torch.randn(out_channel, in_channel, kernel_size, kernel_size)
        )
        self.scale = 1 / math.sqrt(in_channel * kernel_size ** 2)

        self.stride = stride
        self.padding = padding

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channel))

        else:
            self.bias = None

    def forward(self, input):
        out = F.conv2d(
            input,
            self.weight * self.scale,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
        )

        return out

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]},"
            f" {self.weight.shape[2]}, stride={self.stride}, padding={self.padding})"
        )

class EqualLRLinear(nn.Module):
    def __init__(
        self, in_dim, out_dim, bias=True, bias_init=0, lr_mul=0.01, activation=None  # lr_mul from rosinality 
    ):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))

        else:
            self.bias = nn.Parameter(torch.zeros(out_dim))

        self.activation = activation

        self.scale = (1 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, input):
        if self.activation:  # WE NEVER USE THIS, fused_leaky_relu is not defined
            out = F.linear(input, self.weight * self.scale)
            out = fused_leaky_relu(out, self.bias * self.lr_mul)
        else:
            out = F.linear(
                input, self.weight * self.scale, bias=self.bias * self.lr_mul
            )

        return out

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]})"
        )

def EMA(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay) 

class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True)
                                  + 1e-8)

class MiniBatchStdDev(nn.Module):
    # try 32 as it seems to come from batch_size / num_gpu
    def __init__(self, group_size=4):  # 8 from the paper lets test it # But also 4 when they have batch size of 32 group_size=4 
        super().__init__()
        self.group_size = group_size
    
    def forward(self, x):
        N, C, H, W = x.shape 
        G = min(self.group_size, N) 
        
        y = x.view(G, -1, C, H, W)
        
        y = y - torch.mean(y, dim=0, keepdim=True)
        y = torch.mean(torch.square(y), dim=0)
        y = torch.sqrt(y + 1e-8)
        
        y = torch.mean(y, dim=[1,2,3], keepdim=True)
        
        y = y.repeat(G, 1, H, W)
        
        return torch.cat([x,y], dim=1)

class LearnedConstant(nn.Module):
    def __init__(self, in_c):
        super().__init__()
        
        self.constant = nn.Parameter(torch.randn(1, in_c, 4, 4))
        
    def forward(self, batch_size):
        return self.constant.expand(batch_size, -1, -1, -1)

class NoiseLayer(nn.Module):
    def __init__(self, channels):
        super().__init__()
        
        self.weight = nn.Parameter(torch.zeros(1, channels, 1, 1))
    
    def forward(self, gen_image, noise=None):
        if noise is None:
            N, _, H, W = gen_image.shape
            noise = torch.randn(N, 1, H, W, device=gen_image.device)
        
        return gen_image + (noise * self.weight)

class MappingNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.norm = PixelNorm()
        
        # I set output to 512 to match the shortened G case
        self.layers = nn.Sequential(
            EqualLRLinear(512, 512),
            nn.LeakyReLU(0.2),
            EqualLRLinear(512, 512),
            nn.LeakyReLU(0.2),
            EqualLRLinear(512, 512),
            nn.LeakyReLU(0.2),
            EqualLRLinear(512, 512),
            nn.LeakyReLU(0.2),
            EqualLRLinear(512, 512),
            nn.LeakyReLU(0.2),
            EqualLRLinear(512, 512),
            nn.LeakyReLU(0.2),
            EqualLRLinear(512, 512),
            nn.LeakyReLU(0.2),
            EqualLRLinear(512, 512),
            nn.LeakyReLU(0.2),
        )
    
    def forward(self, x):
        # Normalise input latent
        x = self.norm(x)
        
        out = self.layers(x)
        
        return out
    
class Blur(nn.Module):
    def __init__(self):
        super().__init__()
        #f = torch.Tensor([1, 2, 1])
        f = torch.Tensor([1, 3, 3, 1])
        self.register_buffer('f', f)

    def forward(self, x):
        f = self.f
        f = f[None, None, :] * f[None, :, None]
        f = f / f.sum()
        return F.conv2d(x, f.expand(x.size(1), -1, -1, -1), 
                        groups=x.size(1), padding=1)

'''class mod_demod(nn.Module):
    def __init__(self, latent_dim, out_channels):
        super().__init__()
    
        self.map_layer = EqualLRLinear(latent_dim, out_channels, 1)  # bias init of 1
        self.bias = nn.Parameter(torch.ones(1, out_channels, 1, 1))

        self.out_channels = out_channels
    def forward(self, w, conv_weight, batch):
        # conv_weight is the weights of the current convolutional layer
        s = self.map_layer(w)
        s = s.view(w.size(0), -1, 1, 1, 1)
        # Add bias and 1 (as per StyleGAN2)
        s = s + 1 + self.bias.view(1, -1, 1, 1, 1)
        # Modulation
        weight = conv_weight * s

        # Demodulation
        demod = torch.rsqrt(weight.pow(2).sum([2,3,4], keepdim=True) + 1e-8)
        weight = weight * demod

        # Return the weight's of the convolution
        #return nn.Parameter(weight)
        # returning nn.Parameter(weight) caused the latent to have no grad in the ppl regularisation func?!?!
        return weight'''

'''class mod_demod(nn.Module):
    def __init__(self, latent_dim, in_channels):
        super().__init__()
        
        # Change to map to input channels and handle bias init in EqualLRLinear
        self.modulation = EqualLRLinear(latent_dim, in_channels, bias_init=1)
    
    def forward(self, w, conv_weight):
        batch = w.size(0)
        
        # Get style scaling factors - no need for extra bias or +1
        style = self.modulation(w).view(batch, 1, -1, 1, 1)
        
        # Scale weights
        weight = conv_weight.unsqueeze(0) * style
        
        # Demodulation
        demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8)
        weight = weight * demod.unsqueeze(2).unsqueeze(3).unsqueeze(4)
        
        return weight'''

'''class Conv2d_mod(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, latent_dim, padding=1):
        super().__init__()

        # The weights for our conv
        self.weights = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))

        #self.bias = nn.Parameter(torch.ones(1, out_channels, 1, 1))  # b2 in diagram
        
        self.modulator = mod_demod(latent_dim, out_channels)

        self.eq_lr_scale = sqrt(1 / (in_channels * kernel_size ** 2))

        self.padding = kernel_size // 2
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.latent_dim = latent_dim

    def forward(self, x, style_w):
        batch, channels, H, W = x.shape

        weights = self.weights * self.eq_lr_scale
        # Need to create the modulated and demodulated weights
        weights = self.modulator(style_w, weights, batch)

        #print(f'Channels in x: {channels} | self.in_channels: {self.in_channels} | self.out_channels: {self.out_channels}')

        #print(weights.shape)
        weights = weights.view(batch * self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)

        x = x.view(1, batch*channels, H, W)
        #print(f'x.shape: {x.shape} | weights.shape: {weights.shape}')
        out = F.conv2d(x, weights, groups=batch, padding=self.padding)
        out = out.view(batch, self.out_channels, out.shape[-2], out.shape[-1])

        #out += self.bias
            
        return out'''

'''class Conv2d_mod(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, latent_dim, padding):
        super().__init__()
        
        # Initialize the convolution weight with an extra dimension for batch processing
        self.weight = nn.Parameter(
            torch.randn(1, out_channels, in_channels, kernel_size, kernel_size)
        )
        
        # Calculate the scaling factor for equalized learning rate
        self.scale = 1 / math.sqrt(in_channels * kernel_size ** 2)
        
        # Create the modulation/demodulation layer
        self.modulator = mod_demod(latent_dim, in_channels)
        
        # Store all necessary dimensions as instance variables
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.latent_dim = latent_dim
        
        # Calculate proper padding to maintain spatial dimensions
        self.padding = kernel_size // 2

    def forward(self, x, style):
        batch, channels, height, width = x.shape

        weight = self.weight * self.scale
        
        # Apply modulation and demodulation to weights
        weight = self.modulator(style, self.weight)
        
        # Reshape weight for grouped convolution
        weight = weight.view(
            batch * self.out_channels, self.in_channels, 
            self.kernel_size, self.kernel_size
        )
        
        # Reshape input for grouped convolution
        x = x.view(1, batch * channels, height, width)
        
        # Perform the convolution
        out = F.conv2d(
            x, 
            weight,
            padding=self.padding,
            groups=batch
        )
        
        # Reshape output back to proper dimensions
        _, _, height, width = out.shape
        out = out.view(batch, self.out_channels, height, width)
        
        return out'''

class Conv2d_mod(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, latent_dim, padding):
        super().__init__()
        
        # Weight initialization
        self.weight = nn.Parameter(
            torch.randn(1, out_channels, in_channels, kernel_size, kernel_size)
        )
        
        # Scale for equalized learning rate
        self.scale = 1 / math.sqrt(in_channels * kernel_size ** 2)
        
        # Style modulation layer (moved from mod_demod)
        self.modulation = EqualLRLinear(latent_dim, in_channels, bias_init=1)
        
        # Store dimensions
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

    def forward(self, x, style):
        batch, channels, height, width = x.shape
        
        # Style modulation
        style = self.modulation(style).view(batch, 1, -1, 1, 1)
        
        # Scale weights and apply style
        weight = self.scale * self.weight * style
        
        # Demodulation
        demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8)
        weight = weight * demod.view(batch, self.out_channels, 1, 1, 1)
        
        # Reshape for grouped convolution
        weight = weight.view(
            batch * self.out_channels, self.in_channels, 
            self.kernel_size, self.kernel_size
        )
        
        # Reshape input and perform convolution
        x = x.view(1, batch * channels, height, width)
        out = F.conv2d(
            x, 
            weight,
            padding=self.padding,
            groups=batch
        )
        
        # Reshape output
        _, _, height, width = out.shape
        out = out.view(batch, self.out_channels, height, width)
        
        return out

class g_style_block(nn.Module):
    def __init__(
        self, 
        in_c, 
        out_c, 
        ksize1, 
        padding,
        upsample=True,
        latent_dim=512,
    ):
        super().__init__()

        layers_list = []

        if upsample:
            layers_list.extend([
                nn.Upsample(scale_factor=2, mode='bilinear'),
                Conv2d_mod(in_c, out_c, ksize1, latent_dim, padding=padding),  # Will need to provide style_w and x
                NoiseLayer(out_c),
            ])
        else:
            self.learned_constant = LearnedConstant(in_c)

        layers_list.extend([
            nn.LeakyReLU(0.2),
            Conv2d_mod(out_c, out_c, ksize1, latent_dim, padding=padding),
            NoiseLayer(out_c),
            nn.LeakyReLU(0.2),
        ])

        self.layers = nn.ModuleList(layers_list)
        self.upsample = upsample
        
    def forward(self, w, x=None):
        if not self.upsample:
            x = self.learned_constant(w.size(0))
        
        for layer in self.layers:
            #print(layer)
            #print(f'x.shape: {x.shape}')
            if isinstance(layer, LearnedConstant):
                x = layer()
            elif isinstance(layer, Conv2d_mod):
                x = layer(x, w)
            else:
                x = layer(x)
            
        return x


class Generator(nn.Module):
    def __init__(self, in_c=512):
        super().__init__()
        
        self.g_mapping = MappingNetwork()

        fmaps = 0.5
        in_c = int(in_c * fmaps)
        
        self.block_4x4 = g_style_block(in_c, in_c, 3, 1, upsample=False)
        self.block_8x8 = g_style_block(in_c, in_c, 3, 1)
        self.block_16x16 = g_style_block(in_c, in_c, 3, 1)
        self.block_32x32 = g_style_block(in_c, in_c, 3, 1)
        self.block_64x64 = g_style_block(in_c, in_c//2, 3, 1)
        self.block_128x128 = g_style_block(in_c//2, in_c//4, 3, 1)
        self.block_256x256 = g_style_block(in_c//4, in_c//4, 3, 1)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        
        self.to_rgb_4 = EqualLRConv2d(in_c, 3, 1)
        self.to_rgb_8 = EqualLRConv2d(in_c, 3, 1)
        self.to_rgb_16 = EqualLRConv2d(in_c, 3, 1)
        self.to_rgb_32 = EqualLRConv2d(in_c, 3, 1)
        self.to_rgb_64 = EqualLRConv2d(in_c//2, 3, 1)
        self.to_rgb_128 = EqualLRConv2d(in_c//4, 3, 1)
        self.to_rgb_256 = EqualLRConv2d(in_c//4, 3, 1)
                
        self.tanh = nn.Tanh()

    '''def forward(self, z, return_latents=False):   
        w = self.g_mapping(z)
        
        if torch.rand(1).item() < 0.9:
            z2 = torch.randn_like(z)
            w2 = self.g_mapping(z2)

            #crossover_point = random.randint(1, 7)
            crossover_point = random.randint(1, 6)
        else:
            crossover_point = None

        w_to_use = w2 if crossover_point and 1 >= crossover_point else w
        out = self.block_4x4(w_to_use)      
        
        out_4 = self.to_rgb_4(out)
        out_4 = self.upsample(out_4)

        w_to_use = w2 if crossover_point and 2 >= crossover_point else w
        out = self.block_8x8(w_to_use, out)
        
        out_8 = self.to_rgb_8(out)
        out_8 += out_4 * (1 / np.sqrt(2))
        out_8 = self.upsample(out_8)
            
        w_to_use = w2 if crossover_point and 3 >= crossover_point else w
        out = self.block_16x16(w_to_use, out)
        
        out_16 = self.to_rgb_16(out)
        out_16 += out_8 * (1 / np.sqrt(2))
        out_16 = self.upsample(out_16)

        w_to_use = w2 if crossover_point and 4 >= crossover_point else w
        out = self.block_32x32(w_to_use, out)
        
        out_32 = self.to_rgb_32(out)
        out_32 += out_16 * (1 / np.sqrt(2))
        out_32 = self.upsample(out_32)

        w_to_use = w2 if crossover_point and 5 >= crossover_point else w
        out = self.block_64x64(w_to_use, out)
        
        out_64 = self.to_rgb_64(out)
        out_64 += out_32 * (1 / np.sqrt(2))
        out_64 = self.upsample(out_64)

        w_to_use = w2 if crossover_point and 6 >= crossover_point else w    
        out = self.block_128x128(w_to_use, out)
        
        out_128 = self.to_rgb_128(out)
        out_128 += out_64 * (1 / np.sqrt(2))
        #out_128= self.upsample(out_128)

        #w_to_use = w2 if crossover_point and 7 >= crossover_point else w
        #out = self.block_256x256(w_to_use, out)
            
        #out_256 = self.to_rgb_256(out)
        #out_256 += out_128 * (1 / np.sqrt(2))
        
        #if return_latents:
        #    return out_256, w_to_use
        #else:
        #    return out_256
        
        if return_latents:
            return out_128, w_to_use
        else:
            return out_128'''

    def forward(self, z, return_latents=False):
        w = self.g_mapping(z)
        batch_size = z.size(0)
        
        # Determine which samples undergo style mixing
        mixing = torch.rand(batch_size, device=z.device) < 0.9
        
        # Generate z2 and w2 for samples that undergo style mixing
        z2 = torch.randn_like(z)
        w2 = self.g_mapping(z2)
        
        # Generate crossover points for each sample
        crossover_points = torch.randint(1, 7, (batch_size,), device=z.device)
        
        # Initialize a list to hold the styles for each layer
        styles = []
        
        # For each layer, select w or w2 based on crossover points
        for layer_idx in range(7):  # Assuming 7 layers
            # Create a mask for samples that should use w2 at this layer
            use_w2 = mixing & (crossover_points <= layer_idx)
            # Select w or w2 for each sample in this layer
            style = torch.where(use_w2.unsqueeze(1), w2, w)
            styles.append(style)
        
        # Apply styles in generator blocks
        out = self.block_4x4(styles[0])
        out_4 = self.to_rgb_4(out)
        out_4 = self.upsample(out_4)
        
        out = self.block_8x8(styles[1], out)
        out_8 = self.to_rgb_8(out)
        out_8 += out_4 * (1 / np.sqrt(2))
        out_8 = self.upsample(out_8)
        
        out = self.block_16x16(styles[2], out)
        out_16 = self.to_rgb_16(out)
        out_16 += out_8 * (1 / np.sqrt(2))
        out_16 = self.upsample(out_16)
        
        out = self.block_32x32(styles[3], out)
        out_32 = self.to_rgb_32(out)
        out_32 += out_16 * (1 / np.sqrt(2))
        out_32 = self.upsample(out_32)
        
        out = self.block_64x64(styles[4], out)
        out_64 = self.to_rgb_64(out)
        out_64 += out_32 * (1 / np.sqrt(2))
        out_64 = self.upsample(out_64)

        out = self.block_128x128(styles[5], out)            
        out_128 = self.to_rgb_128(out)
        out_128 += out_64 * (1 / np.sqrt(2))
        out_128 = self.upsample(out_128)

        out = self.block_256x256(styles[6], out)
        out_256 = self.to_rgb_128(out)
        out_256 += out_128 * (1 / np.sqrt(2))
        
        if return_latents:
            return out_256, styles[6]
        else:
            return out_256
        

class d_style_block(nn.Module):
    def __init__(
        self,
        in_c,
        out_c,
        ksize1, 
        padding,  
        ksize2=None, 
        padding2=None,
        stride=None,   
        mbatch=False,
    ):
        super().__init__()

        if ksize2 is None:
            ksize2 = ksize1
        if padding2 is None:
            padding2 = padding
        
        layers_list = []

        # TODO Add ConvBlurPool (strided conv with blur) - Not sure if needed???
        self.down_res = nn.Sequential(
            Blur(),
            EqualLRConv2d(in_c, out_c, 3, padding = 1, stride=2),
            EqualLRConv2d(out_c, out_c, 2, padding=0, stride=1) if mbatch else nn.Identity()  # Needed for last layer otherwise res 
            # has two H, W instead of 1
        )
        
        if mbatch:
            layers_list.extend([
                MiniBatchStdDev(),
            ])
            in_c += 1
            
        layers_list.extend([
            EqualLRConv2d(in_c, in_c, ksize1, padding=padding),
            nn.LeakyReLU(0.2),
            EqualLRConv2d(in_c, out_c, ksize2, padding=padding2),
            nn.LeakyReLU(0.2),
        ])

        if not mbatch:  # No downsample if we are on the last layer
            layers_list.extend([
                EqualLRConv2d(out_c, out_c, 3, padding=1, stride=2)
            ])
        
        self.layers = nn.ModuleList(layers_list)
    
    def forward(self, x):
        res = self.down_res(x)
        for layer in self.layers:
            #print(layer)
            #print(f'x.shape: {x.shape}')
            x = layer(x)

        #print(f'res.shape: {res.shape} | x.shape: {x.shape}')
        out = (res + x) * (1/ sqrt(2))       
        
        return out

class Discriminator(nn.Module):
    def __init__(self, out_c=512):
        super().__init__()

        fmaps = 0.5
        out_c = int(out_c * fmaps)

        self.from_rgb = EqualLRConv2d(3, out_c//4, 1)

        self.block_256x256 = d_style_block(out_c//4, out_c//4, 3, 1)
        self.block_128x128 = d_style_block(out_c//4, out_c//2, 3, 1)
        self.block_64x64 = d_style_block(out_c//2, out_c, 3, 1)
        self.block_32x32 =  d_style_block(out_c, out_c, 3, 1)
        self.block_16x16 = d_style_block(out_c, out_c, 3, 1)
        self.block_8x8 = d_style_block(out_c, out_c, 3, 1)
        self.block_4x4 = d_style_block(out_c, out_c, 3, 1, 4, 0, mbatch=True)

        self.linear = EqualLRLinear(out_c, 1)

    def forward(self, x):
        
        out = self.from_rgb(x)

        out = self.block_256x256(out)
        out = self.block_128x128(out)
        out = self.block_64x64(out)
        out = self.block_32x32(out)
        out = self.block_16x16(out)
        out = self.block_8x8(out)
        out = self.block_4x4(out)

        #print(out.shape)
        out = out.view(out.size(0), -1)
        #print(out.shape)
        out = self.linear(out)

        return out

# R1 regulariser uses the Goodfellow GAN training algorithm
def d_loss(real_pred, fake_pred):
    # Do this outside the func
    #real_pred = d(real_images)
    #fake_pred = d(gen_images)

    real_loss = F.softplus(-real_pred).mean()
    fake_loss = F.softplus(fake_pred).mean()

    d_loss = real_loss + fake_loss

    return d_loss

def r1_loss(real_pred, real_images):
    with torch.set_grad_enabled(True):        
        # Get gradients with respect to inputs only
        grad_real, = torch.autograd.grad(
            outputs=real_pred.sum(),
            inputs=real_images,
            grad_outputs=torch.ones([], device=real_pred.device),
            create_graph=True,
            only_inputs=True  # Only compute gradients for inputs, not weights
        )
    #grad_real, = torch.autograd.grad(
    #    outputs=real_pred.sum(), inputs=real_images, create_graph=True
    #)
    grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()

    return grad_penalty

def g_loss(fake_pred):
    fake_loss = F.softplus(-fake_pred).mean()

    return fake_loss

# Credit: https://github.com/rosinality/stylegan2-pytorch/blob/master/train.py#L87
def g_path_regularize(fake_img, latents, mean_path_length, decay=0.01):
    noise = torch.randn_like(fake_img) / math.sqrt(
        fake_img.shape[2] * fake_img.shape[3]
    )
    
    grad, = torch.autograd.grad(
        outputs=(fake_img * noise).sum(), inputs=latents, create_graph=True,
        retain_graph=True, only_inputs=True
    )
    
    path_lengths = torch.sqrt(grad.pow(2).sum(1).mean(0))
    path_mean = mean_path_length + decay * (path_lengths.mean() - mean_path_length)
    path_penalty = (path_lengths - path_mean).pow(2).mean()

    del noise, grad
    torch.cuda.empty_cache()
    
    return path_penalty, path_mean.detach(), path_lengths

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag

def get_dataloader(image_size, batch_size=8):
    # We only call this function once now, unlike in the StyleGAN where it would be required that we change resolution of the dataset when 
    # the network grew
    transform = transforms.Compose([
        #transforms.Resize((image_size, image_size)),  # No longer need to resize the images
        transforms.RandomHorizontalFlip(p=0.5),  # Add a random flip
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Lets now use FFHQ 
    dataset = ImageFolder(root='./ffhq256_imgs', transform=transform)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=1,
        drop_last=True,
        pin_memory=True,
        persistent_workers=True
    )

    return dataloader

# To get the params for the mapping network we look for parameters with "mapping" in the name
def get_params_with_lr(model):
    mapping_params = []
    other_params = []
    for name, param in model.named_parameters():
        if 'mapping' in name:  # Adjust this condition based on your actual naming convention
            mapping_params.append(param)
        else:
            other_params.append(param)
    return mapping_params, other_params

# Now sample 30k fake and add them to fid
def resize(images):
    # Resize to 299x299, the inception v3 model expects 299,299 images so we just resize our images to
    # this size
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
    ])
    return transform(images)

# We make use of the FID class provided by the torchmetrics library provided by PyTorch Lightning
# This class works by "adding" fake and real images to the model
# So I create two functions, one for adding fake images and one for real images
def add_fake_images(g_running, num_images, batch_size, latent_dim, device):
    # The function takes in g_running and all params needed to generate images
    # we use g_running in this function as it is the model we use to output our fake images
    g_running.eval()
    
    # Set torch.no_grad to turn off gradients, it makes the code run faster and use less memory as gradients
    # aren't tracked
    with torch.no_grad():
        # Generate 70000 images to pass to the FID model, we pass them as we generate them to save
        # images
        for _ in tqdm(range(0, num_images, batch_size), desc="Generating images"):
            z = torch.randn(batch_size, latent_dim, device=device)
            batch_images = g_running(z)

            torch.cuda.synchronize()
        
            # resize images
            resize_batch = resize(batch_images)
            # Inception v3 requires pixel ranges to be [0,255] currently it's [-1,1], 
            # this line handles the conversion
            resize_batch = ((resize_batch + 1) * 127.5).clamp(0, 255)
            # Inception v3 also expects input data type to be uint8, this can be handled with a simple cast
            resize_batch = resize_batch.to(torch.uint8)
            
            # Update FID
            fid.update(resize_batch, real=False)
            
            # Clear GPU cache, to save memory
            torch.cuda.empty_cache()

# The second function just takes in the data_loader as a parm
def add_real_imgs(data_loader):
    # we pass all batches to the FID model i.e. 70k images
    for batch in tqdm(data_loader, desc="Processing real images"):
        imgs, _ = batch
        # Resize, convert to [0,255] range and cast to uint8 as before
        imgs = resize(imgs)
        imgs = imgs.to(device)
        imgs = ((imgs + 1) * 127.5).clamp(0, 255)
        imgs = imgs.to(torch.uint8)
        fid.update(imgs, real=True)

        del imgs
        torch.cuda.empty_cache()

# Final function which combines both of the image adding functions
def calculate_and_save_fid(iteration, data_loader, g_running, num_fake_images, batch_size, latent_dim, device, fid_file):
    # We reset the FID score statistics for recalculation
    fid.reset()
    # Add the real and fake images
    add_fake_images(g_running, num_fake_images, batch_size, latent_dim, device)
    add_real_imgs(data_loader)
    # Compute FID score and output it
    fid_score = fid.compute()
    print(f"FID score for iteration {iteration}: {fid_score.item()}")
    
    # We also save the scores to a file
    with open(fid_file, 'a') as f:
        f.write(f"Iteration {iteration}: {fid_score.item()}\n")

# Create new checkpoint dir with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
run_dir = os.path.join('./checkpoints', f'run_{timestamp}')
checkpoint_dir = os.path.join(run_dir, f"checkpoint_{timestamp}")
sample_dir = os.path.join(run_dir, f"sample_{timestamp}")
# Create a file to log FID scores
fid_file = os.path.join(run_dir, 'fid.txt')

os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(sample_dir, exist_ok=True)

# Init models
g = Generator().to(device)
d = Discriminator().to(device)
g_running = Generator().to(device)

g_running.train(False)

fid = FrechetInceptionDistance(feature=2048).to(device)

mapping_params, other_params = get_params_with_lr(g)

# I was wrong about the batching. They used a batch size of 64 total in the StyleGAN2-ADA paper with 16 on each GPU
#lr = 0.002  # Not trying THIS YET 01/12/24
#lr = 0.00025

#lr = 0.001
lr = 0.0025

#lr = 0.00015625
#lr = 0.00016  #https://github.com/NVlabs/stylegan2-ada-pytorch/blob/main/train.py#L157
#lr = 0.00008
#lr = 0.0025  # Way too high. https://github.com/NVlabs/stylegan3/blob/main/docs/configs.md#total-training-time and https://medium.com/@jules.padova/stylegan-deep-dive-from-its-architecture-to-how-to-make-synthetic-humans-e02c242a8b7d
#lr = 0.00005  # 0.00008 is better
# Let's reduce LR after checkpoint 215k which has FID of 113
#lr = 0.00001 - DIDNT DO MUCH LR DECAY NOT THE ANSWER SWAY

# in the paper batch=64 and 8 gpus so 512 imgs total and lr=0.0025. # THIS IS WRONG in StyleGAN2 they use batch 32 with 4 on each GPU and in
# ADA they use batch=64 with 8 on each GPU

# Adjusting LR seemed to improve perf but the training would always break down at iter 25k
# having a lower LR would just make the breakdown smaller
mapping_lr = lr * 0.01

g_reg_freq = 4
d_reg_freq = 16

g_reg_adjustment = g_reg_freq / (g_reg_freq + 1)
d_reg_adjustment = d_reg_freq / (d_reg_freq + 1) 

g_optimizer = torch.optim.Adam([
    {'params': mapping_params, 'lr': mapping_lr},  # 0.01 * LR for mapping network
    {'params': other_params, 'lr': lr * g_reg_adjustment}  # Regular LR for other parts
], betas=(0.0 ** g_reg_adjustment, 0.99 ** g_reg_adjustment))
d_optimizer = torch.optim.Adam(d.parameters(), lr=lr*d_reg_adjustment, betas=(0.0 ** d_reg_adjustment, 0.99 ** d_reg_adjustment))

start_iter = 0
mean_path_length = 0

# You must manually define your checkpoint if you want to resume training.
#resume_checkpoint = ''
resume_checkpoint = False
if resume_checkpoint:
    if os.path.isfile(resume_checkpoint):
        print(f"=> loading checkpoint '{resume_checkpoint}'")
        checkpoint = torch.load(resume_checkpoint, weights_only=False)
        start_iter = checkpoint['iteration']
        g.load_state_dict(checkpoint['g_state_dict'])
        d.load_state_dict(checkpoint['d_state_dict'])
        g_running.load_state_dict(checkpoint['g_running_state_dict'])
        g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
        d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])
        mean_path_length = checkpoint['mean_path_length']
        print(f"=> loaded checkpoint '{resume_checkpoint}' ( iteration {start_iter})")
    else:
        print(f"=> no checkpoint found at '{resume_checkpoint}'")
else:
    print("Starting training from the beginning")

# Init EMA 
EMA(g_running, g, 0)
 
# We evaluate FID every 10k iterations - for more frequent updates
num_iters_for_eval = 10000

# We want to gen 70k fake images for FID calculation to match 30k real images
num_fake_images = 70000
latent_dim = 512  # Adjust based on your model's input size

# Define vars used within training loop
d_loss_val = None
g_loss_val = None
r1_loss_val = None

resolution = 256  # Resolution is always 256
batch_size = 16  # CHANGE to 32, 64 will run but is awfully slow - TRY 4???
total_iters = 25000 * 1000 // batch_size  # k imgs from paper
data_loader = get_dataloader(resolution, batch_size)
dataset = iter(data_loader)

# Init a progress bar
print(f'Training resolution: {resolution}x{resolution}, Batch size: {batch_size}')

if resume_checkpoint:
    remaining_iters = total_iters - start_iter  # Subtract what's been done
    # Create progress bar for remaining iterations but show correct absolute position
    pbar = tqdm(range(remaining_iters), initial=start_iter, total=total_iters)
else:
    pbar = tqdm(range(total_iters))

max_retries = 3

# Begin introducing layer phase
for i in pbar:
    #d.zero_grad()
    requires_grad(g, False)
    requires_grad(d, True)

    try:
        real_imgs, label = next(dataset)
    except (OSError, StopIteration):
        # If we reach the end of the dataset, we reintialise the iterable
        # basically starting again
        dataset = iter(data_loader)
        real_imgs, label = next(dataset)

    # Train D
    real_size = real_imgs.size(0)
    real_imgs = real_imgs.to(device)
    label = label.to(device)
    real_preds = d(real_imgs)
    # The line below implements a small weight penalty
    # It ensures the D loss isnt too far away from 0 preventing extreme outputs
    #real_preds = real_preds.mean() - 0.001 * (real_preds**2).mean()

    # Create gen images and gen preds
    z = torch.randn(real_size, latent_dim, device=device)
    gen_imgs = g(z)
    gen_preds = d(gen_imgs.detach())

    d_loss_val = d_loss(real_preds, gen_preds)

    d.zero_grad()
    d_loss_val.backward()
    d_optimizer.step()

    # Implement a way to run r1 reg every 16 batches

    d_r1_regularise = i % d_reg_freq == 0
    if d_r1_regularise:
        real_imgs.requires_grad_(True)

        real_preds = d(real_imgs)
        r1_loss_val = r1_loss(real_preds, real_imgs)
    
        #print(f"R1 loss value: {r1_loss_val.item()}")

        d.zero_grad()
        #gamma = 10
        #gamma = 1. # from https://github.com/NVlabs/stylegan2-ada-pytorch/blob/main/train.py#L157
        #gamma = 0.01
        gamma = 0.8192
        #gamma = 0.1024
        # *16 from appendix b
        r1_reg = ((gamma*0.5) * r1_loss_val * d_reg_freq + 0 * real_preds[0])
        r1_reg.backward()

        d_optimizer.step()
        torch.cuda.empty_cache()

    # Now lets train the Generator
    #g.zero_grad()
    requires_grad(g, True)
    requires_grad(d, False)

    z = torch.randn(real_size, latent_dim, device=device)
    gen_imgs = g(z)
    gen_preds = d(gen_imgs)

    g_loss_val = g_loss(gen_preds)

    g.zero_grad()
    g_loss_val.backward()
    g_optimizer.step()

    # PPL reg, r1 only used on D
    # PPL a metric for gen images only
    g_ppl_regularise = i % g_reg_freq == 0
    if g_ppl_regularise:
        z = torch.randn(real_size, latent_dim, device=device)
        gen_imgs, latents = g(z, return_latents=True)  # PPL calculation relies on latent vectors which produced imgs
        
        ppl_loss, mean_path_length, path_lengths = g_path_regularize(gen_imgs, latents, mean_path_length)

        #print(f"PPL loss value: {ppl_loss.item()}")

        g.zero_grad()
        
        ppl_loss = 2 * g_reg_freq * ppl_loss
        #ppl_loss += 0 * gen_imgs[0, 0, 0, 0]
        
        ppl_loss.backward()
        g_optimizer.step()
        torch.cuda.empty_cache()
        
    EMA(g_running, g, decay=0.999) # CHANGED EMA AND NEED TO TRY DIFF MBGROUP SIZE
    
    if i > 0 and i % num_iters_for_eval == 0:
        sample_z = torch.randn(16, latent_dim, device=device)
        sample_imgs_EMA = g_running(sample_z)
        save_image(sample_imgs_EMA, f'{sample_dir}/sample__iter_{i}.png', nrow=4, normalize=True)
        print(f'G_running images images after iter: {i+start_iter}')

        #calculate_and_save_fid(i, data_loader, g_running, num_fake_images, batch_size, latent_dim, device, fid_file)

        # Added a try and except loop for the FID calculation, I need to include the different errors I faced here too in a log for blog.
        for attempt in range(max_retries):
           try:
               calculate_and_save_fid(i, data_loader, g_running, num_fake_images, batch_size, latent_dim, device, fid_file)
               break  # If successful, exit retry loop
           except Exception as e:
               print(f"FID calculation failed (attempt {attempt + 1}/{max_retries})")
               print(f"Error: {str(e)}")
               if attempt == max_retries - 1:
                   print("Maximum retries reached. Skipping FID calculation for this iteration.")
               else:
                   print("Retrying...")
                   torch.cuda.empty_cache()
                   time.sleep(5)

        torch.save({
            'g_state_dict': g.state_dict(),
            'g_running_state_dict': g_running.state_dict(),
            'd_state_dict': d.state_dict(),
            'g_optimizer_state_dict': g_optimizer.state_dict(),
            'd_optimizer_state_dict': d_optimizer.state_dict(),
            'iteration': i,
            'mean_path_length': mean_path_length,
        }, f'{checkpoint_dir}/checkpoint_iter_{i}.pth')

with torch.no_grad():
    sample_z = torch.randn(16, latent_dim, device=device)
    sample_imgs = g(sample_z)
    sample_imgs_EMA = g_running(sample_z)
    print('G images')
    show_images(sample_imgs)
    print('G_running images')
    show_images(sample_imgs_EMA)

    calculate_and_save_fid('final', data_loader, g_running, num_fake_images, batch_size, latent_dim, device, fid_file)

# No need for stabilising period

torch.save({
    'g_state_dict': g.state_dict(),
    'g_running_state_dict': g_running.state_dict(),
    'd_state_dict': d.state_dict(),
    'g_optimizer_state_dict': g_optimizer.state_dict(),
    'd_optimizer_state_dict': d_optimizer.state_dict(),
    'iteration': i,
    # Important training statistics
    'mean_path_length': mean_path_length,
}, f'{checkpoint_dir}/completed_checkpoint_g_and_EMA.pth')

# Check 2 things 
# mod and demod ops
# the skip and recurrent aspect
# Do i need to add nograd to the reg parts? 

# 05/12/24
# Made changes to EqLR lr_mlp (set to 0.01). Made change the scaling factor for conv2d_mod_demod. Added bias_init=1 for EqLR in the Style op
# Added bias init in the else, so all EqLRLinear layers get a bias

# 25/12/24
# This training run has been plagued by errors such as seg faults and now a Fatal Python error: Segmentation fault during the FID calc 
# The training has required manual restarts and a lot of my time has been wasted waiting for errors (the main one was a syntax error by me)
# For the next model runs I should think about how to make them more robust

# 26/12/24
# Is there any list of common errors in ML training code? I should make a list if not