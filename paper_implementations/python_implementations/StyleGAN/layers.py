import torch
import torch.nn as nn

from math import sqrt

class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + 1e-8)

class MiniBatchStdDev(nn.Module):
    def __init__(self, group_size=16):
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

class EqualLR:
    def __init__(self, name):
        self.name = name
    
    def compute_weight(self, module):
        weight = getattr(module, self.name + '_orig') 
        fan_in = weight.data.size(1) * weight.data[0][0].numel()

        return weight * sqrt(2 / (fan_in))

    @staticmethod
    def apply(module, name):
        fn = EqualLR(name)
        
        weight = getattr(module, name)
        del module._parameters[name]
        module.register_parameter(name + '_orig', nn.Parameter(weight.data))
        module.register_forward_pre_hook(fn)

        return fn

    def __call__(self, module, input):
        weight = self.compute_weight(module)
        setattr(module, self.name, weight)


def equal_lr(module, name='weight'):
    EqualLR.apply(module, name)

    return module

class EqualLRConv2d(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

        conv = nn.Conv2d(*args, **kwargs)
        conv.weight.data.normal_()
        conv.bias.data.zero_()

        self.conv = equal_lr(conv)

    def forward(self, input):
        return self.conv(input)


class EqualLRConvTranspose2d(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

        conv = nn.ConvTranspose2d(*args, **kwargs)
        conv.weight.data.normal_()
        conv.bias.data.zero_()

        self.conv = equal_lr(conv)

    def forward(self, input):
        return self.conv(input)

class EqualLRLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        linear = nn.Linear(in_dim, out_dim)
        linear.weight.data.normal_()
        linear.bias.data.zero_()

        self.linear = equal_lr(linear)

    def forward(self, input):
        return self.linear(input)

class AdaIN(nn.Module):
    def __init__(self, latent_dim, current_c):
        super().__init__()

        self.map_layer = EqualLRLinear(latent_dim, current_c*2)
        self.IN = nn.InstanceNorm2d(current_c)

    def forward(self, w, image):
        style = self.map_layer(w)
        
        y_s, y_b = style.chunk(2, dim=1)
        
        y_s = y_s.unsqueeze(2).unsqueeze(3)
        y_b = y_b.unsqueeze(2).unsqueeze(3)

        return (y_s * self.IN(image)) + y_b

class NoiseLayer(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.weight = nn.Parameter(torch.zeros(1, channels, 1, 1))
    
    def forward(self, gen_image, noise=None):
        if noise is None:
            N, _, H, W = gen_image.shape
            noise = torch.randn(N, 1, H, W, device=gen_image.device)

        return gen_image + (noise * self.weight)

class LearnedConstant(nn.Module):
    def __init__(self, in_c):
        super().__init__()

        self.constant = nn.Parameter(torch.ones(1, in_c, 4, 4))

    def forward(self, batch_size):
        return self.constant.expand(batch_size, -1, -1, -1)