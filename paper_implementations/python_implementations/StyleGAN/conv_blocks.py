import torch.nn as nn

from layers import *

class D_ConvBlock(nn.Module):
    def __init__(
        self,
        in_c,
        out_c,
        ksize1, 
        padding, 
        ksize2=None, 
        padding2=None,
        stride=None,   
        mbatch=None,
    ):
        super().__init__()
        
        layers_list = []
        
        if ksize2 is None:
            ksize2 = ksize1
        if padding2 is None:
            padding2 = padding
        
        if mbatch:
            layers_list.extend([
                MiniBatchStdDev(),
            ])
            
        layers_list.extend([
            EqualLRConv2d(in_c, out_c, ksize1, padding=padding),
            nn.LeakyReLU(0.2),
            EqualLRConv2d(out_c, out_c, ksize2, padding=padding2),
            nn.LeakyReLU(0.2),
        ])
        
        self.layers = nn.ModuleList(layers_list)
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class G_ConvBlock(nn.Module):
    def __init__(
        self, 
        in_c, 
        out_c, 
        ksize1, 
        padding,
        ksize2=None, 
        padding2=None,
        stride=None, 
        upsample=True,
    ):
        super().__init__()
        
        layers_list = []

        if ksize2 is None:
            ksize2 = ksize1
        if padding2 is None:
            padding2 = padding

        if upsample:
            layers_list.extend([
                nn.Upsample(scale_factor=2, mode='bilinear'),
                EqualLRConv2d(in_c, out_c, ksize1, padding=padding),
                NoiseLayer(out_c),
                AdaIN(256, out_c),
            ])
        else:
            self.learned_constant = LearnedConstant(in_c)
            layers_list.extend([
                NoiseLayer(in_c),
                AdaIN(256, in_c),
            ])

        layers_list.extend([
            nn.LeakyReLU(0.2),
            EqualLRConv2d(out_c, out_c, ksize2, padding=padding2),
            NoiseLayer(out_c),
            AdaIN(256, out_c),
            nn.LeakyReLU(0.2),
        ])
        
        self.layers = nn.ModuleList(layers_list)
        self.upsample = upsample
        
    def forward(self, w, x=None):
        if not self.upsample:
            x = self.learned_constant(w.size(0))
        
        for layer in self.layers:
            if isinstance(layer, LearnedConstant):
                x = layer()
            elif isinstance(layer, AdaIN):
                x = layer(w, x)
            else:
                x = layer(x)
            
        return x