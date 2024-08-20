#| code-fold: true

# Before we continue lets set our inputs and configure the device for our model code
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

import matplotlib.pyplot as plt
import numpy as np
from tqdm.notebook import tqdm
from datetime import datetime
import os
from math import sqrt

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
    
# I choose to implement this as a PyTorch module
# In PyTorch you have a lot of freedom to create modules for many tasks
class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()
    
    # You see it's a relatively straightforward implementation
    # the term inside the sqrt is a summation over all feature maps
    # divided by all feature maps, a.k.a taking the mean!
    def forward(self, x):
        return x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True)
                                  + 1e-8)
    
# Once again we implement Md StdDev as a PyTorch module
class MiniBatchStdDev(nn.Module):
    def __init__(self, group_size=4):
        super().__init__()
        # Group size is from the github repo I linked to
        # It isn't discussed at all in the paper AFAIK
        self.group_size = group_size
    
    def forward(self, x):
        N, C, H, W = x.shape  # N = num feature maps, C = num channels, H = height, W = width
        G = min(self.group_size, N)  # the minibatch must be divisible by group_size
        
        # Here we split up X into groups
        # This line may be a little weird, expand the next code box to explore this line!
        y = x.view(G, -1, C, H, W)
        
        # The 3 following lines see us implement number 1 from the list above.
        y = y - torch.mean(y, dim=0, keepdim=True)
        y = torch.mean(torch.square(y), dim=0)
        y = torch.sqrt(y + 1e-8)
        
        # This is number 2
        y = torch.mean(y, dim=[1,2,3], keepdim=True)
        
        # Finally this is number 3
        y = y.repeat(G, 1, H, W)
        
        # We return the input x with an additional feature map
        return torch.cat([x,y], dim=1)
    
# This is the key class
# It applies EqLR not just at intialisation but dynamically throughout training
# This is done through a forward pre-hook, which essentially is a function
# which runs before the forward method
class EqualLR:
    # We init the EqualLR class with the name of the weight parameter
    def __init__(self, name):
        self.name = name
    
    def compute_weight(self, module):
        # getattr = get attribute, in weight we stored the original
        # in the apply method
        weight = getattr(module, self.name + '_orig')
        # weight.data.size(1) is the number of channels (for a conv layer)
        # or the number of input features (for a linear layer)
        # weight.data[0][0].numel() for a linear layer is 1 and for a conv layer
        # it is the kernel size*kernel_size.  
        fan_in = weight.data.size(1) * weight.data[0][0].numel()

        return weight * sqrt(2 / (fan_in))

    @staticmethod
    def apply(module, name):
        # We create an instance of EqualLR 
        fn = EqualLR(name)
        
        # The original weight is retrived from the layer
        weight = getattr(module, name)
        # The weight is deleted from the layer
        del module._parameters[name]
        # We register a new parameter with the name _orig
        # saving the original parameters
        module.register_parameter(name + '_orig', nn.Parameter(weight.data))
        # We call the pre-hook, which runs before the forward pass
        module.register_forward_pre_hook(fn)

        return fn

    def __call__(self, module, input):
        # We call compute weight before the forward pass
        # When registering the pre_hook this __call__ is what will be called
        weight = self.compute_weight(module)
        setattr(module, self.name, weight)


def equal_lr(module, name='weight'):
    # This is just a simple function to apply the EqualLR regime
    EqualLR.apply(module, name)

    return module

# We redefine the layers simply, initialising the weight with a normal 
# distribution and the biases as 0. 
# By using  *args and **kwargs we allow for full flexibility in these layers
# and we can pass the usual arguments we would to the regular PyTorch versions.
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
        
        # If upsample True, we add the upsample layer before the
        # next set of convolutional blocks
        if upsample:
            layers_list.extend([
                # Upscale, we use the nearest neighbour upsampling technique
                nn.Upsample(scale_factor=2, mode='nearest'),
            ])
        
        # The layers of the block are the same, regardless of if we use upsample or not
        layers_list.extend([
            EqualLRConv2d(in_c, out_c, ksize1, padding=padding),
            PixelNorm(),
            nn.LeakyReLU(0.2),
            EqualLRConv2d(out_c, out_c, ksize2, padding=padding2),
            PixelNorm(),
            nn.LeakyReLU(0.2),
        ])
        
        self.layers = nn.ModuleList(layers_list)
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
# Credit to: https://github.com/odegeasslbc/Progressive-GAN-pytorch/blob/master/progan_modules.py

class Generator(nn.Module):
    def __init__(self, in_c=256):
        super().__init__()
        # The init method of G resembles the Gulrajani
        # The major difference is the addition of multiple to_rgb layers
        # This is to facilitate the growing
        
        self.in_c = in_c
        
        self.block_4x4 = G_ConvBlock(in_c, in_c, 4, 3, 3, 1, upsample=False)
        self.block_8x8 = G_ConvBlock(in_c, in_c, 3, 1)
        self.block_16x16 = G_ConvBlock(in_c, in_c, 3, 1)
        self.block_32x32 = G_ConvBlock(in_c, in_c, 3, 1)
        self.block_64x64 = G_ConvBlock(in_c, in_c//2, 3, 1)
        self.block_128x128 = G_ConvBlock(in_c//2, in_c//4, 3, 1)
        self.block_256x256 = G_ConvBlock(in_c//4, in_c//4, 3, 1)
        
        # no LeakyReLU on the to_RGBs
        self.to_rgb_4 = EqualLRConv2d(in_c, 3, 1)
        self.to_rgb_8 = EqualLRConv2d(in_c, 3, 1)
        self.to_rgb_16 = EqualLRConv2d(in_c, 3, 1)
        self.to_rgb_32 = EqualLRConv2d(in_c, 3, 1)
        self.to_rgb_64 = EqualLRConv2d(in_c//2, 3, 1)
        self.to_rgb_128 = EqualLRConv2d(in_c//4, 3, 1)
        self.to_rgb_256 = EqualLRConv2d(in_c//4, 3, 1)
                
        self.tanh = nn.Tanh()

    def forward(self, x, layer_num, alpha):  
        # Our forward method now takes some new parameters
        # layer_num corresponds to the current layer we are on
        # alpha is the value of the parameter alpha used in the introduction of 
        # new layers to the network
        
        # The first layer is simple, we just pass it through the 4x4 block 
        # and if we are currently on layer_1 we pass it through to_rgb and call it a day
        out_4 = self.block_4x4(x)
        if layer_num == 1:
            out = self.to_rgb_4(out_4)
            out = self.tanh(out)
            return out
        
        # The second layer and onwards are where things heat u
        # Being the second layer we have a previous layer available and it must be used in our progressive growing
        # We pass the out_4 through out_8 
        out_8 = self.block_8x8(out_4)
        if layer_num == 2:
            # if we are currently introducing layer 2 we must implement out formula from below
            # skip corresponds to skip in the formula, as seen in Figure 8 we operate on the outputs
            # passed through to_rgb layers
            skip = self.to_rgb_4(out_4)
            # skip is currently 4x4 images, for the formula to work dimensions must match
            # so we upsample skip to beome 8x8 images, matching the dimensions of out_8 
            skip = F.interpolate(skip, scale_factor=2, mode='bilinear')
            # We also pass out_8 through to_rgb
            out = self.to_rgb_8(out_8)
            
            # Here is our formula, note here I use one out
            # "out = " this one is the out and "* out" is the out_new
            out = ((1-alpha) * skip) + (alpha * out)
            # Our outputs are passed through a tanh, this is to put the values in the range [-1,1]
            out = self.tanh(out)
            return self.tanh(out)
        
        # This scheme continues for all subsequent layers
        out_16 = self.block_16x16(out_8)
        if layer_num == 3:
            skip = self.to_rgb_8(out_8)
            skip = F.interpolate(skip, scale_factor=2, mode='bilinear')
            out = self.to_rgb_16(out_16)
            
            out = ((1-alpha) * skip) + (alpha * out)
            out = self.tanh(out)
            return out
        
        out_32 = self.block_32x32(out_16)
        if layer_num == 4:
            skip = self.to_rgb_16(out_16)
            skip = F.interpolate(skip, scale_factor=2, mode='bilinear')
            out = self.to_rgb_32(out_32)
            
            out = ((1-alpha) * skip) + (alpha * out)
            out = self.tanh(out)
            return out
        
        out_64 = self.block_64x64(out_32)
        if layer_num == 5:
            skip = self.to_rgb_32(out_32)
            skip = F.interpolate(skip, scale_factor=2, mode='bilinear')
            out = self.to_rgb_64(out_64)
            
            out = ((1-alpha) * skip) + (alpha * out)
            out = self.tanh(out)
            return out
        
        out_128 = self.block_128x128(out_64)
        if layer_num == 6:
            skip = self.to_rgb_64(out_64)
            skip = F.interpolate(skip, scale_factor=2, mode='bilinear')
            out = self.to_rgb_128(out_128)
            
            out = ((1-alpha) * skip) + (alpha * out)
            out = self.tanh(out)
            return out
        
        out_256 = self.block_256x256(out_128)
        if layer_num == 7:
            skip = self.to_rgb_128(out_128)
            skip = F.interpolate(skip, scale_factor=2, mode='bilinear')
            out = self.to_rgb_256(out_256)
            
            out = ((1-alpha) * skip) + (alpha * out)
            out = self.tanh(out)
            return out

g = Generator().to(device)
    
class Discriminator(nn.Module):
    def __init__(self, out_c=256):
        super().__init__()
        
        # So in that light, we init a ModuleList with the first
        # convblock having expecting the same number of channels as 
        # the last G output. The first from_rgb will provide us with
        # 64 channels.
        # Also, the last block in this ModuleList will be where we start
        # from. I include some numbers on the right to show the order
        # of growth in the D model (these are index numbers, this makes it easier
        # to understand the forward pass)
        self.blocks = nn.ModuleList([
            D_ConvBlock(out_c//4, out_c//4, 3, 1),  # 6 we finish here            
            D_ConvBlock(out_c//4, out_c//2, 3, 1),  # 5
            D_ConvBlock(out_c//2, out_c, 3, 1),  # 4
            D_ConvBlock(out_c, out_c, 3, 1),  # 3
            D_ConvBlock(out_c, out_c, 3, 1),  # 2
            D_ConvBlock(out_c, out_c, 3, 1),  # 1
            D_ConvBlock(out_c+1, out_c, 3, 1, 4, 0, mbatch=True),  # 0 we start here
        ])
        
        # from_rgb takes the RGB image outputted by the G and outputs the
        # number of channels needed for each layer
        self.from_rgb = nn.ModuleList([
            EqualLRConv2d(3, out_c//4, 1),
            EqualLRConv2d(3, out_c//4, 1),
            EqualLRConv2d(3, out_c//2, 1),
            EqualLRConv2d(3, out_c, 1),
            EqualLRConv2d(3, out_c, 1),
            EqualLRConv2d(3, out_c, 1),
            EqualLRConv2d(3, out_c, 1),
        ])
        # The number of layers is needed to implement the growing and skip formula.
        self.num_layers = len(self.blocks)
        
        # The final layer, this is always the final layer it takes the output
        # and converts it to a prediction
        self.linear = EqualLRLinear(out_c, 1)
    
    def forward(self, x, layer_num, alpha):
        # I use a different method of selecting layers here
        # We know at any given point what number of layers we will use
        # so we index over that number
        for i in reversed(range(layer_num)):
            # idx will give us the index into self.blocks, e.g. if we are 
            # at layer 4 idx starts at 7 - 3 - 1 = 3, then 4, 5, and ends at 6
            idx = self.num_layers - i - 1
            # i is an index which start at 0 and the count layer_num starts at 1 hence the +1
            # since i is inverted i+1 == layer_num means we are at the start of our forward loop
            # and as such we need to convert x from RGB. 
            if i+1 == layer_num:
                out = self.from_rgb[idx](x)
            # We always just pass through the blocks, this starts at the top level block
            # for the current number of layers. E.g. at layer_num=4 we start at index 3 of self.blocks
            out = self.blocks[idx](out)
            # i > 0 means we are not on the last block and for all but the last block
            # we must half the resolution
            if i > 0:
                # Half the resolution of the image
                out = F.interpolate(out, scale_factor=0.5, mode='bilinear')
                
                # as before i+1 == layer_num means we are the start of the forward pass
                # and 0 <= alpha < 1 means we are introducing a new layer
                if i+1 == layer_num and 0 <= alpha < 1:
                    # If you refer back to Figure 8 and observe how the growing works for
                    # the D model, the skip connection involves the input x
                    skip = F.interpolate(x, scale_factor=0.5, mode='bilinear')
                    skip = self.from_rgb[idx + 1](skip)
                    out = ((1 - alpha) * skip) + (alpha * out)
        
        # We flatten the output to be passed through the linear layer
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        
        return out

d = Discriminator().to(device)

# We init a function to calculate the EMA for our parameters.
# This is implemented using two model, one the training G and the other
# a G we do not train but only use to hold the weights after perfoming the EMA
def EMA(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(1 - decay, par2[k].data)

# First off let's load our data again
# The scheme is a little different as due to progressive growing
# Our real images will need to be resized depending upon which layer
# we're currently at

def get_dataloader(image_size, batch_size=4):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),  # Resize images to the required size
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = ImageFolder(root='./celeba_hq_256', transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)

    return dataloader

g = Generator().to(device)  # The G we train
d = Discriminator().to(device)
g_running = Generator().to(device)  # The G we maintain for creating samples

g_running.train(False)

# The LR is quite high but this is specified in the paper
# A high LR has the benefit of allowing us to explore more of the gradient space
g_optimizer = torch.optim.Adam(g.parameters(), lr=0.001, betas=(0.0, 0.99))
d_optimizer = torch.optim.Adam(d.parameters(), lr=0.001, betas=(0.0, 0.99))

EMA(g_running, g, 0)

# We use this list to create a dataloader with the correct resolution for imgs
img_size = [4, 8, 16, 32, 64, 128, 256]

layer_num = 1  # We start at layer 1
start_iter = 0  # We start at iteration 0, of course, we use iterations in this training loop not epochs

# The total number of iterations each phase of training will run for
total_iters = 50000  # The paper states 800k per phase but I go for 50k
# I think 800k is excessive, but feel free to increase the number of training iterations and see how it goes!

# Initialize lists to store loss values for statistics
g_losses = []
d_losses = []

# range 1-7, for 7 layers we will introduce
for layer_num in range(1, 8):
    alpha = 0  # Set alpha to 0 before we introduce each new layer
    
    resolution = img_size[layer_num-1]
    data_loader = get_dataloader(resolution)
    dataset = iter(data_loader)
    
    print(f'Training resolution: {resolution}x{resolution}')
    pbar = tqdm(range(total_iters))
    
    for i in pbar:
        d.zero_grad()
        
        try:
            real_imgs, label = next(dataset)
        except (OSError, StopIteration):
            # If we reach the end of the dataset, we reintialise the iterable
            # basically starting again
            dataset = iter(data_loader)
            real_imgs, label = next(dataset)
        
        # Train D
        # real_size keeps track of the batch_size 
        real_size = real_imgs.size(0)
        real_imgs = real_imgs.to(device)
        label = label.to(device)
        real_preds = d(real_imgs, layer_num=layer_num, alpha=alpha)
        # The line below implements a small weight penalty
        # It is stated in the paper and it's reason is to prevent the 
        # D loss being too far away from 0 preventing extreme outputs 
        real_preds = real_preds.mean() - 0.001 * (real_preds**2).mean()
        
        # sample input for G
        z = torch.randn(real_size, 256, 1, 1).to(device)
        
        gen_imgs = g(z, layer_num=layer_num, alpha=alpha)
        
        gen_preds = d(gen_imgs.detach(), layer_num=layer_num, alpha=alpha)
        gen_preds = gen_preds.mean()  # Make gen_preds a signle value
        
        # Gradient Penalty - GP
        eps = torch.rand((real_size, 1, 1, 1)).to(device)
        x_hat = (eps * real_imgs) + ((1-eps) * gen_imgs.detach())
        x_hat.requires_grad_(True)
        pred_x_hat = d(x_hat, layer_num=layer_num, alpha=alpha)

        WGAN = gen_preds.mean() - real_preds.mean()
        grads = torch.autograd.grad(
            outputs=pred_x_hat, inputs=x_hat,
            grad_outputs=torch.ones_like(pred_x_hat),
            create_graph=True, retain_graph=True
        )[0]
        GP = ((grads.norm(2, dim=1) - 1) ** 2).mean() 

        # WGAN_GP_loss = d_loss
        d_loss = WGAN + 10 * GP
        d_loss.backward()
        d_optimizer.step()

        d_losses.append(d_loss.detach())
        
        # Now lets train the Generator
        g.zero_grad()
        z = torch.randn(real_size, 256, 1, 1).to(device)
        gen_imgs = g(z, layer_num=layer_num, alpha=alpha)
        gen_preds = d(gen_imgs, layer_num=layer_num, alpha=alpha)
        g_loss = -gen_preds.mean()
        g_loss.backward()
        g_optimizer.step()

        g_losses.append(g_loss.detach())
        
        EMA(g_running, g)
    
        alpha += 1 / len(pbar)
        alpha = round(alpha, 2)
    
    # I will show images both after we have introduced the new layer
    # and after we have stabilised it
    with torch.no_grad():
        sample_z = torch.randn(8, 256, 1, 1).to(device)
        sample_imgs = g(sample_z, layer_num=layer_num, alpha=alpha)
        sample_imgs_EMA = g_running(sample_z, layer_num=layer_num, alpha=alpha)
        print(f'Images after introducing layer: {layer_num}')
        print('G images')
        show_images(sample_imgs)
        print('G_running images')
        show_images(sample_imgs_EMA)
    
    
    stabilise_pbar = tqdm(range(total_iters))
    # To stabilise we just run the whole thing again, I do it this
    # way for simplicity
    for i in stabilise_pbar:
        d.zero_grad()
        
        try:
            real_imgs, label = next(dataset)
        except (OSError, StopIteration):
            dataset = iter(data_loader)
            real_imgs, label = next(dataset)
        
        # Train D
        # real_size keeps track of the batch_size 
        real_size = real_imgs.size(0)
        real_imgs = real_imgs.to(device)
        label = label.to(device)
        real_preds = d(real_imgs, layer_num=layer_num, alpha=alpha)
        # The line below implements a small weight penalty
        # It is stated in the paper and it's reason is to prevent the 
        # D loss being too far away from 0 preventing extreme outputs 
        real_preds = real_preds.mean() - 0.001 * (real_preds**2).mean()
        
        # sample input for G
        z = torch.randn(real_size, 256, 1, 1).to(device)
        
        gen_imgs = g(z, layer_num=layer_num, alpha=alpha)
        
        gen_preds = d(gen_imgs.detach(), layer_num=layer_num, alpha=alpha)
        gen_preds = gen_preds.mean()  # Make gen_preds a signle value
        
        # Gradient Penalty - GP
        eps = torch.rand((real_size, 1, 1, 1)).to(device)
        x_hat = (eps * real_imgs) + ((1-eps) * gen_imgs.detach())
        x_hat.requires_grad_(True)
        pred_x_hat = d(x_hat, layer_num=layer_num, alpha=alpha)

        WGAN = gen_preds.mean() - real_preds.mean()
        grads = torch.autograd.grad(
            outputs=pred_x_hat, inputs=x_hat,
            grad_outputs=torch.ones_like(pred_x_hat),
            create_graph=True, retain_graph=True
        )[0]
        GP = ((grads.norm(2, dim=1) - 1) ** 2).mean() 

        # WGAN_GP_loss = d_loss
        d_loss = WGAN + 10 * GP
        d_loss.backward()
        d_optimizer.step()

        d_losses.append(d_loss.detach())
        
        # Now lets train the Generator
        g.zero_grad()
        z = torch.randn(real_size, 256, 1, 1).to(device)
        gen_imgs = g(z, layer_num=layer_num, alpha=alpha)
        gen_preds = d(gen_imgs, layer_num=layer_num, alpha=alpha)
        g_loss = -gen_preds.mean()
        g_loss.backward()
        g_optimizer.step()

        g_losses.append(g_loss.detach())
        
        EMA(g_running, g)
    
    with torch.no_grad():
        sample_z = torch.randn(8, 256, 1, 1).to(device)
        sample_imgs = g(sample_z, layer_num=layer_num, alpha=alpha)
        sample_imgs_EMA = g_running(sample_z, layer_num=layer_num, alpha=alpha)
        print(f'Images after stabilising layer: {layer_num}')
        print('G images')
        show_images(sample_imgs)
        print('G_running images')
        show_images(sample_imgs_EMA)
    
# Assuming d_losses and g_losses are lists of GPU tensors
d_losses_cpu = [loss.cpu().detach().numpy() for loss in d_losses]
g_losses_cpu = [loss.cpu().detach().numpy() for loss in g_losses]

plt.figure(figsize=(10, 5))
plt.plot(d_losses_cpu, label='Discriminator Loss')
plt.plot(g_losses_cpu, label='Generator Loss')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.legend()
plt.title('Generator and Discriminator Loss Over Time in Full Case')
plt.show()