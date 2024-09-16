import torch

from torchvision.utils import save_image

from torchmetrics.image.fid import FrechetInceptionDistance

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from datetime import datetime
import os
from math import sqrt
import sys
import random

from models import *
from utils import *

if torch.backends.mps.is_available():
    device = torch.device("mps")
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

# Reset GPU
torch.cuda.empty_cache()
torch.cuda.ipc_collect()

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
run_dir = os.path.join('./checkpoints', f'run_{timestamp}')
checkpoint_dir = os.path.join(run_dir, f"checkpoint_{timestamp}")
sample_dir = os.path.join(run_dir, f"sample_{timestamp}")
fid_file = os.path.join(run_dir, 'fid.txt')

os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(sample_dir, exist_ok=True)

g = Generator().to(device)
d = Discriminator().to(device)
g_running = Generator().to(device)

g_running.train(False)

fid = FrechetInceptionDistance(feature=2048).to(device)

mapping_params, other_params = get_params_with_lr(g)

lr = 0.001
g_optimizer = torch.optim.Adam([
    {'params': mapping_params, 'lr': lr * 0.01},  
    {'params': other_params, 'lr': lr}  
], betas=(0.0, 0.99))
d_optimizer = torch.optim.Adam(d.parameters(), lr=lr, betas=(0.0, 0.99))

start_layer = 1
start_iter = 0

resume_checkpoint = '/home/yusuf/python/YMPaperImplementations/paper_implementations/python_implementations/StyleGAN/checkpoints/run_20240915_115437/checkpoint_20240915_115437/completed_checkpoint_g_and_EMA_layer_3.pth'
if resume_checkpoint:
    if os.path.isfile(resume_checkpoint):
        print(f"=> loading checkpoint '{resume_checkpoint}'")
        checkpoint = torch.load(resume_checkpoint, weights_only=False)
        start_layer = checkpoint['layer_num'] + 1
        start_iter = checkpoint['iteration'] + 1
        g.load_state_dict(checkpoint['g_state_dict'])
        d.load_state_dict(checkpoint['d_state_dict'])
        g_running.load_state_dict(checkpoint['g_running_state_dict'])
        g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
        d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])

        print(f"=> loaded checkpoint '{resume_checkpoint}' (layer {start_layer}, iteration {start_iter})")
    else:
        print(f"=> no checkpoint found at '{resume_checkpoint}'")
else:
    print("Starting training from the beginning")

EMA(g_running, g, 0)

img_size = [4, 8, 16, 32, 64, 128, 256]

num_iters_for_eval = 10000

num_fake_images = 30000
latent_dim = 256

for layer_num in range(start_layer, 8):
    alpha = 0

    resolution = img_size[layer_num-1]
    batch_size = get_batch_size(resolution)
    total_iters = get_total_iters(resolution)
    data_loader = get_dataloader(resolution, batch_size)
    dataset = iter(data_loader)

    print(f'Training resolution: {resolution}x{resolution}, Batch size: {batch_size}')
    if layer_num != 1:
        pbar = tqdm(range(total_iters))
    
        for i in pbar:
            d.zero_grad()
            
            try:
                real_imgs, label = next(dataset)
            except (OSError, StopIteration):
                dataset = iter(data_loader)
                real_imgs, label = next(dataset)
    
            real_size = real_imgs.size(0)
            real_imgs = real_imgs.to(device)
            label = label.to(device)
            real_preds = d(real_imgs, layer_num=layer_num, alpha=alpha)
            real_preds = real_preds.mean() - 0.001 * (real_preds**2).mean()
    
            z = torch.randn(real_size, latent_dim, device=device)
    
            gen_imgs = g(z, layer_num=layer_num, alpha=alpha)
    
            gen_preds = d(gen_imgs.detach(), layer_num=layer_num, alpha=alpha)
            gen_preds = gen_preds.mean() 
    
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
    
            d_loss = WGAN + 10 * GP
            d_loss.backward()
            d_optimizer.step()
    
            g.zero_grad()
            z = torch.randn(real_size, latent_dim, device=device)
            gen_imgs = g(z, layer_num=layer_num, alpha=alpha)
            gen_preds = d(gen_imgs, layer_num=layer_num, alpha=alpha)
            g_loss = -gen_preds.mean()
            g_loss.backward()
            g_optimizer.step()
    
            EMA(g_running, g)
    
            inc = (1-alpha) / (len(pbar)-i)
            alpha += inc   
    
            if i % num_iters_for_eval == 0:
                sample_z = torch.randn(16, latent_dim, device=device)
                sample_imgs_EMA = g_running(sample_z, layer_num=layer_num, alpha=alpha)
                save_image(sample_imgs_EMA, f'{sample_dir}/sample_intro_layer_{layer_num}_iter_{i}.png', nrow=4, normalize=True)

                print("Introducing layer")
                calculate_and_save_fid(fid, layer_num, i, data_loader, g_running, 
                                       num_fake_images, batch_size, latent_dim, device, fid_file, alpha=alpha)
    
        with torch.no_grad():
            sample_z = torch.randn(16, latent_dim, device=device)
            sample_imgs = g(sample_z, layer_num=layer_num, alpha=alpha)
            sample_imgs_EMA = g_running(sample_z, layer_num=layer_num, alpha=alpha)

            print("Introducing layer")
            calculate_and_save_fid(fid, layer_num, 'final intro', data_loader, g_running, 
                                   num_fake_images, batch_size, latent_dim, device, fid_file, alpha=alpha)

    stabilise_pbar = tqdm(range(total_iters))
    for i in stabilise_pbar:
        d.zero_grad()

        try:
            real_imgs, label = next(dataset)
        except (OSError, StopIteration):
            dataset = iter(data_loader)
            real_imgs, label = next(dataset)

        real_size = real_imgs.size(0)
        real_imgs = real_imgs.to(device)
        label = label.to(device)
        real_preds = d(real_imgs, layer_num=layer_num, alpha=alpha)
        real_preds = real_preds.mean() - 0.001 * (real_preds**2).mean()

        z = torch.randn(real_size, latent_dim, device=device)

        gen_imgs = g(z, layer_num=layer_num, alpha=alpha)

        gen_preds = d(gen_imgs.detach(), layer_num=layer_num, alpha=alpha)
        gen_preds = gen_preds.mean()

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

        d_loss = WGAN + 10 * GP
        d_loss.backward()
        d_optimizer.step()

        g.zero_grad()
        z = torch.randn(real_size, latent_dim, device=device)
        gen_imgs = g(z, layer_num=layer_num, alpha=alpha)
        gen_preds = d(gen_imgs, layer_num=layer_num, alpha=alpha)
        g_loss = -gen_preds.mean()
        g_loss.backward()
        g_optimizer.step()

        EMA(g_running, g)

        if i % num_iters_for_eval == 0:
            sample_z = torch.randn(16, latent_dim, device=device)
            sample_imgs_EMA = g_running(sample_z, layer_num=layer_num, alpha=alpha)
            save_image(sample_imgs_EMA, f'{sample_dir}/sample_stabilising_layer_{layer_num}_iter_{i}.png', nrow=4, normalize=True)

            print("Stabilising layer")
            calculate_and_save_fid(fid, layer_num, i, data_loader, g_running, num_fake_images, batch_size, latent_dim, device, fid_file, alpha=alpha)

    with torch.no_grad():
        sample_z = torch.randn(16, latent_dim, device=device)
        sample_imgs = g(sample_z, layer_num=layer_num, alpha=alpha)
        sample_imgs_EMA = g_running(sample_z, layer_num=layer_num, alpha=alpha)
        save_image(sample_imgs_EMA, f'{sample_dir}/sample_layer_{layer_num}_completed.png', nrow=4, normalize=True)

        print("Stabilising layer")
        calculate_and_save_fid(fid, layer_num, 'final_stable', data_loader, g_running, num_fake_images, batch_size, latent_dim, device, fid_file, alpha=alpha)

    torch.save({
        'g_state_dict': g.state_dict(),
        'g_running_state_dict': g_running.state_dict(),
        'd_state_dict': d.state_dict(),
        'g_optimizer_state_dict': g_optimizer.state_dict(),
        'd_optimizer_state_dict': d_optimizer.state_dict(),
        'layer_num': layer_num,
        'iteration': i
    }, f'{checkpoint_dir}/completed_checkpoint_g_and_EMA_layer_{layer_num}.pth')
