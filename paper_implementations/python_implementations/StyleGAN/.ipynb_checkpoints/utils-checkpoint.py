import torch
import torchvision

from torchvision.datasets import ImageFolder
from torchvision import datasets, transforms, utils
from torch.utils.data import DataLoader

import numpy as np
from tqdm import tqdm

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.backends.cuda.is_built():
    device = torch.device("cuda")
else:
    device = ("cpu")

def show_images(images, num_images=16, figsize=(10,10)):
    images = images.cpu().detach()    
    images = (images + 1) / 2
    images = torch.clamp(images, 0, 1)
    
    grid = torchvision.utils.make_grid(images[:num_images], nrow=4)
    grid = grid.numpy().transpose((1, 2, 0))
    
    plt.figure(figsize=figsize)
    plt.imshow(grid)
    plt.axis('off')
    plt.show()

def EMA(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)    

def get_dataloader(image_size, batch_size=8):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = ImageFolder(root='./celeba_hq_256', transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)

    return dataloader

def get_batch_size(resolution):
    batch_sizes = {
        4: 128,
        8: 128,
        16: 128,
        32: 64,
        64: 32,
        128: 16,
        256: 8,
    }
    return batch_sizes.get(resolution) 

def get_total_iters(resolution):
    total_iters = {
        4: 50000,
        8: 80000,
        16: 100000,
        32: 150000,
        64: 150000,
        128: 200000,
        256: 225000,
    }
    return total_iters.get(resolution)

def get_params_with_lr(model):
    mapping_params = []
    other_params = []
    for name, param in model.named_parameters():
        if 'mapping' in name:
            mapping_params.append(param)
        else:
            other_params.append(param)
    return mapping_params, other_params

def resize(images):
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
    ])
    return transform(images)

def add_fake_images(fid, g_running, num_images, batch_size, latent_dim, device, layer_num, alpha=1):
    g_running.eval()

    with torch.no_grad():
        for _ in tqdm(range(0, num_images, batch_size), desc="Generating images"):
            z = torch.randn(batch_size, latent_dim, device=device)
            batch_images = g_running(z, layer_num=layer_num, alpha=alpha)
        
            resize_batch = resize(batch_images)
            resize_batch = ((resize_batch + 1) * 127.5).clamp(0, 255)
            resize_batch = resize_batch.to(torch.uint8)
            
            fid.update(resize_batch, real=False)
            
            torch.cuda.empty_cache()

def add_real_imgs(fid, data_loader):
    for batch in tqdm(data_loader, desc="Processing real images"):
        imgs, _ = batch
        imgs = resize(imgs)
        imgs = imgs.to(device)
        imgs = ((imgs + 1) * 127.5).clamp(0, 255)
        imgs = imgs.to(torch.uint8)
        fid.update(imgs, real=True)

# Final function which combines both of the image adding functions
def calculate_and_save_fid(fid, layer_num, iteration, data_loader, g_running, num_fake_images, batch_size, latent_dim, device, fid_file, alpha=1):
    fid.reset()
    add_fake_images(fid, g_running, num_fake_images, batch_size, latent_dim, device, layer_num, alpha=alpha)
    add_real_imgs(fid, data_loader)

    fid_score = fid.compute()
    print(f"FID score for layer {layer_num}, iteration {iteration}: {fid_score.item()}")
    
    with open(fid_file, 'a') as f:
        f.write(f"Layer {layer_num}, Iteration {iteration}: {fid_score.item()}\n")