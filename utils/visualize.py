'''
visualize.py 的作用: 可视化模型的内部工作过程——帮助理解
1. PSF 核在不同位置的形状变化
2. Zernike 系数的空间分布
━━━━━━━━━━━━━━━━━━━

1. plot_psf_grid()
   ├─ 采样 5×5 位置的 PSF
   ├─ 展示 PSF 如何随位置变化
   └─ 用对数缩放突出细节

2. plot_coefficient_maps()
   ├─ 采样 128×128 密集网格
   ├─ 显示 4 个关键 Zernike 系数的空间分布
   └─ 揭示网络学到的像差模式

用途:
━━━━
物理层参数
    │
    ├─ PSF 核 → 可视化为网格
    └─ Zernike 系数 → 可视化为热力图
    
用于调试和论文展示
• 调试: 检查网络是否学到了物理约束
• 可视化: 在论文中展示空间变化的模糊
• 监控: 观察训练过程中的变化
'''
import torch
import numpy as np
import matplotlib.pyplot as plt
import os

def plot_psf_grid(physical_layer, H, W, device, filename='psf_grid.png'):
    """
    Plots a grid of PSFs sampled from across the image field.
    在图像不同位置采样 PSF 核，展示 PSF 如何随位置变化
    """
    H_pad = H 
    W_pad = W
    
    # Grid of coordinates
    # We can just manually query AberrationNet at specific points
    rows, cols = 5, 5
    y = torch.linspace(-0.9, 0.9, rows, device=device)
    x = torch.linspace(-0.9, 0.9, cols, device=device)
    yy, xx = torch.meshgrid(y, x, indexing='ij')
    coords = torch.stack([xx, yy], dim=-1).reshape(-1, 2) # [25, 2]
    
    with torch.no_grad():
        coeffs = physical_layer.aberration_net(coords)
        kernels = physical_layer.zernike_generator(coeffs) # [25, 1, K, K]
        
    kernels = kernels.cpu().squeeze(1).numpy()
    
    fig, axes = plt.subplots(rows, cols, figsize=(10, 10))
    for i in range(rows * cols):
        r, c = i // cols, i % cols
        ax = axes[r, c]
        k = kernels[i]
        
        # Log scale for better visibility of rings
        im = ax.imshow(np.log1p(k * 100), cmap='inferno')
        ax.axis('off')
        
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_coefficient_maps(physical_layer, H, W, device, filename='coeff_maps.png'):
    """
    Plots the spatial distribution of first few Zernike coefficients on a dense grid.
    在密集网格上采样 Zernike 系数，展示系数的空间变化
    """
    grid_size = 128
    y = torch.linspace(-1, 1, grid_size, device=device)
    x = torch.linspace(-1, 1, grid_size, device=device)
    yy, xx = torch.meshgrid(y, x, indexing='ij')
    coords = torch.stack([xx, yy], dim=-1).reshape(-1, 2)
    
    with torch.no_grad():
        coeffs = physical_layer.aberration_net(coords) # [G*G, N]
        
    coeffs = coeffs.reshape(grid_size, grid_size, -1).cpu().numpy()
    
    # Plot first 4 non-trivial coefficients (e.g. Defocus, Astig, Coma)
    # Noll 4: Defocus, 5,6 Astig, 7,8 Coma
    # Let's plot indices 3, 4, 5, 6 (0-indexed -> Noll 4,5,6,7)
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    labels = ["Defocus (Noll 4)", "Astig 1 (Noll 5)", "Astig 2 (Noll 6)", "Coma (Noll 7)"]
    indices = [3, 4, 5, 6] 
    
    for i, (idx, label) in enumerate(zip(indices, labels)):
        if idx < coeffs.shape[-1]:
            ax = axes.flatten()[i]
            im = ax.imshow(coeffs[..., idx], cmap='viridis')
            ax.set_title(label)
            plt.colorbar(im, ax=ax)
            ax.axis('off')
            
    plt.savefig(filename)
    plt.close()
