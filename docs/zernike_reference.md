Noll Index (j),名称 (Aberration),极坐标公式 (Zj​),物理直观
1,Piston (平移),1,整体相位移动 (不影响成像质量)
2,Tilt X (倾斜),2ρcosθ,图像位置平移
3,Tilt Y (倾斜),2ρsinθ,图像位置平移
4,Defocus (离焦),3​(2ρ2−1),最核心项：光斑变大变圆
5,Astigmatism 1 (像散),6​ρ2sin(2θ),光斑变椭圆 (45度方向)
6,Astigmatism 2 (像散),6​ρ2cos(2θ),光斑变椭圆 (水平/垂直)
7,Coma 1 (彗差),8​(3ρ3−2ρ)sinθ,彗星状拖尾 (Y轴)
8,Coma 2 (彗差),8​(3ρ3−2ρ)cosθ,彗星状拖尾 (X轴)
11,Spherical (球差),5​(6ρ4−6ρ2+1),光晕/柔焦效果

| **Noll Index (j)** | **名称 (Aberration)** | **极坐标公式 (Zj)**                    | **物理直观**                  |
   | ------------------ | --------------------- | -------------------------------------- | ----------------------------- |
   | **1**              | Piston (平移)         | $1$                                    | 整体相位移动 (不影响成像质量) |
   | **2**              | Tilt X (倾斜)         | $2\rho \cos\theta$                     | 图像位置平移                  |
   | **3**              | Tilt Y (倾斜)         | $2\rho \sin\theta$                     | 图像位置平移                  |
   | **4**              | **Defocus (离焦)**    | $\sqrt{3}(2\rho^2 - 1)$                | **最核心项**：光斑变大变圆    |
   | **5**              | Astigmatism 1 (像散)  | $\sqrt{6}\rho^2 \sin(2\theta)$         | 光斑变椭圆 (45度方向)         |
   | **6**              | Astigmatism 2 (像散)  | $\sqrt{6}\rho^2 \cos(2\theta)$         | 光斑变椭圆 (水平/垂直)        |
   | **7**              | Coma 1 (彗差)         | $\sqrt{8}(3\rho^3 - 2\rho) \sin\theta$ | 彗星状拖尾 (Y轴)              |
   | **8**              | Coma 2 (彗差)         | $\sqrt{8}(3\rho^3 - 2\rho) \cos\theta$ | 彗星状拖尾 (X轴)              |
   | **11**             | **Spherical (球差)**  | $\sqrt{5}(6\rho^4 - 6\rho^2 + 1)$      | **光晕/柔焦效果**             |


### python:

import torch
import numpy as np

def compute_zernike_basis(n_modes, height, width, device='cuda'):
    """
    预计算 Zernike 基函数 (Noll Indices 1 to n_modes)
    返回: tensor shape [n_modes, height, width]
    """
    # 1. 生成归一化网格 (-1 到 1)
    linspace_x = torch.linspace(-1, 1, width, device=device)
    linspace_y = torch.linspace(-1, 1, height, device=device)
    Y, X = torch.meshgrid(linspace_y, linspace_x, indexing='ij')

    # 2. 转换为极坐标 (rho, theta)
    rho = torch.sqrt(X**2 + Y**2)
    theta = torch.atan2(Y, X)

    # 创建孔径掩膜 (只保留单位圆内的值)
    mask = (rho <= 1.0).float()
    rho = rho * mask # 避免圆外数值溢出导致计算错误

    zernikes = []
    
    # 3. 硬编码 Noll Index 1-15 的公式 (最快且最稳健)
    # j=1: Piston
    zernikes.append(torch.ones_like(rho)) 
    
    # j=2: Tilt X
    zernikes.append(2 * rho * torch.cos(theta))
    
    # j=3: Tilt Y
    zernikes.append(2 * rho * torch.sin(theta))
    
    # j=4: Defocus (离焦)
    zernikes.append(torch.sqrt(torch.tensor(3.)) * (2 * rho**2 - 1))
    
    # j=5: Astigmatism (45 deg)
    zernikes.append(torch.sqrt(torch.tensor(6.)) * rho**2 * torch.sin(2*theta))
    
    # j=6: Astigmatism (0 deg)
    zernikes.append(torch.sqrt(torch.tensor(6.)) * rho**2 * torch.cos(2*theta))
    
    # j=7: Coma (Y)
    zernikes.append(torch.sqrt(torch.tensor(8.)) * (3*rho**3 - 2*rho) * torch.sin(theta))
    
    # j=8: Coma (X)
    zernikes.append(torch.sqrt(torch.tensor(8.)) * (3*rho**3 - 2*rho) * torch.cos(theta))
    
    # j=9: Trefoil (Y)
    zernikes.append(torch.sqrt(torch.tensor(8.)) * rho**3 * torch.sin(3*theta))
    
    # j=10: Trefoil (X)
    zernikes.append(torch.sqrt(torch.tensor(8.)) * rho**3 * torch.cos(3*theta))
    
    # j=11: Spherical (球差) - 重点关注
    zernikes.append(torch.sqrt(torch.tensor(5.)) * (6*rho**4 - 6*rho**2 + 1))
    
    # ... 后续高阶项可按需添加，通常前11-15项足矣
    
    # 堆叠并应用掩膜
    basis = torch.stack(zernikes[:n_modes], dim=0) # [N, H, W]
    basis = basis * mask.unsqueeze(0) # 圆外置零
    
    return basis, mask

# 使用示例
# C 是网络输出的系数 [Batch, N_modes]
# Z 是预计算的基函数 [N_modes, H, W]
# Wavefront Phi = torch.einsum('bn, nhw -> bhw', C, Z)