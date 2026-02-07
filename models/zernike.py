# 将 Zernike 像差系数 转换为 PSF (点扩散函数) 卷积核
'''
Zernike 系数 [a₁, a₂, ..., a₁₅]
        ↓ (物理光学计算)
波前相位 φ(x, y)
        ↓ (FFT)
点扩散函数 PSF (卷积核)
'''
import torch
import torch.nn as nn
import torch.fft
import torch.nn.functional as F
import numpy as np
'''
Zernike 系数 (从 AberrationNet)
    [B*N, 15]
        │
        ▼
        │
   ┌────────────────────────────────────────────┐
   │  DifferentiableZernikeGenerator.forward()  │
   └────────────────────────────────────────────┘
        │
        ├─ 步骤 1: ZernikeBasis(系数)
        │          ↓ 计算波前相位 φ [B, 64, 64]
        │
        ├─ 步骤 2: 多波长处理
        │  ┌─ 红光 (650nm)
        │  │   └─ φ_scale × (λ_ref / λ_R)
        │  ├─ 绿光 (550nm) [参考]
        │  │   └─ φ_scale × (λ_ref / λ_G)
        │  └─ 蓝光 (450nm)
        │      └─ φ_scale × (λ_ref / λ_B)
        │
        ├─ 步骤 3: 瞳孔函数
        │   P = A × exp(i×φ)
        │
        ├─ 步骤 4: 过采样 (2×)
        │   64 → 128
        │
        ├─ 步骤 5: FFT (Fourier Transform)
        │   |FFT(P)| → PSF 空间
        │
        ├─ 步骤 6: 下采样 (回到原大小)
        │   128 → 64 → 33×33
        │
        └─ 步骤 7: 归一化
            ∫∫ PSF(x,y) dxdy = 1
        
        ▼
    PSF 卷积核 [B*N, C, 33, 33]
    (如 [128, 3, 33, 33] for RGB)
        │
        ▼
    送入 physical_layer 进行卷积
'''
def noll_to_nm(j):
    """
    索引转换
    Map Noll index j to radial order n and azimuthal frequency m.
    Based on standard Noll numbering sequence.
    """
    if j == 1: return 0, 0
    
    # Solve for n: j = n(n+1)/2 + |m| + ... roughly
    # A bit complex to inverse analytically for all cases, using a lookup for common range
    # and a loop for general case is safer.
    
    # For j=1..15 lookup table:
    # 索引 1-15 对应最常见的 15 种像差
    # j: (n, m)
    mapping = {
        1: (0, 0),
        2: (1, 1), 3: (1, -1),
        4: (2, 0),
        5: (2, -2), 6: (2, 2),
        7: (3, -1), 8: (3, 1),
        9: (3, -3), 10: (3, 3),
        11: (4, 0),
        12: (4, 2), 13: (4, -2), # Note: j=13 is usually 4,-2 (sin)
        14: (4, 4), 15: (4, -4)
    }
    '''
a₁: Piston (活塞)          - 全局亮度偏移
a₂-a₃: Tilt (倾斜)         - 光束方向偏移
a₄: Defocus (离焦)         - 焦点偏移 ✓ 最常见
a₅-a₆: Astigmatism (像散)  - 两个方向焦距不同
a₇-a₈: Coma (彗差)        - 非对称模糊
a₉-a₁₀: Trefoil (三叶)    - 三重对称失真
a₁₁: Spherical (球差)      - 球形透镜失差
a₁₂-a₁₃: Secondary Astig   - 高阶像散
a₁₄-a₁₅: Quadrafoil       - 四重对称失真
    '''
    if j in mapping:
        return mapping[j]
        
    raise NotImplementedError(f"Noll index {j} not implemented in lookup yet.")

def zernike_radial(n, m, rho):
    """
    Compute Radial Zernike Polynomial R_n^|m|(rho).
    Using explicit formulas for low orders or recursion would be better.
    Here we implement explicit polynomials for n<=4 to cover Z1-Z15.
    """
    m = abs(m)
    if n == 0 and m == 0:
        return torch.ones_like(rho)
    if n == 1 and m == 1:
        return rho
    if n == 2 and m == 0:
        return 2 * rho**2 - 1
    if n == 2 and m == 2:
        return rho**2
    if n == 3 and m == 1:
        return 3 * rho**3 - 2 * rho
    if n == 3 and m == 3:
        return rho**3
    if n == 4 and m == 0:
        return 6 * rho**4 - 6 * rho**2 + 1
    if n == 4 and m == 2:
        return 4 * rho**4 - 3 * rho**2
    if n == 4 and m == 4:
        return rho**4
        
    # Validation/Fallback
    return torch.zeros_like(rho)

class ZernikeBasis(nn.Module):
    """
    Precomputes and stores Zernike basis functions on a grid.
    提前计算所有 15 个 Zernike 基函数在空间网格上的值，避免重复计算。
    """
    def __init__(self, n_modes=15, grid_size=64, device='cpu'):
        super().__init__()
        self.n_modes = n_modes
        self.grid_size = grid_size
        self.device = device
        
        # Create coordinate grid
        u = torch.linspace(-1, 1, grid_size, device=device)
        v = torch.linspace(-1, 1, grid_size, device=device)
        v_grid, u_grid = torch.meshgrid(v, u, indexing='ij')
        
        rho = torch.sqrt(u_grid**2 + v_grid**2)
        theta = torch.atan2(v_grid, u_grid)
        
        # Aperture mask
        self.mask = (rho <= 1.0).float()
        rho = rho * self.mask # Zero out rho outside
        
        basis = []
        # Precompute Z1 to Zn
        for j in range(1, n_modes + 1):
            n, m = noll_to_nm(j)
            
            # Normalization factor (Noll convention usually has sqrt(n+1) or similar?)
            # Noll: RMS = 1. 
            # Z_polar = sqrt(n+1) * R_nm(rho) * ...
            #   if m=0: * 1
            #   if m!=0, even j: * sqrt(2) cos(m theta)
            #   if m!=0, odd j:  * sqrt(2) sin(|m| theta)
            
            # Let's verify normalization from reference or standard Noll.
            # Ref j=4 (defocus): sqrt(3)*(2rho^2-1). n=2, sqrt(n+1)=sqrt(3). Correct.
            # Ref j=5 (astig): sqrt(6)*rho^2*sin(2theta). n=2, m=2. sqrt(n+1)=sqrt(3)? 
            #   Actually Noll normalization is sqrt(2)*sqrt(n+1) for m!=0. 
            #   sqrt(2)*sqrt(3) = sqrt(6). Correct.
            
            if m == 0:
                norm = np.sqrt(n + 1)
                term = zernike_radial(n, m, rho)
            else:
                norm = np.sqrt(2 * (n + 1))
                R = zernike_radial(n, m, rho)
                if j % 2 == 0: # Even j -> cos (usually, check mapping)
                    # mapping: 2->(1,1) cos; 3->(1,-1) sin.
                    # 6->(2,2) cos; 5->(2,-2) sin.
                    # 12->(4,2) cos; 13->(4,-2) sin.
                    # So Even j -> cos, Odd j (excluding m=0 cases) -> sin?
                    # Wait, j=3 (odd) is sin. j=5 (odd) is sin. 
                    # j=2 (even) is cos. j=6 (even) is cos.
                    # So yes: for m!=0: j even -> cos, j odd -> sin.
                    term = R * torch.cos(abs(m) * theta)
                else:
                    term = R * torch.sin(abs(m) * theta)
            
            Z = torch.tensor(norm, device=device) * term
            basis.append(Z)
            
        self.basis = torch.stack(basis, dim=0) # [N_modes, G, G]
        self.basis = self.basis * self.mask.unsqueeze(0)
        
        # Register buffer so it saves with state_dict but isn't a parameter
        self.register_buffer('zernike_basis', self.basis)
        self.register_buffer('aperture_mask', self.mask)

    def forward(self, coefficients):
        """
        coefficients: [B, N_modes]
        Returns: wavefront phase [B, G, G]
        """
        # [B, N, 1, 1] * [1, N, G, G] -> [B, N, G, G] -> sum -> [B, G, G]
        # or einsum
        return torch.einsum('bn,nhw->bhw', coefficients, self.zernike_basis)

class DifferentiableZernikeGenerator(nn.Module):
    def __init__(self, n_modes, pupil_size, kernel_size, 
                 oversample_factor=2, 
                 wavelengths=None, ref_wavelength=550e-9,
                 device='cpu'):
        """
        Args:
            n_modes: Zernike 模式数量
            pupil_size: 光瞳网格大小 (simulation grid size)
            kernel_size: 输出 PSF 卷积核的大小
            oversample_factor: 过采样因子 (default: 2)
            wavelengths: List of wavelengths [R, G, B] in meters. If None, mono (ref_wavelength).
            ref_wavelength: Reference wavelength for the coefficients.
            device: 计算设备
        """
        super().__init__()
        self.n_modes = n_modes
        self.pupil_size = pupil_size
        self.kernel_size = kernel_size
        self.oversample_factor = oversample_factor
        self.wavelengths = wavelengths if wavelengths is not None else [ref_wavelength]
        self.ref_wavelength = ref_wavelength
        
        # [Fix] Enforce odd kernel size for alignment
        if kernel_size % 2 == 0:
            raise ValueError(f"Kernel size must be odd to ensure physical alignment, got {kernel_size}")
        
        # 基础 Zernike Basis 依然在原始分辨率 pupil_size 上计算，节省显存
        self.basis = ZernikeBasis(n_modes, pupil_size, device)
        
    def forward(self, coefficients):
        """
        coefficients: [B, N_modes] (defined at ref_wavelength)
        Output: PSF kernels [B, C, K, K] where C = len(wavelengths)
        """
        # 1. 计算波前相位 (Reference Phase)
        # phi_ref = 2pi * OPD / lambda_ref
        # coefficients are in "waves" at lambda_ref => OPD = C * lambda_ref
        # phi_ref = 2pi * C
        phi_ref = 2 * torch.pi * self.basis(coefficients) # [B, G, G]
        
        # 2. Multi-wavelength Loop
        psf_channels = []
        
        for lam in self.wavelengths:
            # Scale phase: phi_lambda = phi_ref * (lambda_ref / lambda)
            scale = self.ref_wavelength / lam
            phi = phi_ref * scale
            
            # Pupil Function P = A * exp(i * phi)
            A = self.basis.aperture_mask
            pupil = A * torch.exp(1j * phi) # [B, G, G]
            
            # Oversampling
            '''
原始 FFT (64×64):
- 分辨率低
- PSF 边界有混叠 (aliasing)

过采样 (128×128):
- 2 倍分辨率
- 减少混叠
- 更准确的 PSF 边界

实际应用:
原始 → 过采样 → FFT → 下采样 → 裁剪
64    128      128    64       33
            '''
            pad_size = self.pupil_size * self.oversample_factor
            pad_total = pad_size - self.pupil_size
            p_l = pad_total // 2
            p_r = pad_total - p_l
            p_t = pad_total // 2
            p_b = pad_total - p_t
            
            pupil_padded = F.pad(pupil, (p_l, p_r, p_t, p_b), mode='constant', value=0)
            
            # FFT
            complex_field = torch.fft.ifftshift(pupil_padded, dim=(-2, -1))
            psf_complex = torch.fft.fft2(complex_field)
            psf_complex = torch.fft.fftshift(psf_complex, dim=(-2, -1))
            
            # Intensity
            psf_high_res = (psf_complex.abs()) ** 2
            '''
瞳孔函数 P(x,y)(复数)
    ↓ FFT
频域复数场
    ↓ |·|
振幅谱
    ↓ (·)²
强度 (PSF)
    
结果: 高斯-like 的亮点
┌──────────────┐
│              │
│    ╱╲╱╲      │
│  ╱      ╲    │
│ │  亮点  │    │  ← PSF 中心集中
│  ╲      ╱    │
│    ╲╱╲╱      │
│              │
└──────────────┘
            '''
            # Downsample：回到原大小
            # 下采样方法：平均池化 (Average Pooling)
            if self.oversample_factor > 1:
                # [Fix] Explicit dimension for pooling to avoid ambiguity
                # psf_high_res: [B, G, G] -> [B, 1, G, G]
                psf = F.avg_pool2d(psf_high_res.unsqueeze(1), 
                                 kernel_size=self.oversample_factor, 
                                 stride=self.oversample_factor).squeeze(1)
            else:
                psf = psf_high_res
                
            # Global Normalize
            psf = psf / (psf.sum(dim=(-2, -1), keepdim=True) + 1e-8)
            
            # Crop
            G = self.pupil_size
            K = self.kernel_size
            if K > G: raise ValueError(f"Kernel size {K} > Pupil size {G}")
            start = G // 2 - K // 2
            end = start + K
            psf_cropped = psf[:, start:end, start:end]
            
            # Re-normalize
            # PSF 代表单位光源的响应，应该守恒能量
            psf_cropped = psf_cropped / (psf_cropped.sum(dim=(-2, -1), keepdim=True) + 1e-8)
            
            psf_channels.append(psf_cropped)
            
        # Stack channels: [B, C, K, K]
        return torch.stack(psf_channels, dim=1)
    
'''
AberrationNet 输出
    │
    ▼
Zernike 系数 [B*N, 15]
(如 [128, 15])
    │
    ├─ a₁ (Piston)
    ├─ a₂ (Tilt-X)
    ├─ a₃ (Tilt-Y)
    ├─ a₄ (Defocus) ← 最重要
    ├─ a₅ (Astigmatism-45°)
    ├─ ...
    └─ a₁₅ (Quadrafoil)
    │
    ▼
DifferentiableZernikeGenerator.forward()
    │
    ├─ 调用 ZernikeBasis(coeffs)
    │  └─ einsum: aⱼ × Zⱼ → φ_ref [B, 64, 64]
    │
    ├─ 多波长循环 (R, G, B)
    │  │
    │  ├─ 红光 (650 nm)
    │  │  ├─ φ = φ_ref × (550/650)
    │  │  ├─ P = A × exp(i×φ)
    │  │  ├─ 过采样: 64 → 128
    │  │  ├─ FFT 计算 PSF
    │  │  ├─ 下采样: 128 → 64
    │  │  ├─ 裁剪: [64, 64] → [33, 33]
    │  │  └─ PSF_R [B, 33, 33]
    │  │
    │  ├─ 绿光 (550 nm)
    │  │  └─ PSF_G [B, 33, 33]
    │  │
    │  └─ 蓝光 (450 nm)
    │     └─ PSF_B [B, 33, 33]
    │
    ▼
堆叠 RGB 通道
    │
    ▼
PSF 卷积核 [B*N, 3, 33, 33]
(如 [128, 3, 33, 33])
    │
    ▼
返回到 physical_layer
进行 FFT 卷积
'''

