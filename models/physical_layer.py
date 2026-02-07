'''
核心功能架构:

输入图像 [B, C, H, W]
  ↓
分割成重叠补丁 (Overlap-Add 策略)
  ↓
对每个补丁中心计算坐标
  ↓
AberrationNet 预测该点的 Zernike 系数
  ↓
ZernikeGenerator 生成局部 PSF 卷积核
  ↓
FFT 频域卷积（高效计算）
  ↓
Hann 窗口加权拼接
  ↓
输出模糊图像 [B, C, H, W]
'''

'''
┌──────────────────────────────────────────────────────────────────────────────┐
│                    SpatiallyVaryingPhysicalLayer                              │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                                │
│  ┌─ 属性 (Attributes) ────────────────────────────────────────────────────┐  │
│  │  • aberration_net: AberrationNet                                       │  │
│  │  • zernike_generator: DifferentiableZernikeGenerator                   │  │
│  │  • patch_size (P): 128                                                 │  │
│  │  • stride (S): 64                                                      │  │
│  │  • kernel_size (K): 31                                                 │  │
│  │  • window: [128, 128] Hann 窗口 (缓冲)                                 │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
│                                                                                │
│  ┌─ 方法 (Methods) ───────────────────────────────────────────────────────┐  │
│  │                                                                         │  │
│  │  get_patch_centers(H, W, device)                                       │  │
│  │  ├─ 输入: 图像尺寸, 设备                                                │  │
│  │  └─ 输出: [N_patches, 2] 归一化坐标                                    │  │
│  │                                                                         │  │
│  │  forward(x_hat)                                                        │  │
│  │  ├─ 步骤 1: Pad(填充)                                                  │  │
│  │  ├─ 步骤 2: Unfold(分割补丁)                                           │  │
│  │  ├─ 步骤 3: Generate Kernels(生成 PSF)                                │  │
│  │  │  ├─ get_patch_centers()                                            │  │
│  │  │  ├─ AberrationNet(坐标) → 系数                                      │  │
│  │  │  └─ ZernikeGenerator(系数) → PSF                                    │  │
│  │  ├─ 步骤 4: FFT Conv(频域卷积)                                         │  │
│  │  ├─ 步骤 5: Window(窗口加权)                                           │  │
│  │  ├─ 步骤 6: Fold(拼接)                                                 │  │
│  │  ├─ 步骤 7: Normalize(归一化)                                          │  │
│  │  └─ 输出: y_hat [B, C, H, W]                                           │  │
│  │                                                                         │  │
│  └─────────────────────────────────────────────────────────────────────────┘  │
│                                                                                │
└──────────────────────────────────────────────────────────────────────────────┘


数据维度变化轨迹 (以 B=2, C=3, H=512, W=512 为例):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[2, 3, 512, 512]  x_hat (输入)
    ↓ Pad
[2, 3, 576, 576]  x_padded
    ↓ Unfold
[2, 3×128×128, 64]  patches_unfolded
    ↓ Reshape
[128, 3, 128, 128]  patches (B*N, C, P, P)

[64, 2]  coords (N_patches, 2)
    ↓ AberrationNet
[64, 15]  coeffs (N_patches, n_coeffs)
    ↓ ZernikeGenerator
[64, 3, 31, 31]  kernels (N_patches, C_k, K, K)

    patches ⊕ kernels (FFT 卷积)
    ↓
[128, 3, 128, 128]  y_patches_large
    ↓ Crop
[128, 3, 128, 128]  y_patches
    ↓ Window
[128, 3, 128, 128]  y_patches (加权)
    ↓ Reshape
[2, 3×128×128, 64]  y_patches_reshaped
    ↓ Fold
[2, 3, 576, 576]  y_accum
    ↓ Crop to [2, 3, 512, 512]
[2, 3, 512, 512]  y_hat (输出)
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
import math
from .zernike import DifferentiableZernikeGenerator
from .aberration_net import AberrationNet
from .newbp_convolution import NewBPConvolutionFunction
'''
┌─────────────────────────────────────────────────────────────┐
│ 输入: x_hat [B, C, H, W]                                      │
│ (如 [2, 3, 512, 512] - 批大小 2, RGB 图, 512x512 像素)      │
└──────────────────┬──────────────────────────────────────────┘
                   │
        ┌──────────▼──────────┐
        │  步骤 1: 填充 (Pad)  │
        └──────────┬──────────┘
                   │
        ┌──────────▼──────────────────────────────────────────┐
        │ x_padded [B, C, H_pad, W_pad]                       │
        │ (如 [2, 3, 576, 576] - 确保整除性)                  │
        └──────────┬──────────────────────────────────────────┘
                   │
        ┌──────────▼──────────┐
        │  步骤 2: Unfold     │
        │ (分割成补丁)        │
        └──────────┬──────────┘
                   │
        ┌──────────▼──────────────────────────────────────────┐
        │ patches [B*N, C, P, P]                              │
        │ (如 [2*64, 3, 128, 128] - 64 个补丁)               │
        └──────────┬──────────────────────────────────────────┘
                   │
        ┌──────────▼──────────────────────────────────────────┐
        │  步骤 3: 生成卷积核                                  │
        │  ├─ 计算补丁中心坐标                               │
        │  ├─ AberrationNet 预测 Zernike 系数                │
        │  └─ ZernikeGenerator 生成 PSF 核                   │
        └──────────┬──────────────────────────────────────────┘
                   │
        ┌──────────▼──────────────────────────────────────────┐
        │ kernels [B*N, C_k, K, K]                            │
        │ (如 [128, 3, 31, 31] - 每个补丁一个 PSF 核)        │
        └──────────┬──────────────────────────────────────────┘
                   │
        ┌──────────▼──────────┐
        │  步骤 4: FFT 卷积    │
        │ (频域相乘)          │
        └──────────┬──────────┘
                   │
        ┌──────────▼──────────────────────────────────────────┐
        │ y_patches [B*N, C_out, P, P]                        │
        │ (如 [128, 3, 128, 128] - 卷积后补丁)               │
        └──────────┬──────────────────────────────────────────┘
                   │
        ┌──────────▼──────────────────────────────────────────┐
        │  步骤 5: 窗口加权                                    │
        │ y_patches *= window_2d                              │
        │ (补丁边界平滑过渡)                                  │
        └──────────┬──────────────────────────────────────────┘
                   │
        ┌──────────▼──────────────────────────────────────────┐
        │  步骤 6: Fold (拼接)                                 │
        │  ├─ 输出拼接                                        │
        │  └─ 权重归一化                                      │
        └──────────┬──────────────────────────────────────────┘
                   │
        ┌──────────▼──────────┐
        │ 裁剪回原尺寸        │
        └──────────┬──────────┘
                   │
        ┌──────────▼──────────────────────────────────────────┐
        │ 输出: y_hat [B, C_out, H, W]                        │
        │ (模糊图像)                                          │
        └──────────────────────────────────────────────────────┘
'''
class SpatiallyVaryingPhysicalLayer(nn.Module):
    def __init__(self, 
                 aberration_net: nn.Module,
                 zernike_generator: DifferentiableZernikeGenerator,
                 patch_size,
                 stride,
                 pad_to_power_2=True,
                 use_newbp=False):
        super().__init__()
        self.aberration_net = aberration_net
        self.zernike_generator = zernike_generator
        self.patch_size = patch_size
        self.stride = stride
        self.kernel_size = zernike_generator.kernel_size
        self.pad_to_power_2 = pad_to_power_2
        self.use_newbp = use_newbp
        
        # Precompute window
        # Hann window 2D
        '''
        1D Hann 窗口:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1.0 |        ╱╲
    |       ╱  ╲
0.5 |      ╱    ╲
    |     ╱      ╲
0.0 |____╱________╲____ 
    0   32   64   96  128

特点:
• 起点 (n=0): w=0
• 中点 (n=64): w=1 (最大值)
• 终点 (n=127): w≈0
• 平滑过渡，无尖角


2D Hann 窗口 (补丁):
━━━━━━━━━━━━━━━━━
    ┌─────────────────┐
    │     亮 (1.0)    │ ← 中心
    │   ╱         ╲   │
    │  ╱           ╲  │
    │ ╱             ╲ │
    │╱_______________╲│
    │ 暗 (0)         │ ← 边缘
    └─────────────────┘
    
中心最亮，边缘逐渐变暗
        '''
        # Hann 窗口是一个从 0 开始，逐渐升到 1，最后又降到 0 的平滑曲线。
        # Hann 窗口的解决了:Overlap-Add 中的补丁拼接,导致重叠区域的像素被重复计算了两次，能量不守恒的问题。
        # 通过在每个补丁上应用 Hann 窗口，补丁的边缘部分会被平滑地衰减到零，从而在拼接时避免了重复计算的问题。
        # 这样在重叠区域，多个补丁的贡献会自然地加权平均，确保最终图像的亮度和对比度保持一致。
        hann = torch.hann_window(patch_size)
        window_2d = torch.outer(hann, hann)
        self.register_buffer('window', window_2d)

    def get_patch_centers(self, H, W, device, H_orig=None, W_orig=None, crop_info=None):
        """
        计算所有补丁的中心坐标，并归一化到 [-1, 1] 范围内。
        支持全局坐标对齐（Global Coordinate Alignment）。
        
        Args:
            H, W: 当前（可能已填充的）图像尺寸
            H_orig, W_orig: 原始图像尺寸（用于全局坐标计算）
            crop_info: [top/H_orig, left/W_orig, crop_h/H_orig, crop_w/W_orig] 
                      表示裁剪区域在原图中的归一化位置
        
        Returns:
            coords [N_patches, 2]: 归一化坐标 (y, x) in [-1, 1]，
                                   在全局坐标系中（如果提供了 crop_info）
        """
        
        # Calculate number of patches along H and W
        n_h = (H - self.patch_size) // self.stride + 1
        n_w = (W - self.patch_size) // self.stride + 1
        
        # Generate coordinates in local (padded image) space
        # Center of first patch: P/2
        # Center of second patch: P/2 + S
        y_centers_local = torch.arange(n_h, device=device) * self.stride + self.patch_size / 2
        x_centers_local = torch.arange(n_w, device=device) * self.stride + self.patch_size / 2
        
        # ============================================================================
        # Global Coordinate Transformation
        # 全局坐标变换：将局部补丁中心坐标还原为原图全局坐标
        # ============================================================================
        
        if crop_info is not None and H_orig is not None and W_orig is not None:
            # crop_info = [top_norm, left_norm, crop_h_norm, crop_w_norm]
            # 其中每个值都是相对于原图尺寸的归一化量
            
            crop_info = crop_info.to(device)
            top_norm, left_norm, crop_h_norm, crop_w_norm = crop_info
            
            # 还原像素坐标
            top_pix = top_norm * H_orig
            left_pix = left_norm * W_orig
            crop_h_pix = crop_h_norm * H_orig
            crop_w_pix = crop_w_norm * W_orig
            
            # 局部坐标 [0, H] 映射到全局坐标
            # y_global_pix = y_local + top_pix
            # x_global_pix = x_local + left_pix
            y_centers_global = y_centers_local + top_pix
            x_centers_global = x_centers_local + left_pix
            
            # 基于原图尺寸进行归一化到 [-1, 1]
            y_norm = (y_centers_global / H_orig) * 2 - 1
            x_norm = (x_centers_global / W_orig) * 2 - 1
        else:
            # 如果没有 crop_info，使用局部坐标（向后兼容）
            y_norm = (y_centers_local / H) * 2 - 1
            x_norm = (x_centers_local / W) * 2 - 1
        
        # Grid [N_h, N_w, 2]
        grid_y, grid_x = torch.meshgrid(y_norm, x_norm, indexing='ij')
        coords = torch.stack([grid_y, grid_x], dim=-1)  # [Nh, Nw, 2] (y, x) order
        
        return coords.reshape(-1, 2)  # [N_patches, 2]

    def compute_coefficient_smoothness(self, grid_size=16):
        """
        计算像差系数在视场上的空间平滑度 (TV)。

        Args:
            grid_size: 采样网格大小

        Returns:
            smoothness: TV loss 标量
        """
        device = next(self.aberration_net.parameters()).device
        y = torch.linspace(-1, 1, grid_size, device=device)
        x = torch.linspace(-1, 1, grid_size, device=device)
        grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
        coords = torch.stack([grid_y.flatten(), grid_x.flatten()], dim=1)

        coeffs = self.aberration_net(coords)
        coeffs_map = coeffs.view(grid_size, grid_size, -1).permute(2, 0, 1)

        dy = torch.abs(coeffs_map[:, 1:, :] - coeffs_map[:, :-1, :]).mean()
        dx = torch.abs(coeffs_map[:, :, 1:] - coeffs_map[:, :, :-1]).mean()
        return dy + dx

    def forward(self, x_hat, crop_info=None):
        """
        x_hat: [B, C, H, W] (输入图像)
        crop_info: 可选，[top/H_orig, left/W_orig, crop_h/H_orig, crop_w/W_orig] 
                   用于全局坐标对齐。如果为 None，使用局部坐标（向后兼容）。
        
        Returns: y_hat [B, C, H, W] (模糊图像)
        """
        B, C, H, W = x_hat.shape
        P = self.patch_size
        S = self.stride
        K = self.kernel_size
        
        # 1. Pad Input to ensure patches cover everything nicely
        # We need H, W to be P + k*S.
        # Or simply Unfold and Fold handles padding?
        # F.fold requires output_size to be specified.
        # If we just pad x_hat so that it fits unfold perfectly.
        '''
        原始图像: 512×512
补丁大小 P = 128, 步长 S = 64

计算补丁数:
n_h = (512 - 128) / 64 + 1 = 7

但最后一个补丁起始位置: 6 × 64 = 384
最后一个补丁范围: [384, 512) - 只覆盖到 512
缺失像素: 无

但如果是 513×513:
n_h = (513 - 128) / 64 + 1 = 7.03 → 7 (整除)
最后像素 513 无法被覆盖

所以需要填充:
513 + pad = 576 (能整除)
pad_h = (64 - (513 - 128) % 64) % 64 = (64 - 1) % 64 = 63
        '''
        pad_h = (S - (H - P) % S) % S
        pad_w = (S - (W - P) % S) % S
        
        # Also need to handle if H < P
        if H < P: pad_h += P - H
        if W < P: pad_w += P - W
        
        # Check if padding is too large for reflect mode
        # Reflect padding requires input_dim >= pad
        # If input size is smaller than padding, reflect mode will crash.
        if H < pad_h or W < pad_w:
            mode_pad = 'replicate' # Fallback for very small images
        else:
            mode_pad = 'reflect'
            
        x_padded = F.pad(x_hat, (0, pad_w, 0, pad_h), mode=mode_pad)
        
        H_pad, W_pad = x_padded.shape[2:]
        
        # 2. Unfold
        # [B, C*P*P, N_blocks]
        '''
[B, C, H_pad, W_pad] 
    ↓
[B, C*P*P, N_patches]  ← 每列是一个 P×P 补丁的展平版本
    ↓
reshape → [B*N_patches, C, P, P]

例如 B=2, C=3, N_patches=64:
[2, 3*128*128, 64]
    ↓
[2, 49152, 64]
    ↓
[128, 3, 128, 128]
        '''
        patches_unfolded = F.unfold(x_padded, kernel_size=P, stride=S)
        N_patches = patches_unfolded.shape[2]
        
        # Reshape to [B * N_patches, C, P, P]
        # Transpose to [B, N_patches, C*P*P]
        patches_unfolded = patches_unfolded.transpose(1, 2)
        # Reshape
        patches = patches_unfolded.reshape(B * N_patches, C, P, P)
        
        # 3. Generate Kernels
        # Get coordinates for ALL patches (same for every item in batch)
        # [N_patches, 2]
        '''
        不同补丁使用不同的卷积核:
        补丁 1 (中心: 图像左上)
  ├─ 坐标 (-0.778, -0.778)
  ├─ AberrationNet 预测系数 [a₁, a₂, ..., a₁₅]
  └─ ZernikeGenerator → PSF 核 K₁ [3, 31, 31]

补丁 2 (中心: 图像中心)
  ├─ 坐标 (0, 0)
  ├─ AberrationNet 预测系数 [a'₁, a'₂, ..., a'₁₅]
  └─ ZernikeGenerator → PSF 核 K₂ [3, 31, 31]

补丁 64 (中心: 图像右下)
  ├─ 坐标 (0.778, 0.778)
  ├─ AberrationNet 预测系数 [a''₁, a''₂, ..., a''₁₅]
  └─ ZernikeGenerator → PSF 核 K₆₄ [3, 31, 31]
        '''
        # 光学系统的像差通常随视场角变化（如边缘失焦）
        # 中心清晰 → 边缘模糊的真实光学现象

        # 获取补丁中心坐标，支持全局坐标对齐（逐样本）
        if crop_info is not None:
            # crop_info: [B, 4] 或 [4,]
            if crop_info.dim() == 1:
                crop_info = crop_info.unsqueeze(0)

            coords_list = []
            for b in range(B):
                crop_info_single = crop_info[b]

                # 反解原始图像尺寸（基于 crop_info 推断）
                crop_h_norm = crop_info_single[2].item()
                crop_w_norm = crop_info_single[3].item()

                if crop_h_norm > 0 and crop_w_norm > 0:
                    H_orig = int(H / crop_h_norm)
                    W_orig = int(W / crop_w_norm)
                else:
                    H_orig = H
                    W_orig = W
                    crop_info_single = None

                coords_1img = self.get_patch_centers(
                    H_pad, W_pad, x_hat.device,
                    H_orig=H_orig, W_orig=W_orig,
                    crop_info=crop_info_single
                )
                coords_list.append(coords_1img)

            # [B * N_patches, 2]
            coords = torch.cat(coords_list, dim=0)
        else:
            # 向后兼容：没有 crop_info 时使用局部坐标
            coords_1img = self.get_patch_centers(H_pad, W_pad, x_hat.device)
            coords = coords_1img.repeat(B, 1)
        
        # AberrationNet -> Coeffs
        coeffs = self.aberration_net(coords) # [B*N, Ncoeff]
        
        # ZernikeGenerator -> Kernels
        kernels = self.zernike_generator(coeffs) # [B*N, C_k, K, K]
        
        # Check channel consistency and determine output channels
        C_k = kernels.shape[1]
        if C == C_k:
            C_out = C
        elif C == 1 and C_k > 1:
            C_out = C_k
        elif C > 1 and C_k == 1:
            C_out = C
        else:
            raise ValueError(f"Channel mismatch: Input ({C}) and Kernel ({C_k}) are not compatible for broadcasting.")
        
        # 4. Convolution Implementation
        # 卷积实现：根据配置选择基于 FFT 的 NewBP 算子或高效的空间域卷积
        
        if self.use_newbp:
            # -------------------------------------------------------------------------
            # NewBP Mode: Custom FFT-based Convolution
            # 仅在需要使用 NewBP 自定义反向传播算子时（科研需求），才进行 FFT 相关计算
            # -------------------------------------------------------------------------
            fft_size = P + K - 1
            if self.pad_to_power_2:
                fft_size = 2 ** math.ceil(math.log2(fft_size))

            # Use NewBP custom autograd function
            # This implements backward pass with explicit non-diagonal Jacobian
            y_patches_result = NewBPConvolutionFunction.apply(
                patches, kernels, K, P, fft_size
            )
            # Determine C_out from result
            assert isinstance(y_patches_result, torch.Tensor), "NewBP output must be Tensor"
            y_patches = y_patches_result
            C_out = y_patches.shape[1]

        else:
            # -------------------------------------------------------------------------
            # Spatial Domain Convolution (GPU-optimized via cuDNN)
            # 空间域卷积：在 GPU 上直接进行卷积运算
            # 完全省略傅里叶变换步骤。对于小尺寸卷积核 (如 31x31)，这比 FFT 转换快得多，
            # 且能充分利用 GPU 的并行计算能力。
            # -------------------------------------------------------------------------
            
            # Flip kernel for F.conv2d (Correlation -> Convolution)
            # 注意：F.conv2d 计算的是相关性 (Correlation)，而物理光学定义的是卷积 (Convolution)
            # 因此需要在空间域对卷积核进行上下左右翻转。
            kernels_flipped = torch.flip(kernels, dims=[-2, -1])
            
            # 使用 Grouped Convolution 实现动态卷积 (Dynamic Convolution)
            # 将 Batch 中每个样本的 Patch 与其对应的 Kernel 进行卷积
            BN = patches.shape[0]  # B * N_patches
            pad = K // 2
            
            # Pad input patches to maintain size: P x P -> P x P
            patches_padded = F.pad(patches, (pad, pad, pad, pad), mode='constant', value=0)
            
            if C == C_k:
                # Case 1: Same channels (Color -> Color)
                # 使用 grouped convolution: groups=BN*C，实现每个通道独立的局部卷积
                patches_grouped = patches_padded.view(1, BN * C,
                                                       patches_padded.shape[2],
                                                       patches_padded.shape[3])
                kernels_grouped = kernels_flipped.view(BN * C, 1, K, K)
                y_grouped = F.conv2d(patches_grouped, kernels_grouped, groups=BN * C)
                y_patches = y_grouped.view(BN, C, P, P)
                C_out = C
                
            elif C == 1 and C_k > 1:
                # Case 2: Grayscale input -> Multi-channel Kernel (e.g., BW -> RGB aberration)
                patches_expanded = patches_padded.expand(-1, C_k, -1, -1)
                patches_grouped = patches_expanded.reshape(1, BN * C_k,
                                                            patches_padded.shape[2],
                                                            patches_padded.shape[3])
                kernels_grouped = kernels_flipped.view(BN * C_k, 1, K, K)
                y_grouped = F.conv2d(patches_grouped, kernels_grouped, groups=BN * C_k)
                y_patches = y_grouped.view(BN, C_k, P, P)
                C_out = C_k
                
            else:  # C > 1 and C_k == 1
                # Case 3: Multi-channel input -> Single Kernel (e.g., RGB -> Scalar aberration)
                kernels_expanded = kernels_flipped.expand(-1, C, -1, -1)
                patches_grouped = patches_padded.view(1, BN * C,
                                                       patches_padded.shape[2],
                                                       patches_padded.shape[3])
                kernels_grouped = kernels_expanded.reshape(BN * C, 1, K, K)
                y_grouped = F.conv2d(patches_grouped, kernels_grouped, groups=BN * C)
                y_patches = y_grouped.view(BN, C, P, P)
                C_out = C
        
        # 5. Apply Window and Fold
        # Explicit dimension expansion for window
        window_4d = self.window.view(1, 1, P, P)
        y_patches = y_patches * window_4d
        
        # Reshape for folding: [B, C_out*P*P, N_patches]
        # Use C_out here
        y_patches_reshaped = y_patches.reshape(B, N_patches, C_out*P*P).transpose(1, 2)
        
        output_h = H_pad
        output_w = W_pad
        
        # Output will have C_out channels
        y_accum = F.fold(y_patches_reshaped, output_size=(output_h, output_w), kernel_size=P, stride=S)
        
        # 6. Normalization Map
        # Weight patches: ones * window
        # Expand match C_out
        w_patches = window_4d.expand(B * N_patches, C_out, P, P)
        w_patches_reshaped = w_patches.reshape(B, N_patches, C_out*P*P).transpose(1, 2)
        
        w_accum = F.fold(w_patches_reshaped, output_size=(output_h, output_w), kernel_size=P, stride=S)
        
        # Normalize
        y_hat_padded = y_accum / (w_accum + 1e-8)
        
        # Crop back to original size
        y_hat = y_hat_padded[..., :H, :W]
        
        return y_hat
'''
Overlap-Add 可视化:

补丁 1 (像素 [0:128, 0:128])  +  Hann 窗口
补丁 2 (像素 [64:192, 0:128]) +  Hann 窗口
         ↓ 重叠区间 [64:128]
    两个补丁的输出在此相加

Fold 过程:
   [0:64]       [64:128]      [128:192]
  ┌─────────┬─────────┬─────────┐
  │ 补丁1   │ 补丁1/2 │ 补丁2   │
  │ (仅窗口)│ (相加)  │ (仅窗口)│
  └─────────┴─────────┴─────────┘
  
加权归一化：
result[64:128] = (patch1_out + patch2_out) / (w1 + w2)
                = (w1*信号1 + w2*信号2) / (w1 + w2)  ← 加权平均
'''

'''
                    输入图像 x_hat
                        │
                        ▼
                    Unfold (分割)
                        │
        ┌───────────────┴───────────────┐
        │                               │
        ▼                               ▼
    补丁集合                        坐标集合
  (N 个 128×128)                   (N 个坐标)
        │                               │
        │                   AberrationNet(坐标)
        │                        │
        │                        ▼
        │                   Zernike 系数
        │                        │
        │                   ZernikeGenerator
        │                        │
        │                        ▼
        │                    PSF 核集合
        │                   (N 个 31×31)
        │                        │
        └───────┬─────────────────┘
                ▼
        FFT 卷积 (补丁 ⊗ PSF)
                │
                ▼
        卷积后补丁
      (N 个 128×128)
                │
                ├─ Hann 窗口加权 ─────┐
                │                     │
                ▼                     ▼
        加权后补丁              权重补丁
        y_patches             w_patches
                │                     │
                └──────┬──────────────┘
                       ▼
                  Fold (拼接)
                       │
        ┌──────────────┴──────────────┐
        ▼                             ▼
   y_accum                       w_accum
  (累积输出)                     (累积权重)
        │                             │
        └──────┬──────────────────────┘
               ▼
        y_accum / w_accum
             (归一化)
               ▼
        裁剪回原尺寸
               ▼
        最终模糊图像 y_hat
'''