"""
NewBP (New Backpropagation) 散焦去模糊自定义 Autograd 函数

本模块实现了一个自定义的自动求导函数，用于处理反向传播过程中的空间能量串扰。
它通过非对角雅可比矩阵来建模光线如何通过点扩散函数 (PSF) 从一个像素扩散到相邻像素的过程。

数学基础:
-----------------------
前向传播: Y = H·X, 其中 H 是串扰矩阵 (与 PSF 进行卷积)
反向传播: ∂L/∂X = H^T · (∂L/∂Y)

梯度分解 (NewBP):
  ∂L/∂X[i,j] = G_direct + G_indirect
  
  G_direct   = ∂L/∂Y[i,j] × K[0,0]           # 自我贡献 (对角线元素)
  G_indirect = Σ_{(m,n)≠(0,0)} ∂L/∂Y[m,n] × K[i-m, j-n]  # 邻域贡献 (非对角线元素)

对于圆形 PSF: K_flipped = K (对称)，因此反向卷积使用相同的核。

参考: 科研日志.md lines 88-145 (NewBP视角下的具体梯度解析)

实现说明 (2026-01-23):
---------------------------------
已从基于 FFT 的频域乘法改为使用 F.conv2d 的空域卷积，以提高小尺寸核 (K <= 33) 在 GPU 上的性能。
当核尺寸相对于图像尺寸较小时，cuDNN 提供的优化算法 (Winograd, im2col) 优于 FFT 方法。
"""

'''
输入数据流 (空域卷积实现)
────────────────────────────────────────────
清晰图像 X [B*N, 3, 128, 128]
  ↓ (reflect 填充 K//2)
X_padded [B*N, 3, 143, 143]  (当 K=31)
  ↓ 
分组卷积 Grouped Conv2d (groups=B*N*C)
  ↓
输出 Y [B*N, 3, 128, 128] ✓

PSF 核 K [B*N, 3, 31, 31]
  ↓ reshape
K_grouped [B*N*3, 1, 31, 31]

────────────────────────────────────────────
反向梯度流 (空域卷积实现)
────────────────────────────────────────────
下游梯度 ∂L/∂Y [B*N, 3, 128, 128]
  ↓ (constant 填充 K//2)
  ↓ conv2d with flip(K)
∂L/∂X [B*N, 3, 128, 128] ✓

对核梯度:
X.unfold(K) → [B*N, C*K*K, P*P]
  ↓ bmm with ∂L/∂Y.flatten
∂L/∂K [B*N, 3, 31, 31] ✓
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
import math


class NewBPConvolutionFunction(torch.autograd.Function):
    """
    实现 NewBP 感知卷积的自定义 Autograd 函数。
    
    该函数显式地对基于 PSF 的空间卷积产生的非对角雅可比矩阵进行建模，
    将梯度分解为直接和间接分量，以正确地考虑能量串扰。
    """
    
    @staticmethod
    def forward(ctx, patches, kernels, kernel_size, patch_size, fft_size):
        """
        前向传播: Patch 与 PSF 核的卷积 (此处主要使用空域实现)。
        
        Args:
            patches: [B*N, C, P, P] - 输入图像 patch
            kernels: [B*N, C_k, K, K] - PSF 卷积核
            kernel_size: int - PSF 核的大小 (K)
            patch_size: int - Patch 的大小 (P)
            fft_size: int - FFT 计算大小 (此处保留该参数以保持接口一致，但主要使用空域卷积)
            
        Returns:
            y_patches: [B*N, C_out, P, P] - 卷积后的 patch
        """
        # 保存输入 patch 用于计算核梯度
        ctx.save_for_backward(patches, kernels)
        ctx.kernel_size = kernel_size
        ctx.patch_size = patch_size
        ctx.fft_size = fft_size
        
        # 确定输出通道数
        C = patches.shape[1]
        C_k = kernels.shape[1]
        if C == C_k:
            C_out = C
        elif C == 1 and C_k > 1:
            C_out = C_k
        elif C > 1 and C_k == 1:
            C_out = C
        else:
            raise ValueError(f"Channel mismatch: patches={C}, kernels={C_k}")
        
        ctx.input_channels = C
        ctx.kernel_channels = C_k
        ctx.output_channels = C_out
        
        # =====================================================================
        # Spatial Domain Convolution (GPU-optimized via cuDNN)
        # 空域卷积 (通过 cuDNN 进行 GPU 优化)
        # 替代了基于 FFT 的频域乘法，以便在小尺寸核 (K <= 33) 上获得更好的性能，
        # 此时 cuDNN 表现更优。
        #
        # 重要提示: FFT 卷积计算的是真正的卷积 (核翻转)，
        # 而 F.conv2d 计算的是互相关 (核不翻转)。
        # 为了匹配 FFT 的结果，我们必须在调用 F.conv2d 之前翻转核。
        # =====================================================================
        
        BN = patches.shape[0]  # B * N_patches
        
        # 翻转核以将相关转换为卷积 (匹配 FFT 行为)
        kernels_flipped = torch.flip(kernels, dims=[-2, -1])
        
        # 填充输入以获得 'same' 输出尺寸 (每侧填充 kernel_size // 2)
        # 使用 'constant' (零) 填充以匹配 FFT 的隐式零填充
        pad = kernel_size // 2
        patches_padded = F.pad(patches, (pad, pad, pad, pad), mode='constant', value=0)
        
        # 逐样本卷积: 每个 patch 都有自己的核
        # 使用 groups=BN 将不同的核应用于不同的样本
        # Reshape: [BN, C, H, W] -> [1, BN*C, H, W] 用于分组卷积
        
        if C == C_k:
            # 情况 1: 通道数相同 - 直接逐通道卷积
            # Reshape patches: [BN, C, H, W] -> [1, BN*C, H, W]
            patches_grouped = patches_padded.view(1, BN * C, 
                                                   patches_padded.shape[2], 
                                                   patches_padded.shape[3])
            # Reshape flipped kernels: [BN, C, K, K] -> [BN*C, 1, K, K]
            kernels_grouped = kernels_flipped.view(BN * C, 1, kernel_size, kernel_size)
            
            # 分组卷积: BN*C 个组中，每组有 1 个输入和 1 个输出通道
            y_grouped = F.conv2d(patches_grouped, kernels_grouped, groups=BN * C)
            
            # Reshape back: [1, BN*C, P, P] -> [BN, C, P, P]
            y_patches = y_grouped.view(BN, C, patch_size, patch_size)
            
        elif C == 1 and C_k > 1:
            # 情况 2: 灰度输入，多通道核 (广播)
            # 复制输入以匹配核通道
            patches_expanded = patches_padded.expand(-1, C_k, -1, -1)
            patches_grouped = patches_expanded.reshape(1, BN * C_k,
                                                        patches_padded.shape[2],
                                                        patches_padded.shape[3])
            kernels_grouped = kernels_flipped.view(BN * C_k, 1, kernel_size, kernel_size)
            
            y_grouped = F.conv2d(patches_grouped, kernels_grouped, groups=BN * C_k)
            y_patches = y_grouped.view(BN, C_k, patch_size, patch_size)
            
        else:  # C > 1 and C_k == 1
            # 情况 3: 多通道输入，单通道核 (广播核)
            kernels_expanded = kernels_flipped.expand(-1, C, -1, -1)
            patches_grouped = patches_padded.view(1, BN * C,
                                                   patches_padded.shape[2],
                                                   patches_padded.shape[3])
            kernels_grouped = kernels_expanded.reshape(BN * C, 1, kernel_size, kernel_size)
            
            y_grouped = F.conv2d(patches_grouped, kernels_grouped, groups=BN * C)
            y_patches = y_grouped.view(BN, C, patch_size, patch_size)
        
        return y_patches
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        反向传播: 使用非对角雅可比矩阵进行 NewBP 梯度计算。
        """
        patches, kernels = ctx.saved_tensors
        K = ctx.kernel_size
        P = ctx.patch_size
        fft_size = ctx.fft_size
        C_in = ctx.input_channels
        C_out = ctx.output_channels
        
        # =====================================================================
        # --- 1. 计算关于输入 Patch 的梯度 (dL/dX) ---
        # =====================================================================
        # 雅可比矩阵 J = ∂Y/∂X 是一个卷积矩阵。
        # J^T 对应于与 180° 旋转 (翻转) 后的核进行卷积。
        # 
        # NewBP 分解:
        #   ∂L/∂X[i,j] = G_direct + G_indirect
        #   - G_direct (对角项):    grad_output[i,j] × K[center, center]
        #   - G_indirect (非对角项): Σ grad_output[m,n] × K_flipped[i-m, j-n]
        # 
        # 由于前向传播使用了翻转后的核 (为了匹配 FFT 卷积)，
        # 反向传播的 J^T 需要使用原始 (未翻转) 的核。
        # =====================================================================
        
        BN = grad_output.shape[0]
        
        # 对 grad_output 进行填充以进行 'same' 卷积
        pad = K // 2
        grad_padded = F.pad(grad_output, (pad, pad, pad, pad), mode='constant', value=0)
        
        # 空域卷积: grad_output * K (原始，未翻转)
        # 因为前向传播使用了 flip(K)，反向 J^T 直接使用 K
        if C_out == kernels.shape[1]:
            grad_grouped = grad_padded.view(1, BN * C_out, 
                                            grad_padded.shape[2], grad_padded.shape[3])
            k_grouped = kernels.view(BN * C_out, 1, K, K)
            
            grad_X_grouped = F.conv2d(grad_grouped, k_grouped, groups=BN * C_out)
            grad_patches = grad_X_grouped.view(BN, C_out, P, P)
        else:
            # 处理广播情况
            grad_grouped = grad_padded.view(1, BN * C_out,
                                            grad_padded.shape[2], grad_padded.shape[3])
            k_expanded = kernels.expand(-1, C_out, -1, -1) if kernels.shape[1] == 1 else kernels
            k_grouped = k_expanded.reshape(BN * C_out, 1, K, K)
            
            grad_X_grouped = F.conv2d(grad_grouped, k_grouped, groups=BN * C_out)
            grad_patches = grad_X_grouped.view(BN, C_out, P, P)
        
        # 匹配输入通道以实现反向兼容性
        if grad_patches.shape[1] != C_in:
            if C_in == 1 and grad_patches.shape[1] > 1:
                grad_patches = grad_patches.sum(dim=1, keepdim=True)
            elif C_in > 1 and grad_patches.shape[1] == 1:
                grad_patches = grad_patches.repeat(1, C_in, 1, 1)

        # =====================================================================
        # --- 2. 计算关于核的梯度 (dL/dK) ---
        # =====================================================================
        # 对于 Y = X * K (卷积)，核梯度为:
        #   dL/dK = X ⊛ dL/dY  (互相关)
        # 
        # 在空域中，这等价于:
        #   dL/dK[ki, kj] = Σ_{i,j} X[i+ki, j+kj] × dL/dY[i, j]
        # 
        # 我们使用带有翻转操作数的 F.conv2d 来高效计算。
        # =====================================================================
        
        # 互相关: patches ⊛ grad_output
        # 等价于: conv2d(patches, grad_output_as_kernel)
        # 但我们需要逐样本相关，因此使用循环或 unfold 技巧
        
        # 为了效率，使用 unfold + matmul 模式计算
        # Unfold patches 以提取所有 K×K 窗口
        patches_unfolded = F.unfold(patches, kernel_size=K, padding=K//2)  # [BN, C*K*K, P*P]
        
        # Reshape grad_output: [BN, C_out, P, P] -> [BN, C_out, P*P]
        grad_flat = grad_output.view(BN, C_out, -1)  # [BN, C_out, P*P]
        
        # 通过批量矩阵乘法计算相关性
        # patches_unfolded: [BN, C_in*K*K, P*P]
        # grad_flat: [BN, C_out, P*P]
        # 结果应为: [BN, C_k, K, K]
        
        C_k = kernels.shape[1]
        
        if C_in == C_out == C_k:
            # 标准情况: 通道数全程相同
            # 逐通道相关
            grad_kernels_list = []
            for c in range(C_in):
                # 提取通道 c 的窗口: [BN, K*K, P*P]
                patch_c = patches_unfolded[:, c*K*K:(c+1)*K*K, :]
                # 梯度通道 c: [BN, 1, P*P]
                grad_c = grad_flat[:, c:c+1, :]
                # 相关性: [BN, K*K, P*P] × [BN, P*P, 1] -> [BN, K*K, 1]
                corr_c = torch.bmm(patch_c, grad_c.transpose(1, 2))  # [BN, K*K, 1]
                grad_kernels_list.append(corr_c.view(BN, 1, K, K))
            grad_kernels = torch.cat(grad_kernels_list, dim=1)  # [BN, C, K, K]
        else:
            # 广播情况: 简化计算
            # 针对 C_in=1, C_k>1 或 C_in>1, C_k=1
            # 在空间位置上求和
            patches_unfolded_sum = patches_unfolded.view(BN, C_in, K*K, P*P)
            grad_flat_expanded = grad_flat.view(BN, C_out, 1, P*P)
            
            # [BN, C_in, K*K, P*P] × [BN, C_out, P*P, 1] via einsum
            # 输出: [BN, max(C_in, C_out), K*K]
            if C_in == 1:
                corr = torch.einsum('bkp,bcp->bck', patches_unfolded_sum.squeeze(1), grad_flat)
            else:  # C_k == 1
                corr = torch.einsum('bckp,bp->bck', patches_unfolded_sum, grad_flat.squeeze(1))
                corr = corr.sum(dim=1, keepdim=True)  # Sum over input channels
            grad_kernels = corr.view(BN, C_k, K, K)
        
        # =====================================================================
        # [BUG FIX] 核梯度方向修正
        # ---------------------------------------------------------------------
        # 前向传播使用了翻转后的核: Y = X * flip(K)
        # 当前计算的 grad_kernels 是关于 flip(K) 的梯度: dL/d(flip(K))
        # 但 Autograd 需要的是关于原始 K 的梯度: dL/dK
        # 
        # 由于 flip 是自逆运算 (flip(flip(x)) = x)，根据链式法则:
        #   dL/dK = flip(dL/d(flip(K)))
        # 
        # 不加此翻转会导致梯度方向在空间上 180° 倒置，
        # 使优化器沿错误方向更新参数，导致 Loss 上升而非下降。
        # =====================================================================
        grad_kernels = torch.flip(grad_kernels, dims=[-2, -1])
        
        return grad_patches, grad_kernels, None, None, None


class NewBPSpatialConvolution(nn.Module):
    """
    NewBP 卷积的包装模块，可作为一个层使用。
    
    该模块提供了一个干净的接口，以便在物理层内使用 NewBP 卷积，
    并具有可选的梯度统计记录功能。
    """
    
    def __init__(self, enable_grad_logging=False):
        super().__init__()
        self.enable_grad_logging = enable_grad_logging
        
    def forward(self, patches, kernels, kernel_size, patch_size, fft_size):
        """
        对 patch 应用 NewBP 卷积。
        
        Args:
            patches: [B*N, C, P, P]
            kernels: [B*N, C_k, K, K]
            kernel_size: int
            patch_size: int
            fft_size: int
            
        Returns:
            [B*N, C_out, P, P]
        """
        return NewBPConvolutionFunction.apply(
            patches, kernels, kernel_size, patch_size, fft_size
        )


def compute_jacobian_structure(kernel, image_size):
    """
    用于可视化给定 PSF 核的雅可比矩阵结构的实用函数。
    
    此函数用于测试和验证。它为一个小图像计算显式雅可比矩阵，以验证非对角结构。
    
    Args:
        kernel: [K, K] - PSF 核
        image_size: int - 测试图像的大小 (应该很小，例如 16)
        
    Returns:
        jacobian: [N, N] - 显式雅可比矩阵，其中 N = image_size^2
        
    Example:
        >>> kernel = torch.randn(5, 5)
        >>> J = compute_jacobian_structure(kernel, image_size=16)
        >>> # Visualize: plt.imshow(J.abs())
    """
    N = image_size * image_size
    K = kernel.shape[0]
    
    # 创建显式雅可比矩阵
    jacobian = torch.zeros(N, N)
    
    # ---------------------------------------------------------------------
    # 重构: 向量化实现 (O(K^2) 循环代替 O(N^2))
    # ---------------------------------------------------------------------
    center = K // 2
    device = jacobian.device # 使用与目标矩阵相同的设备
    
    for ki in range(K):
        for kj in range(K):
            val = kernel[ki, kj]
            # 跳过接近零的值以提高稀疏效率
            if abs(val) < 1e-12: continue
            
            # 计算相对于中心的位移
            # 原始逻辑: ki = center - di => di = center - ki
            di = center - ki
            dj = center - kj
            
            # --- NewBP 梯度分离 ---
            # 1. 直接梯度 (对角元素):
            #    自我贡献，其中 di=0, dj=0 (idx_i == idx_j)
            # 2. 间接梯度 (非对角元素):
            #    来自邻域的串扰，其中 di!=0 或 dj!=0
            is_diagonal = (di == 0) and (dj == 0)
            
            # 计算有效输出像素范围 [i, j]
            # 条件: 0 <= i < S AND 0 <= i+di < S
            i_start = max(0, -di)
            i_end = min(image_size, image_size - di)
            
            # 条件: 0 <= j < S AND 0 <= j+dj < S
            j_start = max(0, -dj)
            j_end = min(image_size, image_size - dj)
            
            if i_start < i_end and j_start < j_end:
                # 向量化索引生成
                i_idx = torch.arange(i_start, i_end, device=device)
                j_idx = torch.arange(j_start, j_end, device=device)
                
                # 输出索引 (雅可比矩阵的行): i * W + j
                rows = (i_idx[:, None] * image_size + j_idx[None, :]).flatten()
                
                # 输入索引 (雅可比矩阵的列): (i+di) * W + (j+dj)
                # 注意: di*W + dj 是此核元素的常数偏移
                cols = rows + (di * image_size + dj)
                
                # 批量赋值
                # 如果是 is_diagonal: 填充主对角线 (直接梯度)
                # 如果不是 is_diagonal: 填充非对角带 (间接梯度)
                jacobian[rows, cols] = val
    
    return jacobian


def analyze_gradient_components(grad_total, grad_output, kernel):
    """
    将梯度分解为直接和间接分量进行分析。
    
    Args:
        grad_total: NewBP 计算的总梯度
        grad_output: 下游传来的梯度
        kernel: 卷积中使用的 PSF 核
        
    Returns:
        dict with keys: 'direct', 'indirect', 'direct_ratio'
    """
    K = kernel.shape[-1]
    center_idx = K // 2
    
    # 直接分量: 由中心核值缩放的 grad_output
    K_center = kernel[..., center_idx:center_idx+1, center_idx:center_idx+1]
    grad_direct = grad_output * K_center
    
    # 间接分量: 剩余部分
    grad_indirect = grad_total - grad_direct
    
    # 统计信息
    direct_norm = grad_direct.norm().item()
    indirect_norm = grad_indirect.norm().item()
    total_norm = grad_total.norm().item()
    
    return {
        'grad_direct': grad_direct,
        'grad_indirect': grad_indirect,
        'direct_norm': direct_norm,
        'indirect_norm': indirect_norm,
        'total_norm': total_norm,
        'direct_ratio': direct_norm / (total_norm + 1e-10),
        'indirect_ratio': indirect_norm / (total_norm + 1e-10)
    }
