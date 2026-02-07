# U-Net 架构的图像复原网络——从模糊图像恢复清晰图像
'''
模糊图像 (来自 physical_layer)
    [B, C, H, W]
        ↓
RestorationNet (U-Net)
    ↓
复原图像 (清晰)
    [B, C, H, W]
'''

'''
模糊图像 [B, 3, H, W]
    │
    ├─ (可选) 坐标注入: 拼接 [y_grid, x_grid]
    │  └─ 变成 [B, 5, H, W]
    │
    ▼
编码器 (Encoder) - 下采样
    │
    ├─ Inc (入口): 3/5 → 64
    │  [B, 64, H, W]
    │
    ├─ Down1: MaxPool + Conv
    │  [B, 128, H/2, W/2]
    │
    ├─ Down2: MaxPool + Conv
    │  [B, 256, H/4, W/4]
    │
    ├─ Down3: MaxPool + Conv
    │  [B, 512, H/8, W/8]
    │
    └─ Down4: MaxPool + Conv
       [B, 512, H/16, W/16]  ← 最深层 (瓶颈)
    │
    ▼
解码器 (Decoder) - 上采样 + 跳跃连接
    │
    ├─ Up1: 上采样 + 跳跃 + Conv
    │  [B, 256, H/8, W/8]
    │
    ├─ Up2: 上采样 + 跳跃 + Conv
    │  [B, 128, H/4, W/4]
    │
    ├─ Up3: 上采样 + 跳跃 + Conv
    │  [B, 64, H/2, W/2]
    │
    ├─ Up4: 上采样 + 跳跃 + Conv
    │  [B, 64, H, W]
    │
    └─ OutConv: 1×1 卷积
       [B, 3, H, W]  ← 差异图 (correction)
    │
    ▼
残差连接 (Residual)
    输出 = 输入 + 差异
    [B, 3, H, W]  ← 复原图像
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    # 功能：提取特征的基本单元
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            # nn.BatchNorm2d(mid_channels), # Batch Norm can sometimes be problematic in low batch size physics sims, but usually fine.
            # Using GroupNorm or InstanceNorm is often safer for restoration, lets stick to simple Conv+ReLU for now or include BN.
            # Deconvolution often prefers removing BN or using Instance Norm. Let's use standard BN for U-Net baseline.
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    '''
输入 [B, in_c, H, W]
    │
    ├─ MaxPool2d(2)  ← 2×2 最大池化
    │  [B, in_c, H/2, W/2]
    │
    ├─ DoubleConv
    │  [B, out_c, H/2, W/2]
    │
    ▼
输出 [B, out_c, H/2, W/2]

分辨率降低 (↓2)
通道增加 (通常翻倍)
    '''
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""
    '''
解码器的上一层 x1       编码器的同级特征 x2
[B, 512, H/16, W/16]   [B, 256, H/8, W/8]
        │                       │
        │ Upsample(×2)         │
        ├─→ [B, 512, H/8, W/8] │
        │                       │
        ├─ 处理尺寸差异 (padding)
        │                       │
        └──────┬────────────────┘
               │
           Concatenate (通道维)
         [B, 768, H/8, W/8]
               │
             Conv
         [B, 256, H/8, W/8]
               │
               ▼
           输出
    '''
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, mid_channels=in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    '''
输入 [B, 64, H, W]
    │
    ├─ Conv1×1 (无 padding，无激活)
    │  降维到输出通道数
    │
    ▼
输出 [B, 3, H, W]
    (RGB 三通道)
    '''
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class RestorationNet(nn.Module):
    def __init__(self, n_channels, n_classes, base_filters=32, bilinear=True, use_coords=False):
        super(RestorationNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.use_coords = use_coords
        
        factor = 2 if bilinear else 1
        
        # Increase input channels if using coordinates
        input_channels = n_channels + 2 if use_coords else n_channels
        
        self.inc = DoubleConv(input_channels, base_filters)
        self.down1 = Down(base_filters, base_filters * 2)
        self.down2 = Down(base_filters * 2, base_filters * 4)
        self.down3 = Down(base_filters * 4, base_filters * 8)
        self.down4 = Down(base_filters * 8, base_filters * 16 // factor)
        
        self.up1 = Up(base_filters * 16, base_filters * 8 // factor, bilinear)
        self.up2 = Up(base_filters * 8, base_filters * 4 // factor, bilinear)
        self.up3 = Up(base_filters * 4, base_filters * 2 // factor, bilinear)
        self.up4 = Up(base_filters * 2, base_filters, bilinear)
        
        self.outc = OutConv(base_filters, n_classes)

    def forward(self, x_input):
        # Input x_input: [B, C, H, W]
        x = x_input
        
        if self.use_coords:
            B, C, H, W = x.shape
            # Create coordinate grid
            # y maps (-1, 1), x maps (-1, 1)
            # Use same device and dtype as input
            y_coords = torch.linspace(-1, 1, H, device=x.device, dtype=x.dtype)
            x_coords = torch.linspace(-1, 1, W, device=x.device, dtype=x.dtype)
            grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')
            
            # [H, W] -> [1, 1, H, W] -> [B, 1, H, W]
            grid_y = grid_y.unsqueeze(0).unsqueeze(0).expand(B, 1, H, W)
            grid_x = grid_x.unsqueeze(0).unsqueeze(0).expand(B, 1, H, W)
            
            # Concatenate: [B, C+2, H, W]
            x = torch.cat([x, grid_y, grid_x], dim=1)
            
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        
        # Residual connection
        # Restored = Input + Correction
        # This forces the net to learn the difference (blur removal) rather than the image itself
        return x_input + logits

'''
                              输入 [B, 3, H, W] (模糊图像)
                                    │
                              + 坐标 [B, 2, H, W]
                                    │
                                    ▼
                              Inc: 3/5 → 64
                          [B, 64, H, W] ← x1
                                    │
                          ┌─────────┴─────────┐
                          │                   │
                    Down1 ▼                   │ (跳跃)
                     MaxPool + Conv           │
                   [B, 128, H/2, W/2] ← x2   │
                          │                   │
                    ┌─────┴─────┐             │
                    │           │             │
              Down2 ▼           │ (跳跃)      │
               MaxPool+Conv     │             │
             [B, 256, H/4, W/4] ← x3  ← ─ ─ ┘
                    │           │
              ┌─────┴───┐       │
              │         │       │
        Down3 ▼         │ (跳跃)│
         MaxPool+Conv   │       │
       [B, 512, H/8, W/8] ← x4  ← ─ ┘
              │         │
        ┌─────┴───┐     │
        │         │     │
  Down4 ▼         │ (跳跃)
   MaxPool+Conv   │
 [B, 512, H/16]   │
      (瓶颈)      │
        │         │
        │    Up1 ← ┘
        ▼    ↑ Upsample + DoubleConv
      [B, 256, H/8, W/8] ← x3 (跳跃连接)
        │
        │    Up2
        ▼    ↑ Upsample + DoubleConv
      [B, 128, H/4, W/4] ← x2 (跳跃连接)
        │
        │    Up3
        ▼    ↑ Upsample + DoubleConv
      [B, 64, H/2, W/2] ← x1 (跳跃连接)
        │
        │    Up4
        ▼    ↑ Upsample + DoubleConv
      [B, 64, H, W]
        │
        ▼
      OutConv (1×1)
      [B, 3, H, W] ← 差异图 (correction)
        │
        │ + 残差连接
        │
        ▼
    [B, 3, H, W] ← 复原图像
'''