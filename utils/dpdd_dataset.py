"""
DPDD (Dual-Pixel Defocus Deblurring) Dataset
=============================================
支持直接从原始数据文件夹加载图像，无需预处理。

特性：
1. 虚拟长度机制：通过 repeat_factor 让每张图在一个 Epoch 内被随机裁剪多次
2. 同步随机裁剪：确保模糊图和清晰图在同一位置裁剪
3. 验证集使用固定中心裁剪，测试集使用全分辨率
4. 直接从 train_c/val_c/test_c 文件夹加载原始图像

DPDD Canon Set 数据量:
- train_c: 350 对图像
- val_c:   74 对图像
- test_c:  76 对图像

原始图像尺寸: 1680 x 1120 像素
"""

import os
from PIL import Image, ImageOps
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import random


class DPDDDataset(Dataset):
    """
    DPDD Dataset with virtual length mechanism and synchronized random cropping.
    
    Expects the following directory structure (original DPDD):
    root_dir/
        train_c/ (or val_c/ or test_c/)
            source/   <- blur images
                img1.png
                ...
            target/   <- sharp images
                img1.png
                ...
    """

    # 映射 mode 到文件夹名
    MODE_TO_FOLDER = {
        'train': 'train_c',
        'val': 'val_c',
        'test': 'test_c'
    }

    def __init__(self, root_dir, mode='train', crop_size=512, 
                 repeat_factor=1, transform=None,
                 val_crop_size=1024, use_full_resolution=False,
                 random_flip=False):
        """
        Args:
            root_dir (str): Root directory of the dataset (e.g., ./data/dd_dp_dataset_png).
            mode (str): One of 'train', 'val', 'test'.
            crop_size (int): Size of the random crop for training (default: 512).
            repeat_factor (int): Virtual length multiplier. Each image is accessed 
                                 `repeat_factor` times per epoch with different random crops.
                                 Only effective for training mode. (default: 1)
            transform (callable, optional): Optional transform to be applied on a sample.
            val_crop_size (int): Size of center crop for validation (default: 1024).
            use_full_resolution (bool): If True, use full resolution images without cropping.
                                       Typically used for testing. (default: False)
        """
        super().__init__()
        self.root_dir = root_dir
        self.mode = mode
        self.crop_size = crop_size
        self.repeat_factor = repeat_factor if mode == 'train' else 1  # 只对训练集生效
        self.val_crop_size = val_crop_size
        self.use_full_resolution = use_full_resolution
        self.random_flip = random_flip

        # Ensure transform is callable
        if transform is None:
            self.transform = transforms.Compose([transforms.ToTensor()])
        else:
            self.transform = transform

        # 根据 mode 确定实际文件夹路径
        folder_name = self.MODE_TO_FOLDER.get(mode)
        if folder_name is None:
            raise ValueError(f"Invalid mode '{mode}'. Must be one of {list(self.MODE_TO_FOLDER.keys())}")
        
        self.split_dir = os.path.join(root_dir, folder_name)
        self.blur_dir = os.path.join(self.split_dir, 'source')   # 模糊图在 source 文件夹
        self.sharp_dir = os.path.join(self.split_dir, 'target')  # 清晰图在 target 文件夹

        if not os.path.exists(self.blur_dir) or not os.path.exists(self.sharp_dir):
            raise FileNotFoundError(
                f"Source or Target directory not found in {self.split_dir}. "
                f"Expected 'source' and 'target' subdirectories."
            )

        # Get file lists
        self.blur_files = sorted([f for f in os.listdir(self.blur_dir) if self._is_image(f)])
        self.sharp_files = sorted([f for f in os.listdir(self.sharp_dir) if self._is_image(f)])

        # Verify integrity
        if len(self.blur_files) != len(self.sharp_files):
            raise ValueError(
                f"Mismatch number of images: {len(self.blur_files)} in source vs "
                f"{len(self.sharp_files)} in target"
            )
        
        self._real_length = len(self.blur_files)
        
        print(f"[DPDDDataset] Mode: {mode}, Real samples: {self._real_length}, "
              f"Repeat factor: {self.repeat_factor}, Virtual length: {len(self)}")

    def _is_image(self, filename):
        return filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))

    def __len__(self):
        """
        返回虚拟长度。
        训练集: real_length * repeat_factor (例如 350 * 100 = 35000)
        验证/测试集: real_length
        """
        return self._real_length * self.repeat_factor

    def get_real_length(self):
        """返回真实样本数量"""
        return self._real_length

    def __getitem__(self, idx):
        """
        获取一个样本。
        对于训练集，idx 会对真实长度取模，所以同一张图会被多次访问，
        但每次都会生成不同的随机裁剪位置。
        """
        # 真实索引映射
        real_idx = idx % self._real_length
        
        # Retrieve filenames
        blur_filename = self.blur_files[real_idx]
        sharp_filename = self.sharp_files[real_idx]

        blur_path = os.path.join(self.blur_dir, blur_filename)
        sharp_path = os.path.join(self.sharp_dir, sharp_filename)

        # Open images and convert to RGB
        blur_img = Image.open(blur_path).convert('RGB')
        sharp_img = Image.open(sharp_path).convert('RGB')

        # Get original image dimensions (PIL uses (width, height))
        W_orig, H_orig = blur_img.size
        
        # 根据模式选择裁剪策略
        if self.use_full_resolution:
            # 测试模式：使用全分辨率，不裁剪
            top, left = 0, 0
            crop_h, crop_w = H_orig, W_orig
            # 不裁剪，直接使用原图
            
        elif self.mode == 'train':
            # 训练模式：随机裁剪
            crop_size = self.crop_size
            if H_orig >= crop_size and W_orig >= crop_size:
                # Generate random crop parameters
                max_top = H_orig - crop_size
                max_left = W_orig - crop_size
                top = random.randint(0, max_top)
                left = random.randint(0, max_left)
                
                # PIL crop: (left, top, right, bottom)
                box = (left, top, left + crop_size, top + crop_size)
                blur_img = blur_img.crop(box)
                sharp_img = sharp_img.crop(box)
                crop_h, crop_w = crop_size, crop_size
            else:
                # Image smaller than crop_size, use padding
                top, left = 0, 0
                crop_h, crop_w = H_orig, W_orig
                
        else:
            # 验证模式：固定中心裁剪
            crop_size = self.val_crop_size
            if H_orig >= crop_size and W_orig >= crop_size:
                top = (H_orig - crop_size) // 2
                left = (W_orig - crop_size) // 2
                
                box = (left, top, left + crop_size, top + crop_size)
                blur_img = blur_img.crop(box)
                sharp_img = sharp_img.crop(box)
                crop_h, crop_w = crop_size, crop_size
            else:
                # Image smaller than crop_size, use full image
                top, left = 0, 0
                crop_h, crop_w = H_orig, W_orig
        
        # Compute normalized crop_info: [top/H_orig, left/W_orig, crop_h/H_orig, crop_w/W_orig]
        # This represents the crop location in the original image coordinates
        crop_info = torch.tensor(
            [top / H_orig, left / W_orig, crop_h / H_orig, crop_w / W_orig],
            dtype=torch.float32
        )
        
        # Data augmentation (train only)
        if self.mode == 'train' and self.random_flip:
            if random.random() < 0.5:
                blur_img = ImageOps.mirror(blur_img)
                sharp_img = ImageOps.mirror(sharp_img)
            if random.random() < 0.5:
                blur_img = ImageOps.flip(blur_img)
                sharp_img = ImageOps.flip(sharp_img)

        # Apply transforms to convert to tensors
        blur_tensor = self.transform(blur_img)
        sharp_tensor = self.transform(sharp_img)

        return {
            'blur': blur_tensor,
            'sharp': sharp_tensor,
            'crop_info': crop_info,
            'filename': blur_filename,
            'original_size': (H_orig, W_orig)
        }


class DPDDTestDataset(Dataset):
    """
    专门用于测试的 DPDD 数据集类。
    使用全分辨率图像，不进行任何裁剪。
    """
    
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (str): Root directory of the dataset.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        super().__init__()
        self.root_dir = root_dir
        
        if transform is None:
            self.transform = transforms.Compose([transforms.ToTensor()])
        else:
            self.transform = transform
        
        # 测试集路径
        self.split_dir = os.path.join(root_dir, 'test_c')
        self.blur_dir = os.path.join(self.split_dir, 'source')
        self.sharp_dir = os.path.join(self.split_dir, 'target')
        
        if not os.path.exists(self.blur_dir) or not os.path.exists(self.sharp_dir):
            raise FileNotFoundError(
                f"Test set directories not found in {self.split_dir}. "
                f"Expected 'source' and 'target' subdirectories."
            )
        
        valid_exts = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')
        self.blur_files = sorted([f for f in os.listdir(self.blur_dir) if f.lower().endswith(valid_exts)])
        self.sharp_files = sorted([f for f in os.listdir(self.sharp_dir) if f.lower().endswith(valid_exts)])
        
        if len(self.blur_files) != len(self.sharp_files):
            raise ValueError("Mismatch in test set image counts")
        
        print(f"[DPDDTestDataset] Loaded {len(self.blur_files)} test image pairs")
    
    def __len__(self):
        return len(self.blur_files)
    
    def __getitem__(self, idx):
        blur_filename = self.blur_files[idx]
        sharp_filename = self.sharp_files[idx]
        
        blur_path = os.path.join(self.blur_dir, blur_filename)
        sharp_path = os.path.join(self.sharp_dir, sharp_filename)
        
        blur_img = Image.open(blur_path).convert('RGB')
        sharp_img = Image.open(sharp_path).convert('RGB')
        
        W_orig, H_orig = blur_img.size
        
        # 全分辨率，crop_info 表示完整图像
        crop_info = torch.tensor([0.0, 0.0, 1.0, 1.0], dtype=torch.float32)
        
        blur_tensor = self.transform(blur_img)
        sharp_tensor = self.transform(sharp_img)
        
        return {
            'blur': blur_tensor,
            'sharp': sharp_tensor,
            'crop_info': crop_info,
            'filename': blur_filename,
            'original_size': (H_orig, W_orig)
        }
