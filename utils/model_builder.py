import torch
import torch.nn as nn
import os
from typing import Optional
from config import Config
from models.zernike import DifferentiableZernikeGenerator
from models.aberration_net import AberrationNet, PolynomialAberrationNet
from models.restoration_net import RestorationNet
from models.physical_layer import SpatiallyVaryingPhysicalLayer
from trainer import DualBranchTrainer
from utils.dpdd_dataset import DPDDDataset, DPDDTestDataset
from torch.utils.data import DataLoader

def build_models_from_config(config: Config, device: str):
    """根据配置构建所有模型组件
    
    Args:
        config: 配置对象
        device: 计算设备
    
    Returns:
        tuple: (zernike_gen, aberration_net, restoration_net, physical_layer)
    """
    
    use_physical_layer = getattr(config.experiment, "use_physical_layer", True)

    zernike_gen = None
    aberration_net = None
    physical_layer = None

    if use_physical_layer:
        # 1. Zernike 生成器
        zernike_gen = DifferentiableZernikeGenerator(
            n_modes=config.physics.n_modes,
            pupil_size=config.physics.pupil_size,
            kernel_size=config.physics.kernel_size,
            oversample_factor=config.physics.oversample_factor,
            wavelengths=config.physics.wavelengths,
            ref_wavelength=config.physics.ref_wavelength,
            device=device
        )
        
        # 2. 像差预测网络
        if config.aberration_net.type == "polynomial":
            aberration_net = PolynomialAberrationNet(
                n_coeffs=config.aberration_net.n_coeffs,
                degree=config.aberration_net.polynomial.degree,
                a_max=config.aberration_net.a_max
            ).to(device)
            print(f"  ├─ 像差网络: PolynomialAberrationNet (degree={config.aberration_net.polynomial.degree})")
        else:
            aberration_net = AberrationNet(
                num_coeffs=config.aberration_net.n_coeffs,
                hidden_dim=config.aberration_net.mlp.hidden_dim,
                a_max=config.aberration_net.mlp.a_max_mlp,
                use_fourier=config.aberration_net.mlp.use_fourier
            ).to(device)
            print(f"  ├─ 像差网络: AberrationNet (hidden_dim={config.aberration_net.mlp.hidden_dim})")
    
    # 3. 图像复原网络
    restoration_net = RestorationNet(
        n_channels=config.restoration_net.n_channels,
        n_classes=config.restoration_net.n_classes,
        bilinear=config.restoration_net.bilinear,
        base_filters=config.restoration_net.base_filters,
        use_coords=config.restoration_net.use_coords
    ).to(device)
    print(f"  ├─ 复原网络: RestorationNet (base_filters={config.restoration_net.base_filters}, use_coords={config.restoration_net.use_coords})")
    
    # 4. 物理卷积层
    if use_physical_layer:
        physical_layer = SpatiallyVaryingPhysicalLayer(
            aberration_net=aberration_net,
            zernike_generator=zernike_gen,
            patch_size=config.ola.patch_size,
            stride=config.ola.stride,
            pad_to_power_2=config.ola.pad_to_power_2,
            use_newbp=config.ola.use_newbp
        ).to(device)
        name_algo = "NewBP" if config.ola.use_newbp else "Standard"
        print(f"  └─ 物理层: OLA (patch={config.ola.patch_size}, stride={config.ola.stride}, algo={name_algo})")
    else:
        print("  └─ 物理层: disabled")
    
    return zernike_gen, aberration_net, restoration_net, physical_layer


def build_trainer_from_config(config: Config, restoration_net, physical_layer, device: str, tensorboard_dir: Optional[str] = None):
    """根据配置构建训练器
    
    Args:
        config: 配置对象
        restoration_net: 复原网络
        physical_layer: 物理卷积层
        device: 计算设备
    
    Returns:
        DualBranchTrainer 对象
    """
    # 获取 accumulation_steps
    if hasattr(config.training, 'accumulation_steps'):
        accumulation_steps = config.training.accumulation_steps
    else:
        accumulation_steps = 1
    
    # TensorBoard 配置
    if tensorboard_dir is None and hasattr(config, 'experiment') and hasattr(config.experiment, 'tensorboard'):
        tb_config = config.experiment.tensorboard
        if getattr(tb_config, 'enabled', False):
            base_dir = getattr(tb_config, 'log_dir', 'runs')
            if os.path.isabs(base_dir):
                tensorboard_dir = os.path.join(base_dir, config.experiment.name)
            else:
                tensorboard_dir = os.path.join(
                    config.experiment.output_dir,
                    base_dir,
                    config.experiment.name
                )
    
    # 熔断机制配置
    circuit_breaker_config = None
    if hasattr(config, 'checkpoint') and hasattr(config.checkpoint, 'circuit_breaker'):
        cb_config = config.checkpoint.circuit_breaker
        circuit_breaker_config = {
            'enabled': getattr(cb_config, 'enabled', False),
            'stage1_min_loss': getattr(cb_config, 'stage1_min_loss', 0.005),
            'stage2_min_psnr': getattr(cb_config, 'stage2_min_psnr', 30.0),
            'stage2_min_ssim': getattr(cb_config, 'stage2_min_ssim', 0.95)
        }
        
    trainer = DualBranchTrainer(
        restoration_net=restoration_net,
        physical_layer=physical_layer,
        lr_restoration=config.training.optimizer.lr_restoration,
        lr_optics=config.training.optimizer.lr_optics,
        optimizer_type=getattr(config.training.optimizer, 'type', 'adamw'),
        weight_decay=getattr(config.training.optimizer, 'weight_decay', 0.0),
        lambda_sup=config.training.loss.lambda_sup,
        lambda_coeff=config.training.loss.lambda_coeff,
        lambda_smooth=config.training.loss.lambda_smooth,
        lambda_image_reg=config.training.loss.lambda_image_reg,
        grad_clip_restoration=getattr(config.training.gradient_clip, 'restoration', 5.0),
        grad_clip_optics=getattr(config.training.gradient_clip, 'optics', 1.0),
        stage_schedule=config.training.stage_schedule,
        stage_weights=config.training.stage_weights,
        smoothness_grid_size=config.training.smoothness_grid_size,
        accumulation_steps=accumulation_steps,
        device=device,
        tensorboard_dir=tensorboard_dir,
        circuit_breaker_config=circuit_breaker_config
    )
    
    return trainer

def build_dataloader_from_config(config: Config, mode: str = 'train'):
    """根据配置构建 DataLoader
    
    Args:
        config: 配置对象
        mode: 数据集模式 ('train', 'val', 'test')
    
    Returns:
        DataLoader 对象
    """
    # 获取配置参数
    crop_size = config.data.crop_size
    val_crop_size = getattr(config.data, 'val_crop_size', 1024)
    repeat_factor = getattr(config.data, 'repeat_factor', 1) if mode == 'train' else 1
    
    # 测试集使用全分辨率
    use_full_resolution = (mode == 'test')
    
    # 创建数据集
    dataset = DPDDDataset(
        root_dir=config.data.data_root, 
        mode=mode, 
        crop_size=crop_size,
        repeat_factor=repeat_factor,
        val_crop_size=val_crop_size,
        use_full_resolution=use_full_resolution,
        random_flip=getattr(getattr(config.data, 'augmentation', None), 'random_flip', False),
        transform=None  # Default ToTensor
    )
    
    # 只有训练集需要 shuffle
    shuffle = (mode == 'train')
    
    # 测试集 batch_size 通常为 1（全分辨率图像较大）
    batch_size = 1 if mode == 'test' else config.data.batch_size
    
    loader = DataLoader(
        dataset, 
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=config.data.num_workers,
        pin_memory=True if config.experiment.device == 'cuda' else False,
        drop_last=(mode == 'train')  # 训练时丢弃不完整的最后一个 batch
    )
    
    return loader


def build_test_dataloader_from_config(config: Config):
    """专门构建测试集 DataLoader（全分辨率）
    
    Args:
        config: 配置对象
    
    Returns:
        DataLoader 对象
    """
    dataset = DPDDTestDataset(
        root_dir=config.data.data_root,
        transform=None
    )
    
    loader = DataLoader(
        dataset,
        batch_size=1,  # 测试集使用 batch_size=1
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=True if config.experiment.device == 'cuda' else False
    )
    
    return loader
