"""
配置管理模块 (Configuration Manager)
=====================================

提供统一的配置加载、验证和访问接口。
支持 YAML 文件加载、命令行覆盖和默认值回退。

使用方法:
---------
    from config import Config, load_config
    
    # 方式 1: 加载默认配置
    config = load_config()
    
    # 方式 2: 加载指定配置文件
    config = load_config('config/experiment1.yaml')
    
    # 方式 3: 命令行覆盖
    config = load_config('config/default.yaml', overrides=['training.epochs=200'])
    
    # 访问配置
    kernel_size = config.physics.kernel_size
    lr = config.training.optimizer.lr_restoration
"""

import os
import yaml
from dataclasses import dataclass, field
from typing import List, Optional, Any, Dict
from pathlib import Path


# =============================================================================
# 数据类定义 (Dataclass Definitions)
# =============================================================================

@dataclass
class PhysicsConfig:
    """物理光学参数配置"""
    n_modes: int = 15
    pupil_size: int = 64
    kernel_size: int = 33
    oversample_factor: int = 2
    ref_wavelength: float = 550e-9
    wavelengths: List[float] = field(default_factory=lambda: [620e-9, 550e-9, 450e-9])
    
    def __post_init__(self):
        if self.kernel_size % 2 == 0:
            raise ValueError(f"kernel_size 必须为奇数，当前值: {self.kernel_size}")
        if self.n_modes < 1 or self.n_modes > 15:
            raise ValueError(f"n_modes 必须在 1-15 之间，当前值: {self.n_modes}")


@dataclass
class OLAConfig:
    """空间变化卷积层配置"""
    patch_size: int = 128
    stride: int = 64
    pad_to_power_2: bool = True
    use_newbp: bool = True
    
    def __post_init__(self):
        if self.stride > self.patch_size:
            raise ValueError(f"stride ({self.stride}) 不能大于 patch_size ({self.patch_size})")


@dataclass
class PolynomialConfig:
    """多项式像差网络配置"""
    degree: int = 2


@dataclass
class MLPConfig:
    """MLP 像差网络配置"""
    hidden_dim: int = 64
    use_fourier: bool = True
    fourier_scale: int = 5
    a_max_mlp: float = 3.0


@dataclass
class AberrationNetConfig:
    """像差预测网络配置"""
    n_coeffs: int = 15
    a_max: float = 2.0
    type: str = "polynomial"  # "polynomial" 或 "mlp"
    polynomial: PolynomialConfig = field(default_factory=PolynomialConfig)
    mlp: MLPConfig = field(default_factory=MLPConfig)
    
    def __post_init__(self):
        if self.type not in ["polynomial", "mlp"]:
            raise ValueError(f"type 必须是 'polynomial' 或 'mlp'，当前值: {self.type}")


@dataclass
class RestorationNetConfig:
    """图像复原网络配置"""
    n_channels: int = 3
    n_classes: int = 3
    base_filters: int = 64
    bilinear: bool = True
    use_coords: bool = True
    channel_multipliers: List[int] = field(default_factory=lambda: [1, 2, 4, 8, 8])


@dataclass
class OptimizerConfig:
    """优化器配置"""
    type: str = "adamw"
    lr_restoration: float = 1e-4
    lr_optics: float = 1e-5
    weight_decay: float = 0.01


@dataclass
class LossConfig:
    """损失函数配置"""
    lambda_sup: float = 0.0
    lambda_coeff: float = 0.01
    lambda_smooth: float = 0.01
    lambda_image_reg: float = 0.001


@dataclass
class GradientClipConfig:
    """梯度裁剪配置"""
    restoration: float = 5.0
    optics: float = 1.0


@dataclass
class StageScheduleConfig:
    """三阶段训练阶段配置"""
    stage1_epochs: int = 50   # Stage 1: Physics Only
    stage2_epochs: int = 200  # Stage 2: Restoration with Fixed Physics
    stage3_epochs: int = 50   # Stage 3: Joint Fine-tuning (学习率减半)


@dataclass
class TrainingConfig:
    """训练配置"""
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    gradient_clip: GradientClipConfig = field(default_factory=GradientClipConfig)
    stage_weights: Dict[str, Dict[str, float]] = field(default_factory=dict)
    smoothness_grid_size: int = 16
    accumulation_steps: int = 1
    stage_schedule: StageScheduleConfig = field(default_factory=StageScheduleConfig)


@dataclass
class AugmentationConfig:
    """数据增强配置"""
    random_flip: bool = True


@dataclass
class DataConfig:
    """数据配置"""
    data_root: str = "data/dd_dp_dataset_png"  # 指向原始 DPDD 数据集
    batch_size: int = 4
    image_height: int = 1120           # 原始图像高度
    image_width: int = 1680            # 原始图像宽度
    crop_size: int = 512               # 训练时随机裁剪尺寸
    val_crop_size: int = 1024          # 验证集中心裁剪尺寸
    num_workers: int = 4
    repeat_factor: int = 100           # 虚拟长度倍数
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)


@dataclass
class PSFGridConfig:
    """PSF 网格可视化配置"""
    rows: int = 5
    cols: int = 5
    coord_range: List[float] = field(default_factory=lambda: [-0.9, 0.9])
    colormap: str = "inferno"


@dataclass
class CoeffMapsConfig:
    """系数分布可视化配置"""
    grid_size: int = 128
    indices: List[int] = field(default_factory=lambda: [3, 4, 5, 6])
    colormap: str = "viridis"


@dataclass
class VisualizationConfig:
    """可视化配置"""
    psf_grid: PSFGridConfig = field(default_factory=PSFGridConfig)
    coeff_maps: CoeffMapsConfig = field(default_factory=CoeffMapsConfig)


@dataclass
class ExperimentConfig:
    """实验配置"""
    name: str = "default"
    seed: int = 42
    device: str = "cuda"
    use_physical_layer: bool = True
    epochs: int = 300  # 总训练轮数 (需与 stage_schedule 总和一致)
    save_interval: int = 20  # 定期存档间隔
    log_interval: int = 1
    output_dir: str = "results"
    run_name: Optional[str] = None
    use_timestamp: bool = True
    timestamp_format: str = "%m%d_%H%M"
    checkpoints_subdir: str = "checkpoints"
    tensorboard: 'TensorBoardConfig' = field(default_factory=lambda: TensorBoardConfig())


@dataclass
class TensorBoardConfig:
    """TensorBoard 配置"""
    enabled: bool = True
    log_dir: str = "runs"
    append_run_name: bool = False
    log_images: bool = True
    image_log_interval: int = 10


@dataclass
class CircuitBreakerConfig:
    """熔断机制配置"""
    enabled: bool = True
    stage1_min_loss: float = 0.5     # Stage 1 验证 Loss 需低于此值才能进入 Stage 2
    stage2_min_psnr: float = 20.0    # Stage 2 验证 PSNR 需高于此值才能进入 Stage 3
    stage2_min_ssim: float = 0.0     # Stage 2 验证 SSIM 需高于此值才能进入 Stage 3 (0.0 表示不启用)


@dataclass
class CheckpointConfig:
    """检查点保存策略配置"""
    save_best_per_stage: bool = True
    stage1_metric: str = "reblur_mse"  # Stage 1 使用重模糊误差
    stage2_metric: str = "psnr"        # Stage 2 使用 PSNR
    stage3_metric: str = "combined"    # Stage 3 使用综合指标
    circuit_breaker: CircuitBreakerConfig = field(default_factory=CircuitBreakerConfig)
    save_interval: int = 10
    log_interval: int = 1
    output_dir: str = "results"


@dataclass
class Config:
    """主配置类"""
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    ola: OLAConfig = field(default_factory=OLAConfig)
    aberration_net: AberrationNetConfig = field(default_factory=AberrationNetConfig)
    restoration_net: RestorationNetConfig = field(default_factory=RestorationNetConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    
    def __str__(self):
        """格式化打印配置"""
        return yaml.dump(self.to_dict(), default_flow_style=False, allow_unicode=True)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return _dataclass_to_dict(self)
    
    def save(self, path: str):
        """保存配置到 YAML 文件"""
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, allow_unicode=True)
        print(f"配置已保存到: {path}")


# =============================================================================
# 辅助函数 (Helper Functions)
# =============================================================================

def _dataclass_to_dict(obj) -> Any:
    """递归将 dataclass 转换为字典"""
    if hasattr(obj, '__dataclass_fields__'):
        return {k: _dataclass_to_dict(v) for k, v in obj.__dict__.items()}
    elif isinstance(obj, list):
        return [_dataclass_to_dict(item) for item in obj]
    else:
        return obj


def _dict_to_dataclass(cls, data: Dict[str, Any]):
    """递归将字典转换为 dataclass"""
    if data is None:
        return cls()
    
    field_types = {f.name: f.type for f in cls.__dataclass_fields__.values()}
    kwargs = {}
    
    for field_name, field_type in field_types.items():
        if field_name in data:
            value = data[field_name]
            
            # 检查是否是嵌套的 dataclass
            if hasattr(field_type, '__dataclass_fields__'):
                kwargs[field_name] = _dict_to_dataclass(field_type, value)
            else:
                kwargs[field_name] = value
    
    return cls(**kwargs)


def _apply_overrides(config_dict: Dict[str, Any], overrides: List[str]) -> Dict[str, Any]:
    """应用命令行覆盖参数
    
    格式: key1.key2.key3=value
    例如: training.optimizer.lr_restoration=0.001
    """
    for override in overrides:
        if '=' not in override:
            raise ValueError(f"无效的覆盖格式: {override}，应为 key=value")
        
        key_path, value_str = override.split('=', 1)
        keys = key_path.split('.')
        
        # 解析值类型
        try:
            # 尝试解析为数字
            if '.' in value_str:
                value = float(value_str)
            elif value_str.lower() in ('true', 'false'):
                value = value_str.lower() == 'true'
            elif value_str.startswith('[') and value_str.endswith(']'):
                # 简单列表解析
                value = yaml.safe_load(value_str)
            else:
                value = int(value_str)
        except ValueError:
            value = value_str
        
        # 设置嵌套值
        current = config_dict
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        current[keys[-1]] = value
    
    return config_dict


# =============================================================================
# 主要 API (Main API)
# =============================================================================

def load_config(
    config_path: Optional[str] = None,
    overrides: Optional[List[str]] = None
) -> Config:
    """加载配置文件
    
    Args:
        config_path: YAML 配置文件路径。如果为 None，使用默认配置。
        overrides: 命令行覆盖参数列表，格式为 ["key1.key2=value", ...]
    
    Returns:
        Config 对象
    
    Examples:
        # 加载默认配置
        config = load_config()
        
        # 加载自定义配置
        config = load_config('config/experiment1.yaml')
        
        # 带命令行覆盖
        config = load_config(overrides=['training.epochs=200', 'data.batch_size=4'])
    """
    # 确定配置文件路径
    if config_path is None:
        # 默认配置文件路径
        default_path = Path(__file__).parent / 'default.yaml'
        if default_path.exists():
            config_path = str(default_path)
        else:
            print("未找到默认配置文件，使用内置默认值")
            return Config()
    
    # 加载 YAML 文件
    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)
    
    # 应用覆盖参数
    if overrides:
        config_dict = _apply_overrides(config_dict, overrides)
    
    # 递归构建 Config 对象
    config = _build_config_from_dict(config_dict)
    
    print(f"✓ 配置已加载: {config_path}")
    return config


def _build_config_from_dict(data: Dict[str, Any]) -> Config:
    """从字典构建 Config 对象"""
    
    # 手动处理嵌套结构
    physics = _dict_to_dataclass(PhysicsConfig, data.get('physics', {}))
    ola = _dict_to_dataclass(OLAConfig, data.get('ola', {}))
    
    # 处理 aberration_net 的嵌套
    ab_data = data.get('aberration_net', {})
    polynomial = _dict_to_dataclass(PolynomialConfig, ab_data.get('polynomial', {}))
    mlp = _dict_to_dataclass(MLPConfig, ab_data.get('mlp', {}))
    aberration_net = AberrationNetConfig(
        n_coeffs=ab_data.get('n_coeffs', 15),
        a_max=ab_data.get('a_max', 2.0),
        type=ab_data.get('type', 'polynomial'),
        polynomial=polynomial,
        mlp=mlp
    )
    
    restoration_net = _dict_to_dataclass(RestorationNetConfig, data.get('restoration_net', {}))
    
    # 处理 training 的嵌套
    tr_data = data.get('training', {})
    optimizer = _dict_to_dataclass(OptimizerConfig, tr_data.get('optimizer', {}))
    loss = _dict_to_dataclass(LossConfig, tr_data.get('loss', {}))
    gradient_clip = _dict_to_dataclass(GradientClipConfig, tr_data.get('gradient_clip', {}))
    stage_schedule = _dict_to_dataclass(StageScheduleConfig, tr_data.get('stage_schedule', {}))
    training = TrainingConfig(
        optimizer=optimizer,
        loss=loss,
        gradient_clip=gradient_clip,
        stage_weights=tr_data.get('stage_weights', {}),
        smoothness_grid_size=tr_data.get('smoothness_grid_size', 16),
        accumulation_steps=tr_data.get('accumulation_steps', 1),
        stage_schedule=stage_schedule
    )
    
    # 处理 data 的嵌套
    # CONFIG_FIX: 添加缺失的 data_root, crop_size, val_crop_size, repeat_factor 字段解析
    d_data = data.get('data', {})
    augmentation = _dict_to_dataclass(AugmentationConfig, d_data.get('augmentation', {}))
    data_config = DataConfig(
        data_root=d_data.get('data_root', 'data/dd_dp_dataset_png'),
        batch_size=d_data.get('batch_size', 2),
        image_height=d_data.get('image_height', 1120),
        image_width=d_data.get('image_width', 1680),
        crop_size=d_data.get('crop_size', 512),
        val_crop_size=d_data.get('val_crop_size', 1024),
        num_workers=d_data.get('num_workers', 4),
        repeat_factor=d_data.get('repeat_factor', 100),
        augmentation=augmentation
    )
    
    # 处理 visualization 的嵌套
    vis_data = data.get('visualization', {})
    psf_grid = _dict_to_dataclass(PSFGridConfig, vis_data.get('psf_grid', {}))
    coeff_maps = _dict_to_dataclass(CoeffMapsConfig, vis_data.get('coeff_maps', {}))
    visualization = VisualizationConfig(psf_grid=psf_grid, coeff_maps=coeff_maps)
    
    # CONFIG_FIX: 修复 experiment 的 tensorboard 嵌套解析
    exp_data = data.get('experiment', {})
    tensorboard = _dict_to_dataclass(TensorBoardConfig, exp_data.get('tensorboard', {}))
    experiment = ExperimentConfig(
        name=exp_data.get('name', 'default'),
        seed=exp_data.get('seed', 42),
        device=exp_data.get('device', 'cuda'),
        use_physical_layer=exp_data.get('use_physical_layer', True),
        epochs=exp_data.get('epochs', 300),
        save_interval=exp_data.get('save_interval', 20),
        log_interval=exp_data.get('log_interval', 1),
        output_dir=exp_data.get('output_dir', 'results'),
        run_name=exp_data.get('run_name'),
        use_timestamp=exp_data.get('use_timestamp', True),
        timestamp_format=exp_data.get('timestamp_format', '%m%d_%H%M'),
        checkpoints_subdir=exp_data.get('checkpoints_subdir', 'checkpoints'),
        tensorboard=tensorboard
    )
    
    # CONFIG_FIX: 添加缺失的 checkpoint 配置解析
    ckpt_data = data.get('checkpoint', {})
    circuit_breaker = _dict_to_dataclass(CircuitBreakerConfig, ckpt_data.get('circuit_breaker', {}))
    checkpoint = CheckpointConfig(
        save_best_per_stage=ckpt_data.get('save_best_per_stage', True),
        stage1_metric=ckpt_data.get('stage1_metric', 'reblur_mse'),
        stage2_metric=ckpt_data.get('stage2_metric', 'psnr'),
        stage3_metric=ckpt_data.get('stage3_metric', 'combined'),
        circuit_breaker=circuit_breaker,
        save_interval=ckpt_data.get('save_interval', 10),
        log_interval=ckpt_data.get('log_interval', 1),
        output_dir=ckpt_data.get('output_dir', 'results')
    )
    
    return Config(
        physics=physics,
        ola=ola,
        aberration_net=aberration_net,
        restoration_net=restoration_net,
        training=training,
        data=data_config,
        visualization=visualization,
        experiment=experiment,
        checkpoint=checkpoint  # CONFIG_FIX: 添加 checkpoint 配置
    )


def get_default_config() -> Config:
    """获取默认配置"""
    return Config()


# =============================================================================
# 命令行接口 (CLI)
# =============================================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='配置管理工具')
    parser.add_argument('--config', '-c', type=str, default=None, help='配置文件路径')
    parser.add_argument('--print', '-p', action='store_true', help='打印配置')
    parser.add_argument('--save', '-s', type=str, default=None, help='保存配置到指定路径')
    parser.add_argument('overrides', nargs='*', help='覆盖参数，格式: key1.key2=value')
    
    args = parser.parse_args()
    
    config = load_config(args.config, args.overrides if args.overrides else None)
    
    if args.print:
        print("\n" + "="*60)
        print("当前配置:")
        print("="*60)
        print(config)
    
    if args.save:
        config.save(args.save)
