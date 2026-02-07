import torch
import torch.nn as nn
import torch.optim as optim
import os
import time
from typing import Any, Mapping, Optional, Dict, Union

# TensorBoard 支持
try:
    # 尝试从 writer 子模块导入 (解决 IDE 静态检查报错)
    from torch.utils.tensorboard.writer import SummaryWriter as _SummaryWriter
    SummaryWriter = _SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    try:
        # 回退到标准导入方式
        from torch.utils.tensorboard import SummaryWriter as _SummaryWriter  # type: ignore[attr-defined]
        SummaryWriter = _SummaryWriter
        TENSORBOARD_AVAILABLE = True
    except ImportError:
        TENSORBOARD_AVAILABLE = False
        SummaryWriter = None

'''
================================================================================
                    三阶段解耦训练策略 (Three-Stage Decoupled Training)
================================================================================

┌─────────────────────────────────────────────────────────────────────────────┐
│  Stage 1: Physics Only (物理层单独训练) - 50 Epochs                          │
├─────────────────────────────────────────────────────────────────────────────┤
│  目的: 利用成对数据，单独训练 AberrationNet 准确拟合数据集的光学像差特性     │
│                                                                              │
│  数据流:                                                                     │
│    X_gt (清晰图像) ──▶ PhysicalLayer ──▶ Y_hat (重模糊)                     │
│                                                                              │
│  Loss = MSE(Y_hat, Y) + λ_coeff × ||coeffs||² + λ_smooth × TV(coeffs)       │
│                                                                              │
│  冻结: RestorationNet (❄️)     更新: AberrationNet (🔥)                      │
│  验证判据: Re-blur MSE (重模糊一致性误差)                                    │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│  Stage 2: Restoration with Fixed Physics (固定物理层训练复原网络) - 200 Epochs│
├─────────────────────────────────────────────────────────────────────────────┤
│  目的: 在已知且准确的物理模型指导下，训练复原网络                            │
│                                                                              │
│  数据流:                                                                     │
│    Y (模糊图像) ──▶ RestorationNet ──▶ X_hat ──▶ PhysicalLayer ──▶ Y_hat   │
│                                                                              │
│  Loss = λ_sup × L1(X_hat, X_gt) + MSE(Y_hat, Y) + λ_image_reg × TV(X_hat)  │
│                                                                              │
│  冻结: AberrationNet (❄️)      更新: RestorationNet (🔥)                     │
│  验证判据: Validation PSNR & SSIM                                            │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│  Stage 3: Joint Fine-tuning (联合微调) - 50 Epochs                           │
├─────────────────────────────────────────────────────────────────────────────┤
│  目的: 联合微调，消除模块间的耦合误差                                        │
│                                                                              │
│  数据流:                                                                     │
│    Y ──▶ RestorationNet ──▶ X_hat ──▶ PhysicalLayer ──▶ Y_hat              │
│                                                                              │
│  Loss = 综合损失（所有项）                                                   │
│  学习率: 减半 (lr_restoration / 2, lr_optics / 2)                           │
│                                                                              │
│  更新: RestorationNet (🔥) + AberrationNet (🔥)                              │
│  验证判据: 综合指标 (PSNR + 物理约束)                                        │
└─────────────────────────────────────────────────────────────────────────────┘
'''
class DualBranchTrainer:
    """
    三阶段解耦训练器 (Three-Stage Decoupled Trainer)

    支持三种训练模式:
    - 'physics_only': 仅训练物理层 (Stage 1)
    - 'restoration_fixed_physics': 固定物理层训练复原网络 (Stage 2)
    - 'joint': 联合训练所有模块 (Stage 3)
    
    特性:
    - 动态损失权重调整
    - 熔断机制 (Circuit Breaker)
    - TensorBoard 日志
    - Stage 3 学习率自动减半
    """

    VALID_STAGES = ('physics_only', 'restoration_fixed_physics', 'joint', 'restoration_only')

    def __init__(self,
                 restoration_net,
                 physical_layer,
                 lr_restoration,
                 lr_optics,
                 optimizer_type="adamw",
                 weight_decay=0.0,
                 lambda_sup=1.0,
                 lambda_coeff=0.05,
                 lambda_smooth=0.1,
                 lambda_image_reg=0.0,
                 grad_clip_restoration=5.0,
                 grad_clip_optics=1.0,
                 stage_schedule=None,
                 stage_weights=None,
                 smoothness_grid_size=16,
                 device='cuda',
                 accumulation_steps=4,
                 tensorboard_dir=None,
                 circuit_breaker_config=None):

        self.device = device
        self.restoration_net = restoration_net.to(device)
        self.physical_layer = physical_layer.to(device) if physical_layer is not None else None
        self.use_physical_layer = self.physical_layer is not None

        # Access internals for regularization
        self.aberration_net = self.physical_layer.aberration_net if self.physical_layer is not None else None

        # 保存原始学习率用于 Stage 3 减半
        self.base_lr_restoration = lr_restoration
        self.base_lr_optics = lr_optics
        
        # 独立优化器
        opt_type = str(optimizer_type).lower()
        if opt_type == "adamw":
            optimizer_cls = optim.AdamW
        elif opt_type == "adam":
            optimizer_cls = optim.Adam
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

        self.optimizer_W = optimizer_cls(self.restoration_net.parameters(), lr=lr_restoration, weight_decay=weight_decay)
        self.optimizer_Theta = None
        if self.aberration_net is not None:
            self.optimizer_Theta = optimizer_cls(self.aberration_net.parameters(), lr=lr_optics, weight_decay=weight_decay)

        # 兼容旧配置（已弃用的固定权重，仅保留字段）
        self.lambda_sup = lambda_sup
        self.lambda_coeff = lambda_coeff
        self.lambda_smooth = lambda_smooth
        self.lambda_image_reg = lambda_image_reg

        self.stage_weights = stage_weights if stage_weights is not None else {}
        
        # 三阶段调度 (可为 dict 或 dataclass)
        default_schedule = {
            'stage1_epochs': 50,
            'stage2_epochs': 200,
            'stage3_epochs': 50
        }
        self.stage_schedule: Any = stage_schedule if stage_schedule is not None else default_schedule

        # 平滑正则采样网格大小
        self.smoothness_grid_size = smoothness_grid_size

        # 梯度累积
        self.accumulation_steps = max(1, accumulation_steps)
        self.accumulation_counter = 0

        # 梯度裁剪阈值
        self.grad_clip_restoration = grad_clip_restoration
        self.grad_clip_optics = grad_clip_optics

        # 损失函数
        self.criterion_mse = nn.MSELoss()
        self.criterion_l1 = nn.L1Loss()

        # 当前训练阶段
        self._current_stage = 'joint' if self.use_physical_layer else 'restoration_only'
        self._previous_stage = None  # 用于检测阶段切换
        self._stage3_lr_halved = False  # 标记 Stage 3 学习率是否已减半
        self._forced_stage = None  # 强制阶段 (用于熔断阻断切换)

        # 熔断机制配置
        self.circuit_breaker_config = circuit_breaker_config or {
            'enabled': True,
            'stage1_min_loss': 0.005,
            'stage2_min_psnr': 30.0
        }
        self.circuit_breaker_triggered = False
        self.circuit_breaker_message = ""

        # TensorBoard
        self.writer = None  # type: Optional[Any]
        if tensorboard_dir and TENSORBOARD_AVAILABLE and SummaryWriter is not None:
            self.writer = SummaryWriter(log_dir=tensorboard_dir)
            print(f"[TensorBoard] Logging to: {tensorboard_dir}")
        elif tensorboard_dir and not TENSORBOARD_AVAILABLE:
            print("[Warning] TensorBoard not available. Install with: pip install tensorboard")

        # History
        self.history = {
            'loss_total': [], 'loss_data': [], 'loss_sup': [],
            'grad_norm_W': [], 'grad_norm_Theta': []
        }
        
        # 各阶段最佳验证指标
        self.best_metrics = {
            'physics_only': {'reblur_mse': float('inf')},
            'restoration_fixed_physics': {'psnr': 0.0, 'ssim': 0.0},
            'joint': {'psnr': 0.0, 'combined': 0.0},
            'restoration_only': {'psnr': 0.0}
        }

    # =========================================================================
    #                          阶段调度与冻结策略
    # =========================================================================
    def _get_stage(self, epoch: int) -> str:
        """根据 epoch(0-indexed) 获取当前阶段"""
        if not self.use_physical_layer:
            return 'restoration_only'
        if isinstance(self.stage_schedule, Mapping):
            s1 = self.stage_schedule.get('stage1_epochs', 50)
            s2 = self.stage_schedule.get('stage2_epochs', 200)
        else:
            s1 = getattr(self.stage_schedule, 'stage1_epochs', 50)
            s2 = getattr(self.stage_schedule, 'stage2_epochs', 200)

        if epoch < s1:
            return 'physics_only'
        elif epoch < s1 + s2:
            return 'restoration_fixed_physics'
        return 'joint'

    def _adjust_learning_rate_for_stage3(self):
        """Stage 3 学习率减半"""
        if not self.use_physical_layer:
            return
        if self._stage3_lr_halved:
            return  # 已经调整过
        
        new_lr_W = self.base_lr_restoration / 2.0
        new_lr_Theta = self.base_lr_optics / 2.0
        
        for param_group in self.optimizer_W.param_groups:
            param_group['lr'] = new_lr_W
        if self.optimizer_Theta is not None:
            for param_group in self.optimizer_Theta.param_groups:
                param_group['lr'] = new_lr_Theta
        
        self._stage3_lr_halved = True
        print(f"[Stage 3] Learning rate halved: lr_restoration={new_lr_W:.2e}, lr_optics={new_lr_Theta:.2e}")

    def check_circuit_breaker(self, val_metrics: Dict[str, float], current_stage: str, next_stage: str) -> bool:
        """
        熔断机制检查：验证当前阶段是否达到切换条件
        
        Args:
            val_metrics: 验证集指标字典
            current_stage: 当前训练阶段
            next_stage: 即将进入的阶段
        
        Returns:
            bool: True 表示可以切换，False 表示熔断（不允许切换）
        """
        if not self.use_physical_layer:
            return True
        if not self.circuit_breaker_config.get('enabled', False):
            return True  # 熔断机制未启用，允许切换
        
        # Stage 1 -> Stage 2: 检查重模糊误差
        if current_stage == 'physics_only' and next_stage == 'restoration_fixed_physics':
            reblur_mse = val_metrics.get('Reblur_MSE', val_metrics.get('reblur_mse', float('inf')))
            threshold = self.circuit_breaker_config.get('stage1_min_loss', 0.5)
            
            if reblur_mse > threshold:
                self.circuit_breaker_triggered = True
                self.circuit_breaker_message = (
                    f"[Circuit Breaker] Stage 1 -> 2 BLOCKED: "
                    f"Reblur MSE ({reblur_mse:.4f}) > threshold ({threshold:.4f}). "
                    f"Physics layer not ready. Continuing Stage 1..."
                )
                return False
        
        # Stage 2 -> Stage 3: 检查 PSNR 和 SSIM
        if current_stage == 'restoration_fixed_physics' and next_stage == 'joint':
            psnr = val_metrics.get('PSNR', val_metrics.get('psnr', 0.0))
            ssim = val_metrics.get('SSIM', val_metrics.get('ssim', 0.0))
            
            psnr_threshold = self.circuit_breaker_config.get('stage2_min_psnr', 20.0)
            ssim_threshold = self.circuit_breaker_config.get('stage2_min_ssim', 0.0)
            
            if psnr < psnr_threshold:
                self.circuit_breaker_triggered = True
                self.circuit_breaker_message = (
                    f"[Circuit Breaker] Stage 2 -> 3 BLOCKED: "
                    f"PSNR ({psnr:.2f}) < threshold ({psnr_threshold:.2f}). "
                    f"Restoration network not ready. Continuing Stage 2..."
                )
                return False

            if ssim < ssim_threshold:
                self.circuit_breaker_triggered = True
                self.circuit_breaker_message = (
                    f"[Circuit Breaker] Stage 2 -> 3 BLOCKED: "
                    f"SSIM ({ssim:.4f}) < threshold ({ssim_threshold:.4f}). "
                    f"Restoration network structural quality low. Continuing Stage 2..."
                )
                return False
        
        self.circuit_breaker_triggered = False
        self.circuit_breaker_message = ""
        return True

    def update_best_metrics(self, val_metrics: Dict[str, float], stage: str) -> Dict[str, bool]:
        """
        更新各阶段最佳指标并判断是否需要保存模型
        
        Returns:
            dict: 各指标是否为新最佳值
        """
        is_best = {}
        
        if stage == 'physics_only':
            reblur_mse = val_metrics.get('Reblur_MSE', val_metrics.get('reblur_mse', float('inf')))
            if reblur_mse < self.best_metrics['physics_only']['reblur_mse']:
                self.best_metrics['physics_only']['reblur_mse'] = reblur_mse
                is_best['reblur_mse'] = True
            else:
                is_best['reblur_mse'] = False
                
        elif stage == 'restoration_fixed_physics':
            psnr = val_metrics.get('PSNR', val_metrics.get('psnr', 0.0))
            ssim = val_metrics.get('SSIM', val_metrics.get('ssim', 0.0))
            
            if psnr > self.best_metrics['restoration_fixed_physics']['psnr']:
                self.best_metrics['restoration_fixed_physics']['psnr'] = psnr
                is_best['psnr'] = True
            else:
                is_best['psnr'] = False
                
            if ssim > self.best_metrics['restoration_fixed_physics']['ssim']:
                self.best_metrics['restoration_fixed_physics']['ssim'] = ssim
                is_best['ssim'] = True
            else:
                is_best['ssim'] = False
                
        elif stage == 'joint':
            psnr = val_metrics.get('PSNR', val_metrics.get('psnr', 0.0))
            reblur_mse = val_metrics.get('Reblur_MSE', val_metrics.get('reblur_mse', float('inf')))
            # 综合指标: PSNR 越高越好，Reblur_MSE 越低越好
            # combined = PSNR - 10 * Reblur_MSE (经验公式)
            combined = psnr - 10.0 * reblur_mse
            
            if psnr > self.best_metrics['joint']['psnr']:
                self.best_metrics['joint']['psnr'] = psnr
                is_best['psnr'] = True
            else:
                is_best['psnr'] = False
                
            if combined > self.best_metrics['joint']['combined']:
                self.best_metrics['joint']['combined'] = combined
                is_best['combined'] = True
            else:
                is_best['combined'] = False
        elif stage == 'restoration_only':
            psnr = val_metrics.get('PSNR', val_metrics.get('psnr', 0.0))
            if psnr > self.best_metrics.get('restoration_only', {}).get('psnr', 0.0):
                self.best_metrics.setdefault('restoration_only', {})['psnr'] = psnr
                is_best['psnr'] = True
            else:
                is_best['psnr'] = False
        
        return is_best

    def log_to_tensorboard(self, metrics: Dict[str, float], epoch: int, prefix: str = 'train'):
        """记录指标到 TensorBoard"""
        if self.writer is None:
            return
        
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and not (isinstance(value, float) and (value != value)):  # 排除 NaN
                self.writer.add_scalar(f'{prefix}/{key}', value, epoch)
    
    def log_gradients_to_tensorboard(self, epoch: int):
        """记录梯度分布到 TensorBoard"""
        if self.writer is None:
            return
        
        # 记录复原网络梯度
        for name, param in self.restoration_net.named_parameters():
            if param.grad is not None:
                self.writer.add_histogram(f'gradients/restoration/{name}', param.grad, epoch)
        
        # 记录像差网络梯度
        if self.aberration_net is not None:
            for name, param in self.aberration_net.named_parameters():
                if param.grad is not None:
                    self.writer.add_histogram(f'gradients/aberration/{name}', param.grad, epoch)
    
    def log_images_to_tensorboard(self, blur_img, sharp_img, restored_img, reblur_img, epoch: int):
        """记录图像到 TensorBoard"""
        if self.writer is None:
            return
        
        # 只取第一张图
        self.writer.add_image('images/blur', blur_img[0].clamp(0, 1), epoch)
        self.writer.add_image('images/sharp_gt', sharp_img[0].clamp(0, 1), epoch)
        self.writer.add_image('images/restored', restored_img[0].clamp(0, 1), epoch)
        if reblur_img is not None:
            self.writer.add_image('images/reblur', reblur_img[0].clamp(0, 1), epoch)
    
    def close_tensorboard(self):
        """关闭 TensorBoard writer"""
        if self.writer is not None:
            self.writer.close()

    def _get_stage_weights(self, stage: str):
        """根据阶段返回动态 Loss 权重"""
        # 默认权重
        weights = {
            'w_data': 1.0,
            'w_sup': 0.0,
            'w_smooth': 0.0,
            'w_coeff': 0.0,
            'w_img_reg': 0.0
        }

        # 优先从 stage_weights 配置中读取
        # stage_weights 应该是一个字典，包含 'physics_only', 'joint' 等键
        if stage in self.stage_weights:
            custom_weights = self.stage_weights[stage]
            weights.update(custom_weights)
            return weights

        # Fallback 到旧逻辑 (如果配置中为空)
        if stage == 'physics_only':
            weights.update({
                'w_data': 1.0, 
                'w_sup': 0.0, 
                'w_smooth': self.lambda_smooth, 
                'w_coeff': self.lambda_coeff, 
                'w_img_reg': 0.0
            })
        elif stage == 'restoration_fixed_physics':
            weights.update({
                'w_data': 0.1, 
                'w_sup': 1.0, 
                'w_smooth': 0.0, 
                'w_coeff': 0.0, 
                'w_img_reg': self.lambda_image_reg
            })
        elif stage == 'joint':
            weights.update({
                'w_data': 0.5, 
                'w_sup': 1.0, 
                'w_smooth': self.lambda_smooth * 0.5, 
                'w_coeff': self.lambda_coeff * 0.2, 
                'w_img_reg': self.lambda_image_reg * 0.1
            })
        elif stage == 'restoration_only':
            weights.update({
                'w_data': 0.0,
                'w_sup': 1.0,
                'w_smooth': 0.0,
                'w_coeff': 0.0,
                'w_img_reg': self.lambda_image_reg
            })

        return weights

    def _set_trainable(self, stage: str):
        """根据阶段快速冻结/解冻网络，并切换 train/eval 模式"""
        if stage == 'physics_only':
            for p in self.restoration_net.parameters():
                p.requires_grad = False
            if self.aberration_net is not None:
                for p in self.aberration_net.parameters():
                    p.requires_grad = True
            self.restoration_net.eval()
            if self.physical_layer is not None:
                self.physical_layer.train()
        elif stage == 'restoration_fixed_physics':
            for p in self.restoration_net.parameters():
                p.requires_grad = True
            if self.aberration_net is not None:
                for p in self.aberration_net.parameters():
                    p.requires_grad = False
            self.restoration_net.train()
            if self.physical_layer is not None:
                self.physical_layer.eval()
        elif stage == 'joint':
            for p in self.restoration_net.parameters():
                p.requires_grad = True
            if self.aberration_net is not None:
                for p in self.aberration_net.parameters():
                    p.requires_grad = True
            self.restoration_net.train()
            if self.physical_layer is not None:
                self.physical_layer.train()
        elif stage == 'restoration_only':
            for p in self.restoration_net.parameters():
                p.requires_grad = True
            self.restoration_net.train()
        else:
            raise ValueError(f"Invalid stage '{stage}'. Must be one of {self.VALID_STAGES}")

    def set_stage(self, stage: str):
        """兼容旧流程的手动设置（仍可用）"""
        if stage not in self.VALID_STAGES:
            raise ValueError(f"Invalid stage '{stage}'. Must be one of {self.VALID_STAGES}")
        self._current_stage = stage
        self._set_trainable(stage)

    def get_stage(self, epoch: int) -> str:
        return self._resolve_stage(epoch)

    def set_forced_stage(self, stage: Optional[str]):
        if stage is None:
            self._forced_stage = None
            return
        if stage not in self.VALID_STAGES:
            raise ValueError(f"Invalid stage '{stage}'. Must be one of {self.VALID_STAGES}")
        self._forced_stage = stage

    def _resolve_stage(self, epoch: int) -> str:
        if self._forced_stage is not None:
            return self._forced_stage
        return self._get_stage(epoch)

    def get_stage_weights(self, epoch: int):
        return self._get_stage_weights(self._get_stage(epoch))

    # =========================================================================
    #                              核心训练步骤
    # =========================================================================
    def train_step(self, Y_blur, X_gt, epoch, crop_info=None):
        """
        执行一个训练步骤，内部根据 epoch 自动切换阶段并分配动态 Loss 权重。
        """
        current_stage = self._resolve_stage(epoch)
        
        # 检测阶段切换
        if self._previous_stage is not None and self._previous_stage != current_stage:
            print(f"\n[Stage Transition] {self._previous_stage} -> {current_stage}")
            
            # Stage 3 学习率减半
            if current_stage == 'joint' and not self._stage3_lr_halved:
                self._adjust_learning_rate_for_stage3()
        
        self._previous_stage = current_stage
        self._current_stage = current_stage
        self._set_trainable(current_stage)

        weights = self._get_stage_weights(current_stage)
        w_data = weights['w_data']
        w_sup = weights['w_sup']
        w_smooth = weights['w_smooth']
        w_coeff = weights['w_coeff']
        w_img_reg = weights['w_img_reg']

        Y_blur = Y_blur.to(self.device)
        X_gt = X_gt.to(self.device)
        if crop_info is not None:
            crop_info = crop_info.to(self.device)

        # 梯度累积：仅在第一个累积步骤清除梯度
        if self.accumulation_counter == 0:
            if not self.use_physical_layer:
                self.optimizer_W.zero_grad()
            elif current_stage == 'physics_only':
                if self.optimizer_Theta is not None:
                    self.optimizer_Theta.zero_grad()
            elif current_stage == 'restoration_fixed_physics':
                self.optimizer_W.zero_grad()
            else:
                self.optimizer_W.zero_grad()
                if self.optimizer_Theta is not None:
                    self.optimizer_Theta.zero_grad()

        # ========================== Forward & Loss ===========================
        loss_data = torch.tensor(0.0, device=self.device)
        loss_sup = torch.tensor(0.0, device=self.device)
        loss_coeff = torch.tensor(0.0, device=self.device)
        loss_smooth = torch.tensor(0.0, device=self.device)
        loss_image_reg = torch.tensor(0.0, device=self.device)

        if not self.use_physical_layer:
            X_hat = self.restoration_net(Y_blur)
            loss_sup = self.criterion_l1(X_hat, X_gt)
            if w_img_reg > 0:
                loss_image_reg = self.compute_image_tv_loss(X_hat)
        elif current_stage == 'physics_only':
            if self.physical_layer is None or self.aberration_net is None:
                raise RuntimeError("physical_layer and aberration_net are required for physics_only stage")
            Y_reblur = self.physical_layer(X_gt, crop_info=crop_info)
            loss_data = self.criterion_mse(Y_reblur, Y_blur)

            if w_coeff > 0 or w_smooth > 0:
                coords = self.physical_layer.get_patch_centers(
                    Y_blur.shape[2], Y_blur.shape[3], self.device
                )
                if coords.shape[0] > 64:
                    indices = torch.randperm(coords.shape[0])[:64]
                    coords = coords[indices]
                coeffs = self.aberration_net(coords)
                if w_coeff > 0:
                    loss_coeff = torch.mean(coeffs**2)
                if w_smooth > 0:
                    loss_smooth = self.physical_layer.compute_coefficient_smoothness(self.smoothness_grid_size)
        else:
            if self.physical_layer is None or self.aberration_net is None:
                raise RuntimeError("physical_layer and aberration_net are required for this stage")
            X_hat = self.restoration_net(Y_blur)
            Y_reblur = self.physical_layer(X_hat, crop_info=crop_info)
            loss_data = self.criterion_mse(Y_reblur, Y_blur)
            loss_sup = self.criterion_l1(X_hat, X_gt)

            if w_img_reg > 0:
                loss_image_reg = self.compute_image_tv_loss(X_hat)

            if current_stage == 'joint' and (w_coeff > 0 or w_smooth > 0):
                coords = self.physical_layer.get_patch_centers(
                    Y_blur.shape[2], Y_blur.shape[3], self.device
                )
                if coords.shape[0] > 64:
                    indices = torch.randperm(coords.shape[0])[:64]
                    coords = coords[indices]
                coeffs = self.aberration_net(coords)
                if w_coeff > 0:
                    loss_coeff = torch.mean(coeffs**2)
                if w_smooth > 0:
                    loss_smooth = self.physical_layer.compute_coefficient_smoothness(self.smoothness_grid_size)

        # ========================== Weighted Loss ============================
        loss_data_w = w_data * loss_data
        loss_sup_w = w_sup * loss_sup
        loss_coeff_w = w_coeff * loss_coeff
        loss_smooth_w = w_smooth * loss_smooth
        loss_image_reg_w = w_img_reg * loss_image_reg

        total_loss = loss_data_w + loss_sup_w + loss_coeff_w + loss_smooth_w + loss_image_reg_w

        scaled_loss = total_loss / self.accumulation_steps
        scaled_loss.backward()

        # ========================== Optimizer Step ============================
        self.accumulation_counter += 1
        should_step = (self.accumulation_counter >= self.accumulation_steps)

        gn_W = torch.tensor(0.0, device=self.device)
        gn_Theta = torch.tensor(0.0, device=self.device)

        if should_step:
            if not self.use_physical_layer:
                gn_W = nn.utils.clip_grad_norm_(self.restoration_net.parameters(), self.grad_clip_restoration)
                self.optimizer_W.step()
            elif current_stage == 'physics_only':
                if self.aberration_net is None:
                    raise RuntimeError("aberration_net is required for physics_only stage")
                gn_Theta = nn.utils.clip_grad_norm_(self.aberration_net.parameters(), self.grad_clip_optics)
                if self.optimizer_Theta is not None:
                    self.optimizer_Theta.step()
            elif current_stage == 'restoration_fixed_physics':
                gn_W = nn.utils.clip_grad_norm_(self.restoration_net.parameters(), self.grad_clip_restoration)
                self.optimizer_W.step()
            else:
                if self.aberration_net is None:
                    raise RuntimeError("aberration_net is required for joint stage")
                gn_W = nn.utils.clip_grad_norm_(self.restoration_net.parameters(), self.grad_clip_restoration)
                gn_Theta = nn.utils.clip_grad_norm_(self.aberration_net.parameters(), self.grad_clip_optics)
                self.optimizer_W.step()
                if self.optimizer_Theta is not None:
                    self.optimizer_Theta.step()

            self.accumulation_counter = 0

            self.history['loss_total'].append(total_loss.item())
            self.history['loss_data'].append(loss_data_w.item())
            self.history['grad_norm_W'].append(gn_W.item() if isinstance(gn_W, torch.Tensor) else gn_W)
            self.history['grad_norm_Theta'].append(gn_Theta.item() if isinstance(gn_Theta, torch.Tensor) else gn_Theta)

        return {
            'loss': total_loss.item(),
            'loss_data': loss_data_w.item(),
            'loss_sup': loss_sup_w.item(),
            'loss_coeff': loss_coeff_w.item(),
            'loss_smooth': loss_smooth_w.item(),
            'loss_image_reg': loss_image_reg_w.item(),
            'loss_data_raw': loss_data.item(),
            'loss_sup_raw': loss_sup.item(),
            'loss_coeff_raw': loss_coeff.item(),
            'loss_smooth_raw': loss_smooth.item(),
            'loss_image_reg_raw': loss_image_reg.item(),
            'grad_W': gn_W.item() if isinstance(gn_W, torch.Tensor) else gn_W,
            'grad_Theta': gn_Theta.item() if isinstance(gn_Theta, torch.Tensor) else gn_Theta,
            'stage': current_stage
        }

    def compute_image_tv_loss(self, img):
        """
        Compute Total Variation (TV) loss on the image.
        L_tv = mean(|dI/dx| + |dI/dy|)
        """
        B, C, H, W = img.shape
        dy = torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :]).mean()
        dx = torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1]).mean()
        return dy + dx

    def save_checkpoint(self, path, epoch=None, stage=None, val_metrics=None):
        """
        保存模型检查点
        
        Args:
            path: 保存路径
            epoch: 当前 epoch (可选)
            stage: 当前训练阶段 (可选)
            val_metrics: 验证指标 (可选)
        """
        checkpoint = {
            'restoration_net': self.restoration_net.state_dict(),
            'optimizer_W': self.optimizer_W.state_dict(),
            'best_metrics': self.best_metrics,
        }
        if self.aberration_net is not None:
            checkpoint['aberration_net'] = self.aberration_net.state_dict()
        if self.optimizer_Theta is not None:
            checkpoint['optimizer_Theta'] = self.optimizer_Theta.state_dict()
        
        if epoch is not None:
            checkpoint['epoch'] = epoch
        if stage is not None:
            checkpoint['stage'] = stage
        if val_metrics is not None:
            checkpoint['val_metrics'] = val_metrics
            
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path, load_optimizer=True):
        """
        加载模型检查点
        
        Args:
            path: 检查点路径
            load_optimizer: 是否加载优化器状态
        
        Returns:
            dict: 检查点中的额外信息 (epoch, stage, val_metrics 等)
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        self.restoration_net.load_state_dict(checkpoint['restoration_net'])
        if self.aberration_net is not None and 'aberration_net' in checkpoint:
            self.aberration_net.load_state_dict(checkpoint['aberration_net'])
        
        if load_optimizer:
            if 'optimizer_W' in checkpoint:
                self.optimizer_W.load_state_dict(checkpoint['optimizer_W'])
            if self.optimizer_Theta is not None and 'optimizer_Theta' in checkpoint:
                self.optimizer_Theta.load_state_dict(checkpoint['optimizer_Theta'])
        
        if 'best_metrics' in checkpoint:
            self.best_metrics = checkpoint['best_metrics']
        
        return {
            'epoch': checkpoint.get('epoch'),
            'stage': checkpoint.get('stage'),
            'val_metrics': checkpoint.get('val_metrics')
        }
    
    def get_current_lr(self) -> Dict[str, float]:
        """获取当前学习率"""
        lr_optics = float('nan')
        if self.optimizer_Theta is not None:
            lr_optics = self.optimizer_Theta.param_groups[0]['lr']
        return {
            'lr_restoration': self.optimizer_W.param_groups[0]['lr'],
            'lr_optics': lr_optics
        }
