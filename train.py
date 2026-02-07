"""
DPDD 物理驱动复原网络 - 训练脚本
================================
使用 utils.model_builder 统一构建组件，支持真实 DPDD 数据集训练。

特性:
- 三阶段训练: 由 training.stage_schedule 定义
- 虚拟长度机制: 每张图在一个 Epoch 内被随机裁剪多次
- TensorBoard 可视化
- 熔断机制: 阶段切换前验证模型质量
- 定期存档: 每 20 个 Epoch 强制保存
- 阶段最佳模型: 各阶段独立保存最佳模型

Usage:
    python train.py --config config/default.yaml
"""

import argparse
import os
import torch
import sys
import math
from typing import Sized, cast, Any, Dict
from tqdm import tqdm
from datetime import datetime
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import load_config
from utils.model_builder import build_models_from_config, build_trainer_from_config, build_dataloader_from_config
from utils.metrics import PerformanceEvaluator


def print_stage_info(stage: str, epoch: int, total_epochs: int):
    """打印阶段信息"""
    stage_names = {
        'physics_only': 'Stage 1: Physics Only (物理层训练)',
        'restoration_fixed_physics': 'Stage 2: Restoration with Fixed Physics (复原网络训练)',
        'joint': 'Stage 3: Joint Fine-tuning (联合微调, 学习率减半)',
        'restoration_only': 'Restoration Only (无物理层)'
    }
    stage_name = stage_names.get(stage, stage)
    print(f"\n{'='*60}")
    print(f"Epoch {epoch}/{total_epochs} - {stage_name}")
    print(f"{'='*60}")


def main():
    # 1. 解析参数
    parser = argparse.ArgumentParser(description='DPDD Training Script')
    parser.add_argument('--config', '-c', type=str, default='config/default.yaml', help='Path to config file')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    args = parser.parse_args()

    # 启用 CUDNN Benchmark 以加速固定尺寸输入的训练
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    # 2. 加载配置
    print(f"Loading config from {args.config}...")
    config = load_config(args.config)
    
    # 3. 设置环境
    device = config.experiment.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA not available, using CPU")
        device = 'cpu'
    config.experiment.device = device
    
    # 设置随机种子
    torch.manual_seed(config.experiment.seed)
    if device == 'cuda':
        torch.cuda.manual_seed(config.experiment.seed)

    print(f"Device: {device}")
    print(f"Seed: {config.experiment.seed}")
    
    use_physical_layer = getattr(config.experiment, 'use_physical_layer', True)
    s1 = 0
    s2 = 0
    s3 = 0
    if use_physical_layer:
        s1 = config.training.stage_schedule.stage1_epochs
        s2 = config.training.stage_schedule.stage2_epochs
        s3 = config.training.stage_schedule.stage3_epochs
        schedule_total = s1 + s2 + s3
        if config.experiment.epochs != schedule_total:
            raise ValueError(
                "config.experiment.epochs must match the sum of training.stage_schedule "
                f"(epochs={config.experiment.epochs}, schedule_total={schedule_total})."
            )

    # 创建输出目录
    base_output_dir = Path(config.experiment.output_dir)
    base_output_dir.mkdir(parents=True, exist_ok=True)

    run_name = config.experiment.run_name or config.experiment.name
    if config.experiment.use_timestamp:
        run_name = f"{run_name}_{datetime.now().strftime(config.experiment.timestamp_format)}"

    output_dir = str(base_output_dir / run_name)
    os.makedirs(output_dir, exist_ok=True)
    checkpoints_subdir = config.experiment.checkpoints_subdir or "checkpoints"
    checkpoints_dir = os.path.join(output_dir, checkpoints_subdir)
    os.makedirs(checkpoints_dir, exist_ok=True)

    print(f"Output directory: {output_dir}")
    print(f"Checkpoints will be saved to: {checkpoints_dir}")

    # 将本次实验输出目录写回配置，确保其他模块使用一致路径
    config.experiment.output_dir = output_dir

    # TensorBoard 日志目录
    tb_log_dir = None
    if getattr(config.experiment.tensorboard, 'enabled', False):
        base_tb_dir = config.experiment.tensorboard.log_dir
        if not os.path.isabs(base_tb_dir):
            base_tb_dir = os.path.join(output_dir, base_tb_dir)
        if config.experiment.tensorboard.append_run_name:
            tb_log_dir = os.path.join(base_tb_dir, run_name)
        else:
            tb_log_dir = base_tb_dir
        Path(tb_log_dir).mkdir(parents=True, exist_ok=True)
        print(f"TensorBoard logs: {tb_log_dir}")

    # 4. 构建数据
    print("\n" + "="*60)
    print("Initializing DataLoaders...")
    print("="*60)
    try:
        train_loader = build_dataloader_from_config(config, mode='train')
        val_loader = build_dataloader_from_config(config, mode='val')
        train_dataset = train_loader.dataset
        val_dataset = val_loader.dataset
        
        # 获取数据集信息
        train_size = len(cast(Sized, train_dataset)) if hasattr(train_dataset, "__len__") else "Unknown"
        val_size = len(cast(Sized, val_dataset)) if hasattr(val_dataset, "__len__") else "Unknown"
        
        # 获取真实样本数
        real_train_size = cast(Any, train_dataset).get_real_length() if hasattr(train_dataset, 'get_real_length') else train_size
        
        print(f"✓ Train set: {real_train_size} real images, virtual length: {train_size}")
        print(f"✓ Val set size: {val_size}")
        print(f"✓ Batch size: {config.data.batch_size}")
        print(f"✓ Crop size (train): {config.data.crop_size}")
        print(f"✓ Center crop size (val): {getattr(config.data, 'val_crop_size', 1024)}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure the dataset exists at the specified path.")
        return

    # 5. 构建模型与训练器
    print("\n" + "="*60)
    print("Building Models...")
    print("="*60)
    zernike_gen, aberration_net, restoration_net, physical_layer = \
        build_models_from_config(config, device)
    
    print("\nInitializing Trainer...")
    trainer = build_trainer_from_config(
        config,
        restoration_net,
        physical_layer,
        device,
        tensorboard_dir=tb_log_dir
    )
    
    # 打印训练计划
    print(f"\n训练计划:")
    if use_physical_layer:
        print(f"  Stage 1 (Physics Only):        Epochs 1-{s1}")
        print(f"  Stage 2 (Restoration):         Epochs {s1+1}-{s1+s2}")
        print(f"  Stage 3 (Joint, LR halved):    Epochs {s1+s2+1}-{s1+s2+s3}")
        print(f"  Total: {s1+s2+s3} epochs")
    else:
        print(f"  Restoration Only:              Epochs 1-{config.experiment.epochs}")
    
    # 6. 恢复训练
    start_epoch = 0
    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        resume_info = trainer.load_checkpoint(args.resume)
        if resume_info.get('epoch') is not None:
            start_epoch = resume_info['epoch']
            print(f"  Resumed at epoch {start_epoch}")

    # 7. 训练循环
    print("\n" + "="*60)
    print("Start Training")
    print("="*60)
    
    epochs = config.experiment.epochs
    save_interval = config.experiment.save_interval
    prev_stage = None
    
    # 初始化变量以防止静态分析错误
    stage = 'physics_only' if use_physical_layer else 'restoration_only'
    val_metrics: Dict[str, Any] = {}

    # 创建评估器
    evaluator = PerformanceEvaluator(device=device)

    for epoch in range(start_epoch, epochs):
        current_epoch = epoch + 1
        
        # --- Stage Scheduling (自动基于 epoch) ---
        stage = trainer.get_stage(epoch)
        
        # 阶段切换时打印信息
        if stage != prev_stage:
            print_stage_info(stage, current_epoch, epochs)
            
            # 熔断机制检查（在阶段切换时）
            if prev_stage is not None and use_physical_layer:
                # 先评估当前验证指标
                if prev_stage == 'physics_only':
                    if trainer.physical_layer is None:
                        raise RuntimeError("physical_layer is required for physics_only evaluation")
                    val_metrics = PerformanceEvaluator.evaluate_stage1(
                        trainer.physical_layer, val_loader, device,
                        config.training.smoothness_grid_size
                    )
                else:
                    val_metrics = evaluator.evaluate(
                        trainer.restoration_net, trainer.physical_layer,
                        val_loader, device, config.training.smoothness_grid_size
                    )
                
                # 检查熔断
                can_switch = trainer.check_circuit_breaker(val_metrics, prev_stage, stage)
                if not can_switch:
                    print(trainer.circuit_breaker_message)
                    # 严格熔断：阻止阶段切换，继续上一阶段训练
                    trainer.set_forced_stage(prev_stage)
                    stage = prev_stage
                else:
                    trainer.set_forced_stage(None)
            
            prev_stage = stage
        else:
            print(f"\nEpoch {current_epoch}/{epochs} [{stage}]")
        
        # --- Training Phase ---
        epoch_loss = 0.0
        epoch_loss_data = 0.0
        epoch_loss_sup = 0.0
        steps = 0
        
        pbar = tqdm(train_loader, desc=f"Train E{current_epoch}")
        
        acc_steps = getattr(trainer, 'accumulation_steps', 1)
        
        for batch_idx, batch in enumerate(pbar):
            # 处理字典格式的数据
            if isinstance(batch, dict):
                blur_imgs = batch['blur']
                sharp_imgs = batch['sharp']
                crop_info = batch.get('crop_info', None)
            else:
                blur_imgs, sharp_imgs = batch
                crop_info = None

            # 训练步骤
            metrics = trainer.train_step(
                Y_blur=blur_imgs,
                X_gt=sharp_imgs,
                epoch=epoch,
                crop_info=crop_info
            )
            
            epoch_loss += metrics['loss']
            epoch_loss_data += metrics['loss_data']
            epoch_loss_sup += metrics['loss_sup']
            steps += 1
            
            # 更新进度条
            if (batch_idx + 1) % acc_steps == 0:
                pbar.set_postfix({
                    'Loss': f"{metrics['loss']:.4f}",
                    'Data': f"{metrics['loss_data']:.4f}",
                    'Sup': f"{metrics['loss_sup']:.4f}",
                    'GradW': f"{metrics.get('grad_W', 0):.2f}"
                })
            
        avg_loss = epoch_loss / max(steps, 1)
        avg_loss_data = epoch_loss_data / max(steps, 1)
        avg_loss_sup = epoch_loss_sup / max(steps, 1)
        
        print(f"  Train Loss: {avg_loss:.6f} (Data: {avg_loss_data:.6f}, Sup: {avg_loss_sup:.6f})")
        
        # --- Validation Phase ---
        print("  Evaluating on validation set...")
        
        if stage == 'physics_only':
            # Stage 1: 专用评估（重模糊一致性）
            if trainer.physical_layer is None:
                raise RuntimeError("physical_layer is required for physics_only evaluation")
            val_metrics = PerformanceEvaluator.evaluate_stage1(
                trainer.physical_layer, val_loader, device,
                config.training.smoothness_grid_size
            )
        else:
            # Stage 2/3: 完整评估
            val_metrics = evaluator.evaluate(
                trainer.restoration_net, trainer.physical_layer,
                val_loader, device, config.training.smoothness_grid_size
            )
        
        # 打印验证指标
        try:
            from tabulate import tabulate  # type: ignore[import-not-found]
            rows = []
            for k, v in val_metrics.items():
                if isinstance(v, float) and math.isnan(v):
                    rows.append([k, "NaN"])
                else:
                    rows.append([k, f"{v:.6f}"] if isinstance(v, float) else [k, str(v)])
            print(tabulate(rows, headers=["Metric", "Value"], tablefmt="github"))
        except Exception:
            for k, v in val_metrics.items():
                if isinstance(v, float) and math.isnan(v):
                    v_str = "NaN"
                else:
                    v_str = f"{v:.6f}" if isinstance(v, float) else str(v)
                print(f"    {k}: {v_str}")

        # --- TensorBoard Logging ---
        train_metrics = {
            'loss': avg_loss,
            'loss_data': avg_loss_data,
            'loss_sup': avg_loss_sup
        }
        trainer.log_to_tensorboard(train_metrics, current_epoch, prefix='train')
        trainer.log_to_tensorboard(val_metrics, current_epoch, prefix='val')
        
        # 记录学习率
        lr_info = trainer.get_current_lr()
        trainer.log_to_tensorboard(lr_info, current_epoch, prefix='lr')

        # --- Best Model Checkpointing (阶段性最佳模型保存) ---
        is_best = trainer.update_best_metrics(val_metrics, stage)
        
        # 保存阶段最佳模型
        if stage == 'physics_only' and is_best.get('reblur_mse', False):
            best_path = os.path.join(checkpoints_dir, "best_stage1_physics.pt")
            trainer.save_checkpoint(best_path, epoch=current_epoch, stage=stage, val_metrics=val_metrics)
            print(f"  ✓ New best Stage 1 model saved: Reblur_MSE={val_metrics.get('Reblur_MSE', 0):.6f}")
            
        elif stage == 'restoration_fixed_physics' and is_best.get('psnr', False):
            best_path = os.path.join(checkpoints_dir, "best_stage2_restoration.pt")
            trainer.save_checkpoint(best_path, epoch=current_epoch, stage=stage, val_metrics=val_metrics)
            print(f"  ✓ New best Stage 2 model saved: PSNR={val_metrics.get('PSNR', 0):.2f}")
            
        elif stage == 'joint' and is_best.get('combined', False):
            best_path = os.path.join(checkpoints_dir, "best_stage3_joint.pt")
            trainer.save_checkpoint(best_path, epoch=current_epoch, stage=stage, val_metrics=val_metrics)
            print(f"  ✓ New best Stage 3 model saved: PSNR={val_metrics.get('PSNR', 0):.2f}")
        elif stage == 'restoration_only' and is_best.get('psnr', False):
            best_path = os.path.join(checkpoints_dir, "best_restoration_only.pt")
            trainer.save_checkpoint(best_path, epoch=current_epoch, stage=stage, val_metrics=val_metrics)
            print(f"  ✓ New best restoration-only model saved: PSNR={val_metrics.get('PSNR', 0):.2f}")

        # --- Periodic Checkpointing (定期存档，每 20 个 epoch) ---
        if current_epoch % save_interval == 0:
            periodic_path = os.path.join(checkpoints_dir, f"checkpoint_epoch{current_epoch:03d}.pt")
            trainer.save_checkpoint(periodic_path, epoch=current_epoch, stage=stage, val_metrics=val_metrics)
            print(f"  ✓ Periodic checkpoint saved: {periodic_path}")

    # 8. 训练结束，保存最终模型
    final_path = os.path.join(checkpoints_dir, "final_model.pt")
    trainer.save_checkpoint(final_path, epoch=epochs, stage=stage, val_metrics=val_metrics)
    print(f"\n✓ Final model saved: {final_path}")
    
    # 关闭 TensorBoard
    trainer.close_tensorboard()
    
    # 打印最终统计
    print("\n" + "="*60)
    print("Training Finished!")
    print("="*60)
    print(f"\nBest metrics achieved:")
    if use_physical_layer:
        print(f"  Stage 1 (Physics): Reblur_MSE = {trainer.best_metrics['physics_only']['reblur_mse']:.6f}")
        print(f"  Stage 2 (Restoration): PSNR = {trainer.best_metrics['restoration_fixed_physics']['psnr']:.2f}")
        print(f"  Stage 3 (Joint): PSNR = {trainer.best_metrics['joint']['psnr']:.2f}")
    else:
        print(f"  Restoration Only: PSNR = {trainer.best_metrics['restoration_only']['psnr']:.2f}")
    print(f"\nOutput directory: {output_dir}")
    if tb_log_dir:
        print(f"Run 'tensorboard --logdir {tb_log_dir}' to view training curves.")
    else:
        print("TensorBoard is disabled.")


if __name__ == "__main__":
    main()
