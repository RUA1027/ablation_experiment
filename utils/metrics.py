import math
import time
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PerformanceEvaluator:
    """
    多维度评估器：图像质量 + 物理一致性 + 计算效率
    
    支持:
    - 图像质量指标: PSNR, SSIM, LPIPS
    - 物理一致性指标: Re-blur MSE, PSF Smoothness
    - 计算效率指标: Parameters, FLOPs, Inference Time
    - 阶段特定评估
    """

    def __init__(self, device: str = "cuda", ssim_window: int = 11, ssim_sigma: float = 1.5):
        self.device = device
        self.ssim_window = ssim_window
        self.ssim_sigma = ssim_sigma

        self._lpips = None
        self._lpips_available = False
        try:
            import lpips  # type: ignore
            self._lpips = lpips.LPIPS(net="alex").to(device)
            self._lpips_available = True
        except Exception:
            self._lpips = None
            self._lpips_available = False

    @staticmethod
    def _psnr(x: torch.Tensor, y: torch.Tensor, max_val: float = 1.0, eps: float = 1e-8) -> torch.Tensor:
        mse = F.mse_loss(x, y, reduction="mean")
        psnr = 10.0 * torch.log10((max_val ** 2) / (mse + eps))
        return psnr

    @staticmethod
    def _gaussian_window(window_size: int, sigma: float, channels: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        window_1d = g.unsqueeze(1)
        window_2d = window_1d @ window_1d.t()
        window_2d = window_2d / window_2d.sum()
        window = window_2d.expand(channels, 1, window_size, window_size).contiguous()
        return window

    def _ssim(self, x: torch.Tensor, y: torch.Tensor, max_val: float = 1.0) -> torch.Tensor:
        b, c, _, _ = x.shape
        window = self._gaussian_window(self.ssim_window, self.ssim_sigma, c, x.device, x.dtype)
        mu_x = F.conv2d(x, window, padding=self.ssim_window // 2, groups=c)
        mu_y = F.conv2d(y, window, padding=self.ssim_window // 2, groups=c)

        mu_x2 = mu_x.pow(2)
        mu_y2 = mu_y.pow(2)
        mu_xy = mu_x * mu_y

        sigma_x = F.conv2d(x * x, window, padding=self.ssim_window // 2, groups=c) - mu_x2
        sigma_y = F.conv2d(y * y, window, padding=self.ssim_window // 2, groups=c) - mu_y2
        sigma_xy = F.conv2d(x * y, window, padding=self.ssim_window // 2, groups=c) - mu_xy

        c1 = (0.01 * max_val) ** 2
        c2 = (0.03 * max_val) ** 2

        ssim_map = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / ((mu_x2 + mu_y2 + c1) * (sigma_x + sigma_y + c2))
        return ssim_map.mean()

    def _lpips_score(self, x: torch.Tensor, y: torch.Tensor) -> Optional[torch.Tensor]:
        if not self._lpips_available or self._lpips is None:
            return None
        # LPIPS expects input in [-1, 1]
        x_norm = (x * 2.0 - 1.0).clamp(-1, 1)
        y_norm = (y * 2.0 - 1.0).clamp(-1, 1)
        return self._lpips(x_norm, y_norm).mean()

    @staticmethod
    def _count_parameters(*models: nn.Module) -> float:
        total = 0
        for model in models:
            total += sum(p.numel() for p in model.parameters())
        return total / 1e6  # Million

    @staticmethod
    def _try_flops(model: nn.Module, device: str, input_shape=(1, 3, 1024, 1024)) -> Optional[float]:
        try:
            from thop import profile  # type: ignore
            dummy = torch.randn(*input_shape, device=device)
            macs, _ = profile(model, inputs=(dummy,), verbose=False)
            return macs / 1e9  # GMACs
        except Exception:
            return None

    @staticmethod
    def _measure_inference_time(model: nn.Module, device: str, input_shape=(1, 3, 1024, 1024),
                                warmup: int = 5, repeat: int = 20) -> float:
        model.eval()
        dummy = torch.randn(*input_shape, device=device)
        with torch.no_grad():
            for _ in range(warmup):
                _ = model(dummy)

            if device.startswith("cuda") and torch.cuda.is_available():
                starter = torch.cuda.Event(enable_timing=True)
                ender = torch.cuda.Event(enable_timing=True)
                torch.cuda.synchronize()
                starter.record(stream=torch.cuda.current_stream())
                for _ in range(repeat):
                    _ = model(dummy)
                ender.record(stream=torch.cuda.current_stream())
                torch.cuda.synchronize()
                elapsed = starter.elapsed_time(ender)  # ms
                return elapsed / repeat
            else:
                start = time.perf_counter()
                for _ in range(repeat):
                    _ = model(dummy)
                end = time.perf_counter()
                return (end - start) * 1000.0 / repeat

    def evaluate(self, restoration_net: nn.Module, physical_layer: Optional[nn.Module], val_loader, device: str,
                 smoothness_grid_size: int = 16) -> Dict[str, float]:
        restoration_net.eval()
        use_physical_layer = physical_layer is not None
        if use_physical_layer:
            physical_layer.eval()

        psnr_total = 0.0
        ssim_total = 0.0
        lpips_total = 0.0
        reblur_total = 0.0
        n = 0
        lpips_count = 0

        with torch.no_grad():
            for batch in val_loader:
                if isinstance(batch, dict):
                    blur = batch["blur"].to(device)
                    sharp = batch["sharp"].to(device)
                    crop_info = batch.get("crop_info", None)
                    if crop_info is not None:
                        crop_info = crop_info.to(device)
                else:
                    blur, sharp = batch
                    blur = blur.to(device)
                    sharp = sharp.to(device)
                    crop_info = None

                x_hat = restoration_net(blur)

                if x_hat.shape[1] == 1 and sharp.shape[1] == 3:
                    x_hat = x_hat.repeat(1, 3, 1, 1)
                if sharp.shape[1] == 1 and x_hat.shape[1] == 3:
                    sharp = sharp.repeat(1, 3, 1, 1)

                psnr_total += self._psnr(x_hat, sharp).item()
                ssim_total += self._ssim(x_hat, sharp).item()

                lp = self._lpips_score(x_hat, sharp)
                if lp is not None:
                    lpips_total += lp.item()
                    lpips_count += 1

                if use_physical_layer:
                    y_reblur = physical_layer(x_hat, crop_info=crop_info)
                    reblur_total += F.mse_loss(y_reblur, blur).item()

                n += 1

        smoothness = float("nan")
        if use_physical_layer and hasattr(physical_layer, "compute_coefficient_smoothness"):
            with torch.no_grad():
                smoothness = physical_layer.compute_coefficient_smoothness(smoothness_grid_size).item()

        if use_physical_layer:
            params_m = self._count_parameters(restoration_net, physical_layer)
        else:
            params_m = self._count_parameters(restoration_net)
        flops_gmacs = self._try_flops(restoration_net, device)
        infer_ms = self._measure_inference_time(restoration_net, device)

        metrics = {
            "PSNR": psnr_total / max(n, 1),
            "SSIM": ssim_total / max(n, 1),
            "LPIPS": (lpips_total / lpips_count) if lpips_count > 0 else float("nan"),
            "Reblur_MSE": (reblur_total / max(n, 1)) if use_physical_layer else float("nan"),
            "PSF_Smoothness": smoothness,
            "Params(M)": params_m,
            "FLOPs(GMACs)": flops_gmacs if flops_gmacs is not None else float("nan"),
            "Inference(ms)": infer_ms
        }
        return metrics

    @staticmethod
    def evaluate_model(restoration_net: nn.Module, physical_layer: Optional[nn.Module], val_loader, device: str,
                       smoothness_grid_size: int = 16) -> Dict[str, float]:
        evaluator = PerformanceEvaluator(device=device)
        return evaluator.evaluate(restoration_net, physical_layer, val_loader, device, smoothness_grid_size)

    @staticmethod
    def evaluate_stage1(physical_layer: nn.Module, val_loader, device: str,
                        smoothness_grid_size: int = 16) -> Dict[str, float]:
        """
        Stage 1 专用评估: 只评估物理层的重模糊一致性
        
        Args:
            physical_layer: 物理层
            val_loader: 验证集 DataLoader
            device: 计算设备
            smoothness_grid_size: 平滑性计算网格大小
        
        Returns:
            dict: 包含 Reblur_MSE 和 PSF_Smoothness
        """
        physical_layer.eval()
        
        reblur_total = 0.0
        n = 0
        
        with torch.no_grad():
            for batch in val_loader:
                if isinstance(batch, dict):
                    blur = batch["blur"].to(device)
                    sharp = batch["sharp"].to(device)
                    crop_info = batch.get("crop_info", None)
                    if crop_info is not None:
                        crop_info = crop_info.to(device)
                else:
                    blur, sharp = batch
                    blur = blur.to(device)
                    sharp = sharp.to(device)
                    crop_info = None
                
                # Stage 1: 用清晰图重模糊，与真实模糊图比较
                y_reblur = physical_layer(sharp, crop_info=crop_info)
                reblur_total += F.mse_loss(y_reblur, blur).item()
                n += 1
        
        # 计算 PSF 平滑性
        smoothness = float("nan")
        if hasattr(physical_layer, "compute_coefficient_smoothness"):
            with torch.no_grad():
                smoothness = physical_layer.compute_coefficient_smoothness(smoothness_grid_size).item()
        
        return {
            "Reblur_MSE": reblur_total / max(n, 1),
            "PSF_Smoothness": smoothness
        }

    def evaluate_full_resolution(self, restoration_net: nn.Module, physical_layer: Optional[nn.Module], 
                                  test_loader, device: str) -> Tuple[Dict[str, float], list]:
        """
        全分辨率测试集评估（用于论文最终结果）
        
        Args:
            restoration_net: 复原网络
            physical_layer: 物理层
            test_loader: 测试集 DataLoader
            device: 计算设备
        
        Returns:
            tuple: (平均指标字典, 每张图像的详细结果列表)
        """
        restoration_net.eval()
        use_physical_layer = physical_layer is not None
        if use_physical_layer:
            physical_layer.eval()
        
        results = []
        
        psnr_total = 0.0
        ssim_total = 0.0
        lpips_total = 0.0
        reblur_total = 0.0
        n = 0
        lpips_count = 0
        
        with torch.no_grad():
            for batch in test_loader:
                if isinstance(batch, dict):
                    blur = batch["blur"].to(device)
                    sharp = batch["sharp"].to(device)
                    crop_info = batch.get("crop_info", None)
                    filename = batch.get("filename", [f"image_{n}"])[0]
                    if crop_info is not None:
                        crop_info = crop_info.to(device)
                else:
                    blur, sharp = batch
                    blur = blur.to(device)
                    sharp = sharp.to(device)
                    crop_info = None
                    filename = f"image_{n}"
                
                # 复原
                x_hat = restoration_net(blur)
                
                # 计算指标
                psnr = self._psnr(x_hat, sharp).item()
                ssim = self._ssim(x_hat, sharp).item()
                
                lp = self._lpips_score(x_hat, sharp)
                lpips_val = lp.item() if lp is not None else float("nan")
                
                # 重模糊误差
                if use_physical_layer:
                    y_reblur = physical_layer(x_hat, crop_info=crop_info)
                    reblur_mse = F.mse_loss(y_reblur, blur).item()
                else:
                    reblur_mse = float("nan")
                
                # 记录单张图像结果
                results.append({
                    "filename": filename,
                    "PSNR": psnr,
                    "SSIM": ssim,
                    "LPIPS": lpips_val,
                    "Reblur_MSE": reblur_mse
                })
                
                psnr_total += psnr
                ssim_total += ssim
                if not math.isnan(lpips_val):
                    lpips_total += lpips_val
                    lpips_count += 1
                if use_physical_layer:
                    reblur_total += reblur_mse
                n += 1
        
        avg_metrics = {
            "PSNR": psnr_total / max(n, 1),
            "SSIM": ssim_total / max(n, 1),
            "LPIPS": (lpips_total / lpips_count) if lpips_count > 0 else float("nan"),
            "Reblur_MSE": (reblur_total / max(n, 1)) if use_physical_layer else float("nan"),
            "Num_Images": n
        }
        
        return avg_metrics, results
