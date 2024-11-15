from typing import Sequence, List

from mmengine.evaluator import BaseMetric
from mmengine.registry import METRICS
from . import metrics_torch


@METRICS.register_module()
class PixelRestoration(BaseMetric):
    default_prefix = "PixelRestoration"

    def process(self, data_batch, data_samples):
        for i, result in enumerate(data_samples):
            y = result.unsqueeze(0)
            gt = data_batch["gt"][i : i + 1, :].to(result.device)
            self.results.append(
                {"psnr": metrics_torch.PSNR(y, gt), "ssim": metrics_torch.SSIM(y, gt), "mae": metrics_torch.MAE(y, gt), "mse": metrics_torch.MSE(y, gt), "sam": metrics_torch.SAM(y, gt)}
            )

    def compute_metrics(self, results: List):
        total_psnr = sum(item["psnr"] for item in results)
        total_ssim = sum(item["ssim"] for item in results)
        total_mae = sum(item["mae"] for item in results)
        total_mse = sum(item["mse"] for item in results)
        total_sam = sum(item["sam"] for item in results)
        return {"PSNR": total_psnr / len(results), "SSIM": total_ssim / len(results), "MAE": total_mae / len(results), "MSE": total_mse / len(results), "SAM": total_sam / len(results)}
