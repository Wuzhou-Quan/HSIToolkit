import torch
import torch.nn.functional as F
from math import log10, sqrt


def MSE(img1, img2):
    return F.mse_loss(img1, img2).item()


def MAE(img1, img2):
    return F.l1_loss(img1, img2).item()


def PSNR(img1, img2):
    mse_val = MSE(img1, img2)
    if mse_val == 0:
        return float("inf")
    max_pixel = 1.0
    return 20 * log10(max_pixel / sqrt(mse_val))


def SSIM(img1, img2):
    C1 = 0.01**2
    C2 = 0.03**2

    mu1 = F.avg_pool2d(img1, 3, 1, 1)
    mu2 = F.avg_pool2d(img2, 3, 1, 1)

    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.avg_pool2d(img1 * img1, 3, 1, 1) - mu1_sq
    sigma2_sq = F.avg_pool2d(img2 * img2, 3, 1, 1) - mu2_sq
    sigma12 = F.avg_pool2d(img1 * img2, 3, 1, 1) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean().item()


def SAM(img1, img2):
    dot_product = torch.sum(img1 * img2, dim=1)
    norm1 = torch.sqrt(torch.sum(img1**2, dim=1))
    norm2 = torch.sqrt(torch.sum(img2**2, dim=1))
    cos_sim = dot_product / (norm1 * norm2)
    sam_val = torch.mean(torch.acos(torch.clamp(cos_sim, -1, 1)))
    return sam_val.item()
