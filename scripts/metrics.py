import torch
import numpy as np
import torch.nn.functional as F
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

def calculate_mse(pred, target):
    return ((pred - target) ** 2).mean().item()

def calculate_rmse(pred, target):
    return torch.sqrt(((pred - target) ** 2).mean()).item()

def calculate_psnr(pred, target):
    pred_np = pred.squeeze().cpu().numpy()
    target_np = target.squeeze().cpu().numpy()
    if pred_np.ndim == 3:
        pred_np = pred_np[0]
    if target_np.ndim == 3:
        target_np = target_np[0]
    return psnr(target_np, pred_np, data_range=1.0)

def calculate_ssim(pred, target):
    pred_np = pred.squeeze().cpu().numpy()
    target_np = target.squeeze().cpu().numpy()
    if pred_np.ndim == 3:
        pred_np = pred_np[0]
    if target_np.ndim == 3:
        target_np = target_np[0]
    return ssim(target_np, pred_np, data_range=1.0)
