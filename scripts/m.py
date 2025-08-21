import torch.nn.functional as F
import numpy as np
from skimage.metrics import structural_similarity as ssim
import pandas as pd
import os


def calculate_psnr(pred, target, max_pixel=2.0):
    mse_val = F.mse_loss(pred, target).item()
    if mse_val == 0:
        return float('inf')
    return 20 * np.log10(max_pixel / np.sqrt(mse_val))

def calculate_rmse(pred, target):
    mse_val = F.mse_loss(pred, target).item()
    return np.sqrt(mse_val)

def calculate_ssim(pred, target):
    # pred and target should be numpy arrays, scaled 0-255, uint8
    pred_np = pred.cpu().numpy().squeeze()
    target_np = target.cpu().numpy().squeeze()
    return ssim(target_np, pred_np, data_range=2.0)
    

