"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os
import nibabel as nib
# from visdom import Visdom
# viz = Visdom(port=8850)
import sys
import random
sys.path.append(".")
import numpy as np
import pandas as pd
import time
import torch as th
import torch.distributed as dist
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from guided_diffusion import dist_util, logger
from guided_diffusion.bratsloader import BRATSDataset
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
from loader import load_LIDC
from m import *
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim
from PIL import Image
from torchvision.transforms.functional import to_pil_image
import torch.nn.functional as F
from metrics import calculate_mse, calculate_rmse, calculate_psnr, calculate_ssim


seed=10
th.manual_seed(seed)
th.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)


def visualize(img):
    _min = img.min()
    _max = img.max()
    normalized_img = (img - _min) / (_max - _min)
    return normalized_img


def show_tensor_images(image, mask, output, num, title=None):
    to_pil = transforms.ToPILImage()

    pil_image1 = to_pil(image.squeeze().cpu())
    pil_image2 = to_pil(mask.squeeze().cpu())
    pil_image3 = to_pil(output)

    # Plotting
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle(title, fontsize=14)

    axes[0].imshow(pil_image1, aspect = 'equal', cmap = 'gray' )
    axes[0].set_title('Image', fontsize=10)

    axes[1].imshow(pil_image2, aspect = 'equal', cmap = 'gray')
    axes[1].set_title('Ground Truth', fontsize=10)

    axes[2].imshow(pil_image3, aspect = 'equal', cmap = 'gray')
    axes[2].set_title('Output', fontsize=10)

    for ax in axes:
        ax.axis('off')

    plt.savefig('./output_images/Output_'+str(num)+'.png')
    # plt.show()

def main():
    args = create_argparser().parse_args()
    # dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )

    # If multiple GPU is connected
    # model = th.nn.DataParallel(model)

    output_dir = './output'
    output_img_dir = './output_images'
    output_metrics_dir = './output_metrics'
    samples_dir = './samples'


    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(output_img_dir):
        os.makedirs(output_img_dir)
    if not os.path.exists(output_metrics_dir):
        os.makedirs(output_metrics_dir)
    if not os.path.exists(samples_dir):
        os.makedirs(samples_dir)

    ds = load_LIDC(image_size=224, combine_train_val=True, mode='Test')
    datal= th.utils.data.DataLoader(
        ds,
        batch_size=1,
        shuffle=False)
    data = iter(datal)

    all_images = []
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    dp = {'title': [], 'mse': [], 'rmse': [], 'psnr': [], 'ssim': []}
    df = pd.DataFrame(dp)

    # for i in range(30):
    #     b, mask, image_path, mask_path = next(data)

    title = ''
    cnt = 1
    while len(all_images) * args.batch_size < args.num_samples:
        # should return an image from the dataloader "data"
        b, mask, image_path, mask_path = next(data)
        b = b[:, :1, ...]
        c = torch.randn_like(b)
          
        img = torch.cat([b, c], dim = 1)
        #img = torch.cat([b,b], dim=1)
        print(f"input shape: {img.shape}")
        print(image_path)
        # slice_ID = path[0].split("/", -1)[3]
        slice_ID = os.path.basename(image_path[0]).split(".")[0]
        title = os.path.basename(mask_path[0]).split(".")[0]

        logger.log("sampling...")

        start = th.cuda.Event(enable_timing=True)
        end = th.cuda.Event(enable_timing=True)

        tensor_list = []
        # this is for the generation of an ensemble of 5 masks.
        for i in range(args.num_ensemble):
            model_kwargs = {}
            start.record()
            sample_fn = (
                diffusion.p_sample_loop_known if not args.use_ddim else diffusion.ddim_sample_loop_known
            )
            print("Input img shape before sampling:", img.shape)
            sample, x_noisy, org = sample_fn(
                model,
                (args.batch_size, 2, args.image_size, args.image_size),
                img
              
            )
            print("Sample shape:", sample.shape)
            end.record()
            th.cuda.synchronize()
            # time measurement for the generation of 1 sample
            print('time for 1 sample', start.elapsed_time(end))

            s = sample.clone().detach()
            tensor_list.append(s.squeeze().cpu())
            # viz.image(visualize(sample[0, 0, ...]), opts=dict(caption="sampled output"))
        

        best_score = -float('inf')
        index = 0
        best_metrics = (None, None, None)
        for i in range(args.num_ensemble):
            pred = tensor_list[i]
            target = mask.squeeze().cpu()
            mse_val = calculate_mse(pred, target)
            psnr_val = calculate_psnr(pred, target)
            ssim_val = calculate_ssim(pred, target)
            score = -0.3 * mse_val + 0.4 * psnr_val + 0.6 * ssim_val
            if score > best_score:
                best_score = score
                index = i
                best_metrics = (mse_val, psnr_val, ssim_val)
        best_output = tensor_list[index]
        mse_val, psnr_val, ssim_val = best_metrics
        rmse_val = calculate_rmse(best_output, mask.squeeze().cpu())

        #index = 0
        #best_mse = calculate_mse(pred=tensor_list[0], target=mask.squeeze().cpu())
        #for i in range(1, args.num_ensemble):
         #   mse_i = calculate_mse(tensor_list[i], target=mask.squeeze().cpu())
          #  if mse_i < best_mse:
           #     best_mse = mse_i
            #    index = i
        #best_output = tensor_list[index]
        #mse_val = best_mse
        #rmse_val = calculate_rmse(best_output, mask.squeeze().cpu())
        #psnr_val = calculate_psnr(best_output, mask.squeeze().cpu())
        #ssim_val = calculate_ssim(best_output, mask.squeeze().cpu())




        #get_np = mask.squeeze().cpu().numpy()
        #best_score = -float('inf')
        #best_output = None
        #index = 0
        #for i in range(len(tensor_list)):
        #for i, pred_tensor in enumerate(tensor_list):
         #   pred = pred_tensor.squeeze().cpu()
          #  pred_np = pred.numpy()
           # mse = calculate_mse(pred, mask.squeeze().cpu())
            #pred = tensor_list[i].squeeze().cpu()
            #mse_val = F.mse_loss(pred, mask.squeeze().cpu()).item()
            #mse_val = calculate_mse(pred, mask.squeeze().cpu())
       #     pred_np = pred.numpy()
       #     mse_val = F.mse_loss(tensor_list[i], mask.squeeze().cpu())
            #psnr_val = calculate_psnr(pred, mask.squeeze().cpu())
            #ssim_val = calculate_ssim(pred, mask.squeeze().cpu())
            #score = -mse + psnr_val + ssim_val
            #if score > best_score:
             #   best_score = score
              #  best_output = pred
               # index = i
        #mse_val = calculate_mse(best_output, mask.squeeze().cpu())
        #rmse_val = calculate_rmse(best_output, mask.squeeze().cpu())
        #psnr_val = calculate_psnr(best_output, mask.squeeze().cpu())
        #ssim_val = calculate_ssim(best_output, mask.squeeze().cpu())
        #    ssim_val = compute_ssim(get_np, pred_np, data_range=1.0)
         #   combined_score = -mse_val + psnr_val + ssim_val
          #  if combined_score > best_score:
           #     best_score = combined_score
            #    best_output = pred
        #pred = tensor_list[0].squeeze().cpu()
        #pred_np = pred.numpy()
        #mse_val = F.mse_loss(tensor_list[0], mask.squeeze().cpu())
        #psnr_val = compute_psnr(get_np, pred_np, data_range = 1.0)
        #ssim_val = compute_ssim(get_np, pred_np, data_range = 1.0)
        #best_score = -mse_val + psnr_val + ssim_val
        #best_output = pred
        #index = 0 
        #for i in range(1, len(tensor_list)):
            #pred=tensor_list[i].squeeze().cpu()
            #pred_np=pred.numpy()
            #mse_val=F.mse_loss(tensor_list[i], mask.squeeze().cpu())
            #psnr_val = compute_psnr(get_np, pred_np, data_range=1.0)
            #ssim_val = compute_ssim(get_np, pred_np, data_range=1.0)
            #combined_score = -mse_val + psnr_val + ssim_val
            #if combined_score > best_score:
             #   best_score = combined_score 
              #  best_output = pred
               # index = i
      #  mse_val = rmse_val = psnr_val= ssim_val = None
       # mse_val = F.mse_loss(best_output, mask.squeeze().cpu()).item()
        #rmse_val = calculate_rmse(best_output, mask.squeeze().cpu())
        #psnr_val = calculate_psnr(best_output, mask.squeeze().cpu())
        #ssim_val = calculate_ssim(best_output, mask.squeeze().cpu())

        new_row = pd.DataFrame([[title, mse_val, rmse_val, psnr_val, ssim_val]],
                    columns=['title', 'mse', 'rmse', 'psnr', 'ssim'])
        df = pd.concat([df, new_row], ignore_index=True)
        output_file_path = os.path.join(output_metrics_dir, 'evaluation_metrics.csv')
        df.to_csv(output_file_path, index=False)

        show_tensor_images(image=b, mask=mask, output=tensor_list[index], num=cnt, title=str(slice_ID))
        th.save(tensor_list[index], './output/'+str(slice_ID)+'_output')  # save the generated mask
        
        def resize_and_save_output(tensor, save_path):
             if tensor.dim() == 2:
                 tensor = tensor.unsqueeze(0)
             elif tensor.dim() == 3 and tensor.shape[0] != 1:
                 raise ValueError("Expected a single-channel tensor.")
             tensor = tensor.unsqueeze(0)
             resized = F.interpolate(tensor, size = (512, 512), mode = 'bicubic', align_corners = False)
             resized = resized.squeeze()
             image_uint8 = (resized.clamp(0,1) * 255).to(torch.uint8)
             #image_uint8 = (tensor.clamp(0,1) * 255).to(torch.uint8)
             resized_pil_img = to_pil_image(image_uint8)
             resized_pil_img.save(save_path)
            
        save_path = os.path.join(samples_dir, f"{slice_ID}_resized_output.png")
        resize_and_save_output(best_output, save_path)



        tensor_list.clear()
        cnt = cnt + 1


def create_argparser():
    defaults = dict(
        data_dir="./data/testing",
        clip_denoised=True,
        num_samples=1,
        batch_size=1,
        use_ddim=False,
        model_path="",
        num_ensemble=5  # number of samples in the ensemble
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":

    main()
