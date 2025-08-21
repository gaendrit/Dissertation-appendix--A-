#!/bin/bash
#$ -cwd
#$ -j y
#$ -pe smp 12         # 12 cores per GPU
#$ -l h_rt=24:0:0    # 240 hours runtime
#$ -l h_vmem=7.5G     # 7.5G RAM per core
#$ -l gpu=1           # request 2 GPU
#$ -l cluster=andrena # use the Andrena nodes and enable 12 cores per GPU

module load python/3.8.5-gcc-12.2.0


source ~/envs/denoising_envs/bin/activate

cd /data/home/bt241032/model

MODEL_FLAGS="--num_channels 128 --class_cond False --num_res_blocks 2 --num_heads 1 --learn_sigma True --use_scale_shift_norm False --attention_resolutions 16"
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule linear --rescale_learned_sigmas False --rescale_timesteps False"
TRAIN_FLAGS="--lr 1e-4 --batch_size 10"

python scripts/s2.py  --data_dir ./data/testing  --model_path ./results/savedmodel048000.pt --num_ensemble=5 $MODEL_FLAGS $DIFFUSION_FLAGS
