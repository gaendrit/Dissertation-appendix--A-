#!/bin/bash
#$ -cwd
#$ -j y
#$ -pe smp 12         # 12 cores per GPU
#$ -l h_rt=24:0:0    # 240 hours runtime
#$ -l h_vmem=7.5G     # 7.5G RAM per core
#$ -l gpu=1          # request 2 GPU
#$ -l cluster=andrena # use the Andrena nodes and enable 12 cores per GPU

module load python/3.8.5-gcc-12.2.0


source ~/envs/denoising_envs/bin/activate

cd /data/home/bt241032/model

python con.py
