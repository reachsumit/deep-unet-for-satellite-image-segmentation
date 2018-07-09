#!/bin/bash

# This script will request one GPU device and 1 CPU core

#SBATCH --account=mscagpu
#SBATCH --job-name=satellite
#SBATCH --output=logout_%j.txt
#SBATCH --error=logerr_%j.txt
#SBATCH --time=3:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=mscagpu
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=20000


# if your executable was built with CUDA, be sure to load the correct CUDA module:
module load Anaconda3 cuda/8.0

#
# your GPU-based executable here
#
python train_unet.py
python predict.py
