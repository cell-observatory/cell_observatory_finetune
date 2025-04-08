#!/bin/bash
#SBATCH --qos=abc_high
#SBATCH --partition=abc_a100
#SBATCH --account=co_abc
#SBATCH --job-name=train_seg
#SBATCH --output=/clusterfs/nvme/segment_4d/final_pipeline_v3/codebase/logs/train_seg.log
#SBATCH --error=/clusterfs/nvme/segment_4d/final_pipeline_v3/codebase/logs/train_seg.err
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-cpu=31000
#SBATCH --ntasks-per-node=1

# USAGE: bash /clusterfs/nvme/segment_4d/final_pipeline_v3/codebase/train/segmentation/src/segmentation/training/train_segmentation.sh

source /global/home/users/hph/miniconda3/etc/profile.d/conda.sh
conda activate 4d_seg

# using tmp/torch_extensions causes stalling in DeepSpeed init 
export TORCH_EXTENSIONS_DIR="/tmp/torch_extensions_$$"

python3 /clusterfs/nvme/segment_4d/final_pipeline_v3/codebase/train/segmentation/src/segmentation/training/train_segmentation.py