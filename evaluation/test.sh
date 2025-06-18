#!/bin/bash
#SBATCH --qos=abc_high
#SBATCH --partition=abc_a100
#SBATCH --account=co_abc
#SBATCH --job-name=train_seg
#SBATCH --output=/clusterfs/nvme/segment_4d/final_pipeline_v3/codebase/logs/train_seg_%A_%a.log
#SBATCH --error=/clusterfs/nvme/segment_4d/final_pipeline_v3/codebase/logs/train_seg_%A_%a.err
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-cpu=31000
#SBATCH --ntasks-per-node=1

# USAGE: bash /clusterfs/nvme/hph/git_managed/cell_observatory_finetune/train/train.sh

source /global/home/users/hph/miniconda3/etc/profile.d/conda.sh
conda activate 4d_seg

# apptainer build --fakeroot develop_torch_cuda_12_8_ops3d.sif apptainerfile.def

# using tmp/torch_extensions causes stalling in DeepSpeed init 
export TORCH_EXTENSIONS_DIR="/tmp/torch_extensions_$$"
export PYTHONPATH="/clusterfs/nvme/hph/git_managed/cell_observatory_finetune:${PYTHONPATH}"
export PYTHONPATH="/clusterfs/nvme/hph/git_managed/cell_observatory_finetune/cell_observatory_platform:${PYTHONPATH}"
export PYTHONPATH="/clusterfs/nvme/hph/git_managed:${PYTHONPATH}"

CFG="base_evaluation_channel_split_masked_predictor.yaml"

python3 /clusterfs/nvme/hph/git_managed/cell_observatory_finetune/train/run.py --config-name=${CFG}