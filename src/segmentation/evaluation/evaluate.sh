#!/bin/bash
#SBATCH --qos=abc_high
#SBATCH --partition=abc_a100
#SBATCH --account=co_abc
#SBATCH --job-name=eval
#SBATCH --output=/clusterfs/nvme/segment_4d/final_pipeline_v3/codebase/logs/eval_seg.log
#SBATCH --error=/clusterfs/nvme/segment_4d/final_pipeline_v3/codebase/logs/eval_seg.err
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-cpu=31000
#SBATCH --ntasks-per-node=1

# USAGE: bash /clusterfs/nvme/hph/git_managed/segmentation/src/segmentation/evaluation/evaluate.sh

source /global/home/users/hph/miniconda3/etc/profile.d/conda.sh
conda activate 4d_seg

# using tmp/torch_extensions causes stalling in DeepSpeed init 
export TORCH_EXTENSIONS_DIR="/tmp/torch_extensions_$$"

# CFG="skittlez_evaluation_mrcnn_resnet.yaml"
# CFG="skittlez_evaluation_mrcnn_resnet_fpn.yaml"
# CFG="skittlez_evaluation_mrcnn_intern_image_fpn.yaml"
# CFG="skittlez_evaluation_mrcnn_hiera_fpn.yaml"
# CFG="skittlez_evaluation_mrcnn_vitDet.yaml"
CFG="skittlez_evaluation_maskdino_resnet.yaml"

python3 /clusterfs/nvme/hph/git_managed/segmentation/src/segmentation/evaluation/evaluate.py --config-name=${CFG}