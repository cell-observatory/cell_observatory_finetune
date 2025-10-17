#!/bin/bash

# --- ABC

# USAGE: bash /clusterfs/nvme/hph/git_managed/cell_observatory_finetune/scripts/inference.sh

export PYTHONPATH="/clusterfs/nvme/hph/git_managed:/clusterfs/nvme/hph/git_managed/cell_observatory_finetune/cell_observatory_platform"

# CFG="experiments/abc/inference/test_inference_mae.yaml"
CFG="experiments/abc/inference/test_inference_jepa.yaml"

python3 /clusterfs/nvme/hph/git_managed/cell_observatory_finetune/manager.py --config-name=${CFG}