#!/bin/bash

# --- ABC

# USAGE: bash /clusterfs/nvme/hph/git_managed/cell_observatory_finetune/scripts/eval.sh

export PYTHONPATH="/clusterfs/nvme/hph/git_managed:/clusterfs/nvme/hph/git_managed/cell_observatory_finetune/cell_observatory_platform"

CFG="experiments/abc/eval/4D/mae/channel_split.yaml"
# CFG="experiments/abc/eval/4D/mae/upsample_space.yaml"
# CFG="experiments/abc/eval/4D/mae/upsample_time.yaml"
# CFG="experiments/abc/eval/4D/mae/upsample_spacetime.yaml"

# CFG="experiments/abc/eval/4D/jepa/channel_split.yaml"
# CFG="experiments/abc/eval/4D/jepa/upsample_space.yaml"
# CFG="experiments/abc/eval/4D/jepa/upsample_time.yaml"
# CFG="experiments/abc/eval/4D/jepa/upsample_spacetime.yaml"

python3 /clusterfs/nvme/hph/git_managed/cell_observatory_finetune/manager.py --config-name=${CFG}