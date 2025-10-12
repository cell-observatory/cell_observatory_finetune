#!/bin/bash

# --- ABC

# USAGE: bash /clusterfs/nvme/hph/git_managed/cell_observatory_finetune/scripts/train.sh

export PYTHONPATH="/clusterfs/nvme/hph/git_managed:/clusterfs/nvme/hph/git_managed/cell_observatory_finetune/cell_observatory_platform"

CFG="experiments/abc/4D/mae/test_channel_split_10_03_25.yaml"
# CFG="experiments/abc/4D/mae/test_upsample_space_10_03_25.yaml"
# CFG="experiments/abc/4D/mae/test_upsample_time_10_03_25.yaml"
# CFG="experiments/abc/4D/mae/test_upsample_spacetime_10_03_25.yaml"

# CFG="experiments/abc/4D/jepa/test_channel_split_10_03_25.yaml"
# CFG="experiments/abc/4D/jepa/test_upsample_space_10_03_25.yaml"
# CFG="experiments/abc/4D/jepa/test_upsample_time_10_03_25.yaml"
# CFG="experiments/abc/4D/jepa/test_upsample_spacetime_10_03_25.yaml"

python3 /clusterfs/nvme/hph/git_managed/cell_observatory_finetune/manager.py --config-name=${CFG}