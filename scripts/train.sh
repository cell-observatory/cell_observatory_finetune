#!/bin/bash

# --- ABC

# USAGE: bash /clusterfs/nvme/hph/git_managed/cell_observatory_finetune/scripts/train.sh

export PYTHONPATH="/clusterfs/nvme/hph/git_managed:/clusterfs/nvme/hph/git_managed/cell_observatory_finetune/cell_observatory_platform"

# --- 4D ----

# CFG="experiments/abc/4D/mae/channel_split/test_channel_split_linear_10_13_25.yaml"
# CFG="experiments/abc/4D/mae/channel_split/test_channel_split_vit_10_13_25.yaml"
# CFG="experiments/abc/4D/mae/channel_split/test_channel_split_dpt_10_13_25.yaml"

# CFG="experiments/abc/4D/mae/upsample_space/test_linear_10_13_25.yaml"
# CFG="experiments/abc/4D/mae/upsample_space/test_vit_10_13_25.yaml"

# CFG="experiments/abc/4D/mae/upsample_time/test_vit_10_13_25.yaml"

# CFG="experiments/abc/4D/mae/upsample_spacetime/test_vit_10_13_25.yaml"

# CFG="experiments/abc/4D/jepa/channel_split/test_linear_10_13_25.yaml"
# CFG="experiments/abc/4D/jepa/channel_split/test_vit_10_13_25.yaml"

# CFG="experiments/abc/4D/jepa/upsample_space/test_linear_10_03_25.yaml"
# CFG="experiments/abc/4D/jepa/upsample_space/test_vit_10_03_25.yaml"

# CFG="experiments/abc/4D/jepa/upsample_time/test_vit_10_03_25.yaml"

# CFG="experiments/abc/4D/jepa/upsample_spacetime/test_vit_10_03_25.yaml"

# --- 3D ----

# CFG="experiments/abc/3D/mae/channel_split/test_channel_split_linear_10_13_25.yaml"
# CFG="experiments/abc/3D/mae/channel_split/test_channel_split_vit_10_13_25.yaml"

# CFG="experiments/abc/3D/mae/upsample_space/test_linear_10_13_25.yaml"
# CFG="experiments/abc/3D/mae/upsample_space/test_vit_10_13_25.yaml"

# CFG="experiments/abc/3D/jepa/channel_split/test_linear_10_13_25.yaml"
# CFG="experiments/abc/3D/jepa/channel_split/test_vit_10_13_25.yaml"

# CFG="experiments/abc/3D/jepa/upsample_space/test_linear_10_03_25.yaml"
CFG="experiments/abc/3D/jepa/upsample_space/test_vit_10_03_25.yaml"

python3 /clusterfs/nvme/hph/git_managed/cell_observatory_finetune/manager.py --config-name=${CFG}