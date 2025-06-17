#!/bin/bash

# USAGE: bash /clusterfs/nvme/hph/git_managed/cell_observatory_finetune/train/train_multinode.sh

# Available configurations:

# CFG="config_mrcnn_resnet.yaml"
# CFG="config_mrcnn_resnet_fpn.yaml"
# CFG="config_mrcnn_convnext_fpn.yaml"
# CFG="config_mrcnn_intern_image_fpn.yaml"
# CFG="config_mrcnn_hiera_fpn.yaml"
# CFG="config_mrcnn_vitDet.yaml"
# CFG="config_maskdino_resnet.yaml"
CFG="config_channel_split_masked_predictor_mae.yaml"
# CFG="config_upsample_masked_predictor_mae.yaml"
# CFG="config_denoise_dnmodel_mae.yaml"
# CFG="config_channel_predict_masked_predictor_mae.yaml"

python /clusterfs/nvme/hph/git_managed/cell_observatory_finetune/clusters/manager.py --config-name=${CFG}