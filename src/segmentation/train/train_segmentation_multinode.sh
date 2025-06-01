#!/bin/bash

# USAGE: bash /clusterfs/nvme/hph/git_managed/segmentation/src/segmentation/training/train_segmentation_multinode.sh

# Available configurations:

# CFG="config_mrcnn_resnet.yaml"
# CFG="config_mrcnn_resnet_fpn.yaml"
# CFG="config_mrcnn_convnext_fpn.yaml"
CFG="config_mrcnn_hiera_fpn.yaml"
# CFG="config_mrcnn_vitDet.yaml"

python /clusterfs/nvme/hph/git_managed/segmentation/src/segmentation/clusters/manager.py --config-name=${CFG}