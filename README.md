# 3D Segmentation Models

3D segmentation models designed specifically for 3D Light Sheet (LS) microscopy datasets. 
Built with [PyTorch](https://pytorch.org/), accelerated and scaled with [Ray](https://www.ray.io/), and flexibly configured using [Hydra](https://hydra.cc/).

## Table of Contents

- [Installation](#installation)
- [Get Started](#get-started)
  - [Training](#training)
  - [Evaluation](#evaluation)
- [Models](#background-and-architecture)
  - [ResNet](#resnet-overview)
  - [ViTDet](#vitdet-overview)
  - [Hiera](#hiera-overview)
  - [FPN](#fpn)
  - [Mask R-CNN](#mask-r-cnn-overview)
- [Data Pipeline](#data-pipeline)
  - [Dataset Example: `Skittlez_Database`](#skittlez_database-example)
- [Model Configurations](#configurations) 

## Installation

This package has been tested on **CUDA 12.4** with **PyTorch 2.4.1+cu124**. We recommend using a dedicated `conda` environment.

```bash
# 1. Create & activate conda environment
conda create -n segmentation python=3.10 -y
conda activate segmentation

# 2. Install matching CUDA‐enabled PyTorch & torchvision
conda install -c pytorch pytorch=2.4.1 torchvision=0.19.1 cudatoolkit=12.4 -c nvidia

# 3. Install package in editable mode
pip install -e . 
```

## Get Started

### Training

Run training locally by specifying a Hydra config file (see [Model Configurations](#configurations) for details on how to configure experiments using Hydra YAML files):

```bash
python3 training/train_segmentation.py --config-name=config_mrcnn_vitDet.yaml
```

Or run via SLURM scheduler using the provided bash script (edit the slurm bash file as needed):

```bash
sbatch training/train_segmentation.sh --config-name=config_mrcnn_vitDet.yaml
```

### Evaluation

Run evaluation locally by specifying a Hydra config file:

```bash
python3 evaluation/evaluate.py --config-name=skittlez_evaluation_vitDet.yaml
```

Or run via SLURM scheduler using the provided bash script (edit the slurm bash file as needed):

```bash
sbatch evaluation/evaluate.sh --config-name=skittlez_evaluation_vitDet.yaml
```

## Models

### ResNet Overview

Adapted from Torchvision’s ResNet [1], our 3D‑ResNet replaces all 2D operations with their 3D counterparts (`nn.Conv3d`, `nn.BatchNorm3d`, `nn.MaxPool3d`, etc.).

- **Stem**  
  - **Conv3d:** `out_channels = 64`, `kernel = 7³`, `stride = 2`, `padding = 3`  
  - **BatchNorm3d** 
  - **ReLU** 
  - **MaxPool3d:** `kernel = 3`, `stride = 2`, `padding = 1`  

- **Residual Stages**  
  Each stage consists of a `block` module (either **Basic** or **Bottleneck** blocks). Depending on volume size, the number of layers used may vary. 
  1. **Stage p2**   
     Output: `(64·expansion, D/4, H/4, W/4)`  
  2. **Stage P3**   
     Output: `(128·expansion, D/8, H/8, W/8)`  
  3. (Optional) **Stage p4**   
     Output: `(256·expansion, D/16, H/16, W/16)`  
  4. (Optional) **Stage p5**   
     Output: `(512·expansion, D/32, H/32, W/32)`

- **Block Types**  
  - **BasicBlock** (`expansion = 1`):  
    ```python
    # Pseudocode
    out  =  ReLU(BN3D(conv3x3(x)))
    out  =  BN3D(conv3x3(x))
    out +=  x
    out  =  ReLU(out)
    ```  
    
  - **Bottleneck** (`expansion = 4`):  
    ```python
    # Pseudocode
    out  = RELU(BN3D(conv1x1(x)))
    out  = ReLU(BN3D(conv3x3(out)))
    out  = BN3D(conv1x1(out))
    out += x
    out  = ReLU(out)
    ```  

### ViTDet Overview

Adapted from Detectron2’s VitDet implementation [3], our 3D-ViTDet replaces any 2D operations with their 3D counterparts:

- **Patch Embedding**  
  - **Conv3d:** `out_channels = embed_dim`, `kernel = (patch_size)³`, `stride = (patch_size)³`  

- **Absolute Positional Embeddings**  
  - Optional **absolute** pos‑embeddings interpolated to match the input size if `image_size != pretrain_image_size` 

- **Transformer Blocks** 
    ```python
    # Pseudocode
    shortcut = x
    x = norm1(x)
    if window_size > 0:
        x = window_partition(x)
    x = attn(x)
    if window_size > 0:
        x = window_unpartition(x)
    x = shortcut + drop_path(x)
    x = x + drop_path(mlp(norm2(x)))
    if use_residual_block:
        x = residual(x)
    ```
  
- **Simple Feature Pyramid**  
  Builds `{p2, p3, …}` from `"last_feat"` through upsamling/downsampling with `scale factors` (e.g. 4×, 2×, 1×):
  1. **Upsample/Downsample**  
     - `ConvTranspose3d` for `scale>1`  
     - `MaxPool3d` for `scale<1`  
  2. **Channel unify**  
     - `Conv3d` (1×1×1 → 3×3×3 ) to set each map to `out_channels`  

### Hiera Overview

Taken from “Hiera: A Hierarchical Vision Transformer without the Bells‑and‑Whistles” [3]:

- **Patch Embedding**  
  - **Conv3d:** `out_channels = embed_dim`, `kernel = (patch_kernel)³`, `stride = (patch_stride)³`
    - Zeros-out masked regions before `Conv3d` to prevent leakage of masked regions when using overlapping kernels

- **Positional Embedding**  
  - Learned `positional embeddings`, optionally decomposed across spatial dimensions (Z, Y, X)

- **Unroll & Reroll**  
  - **`Unroll`** Reorders the tokens such that patches are contiguous in memory (e.g. `[B, (D, H, W), C] -> [B, (Sz, Sy, Sx, D // Sz, H // Sy, W // Sx), C]`).
  - **`Reroll`** undos `unroll` operation at stage end to recover original spatial order for intermediate feature maps.

- **HieraBlock**  
 ```python
    # Pseudocode
    x_norm = norm1(x)
    if dim != dim_out:
        # query pooling
        x = do_pool(proj(x_norm), stride=q_stride)
    x = x + drop_path(attn(x_norm))
    x = x + drop_path(mlp(norm2(x)))
 ```

- **Hierarchical Stages**  
  - After each stage (e.g. `stages=[2, 3, 16]` Hiera Blocks), set `new_embed_dim = embed_dim * dim_mul` and `new_num_heads = num_heads * head_mul`.  
  - If `return_intermediates=True`, applies `Reroll` at each stage end to yield feature maps of shape `(B, embed_dim_intermediate, D_intermediate, H_intermediate, W_intermediate)`.

### FPN Overview

We optionally combine each 3D backbone with a Feature Pyramid Network [5], adapted from Torchvision's 2D FPN by replacing all 2D operations with their 3D counterparts:

- **Return Layers**  
  Select intermediate outputs (e.g. `"p2"`, `"p3"`) from the ResNet to build the pyramid with `return_layers`.
- **FPN**
  - **Lateral Conv3d:** `kernel_size = 1³`, `stride = 1³`, `padding = 0`
  - **Top‑Down Pathway:** `F.interpolate(coarse_map, mode="nearest") + lateral map`  
  - **Merged Conv3d:** `kernel_size = 3³`, `stride = 1³`, `padding = 1`

All 3D‑FPN outputs are returned as an `OrderedDict[str, Tensor]` of multi‑scale feature maps, for downstream detection or segmentation heads.  

### Mask R-CNN Overview

Adapted from TorchVision’s Mask R‑CNN, our 3D mask R-CNN replaces all 2D operations with their 3D counterparts. All custom CUDA kernels are very lightly adapted from https://github.com/TimothyZero/MedVision. 

- **GeneralizedRCNNTransform**:  
  1. **Resize** the shorter side to range `[min_size, max_size]`  
  2. **Normalize** per-channel with `image_mean` and `image_std`  
  3. **Pad & batch** into `ImageList` data structure

- **Backbone**  
  - Any `backbone` above, returning either a single `(B, C_out, D′, H′, W′)` tensor or an `OrderedDict[str, Tensor]` of multi‑scale feature maps.

- **Region Proposal Network (RPN)**  
  1. **AnchorGenerator** generates 3D anchors for each `(D,H,W)` with specified `anchor_sizes`, `aspect_ratios`, and `aspect_ratios_z`
  2. **RPNHead:**  
     - **Conv3dNormActivation:** repeated `conv_depth` times, `kernel_size = 3`, `stride = 1`
     - **Objectness Score Conv3d:** `kernel_size = 1`, `stride = 1`, `out_channels = num_anchors`   
     - **Box Deltas Conv3d**: `kernel_size = 1`, `stride = 1`, `out_channels = 6*num_anchors`
  3. **Decode & Filter** with `BoxCoder.decode`, clip to volume bounds, discard small/low‑score boxes, and apply NMS

- **ROI Heads**  
  1. **MultiScaleRoIAlign:** Leverages feature maps `featmap_names`, with `output_size=7` & `sampling_ratio=2` to align each ROI proposal from the RPN to correct feature map in hierarchy and resize to `(output_size, output_size, output_size)`.  
  2. **TwoMLPHead**:  
     ```python    
        # Pseudocode
        x = flatten(x, start_dim=1)       # [b, ch·d·h·w]
        x = relu(fc6(x))                  # first fully‐connected + activation
        x = relu(fc7(x))                  # second fully‐connected + activation
      ```
  3. **FastRCNNPredictor**:  
      ```python
          # Pseudocode
          scores      = cls_score(x)                 # → [b, num_classes]
          bbox_deltas = bbox_pred(x)                 # → [b, num_classes * 6]
      ```
- **Mask Branch**  
  Extends Faster R‑CNN with a parallel mask roi head:
  1. **MultiScaleRoIAlign**: Leverages feature maps `featmap_names`, with `output_size=14` & `sampling_ratio=2` to align each ROI proposal from the RPN to correct feature map in hierarchy and resize to `(output_size, output_size, output_size)`.   
  2. **MaskRCNNHeads**: `N` layers `Conv3dNormActivation(channels_out=256, kernel=3, padding=1)`  
  3. **MaskRCNNPredictor**:  
     - `ConvTranspose3d(channel_in=256, channels_out=256, kernel=2, stride=2)` 
     - `Relu`
     - `Conv3d(channel_in=256, channels_out=num_classes, kernel=1, stride=1)`  

### References

[1]: “Deep Residual Learning for Image Recognition,” (https://arxiv.org/abs/1512.03385)  

[2]: “Bag of Tricks for Image Classification with Convolutional Neural Networks,” (https://arxiv.org/abs/1812.01187)

[3]: “Exploring Plain Vision Transformer Backbones for Object Detection,” (https://arxiv.org/abs/2203.16527)  

[4]: “Hiera: A Hierarchical Vision Transformer without the Bells-and-Whistles,” (https://arxiv.org/abs/2306.00989)

[5]: “Feature Pyramid Networks for Object Detection,” (https://arxiv.org/abs/1612.03144)

[6]: “Faster R‑CNN: Towards Real-Time Object Detection with Region Proposal Networks,” (https://arxiv.org/abs/1506.01497)  

[7]: “Mask R‑CNN,” (https://arxiv.org/abs/1703.06870)


## Data Pipeline

The data pipeline is centered around a single entry point, `gather_dataset`, which:

1. **Instantiates Transforms**  
   - Reads `config.transforms.transforms_list` (a list of Hydra‐style transform configs)  
   - Wraps them in a custom `Compose` class to apply chained preprocessing (cropping, normalization, augmentations, etc.)

2. **Instantiates Dataset**  
   - Calls `instantiate(config.datasets.database, ...)`  to instantiate the dataset class
   - Currently, only `Skittlez_Database` is supported, but any can `torch.utils.data.Dataset` class may be used  

3. **Creates DataLoader(s)**  
   - If the `config.datasets.return_dataloaders` flag is set, we:  
     - Optionally split into train/val via `random_split`  
     - Wrap each split or the full dataset in a `DataLoader` with:  
       - `batch_size=config.worker_batch_size`  
       - `DistributedSampler` (for multi‑GPU/distributed) or plain sampling for evaluation  

4. **Otherwise** returns the raw `Dataset` for manual handling.

### Dataset Examples

`Skittlez_Database` 

- **Initialization**  
  - Reads metadata from a remote DB or a local SQLite DB (`_query_db`)  
  - Scans TIFF/Zarr files, builds a **local** SQLite index of all valid `(z,y,x,c)` crops with their corresponding bounding boxes and mask IDs  

- **`__len__` / `__getitem__`**  
  - `__len__`: number of crops (i.e. rows in `store_index_map`)  
  - `__getitem__(i)`  
    1. Look up the i‑th crop’s file path, slice coordinates, and instance segmentation target information (bboxes, masks, etc.)   
    2. Load the raw subvolume (`tifffile`/`TensorStore`), crop out the `(Z, Y, X)` window and any selected color channels  
    3. Fetch all ground‑truth bboxes and binary masks for that crop from corresponding DB tables  
    4. Convert to PyTorch tensors:  
       ```python
       image: Tensor[C, Z, Y, X]
       target: {
         "boxes":  Tensor[N, 6],
         "labels": Tensor[N],
         "masks":  Tensor[N, Z, Y, X]
       }
       ```  
    5. Apply `self.transforms(image, target)` if provided  

## Model Configurations with Hydra

We use [Hydra](https://hydra.cc/) for managing experiment configurations. Hydra allows you to construct experiments by composing modular YAML snippets.

### 1. Select Base Configurations

Use the `defaults:` list to select the base YAML configurations for your experiment:

```yaml
defaults:
  - models:           maskrcnn_fpn
  - models/backbones: hiera_fpn
  - datasets:         skittlez
  - transforms:       transforms_skittlez
  - metrics:          metrics
  - _self_            # load this file’s overrides last
```

### 2. Override Only What You Need

Hydra handles overrides, allowing precise experiment adjustments. Scalars and lists are replaced outright, whereas dictionaries are merged recursively (only specified keys change).

Example overrides in your main config file:

```yaml
# Change batch size
datasets:
  batch_size: 4

# Swap out backbone model
backbone_target: segmentation.models.backbones.resnet.ResNet
backbone_out_channels: 2048

# Adjust RPN anchor sizes
models:
  rpn_anchor_generator:
    sizes:
      - [16, 32, 64]
      - [32, 64, 128]
```

### 3. Configuration Directory Structure & Usage

Here's what each configuration subdirectory handles:

- **`models/`**
  - Defines complete model specifications (e.g., Mask R‑CNN variants).
- **`models/backbones/`**
  - Defines backbone networks and corresponding FPN wrappers.
- **`datasets/`**
  - Defines dataset classes and parameters.
- **`transforms/`**
  - Defines data augmentation and preprocessing pipelines and parameters.
- **`metrics/`**
  - Defines evaluation metrics computed during validation by our custom `Evaluator` class.

### 3. Extend or Add New Modules

Add a new YAML under the proper group, e.g. `configs/models/backbones/my_new_backbone.yaml`. To support a brand‑new model or dataset, just drop in your new small YAML and reference it in your defaults: block.

```yaml
defaults:
  - models/backbones: my_new_backbone
  ...
  ...
  ...
  - _self_
```