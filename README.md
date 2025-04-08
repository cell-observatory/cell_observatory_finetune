# 3D Segmentation Models

This repository implements 3D segmentation models for processing 3D Light Sheet Datasets. It extends state-of-the-art computer vision detection and instance segmentation approaches—typically designed for 2D images to higher dimensions by developing custom kernels and logic for high-quality instance segmentation.

## Table of Contents

- [Overview](#overview)
- [Background and Architecture](#background-and-architecture)
  - [Mask R-CNN Overview](#mask-r-cnn-overview)
  - [Mask DINO Overview](#mask-dino-overview)
- [Data Pipeline](#data-pipeline)
- [Model Configurations](#configurations) 
- [Installation](#installation)
- [TO DO:](#to-do)

## Overview

Detection and instance segmentation in computer vision is dominated by two major types of approaches. In the 2D realm, methods such as Mask R-CNN and recent transformer-based approaches (e.g., DETR variants) have been explored extensively. This repo extends these ideas to 3D segmentation. Our focus is on achieving high-quality instance segmentation in 3D volumes.

## Background and Architecture

### Mask R-CNN Overview

- **Two-Stage Pipeline:**
  - **Stage 1: Region Proposal Network (RPN)**
    - Extracts feature maps from an input image using a backbone network (e.g., ResNet).
    - Proposes candidate object regions (RoIs) based on features.
  - **Stage 2: Object Detection and Segmentation Heads**
    - Uses RoI pooling (e.g., RoIAlign) to extract fixed-size feature maps from each proposal.
    - Classifies the objects and refines bounding boxes.
    - Produces high-quality segmentation masks for each object.

- **Dimensional Considerations:**
  - Originally designed for 2D images.
  - Extensions to 3D require adaptation of the pooling (e.g., 3D RoI pooling), nms, and convolution operations to handle volumetric data.

- **Core Idea:**
  - The method decouples object localization from instance segmentation, allowing for dedicated refinement at each stage.
  - Maintains high accuracy by focusing on regions with a high likelihood of containing objects.

---

### Mask DINO Overview

- **Unified Transformer-Based Framework:**
  - Eliminates the need for an RPN by directly processing the image with transformer layers.
  - Uses learned object queries to directly predict object locations and segmentation masks.

- **Self-Attention Mechanism:**
  - Integrates global context through self-attention, enabling the model to model relationships between different parts of an image (or volume).
  - Iteratively refines object representations by interacting with features across the entire input.

- **Dimensional and Latent Space Adaptation:**
  - Typically operates in 2D, but the underlying transformer architecture is flexible.
  - To extend to 3D, the query and feature mechanisms must incorporate volumetric (3D) information.

- **Core Idea:**
  - Emphasizes a one-stage, end-to-end approach where object proposals and segmentation masks are generated simultaneously.
  - Leverages transformer decoders to associate high-level latent queries with fine-grained spatial details for mask prediction.

## Data Pipeline

A custom database class is used to manage data stored in TIFF or Zarr formats, supporting both detection and instance segmentation workflows.

### Key Features

- **Flexible File Handling**
  - Database metadata-driven file loading with filtering for quality flags.

- **Efficient Crop-Based Indexing**
  - Crops 3D volumes (Z, Y, X, C) into fixed-size blocks using a configurable `DataConfig`.
  - Index mapping is cached using a lightweight **SQLite** database for fast access and reproducibility.
  - Index entries store coordinates and instance segmentation targets for each crop.

## Model Configurations

This repository uses [Hydra](https://hydra.cc/) for managing all experiment configurations in a modular and hierarchical fashion. Each experiment is driven by a top-level config file that references sub-configs for the model, dataset, metrics, and backbone.

### Key Features

- **Modular Configuration Structure**  
  - All configs live in a `configs/` directory and are composed using a `defaults` list:
    ```yaml
    defaults:
      - models: maskrcnn
      - models/backbones: resnet
      - datasets: skittlez
      - metrics: metrics
    ```

- **Model Architecture Selection**  
  - Supports switching between architectures like `maskrcnn` and `maskdino`.
  - Corresponding backbones (e.g., `resnet50`, `resnet101`, etc.) are loaded dynamically.

- **Training Paradigm**  
  - The `paradigm` key specifies the training loop to use (e.g., `segmentation.training.backend_segmentation.supervised`).

- **Configurable Resource Usage**  
  - Training is parallelized across GPUs and nodes using Ray + DeepSpeed, with config entries for:
    - Number of workers and GPUs (`gpu_workers`)
    - Gradient accumulation, clipping, mixed precision, etc.
    - Full ZeRO Stage 3 optimization for large-scale training.

- **Custom Output Paths**  
  - All logs, checkpoints, and intermediate databases are routed through `outdir`, making runs easy to isolate and track.

- **Optional Profiling and Logging**  
  - Built-in support for TensorBoard, CSV logging, and DeepSpeed FLOPs profiler is configurable via the `deepspeed_config`

## Installation

This project includes custom CUDA extensions, so you'll need a working PyTorch + CUDA environment.

### Installation (Editable Mode)

Clone the repo and install it with `pip`:

```bash
git clone https://github.com/HugoHamilton/segmentation.git
cd segmentation
pip install -e .
```

## TO DO:

-  **Refactor Data Pipeline**  
  The current database + dataloader setup (`Skittlez_Database`) is a **temporary workaround** for testing. It is not cleanly integrated with the synthetic data generation or pre-training dataset generation pipeline.

-  **Troubleshoot Mask R-CNN Implementation**  
  Currently debugging and refining the 3D Mask R-CNN pipeline.

-  **Implement Mask DINO and Mask2Former in 3D**  
  Working on a **3D adaptation of Mask DINO and Mask2Former**.
