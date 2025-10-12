# Cell Observatory Finetune

`cell_observatory_finetune` extends the [`cell_observatory_platform`](../cell_observatory_platform/README.md) training engine to specialized microscopy fine-tuning pipelines. See the platform README for environment setup, logging, and launch patterns.

## Scope
Fine-tuned pretrained backbones across volumetric and time-lapse microscopy datasets for:
  - Segmentation
  - Channel splitting
  - Upsampling in Space, Time, or SpaceTime

## Key Components
- `models/heads/`: Mask2Former, MaskDINO, DensePredictor, and linear heads plus the associated multi-scale and deformable decoders.
- `models/adapters/vit_adapter.py`: Multi-scale ViT adapter with deformable cross-attention.
- `models/ops/`: Deformable attention ops binding our custom CUDA kernels.
- `configs/`: Configs for various finetuning tasks in 3D and 4D for backbones pretrained using self-supervised learning paradigms (MAE, JEPA).