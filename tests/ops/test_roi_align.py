import torch

try:
    import ops3d._C as _C
except ImportError:
    print("3D NMS op failed to load. Please compile ops3d if needed.")

# TODO: Implement unit test here