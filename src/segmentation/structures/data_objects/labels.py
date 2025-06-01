from typing import Any, List, Union, Sequence

import torch
import torch.nn.functional as F

import numpy as np

from segmentation.structures.sample_objects.utils import record_init

class Labels:
    """
    Stores the instance labels for an image.

    Each row corresponds to one instance, each column to one category.

    Attributes:
        tensor : BoolTensor, shape (N, )
            N instances for image.
    """
    @record_init
    def __init__(
        self,
        tensor: Union[torch.Tensor, np.ndarray, Sequence[int]],
        num_classes: int,
    ):
        if isinstance(tensor, torch.Tensor):
            tensor = tensor.to(torch.long)
        else:
            tensor = torch.as_tensor(tensor, dtype=torch.long, device="cpu")

        assert tensor.dim() == 1, f"Expected shape (N,), got {tuple(tensor.shape)}"
        assert num_classes > 0, "num_classes must be a positive integer"
        assert (
            (tensor >= 0).all() and (tensor < num_classes).all()
        ), "Class ids must be in the range [0, num_classes-1] but got " \
            f"{tensor.min()} to {tensor.max()} for num_classes={num_classes}" \
            f"with tensor {tensor}"

        self.tensor = tensor
        self.num_classes = int(num_classes)
    
    @property
    def dtype(self) -> torch.dtype:
        """Return the dtype of the underlying tensor."""
        return self.tensor.dtype

    def clone(self) -> "Labels":
        """
        Clone the Labels.

        Returns:
            Labels
        """
        return Labels(self.tensor.clone(), self.num_classes)

    def cpu(self) -> "Labels":
        """Move Labels to CPU."""
        return Labels(self.tensor.cpu())

    def cuda(self, device=None) -> "Labels":
        """Move Labels to CUDA device."""
        return Labels(self.tensor.cuda(device))

    def detach(self) -> "Labels":
        """Detach from computation graph."""
        return Labels(self.tensor.detach())

    def to(self, *args: Any, **kwargs: Any) -> "Labels":          
        """Return a copy cast to the given *device* / *dtype*."""
        return Labels(self.tensor.to(*args, **kwargs), self.num_classes)

    @property
    def device(self) -> torch.device:
        return self.tensor.device

    def __len__(self) -> int:
        """Number of instances in this image."""
        return self.tensor.numel()

    def __repr__(self) -> str:                              
        return (
            f"{self.__class__.__name__}(num_instances={len(self)}, "
            f"num_classes={self.num_classes})"
        )

    def __iter__(self):
        """Yield a single-instance *Labels* (length = 1) at a time."""
        for cls_id in self.tensor:
            yield Labels(cls_id.unsqueeze(0), self.num_classes)

    def __getitem__(self, item: Union[int, slice, torch.BoolTensor, str]) -> Union["Labels", Any]:
        return Labels(self.tensor[item], self.num_classes)  

    def to_onehot(self) -> torch.BoolTensor:
        """Return a Boolean one-hot matrix, shape ``(N, num_classes)``."""
        return F.one_hot(self.tensor, num_classes=self.num_classes).bool()

    def nonempty(self) -> torch.BoolTensor:
        """All instances are “non-empty” by definition; kept for API parity."""
        return torch.ones_like(self.tensor, dtype=torch.bool)

    @staticmethod
    def cat(labels_list: List["Labels"]) -> "Labels":
        """Concatenate multiple *Labels* objects along the instance axis."""
        if not labels_list:
            raise ValueError("labels_list must be non-empty")

        first_nc = labels_list[0].num_classes
        if any(l.num_classes != first_nc for l in labels_list):
            raise ValueError("All Labels must share the same num_classes")

        cat_tensor = torch.cat([l.tensor for l in labels_list], dim=0)
        return Labels(cat_tensor, num_classes=first_nc)