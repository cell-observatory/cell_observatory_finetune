"""
https://github.com/pytorch/vision/blob/95f10a4ec9e43b2c8072ae5a68edd5700f9b1e45/torchvision/models/detection/image_list.py#L8

BSD 3-Clause License

Copyright (c) Soumith Chintala 2016, 
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

from typing import List, Tuple

import torch
from torch import Tensor


# TODO: move to a file with other custom data structures
class ImageList:
  """
  Structure that holds a list of images (of possibly
  varying sizes) as a single tensor.
  This works by padding the images to the same size,
  and storing in a field the original sizes of each image

  Args:
    tensors (tensor): Tensor containing images.
    image_sizes (list[tuple[int, int]]): List of Tuples each containing size of images.
  """

  def __init__(self, tensors: Tensor, image_sizes: List[Tuple[int, int]]) -> None:
    self.tensors = tensors
    self.image_sizes = image_sizes

  def to(self, device: torch.device) -> "ImageList":
    cast_tensor = self.tensors.to(device)
    return ImageList(cast_tensor, self.image_sizes)