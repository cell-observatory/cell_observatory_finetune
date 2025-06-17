"""
https://github.com/IDEA-Research/MaskDINO/blob/3831d8514a3728535ace8d4ecc7d28044c42dd14/maskdino/modeling/meta_arch/maskdino_head.py#L4

Apache License
Version 2.0, January 2004
http://www.apache.org/licenses/

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""


from torch import nn


class MaskDINOHead(nn.Module):
    def __init__(
        self,
        *,
        num_classes: int,
        pixel_decoders: nn.Module,
        decoders: nn.Module,
    ):
        """
        Args:
            input_shape: shapes (channels and stride) of the input features
            num_classes: number of classes to predict
            pixel_decoder: the pixel decoder module
            ignore_value: category id to be ignored during training.
            decoder: the transformer decoder that makes prediction
        """
        super().__init__()

        self.num_classes = num_classes

        self.pixel_decoder = pixel_decoders
        self.decoder = decoders

    def forward(self, features, mask = None, targets = None):
        mask_features, transformer_encoder_features, multi_scale_features = self.pixel_decoder.forward_features(features, mask)
        predictions = self.decoder(multi_scale_features, mask_features, mask, targets = targets)
        return predictions 