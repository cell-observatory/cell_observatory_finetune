from typing import Optional

from torch import nn

from finetune.models.meta_arch.base_model import BaseModel
from finetune.models.meta_arch.preprocessor import PreProcessor


class EncoderDecoder(BaseModel):
    """
    General encoder-decoder model meta architecture.
    """
    def __init__(self, 
                 masking: bool,
                #  mask_generators: Optional[callable],
                 patch_embeddings: Optional[callable],
                 preprocessors: PreProcessor, 
                 backbones: nn.Module, 
                 heads: nn.Module, 
                 losses: callable
    ):
        super().__init__(preprocessor=preprocessors)

        self.masking = masking
        # self.mask_generator = mask_generators
        
        self.patch_embedding = patch_embeddings

        self.encoder = backbones
        self.decoder = heads
        self.loss = losses

    def _forward(self, data_sample):
        """
        Forward pass of the encoder-decoder model.
        """
        # masking used for channel prediction 
        # and temporal upsampling
        if self.masking:
            mask = data_sample.gt_instances.mask.tensor
            original_patch_indices = data_sample.gt_instances.original_patch_indices.tensor
        else:
            mask, original_patch_indices = None, None
        
        # TODO: is all this branching logic necessary, should be 
        #       possible to just set branching logic in encoder/decoder
        if self.masking:
            # tensor: (B, (T), D, H, W, C) -> (B, L, C) -> (B, L, C')
            latent, patches = self.encoder(data_sample.data_tensor.tensor, masks=mask)
        else:
            latent, *_ = self.encoder(data_sample.data_tensor.tensor)

        if self.masking:
            # tensor: (B, L, C') -> (B, L, C)
            pred = self.decoder(latent, target_masks=mask, original_patch_indices=original_patch_indices) 
        else:
            pred = self.decoder(latent)
        
        # patchify target if patch embedding is used 
        # TODO: might be easier to just reshape pred instead?
        if self.patch_embedding is not None:
            # tensor: (B, (T), D, H, W, C) -> (B, L, C)
            target = self.patch_embedding.patchify(data_sample.gt_instances.image.tensor)
        else:
            # TODO: not currently used, consider 
            #       simplifying branching logic
            target = data_sample.gt_instances.image.tensor

        if self.masking:
            # loss: ((B, L, C), (B, L, C)) -> loss value
            loss = {"step_loss": self.loss(target, pred, masks=mask)}
        else:
            loss = {"step_loss": self.loss(target, pred)}
        return loss, pred