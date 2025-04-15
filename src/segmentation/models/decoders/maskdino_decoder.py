"""
https://github.com/IDEA-Research/MaskDINO/blob/3831d8514a3728535ace8d4ecc7d28044c42dd14/maskdino/modeling/transformer_decoder/maskdino_decoder.py

(ADD COPYRIGHT HERE)
"""


import fvcore.nn.weight_init as weight_init

import torch
from torch import nn
from torch.nn import functional as F

from segmentation.models.decoders.dino_decoder import TransformerDecoder, DeformableTransformerDecoderLayer
from segmentation.models.utils.model_utils import Conv3d, MLP, inverse_sigmoid, compute_unmasked_ratio, gen_encoder_output_proposals
from segmentation.models.rpn.box_ops import box_ops


class MaskDINODecoder(nn.Module):
    def __init__(
            self,
            in_channels,
            mask_classification=True,
            *,
            num_classes: int,
            hidden_dim: int,
            num_queries: int,
            n_heads: int,
            dim_feedforward: int,
            dec_layers: int,
            mask_dim: int,
            enforce_input_proj: bool,
            two_stage_flag: bool,
            denoise_queries_flag: str,
            noise_scale:float,
            total_denosing_queries:int,
            initialize_box_type:bool,
            initial_prediction_flag:bool,
            learn_query_embeddings: bool,
            total_num_feature_levels: int = 4,
            dropout: float = 0.0,
            activation: str = 'relu',
            nhead: int = 8,
            dec_n_points: int = 4,
            return_intermediate_dec: bool = True,
            query_dim: int = 6,
            dec_layer_share: bool = False,
            semantic_ce_loss: bool = False,
    ):
        """
        Args:
            in_channels: channels of the input features
            mask_classification: whether to add mask classifier or not
            num_classes: number of classes
            hidden_dim: Transformer feature dimension
            num_queries: number of queries
            n_heads: number of heads
            dim_feedforward: feature dimension in feedforward network
            enc_layers: number of Transformer encoder layers
            dec_layers: number of Transformer decoder layers
            pre_norm: whether to use pre-LayerNorm or not
            mask_dim: mask feature dimension
            enforce_input_project: add input project 1x1 conv even wehen input
                channels and hidden dim are identical
            d_model: transformer dimension
            dropout: dropout rate
            activation: activation function
            nhead: num heads in multi-head attention
            dec_n_points: number of sampling points in decoder
            return_intermediate_dec: return the intermediate results of decoder
            query_dim: 6 -> (x, y, z, w, h, d)
            dec_layer_share: whether to share each decoder layer
            semantic_ce_loss: use ce loss for semantic segmentation
        """
        super().__init__()

        assert mask_classification, "MaskDINO only supports mask classification mode"
        self.mask_classification = mask_classification
        
        self.num_feature_levels = total_num_feature_levels
        self.total_num_feature_levels = total_num_feature_levels # TODO: Is this safe to remove?

        # flag to support prediction from initial matching and denoising outputs
        self.initial_prediction_flag = initial_prediction_flag 

        # transformer decoder 
        self.denoise_queries_flag = denoise_queries_flag
        # flag to generate learnable embeddings or to use encoder outputs (two-stage mode)
        self.learn_query_embeddings = learn_query_embeddings
        
        # noise_scale for bbox: max = 1.0 => noise is up to full d/h/w range (50% shift) 
        # noise_scale for labels: corrupt labels with p = noise_scale * 0.5
        self.noise_scale = noise_scale 
        
        self.num_heads = n_heads
        self.total_denosing_queries = total_denosing_queries # total number of noisy copies of ground-truth targets injected into the decoder
        self.num_layers = dec_layers # number of decoder layers
        
        self.two_stage_flag = two_stage_flag
        self.initialize_box_type = initialize_box_type

        self.num_queries = num_queries # nr. object queries transformer decoder uses => nr. predictions makes per image
        self.semantic_ce_loss = semantic_ce_loss

        # define learnable query features (in two-stage queries are generated from encoder output)
        if not two_stage_flag or self.learn_query_embeddings:
            self.query_features = nn.Embedding(num_queries, hidden_dim)
        
        if not two_stage_flag and initialize_box_type == 'no':
            self.query_embeddings = nn.Embedding(num_queries, 6)
        
        # Encoder generates class logits and proposals
        # project and normalize proposals before feeding into the decoder
        if two_stage_flag:
            self.encoder_output_mlp = nn.Linear(hidden_dim, hidden_dim)
            self.encoder_output_norm = nn.LayerNorm(hidden_dim)

        self.input_proj = nn.ModuleList()
        for _ in range(self.num_feature_levels):
            if in_channels != hidden_dim or enforce_input_proj:
                self.input_proj.append(Conv3d(in_channels, hidden_dim, kernel_size=1))
                weight_init.c2_xavier_fill(self.input_proj[-1])
            else:
                self.input_proj.append(nn.Sequential())

        self.num_classes = num_classes
        # output FFNs (TODO: Ensure new logic works)
        if self.semantic_ce_loss:
            self.class_predictor = nn.Linear(hidden_dim, num_classes+1)
        else:
            self.class_predictor = nn.Linear(hidden_dim, num_classes)
        
        self.label_embeddings = nn.Embedding(num_classes,hidden_dim)
        self.mask_embeddings = MLP(hidden_dim, hidden_dim, mask_dim, 3)

        # initialize deformable transformer decoder
        self.decoder_norm = decoder_norm = nn.LayerNorm(hidden_dim)
        decoder_layer = DeformableTransformerDecoderLayer(hidden_dim, 
                                                          dim_feedforward,
                                                          dropout, 
                                                          activation,
                                                          self.num_feature_levels, 
                                                          nhead, 
                                                          dec_n_points)
        self.decoder = TransformerDecoder(decoder_layer, 
                                          self.num_layers, 
                                          decoder_norm,
                                          return_intermediates=return_intermediate_dec,
                                          embed_dim=hidden_dim, 
                                          query_dim=query_dim,
                                          num_feature_levels=self.num_feature_levels,
                                          dec_layer_share=dec_layer_share,
                                          )
        
        # define bbox regressor FFN
        self.hidden_dim = hidden_dim
        self._bbox_regressor = _bbox_regressor = MLP(hidden_dim, hidden_dim, 6, 3) 
        
        # set the last layer of the box prediction FFN to 0
        # implies decoder's first prediction will be exactly the initial reference point
        nn.init.constant_(_bbox_regressor.layers[-1].weight.data, 0)
        nn.init.constant_(_bbox_regressor.layers[-1].bias.data, 0)
        
        _bbox_regressor_layerlist = [_bbox_regressor for i in range(self.num_layers)]  # share box prediction each layer
        self.bbox_regressor = nn.ModuleList(_bbox_regressor_layerlist)
        self.decoder.bbox_regressor = self.bbox_regressor


    def generate_denoising_queries(self, targets, target_query_embeddings, reference_point_embeddings, batch_size):
        if self.training:
            total_denosing_queries, noise_scale = self.total_denosing_queries, self.noise_scale

            labels_per_image = [(torch.ones_like(t['labels'])).cuda() for t in targets] # (bs, num_known_per_image)
            label_indices_per_image = [torch.nonzero(t) for t in labels_per_image] # (bs, num_known_per_image) 
            num_labels_per_image = [sum(gt_labels) for gt_labels in labels_per_image] # (bs, )

            if max(num_labels_per_image) == 0:
                return None, None, None, None

            # fix number of denoising queries such that each label gets the same number of queries with total = total query num
            denoise_queries_per_label = total_denosing_queries // (int(max(num_labels_per_image))) 

            # binary mask indicating which GT boxes/labels should be used for denoising (currently all 1s)
            # can be modified to selectively denosie some labels or boxes (hence the overcomplicated logic)
            bboxes_denoise_index_mask = labels_denoise_index_mask = torch.cat(labels_per_image) # (total_num_labels, ) all 1s
            labels = torch.cat([t['labels'] for t in targets]) # (total num labels)
            bboxes = torch.cat([t['boxes'] for t in targets]) # (total num bboxes)
            
            # allows for assigning each DN query to the correct image in the batch
            batch_idx = torch.cat([torch.full_like(t['labels'].long(), i) for i, t in enumerate(targets)]) # [0] * num_labels_image1 + [1] * num_labels_image2 + ... 

            # roundabout way to get the num of labels and bboxes to denoise (legacy code)
            # DIMS: (total_num_objects_denoise,) = 2 x (total_num_labels)
            denoise_target_indices = torch.nonzero(bboxes_denoise_index_mask + labels_denoise_index_mask).view(-1)
            
            # generate denoise_queries_per_label num of queries for each denoise_item (copies to be populated) 
            denoise_target_indices = denoise_target_indices.repeat(denoise_queries_per_label, 1).view(-1)
            total_denoise_labels = labels.repeat(denoise_queries_per_label, 1).view(-1)
            denoise_query_batch_id = batch_idx.repeat(denoise_queries_per_label, 1).view(-1)
            total_denoise_bboxes = bboxes.repeat(denoise_queries_per_label, 1)
            
            denoise_target_indices_copy = denoise_target_indices.clone()
            total_denoise_bboxes_copy = total_denoise_bboxes.clone()

            # add noise to labels and bboxes
            # label ids are randomly assigned with prob p = noise_scale * 0.5
            # bbox coordinates are randomly shifted 
            if noise_scale > 0:
                label_flip_probs = torch.rand_like(denoise_target_indices_copy.float()) # uniform sampling [0,1]
                flipped_indices = torch.nonzero(label_flip_probs < (noise_scale * 0.5)).view(-1) # 50% chance to flip with max noise_scale = 1  
                new_label_ids = torch.randint_like(flipped_indices, 0, self.num_classes)  
                total_denoise_labels.scatter_(0, flipped_indices, new_label_ids)

                denoise_bbox_deltas = torch.zeros_like(total_denoise_bboxes_copy)
                denoise_bbox_deltas[:, :3] = total_denoise_bboxes_copy[:, 3:] / 2 # shift amount of dd, dh, dw 
                denoise_bbox_deltas[:, 3:] = total_denoise_bboxes_copy[:, 3:] # starting dd, dw, dh

                # randomly shift bbox coordinates ([-1,1] * bbox_deltas * noise_scale)
                total_denoise_bboxes_copy += torch.mul((torch.rand_like(total_denoise_bboxes_copy) * 2 - 1.0), denoise_bbox_deltas).cuda() * noise_scale
                total_denoise_bboxes_copy = total_denoise_bboxes_copy.clamp(min=0.0, max=1.0) # clamp new bbox coordinates to [0,1] range

            # embed/encode noised labels and bboxes
            total_denoise_label_embeddings = self.label_embeddings(total_denoise_labels.long().cuda())
            total_denoise_bboxes_encoded = inverse_sigmoid(total_denoise_bboxes_copy) # encode bboxes into sigmoid space
            
            # pad all denoising queries to the same size
            max_labels_per_image = int(max(num_labels_per_image))
            max_query_pad_size = int(max_labels_per_image * denoise_queries_per_label)

            # pad the number of denoise queries (labels per image * num queries per label) per image 
            # to the same size
            denoise_labels_padded = torch.zeros(max_query_pad_size, self.hidden_dim).cuda()
            denoise_bboxes_padded = torch.zeros(max_query_pad_size, 6).cuda()

            # combine the denoised labels and bboxes with target queries/bbox embeddings if they exist
            if reference_point_embeddings is not None:
                label_queries = torch.cat([denoise_labels_padded, target_query_embeddings], dim=0).repeat(batch_size, 1, 1) # (batch_size, num_dn + num_queries, d_model)
                bbox_queries = torch.cat([denoise_bboxes_padded, reference_point_embeddings], dim=0).repeat(batch_size, 1, 1) # (batch_size, num_dn + num_queries, 6)
            else:
                label_queries = denoise_labels_padded.repeat(batch_size, 1, 1) # (batch_size, num_dn, d_model)
                bbox_queries = denoise_bboxes_padded.repeat(batch_size, 1, 1) # (batch_size, num_dn, 6)

            # create mapping of 
            denoise_target_indices_map = torch.tensor([]).cuda()
            if len(num_labels_per_image) > 0:
                # For each image, create range [0, ..., num_labels_image_i-1] and cat them together
                denoise_target_indices_map = torch.cat([torch.tensor(range(num)) for num in num_labels_per_image])  
                # each group (for replication i of denoise queries) is assigned a slot of size labels_per_image 
                # hence shift the indices (elemnt values) by i × max_labels_per_image
                # to write to the correct row in the padded tensor we need to shift the indices by i * max_labels_per_image
                denoise_target_indices_map = torch.cat([denoise_target_indices_map + max_labels_per_image * query_num for query_num in range(denoise_queries_per_label)]).long()
            
            if len(denoise_query_batch_id) > 0:
                # batch ids are of form [0, 0, 1, 1, 1, ...] x N copies for N denoising queries per label
                # denoise_target_indices_map is of form [0, 1, 0, 1, 2, ..., max_labels_per_image, max_labels_per_image + 1, max_labels_per_image, ...] 
                # this way we assign the embeddings of the queries_per_label number of copies of flipped labels to position in matrix where  
                # previously we had vectors of zeros from denoise_labels_padded
                label_queries[(denoise_query_batch_id.long(), denoise_target_indices_map)] = total_denoise_label_embeddings
                bbox_queries[(denoise_query_batch_id.long(), denoise_target_indices_map)] = total_denoise_bboxes_encoded

            # num_queries regular queries and max_query_pad_size denoising queries for each image
            # attention mask will determine which queries can attend to which 
            # rules:  
            # regular queries cannot attend to denoising queries
            # denoising queries cannot attend to each other
            # denoising queries can only attend within their group
            atten_mask_size = max_query_pad_size + self.num_queries
            attn_mask = torch.ones(atten_mask_size, atten_mask_size).cuda() < 0 # all False so everyone can see everyone initially

            # regular queries cannot see the denoising queries
            attn_mask[max_query_pad_size:, :max_query_pad_size] = True

            # denoising queries cannot see each other
            for i in range(denoise_queries_per_label):
                if i == 0:
                    attn_mask[max_query_pad_size * i:max_query_pad_size * (i + 1), max_query_pad_size * (i + 1):max_query_pad_size] = True
                if i == denoise_queries_per_label - 1:
                    attn_mask[max_query_pad_size * i:max_query_pad_size * (i + 1), :max_query_pad_size * i] = True
                else:
                    attn_mask[max_query_pad_size * i:max_query_pad_size * (i + 1), max_query_pad_size * (i + 1):max_query_pad_size] = True
                    attn_mask[max_query_pad_size * i:max_query_pad_size * (i + 1), :max_query_pad_size * i] = True

            mask_dict = {
                'denoise_target_indices': torch.as_tensor(denoise_target_indices).long(),
                'batch_idx': torch.as_tensor(batch_idx).long(),
                'denoise_target_indices_map': torch.as_tensor(denoise_target_indices_map).long(),
                'known_lbs_bboxes': (total_denoise_labels, total_denoise_bboxes),
                'label_indices_per_image': label_indices_per_image,
                'max_query_pad_size': max_query_pad_size,
                'denoise_queries_per_label': denoise_queries_per_label,
            }

        else:
            if reference_point_embeddings is not None:
                label_queries = target_query_embeddings.repeat(batch_size, 1, 1)
                bbox_queries = reference_point_embeddings.repeat(batch_size, 1, 1)
            else:
                label_queries = None
                bbox_queries = None

            attn_mask = None
            mask_dict = None

        return label_queries, bbox_queries, attn_mask, mask_dict


    def denoise_post_process(self, outputs_class, outputs_bboxes, mask_dict, outputs_mask):
        assert mask_dict['max_query_pad_size'] > 0, "max_query_pad_size must be greater than 0"
        # first max_query_pad_size queries are the denoising queries
        # we return the non-denoising queries and store the denoising queries in the mask_dict
        outputs_class_dn = outputs_class[:, :, :mask_dict['max_query_pad_size'], :]
        outputs_class = outputs_class[:, :, mask_dict['max_query_pad_size']:, :]
        
        outputs_bboxes_dn = outputs_bboxes[:, :, :mask_dict['max_query_pad_size'], :] 
        outputs_bboxes = outputs_bboxes[:, :, mask_dict['max_query_pad_size']:, :]
        
        if outputs_mask is not None:
            outputs_mask_dn = outputs_mask[:, :, :mask_dict['max_query_pad_size'], :]
            outputs_mask = outputs_mask[:, :, mask_dict['max_query_pad_size']:, :]
        
        out = {'pred_logits': outputs_class_dn[-1], 'pred_boxes': outputs_bboxes_dn[-1], 'pred_masks': outputs_mask_dn[-1]}
        out['aux_outputs'] = self._set_aux_loss(outputs_class_dn, outputs_mask_dn, outputs_bboxes_dn)
        mask_dict['output_known_lbs_bboxes'] = out
        return outputs_class, outputs_bboxes, outputs_mask


    def pred_bbox(self, reference_pts_list, intermediates, initial_reference_pts = None):
        device = reference_pts_list[0].device

        if initial_reference_pts is None:
            outputs_bbox_list = []
        else:
            outputs_bbox_list = [initial_reference_pts.to(device)]

        # iterate over all decoder layers and, for each layer, update the reference points 
        for layer_id, (reference_pts, layer_bbox_regressor, intermediate) in enumerate(zip(reference_pts_list[:-1], self.bbox_regressor, intermediates)):
            layer_deltas = layer_bbox_regressor(intermediate).to(device)
            reference_pts_output = layer_deltas + inverse_sigmoid(reference_pts).to(device)
            outputs_bbox_list.append(reference_pts_output.sigmoid())
        
        outputs_bboxes = torch.stack(outputs_bbox_list)
        return outputs_bboxes
    

    def forward_prediction_heads(self, output, pixel_decoder_output, predict_masks = True):
        decoder_output = self.decoder_norm(output) 
        decoder_output = decoder_output.transpose(0, 1) # (bs, num_queries, C)
        outputs_class = self.class_predictor(decoder_output) # (bs, num_queries, num_classes)
        outputs_mask = None
        
        if predict_masks:
            mask_embeddings = self.mask_embeddings(decoder_output)
            outputs_mask = torch.einsum("bqc, bcdhw->bqdhw", mask_embeddings, pixel_decoder_output) # (bs, num_queries, d, h, w)

        return outputs_class, outputs_mask


    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_masks, out_bboxes = None):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        if out_bboxes is None:
            return [
                {"pred_logits": a, "pred_masks": b}
                for a, b in zip(outputs_class[:-1], outputs_masks[:-1])
            ]
        else:
            return [
                {"pred_logits": a, "pred_masks": b, "pred_boxes":c}
                for a, b, c in zip(outputs_class[:-1], outputs_masks[:-1], out_bboxes[:-1])
            ]


    def forward(self, x, pixel_decoder_output, masks, targets = None):
        assert len(x) == self.num_feature_levels, "The number of feature maps much match self.num_feature_levels"
        device = x[0].device

        # disable mask — it does not affect performance unless feature map is not divisible by 32
        if masks is None or not any(feature_map.size(2) % 32 or feature_map.size(3) % 32 or feature_map.size(4) % 32 for feature_map in x):
            masks = [
                torch.zeros((feature_map.size(0), feature_map.size(2), feature_map.size(3), feature_map.size(4)), 
                            device=feature_map.device, dtype=torch.bool) 
                            for feature_map in x
            ]

        x_flatten = []
        mask_flatten = []
        shapes = []
        # iterate over feature levels in reverse order
        for idx in range(self.num_feature_levels - 1, -1, -1):
            bs, c , d, h, w = x[idx].shape
            shapes.append(torch.as_tensor([d, h, w], dtype=torch.long, device=device))
            # (bs, c, d, h, w) -> (bs, d_model, d, h, w) → (bs, d_model, d×h×w) -> (bs, d×h×w, d_model)
            x_flatten.append(self.input_proj[idx](x[idx]).flatten(2).transpose(1, 2)) 
            mask_flatten.append(masks[i].flatten(1)) # (bs, d×h×w)
        
        # concatenate all feature levels
        x_flatten = torch.cat(x_flatten, 1)  # bs, \sum{d x h x w}, c
        mask_flatten = torch.cat(mask_flatten, 1)  # bs, \sum{d x h x w}
        shapes = torch.as_tensor(shapes, dtype=torch.long, device=x_flatten.device) # (num_feature_levels, 3)
        
        # get the start index of each feature level in token sequence [0, d1*h1*w1, d1*h1*w1 + d2*h2*w2, ...]
        level_start_index = torch.cat((shapes.new_zeros((1,)), shapes.prod(1).cumsum(0)[:-1])) # (num_feature_levels, )
        # get ratio mask vs unmasked volume for each feature level (padded, not real data vs real data)
        valid_ratios = torch.stack([compute_unmasked_ratio(m) for m in masks], 1) # (B, num_feature_levels, 3)

        predictions_class = []
        predictions_mask = []
        queries_learned_topk = None # is populated if learned query_features are used
        if self.two_stage_flag:
            # generate proposals for topk query selection
            output_memory, output_proposals = gen_encoder_output_proposals(x_flatten, mask_flatten, shapes)
            output_memory = self.encoder_output_norm(self.encoder_output_mlp(output_memory))
            
            # predict class logits and proposals from encoder memory 
            class_predictions = self.class_predictor(output_memory)
            # TODO: Check that is equivalent to the original implementation
            bbox_preds = self.bbox_regressor(output_memory) + output_proposals  # (bs, \sum{dxhxw}, 6) 
            
            # select topk proposals based on class prediction confidence (regardless of class)
            # returns: indices encoder memory
            topk_proposals = torch.topk(class_predictions.max(-1)[0], self.num_queries, dim=1)[1] # (B, k)
            
            bbox_preds_topk = torch.gather(bbox_preds, 
                                                   1, 
                                                   topk_proposals.unsqueeze(-1).repeat(1, 1, 6))  # select (B, k, 6) from bbox_preds
            bbox_preds_topk = bbox_preds_topk.detach()

            # select topk encoder memory
            # output[b, k, c] = input[b, index[b, k, c], c]
            output_memory_topk = torch.gather(output_memory, # input   
                                        1, # dim
                                        topk_proposals.unsqueeze(-1).repeat(1, 1, self.hidden_dim))  # select (B, k, d_model)

            outputs_class, outputs_mask = self.forward_prediction_heads(output_memory_topk.transpose(0, 1), pixel_decoder_output)
            
            output_memory_topk = output_memory_topk.detach()
            if self.learn_query_embeddings:
                # optionally override topk query features with learned query features
                queries_learned_topk = self.query_features.weight[None].repeat(bs, 1, 1) # (1, num_queries, d_model) -> (bs, num_queries, d_model)
            
            #  store initial predictions from the encoder
            intermediate_outputs=dict()
            intermediate_outputs['pred_logits'] = outputs_class
            intermediate_outputs['pred_boxes'] = bbox_preds_topk.sigmoid()
            intermediate_outputs['pred_masks'] = outputs_mask

            # optionally: initialize decoder box queries using predicted masks
            if self.initialize_box_type != 'no':
                # convert masks into boxes to better initialize boxes in the decoder
                assert self.initial_prediction_flag, "Initial prediction flag must be set to True"
                flatten_mask = outputs_mask.detach().flatten(0, 1) # (B * num_queries, D, H, W)
                d, h, w = outputs_mask.shape[-2:]
                
                if self.initialize_box_type == 'bitmask':  # slower, but more accurate
                    refpoint_embeddings = BitMasks(flatten_mask > 0).get_bounding_boxes().tensor.to(device)
                elif self.initialize_box_type == 'mask2box':  # faster 
                    refpoint_embeddings = box_ops.masks_to_boxes(flatten_mask > 0).to(device)
                else:
                    assert NotImplementedError, "Unknown box initialization type: {}".format(self.initialize_box_type)
                
                refpoint_embeddings = box_ops.box_xyzxyz_to_cxcyczwhd(refpoint_embeddings) / torch.as_tensor([w, h, d, w, h, d], # TODO: Double check this logic
                                                                                              dtype=torch.float).to(device)
                refpoint_embeddings = refpoint_embeddings.reshape(outputs_mask.shape[0], outputs_mask.shape[1], 6) # (B * num_queries, 6) -> (B, num_queries, 6)
                refpoint_embeddings = inverse_sigmoid(refpoint_embeddings)
        else:
            queries_learned_topk = self.query_features.weight[None].repeat(bs, 1, 1) # (1, num_queries, d_model) -> (bs, num_queries, d_model)
            refpoint_embeddings = self.query_embeddings.weight[None].repeat(bs, 1, 1) # (1, num_queries, 6) -> (bs, num_queries, 6)

        attn_mask = None
        mask_dict = None
        queries = queries_learned_topk if queries_learned_topk is not None else output_memory_topk
        # generate denoising queries if training and denoising queries flag is set
        if self.denoise_queries_flag != "no" and self.training:
            assert targets is not None, "If denoising queries are used, targets must be provided"
            # generates noisy copies of the ground-truth labels (fliped) and bboxes (shifted)
            label_queries, bbox_queries, attn_mask, mask_dict = self.generate_denoising_queries(targets, None, None, x[0].shape[0])
            
            # mask dict contains metadata for the denoising queries
            if mask_dict is not None: # TODO: Why set logic based on mask_dict?
                queries = torch.cat([label_queries, queries], dim=1) # (bs, num_dn + num_queries, d_model)

        if self.initial_prediction_flag:
            # predict class logits and masks from queries and pixel decoder output
            outputs_class, outputs_mask = self.forward_prediction_heads(queries.transpose(0, 1), pixel_decoder_output, self.training)
            predictions_class.append(outputs_class)
            predictions_mask.append(outputs_mask)
        
        if self.denoise_queries_flag != "no" and self.training and mask_dict is not None:
            refpoint_embeddings=torch.cat([bbox_queries, refpoint_embeddings], dim=1)

        intermediates, reference_points_list = self.decoder(
            target=queries.transpose(0, 1),
            memory=x_flatten.transpose(0, 1),
            memory_key_padding_mask=mask_flatten,
            pos_embeddings=None,
            reference_points=refpoint_embeddings.transpose(0, 1), # ()
            level_start_index=level_start_index,
            shapes=shapes,
            valid_ratios=valid_ratios,
            target_mask=attn_mask # target = queries in this case (standard nn.Transformer vs DETR nomenclature) 
        )

        num_intermediates = len(intermediates)
        for idx, output in enumerate(intermediates):
            outputs_class, outputs_mask = self.forward_prediction_heads(output.transpose(0, 1), 
                                                                        pixel_decoder_output, 
                                                                        self.training or (idx == num_intermediates - 1))
            predictions_class.append(outputs_class)
            predictions_mask.append(outputs_mask)

        # bbox prediction
        if self.initial_prediction_flag:
            # include refpoint embeddings in pred_box predictions
            # recall: refpoint_embeddings are given by encoded mask preds of encoder proposals
            # (+ optional denosing bboxes) or learned query embeddings
            out_bboxes = self.pred_bbox(reference_points_list, intermediates, refpoint_embeddings.sigmoid()) # (num_layers (+1 if init), B, num_queries, 4 or 6)
            assert len(predictions_class) == self.num_layers + 1, "predictions_class should be of size self.num_layers + 1"
        else:
            out_bboxes = self.pred_bbox(reference_points_list, intermediates)
        
        if mask_dict is not None:
            predictions_mask = torch.stack(predictions_mask) # (decode_layers, bs, num_queries + num_dn_queries, D, H, W)
            predictions_class = torch.stack(predictions_class) # (decode_layers, bs, num_queries + num_dn_queries, class_num)
            predictions_class, out_bboxes, predictions_mask = self.denoise_post_process(predictions_class, out_bboxes, mask_dict, predictions_mask)            
            predictions_class, predictions_mask = list(predictions_class), list(predictions_mask)
        elif self.training:  # ensure self.label_embeddings is marked as used for computation graph
            predictions_class[-1] += 0.0*self.label_embeddings.weight.sum()

        outputs = {
            'pred_logits': predictions_class[-1],
            'pred_masks': predictions_mask[-1],
            'pred_bboxes': out_bboxes[-1],
            'aux_outputs': self._set_aux_loss(
                predictions_class if self.mask_classification else None, predictions_mask, out_bboxes
            )
        }
        
        if self.two_stage_flag:
            outputs['intermediates'] = intermediate_outputs
        
        return outputs, mask_dict

################################################################################ OLD CODE ################################################################################
         
# disable mask, it does not affect performance
# enable_mask = 0
# if masks is not None:
#     for src in x:
#         if src.size(2) % 32 or src.size(3) % 32:
#             enable_mask = 1

# if enable_mask == 0:
#     masks = [torch.zeros((src.size(0), src.size(2), src.size(3)), device=src.device, dtype=torch.bool) for src in x]

################################################################################################################################################################