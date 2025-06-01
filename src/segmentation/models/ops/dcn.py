from segmentation.models.ops.dcnv4 import DCNv4


def DCN(channels=64,
        kernel_size=3,
        stride=1,
        pad=1,
        dilation=1,
        group=4,
        offset_scale=1.0,
        dw_kernel_size=None,
        center_feature_scale=False,
        remove_center=False,
        output_bias=True,
        without_pointwise=False,
        norm_layer='LN',
        act_layer='GELU',
        use_dcn_v4_op=False,):
    if use_dcn_v4_op:
        return DCNv4(channels=channels,
                     kernel_size=kernel_size,
                     stride=stride,
                     pad=pad,
                     dilation=dilation,
                     group=group,
                     offset_scale=offset_scale,
                     dw_kernel_size=dw_kernel_size,
                     center_feature_scale=center_feature_scale,
                     remove_center=remove_center,
                     output_bias=output_bias,
                     without_pointwise=without_pointwise)
    else:
        raise NotImplementedError("DCNv3 is not implemented yet. Please use DCNv4 instead.")