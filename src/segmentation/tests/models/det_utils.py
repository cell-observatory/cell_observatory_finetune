from segmentation.models.rpn import det_utils

import torch


def  main():
    # example predicted deltas in encoded box format:
    # [dx, dy, dz, dw, dh, dl] where dx = delta centre x, dy = delta centre y, dz = delta centre z,
    # means => move centre 0 units and move width 0.1 units in x, height 0.1 units in y, depth 0.1 units in z (encoded space)
    # equivalent to approx. 5 units in x, 5 units in y, 5 units in z (in original space) since exp(0.1) = 1.1
    # and 1.1 * 50 = 55 - 50 = 5
    example_box_deltas = torch.tensor([[0.0, 0.0, 0.0, 0.1, 0.1, 0.1]])

    example_anchor = torch.tensor([[0.0, 0.0, 0.0, 100.0, 100.0, 100.0]])

    box_coder = det_utils.BoxCoder(weights=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0))
    
    print(f"Example box deltas shape: {example_box_deltas.shape}")
    print(f"Example anchor shape: {example_anchor.shape}")

    # decoder will do
    # 1. dx * x_width + ctr_x (new pred centre x after apply dx i.e. delta centre x), ... (same for y, z) 
    # 2. exp(dw) * width_x * 0.5 (new pred width from centre from
    # sigmoid space delta w applied to half width), ... (same for y, z) 
    # 3. pred_ctr_x +- c_to_c_w => predicted new box coords for x, ... (same for y, z)
    encoded_box = box_coder.decode(example_box_deltas, [example_anchor])
    # should return back: [-5, -5, -5, 105, 105, 105]
    print(f"Encoded Box: {encoded_box}")

    decoded_box = box_coder.encode([encoded_box[0]], [example_anchor])
    # should return back: [0.0, 0.0, 0.0, 1.0, 1.0, 1.0] 
    print(f"Decoded Box: {decoded_box}")


if __name__ == "__main__":
    main()