from segmentation.models.rpn.anchor_generator import AnchorGenerator


def main():
    anchor_gen_sizes = ((2,),)
    anchor_gen_aspect_ratios = ((1.0,),)
    anchor_gen_aspect_ratios_z = ((1.0,),)

    # assuming original image size is also 2x2x2
    grid_sizes = [(2,2,2)] # feature map size
    strides = [(1,1,1)] # stride for feature map, i.e. image size / feature map size

    # Create an instance of AnchorGenerator
    # and generate anchors
    anchor_generator = AnchorGenerator(sizes=anchor_gen_sizes,
                                       aspect_ratios=anchor_gen_aspect_ratios,
                                       aspect_ratios_z=anchor_gen_aspect_ratios_z)
    
    print(f"Anchor generator self.anchors: {anchor_generator.cell_anchors}")

    # expectation: 
    # 0. given stride of 4 and feature map size of 64x64x64
    # 1. generate grid of anchors at index [0 4 8 12 ...] for each dim in original image 
    # 2. generate anchors at each grid point of size 32x32x32 (only one size for simplicity)
    # 3. example for index (64,64,64) in original image (or (16,16,16) in feature map):
    # 4. (32,32,32,96,96,96)
    anchors = anchor_generator.grid_anchors(grid_sizes, strides)

    print(f"Anchors shape: ", anchors[0].shape)
    print("Generated anchors:")
    for i, anchor in enumerate(anchors[0]):
        print(f"Anchor {i}:")
        print(anchor)

    # should return one box for each location in 2x2x2 grid (i.e. 8 boxes)


if __name__ == "__main__":
    main()