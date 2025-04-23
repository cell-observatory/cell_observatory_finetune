import random
from typing import List, Tuple, Optional

import skimage.io
import numpy as np


def plot_boxes(
    boxes: List[Tuple[float, float, float, float, float, float]],
    image_shape: Tuple[int, int, int],
    save_path: str,
    thickness: int = 1,
    sample_num: Optional[int] = None,
    sample_indices: Optional[List[int]] = None,
) -> np.ndarray:
    """
    Create plot with predicted bounding box edges drawn.
    
    Args:
        boxes (list): List of bounding boxes. Each box must have the format [x1, y1, z1, x2, y2, z2].
        image_shape (tuple): Desired shape of the output image as (depth, height, width).
        thickness (int): The thickness (in pixels) of the edges. Default is 1.
        sample_num (int): Number of boxes to sample from the list. If None, all boxes are used.
        sample_indices (list): Indices of boxes to sample from the list. If None, all boxes are used.
        save_path (str): File path where the resulting TIFF image is saved.
        
    Returns:
        np.ndarray: The resulting 3D image with box edges drawn.
    """
    output_img = np.zeros(image_shape, dtype=np.uint8)

    if sample_indices is not None:
        boxes = [boxes[idx] for idx in sample_indices]
    elif sample_num is not None and sample_num < len(boxes):
        boxes = random.sample(boxes, sample_num)

    for box in boxes:
        if len(box) != 6:
            print("Skipping invalid box (incorrect format):", box)
            continue
        
        try:
            x_min, y_min, z_min, x_max, y_max, z_max = [int(round(v)) for v in box]
        except Exception as e:
            print("Error converting box to integers:", box, e)
            continue

        # bottom edge
        output_img[z_min:z_max+1, y_min:y_min+thickness, x_min:x_max+1] = 1
        # top edge
        output_img[z_min:z_max+1, y_max:y_max+thickness, x_min:x_max+1] = 1

        # left edge
        output_img[z_min:z_max+1, y_min:y_max+1, x_min:x_min+thickness] = 1
        # right edge
        output_img[z_min:z_max+1, y_min:y_max+1, x_max:x_max+thickness] = 1

    skimage.io.imsave(save_path, 256*output_img)
    print(f"Saved output image with boxes to {save_path}")
    return output_img