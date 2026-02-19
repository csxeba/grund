import numpy as np


def is_neighbour(entity, other, diag=True):
    thresh = np.sqrt(2.0) if diag else 1.0
    return np.linalg.norm(entity.position - other.position) <= thresh


def boxes_overlap(box_x0y0x1y1: np.ndarray, other_boxes: np.ndarray) -> np.ndarray:
    """
    :param box_x0y0x1y1: np.ndarray[integral | float], shape: [4]
        Box to be checked if it overlaps with existing boxes. Represented as top left, bottom right
    :param other_boxes: np.ndarray[integral | float], shape: [N, 4]
        Existing array of boxes. Represented as N top left, bottom right.
    :return: np.ndarray[bool], shape: [N]
        True, where the box overlapped.
    """
    assert box_x0y0x1y1.ndim == 1
    assert other_boxes.ndim == 2
    assert box_x0y0x1y1.shape[0] == other_boxes.shape[1] == 4

    overlaps = np.logical_or.reduce(
        [
            box_x0y0x1y1[2] >= other_boxes[:, 0],  # rx < lx
            box_x0y0x1y1[0] <= other_boxes[:, 2],  # lx > rx
            box_x0y0x1y1[1] >= other_boxes[:, 3],  # ty < by
            box_x0y0x1y1[3] <= other_boxes[:, 1],  # by > ty
        ],
        axis=0,
    )

    return overlaps
