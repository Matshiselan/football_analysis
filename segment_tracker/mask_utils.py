# segment_tracker/mask_utils.py

import numpy as np
import cv2

def clean_mask(mask, min_area=500):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )

    if num_labels <= 1:
        return mask

    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return (labels == largest).astype(np.uint8)


def get_bbox_from_mask(mask: np.ndarray):
    """
    Converts a binary mask to a bounding box [x1, y1, x2, y2]
    """
    if mask is None or mask.sum() == 0:
        return None

    mask = mask.astype(np.uint8)

    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None

    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()

    return [int(x1), int(y1), int(x2), int(y2)]
