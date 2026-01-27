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

def get_bbox_from_mask(mask):
    ys, xs = np.where(mask > 0)

    if len(xs) == 0:
        return None

    return [xs.min(), ys.min(), xs.max(), ys.max()]
