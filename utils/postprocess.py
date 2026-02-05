
import numpy as np
from scipy.optimize import linear_sum_assignment

def iou(bbox1, bbox2):
    xA = max(bbox1[0], bbox2[0])
    yA = max(bbox1[1], bbox2[1])
    xB = min(bbox1[2], bbox2[2])
    yB = min(bbox1[3], bbox2[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    boxBArea = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou

def smooth_player_ids(tracks, iou_thresh=0.5, window=5):
    """
    Post-process player tracks to relink IDs based on IoU and temporal continuity.
    Args:
        tracks: dict with 'players' key, list of dicts per frame {id: {...}}
        iou_thresh: IoU threshold for relinking
        window: number of previous frames to consider
    Returns:
        tracks with smoothed player IDs
    """
    new_tracks = []
    prev_ids = []
    prev_bboxes = []
    id_map = {}
    next_id = 1
    for frame_idx, player_dict in enumerate(tracks['players']):
        curr_bboxes = [pdata['bbox'] for pdata in player_dict.values()]
        curr_pdatas = list(player_dict.values())
        curr_ids = [None] * len(curr_bboxes)
        if prev_bboxes:
            cost_matrix = np.zeros((len(prev_bboxes), len(curr_bboxes)), dtype=np.float32)
            for i, prev_bbox in enumerate(prev_bboxes):
                for j, curr_bbox in enumerate(curr_bboxes):
                    cost_matrix[i, j] = 1 - iou(prev_bbox, curr_bbox)
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            assigned_prev = set()
            assigned_curr = set()
            for i, j in zip(row_ind, col_ind):
                if cost_matrix[i, j] < (1 - iou_thresh):
                    curr_ids[j] = prev_ids[i]
                    assigned_prev.add(i)
                    assigned_curr.add(j)
            # Assign new IDs to unassigned detections
            for j in range(len(curr_bboxes)):
                if curr_ids[j] is None:
                    curr_ids[j] = next_id
                    next_id += 1
        else:
            for j in range(len(curr_bboxes)):
                curr_ids[j] = next_id
                next_id += 1
        # Build new frame dict
        new_frame = {}
        for idx, pdata in enumerate(curr_pdatas):
            new_frame[curr_ids[idx]] = pdata.copy()
        new_tracks.append(new_frame)
        prev_ids = curr_ids
        prev_bboxes = curr_bboxes
    tracks['players'] = new_tracks
    return tracks
