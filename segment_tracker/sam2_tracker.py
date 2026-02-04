# segment_tracker/sam2_tracker.py

# segment_tracker/sam2_tracker.py

import numpy as np
import cv2
from sam2.sam2_video_predictor import SAM2VideoPredictor
from .mask_utils import clean_mask, get_bbox_from_mask


class SAM2Tracker:
    def __init__(self, model_id, device="cuda"):
        self.predictor = SAM2VideoPredictor.from_pretrained(
            model_id,
            device=device
        )
        self.state = None

    def initialize(self, video_path, detections):
        """
        video_path: path to MP4 video
        detections: list of (track_id, bbox) from FIRST frame
        """

        # ✅ SAM2 expects a VIDEO PATH
        self.state = self.predictor.init_state(video_path)

        # Add detected players
        for track_id, bbox in detections:
            self.predictor.add_object(
                self.state,
                object_id=int(track_id),
                box=bbox,
                frame_idx=0
            )

    def track(self):
        tracks = {
            "players": [],
            "referees": [],
            "ball": []
        }

        prev_bboxes = None
        prev_ids = None

        def bbox_iou(boxA, boxB):
            # box: [x1, y1, x2, y2]
            xA = max(boxA[0], boxB[0])
            yA = max(boxA[1], boxB[1])
            xB = min(boxA[2], boxB[2])
            yB = min(boxA[3], boxB[3])
            interArea = max(0, xB - xA) * max(0, yB - yA)
            boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
            boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
            iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
            return iou

        for frame_idx, obj_ids, masks in self.predictor.propagate_in_video(self.state):
            players_frame = {}
            curr_bboxes = []
            curr_ids = []
            curr_masks = []

            for obj_id, mask in zip(obj_ids, masks):
                mask = clean_mask(mask)
                bbox = get_bbox_from_mask(mask)
                if bbox is None:
                    continue
                curr_bboxes.append(bbox)
                curr_ids.append(int(obj_id))
                curr_masks.append(mask)

            # Post-process to reduce ID switches (simple IoU matching)
            if prev_bboxes is not None and prev_ids is not None:
                assigned = set()
                id_map = {}
                for i, bbox in enumerate(curr_bboxes):
                    best_iou = 0
                    best_j = -1
                    for j, prev_bbox in enumerate(prev_bboxes):
                        if j in assigned:
                            continue
                        iou = bbox_iou(bbox, prev_bbox)
                        if iou > best_iou:
                            best_iou = iou
                            best_j = j
                    # If IoU is high, keep previous ID
                    if best_iou > 0.5 and best_j != -1:
                        id_map[i] = prev_ids[best_j]
                        assigned.add(best_j)
                    else:
                        id_map[i] = curr_ids[i]
                # Build frame with mapped IDs
                for i, mask in enumerate(curr_masks):
                    players_frame[id_map[i]] = {
                        "bbox": curr_bboxes[i],
                        "mask": mask
                    }
            else:
                for i, mask in enumerate(curr_masks):
                    players_frame[curr_ids[i]] = {
                        "bbox": curr_bboxes[i],
                        "mask": mask
                    }

            tracks["players"].append(players_frame)
            tracks["referees"].append({})
            tracks["ball"].append({})

            prev_bboxes = curr_bboxes
            prev_ids = list(players_frame.keys())

        return tracks
