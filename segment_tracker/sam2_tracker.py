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

        for frame_idx, obj_ids, masks in self.predictor.propagate_in_video(self.state):

            players_frame = {}

            for obj_id, mask in zip(obj_ids, masks):
                mask = clean_mask(mask)
                bbox = get_bbox_from_mask(mask)

                if bbox is None:
                    continue

                players_frame[int(obj_id)] = {
                    "bbox": bbox,
                    "mask": mask
                }

            tracks["players"].append(players_frame)
            tracks["referees"].append({})
            tracks["ball"].append({})

        return tracks
