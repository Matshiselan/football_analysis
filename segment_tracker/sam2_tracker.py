# segment_tracker/sam2_tracker.py

import numpy as np
import cv2
from sam2.sam2_video_predictor import SAM2VideoPredictor
from utils import get_bbox_from_mask
from utils import get_foot_position


class SAM2Tracker:
    def __init__(self, sam2_ckpt, device="cuda"):
        self.predictor = SAM2VideoPredictor.from_pretrained(
            sam2_ckpt,
            device=device
        )

    def initialize(self, frame, detections):
        """
        detections: list of (track_id, bbox)
        """
        self.predictor.reset()
        for track_id, bbox in detections:
            self.predictor.add_object(
                object_id=track_id,
                box=bbox,
                frame=frame
            )

    def track(self, frames):
        tracks = {
            "players": [],
            "referees": [],
            "ball": []
        }

        for f_idx, frame in enumerate(frames):
            masks = self.predictor.track(frame)

            players_frame = {}

            for obj_id, mask in masks.items():
                bbox = get_bbox_from_mask(mask)

                players_frame[obj_id] = {
                    "bbox": bbox,
                    "mask": mask
                }

            tracks["players"].append(players_frame)
            tracks["referees"].append({})
            tracks["ball"].append({})

        return tracks
