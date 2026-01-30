from ultralytics import YOLO
import supervision as sv
import pickle
import os
import numpy as np
import pandas as pd
import cv2
import sys

sys.path.append('../')
from utils import get_center_of_bbox, get_bbox_width, get_foot_position


from .botsort import BoTSORT

class Tracker:
    def __init__(self, model_path, tracker_type="bytetrack"):
        self.model = YOLO(model_path)
        if tracker_type == "botsort":
            self.tracker = BoTSORT()
        else:
            self.tracker = sv.ByteTrack()

    # -----------------------------------------------------------
    # POSITION ASSIGNMENT
    # -----------------------------------------------------------
    def add_position_to_tracks(self, tracks):
        for obj, obj_tracks in tracks.items():
            for frame_num, track in enumerate(obj_tracks):
                for track_id, track_info in track.items():
                    bbox = track_info['bbox']
                    if obj == 'ball':
                        position = get_center_of_bbox(bbox)
                    else:
                        position = get_foot_position(bbox)

                    tracks[obj][frame_num][track_id]['position'] = position

    # -----------------------------------------------------------
    # BALL INTERPOLATION
    # -----------------------------------------------------------
    def interpolate_ball_positions(self, ball_tracks):
        ball_positions = [x.get(1, {}).get('bbox', []) for x in ball_tracks]
        df = pd.DataFrame(ball_positions, columns=['x1', 'y1', 'x2', 'y2'])

        df = df.interpolate().bfill()
        ball_positions = [{1: {"bbox": row}} for row in df.to_numpy().tolist()]
        return ball_positions

    # -----------------------------------------------------------
    # BATCH FRAME DETECTION
    # -----------------------------------------------------------
    def detect_frames(self, frames, batch_size=20):
        detections = []
        for i in range(0, len(frames), batch_size):
            batch = self.model.predict(frames[i:i + batch_size], conf=0.1)
            detections += batch
        return detections

    # -----------------------------------------------------------
    # BUILD TRACK DICTIONARY
    # -----------------------------------------------------------
    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        # load stub
        if read_from_stub and stub_path and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                return pickle.load(f)

        detections = self.detect_frames(frames)

        tracks = {
            "players": [],
            "referees": [],
            "ball": []
        }

        for frame_num, det in enumerate(detections):

            # convert YOLO detection → Supervision object
            detection_sup = sv.Detections.from_ultralytics(det)
            cls_names = det.names
            cls_inv = {v: k for k, v in cls_names.items()}

            # convert goalkeepers → players
            for i, cls_id in enumerate(detection_sup.class_id):
                if cls_names[cls_id] == "goalkeeper":
                    detection_sup.class_id[i] = cls_inv["player"]


            # tracking
            if hasattr(self.tracker, "update_with_detections"):
                tracked = self.tracker.update_with_detections(detection_sup)
            else:
                # Convert detection_sup to numpy array for BoT-SORT
                # Example: [[x1, y1, x2, y2, score, class_id], ...]
                dets = []
                for i, bbox in enumerate(detection_sup.xyxy):
                    score = detection_sup.confidence[i] if hasattr(detection_sup, 'confidence') else 1.0
                    class_id = detection_sup.class_id[i]
                    dets.append([
                        bbox[0], bbox[1], bbox[2], bbox[3], score, class_id
                    ])
                dets = np.array(dets)
                tracked = self.tracker.update(dets)

            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            # tracked players & referees
            for tr in tracked:
                bbox = tr[0].tolist()
                cls_id = tr[3]
                track_id = tr[4]

                if cls_id == cls_inv['player']:
                    tracks["players"][frame_num][track_id] = {"bbox": bbox}

                if cls_id == cls_inv['referee']:
                    tracks["referees"][frame_num][track_id] = {"bbox": bbox}

            # ball (non-tracked)
            for det_sup in detection_sup:
                bbox = det_sup[0].tolist()
                cls_id = det_sup[3]
                if cls_id == cls_inv['ball']:
                    tracks["ball"][frame_num][1] = {"bbox": bbox}

        # save stub
        if stub_path:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)

        return tracks

    # -----------------------------------------------------------
    # DRAW ELLIPSE FOR PLAYER
    # -----------------------------------------------------------
    def draw_ellipse(self, frame, bbox, color, track_id=None):
        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)

        cv2.ellipse(
            frame,
            center=(x_center, y2),
            axes=(int(width), int(0.35 * width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color=color,
            thickness=2,
            lineType=cv2.LINE_4
        )

        # ID rectangle
        if track_id is not None:
            w, h = 40, 20
            x1 = x_center - w // 2
            y1 = y2 + 15 - h // 2
            x2 = x_center + w // 2
            y2_ = y2 + 15 + h // 2

            cv2.rectangle(frame, (x1, y1), (x2, y2_), color, cv2.FILLED)

            cv2.putText(
                frame,
                f"{track_id}",
                (x1 + 12, y1 + 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2
            )
        return frame

    # -----------------------------------------------------------
    # DRAW TRIANGLE (BALL / PLAYER POSSESSION)
    # -----------------------------------------------------------
    def draw_triangle(self, frame, bbox, color):
        y = int(bbox[1])
        x, _ = get_center_of_bbox(bbox)

        pts = np.array([
            [x, y],
            [x - 10, y - 20],
            [x + 10, y - 20]
        ])

        cv2.drawContours(frame, [pts], 0, color, cv2.FILLED)
        cv2.drawContours(frame, [pts], 0, (0, 0, 0), 2)
        return frame

    # -----------------------------------------------------------
    # SAFE TEAM CONTROL DRAW
    # -----------------------------------------------------------
    def draw_team_ball_control(self, frame, frame_num, team_control):
        """
        SAFE VERSION — Prevents ZeroDivisionError when no team touched the ball.
        """
        overlay = frame.copy()
        cv2.rectangle(overlay, (1350, 850), (1900, 970), (255, 255, 255), -1)
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

        control_up_to_now = team_control[:frame_num + 1]

        t1 = (control_up_to_now == 1).sum()
        t2 = (control_up_to_now == 2).sum()

        total = t1 + t2

        if total == 0:
            pct1 = pct2 = 0.0
        else:
            pct1 = t1 / total
            pct2 = t2 / total

        cv2.putText(frame,
                    f"Team 1 Ball Control: {pct1*100:.1f}%",
                    (1400, 900),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 0),
                    3)

        cv2.putText(frame,
                    f"Team 2 Ball Control: {pct2*100:.1f}%",
                    (1400, 950),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 0),
                    3)

        return frame

    # -----------------------------------------------------------
    # MAIN DRAW FUNCTION
    # -----------------------------------------------------------
    def draw_annotations(self, video_frames, tracks, team_control):
        out_frames = []

        for fnum, frame in enumerate(video_frames):
            frame = frame.copy()

            players = tracks["players"][fnum]
            refs = tracks["referees"][fnum]
            ball = tracks["ball"][fnum]

            # players
            for tid, info in players.items():
                frame = self.draw_ellipse(frame,
                                          info["bbox"],
                                          info.get("team_color", (0, 0, 255)),
                                          tid)
                if info.get("has_ball", False):
                    frame = self.draw_triangle(frame, info["bbox"], (0, 0, 255))

            # referees
            for _, info in refs.items():
                frame = self.draw_ellipse(frame, info["bbox"], (0, 255, 255))

            # ball
            for _, info in ball.items():
                frame = self.draw_triangle(frame, info["bbox"], (0, 255, 0))

            # safe draw
            frame = self.draw_team_ball_control(frame, fnum, team_control)

            out_frames.append(frame)

        return out_frames
