
# Minimal BoT-SORT tracker with embedding output (stub for integration)
import numpy as np
from collections import deque

class BOTrack:
    def __init__(self, bbox, track_id, embedding=None):
        self.bbox = bbox
        self.track_id = track_id
        self.embedding = embedding if embedding is not None else np.random.rand(128)

class BoTSORT:
    def __init__(self, track_buffer=30, match_thresh=0.8):
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        # In a real implementation, load ReID model and tracker state here

    def update(self, detections):
        # detections: np.ndarray of shape (N, 9) [x1, y1, x2, y2, score, class_id, color0, color1, color2]
        # This version uses color similarity for association
        if not hasattr(self, 'active_tracks'):
            self.active_tracks = []  # Each: [bbox, track_id, embedding, class_id, color, age]
            self.next_id = 1

        updated_tracks = []
        used_det = set()
        # Association: for each active track, find best matching detection by color (and optionally embedding)
        for track in self.active_tracks:
            best_idx = -1
            best_score = float('inf')
            for i, det in enumerate(detections):
                if i in used_det:
                    continue
                det_color = np.array([float(det[6]), float(det[7]), float(det[8])]) if det[6] is not None else np.array([0,0,0])
                track_color = np.array(track[4]) if track[4] is not None else np.array([0,0,0])
                color_dist = np.linalg.norm(det_color - track_color)
                # Optionally, add embedding distance here
                score = color_dist
                if score < best_score:
                    best_score = score
                    best_idx = i
            # Threshold for color match (tune as needed)
            color_thresh = 60.0  # Euclidean distance in RGB
            if best_idx != -1 and best_score < color_thresh:
                det = detections[best_idx]
                bbox = det[:4]
                embedding = np.random.rand(128)  # Replace with real ReID embedding
                color = [float(det[6]), float(det[7]), float(det[8])] if det[6] is not None else [0,0,0]
                updated_tracks.append([bbox, track[1], embedding, int(det[5]), color, 0])
                used_det.add(best_idx)
            else:
                # Track lost, do not add
                pass

        # Add unmatched detections as new tracks
        for i, det in enumerate(detections):
            if i in used_det:
                continue
            bbox = det[:4]
            embedding = np.random.rand(128)  # Replace with real ReID embedding
            color = [float(det[6]), float(det[7]), float(det[8])] if det[6] is not None else [0,0,0]
            updated_tracks.append([bbox, self.next_id, embedding, int(det[5]), color, 0])
            self.next_id += 1

        # Update active tracks
        self.active_tracks = updated_tracks

        # Return tracks in expected format (without age)
        return [[t[0], t[1], t[2], t[3], t[4]] for t in self.active_tracks]
