import cv2
import os
import pandas as pd

from utils import measure_distance, get_foot_position


class SpeedAndDistance_Estimator:
    """
    Computes per-player cumulative distance and speed using perspective-transformed coordinates.
    Provides:
        - add_speed_and_distance_to_tracks(): updates the tracking dict
        - draw_speed_and_distance(): draws overlays on video frames
        - export_full_csv(): saves per-frame speed + distance
        - export_summary_csv(): saves one row per player (final distance, avg/max speed)
    """

    def __init__(self, frame_rate=24, frame_window=5):
        self.frame_rate = frame_rate       # video FPS
        self.frame_window = frame_window   # window for speed calculation

    # ---------------------------------------------------------------------
    # 1. COMPUTE SPEED + CUMULATIVE DISTANCE
    # ---------------------------------------------------------------------
    def add_speed_and_distance_to_tracks(self, tracks):
        """
        Adds speed (km/h) and cumulative distance (meters) into each track entry.
        """

        total_distance = {}   # {object: {track_id: cumulative_dist}}

        for object_name, object_tracks in tracks.items():
            if object_name in ["ball", "referees"]:
                continue

            num_frames = len(object_tracks)

            for start_frame in range(0, num_frames, self.frame_window):
                end_frame = min(start_frame + self.frame_window, num_frames - 1)

                frame_tracks_start = object_tracks[start_frame]
                frame_tracks_end = object_tracks[end_frame]

                for track_id in frame_tracks_start.keys():

                    # skip players not present in the end-frame
                    if track_id not in frame_tracks_end:
                        continue

                    start_pos = frame_tracks_start[track_id].get("position_transformed", None)
                    end_pos = frame_tracks_end[track_id].get("position_transformed", None)

                    if start_pos is None or end_pos is None:
                        continue

                    # --- distance + speed ---
                    distance_m = measure_distance(start_pos, end_pos)
                    time_s = (end_frame - start_frame) / self.frame_rate
                    if time_s == 0:
                        continue

                    speed_mps = distance_m / time_s
                    speed_kmh = speed_mps * 3.6

                    # --- initialize dict for team type ---
                    if object_name not in total_distance:
                        total_distance[object_name] = {}

                    # --- initialize and update cumulative distance ---
                    if track_id not in total_distance[object_name]:
                        total_distance[object_name][track_id] = 0

                    total_distance[object_name][track_id] += distance_m

                    # --- update all frames in this window ---
                    for f in range(start_frame, end_frame + 1):
                        if track_id not in object_tracks[f]:
                            continue

                        object_tracks[f][track_id]["speed"] = speed_kmh
                        object_tracks[f][track_id]["distance"] = total_distance[object_name][track_id]

    # ---------------------------------------------------------------------
    # 2. DRAW OVERLAYS ON VIDEO
    # ---------------------------------------------------------------------
    def draw_speed_and_distance(self, frames, tracks):
        """
        Draws speed (km/h) and cumulative distance on every frame.
        """
        output_frames = []

        for f_idx, frame in enumerate(frames):
            for object_name, obj_tracks in tracks.items():
                if object_name in ["ball", "referees"]:
                    continue

                for _, info in obj_tracks[f_idx].items():
                    if "speed" not in info or "distance" not in info:
                        continue

                    speed = info["speed"]
                    distance = info["distance"]

                    bbox = info["bbox"]
                    pos = list(get_foot_position(bbox))
                    pos[1] += 40
                    pos = tuple(map(int, pos))

                    cv2.putText(frame, f"{speed:.2f} km/h", pos,
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                    cv2.putText(frame, f"{distance:.2f} m", (pos[0], pos[1] + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

            output_frames.append(frame)

        return output_frames

    # ---------------------------------------------------------------------
    # 3. EXPORT FULL PER-FRAME DATA
    # ---------------------------------------------------------------------
    def export_full_csv(self, tracks, output_folder="output_videos"):
        """
        Exports a full CSV containing all frame-by-frame statistics.
        """
        all_rows = []

        for object_name, object_tracks in tracks.items():
            if object_name in ["ball", "referees"]:
                continue

            for frame_num in range(len(object_tracks)):
                for track_id, info in object_tracks[frame_num].items():
                    if "speed" in info and "distance" in info:
                        all_rows.append({
                            "object_type": object_name,
                            "track_id": track_id,
                            "frame_num": frame_num,
                            "speed_kmh": info["speed"],
                            "total_distance_m": info["distance"]
                        })

        if not all_rows:
            print("No speed/distance data found. CSV skipped.")
            return None

        df = pd.DataFrame(all_rows)
        os.makedirs(output_folder, exist_ok=True)
        path = os.path.join(output_folder, "full_speed_distance.csv")
        df.to_csv(path, index=False)
        print(f"Full per-frame CSV exported: {path}")
        return path

    # ---------------------------------------------------------------------
    # 4. EXPORT FINAL SUMMARY (ONE ROW PER PLAYER)
    # ---------------------------------------------------------------------
    def export_summary_csv(self, tracks, output_folder="output_videos"):
        """
        Exports summary statistics for each tracked player:
            - final cumulative distance
            - average speed
            - maximum speed
        """

        summary = {}

        for object_name, object_tracks in tracks.items():
            if object_name in ["ball", "referees"]:
                continue

            num_frames = len(object_tracks)

            for f in range(num_frames):
                for track_id, info in object_tracks[f].items():

                    if "distance" not in info:
                        continue

                    if track_id not in summary:
                        summary[track_id] = {
                            "object_type": object_name,
                            "track_id": track_id,
                            "final_distance": 0,
                            "speed_values": []
                        }

                    # update cumulative distance
                    summary[track_id]["final_distance"] = info["distance"]

                    # store speed history
                    if "speed" in info:
                        summary[track_id]["speed_values"].append(info["speed"])

        # Convert to CSV-ready table
        rows = []
        for track_id, s in summary.items():
            speeds = s["speed_values"]

            rows.append({
                "object_type": s["object_type"],
                "track_id": track_id,
                "final_distance_m": s["final_distance"],
                "avg_speed_kmh": sum(speeds) / len(speeds) if speeds else 0,
                "max_speed_kmh": max(speeds) if speeds else 0
            })

        df = pd.DataFrame(rows)
        os.makedirs(output_folder, exist_ok=True)
        path = os.path.join(output_folder, "player_summary_stats.csv")
        df.to_csv(path, index=False)
        print(f"Player summary CSV exported: {path}")
        return path
