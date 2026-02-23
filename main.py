import os
import cv2
import numpy as np
import pandas as pd

from utils import read_video, save_video
from trackers import Tracker
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
from speed_and_distance_estimator import SpeedAndDistance_Estimator


def main():

    # ----------------------------
    # INPUTS
    # ----------------------------
    video_path = 'input_videos/08fd33_4.mp4'
    weights_path = 'models/best.pt'
    # video_path = '/content/drive/MyDrive/Computer Vision/input_videos/08fd33_4.mp4'
    # weights_path = '/content/drive/MyDrive/Computer Vision/models/best.pt'

    output_dir = "output_videos"
    os.makedirs(output_dir, exist_ok=True)

    # ----------------------------
    # READ VIDEO
    # ----------------------------
    video_frames = read_video(video_path)

    # ----------------------------
    # TRACKING
    # ----------------------------

    # Use BoT-SORT with ReID enabled by passing a custom botsort.yaml config
    # Create a custom botsort.yaml if not already present, with 'with_reid: True'
    botsort_yaml = "models/botsort_reid.yaml"
    # Example botsort_reid.yaml content:
    # tracker_type: botsort
    # with_reid: True
    # model: auto
    # (other parameters can be set as needed)

    tracker = Tracker(weights_path, tracker_type="botsort")

    tracks = tracker.get_object_tracks(
        video_frames,
        read_from_stub=True,
        stub_path='stubs/track_stubs.pkl',
        tracker_yaml=botsort_yaml
    )



    tracker.add_position_to_tracks(tracks)

    # Robust tracked player continuity: maintain tracking even if ID changes
    tracked_id = tracker.tracked_player_id
    prev_embedding = None
    lost_counter = 0
    max_lost = 10  # frames to interpolate if lost
    tracked_log = []  # For debug visualization
    last_known_bbox = None
    for frame_num, player_track in enumerate(tracks["players"]):
        log_entry = {"frame": frame_num, "status": "", "tracked_id": tracked_id}
        if tracked_id in player_track:
            tracks["players"][frame_num] = {tracked_id: player_track[tracked_id]}
            bbox = player_track[tracked_id]["bbox"]
            last_known_bbox = bbox
            prev_embedding = player_track[tracked_id].get("embedding")
            lost_counter = 0
            log_entry["status"] = "tracked"
        else:
            if len(player_track) == 0:
                tracks["players"][frame_num] = {}
                log_entry["status"] = "no players"
                tracked_log.append(log_entry)
                continue
            prev_bbox = last_known_bbox
            best_score = float('inf')
            best_id = None
            for pid, info in player_track.items():
                curr_bbox = info["bbox"]
                curr_embedding = info.get("embedding")
                curr_color = info.get("jersey_color")
                # Print jersey color for debugging
                print(f"Frame {frame_num}, Player {pid}, Jersey Color: {curr_color}")
                # Calculate embedding distance if available
                if prev_embedding is not None and curr_embedding is not None:
                    emb_dist = np.linalg.norm(np.array(prev_embedding) - np.array(curr_embedding))
                else:
                    emb_dist = None
                # Calculate bbox distance if available
                if prev_bbox is not None:
                    prev_center = ((prev_bbox[0]+prev_bbox[2])/2, (prev_bbox[1]+prev_bbox[3])/2)
                    curr_center = ((curr_bbox[0]+curr_bbox[2])/2, (curr_bbox[1]+curr_bbox[3])/2)
                    bbox_dist = np.linalg.norm(np.array(prev_center) - np.array(curr_center))
                else:
                    bbox_dist = None
                # Calculate jersey color distance if available
                prev_color = None
                if tracked_id in player_track and player_track[tracked_id].get("jersey_color") is not None:
                    prev_color = player_track[tracked_id]["jersey_color"]
                elif frame_num > 0 and tracked_id in tracks["players"][frame_num-1] and tracks["players"][frame_num-1][tracked_id].get("jersey_color") is not None:
                    prev_color = tracks["players"][frame_num-1][tracked_id]["jersey_color"]
                color_dist = np.linalg.norm(np.array(prev_color) - np.array(curr_color)) if prev_color is not None and curr_color is not None else None
                # Use color as a hard constraint: only consider matches if color distance is below threshold
                color_hard_thresh = 40.0  # Lower threshold for stricter color matching
                if color_dist is not None and color_dist > color_hard_thresh:
                    continue  # Skip this candidate if color is too different
                # Combine distances: increase color weight
                score = float('inf')
                if emb_dist is not None and bbox_dist is not None and color_dist is not None:
                    score = 0.3 * emb_dist + 0.2 * (bbox_dist / 100.0) + 0.5 * (color_dist / 100.0)
                elif emb_dist is not None and color_dist is not None:
                    score = 0.5 * emb_dist + 0.5 * (color_dist / 100.0)
                elif emb_dist is not None and bbox_dist is not None:
                    score = 0.7 * emb_dist + 0.3 * (bbox_dist / 100.0)
                elif color_dist is not None:
                    score = color_dist / 100.0
                elif emb_dist is not None:
                    score = emb_dist
                elif bbox_dist is not None:
                    score = bbox_dist / 100.0
                if score < best_score:
                    best_score = score
                    best_id = pid
            # Thresholds for lost: tune as needed
            emb_thresh = 0.6
            bbox_thresh = 60.0 / 100.0  # 60 pixels, scaled
            color_thresh = 40.0 / 100.0  # 40 in RGB, scaled (stricter)
            score_thresh = 0.3 * emb_thresh + 0.2 * bbox_thresh + 0.5 * color_thresh
            if best_score < score_thresh:
                tracked_id = best_id
                tracks["players"][frame_num] = {tracked_id: player_track[tracked_id]}
                bbox = player_track[tracked_id]["bbox"]
                last_known_bbox = bbox
                prev_embedding = player_track[tracked_id].get("embedding")
                lost_counter = 0
                log_entry["status"] = "recovered"
            else:
                # Mark as lost, interpolate bbox if possible
                lost_counter += 1
                if last_known_bbox is not None and lost_counter <= max_lost:
                    # Interpolate: keep last known bbox
                    tracks["players"][frame_num] = {tracked_id: {"bbox": last_known_bbox}}
                    log_entry["status"] = f"lost({lost_counter})"
                else:
                    tracks["players"][frame_num] = {}
                    log_entry["status"] = "lost(terminated)"
        tracked_log.append(log_entry)
    # Save debug log to CSV
    import csv
    with open(os.path.join(output_dir, "tracking_debug_log.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["frame", "status", "tracked_id"])
        writer.writeheader()
        writer.writerows(tracked_log)

    # ----------------------------
    # CAMERA MOVEMENT
    # ----------------------------
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])

    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(
        video_frames,
        read_from_stub=True,
        stub_path='stubs/camera_movement_stub.pkl'
    )

    camera_movement_estimator.add_adjust_positions_to_tracks(
        tracks, camera_movement_per_frame
    )

    # ----------------------------
    # VIEW TRANSFORMATION
    # ----------------------------
    view_transformer = ViewTransformer()
    view_transformer.add_transformed_position_to_tracks(tracks)

    # ----------------------------
    # FIX BALL (INTERPOLATE)
    # ----------------------------
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

    # ----------------------------
    # SPEED & DISTANCE
    # ----------------------------
    speed_and_distance_estimator = SpeedAndDistance_Estimator()
    speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)

    # ----------------------------
    # EXPORT CSVs (FULL + SUMMARY)
    # ----------------------------
    speed_and_distance_estimator.export_full_csv(
        tracks, output_folder=output_dir
    )

    speed_and_distance_estimator.export_summary_csv(
        tracks, output_folder=output_dir, tracked_player_id=tracker.tracked_player_id
    )

    print("CSV files saved inside:", output_dir)

    # ----------------------------
    # TEAM ASSIGNMENT
    # ----------------------------
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(
        video_frames[0], tracks['players'][0]
    )

    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(
                video_frames[frame_num],
                track["bbox"],
                player_id
            )
            track["team"] = team
            track["team_color"] = team_assigner.team_colors[team]

    # ----------------------------
    # BALL OWNERSHIP
    # ----------------------------
    player_assigner = PlayerBallAssigner()
    team_ball_control = []

    for frame_num, player_track in enumerate(tracks["players"]):
        ball_bbox = tracks["ball"][frame_num][1]["bbox"]
        assigned_player = player_assigner.assign_ball_to_player(
            player_track, ball_bbox
        )

        if assigned_player != -1:
            player_track[assigned_player]["has_ball"] = True
            team_ball_control.append(player_track[assigned_player]["team"])
        else:
            team_ball_control.append(team_ball_control[-1] if team_ball_control else 0)

    team_ball_control = np.array(team_ball_control)

    # ----------------------------
    # DRAW ANNOTATIONS
    # ----------------------------
    output_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control)
    output_frames = camera_movement_estimator.draw_camera_movement(output_frames, camera_movement_per_frame)
    speed_and_distance_estimator.draw_speed_and_distance(output_frames, tracks)

    # ----------------------------
    # SAVE OUTPUT VIDEO
    # ----------------------------
    output_video_path = os.path.join(output_dir, "output_video.avi")
    save_video(output_frames, output_video_path)

    print(f"\n🎉 Video saved to: {output_video_path}")
    print("🚀 Processing complete!")


if __name__ == "__main__":
    main()