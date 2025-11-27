from utils import read_video, save_video
from trackers import Tracker
import cv2
import numpy as np
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
from speed_and_distance_estimator import SpeedAndDistance_Estimator
import os
import pandas as pd

def main():
    # ----------------------------
    # INPUTS
    # ----------------------------
    video_path = '/content/drive/MyDrive/Computer Vision/input_videos/08fd33_4.mp4'
    weights_path = '/content/drive/MyDrive/Computer Vision/models/best.pt'

    # Output folder
    output_dir = "output_videos"
    os.makedirs(output_dir, exist_ok=True)

    # ----------------------------
    # READ VIDEO
    # ----------------------------
    video_frames = read_video(video_path)

    # ----------------------------
    # TRACKING
    # ----------------------------
    tracker = Tracker(weights_path)

    tracks = tracker.get_object_tracks(
        video_frames,
        read_from_stub=True,
        stub_path='stubs/track_stubs.pkl'
    )

    tracker.add_position_to_tracks(tracks)

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
    # INTERPOLATE BALL
    # ----------------------------
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

    # ----------------------------
    # SPEED & DISTANCE
    # ----------------------------
    speed_and_distance_estimator = SpeedAndDistance_Estimator()
    speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)

    # ----------------------------
    # SAVE SPEED/DISTANCE CSV
    # ----------------------------
    csv_path = os.path.join(output_dir, "player_speed_distance.csv")
    speed_and_distance_estimator.export_to_csv(tracks, csv_path)
    print(f"CSV saved to: {csv_path}")

    # ----------------------------
    # TEAM ASSIGNMENT
    # ----------------------------
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], tracks['players'][0])

    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(
                video_frames[frame_num],
                track['bbox'],
                player_id
            )
            tracks['players'][frame_num][player_id]['team'] = team
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]

    # ----------------------------
    # BALL OWNERSHIP
    # ----------------------------
    player_assigner = PlayerBallAssigner()
    team_ball_control = []
    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)

        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
        else:
            team_ball_control.append(team_ball_control[-1] if len(team_ball_control) > 0 else 0)

    team_ball_control = np.array(team_ball_control)

    # ----------------------------
    # DRAW ANNOTATIONS
    # ----------------------------
    output_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control)

    output_frames = camera_movement_estimator.draw_camera_movement(
        output_frames, camera_movement_per_frame
    )

    speed_and_distance_estimator.draw_speed_and_distance(output_frames, tracks)

    # ----------------------------
    # SAVE OUTPUT VIDEO
    # ----------------------------
    output_video_path = os.path.join(output_dir, "output_video.avi")
    save_video(output_frames, output_video_path)
    print(f"Video saved to: {output_video_path}")

    print("\n🚀 Processing complete!")

if __name__ == '__main__':
    main()
