from utils import read_video, save_video
from trackers import Tracker
import numpy as np
from team_assigner import  TeamAssigner
from player_ball_assigner import PlayerBallAssigner
import streamlit as st
from io import BytesIO

def process_video(video_path):
    # Read Video
    video_frames = read_video(video_path)

    # Initialize Tracker
    tracker = Tracker('models/best.pt')

    tracks = tracker.get_object_tracks(video_frames,
                                       read_from_stub=True,
                                       stub_path='stubs/track_stubs.pkl')

    # Interpolate Ball Positions
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

    # Assign Player Teams
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], tracks['players'][0])

    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num],
                                                 track['bbox'],
                                                 player_id)
            tracks['players'][frame_num][player_id]['team'] = team
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]

    # Assign Ball Acquisition
    player_assigner = PlayerBallAssigner()
    team_ball_control = []
    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)

        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
        else:
            team_ball_control.append(team_ball_control[-1])
    team_ball_control = np.array(team_ball_control)

    # Draw output
    ## Draw object Tracks
    output_video_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control)

    return output_video_frames

def main():
    st.title("Video Tracking App")

    # File uploader
    uploaded_file = st.file_uploader("Upload a video file", type=["mp4"])

    if uploaded_file is not None:
        st.text("Processing... This might take a while depending on the video length.")

        uploaded_file = "C:/Users/kalya/OneDrive/Desktop/Project_DL/DL/clips/08fd33_4.mp4"

        # Process the video
        output_frames = process_video(uploaded_file)

        output_buffer = BytesIO()
        save_video(output_frames, output_buffer)

        # Provide download link
        st.text("Video processed successfully. You can download the output file below.")
        st.download_button(label="Download Output Video", data=output_buffer.getvalue(), file_name="output123")

if __name__ == "__main__":
    main()