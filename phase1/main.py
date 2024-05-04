import numpy as np

from utils import read_video, save_video
from trackers import Tracker
import cv2
import numpy as np
from team_assigner import  TeamAssigner
from player_ball_assigner import PlayerBallAssigner

def main():
    # Read Video
    video_frames = read_video('/Users/vishnuarun/Desktop/DL/clips_output/08fd33_4.mp4')
    print(len(video_frames))
    # Initialize Tracker
    tracker = Tracker('/Users/vishnuarun/Desktop/DL/models/best.pt')

    tracks = tracker.get_object_tracks(video_frames,
                                       read_from_stub=True,
                                       stub_path='/Users/vishnuarun/Desktop/DL/stubs/track_stubs.pkl')


    # Interpolate Ball Positions
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])


    # Assign Player Teams
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[2],tracks.get('players', []))

    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num],
                                                 track['bbox'],
                                                 player_id)
            tracks['players'][frame_num][player_id]['team'] = team
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]


    # Assign Ball Aquisition
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



    # Save Video
    save_video(output_video_frames, 'clips_output/output.avi')

if __name__ == "__main__":
    main()

