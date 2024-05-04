import cv2
import numpy as np
import tempfile
import os

def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    return frames

def save_video(output_video_frames, output_video_path):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, 24, (output_video_frames[0].shape[1], output_video_frames[0].shape[0]))
    for frame in output_video_frames:
        out.write(frame)
    out.release()

# # def save_video(output_video_frames, output_buffer):
# #     frames = []
# #     for frame in output_video_frames:
# #         # Encode the frame as a JPEG image
# #         success, encoded_frame = cv2.imencode('.jpg', frame)
# #         if success:
# #             frames.append(encoded_frame.tobytes())
# #         else:
# #             print("Failed to encode frame")
# #
# #     # Concatenate all the encoded frames into a single byte array
# #     output_buffer.write(b''.join(frames))
#
#
# def save_video(output_video_frames, output_buffer):
#     # Create a temporary file
#     with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
#         fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#         height, width, _ = output_video_frames[0].shape
#         out = cv2.VideoWriter(temp_file.name, fourcc, 24, (width, height))
#
#         for frame in output_video_frames:
#             out.write(frame)
#
#         out.release()
#
#         # Read the temp file into the output buffer
#         temp_file.seek(0)
#         output_buffer.write(temp_file.read())
#
#     # Remove the temporary file
#     os.remove(temp_file.name)