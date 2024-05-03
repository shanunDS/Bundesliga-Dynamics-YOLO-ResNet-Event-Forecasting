
import time
import os
import numpy as np
import pandas as pd
import cv2
import random
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import glob
from sklearn.metrics import confusion_matrix
import copy


IMG_SIZE = (640, 768)
# modelname = "tf_efficientnet_b0"
use_amp = True
batch_size = 40
n_epochs = 5
num_workers = 2
COSINE = True
init_lr = 1e-4
IMG_SOURCE = "img"
BACK_INTERVAL = 20
BACK_INTERVAL_VAL = 1
ERR_TOL = 1
mixup = True


selected_videos = [
    "1606b0e6_0.mp4",
    "35bd9041_0.mp4",
    "3c993bd2_0.mp4",
    '35bd9041_1.mp4',
    '407c5a9e_1.mp4'

]




err_tol = {
    'challenge': [0.30, 0.40, 0.50, 0.60, 0.70],
    'play': [0.15, 0.20, 0.25, 0.30, 0.35],
    'throwin': [0.15, 0.20, 0.25, 0.30, 0.35]
}





event_names = ['challenge', 'throwin', 'play']

df = pd.read_csv("/home/ubuntu/bundesliga/src/Data/train.csv")

print(len(df))

print(df.head(15))


additional_events = []
for arr in df.sort_values(['video_id', 'time', 'event', 'event_attributes']).values:
    if arr[2] in err_tol:
        tol = err_tol[arr[2]][ERR_TOL] / 2
        additional_events.append([arr[0], arr[1] - tol, 'start_' + arr[2], arr[3]])
        additional_events.append([arr[0], arr[1] + tol, 'end_' + arr[2], arr[3]])

for arr in df.sort_values(['video_id', 'time', 'event', 'event_attributes']).values:
    if arr[2] in err_tol:
        tol = err_tol[arr[2]][ERR_TOL] / 2
        additional_events.append([arr[0], arr[1] - tol * 2, 'pre_' + arr[2], arr[3]])
        additional_events.append([arr[0], arr[1] + tol * 2, 'post_' + arr[2], arr[3]])

df = pd.concat([df, pd.DataFrame(additional_events, columns=df.columns)])
df = df[~df['event'].isin(event_names)]
df = df.sort_values(['video_id', 'time'])


print('main dataset')

print(df.head())






cap = cv2.VideoCapture("/home/ubuntu/bundesliga/src/Data/train/3c993bd2_0.mp4")

fps = cap.get(cv2.CAP_PROP_FPS)

print("fps:", fps)
df["frame"] = df["time"] * fps

print(df.head(10))





def extract_images(video_path, out_dir):
    video_name = os.path.basename(video_path).split('.')[0]
    cam = cv2.VideoCapture(video_path)
    print(video_path)
    frame_count = 1
    while True:
        successed, img = cam.read()
        if not successed:
            break
        print('output_img_file')
        outfile = f'{out_dir}/{video_name}-{frame_count:06}.jpg'
        print(outfile)
        # print('img')
        img = cv2.resize(img, dsize=IMG_SIZE, interpolation=cv2.INTER_AREA)
        # print(img)
        cv2.imwrite(outfile, img)
        frame_count += 1





# Path to the directory containing the MP4 files
directory = "/home/ubuntu/bundesliga/src/Data/train"


OUT_DIR = "/home/ubuntu/bundesliga/src/work"


train_out_dir = os.path.join(OUT_DIR, "img_train")
val_out_dir = os.path.join(OUT_DIR, "img_val")


os.makedirs(train_out_dir, exist_ok=True)
os.makedirs(val_out_dir, exist_ok=True)

random.shuffle(selected_videos)


split_ratio = 0.8

# Split the videos into training and validation sets
split_index = int(len(selected_videos) * split_ratio)

print('train videos')
train_videos = selected_videos[:split_index]

print(train_videos)

print('validation videos')
valid_videos = selected_videos[split_index:]
print(valid_videos)

# Extract images from training videos
for video_file in train_videos:
    video_path = os.path.join(directory, video_file)
    extract_images(video_path, train_out_dir)

# Extract images from validation videos
for video_file in valid_videos:
    video_path = os.path.join(directory, video_file)
    extract_images(video_path, val_out_dir)






#
# for video_file in selected_videos:
#     video_path = os.path.join(directory, video_file)
#     extract_images(video_path, OUT_DIR)






