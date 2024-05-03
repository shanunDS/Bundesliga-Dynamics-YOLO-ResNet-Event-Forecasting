
import time
import os
import numpy as np
import pandas as pd
import cv2
import random


import glob
from sklearn.metrics import confusion_matrix
import copy



IMSIZE = [540, 768]
IMG_SIZE = (540, 768)
use_amp = True
batch_size = 40
n_epochs = 30
num_workers = 2
COSINE = True
init_lr = 1e-4

IMG_SOURCE = "img"
BACK_INTERVAL = 20
BACK_INTERVAL_VAL = 1
ERR_TOL = 1
mixup = True





event_names = ['challenge', 'throwin', 'play']


err_tol = {
    'challenge': [0.30, 0.40, 0.50, 0.60, 0.70],
    'play': [0.15, 0.20, 0.25, 0.30, 0.35],
    'throwin': [0.15, 0.20, 0.25, 0.30, 0.35]
}




# df = pd.read_csv("/home/ubuntu/DL/src/Data/train.csv")


df = pd.read_csv("/home/ubuntu/bundesliga/src/Data/train.csv")
print(len(df))
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




cap = cv2.VideoCapture("/home/ubuntu/bundesliga/src/Data/train/3c993bd2_0.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)
print("fps:", fps)
df["frame"] = df["time"] * fps


print(df.head(10))


df.to_csv('/home/ubuntu/bundesliga/src/Data/updated_frames_new.csv', index=False)

