import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import albumentations as A
import cv2
import os
import timm
import glob

import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler, RandomSampler, SequentialSampler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR, ReduceLROnPlateau

IMSIZE = [540, 768]
IMG_SIZE = (540, 768)
modelname = "tf_efficientnet_b0"
use_amp = True
batch_size = 40
n_epochs = 30
num_workers = 2
COSINE = True
init_lr = 2e-4
kernel_type = "{}-{}".format(modelname, IMSIZE[0])

IMG_SOURCE = "img/"
BACK_INTERVAL = 20
BACK_INTERVAL_VAL = 1
ERR_TOL = 1
mixup = True
DEBUG = True




# video_id_split = {
#     'val':[
#          '3c993bd2_0',
#          '3c993bd2_1',
#          '35bd9041_0',
#          '35bd9041_1',
#     ],
#     'train':[
#          '1606b0e6_0',
#          '1606b0e6_1',
#          '407c5a9e_1',
#          '4ffd5986_0',
#          '9a97dae4_1',
#          'cfbe2e94_0',
#          'cfbe2e94_1',
#          'ecf251d4_0',
#     ]
# }




# selected_videos = [
#     "1606b0e6_0.mp4",
#     "35bd9041_0.mp4",
#     "3c993bd2_0.mp4",
#     "407c5a9e_1.mp4",
#     "9a97dae4_1.mp4"
# ]

# a97dae4_1

video_id_split = {

    'val': ['407c5a9e_1'],

    'train': ["1606b0e6_0", "35bd9041_0", "3c993bd2_0"]
}








df = pd.read_csv('/home/ubuntu/bundesliga/src/Data/updated_frames.csv')

print(df.head(10))
def get_df(video_id, VAL=False):
    df_video = df[df.video_id == video_id]

    print(video_id, df_video.shape)

    # crr_statu => background, play, challenge, throwin
    arr = df_video[['frame', 'event']].values

    # print(arr)

    start = None
    data = []
    for a in arr:
        if "pre_" in a[1]:
            start = a[0]
            # print(start)
            cls = a[1]
            # print(cls)
        if "start_" in a[1]:
            # print('data upload')
            data.append({"start": start, "end": a[0], "cls": cls})
            # print(data)
            start = a[0]
            cls = a[1].split("_")[-1]
        if "end_" in a[1]:
            end = a[0]
            data.append({"start": start, "end": end, "cls": cls})
        if "post_" in a[1]:
            data.append({"start": end, "end": a[0], "cls": a[1]})
    # make events
    out = []
    # print('data')
    # print(data)
    for d in data:
        # print('starting')
        start = int(d["start"])
        # print(start)

        if os.path.isfile(os.path.join("/home/ubuntu/bundesliga/src/work/", IMG_SOURCE, video_id + "-" + str(start).zfill(6) + ".jpg")):
            print('out')
            out.append({"frame": start, "cls": d["cls"], "video": video_id})
            print(len(out))
        start += 1
        while start <= d["end"]:
            if os.path.isfile(os.path.join("/home/ubuntu/bundesliga/src/work/", IMG_SOURCE, video_id + "-" + str(start).zfill(6) + ".jpg")):
                out.append({"frame": start, "cls": d["cls"], "video": video_id})
            start += 1

    df2 = pd.DataFrame(out)

    # print(df2)
    if not VAL:
        for i in range(10, df2.frame.max(), BACK_INTERVAL):
            if np.sum(df2.frame.isin([i])) == 0:
                if os.path.isfile(os.path.join("/home/ubuntu/bundesliga/src/work/", IMG_SOURCE, video_id + "-" + str(i).zfill(6) + ".jpg")):
                    out.append({"frame": i, "cls": "background", "video": video_id})
                else:
                    print("pass:", i)
    else:
        for i in range(10, df2.frame.max(), BACK_INTERVAL_VAL):
            if np.sum(df2.frame.isin([i])) == 0:
                if os.path.isfile(os.path.join("/home/ubuntu/bundesliga/src/work/", IMG_SOURCE, video_id + "-" + str(i).zfill(6) + ".jpg")):
                    out.append({"frame": i, "cls": "background", "video": video_id})
                else:
                    print("pass:", i)
    df2 = pd.DataFrame(out)
    return df2




for i,video_id in enumerate(video_id_split["train"]):
    # print(video_id)
    df2 = get_df(video_id)
    if i > 0:
        df_train = pd.concat([df_train, df2])
    else:
        df_train = df2

for i,video_id in enumerate(video_id_split["val"]):
    df2 = get_df(video_id, True)
    if i > 0:
        df_val = pd.concat([df_val, df2])
    else:
        df_val = df2


print('training dataframe')

print(df_train.head(10))

print(len(df_train))

print('val dataframe')

print(df_val.head(10))


print(len(df_val))


df_train.to_csv("/home/ubuntu/bundesliga/src/Data/training_folds_iter.csv", index=False)

df_val.to_csv("/home/ubuntu/bundesliga/src/Data/val_folds_iter.csv", index=False)

