import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler, RandomSampler, SequentialSampler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR, ReduceLROnPlateau
import albumentations as A
import albumentations
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from pandas.testing import assert_index_equal
from typing import Dict, Tuple

import torchvision
import glob
import timm
from sklearn.metrics import confusion_matrix
import copy
import cv2
import time
import torch.optim.lr_scheduler as lr_scheduler
import torchvision.models as models



import sys
sys.path.append('/C:/Users/Asus/OneDrive/Desktop\Bundesliga\src\dflfiles\ timm.tgz\timm\pytorch-image-models')



IMSIZE = [540, 768]
IMG_SIZE = (540, 768)
modelname = "tf_efficientnet_b0"

use_amp = True
batch_size = 40
n_epochs = 1
num_workers = 2
COSINE = True
init_lr = 2e-4
kernel_type = "{}-{}".format(modelname, IMSIZE[0])

IMG_SOURCE = "img"
BACK_INTERVAL = 20
BACK_INTERVAL_VAL = 1
ERR_TOL = 1
mixup = True
# DEBUG = True

print('len df train')
df_train = pd.read_csv("/home/ubuntu/bundesliga/src/Data/training_folds_iter.csv")

print(len(df_train))

print('len df valid')
df_valid = pd.read_csv("/home/ubuntu/bundesliga/src/Data/val_folds_iter.csv")
print(len(df_valid))
class net(nn.Module):
    def __init__(self, modelname, out_dim=10, freeze_bn=True):
        super(net, self).__init__()
        self.model = timm.create_model(modelname, pretrained=True)
        self.model.reset_classifier(out_dim)

    def forward(self, x):
        x = self.model(x)
        return x


# class net(nn.Module):
#     def __init__(self, modelname, out_dim=10, freeze_bn=True):
#         super(net, self).__init__()
#         self.model = models.efficientnet_b0(pretrained=True)  # Instantiate EfficientNet directly
#         num_ftrs = self.model._fc.in_features
#         self.model._fc = nn.Linear(num_ftrs, out_dim)
#
#
#     def forward(self, x):
#         x = self.model(x)
#         return x


def get_label(label):
    if label == "background":
        label = 0
        label2 = 0
    elif label == "challenge":
        label = 1
        label2 = 1
    elif label == "play":
        label = 2
        label2 = 1
    elif label == "throwin":
        label = 3
        label2 = 1
    elif label == "pre_challenge":
        label = 4
        label2 = 1
    elif label == "pre_play":
        label = 5
        label2 = 1
    elif label == "pre_throwin":
        label = 6
        label2 = 1
    elif label == "post_challenge":
        label = 7
        label2 = 1
    elif label == "post_play":
        label = 8
        label2 = 1
    elif label == "post_throwin":
        label = 9
        label2 = 1
    label = np.array(label)
    return label


class dflDataset(Dataset):
    def __init__(self,
                 df,
                 test=False,
                 transform=None,
                 ):

        self.df = df.reset_index(drop=True)
        self.test = test
        self.transform = transform

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        row = self.df.iloc[index]
        images = []
        for i in [-1, 0, 1]:
            tiff_file = "work/{}/".format(IMG_SOURCE) + row.video + "-" + str(row.frame + i).zfill(6) + ".jpg"
            im = cv2.imread(tiff_file, cv2.IMREAD_GRAYSCALE)
            images.append(im)
        images = np.array(images).transpose(1, 2, 0)

        # aug
        if self.transform is not None:
            images = self.transform(image=images)['image']
        file = copy.copy(tiff_file)
        images = images.astype(np.float32)
        images /= 255
        images = images.transpose(2, 0, 1)

        # Load labels
        label = get_label(row.cls)

        # Mixup part
        rd = np.random.rand()
        label2 = label
        gamma = np.array(np.ones(1)).astype(np.float32)[0]
        if mixup and rd < 0.9 and not self.test:
            mix_idx = np.random.randint(0, len(self.df) - 1)
            row2 = self.df.iloc[mix_idx]
            images2 = []
            for i in [-1, 0, 1]:
                tiff_file = "work/{}/".format(IMG_SOURCE) + row2.video + "-" + str(row2.frame + i).zfill(6) + ".jpg"
                im = cv2.imread(tiff_file, cv2.IMREAD_GRAYSCALE)
                images2.append(im)
            images2 = np.array(images2).transpose(1, 2, 0)

            if self.transform is not None:
                images2 = self.transform(image=images2)['image']

            images2 = images2.astype(np.float32)
            images2 /= 255
            images2 = images2.transpose(2, 0, 1)
            # blend image
            gamma = np.array(np.random.beta(1, 1)).astype(np.float32)
            images = ((images * gamma + images2 * (1 - gamma)))
            # blend labels
            label2 = get_label(row2.cls)

        return torch.tensor(images), torch.tensor(label), torch.tensor(label2), torch.tensor(gamma), file


transforms_train = albumentations.Compose([
    albumentations.ShiftScaleRotate(scale_limit=0.3, rotate_limit=45,p=0.5),
    A.OneOf([
       A.RandomBrightnessContrast(brightness_limit=0.2,
                                  contrast_limit=0.2, p=0.5),
    ],p=0.9),
    A.CoarseDropout(max_holes=12, max_height=12, max_width=12, min_holes=1, min_height=1, min_width=1, p=0.5),
    albumentations.Rotate(p=0.5),
    albumentations.Transpose(p=0.5),
    albumentations.VerticalFlip(p=0.5),
    albumentations.HorizontalFlip(p=0.5),
    albumentations.Resize(IMSIZE[0], IMSIZE[1], p=1.0),
])
transforms_val = albumentations.Compose([albumentations.Resize(IMSIZE[0], IMSIZE[1], p=1.0),])





dataset = dflDataset(df_train, transform=transforms_train)
    # Setup dataloader
loader = torch.utils.data.DataLoader(dataset, batch_size=8, sampler=RandomSampler(dataset))
# for (d,label,_,_,_) in loader:
#     for b,l in zip(d, label):
#         plt.figure(figsize=(10,10))
#         plt.imshow(b.transpose(0, 2).transpose(0,1).numpy()[:,:,::-1])
#         plt.show()
#         print(l)
#     break



#model training

scaler = torch.cuda.amp.GradScaler(enabled=use_amp)


def train_epoch(loader, optimizer):
    model.train()
    train_loss = []
    # print(train_loss)
    bar = tqdm(loader)
    for (data, target, target2, lam, _) in bar:
        data, target, target2, lam = data.to(device), target.to(device).long(), target2.to(device).long(), lam.to(
            device).float()
        loss_func = criterion
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(data).squeeze(1)
            if mixup:
                loss = mixup_criterion(criterion, logits, target, target2, lam).mean()
            else:
                loss = loss_func(logits, target)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        loss_np = loss.detach().cpu().numpy()
        print('training loss')
        train_loss.append(loss_np)
        print(train_loss)

        smooth_loss = sum(train_loss[-100:]) / min(len(train_loss), 100)
        bar.set_description('loss: %.5f, smth: %.5f' % (loss_np, smooth_loss))
    return np.mean(train_loss)


tolerances = {
    "challenge": [0.3, 0.4, 0.5, 0.6, 0.7],
    "play": [0.15, 0.20, 0.25, 0.30, 0.35],
    "throwin": [0.15, 0.20, 0.25, 0.30, 0.35],
}


def filter_detections(
        detections: pd.DataFrame, intervals: pd.DataFrame
) -> pd.DataFrame:
    """Drop detections not inside a scoring interval."""
    detection_time = detections.loc[:, 'time'].sort_values().to_numpy()
    intervals = intervals.to_numpy()
    is_scored = np.full_like(detection_time, False, dtype=bool)

    i, j = 0, 0
    while i < len(detection_time) and j < len(intervals):
        time = detection_time[i]
        int_ = intervals[j]

        # If the detection is prior in time to the interval, go to the next detection.
        if time < int_.left:
            i += 1
        # If the detection is inside the interval, keep it and go to the next detection.
        elif time in int_:
            is_scored[i] = True
            i += 1
        # If the detection is later in time, go to the next interval.
        else:
            j += 1

    return detections.loc[is_scored].reset_index(drop=True)


def match_detections(
        tolerance: float, ground_truths: pd.DataFrame, detections: pd.DataFrame
) -> pd.DataFrame:
    """Match detections to ground truth events. Arguments are taken from a common event x tolerance x video evaluation group."""
    detections_sorted = detections.sort_values('score', ascending=False).dropna()

    is_matched = np.full_like(detections_sorted['event'], False, dtype=bool)
    gts_matched = set()
    for i, det in enumerate(detections_sorted.itertuples(index=False)):
        best_error = tolerance
        best_gt = None

        for gt in ground_truths.itertuples(index=False):
            error = abs(det.time - gt.time)
            if error < best_error and not gt in gts_matched:
                best_gt = gt
                best_error = error

        if best_gt is not None:
            is_matched[i] = True
            gts_matched.add(best_gt)

    detections_sorted['matched'] = is_matched

    return detections_sorted


def precision_recall_curve(
        matches: np.ndarray, scores: np.ndarray, p: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(matches) == 0:
        return [1], [0], []

    # Sort matches by decreasing confidence
    idxs = np.argsort(scores, kind='stable')[::-1]
    scores = scores[idxs]
    matches = matches[idxs]

    distinct_value_indices = np.where(np.diff(scores))[0]
    threshold_idxs = np.r_[distinct_value_indices, matches.size - 1]
    thresholds = scores[threshold_idxs]

    # Matches become TPs and non-matches FPs as confidence threshold decreases
    tps = np.cumsum(matches)[threshold_idxs]
    fps = np.cumsum(~matches)[threshold_idxs]

    precision = tps / (tps + fps)
    precision[np.isnan(precision)] = 0
    recall = tps / p  # total number of ground truths might be different than total number of matches

    # Stop when full recall attained and reverse the outputs so recall is non-increasing.
    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)

    # Final precision is 1 and final recall is 0
    return np.r_[precision[sl], 1], np.r_[recall[sl], 0], thresholds[sl]


def average_precision_score(matches: np.ndarray, scores: np.ndarray, p: int) -> float:
    precision, recall, _ = precision_recall_curve(matches, scores, p)
    # Compute step integral
    return -np.sum(np.diff(recall) * np.array(precision)[:-1])


def recall_score(matches: np.ndarray, scores: np.ndarray, p: int) -> float:
    precision, recall, _ = precision_recall_curve(matches, scores, p)
    # Compute step integral
    return np.mean(recall)


def precision_score(matches: np.ndarray, scores: np.ndarray, p: int) -> float:
    precision, recall, _ = precision_recall_curve(matches, scores, p)
    # Compute step integral
    return np.mean(precision)


def event_detection_ap(
        solution: pd.DataFrame,
        submission: pd.DataFrame,
        tolerances: Dict[str, float],
) -> float:
    assert_index_equal(solution.columns, pd.Index(['video_id', 'time', 'event']))
    assert_index_equal(submission.columns, pd.Index(['video_id', 'time', 'event', 'score']))

    # Ensure solution and submission are sorted properly
    solution = solution.sort_values(['video_id', 'time'])
    submission = submission.sort_values(['video_id', 'time'])

    # Extract scoring intervals.
    intervals = (
        solution
        .query("event in ['start', 'end']")
        .assign(interval=lambda x: x.groupby(['video_id', 'event']).cumcount())
        .pivot(index='interval', columns=['video_id', 'event'], values='time')
        .stack('video_id')
        .swaplevel()
        .sort_index()
        .loc[:, ['start', 'end']]
        .apply(lambda x: pd.Interval(*x, closed='both'), axis=1)
    )

    # Extract ground-truth events.
    ground_truths = (
        solution
        .query("event not in ['start', 'end']")
        .reset_index(drop=True)
    )

    # Map each event class to its prevalence (needed for recall calculation)
    class_counts = ground_truths.value_counts('event').to_dict()

    # Create table for detections with a column indicating a match to a ground-truth event
    detections = submission.assign(matched=False)

    # Remove detections outside of scoring intervals
    detections_filtered = []
    for (det_group, dets), (int_group, ints) in zip(
            detections.groupby('video_id'), intervals.groupby('video_id')
    ):
        assert det_group == int_group
        detections_filtered.append(filter_detections(dets, ints))
    detections_filtered = pd.concat(detections_filtered, ignore_index=True)

    # Create table of event-class x tolerance x video_id values
    aggregation_keys = pd.DataFrame(
        [(ev, tol, vid)
         for ev in tolerances.keys()
         for tol in tolerances[ev]
         for vid in ground_truths['video_id'].unique()],
        columns=['event', 'tolerance', 'video_id'],
    )

    # Create match evaluation groups: event-class x tolerance x video_id
    detections_grouped = (
        aggregation_keys
        .merge(detections_filtered, on=['event', 'video_id'], how='left')
        .groupby(['event', 'tolerance', 'video_id'])
    )
    ground_truths_grouped = (
        aggregation_keys
        .merge(ground_truths, on=['event', 'video_id'], how='left')
        .groupby(['event', 'tolerance', 'video_id'])
    )

    # Match detections to ground truth events by evaluation group
    detections_matched = []
    for key in aggregation_keys.itertuples(index=False):
        dets = detections_grouped.get_group(key)
        gts = ground_truths_grouped.get_group(key)
        detections_matched.append(
            match_detections(dets['tolerance'].iloc[0], gts, dets)
        )
    detections_matched = pd.concat(detections_matched)

    # Compute AP per event x tolerance group
    event_classes = ground_truths['event'].unique()
    ap_table = (
        detections_matched
        .query("event in @event_classes")
        .groupby(['event', 'tolerance']).apply(
            lambda group: average_precision_score(
                group['matched'].to_numpy(),
                group['score'].to_numpy(),
                class_counts[group['event'].iat[0]],
            )
        )
    )

    recall_table = (
        detections_matched
        .query("event in @event_classes")
        .groupby(['event', 'tolerance']).apply(
            lambda group: recall_score(
                group['matched'].to_numpy(),
                group['score'].to_numpy(),
                class_counts[group['event'].iat[0]],
            )
        )
    )

    print("recall:")
    print(recall_table)

    pre_table = (
        detections_matched
        .query("event in @event_classes")
        .groupby(['event', 'tolerance']).apply(
            lambda group: precision_score(
                group['matched'].to_numpy(),
                group['score'].to_numpy(),
                class_counts[group['event'].iat[0]],
            )
        )
    )

    print("precision:")
    print(pre_table)

    # Average over tolerances, then over event classes
    mean_ap = ap_table.groupby('event').mean().mean()
    return mean_ap, ap_table





solution = pd.read_csv("/home/ubuntu/bundesliga/src/Data/train.csv", usecols=['video_id', 'time', 'event'])
tolerances = {
    "challenge": [0.3, 0.4, 0.5, 0.6, 0.7],
    "play": [0.15, 0.20, 0.25, 0.30, 0.35],
    "throwin": [0.15, 0.20, 0.25, 0.30, 0.35],
}


def val_epoch(loader):
    model.eval()
    val_loss = []
    LOGITS = []
    PREDS = []
    TARGETS = []
    FILES = []

    with torch.no_grad():
        for (data, target, _, _, files) in tqdm(loader):
            data, target = data.to(device), target.to(device).long()
            logits = model(data)
            logits = logits.squeeze(1)
            loss_func = criterion
            loss = loss_func(logits, target)

            pred = logits.sigmoid().detach()
            LOGITS.append(logits)
            PREDS.append(pred)
            TARGETS.append(target)
            FILES.extend(files)

            val_loss.append(loss.detach().cpu().numpy())
        val_loss = np.mean(val_loss)

    LOGITS = torch.cat(LOGITS).cpu().numpy()
    PREDS = torch.cat(PREDS).cpu().numpy()
    TARGETS = torch.cat(TARGETS).cpu().numpy()
    acc = np.sum(PREDS.argmax(1) == TARGETS) / len(PREDS.argmax(1)) * 100

    # analyze
    cm = confusion_matrix(TARGETS, PREDS.argmax(1))
    print(cm)
    for i, val in enumerate(cm):
        print("for class {}: accuracy: {}".format(i, val[i] / sum(val) * 100))

    # val_df = make_sub(PREDS[:, :4], FILES)
    score, ap_table = event_detection_ap(solution[solution['video_id'].isin(df_valid['video_id'].unique())],
                                         df_valid[['video_id', 'time', 'event', 'score']], tolerances)
    print(score)
    print(ap_table)

    return val_loss, acc, score, PREDS, FILES, TARGETS






device = "cuda"
criterion = nn.CrossEntropyLoss()

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)



dataset_train = dflDataset(df_train, transform=transforms_train)
dataset_valid = dflDataset(df_valid, transform=transforms_val, test=True)

# Setup dataloader
train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, sampler=RandomSampler(dataset_train), num_workers=num_workers)
valid_loader = torch.utils.data.DataLoader(dataset_valid, batch_size=batch_size, sampler=SequentialSampler(dataset_valid), num_workers=num_workers)

# Initialize model
model = net(modelname)
model = model.to(device)
print('length of train')
print(len(dataset_train))

print('length of valid')
print(len(dataset_valid))


# We use Cosine annealing LR scheduling
optimizer = optim.Adam(model.parameters(), lr=init_lr)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)

scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

# C:\Users\Asus\OneDrive\Desktop\Bundesliga\input\dflfiles\timm.tgz\timm.tgz\timm\pytorch-image-models\timm
# best_file = f'./models/{kernel_type}_best_fold{fold}.pth'

# best_file = f'./home/ubuntu/bundesliga/input/dflfiles/timm.tgz/timm.tgz/timm/pytorch-image-models/timm/{kernel_type}.pth'
# print(best_file)

# best = 0
for epoch in range(1, n_epochs+1):
    start = time.time()
    torch.cuda.empty_cache()
    print(time.ctime(), 'Epoch:', epoch)

    # scheduler.step(epoch-1)

    optimizer.step()

    scheduler.step()
    print('training loss')
    train_loss = train_epoch(train_loader, optimizer)
    print(train_loss)
    val_loss, acc, score, PREDS, FILES, TARGETS  = val_epoch(valid_loader)

    print('validation loss')
    print(val_loss)


    # torch.save(model.state_dict(), os.path.join(f'/home/ubuntu/bundesliga/input/dflfiles/timm.tgz/timm.tgz/timm/pytorch-image-models/timm/{kernel_type}_epoch{epoch}.pth'))
    torch.save(model.state_dict(), os.path.join('/home/ubuntu/bundesliga/src/saved_models', f'{kernel_type}_epoch{epoch}.pth'))

    # if score > best:
    #     torch.save(model.state_dict(), os.path.join(f'models/{kernel_type}_epoch{epoch}_fold{fold}.pth'))
    #     best = score























