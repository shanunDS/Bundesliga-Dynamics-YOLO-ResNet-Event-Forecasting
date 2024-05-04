
---

# Object Detection and Prediction Using YOLO and ResNet

## Project Overview
The goal of this project is to detect football passes, including throw-ins and crosses, as well as challenges in original Bundesliga matches. We have trained YOLOv5, YOLOv8, and ResNet models to detect various objects present in videos and to classify entities into different classes. Automatic event detection could provide event data faster and with greater depth. Having access to a broader range of competitions and match conditions would enable data scouts to ensure no talented player is overlooked.

## Commands to Train and Run the ResNet Model

### Step 1: Data Preprocessing
```bash
python3 dataset.py
```
- Main preprocessing steps include creating additional events (start/end events and pre/post events for each original event).
- Computation of the total number of frames in each video.

### Step 2: Frame Extraction
```bash
python3 main_final.py
```
- Extracting frames from videos and saving them as .jpg images.
- The `extract_images` function takes a video path and an output directory as input arguments. The video frames are extracted and saved to output directories divided into `img_train` (for training split) and `img_valid` (for validation split images).

### Step 3: Data Preprocessing for Model
```bash
python3 train.py
```
- The `get_df` function extracts frames, assigns event classes, and identifies background frames.

### Step 4: Model Training and Evaluation
```bash
python3 model.py
```
- ResNet model training and evaluation script.
- Currently, the script processes only 1 train and 1 validation video (it takes ~30-35 min for training) as there are roughly 80,000 frames per video, but we can scale the size of training files depending on compute resources.

### Step 5: Web Application for Football Video Analysis
```bash
python3 webapp.py
```
- A Streamlit-powered Python script that serves as the frontend for our football video analysis tool.
- This web application allows users to upload football match videos and receive real-time annotations for various game elements such as players, referees, goalkeepers, and the ball.
- The application leverages our custom-trained models.

---
