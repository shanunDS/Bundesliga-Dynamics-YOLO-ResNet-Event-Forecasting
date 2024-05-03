                                                                                     ## Video Segmentation using YOLO and RESNET

The goal of this Project is to detect football passes - including throw-ins and crosses and challenges in original Bundesliga matches. We have trained a YOLOV5 which is detecting different objects present in the video
and we also train a Resnet model to classify entities into different classes.

Automatic event detection could provide event data faster and with greater depth. Having access to a broader range of competitions, match conditions and data scouts would be able to ensure no talented player is overlooked


Commands to train and run the RESNET model

1.Python3 dataset.py
->Main preprocessing steps like creating additional events (start/end events and pre/post events for each original event are created.
->Computation of total number of frames in each video is done in this step



2. Python3 main_final.py
->Extracting frames from the images and save them as .jpg images - The "extract_images" function takes a video path and an out directory as input arguments. The video frames are extracted and saved to output directories
and are divided into img_train(for training split) and (img_valid) fro validation split images

3.Python3 train.py
->Main Data Preprocessing function - the get_df function extracts frmaes, assigns event classes and identifies background frames

4. Python3 model.py
   ->Resnet model training and evaluation happens in this script, Right now in the script i am taking only 1 train and 1 val video(it takes ~30-35 min for training) as there are roughly 80000 frames per video, but we can increase/decrease the size of training files as per the compute resources
 
   

