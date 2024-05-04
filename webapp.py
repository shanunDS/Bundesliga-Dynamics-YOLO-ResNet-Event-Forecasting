
## NOTE: Custom  Best YOLO model needs to be downloaded from this drive link since file size was big and wasn't getting uploaded to github
## DRIVE LINK: https://drive.google.com/drive/folders/1Pc0UXMM62va9je8aK9-3nLtfAWahDcHx?usp=sharing
## MODEL_DIR = '/Users/vishnuarun/Desktop/Exam1_Deep_Learning/FInal_Project/Webapp/Webapp' needs to updated after downloading

#pip install streamlit
import streamlit as st
import tempfile
import os
import subprocess
from pathlib import Path
import time


st.set_page_config(page_title="Football Player Detection", page_icon=":soccer:")


st.markdown("""
    <style>
        body {
            background-color: #F5F5DC;
        }
        .title {
            font-size: 48px;
            font-weight: bold;
            color: #FFD700;
            margin-bottom: 20px;
            animation: bounce 2s infinite;
        }
        .subtitle {
            font-size: 28px;
            color: #1E90FF;
            margin-bottom: 10px;
        }
        .processing-text {
            font-size: 22px;
            color: #1E90FF;
            margin-bottom: 20px;
        }
        .footer {
            font-size: 14px;
            color: #FFD700;
            position: fixed;
            bottom: 0;
            left: 0;
            right: 0;
            padding: 10px;
            text-align: center;
        }
        @keyframes bounce {
            0%, 20%, 50%, 80%, 100% {transform: translateY(0);}
            40% {transform: translateY(-20px);}
            60% {transform: translateY(-10px);}
        }
    </style>
""", unsafe_allow_html=True)


MODEL_DIR = '/Users/vishnuarun/Desktop/Exam1_Deep_Learning/FInal_Project/Webapp/Webapp'
MODEL_WEIGHTS = 'yolov8m.pt'

st.markdown('<div class="title">PlayerTrack: Football Video Analysis</div>', unsafe_allow_html=True)

st.balloons()

uploaded_file = st.file_uploader("Upload a video...", type=["mp4", "avi"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmpfile:
        tmpfile.write(uploaded_file.read())
        video_path = tmpfile.name

    # Display input video
    st.markdown('<div class="subtitle">Input Video</div>', unsafe_allow_html=True)
    st.video(video_path)

    # Inform user about video processing
    processing_placeholder = st.empty()
    processing_placeholder.markdown('<div class="subtitle">Processing the Video with Custom YoloV8 our Model!</div>', unsafe_allow_html=True)

    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = Path(temp_dir)


        process_command = [
            'yolo', 'detect', 'predict',
            f'model={os.path.join(MODEL_DIR, MODEL_WEIGHTS)}',
            f'source={video_path}', 'save=True',
            f'project={output_dir}', 'conf=0.25'
        ]

        result = subprocess.run(process_command, capture_output=True, text=True, shell=False)

        output_video_path = next((output_dir / "predict").glob("*.mp4"), None)


        processing_placeholder.empty()


        if output_video_path is not None:
            st.markdown('<div class="subtitle">The Processed Video!</div>', unsafe_allow_html=True)
            st.video(str(output_video_path))
            # Display balloons after the model gives the output
            st.balloons()
        else:
            st.write("Output video not found.")


st.markdown('<div class="footer">Made with &#10084; by Deep Learning Class: Group 3</div>', unsafe_allow_html=True)
