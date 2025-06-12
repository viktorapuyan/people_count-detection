import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import torch

# Load the trained YOLO model (assuming it's saved as 'yolo11n.torchscript')
# Make sure the model path is correct
model = YOLO('best.pt') # or load the exported torchscript model: torch.jit.load('yolo11n.torchscript')

st.title("Person Detection with YOLO")

# Option to upload a video file or use webcam
video_source = st.radio("Choose video source:", ('Upload Video', 'Webcam'))

uploaded_file = None
if video_source == 'Upload Video':
    uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov", "mkv"])

stframe = st.empty()

if video_source == 'Webcam' or uploaded_file is not None:
    if video_source == 'Webcam':
        cap = cv2.VideoCapture(0) # 0 for default webcam
    elif uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        cap = cv2.VideoCapture(tfile.name)

    if not cap.isOpened():
        st.error("Error: Could not open video source.")
    else:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.write("End of video or error reading frame.")
                break

            # Perform detection
            results = model(frame)

            # Overlay results on the frame
            for r in results:
                annotated_frame = r.plot(conf=False) # plot with confidence scores hidden

            # Display the frame
            stframe.image(annotated_frame, channels="BGR")

        cap.release()
        if video_source == 'Upload Video':
            tfile.close()
            os.unlink(tfile.name) # Clean up the temporary file

elif video_source == 'Upload Video' and uploaded_file is None:
    st.info("Please upload a video file.")
