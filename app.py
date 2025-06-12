# app.py
import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import torch
import tempfile
import os

# Load YOLO model
model = YOLO('yolo11n.pt')

st.title("Person Detection with YOLO")

# Choose video source
video_source = st.radio("Choose video source:", ('Upload Video', 'Webcam'))

uploaded_file = None
if video_source == 'Upload Video':
    uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov", "mkv"])

# Create Streamlit UI elements
stframe = st.empty()
playback_state = st.checkbox("Play Video", value=False)
restart_button = st.button("Restart Video")

# Persistent session state for controlling video playback
if "cap" not in st.session_state:
    st.session_state.cap = None
if "restart" not in st.session_state:
    st.session_state.restart = False

# Video source setup
if (video_source == 'Webcam' or uploaded_file is not None) and (playback_state or st.session_state.cap is not None):
    if st.session_state.cap is None or st.session_state.restart:
        if video_source == 'Webcam':
            st.session_state.cap = cv2.VideoCapture(0)
        else:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_file.read())
            tfile.flush()
            st.session_state.video_path = tfile.name
            st.session_state.cap = cv2.VideoCapture(tfile.name)
        st.session_state.restart = False

    cap = st.session_state.cap

    if not cap or not cap.isOpened():
        st.error("Error: Could not open video source.")
    else:
        if playback_state:
            ret, frame = cap.read()
            if not ret:
                st.write("End of video.")
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Loop video
            else:
                results = model(frame)
                for r in results:
                    annotated_frame = r.plot(conf=False)
                stframe.image(annotated_frame, channels="BGR")
        else:
            stframe.info("Paused")

    # Restart logic
    if restart_button:
        if cap:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        st.session_state.restart = True

# Cleanup if file uploaded
if video_source == 'Upload Video' and uploaded_file is None:
    st.info("Please upload a video file.")
