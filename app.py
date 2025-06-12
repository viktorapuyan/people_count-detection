import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import torch
import tempfile
import os

# Load YOLO model
model = YOLO('yolo11n.pt')

st.title("YOLO Person Detection with Video Playback")

video_source = st.radio("Choose video source:", ("Upload Video", "Webcam"))

uploaded_file = None
if video_source == "Upload Video":
    uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov", "mkv"])

# For uploaded video
if video_source == "Upload Video" and uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    tfile.flush()
    cap = cv2.VideoCapture(tfile.name)

    if not cap.isOpened():
        st.error("Error: Could not open video file.")
    else:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        stframe = st.empty()

        # Set initial state
        if "current_frame" not in st.session_state:
            st.session_state.current_frame = 0
        if "is_playing" not in st.session_state:
            st.session_state.is_playing = False

        # Playback control UI
        col1, col2 = st.columns([1, 6])
        with col1:
            play_pause = st.button("▶️ Play" if not st.session_state.is_playing else "⏸️ Pause")
        with col2:
            slider = st.slider("Frame", 0, total_frames - 1, st.session_state.current_frame, key="slider")

        if play_pause:
            st.session_state.is_playing = not st.session_state.is_playing

        # Use slider frame if paused
        if not st.session_state.is_playing:
            st.session_state.current_frame = slider
            cap.set(cv2.CAP_PROP_POS_FRAMES, slider)
            ret, frame = cap.read()
            if ret:
                results = model(frame)
                for r in results:
                    annotated_frame = r.plot(conf=False)
                stframe.image(annotated_frame, channels="BGR", use_column_width=True)
        else:
            # Playing: advance frame by frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, st.session_state.current_frame)
            ret, frame = cap.read()
            if ret:
                results = model(frame)
                for r in results:
                    annotated_frame = r.plot(conf=False)
                stframe.image(annotated_frame, channels="BGR", use_column_width=True)
                st.session_state.current_frame += 1
                if st.session_state.current_frame >= total_frames:
                    st.session_state.is_playing = False
                    st.session_state.current_frame = 0

        cap.release()
        os.unlink(tfile.name)

elif video_source == "Webcam":
    st.warning("Slider-based playback is only supported for uploaded videos. Use webcam mode for live feed.")
