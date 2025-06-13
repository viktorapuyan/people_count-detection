import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile
import os

# Load YOLO model
model = YOLO('best.torchscript')

st.title("🎥 Person Count Detection with YOLO and Frame Playback")

# File uploader only
uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov", "mkv"])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    tfile.flush()
    cap = cv2.VideoCapture(tfile.name)

    if not cap.isOpened():
        st.error("❌ Could not open uploaded video.")
    else:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        stframe = st.empty()
        person_count_placeholder = st.empty()

        # Initialize playback state
        if "current_frame" not in st.session_state:
            st.session_state.current_frame = 0
        if "is_playing" not in st.session_state:
            st.session_state.is_playing = False

        # Playback controls
        col1, col2 = st.columns([1, 6])
        with col1:
            play_pause = st.button("▶️ Play" if not st.session_state.is_playing else "⏸️ Pause")
        with col2:
            slider = st.slider("Frame", 0, total_frames - 1, st.session_state.current_frame, key="slider")

        if play_pause:
            st.session_state.is_playing = not st.session_state.is_playing

        # Handle frame reading and detection
        if not st.session_state.is_playing:
            st.session_state.current_frame = slider
            cap.set(cv2.CAP_PROP_POS_FRAMES, slider)
            ret, frame = cap.read()
        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, st.session_state.current_frame)
            ret, frame = cap.read()
            st.session_state.current_frame += 1
            if st.session_state.current_frame >= total_frames:
                st.session_state.is_playing = False
                st.session_state.current_frame = 0

        if ret:
            results = model(frame)
            person_count = 0
            for r in results:
                boxes = r.boxes
                classes = boxes.cls
                person_count = sum(1 for c in classes if int(c) == 0)  # Class 0 = person
                annotated_frame = r.plot(conf=False)

            stframe.image(annotated_frame, channels="BGR", use_column_width=True)
            person_count_placeholder.markdown(f"### 👥 Persons Detected: `{person_count}`")

        cap.release()
        os.unlink(tfile.name)

else:
    st.info("📁 Please upload a video to start detection.")
