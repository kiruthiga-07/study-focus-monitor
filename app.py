import streamlit as st
import cv2
import numpy as np
import time

# Page Config
st.set_page_config(page_title="Study Focus Monitor", page_icon="📚")
st.title("📚 Study Focus Monitor")
st.subheader("Stay focused! The timer stops if you look away.")

# Initialize Session State for the timer
if 'start_time' not in st.session_state:
    st.session_state.start_time = None
if 'total_focus_time' not in st.session_state:
    st.session_state.total_focus_time = 0

# Load the pre-trained Face Detector (Haar Cascade)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Sidebar Controls
st.sidebar.header("Controls")
run = st.sidebar.checkbox('Start Monitoring')

# Camera Input
img_file_buffer = st.camera_input("Take a snapshot to check focus", disabled=not run)

if img_file_buffer is not None and run:
    # Convert the file to an opencv image
    bytes_data = img_file_buffer.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    
    # Convert to Grayscale for detection
    gray = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    if len(faces) > 0:
        # FACE DETECTED
        st.success("✅ You are focused! Keep it up.")
        if st.session_state.start_time is None:
            st.session_state.start_time = time.time()
        
        # Calculate current session time
        current_session = time.time() - st.session_state.start_time
        st.metric("Current Focus Session", f"{int(current_session)} seconds")
    else:
        # NO FACE DETECTED
        st.error("⚠️ Face not detected! Are you still studying?")
        st.session_state.start_time = None  # Reset session start
        st.snow() # Visual feedback for losing focus

else:
    st.info("Check the 'Start Monitoring' box in the sidebar to begin.")
    st.session_state.start_time = None

st.write("---")
st.caption("Tip: Use good lighting so the camera can detect your face clearly.")
