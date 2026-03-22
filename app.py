import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import cv2
import threading
import time

# --- INITIAL SETUP ---
st.set_page_config(page_title="Deep Focus Monitor", page_icon="🧘")
st.title("🧘 Deep Focus Monitor")
st.write("Timer starts automatically when the camera sees your face!")

# Shared variables between the Camera and the Webpage
# We use a list for the timer so it's "mutable" across threads
if "focus_data" not in st.session_state:
    st.session_state.focus_data = {"seconds": 0, "active": False}

lock = threading.Lock()

# --- THE VIDEO BRAIN ---
class FocusProcessor(VideoProcessorBase):
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)

        # Update the global status
        with lock:
            st.session_state.focus_data["active"] = len(faces) > 0

        # Visual Feedback on Camera
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 3)
            cv2.putText(img, "FOCUSING", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return frame.from_ndarray(img, format="bgr24")

# --- THE UI LAYOUT ---
col1, col2 = st.columns([2, 1])

with col1:
    # This turns on the camera
    ctx = webrtc_streamer(
        key="main-stream",
        video_processor_factory=FocusProcessor,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": False},
    )

with col2:
    st.subheader("Live Stats")
    timer_display = st.empty()
    status_display = st.empty()

# --- THE AUTOMATIC TIMER LOOP ---
# This runs only when the camera is 'Playing'
if ctx.state.playing:
    while True:
        with lock:
            is_focused = st.session_state.focus_data["active"]
        
        if is_focused:
            st.session_state.focus_data["seconds"] += 1
            msg = "✅ FOCUS MODE: ON"
            color = "green"
        else:
            msg = "⚠️ NOT FOCUSED"
            color = "red"

        # Update the UI without refreshing the whole page
        timer_display.metric("Total Focus Time", f"{st.session_state.focus_data['seconds']} sec")
        status_display.markdown(f"<h3 style='color:{color};'>{msg}</h3>", unsafe_allow_html=True)
        
        time.sleep(1) # Wait 1 second for the next tick
