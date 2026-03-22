import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import cv2
import mediapipe as mp
import time

# --- SETUP ---
st.set_page_config(page_title="Driver Drowsiness Alert", page_icon="🚗")
st.title("🚗 AI Drowsiness Detector")
st.write("Keep your eyes on the road! The app will alert you if you fall asleep.")

# MediaPipe Face Mesh setup (better for eye tracking)
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

class DrowsinessProcessor(VideoProcessorBase):
    def __init__(self):
        self.closed_start_time = None
        self.drowsy_alert = False

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        results = face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Get specific points for top and bottom eyelid
                # Landmark 159 is top, 145 is bottom for the right eye
                top = face_landmarks.landmark[159].y
                bottom = face_landmarks.landmark[145].y
                
                distance = abs(top - bottom)

                # Threshold: If distance is very small, eyes are closed
                if distance < 0.015: 
                    if self.closed_start_time is None:
                        self.closed_start_time = time.time()
                    
                    # Check if closed for more than 2 seconds
                    if time.time() - self.closed_start_time > 2:
                        self.drowsy_alert = True
                else:
                    self.closed_start_time = None
                    self.drowsy_alert = False

                # Visual Feedback
                color = (0, 0, 255) if self.drowsy_alert else (0, 255, 0)
                status = "!!! WAKE UP !!!" if self.drowsy_alert else "AWAKE"
                cv2.putText(img, status, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        return frame.from_ndarray(img, format="bgr24")

webrtc_streamer(key="drowsy-check", video_processor_factory=DrowsinessProcessor)
