import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
from streamlit_autorefresh import st_autorefresh
import cv2
import time
import threading

st.set_page_config(page_title="Deep Focus Monitor", page_icon="🧘")

st.title("🧘 Deep Focus Monitor")
st.write("Stay in frame to keep timer running")

lock = threading.Lock()

shared = {
    "focused": False
}


# ------------------------
# Face Detector
# ------------------------

class FaceDetector(VideoProcessorBase):

    def __init__(self):

        self.face = cv2.CascadeClassifier(
            cv2.data.haarcascades +
            "haarcascade_frontalface_default.xml"
        )

    def recv(self, frame):

        img = frame.to_ndarray(format="bgr24")

        gray = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2GRAY
        )

        faces = self.face.detectMultiScale(
            gray,
            1.1,
            4
        )

        with lock:
            shared["focused"] = len(faces) > 0

        for (x, y, w, h) in faces:

            cv2.rectangle(
                img,
                (x, y),
                (x+w, y+h),
                (0,255,0),
                2
            )

        return frame.from_ndarray(img, format="bgr24")


# ------------------------
# Camera
# ------------------------

ctx = webrtc_streamer(
    key="focus",
    video_processor_factory=FaceDetector,
    media_stream_constraints={"video": True, "audio": False},
)


# ------------------------
# Auto refresh every 1 sec
# ------------------------

st_autorefresh(interval=1000, key="timer")


# ------------------------
# Session state
# ------------------------

if "focus_time" not in st.session_state:
    st.session_state.focus_time = 0

if "last_time" not in st.session_state:
    st.session_state.last_time = time.time()


# ------------------------
# Status
# ------------------------

camera_on = ctx.state.playing if ctx else False

with lock:
    focused = shared["focused"]

now = time.time()


if camera_on and focused:

    st.session_state.focus_time += int(
        now - st.session_state.last_time
    )

    status = "✅ Focused"
    color = "green"

elif camera_on and not focused:

    status = "❌ Not Focused"
    color = "red"

else:

    status = "📷 Camera Off"
    color = "orange"


st.session_state.last_time = now


# ------------------------
# UI
# ------------------------

st.markdown(
    f"<h2 style='text-align:center;color:{color}'>{status}</h2>",
    unsafe_allow_html=True
)

st.metric(
    "Focus Time",
    f"{st.session_state.focus_time} sec"
)
