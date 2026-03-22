import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import cv2
import time
import threading

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="Deep Focus Monitor", page_icon="🧘")

st.title("🧘 Deep Focus Monitor")
st.write("Stay in frame to keep your focus timer running!")

lock = threading.Lock()
container = {"is_focused": False, "camera_on": False}


# -----------------------------
# Video Processor
# -----------------------------
class FaceDetector(VideoProcessorBase):

    def __init__(self):
        self.face = cv2.CascadeClassifier(
            cv2.data.haarcascades +
            "haarcascade_frontalface_default.xml"
        )

    def recv(self, frame):

        img = frame.to_ndarray(format="bgr24")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = self.face.detectMultiScale(
            gray,
            1.1,
            4
        )

        with lock:
            container["camera_on"] = True
            container["is_focused"] = len(faces) > 0

        for (x, y, w, h) in faces:

            cv2.rectangle(
                img,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

            cv2.putText(
                img,
                "FOCUSED",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

        return frame.from_ndarray(img, format="bgr24")


# -----------------------------
# Start camera
# -----------------------------
ctx = webrtc_streamer(
    key="focus",
    video_processor_factory=FaceDetector,
    media_stream_constraints={"video": True, "audio": False},
)


# -----------------------------
# Session State
# -----------------------------
if "focus_time" not in st.session_state:
    st.session_state.focus_time = 0

if "last_update" not in st.session_state:
    st.session_state.last_update = time.time()


# -----------------------------
# Timer Logic
# -----------------------------
placeholder = st.empty()

with lock:
    focused = container["is_focused"]
    camera_on = ctx.state.playing if ctx else False


now = time.time()

if camera_on and focused:

    st.session_state.focus_time += int(
        now - st.session_state.last_update
    )

    status = "✅ Focused"
    color = "green"

elif camera_on and not focused:

    status = "⚠️ Look at screen"
    color = "red"

else:

    status = "📷 Camera Off"
    color = "orange"


st.session_state.last_update = now


# -----------------------------
# UI
# -----------------------------
with placeholder.container():

    st.markdown(
        f"<h2 style='color:{color}; text-align:center'>{status}</h2>",
        unsafe_allow_html=True,
    )

    st.metric(
        "Total Focus Time",
        f"{st.session_state.focus_time} sec"
    )


# -----------------------------
# Auto refresh every second
# -----------------------------
time.sleep(1)
st.rerun()
