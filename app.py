import streamlit as st
import numpy as np
from PIL import Image
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av

# Load the YOLOv8 model
model = YOLO('C:\WORLD\yolo-env\webapp\models\yolo-object-detection\yolov8m.onnx', task='detect')

# Function to perform object detection on an image
def detect_objects(image):
    results = model(image)
    annotated_image = results[0].plot()  # Get the annotated image with bounding boxes and labels
    return annotated_image

# Class for WebRTC video processing
class YOLOVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.yolo = YOLO('C:\WORLD\yolo-env\webapp\models\yolo-object-detection\yolov8m.onnx', task='detect')

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        pred_img = detect_objects(img)
        return av.VideoFrame.from_ndarray(pred_img, format="bgr24")

# Streamlit UI
st.title("YOLOv8 Object Detection Web App")

st.write("""
         Upload an image or use real-time video streaming, and the app will detect objects using a YOLOv8 model.
         """)

# Select between image and real-time options
option = st.selectbox("Choose an option:", ["Image", "Real-time Video"], key="option_select")

if option == "Image":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], key="image_uploader")

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)  # Removed key argument

        # Convert to OpenCV format
        opencv_image = np.array(image.convert('RGB'))

        # Perform object detection
        st.write("Detecting objects...")
        detected_image = detect_objects(opencv_image)

        # Display the output image
        st.image(detected_image, caption='Detected Image.', use_column_width=True)  # Removed key argument

elif option == "Real-time Video":
    st.write("Starting real-time video stream...")
    
    webrtc_streamer(key="real_time_video_streamer",
                    video_processor_factory=YOLOVideoProcessor,
                    media_stream_constraints={"video": True, "audio": False},
                    mode=WebRtcMode.SENDRECV)
