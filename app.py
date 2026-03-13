import streamlit as st
import numpy as np
from ultralytics import YOLO
from PIL import Image

# Load YOLO model
model = YOLO("yolov8n.pt")

st.title("AI Object Detection (YOLOv8)")

st.write("Upload an image to detect objects")

uploaded_file = st.file_uploader("Choose an image", type=["jpg","png","jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    frame = np.array(image)

    results = model(frame)

    annotated_frame = results[0].plot()

    st.image(annotated_frame, caption="Detected Objects", use_container_width=True)
