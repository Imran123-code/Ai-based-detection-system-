import streamlit as st
import numpy as np
from ultralytics import YOLO
from PIL import Image

from ultralytics import YOLO
from google.colab.output import eval_js
from base64 import b64decode
from PIL import Image
import numpy as np
import io
import time
from IPython.display import display, Javascript, clear_output

# Load model
model = YOLO("yolov8n.pt")

# Start camera
display(Javascript('''
async function setupCamera(){
  const video=document.createElement('video');
  const stream=await navigator.mediaDevices.getUserMedia({video:true});

  document.body.appendChild(video);
  video.srcObject=stream;
  await video.play();

  window.video=video;
  window.stream=stream;
}
setupCamera();
'''))

time.sleep(3)

# Capture frame function
def capture_frame():
  js = '''
  async function capture(){
    const canvas = document.createElement("canvas");
    canvas.width = window.video.videoWidth;
    canvas.height = window.video.videoHeight;
    canvas.getContext("2d").drawImage(window.video, 0, 0);
    return canvas.toDataURL("image/jpeg", 0.8);
  }
  capture()
  '''
  
  data = eval_js(js)
  binary = b64decode(data.split(',')[1])
  img = Image.open(io.BytesIO(binary))
  return np.array(img)

# Detection loop
for i in range(20):

  frame=capture_frame()

  results=model(frame)
  annotated=results[0].plot()

  clear_output(wait=True)
  display(Image.fromarray(annotated))

  time.sleep(0.3)

# Stop camera after loop
try:
    eval_js("window.stream.getTracks().forEach(track => track.stop())")
except:
    pass



