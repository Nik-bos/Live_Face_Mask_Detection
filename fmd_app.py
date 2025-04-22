# Before moving ahead, kindly note that this code is taken from ChatGPT and I dont know streamlit and cv2 .
# So, don't waste time in understandiing the code


import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import numpy as np

import gdown
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase


model = load_model(r'FMD.h5')    # Loading model
# https://drive.google.com/file/d/1sYkB8epf6y0JrVLDis4n8O-Z9dMjZX8A/view?usp=drive_link    Google's like to the saved mofel

# Only download if it doesn't already exist
if not os.path.exists(model_path):
    file_id = "1sYkB8epf6y0JrVLDis4n8O-Z9dMjZX8A"
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, model_path, quiet=False)


best_threshold = 0.4560450613498688

# Creating require functions

def detect_face_mask(img):
    img = cv2.resize(img, (224,224))    # Resizing image
    norm_img = img/255.0    # Normalizing image
    batch_img = np.expand_dims(norm_img, axis = 0)

    pred = model.predict(batch_img, verbose = 0)    # Verbose = 0 will not show progress while processing the image

    return 'Without Mask' if pred> best_threshold else 'With Mask'

    # We don't need to read image here bcz when it will capture images from videos and store it as numpy array in 'frame' variable

def draw_label(img, text, position, bg_color):
    # Get text size to fit rectangle around text
    text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)

    # Calculate bottom-right coordinates of rectangle
    end_x = position[0] + text_size[0] + 10
    end_y = position[1] + text_size[1] + 10

    # Draw filled rectangle (background for the label)
    cv2.rectangle(img, position, (end_x, end_y), bg_color, -1)  # -1 to fill the rectangle

    # Draw the text inside the rectangle
    cv2.putText(img, text,
                (position[0] + 5, position[1] + text_size[1] + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)


class FaceMaskDetector(VideoTransformerBase):
    
    def transform(self, frame):
        img = frame.to_ndarray(format = 'bgr24')

        label = detect_face_mask(img)

        if label == 'With Mask':
            draw_label(img, label, (30,30), (0, 255, 0))    # (0,255,0) Stands for (B,G,R)
        else:
            draw_label(img, label, (30,30), (0, 0, 255))    # (0,255,0) Stands for (B,G,R)

        return img

st.title("Live Face Mask Detection")
st.markdown("Allow camera access to your browser to see live prediction.")

webrtc_streamer(key = 'mask_detection', video_transformer_factory = FaceMaskDetector)
