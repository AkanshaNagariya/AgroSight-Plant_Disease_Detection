# Library imports
import numpy as np
import streamlit as st
import cv2
from keras.models import load_model
import tensorflow as tf

# Loading the Model
model = load_model('plant_disease_model.h5')

# Name of Classes
CLASS_NAMES = ('Tomato-Bacterial_spot', 'Potato-Barly blight', 'Corn-Common_rust')

# Custom CSS for styling
st.markdown("""
    <style>
        .main {
            background-color: #f0f2f6;
        }
        .title {
            font-size:40px;
            font-weight:700;
            color: #2e8b57;
            text-align: center;
            padding: 10px;
        }
        .subheader {
            font-size:20px;
            color: #444444;
            text-align: center;
            margin-bottom: 20px;
        }
        .result {
            font-size:28px;
            font-weight:bold;
            color: #d9534f;
            text-align: center;
            margin-top: 20px;
        }
    </style>
""", unsafe_allow_html=True)

# Title & Instruction
st.markdown('<div class="title">🌿 Plant Disease Detection App</div>', unsafe_allow_html=True)
st.markdown('<div class="subheader">Upload an image of the plant leaf to detect the disease</div>', unsafe_allow_html=True)

# Uploading the image
plant_image = st.file_uploader("📷 Choose an image...", type="jpg")
submit = st.button('🔍 Predict Disease')

# On predict button click
if submit:
    if plant_image is not None:
        # Convert the file to an OpenCV image
        file_bytes = np.asarray(bytearray(plant_image.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)

        # Display the image
        st.image(opencv_image, channels="BGR", caption="Uploaded Leaf Image", use_column_width=True)
        st.write("Image shape:", opencv_image.shape)

        # Resize image
        opencv_image = cv2.resize(opencv_image, (256, 256))

        # Convert image to 4D
        opencv_image.shape = (1, 256, 256, 3)

        # Make prediction
        Y_pred = model.predict(opencv_image)
        result = CLASS_NAMES[np.argmax(Y_pred)]
        disease = result.split('-')[1]

        # Display result
        st.markdown(f'<div class="result">🦠 Detected: <span style="color:#2e8b57;">{disease}</span> Disease</div>', unsafe_allow_html=True)
