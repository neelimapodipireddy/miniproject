# app.py

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# --- Constants ---
IMG_SIZE = 64
CLASS_NAMES = ['Light', 'Medium', 'Dark']  # Match your folder names in correct order

# --- Load Model ---
@st.cache_resource
def load_model():
    if not os.path.exists('skin_tone_model.h5'):
        st.error("Model file not found. Please train the model first.")
        return None
    return tf.keras.models.load_model('skin_tone_model.h5')

model = load_model()
if model is None:
    st.stop()

# --- Streamlit UI ---
st.title("🧑🏽‍🦱 Skin Tone Classifier")
st.write("Upload an image, and the model will classify it as Light, *Medium, or **Dark* skin tone.")

# --- File Upload ---
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img_resized = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img_resized).astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction[0])
    confidence = prediction[0][predicted_class]

    st.markdown("### 🧠 Prediction")
    st.write(f"Predicted Class: {CLASS_NAMES[predicted_class]}")
    st.write(f"Confidence: {confidence:.4f}")

    # Optional: Show all class probabilities
    st.markdown("#### 🔍 All Class Probabilities")
    for i, class_name in enumerate(CLASS_NAMES):
        st.write(f"{class_name}: {prediction[0][i]:.4f}")
