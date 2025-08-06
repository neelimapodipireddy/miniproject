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

# --- File Upload (Multi-image) ---
uploaded_files = st.file_uploader("Choose image(s)...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

def detect_cosmetics(image):
    # Placeholder logic: In real use, replace with ML model or image analysis
    # Example: return ["Lipstick", "Foundation"] if red/pink tones detected
    return ["Lipstick", "Foundation"]  # Stub result

def detect_camera_filter(image):
    # Placeholder logic: In real use, replace with ML model or heuristics
    # Example: return "Sepia" if brown/yellow tint detected
    return "None (No obvious filter detected)"  # Stub result

if uploaded_files:
    for idx, uploaded_file in enumerate(uploaded_files):
        st.markdown(f"---\n### Image {idx+1}")
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption=f"Uploaded Image {idx+1}", use_column_width=True)

        img_resized = image.resize((IMG_SIZE, IMG_SIZE))
        img_array = np.array(img_resized).astype('float32') / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction[0])
        confidence = prediction[0][predicted_class]

        st.markdown("#### 🧠 Prediction")
        st.write(f"Predicted Skin Tone: {CLASS_NAMES[predicted_class]}")
        st.write(f"Confidence: {confidence:.4f}")

        cosmetics = detect_cosmetics(image)
        st.markdown("#### 💄 Detected Cosmetics (Experimental)")
        if cosmetics:
            st.write(", ".join(cosmetics))
        else:
            st.write("No obvious cosmetics detected.")

        camera_filter = detect_camera_filter(image)
        st.markdown("#### 📷 Camera Filter (Experimental)")
        st.write(camera_filter)

        # Optional: Show all class probabilities
        st.markdown("##### 🔍 All Class Probabilities")
        for i, class_name in enumerate(CLASS_NAMES):
            st.write(f"{class_name}: {prediction[0][i]:.4f}")
