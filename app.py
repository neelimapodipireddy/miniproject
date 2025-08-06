import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# --- 1. Constants ---
IMG_SIZE = 64
CLASS_NAMES = ['Light', 'Medium', 'Dark']

# --- 2. Load Model ---
@st.cache_resource
def load_model():
    if not os.path.exists('skin_tone_model.h5'):
        st.error("Model file 'skin_tone_model.h5' not found. Please ensure the file is in the correct directory.")
        return None
    try:
        model = tf.keras.models.load_model('skin_tone_model.h5')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()
if model is None:
    st.stop()

# --- 3. Streamlit UI ---
st.title("🧑🏽‍🦱 Skin Tone Classifier")
st.write("Upload an image, and the model will classify it as **Light**, **Medium**, or **Dark** skin tone.")

# --- 4. File Upload ---
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # --- 5. Preprocessing ---
    img_resized = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img_resized).astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # --- 6. Prediction ---
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction[0])
    confidence = prediction[0][predicted_class]

    # --- 7. Output ---
    st.markdown("### 🧠 Prediction")
    st.write(f"**Predicted Class:** {CLASS_NAMES[predicted_class]}")
    st.write(f"**Confidence:** {confidence:.4f}")
