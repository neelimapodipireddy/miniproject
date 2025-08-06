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
    # Simple heuristic: Detect lipstick by checking for high red/pink pixel ratio
    np_img = np.array(image)
    # Focus on lower half of the image (where lips are likely to be)
    h = np_img.shape[0]
    lower_half = np_img[h//2:]
    # Count pixels that are 'red/pink' (R > 120, G < 80, B < 100)
    red_pixels = ((lower_half[:,:,0] > 120) & (lower_half[:,:,1] < 80) & (lower_half[:,:,2] < 100)).sum()
    total_pixels = lower_half.shape[0] * lower_half.shape[1]
    red_ratio = red_pixels / total_pixels
    cosmetics = []
    if red_ratio > 0.02:  # If more than 2% of lower half pixels are red/pink
        cosmetics.append("Lipstick")
    # Foundation detection is not implemented (would require face/skin analysis)
    return cosmetics

def detect_camera_filter(image):
    # Placeholder logic: In real use, replace with ML model or heuristics
    # Example: return "Sepia" if brown/yellow tint detected
    return "None (No obvious filter detected)"  # Stub result


import pandas as pd
import matplotlib.pyplot as plt

results = []
feedbacks = []
skin_tone_counts = {name: 0 for name in CLASS_NAMES}
if uploaded_files:
    show_probs = st.checkbox("Show all class probabilities for each image", value=True)
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
        st.progress(float(confidence), text=f"Confidence: {confidence:.2%}")

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
        if show_probs:
            st.markdown("##### 🔍 All Class Probabilities")
            for i, class_name in enumerate(CLASS_NAMES):
                st.write(f"{class_name}: {prediction[0][i]:.4f}")

        # Collect results for CSV
        results.append({
            "Image Name": uploaded_file.name,
            "Predicted Skin Tone": CLASS_NAMES[predicted_class],
            "Confidence": confidence,
            "Cosmetics": ", ".join(cosmetics) if cosmetics else "None",
            "Camera Filter": camera_filter
        })

        # Count for pie chart
        skin_tone_counts[CLASS_NAMES[predicted_class]] += 1

        # User feedback
        feedback = st.radio(
            f"Was the prediction correct for Image {idx+1}?",
            ("Yes", "No"),
            key=f"feedback_{idx}"
        )
        feedbacks.append({
            "Image Name": uploaded_file.name,
            "User Feedback": feedback
        })

    # Pie chart of predictions
    if sum(skin_tone_counts.values()) > 0:
        st.markdown("---")
        st.markdown("### 📊 Skin Tone Prediction Distribution")
        fig, ax = plt.subplots()
        ax.pie(list(skin_tone_counts.values()), labels=list(skin_tone_counts.keys()), autopct='%1.1f%%', startangle=90)
        ax.axis('equal')
        st.pyplot(fig)

    # Download CSV button for results
    st.markdown("---")
    st.markdown("### 📥 Download Results")
    df_results = pd.DataFrame(results)
    st.download_button(
        label="Download CSV of Results",
        data=df_results.to_csv(index=False),
        file_name="skin_tone_results.csv",
        mime="text/csv"
    )

    # Download CSV button for feedback
    df_feedback = pd.DataFrame(feedbacks)
    st.download_button(
        label="Download CSV of User Feedback",
        data=df_feedback.to_csv(index=False),
        file_name="user_feedback.csv",
        mime="text/csv"
    )
