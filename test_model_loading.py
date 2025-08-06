from tensorflow.keras.models import load_model

try:
    model = load_model('skin_tone_model.h5')
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
