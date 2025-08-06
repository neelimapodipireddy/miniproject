import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten

# Example model
model = Sequential([
    Flatten(input_shape=(64, 64, 3)),  # Input shape (e.g., 64x64 RGB images)
    Dense(128, activation='relu'),
    Dense(3, activation='softmax')  # Assuming 3 classes (Light, Medium, Dark)
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Dummy data for demonstration (replace with your dataset)
import numpy as np
x_train = np.random.random((100, 64, 64, 3))  # 100 random images
y_train = np.random.randint(3, size=(100,))   # 100 random labels (0, 1, 2)

# Train the model
model.fit(x_train, y_train, epochs=5)

# Save the model
model.save('skin_tone_model.h5')
print("Model saved as 'skin_tone_model.h5'")
