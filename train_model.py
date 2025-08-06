#trainmodel.py
# train_skin_tone_model.py

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing import image_dataset_from_directory

# --- 1. Load Dataset ---
IMG_SIZE = (64, 64)
BATCH_SIZE = 32
DATA_DIR = "skin_tone_dataset"

dataset = image_dataset_from_directory(
    DATA_DIR,
    image_size=IMG_SIZE,
    label_mode='int',
    batch_size=BATCH_SIZE,
    shuffle=True
)

class_names = dataset.class_names
print("Classes:", class_names)

# --- 2. Preprocessing ---
AUTOTUNE = tf.data.AUTOTUNE
dataset = dataset.prefetch(buffer_size=AUTOTUNE)

# --- 3. Model ---
base_model = MobileNetV2(input_shape=IMG_SIZE + (3,), include_top=False, weights='imagenet')
base_model.trainable = False

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dense(len(class_names), activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# --- 4. Train ---
model.fit(dataset, epochs=10)

# --- 5. Save ---
model.save("skin_tone_model.h5")
print("✅ Model trained and saved as 'skin_tone_model.h5'")

