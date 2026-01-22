"""import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models

# Image settings
IMG_SIZE = 224
BATCH_SIZE = 32

# Dataset loading
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_data = datagen.flow_from_directory(
    "dataset",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training"
)

val_data = datagen.flow_from_directory(
    "dataset",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation"
)

# Load pre-trained model
base_model = MobileNetV2(include_top=False, weights="imagenet", input_shape=(224,224,3))
base_model.trainable = False

# Build model
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation="relu"),
    layers.Dense(train_data.num_classes, activation="softmax")
])

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Train model
model.fit(train_data, validation_data=val_data, epochs=5)

# Save model
model.save("buffalo_model.h5")

print("Model trained and saved!")"""

import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

print("🚀 Buffalo Classifier Starting...")

# Load MobileNetV2 from TF Hub (NO DOWNLOAD ISSUES)
hub_url = "https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/5"
base_model = hub.KerasLayer(hub_url, trainable=False, input_shape=(224, 224, 3))

# Build full model
inputs = tf.keras.Input(shape=(224, 224, 3))
x = base_model(inputs)
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
outputs = Dense(2, activation='softmax')(x)
model = Model(inputs, outputs)

model.compile(optimizer=Adam(0.001), loss='categorical_crossentropy', metrics=['accuracy'])
print("✅ Model ready!")

# UPDATE THIS PATH TO YOUR DATA FOLDER
train_folder = "dataset/train"  # ← CHANGE THIS
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
train_generator = train_datagen.flow_from_directory(train_folder, target_size=(224,224), batch_size=32, class_mode='categorical', subset='training')
val_generator = train_datagen.flow_from_directory(train_folder, target_size=(224,224), batch_size=32, class_mode='categorical', subset='validation')

# Train
model.fit(train_generator, epochs=10, validation_data=val_generator)
model.save('buffalo_model.h5')
print("🎉 DONE! Model saved.")
