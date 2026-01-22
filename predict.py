import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing import image

# Load model
model = tf.keras.models.load_model("buffalo_model.h5")

# Load dataset CSV
data = pd.read_csv("buffalo_dataset.csv")

# Class labels (same order as folders)
class_names = sorted(list(data["Breed"].str.lower()))

# Load image
img_path = "test.jpg"   # Put any buffalo image here
img = image.load_img(img_path, target_size=(224,224))
img_array = image.img_to_array(img)/255.0
img_array = np.expand_dims(img_array, axis=0)

# Predict
pred = model.predict(img_array)
breed = class_names[np.argmax(pred)]

# Get details
details = data[data["Breed"].str.lower() == breed]

print("\nPredicted Breed:", breed.capitalize())
print(details)
