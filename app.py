import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# 1. Load the model you already have
model = tf.keras.models.load_model('model.h5')
class_names = ['Early Blight', 'Late Blight', 'Healthy']

st.title("Potato Leaf Disease Classifier")

# 2. Upload Image
file = st.file_uploader("Upload Leaf Image", type=["jpg", "png"])

if file:
    img = Image.open(file).resize((256, 256)) # Resizing to match your training
    st.image(img)
    
    # 3. Predict
    img_array = np.array(img) / 255.0 # Normalize if you did in training
    img_batched = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_batched)
    result = class_names[np.argmax(prediction)]
    
    st.success(f"Result: {result}")