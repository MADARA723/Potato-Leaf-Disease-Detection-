import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

st.set_page_config(page_title="Potato Disease Detector", layout="centered")

# Helper function to load model safely
@st.cache_resource
def load_model():
    model_path = 'model.keras' # Ensure this matches your file name on GitHub
    if os.path.exists(model_path):
        return tf.keras.models.load_model(model_path)
    else:
        st.error(f"Model file '{model_path}' not found. Please check your GitHub repository.")
        return None

model = load_model()
class_names = ['Late Blight', 'Early Blight', 'Healthy']

st.title("🥔 Potato Leaf Disease Classifier")
st.write("Upload a leaf image to detect diseases and see the confidence score.")

file = st.file_uploader("Choose a potato leaf image...", type=["jpg", "jpeg", "png"])

if file is not None and model is not None:
    # 1. Display the uploaded image
    img = Image.open(file)
    st.image(img, caption="Uploaded Image", use_container_width=True)
    
    # 2. Preprocessing (Matching your model1.ipynb logic)
    img_resized = img.resize((256, 256))
    img_array = np.array(img_resized)
    
    # In your main.py, you didn't divide by 255. 
    # If your model has a Rescaling layer, don't divide here.
    # If it DOES NOT, keep the / 255.0
    img_batched = np.expand_dims(img_array, axis=0)
    
    # 3. Prediction Logic (Like main.py)
    with st.spinner('Analyzing image...'):
        predictions = model.predict(img_batched)
        
        # Calculate Predicted Class and Confidence (Percentage)
        predicted_class = class_names[np.argmax(predictions[0])]
        confidence = np.max(predictions[0]) * 100
    
    # 4. Result Display
    st.subheader(f"Prediction: {predicted_class}")
    
    # Progress bar for confidence
    st.progress(int(confidence))
    st.write(f"**Confidence Score:** {confidence:.2f}%")

    # Business Logic / Advice
    if predicted_class == "Early Blight":
        st.warning("Advice: Apply fungicides and ensure proper crop rotation.")
    elif predicted_class == "Late Blight":
        st.error("Advice: High risk! Remove infected plants immediately to prevent spreading.")
    else:
        st.success("Advice: Your plant looks healthy. Keep maintaining good irrigation!")

