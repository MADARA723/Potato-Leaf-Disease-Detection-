import streamlit as st
from gradio_client import Client, handle_file
import tempfile
import os

st.set_page_config(page_title="Potato Disease Detector", layout="centered")

# Connect to your live Hugging Face backend
hf_client = Client("ASITMORE/Potato_Leaf_Disease_Detection")

st.title("🥔 Potato Leaf Disease Classifier")
file = st.file_uploader("Choose a potato leaf image...", type=["jpg", "jpeg", "png"])

if file is not None:
    st.image(file, caption="Uploaded Image", use_container_width=True)
    
    with st.spinner('Analyzing image via Hugging Face Backend...'):
        # Save uploaded file temporarily so gradio_client can transmit it
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
            tmp_file.write(file.getvalue())
            tmp_path = tmp_file.name
            
        try:
            # Matches the exact endpoint and parameter discovered from client.view_api()
            result = hf_client.predict(
                img=handle_file(tmp_path),
                api_name="/predict_disease"
            )
            
            # Parse response string: "Class|Confidence"
            predicted_class, confidence_str = result.split("|")
            confidence = float(confidence_str)
            
            # Display results
            st.subheader(f"Prediction: {predicted_class}")
            st.progress(int(confidence))
            st.write(f"**Confidence Score:** {confidence:.2f}%")

            if predicted_class == "Early Blight":
                st.warning("Advice: Apply fungicides and ensure proper crop rotation.")
            elif predicted_class == "Late Blight":
                st.error("Advice: High risk! Remove infected plants immediately.")
            else:
                st.success("Advice: Your plant looks healthy. Keep maintaining good irrigation!")
            
        except Exception as e:
            st.error(f"Failed to connect to ML backend: {e}")
            
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
