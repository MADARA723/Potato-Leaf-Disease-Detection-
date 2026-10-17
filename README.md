# 🥔 Potato Leaf Disease Classifier: Decoupled AI Microservice

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange)
![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-red)
![Hugging Face](https://img.shields.io/badge/Backend-Hugging_Face-yellow)
![Architecture](https://img.shields.io/badge/Architecture-Microservice-success)

## 📌 Project Overview
This project is an end-to-end Machine Learning web application that detects agricultural diseases (Early Blight and Late Blight) in potato plants from leaf images. 

Rather than deploying a monolithic application, this project is built using a **decoupled microservice architecture**. By separating the lightweight user interface from the heavy machine learning inference engine, the application achieves high scalability, bypasses strict cloud memory limits, and replicates industry-standard production environments.

## 🏗️ Architecture & Data Flow

The application is split into two distinct environments communicating via a RESTful API tunnel:

1. **Frontend (Streamlit Community Cloud):** A lightweight, highly responsive user interface. It handles image capture and user interaction, utilizing near-zero RAM.
2. **Backend API (Hugging Face Gradio Space):** A dedicated machine learning inference server. It loads the heavy `model.keras` TensorFlow weights into memory and exposes a secure prediction endpoint.

### System Workflow
1. The user uploads a leaf image via the **Streamlit** frontend.
2. The Streamlit server temporarily stores the image and securely authenticates with the backend using a hidden `HF_TOKEN`.
3. The image is transmitted via the `gradio_client` API payload to the **Hugging Face** server.
4. The backend preprocesses the image, runs the TensorFlow inference, and returns a serialized string `(Class|Confidence)`.
5. The frontend parses the response and renders the prediction and actionable agricultural advice to the user.

## 🛠️ Technologies Used
* **Deep Learning:** TensorFlow / Keras, Convolutional Neural Networks (CNNs)
* **Backend API:** Python, Gradio API endpoints, Pillow, NumPy
* **Frontend UI:** Streamlit
* **DevOps & Security:** Streamlit Secrets Management, Token Authentication

## 💡 Engineering Challenges Solved

* **Overcoming Cloud Memory Constraints:** Free-tier cloud hosting (like Streamlit Cloud) typically limits memory to 1GB, which is immediately consumed by TensorFlow overhead. Decoupling the app allowed the frontend to run smoothly on Streamlit while offloading the heavy computational lifting to a 16GB Hugging Face container.
* **Bypassing ZeroGPU Quota Limits:** To ensure 100% uptime without hitting anonymous API rate limits, the client-server connection was secured using Hugging Face Access Tokens injected dynamically through Streamlit's Vault (`st.secrets`).
* **Hardware Compatibility:** Engineered a dummy `@spaces.GPU` decorator workaround to satisfy Hugging Face's server startup checks while forcing TensorFlow to evaluate safely on the CPU, preventing tensor memory allocation crashes.

## 🚀 Future Improvements
* Containerize the backend using Docker and FastAPI for provider-agnostic deployment.
* Expand the dataset to include multi-crop disease detection.
* Implement a PostgreSQL database to track disease outbreak locations geographically. 

---
*Developed as part of a portfolio demonstrating scalable AI deployment strategies.*
