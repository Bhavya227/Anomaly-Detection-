import streamlit as st
from PIL import Image, ImageOps
import numpy as np
from keras.models import load_model
import os

# ---- PAGE CONFIG ----
st.set_page_config(
    page_title="Toothbrush Anomaly Detector",
    page_icon="🪥",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ---- TITLE ----
st.markdown(
    """
    <h1 style='text-align: center;'>🪥 Toothbrush Anomaly Detector</h1>
    <p style='text-align: center;'>AI-powered app to detect defective toothbrushes</p>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

# ---- SIDEBAR ----
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2752/2752807.png", width=100)
    st.header("🔍 About")
    st.markdown(
        """
        This AI app helps you classify toothbrush images as either:
        - ✅ Good
        - ⚠️ Anomaly

        Model is trained using [Teachable Machine](https://teachablemachine.withgoogle.com/) with TensorFlow.
        """
    )
    st.markdown("---")
    st.info("Upload a toothbrush image or use your camera to test it!")

# ---- LOAD MODEL & LABELS ----
@st.cache_resource
def load_tm_model():
    try:
        model = load_model("keras_model.h5", compile=False)
        labels = open("labels.txt", "r").readlines()
        return model, labels
    except Exception as e:
        st.error(f"❌ Error loading model or labels:\n{e}")
        return None, None

model, class_names = load_tm_model()

# ---- IMAGE PROCESSING ----
def predict_image(image):
    try:
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.LANCZOS)
        image_array = np.asarray(image)
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        data[0] = normalized_image_array

        prediction = model.predict(data)
        index = np.argmax(prediction)
        class_name = class_names[index].strip()
        confidence_score = prediction[0][index]
        return class_name, confidence_score
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        return None, None

# ---- IMAGE INPUT ----
st.subheader("📸 Choose Image Input Method")
input_method = st.radio("", ["📁 Upload Image", "📷 Use Camera"])

image = None
if input_method == "📁 Upload Image":
    uploaded_file = st.file_uploader("Upload a toothbrush image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="🖼️ Uploaded Image", width=300)

elif input_method == "📷 Use Camera":
    camera_image = st.camera_input("Take a photo")
    if camera_image:
        image = Image.open(camera_image)
        st.image(image, caption="📸 Captured Image", width=300)

# ---- PREDICTION BUTTON ----
if image is not None and st.button("🧠 Detect Anomaly"):
    st.subheader("🔎 Prediction Result")
    with st.spinner("Analyzing image..."):
        label, confidence = predict_image(image)

        if label:
            st.markdown(
                f"""
                <div style="border: 1px solid #eee; border-radius: 10px; padding: 1.2em; background-color: #f9f9f9;">
                    <h3 style="margin-bottom: 0;">🧾 Prediction: <strong>{label}</strong></h3>
                    <p style="margin-top: 0.5em;">🔢 Confidence Score: <strong>{confidence:.2f}</strong></p>
                </div>
                """,
                unsafe_allow_html=True
            )

            if "anomaly" in label.lower():
                st.error("⚠️ Anomaly Detected! Please check the toothbrush.")
            elif "good" in label.lower():
                st.success("✅ This is a Good Toothbrush!")
            else:
                st.warning("🤔 Prediction is uncertain. Please review image quality.")
