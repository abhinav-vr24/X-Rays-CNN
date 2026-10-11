"""Simple Streamlit interface for the trained model."""
import json
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from config import MODEL_PATH, CLASS_NAMES_PATH, IMAGE_SIZE

st.set_page_config(page_title="Chest X-ray Classifier", page_icon="🩻")
st.title("Chest X-ray Classification")
st.write("Upload an X-ray image to get a model prediction.")
st.warning("For learning and demonstration only. This is not a medical diagnostic tool.")
if not MODEL_PATH.exists() or not CLASS_NAMES_PATH.exists():
    st.error("No trained model found. In the project terminal, run `python train.py` first.")
    st.stop()

@st.cache_resource
def get_model():
    return tf.keras.models.load_model(MODEL_PATH)

class_names = json.loads(CLASS_NAMES_PATH.read_text(encoding="utf-8"))
uploaded = st.file_uploader("Choose a chest X-ray image", type=["png", "jpg", "jpeg"])
if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded image", use_container_width=True)
    if st.button("Classify image"):
        image = image.resize((IMAGE_SIZE[1], IMAGE_SIZE[0]))
        arr = np.expand_dims(np.asarray(image, dtype=np.float32), axis=0)
        probs = get_model().predict(arr, verbose=0)[0]
        idx = int(np.argmax(probs))
        st.subheader(f"Prediction: {class_names[idx]}")
        for name, probability in zip(class_names, probs):
            st.write(f"{name}: {float(probability):.2%}")
