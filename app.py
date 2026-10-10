"""Streamlit interface for the trained Chest X-ray CNN."""
import json
from pathlib import Path
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from config import MODEL_PATH, CLASS_NAMES_PATH, IMAGE_SIZE

st.set_page_config(page_title="Chest X-ray CNN", page_icon="🩻", layout="centered")
st.title("🩻 Chest X-ray Classification")
st.write("Educational demonstration: classify a chest X-ray as one of the model's trained classes.")

st.warning(
    "This application is not a medical device and must not be used to diagnose, "
    "exclude, or make treatment decisions about pneumonia."
)

@st.cache_resource
def load_model_and_classes():
    if not MODEL_PATH.exists() or not CLASS_NAMES_PATH.exists():
        return None, None
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(CLASS_NAMES_PATH, encoding="utf-8") as f:
        class_names = json.load(f)
    return model, class_names

model, class_names = load_model_and_classes()

if model is None:
    st.error("No trained model was found.")
    st.markdown(
        "### Train the model first\n"
        "1. Put the dataset into `data/chest_xray/train/` and `data/chest_xray/test/`.\n"
        "2. Each folder must contain one subfolder per class, for example `NORMAL` and `PNEUMONIA`.\n"
        "3. In the VS Code terminal run: `python train.py`.\n"
        "4. When training finishes, restart or refresh this app."
    )
    st.stop()

st.caption("Class mapping: " + " | ".join(f"{i}: {name}" for i, name in enumerate(class_names)))
uploaded = st.file_uploader("Upload a chest X-ray image", type=["png", "jpg", "jpeg"])

if uploaded is not None:
    try:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, caption="Uploaded image", use_container_width=True)
        resized = image.resize(IMAGE_SIZE)
        array = np.asarray(resized, dtype=np.float32)
        probabilities = model.predict(array[None, ...], verbose=0)[0]
        index = int(np.argmax(probabilities))
        st.subheader(f"Model prediction: {class_names[index]}")
        st.write("Model score (not a calibrated clinical probability): "
                 f"**{float(probabilities[index]) * 100:.1f}%**")
        st.write("Scores by class")
        for i, name in enumerate(class_names):
            st.progress(float(np.clip(probabilities[i], 0, 1)), text=f"{name}: {probabilities[i] * 100:.1f}%")
        st.info(
            "Predictions may be wrong because of image quality, dataset bias, or distribution shift. "
            "A clinician must interpret medical images in context."
        )
    except Exception as exc:
        st.error(f"Could not process this image: {exc}")
