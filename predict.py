"""Predict the class of a single chest X-ray image from the command line."""
import argparse
import json
import numpy as np
import tensorflow as tf
from PIL import Image
from config import MODEL_PATH, CLASS_NAMES_PATH, IMAGE_SIZE


def predict_image(image_path):
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run `python train.py` first.")
    if not CLASS_NAMES_PATH.exists():
        raise FileNotFoundError(f"Class mapping not found at {CLASS_NAMES_PATH}. Run `python train.py` first.")

    with open(CLASS_NAMES_PATH, encoding="utf-8") as f:
        class_names = json.load(f)

    image = Image.open(image_path).convert("RGB").resize(IMAGE_SIZE)
    array = np.asarray(image, dtype=np.float32)
    probabilities = tf.keras.models.load_model(MODEL_PATH).predict(array[None, ...], verbose=0)[0]
    index = int(np.argmax(probabilities))
    return class_names[index], float(probabilities[index]), {
        name: float(probabilities[i]) for i, name in enumerate(class_names)
    }


def main():
    parser = argparse.ArgumentParser(description="Predict Normal/Pneumonia from a chest X-ray.")
    parser.add_argument("image", help="Path to a chest X-ray image file.")
    args = parser.parse_args()
    label, confidence, scores = predict_image(args.image)
    print(f"Predicted class: {label}")
    print(f"Model score for predicted class: {confidence:.4f}")
    print("Class scores:")
    for name, score in scores.items():
        print(f"  {name}: {score:.4f}")
    print("\nEducational demonstration only; not for clinical diagnosis.")


if __name__ == "__main__":
    main()
