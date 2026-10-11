"""Predict one X-ray from the terminal using the saved baseline model."""
import argparse, json
import numpy as np
import tensorflow as tf
from config import MODEL_PATH, CLASS_NAMES_PATH, IMAGE_SIZE

def main():
    p = argparse.ArgumentParser()
    p.add_argument("image", help="Path to one chest X-ray image")
    args = p.parse_args()
    if not MODEL_PATH.exists() or not CLASS_NAMES_PATH.exists():
        raise FileNotFoundError("Model not found. Run `python train.py` first.")
    class_names = json.loads(CLASS_NAMES_PATH.read_text(encoding="utf-8"))
    img = tf.keras.utils.load_img(args.image, target_size=IMAGE_SIZE)
    arr = tf.keras.utils.img_to_array(img)
    probs = tf.keras.models.load_model(MODEL_PATH).predict(np.expand_dims(arr, 0), verbose=0)[0]
    idx = int(np.argmax(probs))
    print(f"Prediction: {class_names[idx]}")
    print("Class probabilities:")
    for name, probability in zip(class_names, probs):
        print(f"  {name}: {float(probability):.4f}")
    print("Educational project only — not a medical diagnosis.")

if __name__ == "__main__":
    main()
