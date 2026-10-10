"""Train a CNN to classify chest X-rays into dataset-defined classes.

Expected data layout:
data/chest_xray/
    train/
        NORMAL/
        PNEUMONIA/
    test/
        NORMAL/
        PNEUMONIA/

The class names are discovered from folder names and saved beside the model.
Images are kept in the 0..255 range; rescaling is part of the model so training
and Streamlit inference use the same preprocessing.
"""
import argparse
import json
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from config import (
    TRAIN_DIR, TEST_DIR, MODEL_PATH, CLASS_NAMES_PATH, HISTORY_PATH,
    IMAGE_SIZE, BATCH_SIZE, SEED, VALIDATION_SPLIT, DEFAULT_EPOCHS
)


def load_datasets(batch_size=BATCH_SIZE):
    if not TRAIN_DIR.exists():
        raise FileNotFoundError(
            f"Training directory not found: {TRAIN_DIR}\n"
            "Set CHEST_XRAY_DATA_DIR or CHEST_XRAY_TRAIN_DIR, or place the "
            "dataset in data/chest_xray/train."
        )
    if not TEST_DIR.exists():
        raise FileNotFoundError(
            f"Test directory not found: {TEST_DIR}\n"
            "Expected data/chest_xray/test/<class_name>/."
        )

    train_ds = keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        validation_split=VALIDATION_SPLIT,
        subset="training",
        seed=SEED,
        image_size=IMAGE_SIZE,
        batch_size=batch_size,
        label_mode="int",
        shuffle=True,
    )
    val_ds = keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        validation_split=VALIDATION_SPLIT,
        subset="validation",
        seed=SEED,
        image_size=IMAGE_SIZE,
        batch_size=batch_size,
        label_mode="int",
        shuffle=False,
    )
    test_ds = keras.utils.image_dataset_from_directory(
        TEST_DIR,
        image_size=IMAGE_SIZE,
        batch_size=batch_size,
        label_mode="int",
        shuffle=False,
    )

    class_names = train_ds.class_names
    if len(class_names) != 2:
        raise ValueError(
            f"This project expects exactly 2 classes, but found {len(class_names)}: "
            f"{class_names}"
        )
    if test_ds.class_names != class_names:
        raise ValueError(
            "Train and test class folders do not match. "
            f"Train classes: {class_names}; test classes: {test_ds.class_names}"
        )

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(autotune)
    val_ds = val_ds.prefetch(autotune)
    test_ds = test_ds.prefetch(autotune)
    return train_ds, val_ds, test_ds, class_names


def build_model(num_classes=2):
    inputs = keras.Input(shape=(*IMAGE_SIZE, 3), name="chest_xray")
    x = layers.Rescaling(1.0 / 255.0, name="rescale")(inputs)

    # Augmentation runs only during training.
    x = layers.RandomRotation(0.04, name="random_rotation")(x)
    x = layers.RandomZoom(0.08, name="random_zoom")(x)
    x = layers.RandomTranslation(0.04, 0.04, name="random_translation")(x)

    x = layers.Conv2D(32, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(64, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(128, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(192, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.40, name="dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="class_probabilities")(x)

    model = keras.Model(inputs, outputs, name="chest_xray_cnn")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    parser = argparse.ArgumentParser(description="Train the chest X-ray CNN.")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1 or args.learning_rate <= 0:
        parser.error("epochs and batch-size must be positive; learning-rate must be > 0.")

    tf.keras.utils.set_random_seed(SEED)
    train_ds, val_ds, test_ds, class_names = load_datasets(args.batch_size)
    print("\nClass mapping (label index -> class):")
    for i, name in enumerate(class_names):
        print(f"  {i} -> {name}")

    model = build_model(len(class_names))
    model.optimizer.learning_rate.assign(args.learning_rate)
    model.summary()

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=str(MODEL_PATH),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=4,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    # Save the best in-memory model as well, ensuring the final artifact matches
    # the restored best validation weights.
    model.save(MODEL_PATH)
    with open(CLASS_NAMES_PATH, "w", encoding="utf-8") as f:
        json.dump(class_names, f, indent=2)
    with open(HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {key: [float(v) for v in values] for key, values in history.history.items()},
            f,
            indent=2,
        )

    print(f"\nSaved model: {MODEL_PATH}")
    print(f"Saved class names: {CLASS_NAMES_PATH}")
    print("Training complete. Run `python evaluate.py` for held-out test evaluation.")


if __name__ == "__main__":
    main()
