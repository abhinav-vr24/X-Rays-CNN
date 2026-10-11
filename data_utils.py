"""Shared dataset utilities. Validation data is split from the training folder."""
import tensorflow as tf
from tensorflow import keras
from config import TRAIN_DIR, TEST_DIR, IMAGE_SIZE, BATCH_SIZE, SEED, VALIDATION_SPLIT

def load_train_validation(batch_size=BATCH_SIZE):
    if not TRAIN_DIR.is_dir():
        raise FileNotFoundError(
            f"Training folder not found: {TRAIN_DIR}\n"
            "Expected data/chest_xray/train/NORMAL and data/chest_xray/train/PNEUMONIA."
        )
    train_ds = keras.utils.image_dataset_from_directory(
        TRAIN_DIR, validation_split=VALIDATION_SPLIT, subset="training",
        seed=SEED, image_size=IMAGE_SIZE, batch_size=batch_size,
        label_mode="int", shuffle=True
    )
    val_ds = keras.utils.image_dataset_from_directory(
        TRAIN_DIR, validation_split=VALIDATION_SPLIT, subset="validation",
        seed=SEED, image_size=IMAGE_SIZE, batch_size=batch_size,
        label_mode="int", shuffle=False
    )
    names = train_ds.class_names
    if len(names) != 2:
        raise ValueError(f"Expected exactly two class folders, found: {names}")
    if val_ds.class_names != names:
        raise ValueError("Could not match training and validation class folders.")
    autotune = tf.data.AUTOTUNE
    return train_ds.prefetch(autotune), val_ds.prefetch(autotune), names

def load_test(batch_size=BATCH_SIZE, expected_class_names=None):
    if not TEST_DIR.is_dir():
        raise FileNotFoundError(
            f"Test folder not found: {TEST_DIR}\n"
            "Expected data/chest_xray/test/NORMAL and data/chest_xray/test/PNEUMONIA."
        )
    ds = keras.utils.image_dataset_from_directory(
        TEST_DIR, image_size=IMAGE_SIZE, batch_size=batch_size,
        label_mode="int", shuffle=False
    )
    if expected_class_names is not None and ds.class_names != list(expected_class_names):
        raise ValueError(
            f"Test class folders {ds.class_names} do not match training classes "
            f"{list(expected_class_names)}."
        )
    return ds.prefetch(tf.data.AUTOTUNE), ds.class_names
