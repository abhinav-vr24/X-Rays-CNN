"""Train the baseline CNN. A validation subset is created from train/ automatically."""
import argparse, json
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from config import (
    MODEL_PATH, CLASS_NAMES_PATH, HISTORY_PATH, IMAGE_SIZE, BATCH_SIZE,
    SEED, DEFAULT_EPOCHS
)
from data_utils import load_train_validation

def build_model(num_classes=2, dropout=0.40, activation="relu", filters=(32, 64, 128, 192)):
    inputs = keras.Input(shape=(*IMAGE_SIZE, 3), name="chest_xray")
    x = layers.Rescaling(1.0 / 255.0)(inputs)
    x = layers.RandomRotation(0.04)(x)
    x = layers.RandomZoom(0.08)(x)
    x = layers.RandomTranslation(0.04, 0.04)(x)
    for i, f in enumerate(filters):
        x = layers.Conv2D(f, 3, padding="same", use_bias=False,
                          kernel_initializer="he_normal")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation(activation)(x)
        if i < len(filters) - 1:
            x = layers.MaxPooling2D()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation=activation)(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = keras.Model(inputs, outputs, name="baseline_chest_xray_cnn")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    args = p.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.learning_rate <= 0:
        p.error("epochs/batch-size must be positive and learning-rate must be > 0")
    tf.keras.utils.set_random_seed(SEED)
    train_ds, val_ds, class_names = load_train_validation(args.batch_size)
    print("\nClass mapping:")
    for i, name in enumerate(class_names):
        print(f"  {i}: {name}")
    model = build_model(len(class_names))
    model.optimizer.learning_rate.assign(args.learning_rate)
    model.summary()
    callbacks = [
        keras.callbacks.ModelCheckpoint(str(MODEL_PATH), monitor="val_loss",
                                        save_best_only=True, verbose=1),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=4,
                                      restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                          patience=2, min_lr=1e-6, verbose=1)
    ]
    history = model.fit(train_ds, validation_data=val_ds,
                        epochs=args.epochs, callbacks=callbacks)
    model.save(MODEL_PATH)
    CLASS_NAMES_PATH.write_text(json.dumps(class_names, indent=2), encoding="utf-8")
    HISTORY_PATH.write_text(json.dumps(
        {k: [float(x) for x in v] for k, v in history.history.items()}, indent=2
    ), encoding="utf-8")
    print(f"\nSaved model: {MODEL_PATH}")
    print("Next run: python evaluate.py")

if __name__ == "__main__":
    main()
