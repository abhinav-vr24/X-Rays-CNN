"""Train ImageNet-pretrained ResNet50 or DenseNet121; validation comes from train/."""
import argparse
import csv
import json
import time
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from config import BATCH_SIZE, IMAGE_SIZE, SEED, MODEL_DIR, REPORT_DIR
from data_utils import load_train_validation


@keras.utils.register_keras_serializable(package="ChestXRay")
class ResNetPreprocess(layers.Layer):
    """ResNet50 preprocessing: RGB to BGR, then subtract ImageNet channel means."""
    def call(self, inputs):
        x = tf.reverse(inputs, axis=[-1])
        means = tf.constant([103.939, 116.779, 123.680], dtype=x.dtype)
        return x - means


@keras.utils.register_keras_serializable(package="ChestXRay")
class DenseNetPreprocess(layers.Layer):
    """DenseNet ImageNet preprocessing (RGB channel mean/std normalization)."""
    def call(self, inputs):
        x = inputs / 255.0
        mean = tf.constant([0.485, 0.456, 0.406], dtype=x.dtype)
        std = tf.constant([0.229, 0.224, 0.225], dtype=x.dtype)
        return (x - mean) / std


def build_transfer_model(model_name, num_classes=2, dropout=0.35):
    name = model_name.lower()
    inputs = keras.Input(shape=(*IMAGE_SIZE, 3), name="xray_rgb_0_255")
    x = layers.RandomRotation(0.03, name="augmentation_rotation")(inputs)
    x = layers.RandomZoom(0.05, name="augmentation_zoom")(x)

    if name == "resnet50":
        x = ResNetPreprocess(name="resnet50_preprocessing")(x)
        backbone = keras.applications.ResNet50(
            include_top=False, weights="imagenet", input_shape=(*IMAGE_SIZE, 3)
        )
    elif name == "densenet121":
        x = DenseNetPreprocess(name="densenet121_preprocessing")(x)
        backbone = keras.applications.DenseNet121(
            include_top=False, weights="imagenet", input_shape=(*IMAGE_SIZE, 3)
        )
    else:
        raise ValueError("model_name must be 'resnet50' or 'densenet121'")

    backbone.trainable = False
    # Keep pretrained BatchNorm layers in inference mode during both stages.
    x = backbone(x, training=False)
    x = layers.GlobalAveragePooling2D(name="global_average_pooling")(x)
    x = layers.BatchNormalization(name="head_batch_normalization")(x)
    x = layers.Dropout(dropout, name="head_dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="class_predictions")(x)
    return keras.Model(inputs, outputs, name=f"{name}_transfer_learning"), backbone


def compile_model(model, learning_rate):
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])


def main():
    parser = argparse.ArgumentParser(description="Transfer learning for chest X-ray classification")
    parser.add_argument("--model", choices=["resnet50", "densenet121"], required=True)
    parser.add_argument("--epochs", type=int, default=8, help="Epochs for training the new classifier head")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--fine-tune", action="store_true", help="Optionally fine-tune final backbone layers")
    parser.add_argument("--fine-tune-epochs", type=int, default=5)
    parser.add_argument("--unfreeze-layers", type=int, default=30)
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.fine_tune_epochs, args.unfreeze_layers) < 1:
        parser.error("epochs, batch-size, fine-tune-epochs and unfreeze-layers must be positive")

    tf.keras.utils.set_random_seed(SEED)
    train_ds, val_ds, class_names = load_train_validation(args.batch_size)
    print("Class mapping:", {i: name for i, name in enumerate(class_names)})
    print("No val/ folder is needed: 20% of train/ is used for validation.")
    print("The test folder is not used during training or model selection.")

    model, backbone = build_transfer_model(args.model, num_classes=len(class_names))
    compile_model(model, 1e-3)
    model.summary()
    model_path = MODEL_DIR / f"{args.model}_transfer.keras"
    class_path = MODEL_DIR / f"{args.model}_class_names.json"
    checkpoint_path = MODEL_DIR / f"{args.model}_transfer_best.keras"
    callbacks = [
        keras.callbacks.ModelCheckpoint(str(checkpoint_path), monitor="val_loss", save_best_only=True, verbose=1),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-7, verbose=1),
    ]

    started = time.time()
    records = []
    history = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks, verbose=1)
    for epoch, vals in enumerate(zip(history.history["loss"], history.history["accuracy"],
                                    history.history["val_loss"], history.history["val_accuracy"]), 1):
        loss, acc, val_loss, val_acc = vals
        records.append({"stage": "frozen_backbone", "epoch": epoch, "loss": float(loss),
                        "accuracy": float(acc), "val_loss": float(val_loss), "val_accuracy": float(val_acc)})

    if args.fine_tune:
        print(f"\nFine-tuning the final {args.unfreeze_layers} backbone layers at learning rate 1e-5.")
        backbone.trainable = True
        for layer in backbone.layers[:-args.unfreeze_layers]:
            layer.trainable = False
        for layer in backbone.layers:
            if isinstance(layer, layers.BatchNormalization):
                layer.trainable = False
        compile_model(model, 1e-5)
        fine_history = model.fit(train_ds, validation_data=val_ds, epochs=args.fine_tune_epochs,
                                 callbacks=callbacks, verbose=1)
        for epoch, vals in enumerate(zip(fine_history.history["loss"], fine_history.history["accuracy"],
                                        fine_history.history["val_loss"], fine_history.history["val_accuracy"]), 1):
            loss, acc, val_loss, val_acc = vals
            records.append({"stage": "fine_tuning", "epoch": epoch, "loss": float(loss),
                            "accuracy": float(acc), "val_loss": float(val_loss), "val_accuracy": float(val_acc)})

    # EarlyStopping restores the best weights from the most recent fit stage.
    model.save(model_path)
    class_path.write_text(json.dumps(class_names, indent=2), encoding="utf-8")
    (REPORT_DIR / f"{args.model}_training_history.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    best_acc = max(r["val_accuracy"] for r in records)
    best_loss = min(r["val_loss"] for r in records)
    elapsed = round(time.time() - started, 2)
    row = {"model": args.model, "fine_tuning": args.fine_tune,
           "head_epochs_requested": args.epochs,
           "fine_tune_epochs_requested": args.fine_tune_epochs if args.fine_tune else 0,
           "best_validation_accuracy": round(best_acc, 6), "best_validation_loss": round(best_loss, 6),
           "duration_seconds": elapsed, "class_names": "|".join(class_names)}
    results_path = REPORT_DIR / "transfer_learning_results.csv"
    write_header = not results_path.exists()
    with results_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    print(f"\nSaved model: {model_path}")
    print(f"Best validation accuracy recorded: {best_acc:.4f}; best validation loss: {best_loss:.4f}")
    print(f"Duration: {elapsed} seconds; comparison CSV: {results_path}")
    print(f"Next: python evaluate_transfer.py --model {args.model}")


if __name__ == "__main__":
    main()
