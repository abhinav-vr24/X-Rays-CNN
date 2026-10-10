"""Controlled experiments for the Chest X-ray CNN project.

Usage examples:
    python experiments.py --group all --epochs 8
    python experiments.py --group optimizers --epochs 10
    python experiments.py --group activations --epochs 8
    python experiments.py --group dropout --epochs 8
    python experiments.py --group architecture --epochs 8
    python experiments.py --group transfer --epochs 5

Each run uses the same train/validation split seed. The test set is not used
during experiments. Metrics are appended to reports/experiment_results.csv.
"""
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from config import (
    TRAIN_DIR, TEST_DIR, IMAGE_SIZE, BATCH_SIZE, SEED,
    VALIDATION_SPLIT, MODEL_DIR, REPORT_DIR
)

RESULTS_PATH = REPORT_DIR / "experiment_results.csv"
EXPERIMENT_MODEL_DIR = MODEL_DIR / "experiments"
EXPERIMENT_MODEL_DIR.mkdir(parents=True, exist_ok=True)

CSV_FIELDS = [
    "run_id", "group", "experiment", "optimizer", "learning_rate",
    "activation", "initializer", "dropout", "filters", "kernel_size",
    "batch_size", "epochs_requested", "epochs_completed", "train_accuracy",
    "val_accuracy", "val_loss", "val_precision_class1", "val_recall_class1",
    "val_f1_class1", "duration_seconds", "model_path", "status", "error"
]


def load_data(batch_size):
    if not TRAIN_DIR.exists() or not TEST_DIR.exists():
        raise FileNotFoundError(
            f"Dataset not found. Expected train directory: {TRAIN_DIR} and test directory: {TEST_DIR}. "
            "See README.md for dataset setup."
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
    train_classes = train_ds.class_names
    val_classes = val_ds.class_names
    test_classes = sorted([p.name for p in TEST_DIR.iterdir() if p.is_dir()])
    if len(train_classes) != 2:
        raise ValueError(f"Expected two classes, found: {train_classes}")
    if train_classes != val_classes or train_classes != test_classes:
        raise ValueError(
            f"Class-folder mismatch. Train={train_classes}, validation={val_classes}, test={test_classes}"
        )
    return train_ds.prefetch(tf.data.AUTOTUNE), val_ds.prefetch(tf.data.AUTOTUNE), train_classes


def activation_layer(name):
    name = name.lower()
    if name == "leaky_relu":
        return layers.LeakyReLU(negative_slope=0.1)
    if name in {"relu", "tanh", "sigmoid"}:
        return layers.Activation(name)
    raise ValueError(f"Unsupported activation: {name}")


def make_optimizer(name, lr):
    name = name.lower()
    if name == "sgd":
        return keras.optimizers.SGD(learning_rate=lr)
    if name == "momentum":
        return keras.optimizers.SGD(learning_rate=lr, momentum=0.9)
    if name == "adam":
        return keras.optimizers.Adam(learning_rate=lr)
    raise ValueError(f"Unsupported optimizer: {name}")


def build_custom_cnn(
    num_classes, activation="relu", initializer="he_normal",
    dropout=0.4, filters=(32, 64, 128), kernel_size=3,
    optimizer="adam", learning_rate=1e-3
):
    inputs = keras.Input(shape=(*IMAGE_SIZE, 3), name="chest_xray")
    x = layers.Rescaling(1.0 / 255.0)(inputs)
    x = layers.RandomRotation(0.04)(x)
    x = layers.RandomZoom(0.08)(x)
    for n_filters in filters:
        x = layers.Conv2D(
            n_filters, kernel_size, padding="same",
            kernel_initializer=initializer, use_bias=False
        )(x)
        x = layers.BatchNormalization()(x)
        x = activation_layer(activation)(x)
        x = layers.MaxPooling2D()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, kernel_initializer=initializer)(x)
    x = activation_layer(activation)(x)
    if dropout > 0:
        x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=make_optimizer(optimizer, learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_transfer_model(num_classes, backbone_name, learning_rate=1e-4, dropout=0.3):
    if backbone_name == "resnet50":
        base = keras.applications.ResNet50(
            include_top=False, weights="imagenet",
            input_shape=(*IMAGE_SIZE, 3)
        )
        preprocess = keras.applications.resnet50.preprocess_input
    elif backbone_name == "densenet121":
        base = keras.applications.DenseNet121(
            include_top=False, weights="imagenet",
            input_shape=(*IMAGE_SIZE, 3)
        )
        preprocess = keras.applications.densenet.preprocess_input
    else:
        raise ValueError(backbone_name)

    base.trainable = False
    inputs = keras.Input(shape=(*IMAGE_SIZE, 3))
    x = layers.Lambda(lambda z: preprocess(z), name="backbone_preprocessing")(inputs)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = keras.Model(inputs, outputs, name=backbone_name)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def evaluate_predictions(model, dataset, class_names):
    y_true = np.concatenate([labels.numpy() for _, labels in dataset])
    probs = model.predict(dataset, verbose=0)
    y_pred = probs.argmax(axis=1)
    # Binary metrics explicitly refer to class index 1. Class mapping is logged.
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true != 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred != 1))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "val_precision_class1": float(precision),
        "val_recall_class1": float(recall),
        "val_f1_class1": float(f1),
        "class_index_1_name": class_names[1],
    }


def experiment_specs(group):
    specs = []
    def add(group_name, experiment, **kwargs):
        specs.append({"group": group_name, "experiment": experiment, **kwargs})

    if group in ("all", "optimizers"):
        add("optimizers", "sgd_lr_0.001", optimizer="sgd", learning_rate=0.001)
        add("optimizers", "momentum_lr_0.001", optimizer="momentum", learning_rate=0.001)
        add("optimizers", "adam_lr_0.001", optimizer="adam", learning_rate=0.001)
    if group in ("all", "activations"):
        for activation in ("relu", "leaky_relu", "tanh", "sigmoid"):
            # Lower learning rate for sigmoid/tanh can be useful, but keep equal here
            # for a simple controlled comparison; results are not guaranteed optimal.
            add("activations", f"activation_{activation}", activation=activation)
    if group in ("all", "initialization"):
        add("initialization", "glorot_uniform", initializer="glorot_uniform")
        add("initialization", "he_normal", initializer="he_normal")
        add("initialization", "random_normal", initializer="random_normal")
    if group in ("all", "dropout"):
        for rate in (0.0, 0.2, 0.4, 0.5):
            add("dropout", f"dropout_{rate}", dropout=rate)
    if group in ("all", "architecture"):
        add("architecture", "filters_16_32_64", filters=(16, 32, 64))
        add("architecture", "filters_32_64_128", filters=(32, 64, 128))
        add("architecture", "filters_64_128_256", filters=(64, 128, 256))
        add("architecture", "kernel_3", kernel_size=3)
        add("architecture", "kernel_5", kernel_size=5)
    if group in ("all", "learning_rate"):
        for lr in (1e-2, 1e-3, 1e-4):
            add("learning_rate", f"learning_rate_{lr}", learning_rate=lr)
    if group in ("all", "transfer"):
        add("transfer", "resnet50_frozen", backbone="resnet50", learning_rate=1e-4)
        add("transfer", "densenet121_frozen", backbone="densenet121", learning_rate=1e-4)
    return specs


def append_result(row):
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    exists = RESULTS_PATH.exists()
    with open(RESULTS_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in CSV_FIELDS})


def run_one(spec, train_ds, val_ds, class_names, epochs, batch_size, save_models):
    run_id = f"{time.strftime('%Y%m%d_%H%M%S')}_{spec['group']}_{spec['experiment']}"
    start = time.time()
    row = {
        "run_id": run_id,
        "group": spec["group"],
        "experiment": spec["experiment"],
        "optimizer": spec.get("optimizer", "adam"),
        "learning_rate": spec.get("learning_rate", 1e-3),
        "activation": spec.get("activation", "relu"),
        "initializer": spec.get("initializer", "he_normal"),
        "dropout": spec.get("dropout", 0.4),
        "filters": "-".join(map(str, spec.get("filters", (32, 64, 128)))),
        "kernel_size": spec.get("kernel_size", 3),
        "batch_size": batch_size,
        "epochs_requested": epochs,
        "status": "running",
    }
    try:
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(SEED)
        if "backbone" in spec:
            model = build_transfer_model(
                len(class_names), spec["backbone"], spec.get("learning_rate", 1e-4)
            )
        else:
            model = build_custom_cnn(
                num_classes=len(class_names),
                activation=spec.get("activation", "relu"),
                initializer=spec.get("initializer", "he_normal"),
                dropout=spec.get("dropout", 0.4),
                filters=spec.get("filters", (32, 64, 128)),
                kernel_size=spec.get("kernel_size", 3),
                optimizer=spec.get("optimizer", "adam"),
                learning_rate=spec.get("learning_rate", 1e-3),
            )
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=3, restore_best_weights=True, verbose=0
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=0
            ),
        ]
        history = model.fit(
            train_ds, validation_data=val_ds, epochs=epochs,
            callbacks=callbacks, verbose=2
        )
        best_epoch = int(np.argmin(history.history["val_loss"]))
        val_loss = float(min(history.history["val_loss"]))
        train_acc = float(history.history["accuracy"][best_epoch])
        val_acc = float(history.history["val_accuracy"][best_epoch])
        extra = evaluate_predictions(model, val_ds, class_names)
        model_path = ""
        if save_models:
            model_path = str(EXPERIMENT_MODEL_DIR / f"{run_id}.keras")
            model.save(model_path)
        row.update({
            "epochs_completed": len(history.history["loss"]),
            "train_accuracy": train_acc,
            "val_accuracy": val_acc,
            "val_loss": val_loss,
            **extra,
            "duration_seconds": round(time.time() - start, 2),
            "model_path": model_path,
            "status": "success",
        })
        print(
            f"RESULT {spec['experiment']}: val_accuracy={val_acc:.4f}, "
            f"val_recall(class 1: {class_names[1]})={extra['val_recall_class1']:.4f}, "
            f"val_f1={extra['val_f1_class1']:.4f}"
        )
    except Exception as exc:
        row.update({
            "duration_seconds": round(time.time() - start, 2),
            "status": "failed",
            "error": repr(exc),
        })
        print(f"FAILED {spec['experiment']}: {exc}")
    append_result(row)


def main():
    parser = argparse.ArgumentParser(description="Run controlled CNN experiments.")
    parser.add_argument(
        "--group",
        choices=["all", "optimizers", "activations", "initialization", "dropout",
                 "architecture", "learning_rate", "transfer"],
        default="all",
    )
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--save-models", action="store_true",
                        help="Save every experiment model; uses additional disk space.")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("epochs and batch size must be positive.")

    train_ds, val_ds, class_names = load_data(args.batch_size)
    print("Class mapping:")
    for index, name in enumerate(class_names):
        print(f"  {index}: {name}")
    print(f"Positive class for class-1 precision/recall/F1: {class_names[1]}")
    specs = experiment_specs(args.group)
    print(f"\nRunning {len(specs)} experiments in group '{args.group}'.")
    print(f"Results will be appended to: {RESULTS_PATH}")
    for index, spec in enumerate(specs, start=1):
        print(f"\n[{index}/{len(specs)}] {spec['group']} / {spec['experiment']}")
        run_one(spec, train_ds, val_ds, class_names, args.epochs, args.batch_size, args.save_models)
    print("\nExperiments finished. Open reports/experiment_results.csv to compare runs.")
    print("Validation metrics are for model selection only; run final test evaluation once after selecting a model.")


if __name__ == "__main__":
    main()
