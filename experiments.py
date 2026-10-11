"""Controlled model experiments. Validation is automatically split from train/."""
import argparse, csv, time
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from config import (
    IMAGE_SIZE, BATCH_SIZE, SEED, EXPERIMENT_RESULTS_PATH, MODEL_DIR
)
from data_utils import load_train_validation
from train import build_model

def compile_model(model, optimizer_name="adam", learning_rate=1e-3):
    name = optimizer_name.lower()
    if name == "sgd":
        opt = keras.optimizers.SGD(learning_rate=learning_rate)
    elif name == "sgd_momentum":
        opt = keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
    elif name == "rmsprop":
        opt = keras.optimizers.RMSprop(learning_rate=learning_rate)
    else:
        opt = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=opt, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

def create_experiment_model(setting):
    kind = setting["group"]
    if kind == "optimizers":
        model = build_model()
        return compile_model(model, setting["value"], 1e-3)
    if kind == "activations":
        return compile_model(build_model(activation=setting["value"]))
    if kind == "initialization":
        init = setting["value"]
        inputs = keras.Input(shape=(*IMAGE_SIZE, 3))
        x = layers.Rescaling(1./255)(inputs)
        for i, f in enumerate((32, 64, 128)):
            x = layers.Conv2D(f, 3, padding="same", kernel_initializer=init, use_bias=False)(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation("relu")(x)
            x = layers.MaxPooling2D()(x)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.4)(x)
        out = layers.Dense(2, activation="softmax")(x)
        return compile_model(keras.Model(inputs, out))
    if kind == "dropout":
        return compile_model(build_model(dropout=float(setting["value"])))
    if kind == "architecture":
        filters = setting["value"]
        return compile_model(build_model(filters=filters))
    if kind == "learning_rate":
        return compile_model(build_model(), "adam", float(setting["value"]))
    raise ValueError(f"Unknown experiment group: {kind}")

GROUPS = {
    "optimizers": [
        {"group":"optimizers","name":"Adam","value":"adam"},
        {"group":"optimizers","name":"SGD","value":"sgd"},
        {"group":"optimizers","name":"SGD_momentum","value":"sgd_momentum"},
        {"group":"optimizers","name":"RMSprop","value":"rmsprop"},
    ],
    "activations": [
        {"group":"activations","name":"relu","value":"relu"},
        {"group":"activations","name":"tanh","value":"tanh"},
        {"group":"activations","name":"elu","value":"elu"},
        {"group":"activations","name":"selu","value":"selu"},
    ],
    "initialization": [
        {"group":"initialization","name":"he_normal","value":"he_normal"},
        {"group":"initialization","name":"glorot_uniform","value":"glorot_uniform"},
        {"group":"initialization","name":"random_normal","value":"random_normal"},
    ],
    "dropout": [
        {"group":"dropout","name":"dropout_0.0","value":0.0},
        {"group":"dropout","name":"dropout_0.2","value":0.2},
        {"group":"dropout","name":"dropout_0.4","value":0.4},
        {"group":"dropout","name":"dropout_0.5","value":0.5},
    ],
    "architecture": [
        {"group":"architecture","name":"filters_16_32_64","value":(16,32,64)},
        {"group":"architecture","name":"filters_32_64_128","value":(32,64,128)},
        {"group":"architecture","name":"filters_32_64_128_192","value":(32,64,128,192)},
    ],
    "learning_rate": [
        {"group":"learning_rate","name":"lr_0.01","value":0.01},
        {"group":"learning_rate","name":"lr_0.001","value":0.001},
        {"group":"learning_rate","name":"lr_0.0001","value":0.0001},
    ],
}

def main():
    p = argparse.ArgumentParser(description="Run controlled CNN experiments.")
    p.add_argument("--group", choices=["all", *GROUPS.keys()], default="all")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--save-models", action="store_true")
    args = p.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        p.error("epochs and batch-size must be positive")
    tf.keras.utils.set_random_seed(SEED)
    train_ds, val_ds, class_names = load_train_validation(args.batch_size)
    settings = sum(GROUPS.values(), []) if args.group == "all" else GROUPS[args.group]
    EXPERIMENT_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    columns = ["group","experiment","epochs_requested","best_val_accuracy","best_val_loss","duration_seconds","class_names"]
    write_header = not EXPERIMENT_RESULTS_PATH.exists()
    with EXPERIMENT_RESULTS_PATH.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        if write_header:
            writer.writeheader()
        for i, setting in enumerate(settings, start=1):
            print(f"\n[{i}/{len(settings)}] {setting['group']} — {setting['name']}")
            keras.backend.clear_session()
            tf.keras.utils.set_random_seed(SEED)
            model = create_experiment_model(setting)
            start = time.time()
            history = model.fit(
                train_ds, validation_data=val_ds, epochs=args.epochs, verbose=2,
                callbacks=[keras.callbacks.EarlyStopping(
                    monitor="val_loss", patience=3, restore_best_weights=True
                )]
            )
            duration = round(time.time() - start, 2)
            best_acc = max(history.history.get("val_accuracy", [0.0]))
            best_loss = min(history.history.get("val_loss", [float("inf")]))
            writer.writerow({
                "group": setting["group"], "experiment": setting["name"],
                "epochs_requested": args.epochs, "best_val_accuracy": round(float(best_acc), 6),
                "best_val_loss": round(float(best_loss), 6), "duration_seconds": duration,
                "class_names": "|".join(class_names)
            })
            f.flush()
            if args.save_models:
                safe_name = setting["name"].replace(".", "_")
                out = MODEL_DIR / "experiments" / f"{setting['group']}_{safe_name}.keras"
                out.parent.mkdir(parents=True, exist_ok=True)
                model.save(out)
                print(f"Saved model: {out}")
            print(f"Best validation accuracy: {best_acc:.4f}; best validation loss: {best_loss:.4f}")
            del model
    print(f"\nResults appended to: {EXPERIMENT_RESULTS_PATH}")
    print("The test folder was not used for experiment selection.")

if __name__ == "__main__":
    main()
