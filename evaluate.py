"""Evaluate the saved model on the held-out test set."""
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_score, recall_score, f1_score, roc_auc_score, ConfusionMatrixDisplay
)
from config import (
    TEST_DIR, MODEL_PATH, CLASS_NAMES_PATH, METRICS_PATH,
    CONFUSION_MATRIX_PATH, IMAGE_SIZE, BATCH_SIZE
)


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run `python train.py` first.")
    if not TEST_DIR.exists():
        raise FileNotFoundError(f"Test directory not found: {TEST_DIR}")

    test_ds = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
        shuffle=False,
    )
    class_names = test_ds.class_names
    if CLASS_NAMES_PATH.exists():
        with open(CLASS_NAMES_PATH, encoding="utf-8") as f:
            saved_names = json.load(f)
        if saved_names != class_names:
            raise ValueError(
                f"Test classes {class_names} do not match saved model classes {saved_names}."
            )

    model = tf.keras.models.load_model(MODEL_PATH)
    y_true = np.concatenate([labels.numpy() for _, labels in test_ds])
    probabilities = model.predict(test_ds, verbose=1)
    y_pred = np.argmax(probabilities, axis=1)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="binary", zero_division=0)
    recall = recall_score(y_true, y_pred, average="binary", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="binary", zero_division=0)
    try:
        auc = roc_auc_score(y_true, probabilities[:, 1])
    except ValueError:
        auc = None

    metrics = {
        "class_names": class_names,
        "positive_class_for_binary_metrics": class_names[1],
        "accuracy": float(accuracy),
        "precision_class_1": float(precision),
        "recall_class_1": float(recall),
        "f1_class_1": float(f1),
        "roc_auc_class_1": None if auc is None else float(auc),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(
            y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0
        ),
        "note": "Metrics describe this dataset split only; this model is not a validated medical diagnostic device."
    }
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("\nTest metrics")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision (class index 1, {class_names[1]}): {precision:.4f}")
    print(f"Recall    (class index 1, {class_names[1]}): {recall:.4f}")
    print(f"F1        (class index 1, {class_names[1]}): {f1:.4f}")
    print(f"ROC-AUC   (class index 1, {class_names[1]}): {auc if auc is not None else 'undefined'}")
    print("\nClassification report:")
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))

    disp = ConfusionMatrixDisplay(
        confusion_matrix=confusion_matrix(y_true, y_pred),
        display_labels=class_names,
    )
    disp.plot(values_format="d")
    plt.title("Chest X-ray CNN — Test Confusion Matrix")
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH, dpi=160)
    plt.close()
    print(f"\nSaved metrics: {METRICS_PATH}")
    print(f"Saved confusion matrix: {CONFUSION_MATRIX_PATH}")


if __name__ == "__main__":
    main()
