"""Evaluate the saved baseline model on the separate test folder only."""
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_score, recall_score, f1_score, roc_auc_score, ConfusionMatrixDisplay
)
from config import MODEL_PATH, CLASS_NAMES_PATH, METRICS_PATH, CONFUSION_MATRIX_PATH, BATCH_SIZE
from data_utils import load_test

def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Baseline model missing. Run `python train.py` first.")
    if not CLASS_NAMES_PATH.exists():
        raise FileNotFoundError("Class mapping missing. Run `python train.py` first.")
    class_names = json.loads(CLASS_NAMES_PATH.read_text(encoding="utf-8"))
    test_ds, class_names = load_test(BATCH_SIZE, class_names)
    model = tf.keras.models.load_model(MODEL_PATH)
    y_true = np.concatenate([y.numpy() for _, y in test_ds])
    probs = model.predict(test_ds, verbose=1)
    y_pred = probs.argmax(axis=1)
    auc = None
    try:
        auc = float(roc_auc_score(y_true, probs[:, 1]))
    except ValueError:
        pass
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=class_names,
                                   output_dict=True, zero_division=0)
    metrics = {
        "class_names": class_names,
        "positive_class_for_binary_metrics": class_names[1],
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_class_1": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_class_1": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_class_1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc_class_1": auc,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "note": "Research/learning output only; not a validated medical diagnostic device."
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("\nTEST RESULTS")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision ({class_names[1]}): {metrics['precision_class_1']:.4f}")
    print(f"Recall ({class_names[1]}): {metrics['recall_class_1']:.4f}")
    print(f"F1 ({class_names[1]}): {metrics['f1_class_1']:.4f}")
    print(f"ROC-AUC ({class_names[1]}): {auc}")
    print("\n", classification_report(y_true, y_pred, target_names=class_names, zero_division=0))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(values_format="d")
    plt.title("Baseline CNN — Test Confusion Matrix")
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH, dpi=160)
    plt.close()
    print(f"Saved metrics: {METRICS_PATH}")
    print(f"Saved confusion matrix: {CONFUSION_MATRIX_PATH}")

if __name__ == "__main__":
    main()
