"""Evaluate a trained ResNet50 or DenseNet121 transfer model on test/."""
import argparse
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
    precision_score, recall_score, f1_score, roc_auc_score, ConfusionMatrixDisplay)
# Import registers the custom preprocessing layers used by the saved models.
from transfer_learning import ResNetPreprocess, DenseNetPreprocess  # noqa: F401
from config import MODEL_DIR, REPORT_DIR, BATCH_SIZE
from data_utils import load_test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["resnet50", "densenet121"], required=True)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()
    model_path = MODEL_DIR / f"{args.model}_transfer.keras"
    class_path = MODEL_DIR / f"{args.model}_class_names.json"
    if not model_path.exists() or not class_path.exists():
        raise FileNotFoundError(f"Train the model first: python transfer_learning.py --model {args.model}")
    class_names = json.loads(class_path.read_text(encoding="utf-8"))
    test_ds, _ = load_test(args.batch_size, class_names)
    model = tf.keras.models.load_model(model_path)
    y_true = np.concatenate([y.numpy() for _, y in test_ds])
    probs = model.predict(test_ds, verbose=1)
    y_pred = np.argmax(probs, axis=1)
    cm = confusion_matrix(y_true, y_pred)
    try:
        auc = float(roc_auc_score(y_true, probs[:, 1]))
    except ValueError:
        auc = None
    report = classification_report(y_true, y_pred, target_names=class_names,
                                   output_dict=True, zero_division=0)
    metrics = {"model": args.model, "class_names": class_names,
        "positive_class_for_binary_metrics": class_names[1],
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_class_1": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_class_1": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_class_1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc_class_1": auc, "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "note": "Educational/research use only; not a validated medical diagnostic device."}
    metrics_path = REPORT_DIR / f"{args.model}_test_metrics.json"
    image_path = REPORT_DIR / f"{args.model}_test_confusion_matrix.png"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"\n{args.model.upper()} TEST RESULTS")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision ({class_names[1]}): {metrics['precision_class_1']:.4f}")
    print(f"Recall ({class_names[1]}): {metrics['recall_class_1']:.4f}")
    print(f"F1 ({class_names[1]}): {metrics['f1_class_1']:.4f}")
    print(f"ROC-AUC ({class_names[1]}): {auc}")
    print("\n", classification_report(y_true, y_pred, target_names=class_names, zero_division=0))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names).plot(values_format="d")
    plt.title(f"{args.model.upper()} — Test Confusion Matrix")
    plt.tight_layout()
    plt.savefig(image_path, dpi=160)
    plt.close()
    print(f"Saved metrics: {metrics_path}\nSaved confusion matrix: {image_path}")


if __name__ == "__main__":
    main()
