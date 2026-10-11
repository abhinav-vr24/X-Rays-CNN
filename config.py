"""Project settings. No separate validation folder is required."""
from pathlib import Path
import os

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.getenv("CHEST_XRAY_DATA_DIR", str(PROJECT_DIR / "data" / "chest_xray")))
TRAIN_DIR = Path(os.getenv("CHEST_XRAY_TRAIN_DIR", str(DATA_DIR / "train")))
TEST_DIR = Path(os.getenv("CHEST_XRAY_TEST_DIR", str(DATA_DIR / "test")))
MODEL_DIR = PROJECT_DIR / "models"
REPORT_DIR = PROJECT_DIR / "reports"
MODEL_PATH = MODEL_DIR / "baseline_cnn.keras"
CLASS_NAMES_PATH = MODEL_DIR / "class_names.json"
HISTORY_PATH = REPORT_DIR / "training_history.json"
METRICS_PATH = REPORT_DIR / "test_metrics.json"
CONFUSION_MATRIX_PATH = REPORT_DIR / "confusion_matrix.png"
EXPERIMENT_RESULTS_PATH = REPORT_DIR / "experiment_results.csv"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42
VALIDATION_SPLIT = 0.20  # 20% of the training folder is held back automatically
DEFAULT_EPOCHS = 15

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
