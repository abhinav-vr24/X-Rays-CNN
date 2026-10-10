# Chest X-ray CNN — rebuilt Python project

This project trains a convolutional neural network to classify chest X-ray images into two folder-defined classes (commonly `NORMAL` and `PNEUMONIA`), evaluates the saved model, predicts one image, and provides a Streamlit interface.

## 1. Requirements

- Windows 10/11
- Python 3.10–3.12 recommended
- VS Code
- Dataset organized into class folders

## 2. Expected folder structure

Copy your dataset so the folders look like this:

```text
chest_xray_cnn_rebuilt/
├── app.py
├── train.py
├── evaluate.py
├── predict.py
├── config.py
├── requirements.txt
├── data/
│   └── chest_xray/
│       ├── train/
│       │   ├── NORMAL/
│       │   └── PNEUMONIA/
│       └── test/
│           ├── NORMAL/
│           └── PNEUMONIA/
├── models/       # created/filled during training
└── reports/      # created/filled during training/evaluation
```

Use the actual folder names from your dataset. Both `train` and `test` must contain the same class folders. The class mapping is discovered from folder names and saved in `models/class_names.json`; do not assume the positive class index without checking it.

### If your dataset is somewhere else

PowerShell example:

```powershell
$env:CHEST_XRAY_DATA_DIR = "C:\Users\YourName\Downloads\ChestXRay2017\chest_xray"
```

This base folder should contain `train` and `test`. Set this environment variable in the same terminal where you run the scripts.

## 3. Create and activate a virtual environment

Open the project folder in VS Code, then open **Terminal → New Terminal**.

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, you can use Command Prompt instead:

```bat
.venv\Scripts\activate.bat
```

To exit the virtual environment later, run:

```powershell
deactivate
```

## 4. Train the model

```powershell
python train.py
```

Optional settings:

```powershell
python train.py --epochs 20 --batch-size 32 --learning-rate 0.001
```

The script reserves 20% of the training directory for validation. It saves the best model and class mapping in `models/`, and the training history in `reports/`.

## 5. Evaluate on the held-out test set

Run this only after training:

```powershell
python evaluate.py
```

Outputs include accuracy, precision, recall, F1, ROC-AUC when defined, a classification report, and a confusion-matrix plot. Binary precision/recall/F1 and ROC-AUC are calculated for class index 1, whose name is printed and saved. Confirm that this is the class you intend to treat as positive.

Do not tune repeatedly against the test set; use validation data for model selection.

## 6. Predict one image

```powershell
python predict.py "C:\path\to\chest_xray.jpg"
```

## 7. Run the Streamlit app

```powershell
streamlit run app.py
```

Streamlit will print a local URL, usually `http://localhost:8501`. Open it in your browser.

## Troubleshooting

- **`requirements.txt` not found:** in the terminal, run `dir` and confirm the current folder is `chest_xray_cnn_rebuilt`. Use `cd "path\to\chest_xray_cnn_rebuilt"` if needed.
- **Training folder not found:** confirm `data/chest_xray/train` exists or set `CHEST_XRAY_DATA_DIR`.
- **Model not found in Streamlit:** train first with `python train.py`.
- **TensorFlow installation issues:** use a supported Python version (Python 3.10–3.12 recommended) and check `python --version` inside the activated environment.
- **Out of memory:** try `python train.py --batch-size 8`.

## Important limitations

This is a learning project, not a validated medical diagnostic system. Dataset leakage, patient overlap, class imbalance, image artifacts, and distribution shift can make reported performance misleading. If patient identifiers are available, split at patient level. Never use this model alone for clinical decisions.

## 8. Run separate controlled experiments

`experiments.py` compares model variants and appends results to `reports/experiment_results.csv`.
The test set is not used during these experiments.

Run one group at a time:

```powershell
python experiments.py --group optimizers --epochs 8
python experiments.py --group activations --epochs 8
python experiments.py --group initialization --epochs 8
python experiments.py --group dropout --epochs 8
python experiments.py --group architecture --epochs 8
python experiments.py --group learning_rate --epochs 8
python experiments.py --group transfer --epochs 5
```

Or run every group (this can take a long time and transfer learning may need substantial RAM/GPU):

```powershell
python experiments.py --group all --epochs 8
```

Use `--save-models` to save every run's model; otherwise only the CSV results are kept:

```powershell
python experiments.py --group dropout --epochs 8 --save-models
```

### How to interpret the results

- `val_accuracy`, `val_loss`, `val_precision_class1`, `val_recall_class1`, and `val_f1_class1` are computed on the validation set.
- Check the printed class mapping. Binary metrics ending in `class1` refer to class index 1, which is usually `PNEUMONIA` when folder names sort alphabetically as `NORMAL`, `PNEUMONIA`; always verify.
- `all` means one-factor-at-a-time screening, not an exhaustive grid search. Each experiment changes the named factor from the common baseline; use repeated runs/seeds for stronger conclusions.
- For a fair comparison, use the same dataset split, epoch budget, and batch size. Optimizers can require different learning-rate tuning, so a second round should tune the best candidates.
- Do not choose a final model by repeatedly inspecting test performance. Select with validation data, then run `python evaluate.py` once for the final held-out test evaluation.

