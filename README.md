# Chest X-ray CNN project — no separate validation folder required

This project trains a binary image classifier using a `train/` and `test/` folder. **You do not need to create a separate `val/` folder.** `data_utils.py` automatically holds out 20% of images in `train/` as validation data.

## 1. Required dataset layout

```text
data/
└── chest_xray/
    ├── train/
    │   ├── NORMAL/
    │   │   └── image files...
    │   └── PNEUMONIA/
    │       └── image files...
    └── test/
        ├── NORMAL/
        │   └── image files...
        └── PNEUMONIA/
            └── image files...
```

Folder names define the class labels. Both `train` and `test` must contain the same two class folders. If your dataset is in another location, set `CHEST_XRAY_DATA_DIR` or edit paths in `config.py`.

## 2. Run in VS Code (Windows PowerShell)

Open this project folder in VS Code, then Terminal → New Terminal. Run one command at a time:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If the virtual environment already exists, just activate it and install requirements.

## 3. First run: train and evaluate

```powershell
python train.py
python evaluate.py
```

`train.py` uses 80% of the `train/` images for fitting and 20% for validation. The split is reproducible using seed 42. The separate `test/` folder is not used for training or model selection; `evaluate.py` uses it for final evaluation.

Model output: `models/baseline_cnn.keras`
Metrics: `reports/test_metrics.json`
Confusion matrix: `reports/confusion_matrix.png`

## 4. Run model comparison experiments

Start with one group at a time. Each experiment gets the same training/validation split.

```powershell
python experiments.py --group optimizers --epochs 8
python experiments.py --group activations --epochs 8
python experiments.py --group initialization --epochs 8
python experiments.py --group dropout --epochs 8
python experiments.py --group architecture --epochs 8
python experiments.py --group learning_rate --epochs 8
```

To run every group in one command (this may take a long time):

```powershell
python experiments.py --group all --epochs 8
```

Results are appended to `reports/experiment_results.csv`. Compare best validation accuracy/loss and duration. Choose settings using validation results, then train a final model and evaluate once on the test folder. Do not select models based on repeated test-set checks.

## 5. Run the web app or predict one image

```powershell
streamlit run app.py
```

For a terminal prediction:

```powershell
python predict.py "C:\path\to\one_xray.jpeg"
```

## Troubleshooting

- `No matching distribution found for tensorflow`: check Python version and architecture. TensorFlow wheels depend on Python version; Python 3.10–3.12 is generally the safer range for TensorFlow 2.16–2.19 on Windows. Use `python --version` to check.
- `Training folder not found`: make sure the `data/chest_xray/train/NORMAL` and `.../PNEUMONIA` folders exist, or update `config.py`.
- Out of memory: use `--batch-size 16` or `8`, e.g. `python train.py --batch-size 16`.
- The model is for educational/research use only, not clinical diagnosis.
