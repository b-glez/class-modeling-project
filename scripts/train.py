"""
Train the model and save to models/model.joblib.
Run from project root:  python scripts/train.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib

from src.config import MODEL_PATH, MODELS_DIR
from src.data import load_raw_dataframe, train_val_split
from src.modeling import build_pipeline, evaluate


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    df = load_raw_dataframe()
    X_train, X_test, y_train, y_test = train_val_split(df)

    model = build_pipeline()
    model.fit(X_train, y_train)
    metrics = evaluate(model, X_test, y_test)

    joblib.dump({"model": model, "feature_columns": list(X_train.columns), "metrics": metrics}, MODEL_PATH)
    print("Saved:", MODEL_PATH)
    print("Accuracy:", metrics["accuracy"])
    if metrics.get("roc_auc") is not None:
        print("ROC-AUC:", metrics["roc_auc"])
    print(metrics["report"])


if __name__ == "__main__":
    main()
