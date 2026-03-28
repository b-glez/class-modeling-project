"""Paths and constants — adjust for your topic and files."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
MODEL_PATH = MODELS_DIR / "model.joblib"

# If you use a real CSV, put it in data/raw/ and set the filename here.
RAW_CSV = DATA_RAW / "dataset.csv"

RANDOM_STATE = 42
