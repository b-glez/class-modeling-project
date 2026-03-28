"""Load and split data — replace synthetic data with your real dataset."""
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from src.config import RAW_CSV, RANDOM_STATE


def load_raw_dataframe():
    """
    If data/raw/dataset.csv exists, load it. The last column is assumed to be the target.
    Otherwise, build a small synthetic classification dataset (placeholder for your topic).
    """
    if RAW_CSV.exists():
        df = pd.read_csv(RAW_CSV)
        return df

    X, y = make_classification(
        n_samples=800,
        n_features=6,
        n_informative=4,
        n_redundant=0,
        random_state=RANDOM_STATE,
    )
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feature_names)
    df["target"] = y
    return df


def train_val_split(df: pd.DataFrame, target_col: str = "target", test_size: float = 0.2):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return train_test_split(X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y)
