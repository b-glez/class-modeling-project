"""Train a simple baseline model — swap pipeline/steps for your problem."""
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_pipeline():
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )


def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    out = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "report": classification_report(y_test, y_pred, digits=3),
    }
    if proba is not None and y_test.nunique() == 2:
        try:
            out["roc_auc"] = float(roc_auc_score(y_test, proba))
        except ValueError:
            out["roc_auc"] = None
    return out
