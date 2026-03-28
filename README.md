# Class modeling project

End-to-end starter for a **topic of your choice**: data → model → saved artifact → small **Streamlit** app you can share.

## What is already here

| Piece | Role |
|--------|------|
| `data/raw/` | Put your dataset (e.g. `dataset.csv`). If missing, the code uses **synthetic** data so everything runs. |
| `notebooks/` | Jupyter notebooks for EDA and experiments. |
| `src/` | Reusable Python: loading data, training pipeline, metrics. |
| `scripts/train.py` | Trains the model and writes `models/model.joblib`. |
| `streamlit_app.py` | Web UI for metrics + manual predictions (good for class demo). |

## Prerequisites

- **Python 3.10+** from [python.org](https://www.python.org/downloads/) (during setup, enable *Add python.exe to PATH*), or install via your course’s recommended method. Check in a terminal: `python --version`.

## Local setup (Windows)

1. Open a terminal in this folder: `class-modeling-project`.

2. Create a virtual environment and install dependencies:

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

3. Train the model:

   ```powershell
   python scripts/train.py
   ```

4. Run the app:

   ```powershell
   streamlit run streamlit_app.py
   ```

## Using your own topic and data

1. Add `data/raw/dataset.csv` (or change `RAW_CSV` / loading logic in `src/config.py` and `src/data.py`).
2. Adjust the target column name in `src/data.py` if it is not `target`.
3. Improve the model in `src/modeling.py` (other algorithms, hyperparameters, cross-validation).
4. Update titles and text in `streamlit_app.py` so it matches your story for the class.

## Share with the class (deploy)

1. Put the project on **GitHub** (new repository, push this folder).
2. On [Streamlit Community Cloud](https://streamlit.io/cloud), connect the repo, set the main file to `streamlit_app.py`, and deploy.
3. **Important:** The hosted app only has files from the repo. Either commit a small trained `models/model.joblib` or run training in the app / at build time (your professor may prefer documenting that the teacher runs `train.py` once locally before deploy).

For a first assignment, committing a small `model.joblib` after training on non-sensitive data is usually acceptable; confirm with your professor.

## Aligning with your professor’s project

Match the **same story arc** they used (problem → data → method → metrics → discussion), even if your **domain** is different. Reuse their checklist: train/validation split, sensible baseline, at least one improvement, and clear limitations.
