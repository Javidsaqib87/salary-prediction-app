"""
Train the salary prediction model and save it to model/model.pkl.

This is the script version of notebooks/model_training.ipynb. It produces the
same model and is handy when the model needs to be rebuilt without opening
the notebook (for example after changing the scikit-learn version).

Usage:  python train_model.py
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split

RANDOM_STATE = 42
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "salary_dataset.csv"
MODEL_DIR = BASE_DIR / "model"


def evaluate(model, X, y):
    pred = model.predict(X)
    return {
        "r2": float(r2_score(y, pred)),
        "mae": float(mean_absolute_error(y, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y, pred))),
    }


def main():
    # Load and tidy the data
    df = pd.read_csv(DATA_PATH).rename(columns={"Experience Years": "YearsExperience"})
    df = df.dropna().drop_duplicates()

    X = df[["YearsExperience"]]
    y = df["Salary"]

    # Hold back 20% for testing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    # The CSV is sorted by experience, so cross-validation folds are shuffled
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_r2 = cross_val_score(LinearRegression(), X, y, cv=cv, scoring="r2")

    metrics = {
        "model": "LinearRegression",
        "feature": "YearsExperience",
        "target": "Salary",
        "slope": float(model.coef_[0]),
        "intercept": float(model.intercept_),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "train": evaluate(model, X_train, y_train),
        "test": evaluate(model, X_test, y_test),
        "cv_r2_mean": float(cv_r2.mean()),
        "experience_min": float(X["YearsExperience"].min()),
        "experience_max": float(X["YearsExperience"].max()),
        "sklearn_version": sklearn.__version__,
    }

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_DIR / "model.pkl")
    with open(MODEL_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("Model saved to", MODEL_DIR / "model.pkl")
    print(f"Test R2:   {metrics['test']['r2']:.3f}")
    print(f"Test MAE:  {metrics['test']['mae']:,.0f}")
    print(f"Test RMSE: {metrics['test']['rmse']:,.0f}")


if __name__ == "__main__":
    main()
