"""
train_baseline.py

Re-creates the Module 2 baseline: Logistic Regression + Random Forest on
the processed UCI readmission data. We keep Random Forest as the model
going forward (Module 4 explains it with SHAP).
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "processed"
MODEL_PATH = ROOT / "models" / "rf_model.pkl"

DROP_FOR_MODEL = ["readmitted_30d", "patient_nbr", "anomaly_score", "is_anomaly"]


def load():
    train = pd.read_csv(f"{DATA_DIR}/train.csv")
    val = pd.read_csv(f"{DATA_DIR}/val.csv")
    return train, val


def build_xy(train, val):
    feature_cols = [c for c in train.columns if c not in DROP_FOR_MODEL]
    cat_cols = [c for c in feature_cols if not pd.api.types.is_numeric_dtype(train[c])]
    num_cols = [c for c in feature_cols if c not in cat_cols]

    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    encoder.fit(train[cat_cols].astype(str))

    def transform(df):
        num = df[num_cols].fillna(0).to_numpy()
        cat = encoder.transform(df[cat_cols].astype(str))
        return np.hstack([num, cat])

    feature_names = num_cols + list(encoder.get_feature_names_out(cat_cols))
    X_train, X_val = transform(train), transform(val)
    y_train, y_val = train["readmitted_30d"], val["readmitted_30d"]
    return X_train, X_val, y_train, y_val, feature_names, encoder, num_cols, cat_cols


def run():
    train, val = load()
    X_train, X_val, y_train, y_val, feature_names, encoder, num_cols, cat_cols = build_xy(train, val)

    logreg = LogisticRegression(max_iter=1000, class_weight="balanced")
    logreg.fit(X_train, y_train)
    logreg_auc = roc_auc_score(y_val, logreg.predict_proba(X_val)[:, 1])

    rf = RandomForestClassifier(
        n_estimators=300, max_depth=8, min_samples_leaf=20,
        class_weight="balanced", random_state=42, n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    rf_proba = rf.predict_proba(X_val)[:, 1]
    rf_auc = roc_auc_score(y_val, rf_proba)

    print(f"Logistic Regression AUROC: {logreg_auc:.3f}")
    print(f"Random Forest AUROC:       {rf_auc:.3f}")
    print("\nRandom Forest classification report (val, threshold=0.5):")
    print(classification_report(y_val, (rf_proba >= 0.5).astype(int),
                                 target_names=["No Readmit", "Readmit"]))

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({
            "model": rf, "encoder": encoder,
            "num_cols": num_cols, "cat_cols": cat_cols,
            "feature_names": feature_names,
        }, f)
    print(f"Saved -> {MODEL_PATH}")


if __name__ == "__main__":
    run()
