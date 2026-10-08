"""
train_baseline.py

Re-creates the Module 2 baseline: Logistic Regression + Random Forest on
the processed UCI readmission data. We keep Random Forest as the model
going forward (Module 4 explains it with SHAP).

Note: class weighting is intentionally NOT used. With class_weight="balanced"
predicted probabilities get pushed toward 0.5 and stop reflecting the real
~9% readmission rate. Without it, predicted risk is a meaningful probability.
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report, brier_score_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from risk_model import calibrate, fit_calibration

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "processed"
MODEL_PATH = ROOT / "models" / "rf_model.pkl"

DROP_FOR_MODEL = ["readmitted_30d",
                  "patient_nbr", "anomaly_score", "is_anomaly"]


def load():
    train = pd.read_csv(f"{DATA_DIR}/train.csv")
    val = pd.read_csv(f"{DATA_DIR}/val.csv")
    return train, val


def build_xy(train, val):
    feature_cols = [c for c in train.columns if c not in DROP_FOR_MODEL]
    cat_cols = [
        c for c in feature_cols if not pd.api.types.is_numeric_dtype(train[c])]
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
    X_train, X_val, y_train, y_val, feature_names, encoder, num_cols, cat_cols = build_xy(
        train, val)

    # Scaling fixes the ConvergenceWarning; no class weights so probabilities stay honest
    logreg = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    logreg.fit(X_train, y_train)
    logreg_auc = roc_auc_score(y_val, logreg.predict_proba(X_val)[:, 1])

    rf = RandomForestClassifier(
        n_estimators=300, max_depth=8, min_samples_leaf=20,
        random_state=42, n_jobs=-1,  # class_weight removed
    )
    rf.fit(X_train, y_train)
    rf_proba = rf.predict_proba(X_val)[:, 1]
    rf_auc = roc_auc_score(y_val, rf_proba)

    print(f"Logistic Regression AUROC: {logreg_auc:.3f}")
    print(f"Random Forest AUROC:       {rf_auc:.3f}")

    # Platt scaling: slope + intercept fitted on the validation set.
    cal = fit_calibration(rf_proba, y_val)
    cal_proba = calibrate(cal, rf_proba)
    auc_after = roc_auc_score(y_val, cal_proba)
    print("\nCalibration check (val, calibrator was fitted here, so judge on the test set):")
    print(f"  before: mean predicted {rf_proba.mean():.3f} | "
          f"Brier {brier_score_loss(y_val, rf_proba):.4f}")
    print(f"  after:  mean predicted {cal_proba.mean():.3f} | "
          f"Brier {brier_score_loss(y_val, cal_proba):.4f}")
    print(f"  AUROC before/after: {rf_auc:.3f} / {auc_after:.3f} (must match)")
    print(
        f"  calibration slope {cal['slope']:.3f}, intercept {cal['intercept']:.3f}")

    # Decision threshold lives on the CALIBRATED risk scale (top 20% flagged).
    threshold = float(np.quantile(cal_proba, 0.80))
    print(
        f"\nRandom Forest classification report (val, threshold={threshold:.3f}):")
    print(classification_report(y_val, (cal_proba >= threshold).astype(int),
                                target_names=["No Readmit", "Readmit"]))

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({
            "model": rf, "encoder": encoder,
            "num_cols": num_cols, "cat_cols": cat_cols,
            "feature_names": feature_names,
            "threshold": threshold,
            "calibration": cal,
        }, f)
    print(f"Saved -> {MODEL_PATH}")


if __name__ == "__main__":
    run()
