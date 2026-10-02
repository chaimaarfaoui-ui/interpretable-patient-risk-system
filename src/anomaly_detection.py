"""
anomaly_detection.py

Module 3: Anomaly Detection Layer.

Trains an Isolation Forest on the processed training data to flag patients
whose overall profile is statistically unusual -- regardless of what the
Module 2 risk model predicts. This is a "safety net": it doesn't know or
care about the readmission label, only whether a patient's combination of
features looks like it belongs to the population the model was trained on.

Adds two columns to each processed split:
  - anomaly_score : raw Isolation Forest score (lower = more anomalous)
  - is_anomaly    : 1 if flagged (bottom ~2% of scores), else 0
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "processed"

# Numeric features: these describe the "shape" of a patient's utilization
# and clinical complexity -- exactly where a bizarre outlier would show up
# (e.g. 60 medications, 40 days in hospital).
NUMERIC_FEATURES = [
    "time_in_hospital", "num_lab_procedures", "num_procedures",
    "num_medications", "number_outpatient", "number_emergency",
    "number_inpatient", "number_diagnoses", "total_prior_visits",
]

# A few categoricals worth including -- admission urgency and whether
# meds changed are part of what makes a case look "normal" or not.
CATEGORICAL_FEATURES = ["admission_type_id", "change", "diabetesMed"]

CONTAMINATION = 0.02  # expect ~2% of patients to be flagged, matching
                       # the synthetic data's injected anomaly rate


def load_splits():
    train = pd.read_csv(f"{DATA_DIR}/train.csv")
    val = pd.read_csv(f"{DATA_DIR}/val.csv")
    test = pd.read_csv(f"{DATA_DIR}/test.csv")
    return train, val, test


def build_features(train, val, test):
    """Fit encoders on train only, then transform all three splits --
    never let val/test leak into how we define 'normal'."""
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    encoder.fit(train[CATEGORICAL_FEATURES].astype(str))

    def transform(df):
        num = df[NUMERIC_FEATURES].fillna(0).to_numpy()
        cat = encoder.transform(df[CATEGORICAL_FEATURES].astype(str))
        return np.hstack([num, cat])

    return transform(train), transform(val), transform(test)


def run():
    train, val, test = load_splits()
    X_train, X_val, X_test = build_features(train, val, test)

    print(f"Training Isolation Forest on {len(train):,} patients, "
          f"{X_train.shape[1]} features...")

    model = IsolationForest(
        n_estimators=200,
        contamination=CONTAMINATION,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train)

    results = {}
    for name, df, X in [("train", train, X_train), ("val", val, X_val), ("test", test, X_test)]:
        df = df.copy()
        df["anomaly_score"] = model.score_samples(X)  # higher = more normal
        df["is_anomaly"] = (model.predict(X) == -1).astype(int)
        out_path = f"{DATA_DIR}/{name}.csv"
        df.to_csv(out_path, index=False)
        results[name] = df
        print(f"  {name}: {df['is_anomaly'].sum():,} flagged "
              f"({df['is_anomaly'].mean():.1%}) -> {out_path}")

    return results


def summarize(results):
    print("\n--- Do flagged patients actually look different? ---")
    for name, df in results.items():
        normal = df[df["is_anomaly"] == 0]
        anomalous = df[df["is_anomaly"] == 1]
        print(f"\n[{name}]")
        print(f"  Readmit rate -- normal: {normal['readmitted_30d'].mean():.2%}  "
              f"|  flagged: {anomalous['readmitted_30d'].mean():.2%}")
        for col in ["num_medications", "time_in_hospital", "number_inpatient", "number_emergency"]:
            print(f"  {col:<18} -- normal median: {normal[col].median():>5.0f}  "
                  f"|  flagged median: {anomalous[col].median():>5.0f}")

    print("\n--- 5 most anomalous patients in the training set ---")
    top5 = results["train"].nsmallest(5, "anomaly_score")
    cols = ["patient_nbr", "time_in_hospital", "num_medications",
            "number_inpatient", "number_emergency", "readmitted_30d", "anomaly_score"]
    print(top5[cols].to_string(index=False))


if __name__ == "__main__":
    results = run()
    summarize(results)
