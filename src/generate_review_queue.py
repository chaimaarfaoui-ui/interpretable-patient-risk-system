"""
generate_review_queue.py

Module 5 data prep: applies the confidence/anomaly/high-stakes gate rules
to the validation set, picks a sample of gated patients, and generates a
plain-English SHAP explanation for each -- exported as JSON for the
human-in-the-loop review UI to consume.

Gate rules (all OR'd together -- any one triggers review):
  - Low confidence: predicted risk between 0.40 and 0.60 (model is unsure)
  - Anomaly flagged: Module 3's Isolation Forest flagged this patient
  - High stakes: predicted risk >= 0.70 (acting on a wrong call here is costly)
"""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import shap

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "processed"
MODEL_PATH = ROOT / "models" / "rf_model.pkl"
OUT_PATH = ROOT / "outputs" / "review_queue.json"

N_SAMPLE = 18
# NOTE: thresholds recalibrated after inspecting the actual prediction
# distribution -- with an AUROC of ~0.64 and class_weight="balanced",
# this Random Forest's outputs cluster tightly around 0.47-0.53 (std
# ~0.065, max ~0.66). Fixed thresholds like ">=0.70" or "0.40-0.60"
# (reasonable-sounding in the abstract) turned out to gate 85% of
# patients -- useless for a review queue. Real weak-signal models often
# look like this: use PERCENTILES of the model's own output, not
# absolute cutoffs pulled from intuition.
HIGH_STAKES_PERCENTILE = 0.92   # top 8% of predicted risk
LOW_CONF_PERCENTILE = 0.10      # closest 10% to the 0.5 toss-up line

READABLE = {
    "number_inpatient": "prior inpatient stays",
    "total_prior_visits": "total prior visits",
    "time_in_hospital": "length of this hospital stay",
    "num_medications": "number of medications",
    "number_diagnoses": "number of diagnoses recorded",
    "discharge_disposition_id": "discharge disposition (where they're sent after leaving)",
    "num_lab_procedures": "number of lab procedures run",
    "has_prior_emergency": "history of ER visits",
    "payer_code_Missing": "missing payer/insurance info",
    "diabetesMed_Yes": "currently on diabetes medication",
}


def readable_name(feat: str) -> str:
    if feat in READABLE:
        return READABLE[feat]
    if feat.startswith("age_"):
        return f"age bracket {feat[4:]}"
    return feat.replace("_", " ")


def build_X(df, bundle):
    num = df[bundle["num_cols"]].fillna(0).to_numpy()
    cat = bundle["encoder"].transform(df[bundle["cat_cols"]].astype(str))
    return np.hstack([num, cat])


def gate_reasons(row, high_stakes_cut, low_conf_cut) -> list:
    reasons = []
    if abs(row["predicted_risk"] - 0.5) <= low_conf_cut:
        reasons.append("Low model confidence")
    if row["is_anomaly"] == 1:
        reasons.append("Anomaly flagged (Module 3)")
    if row["predicted_risk"] >= high_stakes_cut:
        reasons.append("High-stakes prediction")
    return reasons


def run():
    with open(MODEL_PATH, "rb") as f:
        bundle = pickle.load(f)
    model = bundle["model"]
    feature_names = bundle["feature_names"]

    val = pd.read_csv(f"{DATA_DIR}/val.csv")
    X_val = build_X(val, bundle)
    val["predicted_risk"] = model.predict_proba(X_val)[:, 1]

    high_stakes_cut = val["predicted_risk"].quantile(HIGH_STAKES_PERCENTILE)
    dist_from_mid = (val["predicted_risk"] - 0.5).abs()
    low_conf_cut = dist_from_mid.quantile(LOW_CONF_PERCENTILE)
    print(f"Calibrated cutoffs -- high-stakes: risk >= {high_stakes_cut:.3f}  |  "
          f"low-confidence: |risk-0.5| <= {low_conf_cut:.3f}")

    gated_mask = (
        (dist_from_mid <= low_conf_cut)
        | (val["is_anomaly"] == 1)
        | (val["predicted_risk"] >= high_stakes_cut)
    )
    gated = val[gated_mask].copy()
    print(f"{gated_mask.sum():,} of {len(val):,} patients ({gated_mask.mean():.1%}) hit the review gate")

    # Sample a mix so the UI shows variety, not all one gate reason
    sample = gated.sample(n=min(N_SAMPLE, len(gated)), random_state=7).reset_index(drop=True)
    sample_positions = val.index.get_indexer(val[val["patient_nbr"].isin(sample["patient_nbr"])].index)
    X_sample = build_X(sample, bundle)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    if isinstance(shap_values, list):
        shap_pos = shap_values[1]
    elif shap_values.ndim == 3:
        shap_pos = shap_values[:, :, 1]
    else:
        shap_pos = shap_values

    records = []
    for i, row in sample.iterrows():
        contributions = sorted(zip(feature_names, shap_pos[i]), key=lambda x: abs(x[1]), reverse=True)[:3]
        reasons = []
        for feat, val_ in contributions:
            direction = "increased" if val_ > 0 else "decreased"
            reasons.append(f"{readable_name(feat)} {direction} risk")

        records.append({
            "id": int(row["patient_nbr"]),
            "age": row["age"],
            "gender": row["gender"],
            "admission_type_id": int(row["admission_type_id"]),
            "time_in_hospital": int(row["time_in_hospital"]),
            "num_medications": int(row["num_medications"]),
            "number_inpatient": int(row["number_inpatient"]),
            "number_diagnoses": int(row["number_diagnoses"]),
            "diag_1_group": row["diag_1_group"],
            "predicted_risk": round(float(row["predicted_risk"]), 3),
            "is_anomaly": bool(row["is_anomaly"]),
            "gate_reasons": gate_reasons(row, high_stakes_cut, low_conf_cut),
            "top_reasons": reasons,
            "actual_outcome": "Readmitted <30d" if row["readmitted_30d"] == 1 else "Not readmitted",
        })

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(records, f, indent=2)
    print(f"Wrote {len(records)} review cases -> {OUT_PATH}")


if __name__ == "__main__":
    run()
