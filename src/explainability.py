"""
explainability.py

Module 4: Explainability Layer.

Uses SHAP (SHapley Additive exPlanations) to answer, per patient: "which
features pushed this specific risk score up or down, and by how much?"
Also produces a global summary: which features matter most across the
whole population (useful for sanity-checking the model itself).

SHAP's core idea: take a patient's prediction, then ask "how much did
each feature contribute, on average, across every possible order you
could reveal the features in?" That averaging is what makes the
contributions fair and additive -- they sum exactly to (prediction - the
model's average prediction).
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import shap

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "processed"
MODEL_PATH = ROOT / "models" / "rf_model.pkl"
OUTPUTS_DIR = ROOT / "outputs"
N_EXPLAIN = 3  # number of individual patients to explain in detail


def load_model():
    with open(MODEL_PATH, "rb") as f:
        bundle = pickle.load(f)
    return bundle


def build_X(df, bundle):
    num = df[bundle["num_cols"]].fillna(0).to_numpy()
    cat = bundle["encoder"].transform(df[bundle["cat_cols"]].astype(str))
    return np.hstack([num, cat])


def explain_patient(shap_values, feature_names, patient_row, base_value, top_n=5):
    """Turn one patient's raw SHAP vector into a human-readable reason list."""
    contributions = list(zip(feature_names, shap_values))
    contributions.sort(key=lambda x: abs(x[1]), reverse=True)
    lines = []
    for feat, val in contributions[:top_n]:
        direction = "increased" if val > 0 else "decreased"
        lines.append(f"    {feat:<30} {direction} risk by {abs(val):.3f}")
    return lines


def run():
    bundle = load_model()
    model = bundle["model"]
    feature_names = bundle["feature_names"]

    val = pd.read_csv(f"{DATA_DIR}/val.csv")
    X_val = build_X(val, bundle)

    print("Building SHAP explainer (TreeExplainer -- exact & fast for tree models)...")
    explainer = shap.TreeExplainer(model)

    # Explain a sample of the val set (SHAP on the full RF can be slow --
    # 1500 patients is plenty to get a reliable global picture).
    sample_idx = np.random.RandomState(42).choice(len(X_val), size=min(1500, len(X_val)), replace=False)
    X_sample = X_val[sample_idx]

    shap_values = explainer.shap_values(X_sample)
    # shap_values shape for binary RF: (n_samples, n_features, 2) or list of 2 arrays
    if isinstance(shap_values, list):
        shap_pos = shap_values[1]
        base_value = explainer.expected_value[1]
    elif shap_values.ndim == 3:
        shap_pos = shap_values[:, :, 1]
        base_value = explainer.expected_value[1]
    else:
        shap_pos = shap_values
        base_value = explainer.expected_value

    # --- Global importance: mean absolute SHAP value per feature ---
    mean_abs = np.abs(shap_pos).mean(axis=0)
    global_importance = sorted(zip(feature_names, mean_abs), key=lambda x: x[1], reverse=True)

    print(f"\nBase rate (average predicted risk across the sample): {base_value:.3f}")
    print("\n--- Global feature importance (top 10, mean |SHAP value|) ---")
    for feat, imp in global_importance[:10]:
        print(f"  {feat:<30} {imp:.4f}")

    # --- Individual explanations: pick a few interesting patients ---
    proba = model.predict_proba(X_sample)[:, 1]
    val_sample = val.iloc[sample_idx].reset_index(drop=True)

    # Show the highest-risk patient, the highest anomaly-flagged patient,
    # and a random typical one -- variety demonstrates the layer well.
    highest_risk_i = int(np.argmax(proba))
    picks = {"Highest predicted risk": highest_risk_i}
    if "is_anomaly" in val_sample.columns and val_sample["is_anomaly"].sum() > 0:
        anomaly_candidates = val_sample.index[val_sample["is_anomaly"] == 1]
        picks["Flagged anomaly (from Module 3)"] = int(anomaly_candidates[0])
    picks["A typical / average-risk patient"] = int(np.argmin(np.abs(proba - proba.mean())))

    print("\n--- Individual patient explanations ---")
    for label, i in picks.items():
        print(f"\n[{label}] patient_nbr={val_sample.loc[i, 'patient_nbr']}")
        print(f"  Predicted risk: {proba[i]:.1%}   "
              f"(Actual outcome: {'Readmitted <30d' if val_sample.loc[i, 'readmitted_30d'] == 1 else 'Not readmitted'})")
        print("  Top reasons:")
        for line in explain_patient(shap_pos[i], feature_names, val_sample.loc[i], base_value):
            print(line)

    # Save global importance + the sampled SHAP values for reuse in Module 5
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(
        OUTPUTS_DIR / "shap_sample.npz",
        shap_values=shap_pos, sample_idx=sample_idx, base_value=base_value,
    )
    pd.DataFrame(global_importance, columns=["feature", "mean_abs_shap"]).to_csv(
        OUTPUTS_DIR / "global_feature_importance.csv", index=False
    )
    print("\nSaved -> outputs/shap_sample.npz, outputs/global_feature_importance.csv")


if __name__ == "__main__":
    run()
