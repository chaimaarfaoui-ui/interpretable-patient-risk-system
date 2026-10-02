"""
generate_synthetic_data.py

Generates a synthetic patient dataset shaped like the UCI Diabetes 130-US
Hospitals readmission dataset. Use this to build/test the full pipeline
now. Swap in the real CSV later -- as long as you rename columns to match,
nothing downstream needs to change.

Target: readmitted_30d (1 = readmitted within 30 days, 0 = not)
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
N_PATIENTS = 6000


def generate(n=N_PATIENTS) -> pd.DataFrame:
    age = RNG.integers(18, 95, n)
    sex = RNG.choice(["M", "F"], n)

    # Encounter / history features
    num_prior_visits = RNG.poisson(1.2, n)
    num_prior_visits = np.clip(num_prior_visits, 0, 15)
    num_medications = RNG.poisson(8, n)
    num_diagnoses = RNG.integers(1, 10, n)
    time_in_hospital = RNG.integers(1, 14, n)

    # Labs (with realistic missingness baked in)
    hba1c = RNG.normal(7.0, 1.6, n)
    hba1c = np.clip(hba1c, 4.0, 14.0)
    hba1c_missing_mask = RNG.random(n) < 0.35  # labs often missing
    hba1c_observed = hba1c.copy()
    hba1c_observed[hba1c_missing_mask] = np.nan

    glucose_serum = RNG.normal(140, 45, n)
    glucose_missing_mask = RNG.random(n) < 0.5
    glucose_observed = glucose_serum.copy()
    glucose_observed[glucose_missing_mask] = np.nan

    # Categorical clinical context
    admission_type = RNG.choice(
        ["Emergency", "Urgent", "Elective"], n, p=[0.55, 0.25, 0.20]
    )
    has_diabetes_complication = RNG.choice([0, 1], n, p=[0.7, 0.3])
    insulin_use = RNG.choice(["No", "Steady", "Up", "Down"], n, p=[0.4, 0.35, 0.15, 0.10])
    discharge_disposition = RNG.choice(
        ["Home", "SNF", "Home Health", "Other"], n, p=[0.6, 0.15, 0.15, 0.10]
    )

    # --- Construct a real (but noisy) underlying risk signal ---
    # This is what our model SHOULD learn to recover.
    risk_score = (
        0.02 * (age - 50)
        + 0.35 * num_prior_visits
        + 0.05 * num_medications
        + 0.15 * num_diagnoses
        + 0.25 * has_diabetes_complication
        + 0.06 * np.nan_to_num(hba1c_observed, nan=7.0)
        + 0.01 * np.nan_to_num(glucose_observed, nan=140.0)
        + np.where(admission_type == "Emergency", 0.6, 0.0)
        + np.where(discharge_disposition == "SNF", 0.5, 0.0)
        + np.where(insulin_use == "Up", 0.3, 0.0)
        + RNG.normal(0, 1.5, n)  # irreducible noise
    )

    # Shift the decision boundary up (not centered on the mean) so the
    # positive rate lands around a realistic ~15%, producing a class
    # imbalance you'll need to handle deliberately in Module 2.
    threshold = np.quantile(risk_score, 0.85)
    prob = 1 / (1 + np.exp(-(risk_score - threshold) / risk_score.std()))
    readmitted_30d = RNG.binomial(1, np.clip(prob, 0.01, 0.9))

    # Inject a handful of true anomalies: patients whose profile is
    # statistically bizarre regardless of their label (e.g. extreme values
    # combined in ways that basically never happen together).
    n_anomalies = int(0.02 * n)
    anomaly_idx = RNG.choice(n, n_anomalies, replace=False)
    num_medications = num_medications.astype(float)
    num_medications[anomaly_idx] = RNG.integers(35, 60, n_anomalies)
    time_in_hospital[anomaly_idx] = RNG.integers(30, 45, n_anomalies)

    df = pd.DataFrame(
        {
            "patient_id": np.arange(1, n + 1),
            "age": age,
            "sex": sex,
            "admission_type": admission_type,
            "time_in_hospital": time_in_hospital,
            "num_prior_visits": num_prior_visits,
            "num_medications": num_medications,
            "num_diagnoses": num_diagnoses,
            "hba1c": hba1c_observed,
            "glucose_serum": glucose_observed,
            "has_diabetes_complication": has_diabetes_complication,
            "insulin_use": insulin_use,
            "discharge_disposition": discharge_disposition,
            "readmitted_30d": readmitted_30d,
        }
    )

    return df


if __name__ == "__main__":
    df = generate()
    out_path = "/mnt/user-data/outputs/project_b/data/patients_raw.csv"
    import os

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Generated {len(df)} synthetic patients -> {out_path}")
    print(f"Readmission rate: {df['readmitted_30d'].mean():.2%}")
    print(f"Missingness -> hba1c: {df['hba1c'].isna().mean():.1%}, "
          f"glucose_serum: {df['glucose_serum'].isna().mean():.1%}")
