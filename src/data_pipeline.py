"""
data_pipeline.py

Cleans and prepares the real UCI "Diabetes 130-US Hospitals" dataset for
readmission prediction. Run after downloading diabetic_data.csv from UCI
(dataset 296) and placing it in project_b/data/raw/.

Target: binary readmitted_30d (1 = readmitted within 30 days, 0 = otherwise)
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = ROOT / "data" / "raw" / "diabetic_data.csv"
IDS_PATH = ROOT / "data" / "raw" / "IDS_mapping.csv"
OUT_DIR = ROOT / "data" / "processed"

# Columns known to be near-unusable: see EDA. weight/max_glu_serum/A1Cresult
# are >80% missing; examide/citoglipton have zero variance (single value
# for every single patient, so they carry no information).
DROP_COLS = [
    "weight", "max_glu_serum", "A1Cresult",
    "examide", "citoglipton",
    "encounter_id",  # identifier, not a feature
]

# Discharge codes meaning the patient died or went to hospice. These
# patients cannot be "readmitted" in any meaningful sense, and including
# them would bias the readmission rate and leak information the model
# has no business using.
HOSPICE_OR_DEATH_CODES = {11, 13, 14, 19, 20, 21}


def load_raw() -> pd.DataFrame:
    df = pd.read_csv(RAW_PATH, na_values="?", low_memory=False)
    return df


def dedupe_to_first_encounter(df: pd.DataFrame) -> pd.DataFrame:
    """Each patient may appear multiple times (multiple hospital stays).
    Using every encounter as an independent row leaks patient identity
    across train/val/test. Keep only each patient's first encounter."""
    first_ids = df.groupby("patient_nbr")["encounter_id"].transform("min")
    return df[df["encounter_id"] == first_ids].copy()


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Remove hospice/death dispositions before anything else.
    df = df[~df["discharge_disposition_id"].isin(HOSPICE_OR_DEATH_CODES)]

    # Drop unusable columns.
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

    # Binarize the target: "<30" = 1, everything else ("NO", ">30") = 0.
    df["readmitted_30d"] = (df["readmitted"] == "<30").astype(int)
    df = df.drop(columns=["readmitted"])

    # payer_code and medical_specialty are ~40-50% missing but the
    # missingness itself may be informative (e.g. specialty not recorded
    # could correlate with less-specialized care) -- keep as an explicit
    # "Missing" category rather than dropping or imputing blindly.
    for col in ["payer_code", "medical_specialty", "race"]:
        df[col] = df[col].fillna("Missing")

    # diag_1/2/3 are ICD-9 codes -- collapse into clinical categories per
    # the standard grouping used in the original Strack et al. paper.
    df["diag_1_group"] = df["diag_1"].apply(_map_icd9_group)
    df = df.drop(columns=["diag_1", "diag_2", "diag_3"])

    # Cap number_outpatient/emergency/inpatient outliers (a few patients
    # have 40+ prior visits -- keep as a flag rather than deleting rows,
    # since these are exactly the anomalies Module 3 should catch).
    return df


def _map_icd9_group(code) -> str:
    """Collapse an ICD-9 diagnosis code into a broad clinical category."""
    if pd.isna(code):
        return "Missing"
    try:
        if str(code).startswith(("V", "E")):
            return "Other"
        c = float(code)
    except ValueError:
        return "Other"

    if c == 250 or (250 <= c < 251):
        return "Diabetes"
    if (390 <= c <= 459) or c == 785:
        return "Circulatory"
    if (460 <= c <= 519) or c == 786:
        return "Respiratory"
    if (520 <= c <= 579) or c == 787:
        return "Digestive"
    if 800 <= c <= 999:
        return "Injury"
    if 710 <= c <= 739:
        return "Musculoskeletal"
    if (580 <= c <= 629) or c == 788:
        return "Genitourinary"
    if (140 <= c <= 239):
        return "Neoplasms"
    return "Other"


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["total_prior_visits"] = (
        df["number_outpatient"] + df["number_emergency"] + df["number_inpatient"]
    )
    df["has_prior_emergency"] = (df["number_emergency"] > 0).astype(int)
    df["med_change"] = (df["change"] == "Ch").astype(int)
    return df


def split(df: pd.DataFrame, seed=42):
    from sklearn.model_selection import train_test_split

    y = df["readmitted_30d"]
    train, temp = train_test_split(df, test_size=0.30, stratify=y, random_state=seed)
    val, test = train_test_split(
        temp, test_size=0.50, stratify=temp["readmitted_30d"], random_state=seed
    )
    return train, val, test


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Loading raw data...")
    df = load_raw()
    print(f"  {len(df):,} rows, {df['patient_nbr'].nunique():,} unique patients")

    print("Deduplicating to first encounter per patient...")
    df = dedupe_to_first_encounter(df)
    print(f"  {len(df):,} rows remain")

    print("Cleaning...")
    df = clean(df)
    print(f"  {len(df):,} rows remain after removing hospice/death dispositions")
    print(f"  Readmit-30d rate: {df['readmitted_30d'].mean():.2%}")

    print("Engineering features...")
    df = engineer_features(df)

    print("Splitting train/val/test (70/15/15, stratified)...")
    train, val, test = split(df)
    for name, part in [("train", train), ("val", val), ("test", test)]:
        path = f"{OUT_DIR}/{name}.csv"
        part.to_csv(path, index=False)
        print(f"  {name}: {len(part):,} rows, readmit rate "
              f"{part['readmitted_30d'].mean():.2%} -> {path}")

    print("\nDone. Columns in final dataset:")
    print(list(df.columns))
