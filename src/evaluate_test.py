"""
evaluate_test.py

Final evaluation on the held-out TEST set. Run this once, after all model and
threshold decisions are finished -- the test set should never guide tuning.

What it reports:
  1. Headline metrics with 95% bootstrap confidence intervals
     (AUROC, AUPRC, Brier score vs a "predict the base rate" baseline,
      precision/recall at the decision threshold)
  2. Reliability table: predicted risk vs actual readmission rate by decile
  3. Gate-bucket analysis: does the human-review gate actually route riskier
     or harder cases? (cutoffs are learned on VAL, then applied to TEST)
  4. Subgroup audit by gender, age, and race: calibration, flag rate, recall

Outputs: console report + outputs/evaluation_report.json
"""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from generate_review_queue import HIGH_STAKES_PERCENTILE, LOW_CONF_PERCENTILE, build_X

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "processed"
MODEL_PATH = ROOT / "models" / "rf_model.pkl"
OUT_PATH = ROOT / "outputs" / "evaluation_report.json"

N_BOOT = 1000
MIN_GROUP = 300      # subgroups smaller than this get no AUROC (too noisy)
MIN_POSITIVES = 30
SUBGROUP_COLUMNS = ["gender", "age", "race"]


def score(df, bundle):
    return bundle["model"].predict_proba(build_X(df, bundle))[:, 1]


def bootstrap_ci(y, p, threshold, n_boot=N_BOOT, seed=42):
    """95% percentile bootstrap intervals, resampling patients with replacement."""
    rng = np.random.RandomState(seed)
    n = len(y)
    stats = {k: [] for k in ["auroc", "auprc", "brier", "precision", "recall"]}
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        yb, pb = y[idx], p[idx]
        if yb.min() == yb.max():
            continue
        flag = pb >= threshold
        tp = int((flag & (yb == 1)).sum())
        stats["auroc"].append(roc_auc_score(yb, pb))
        stats["auprc"].append(average_precision_score(yb, pb))
        stats["brier"].append(brier_score_loss(yb, pb))
        stats["precision"].append(tp / max(int(flag.sum()), 1))
        stats["recall"].append(tp / max(int((yb == 1).sum()), 1))
    return {k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
            for k, v in stats.items()}


def headline(y, p, threshold, train_rate):
    flag = p >= threshold
    tp = int((flag & (y == 1)).sum())
    point = {
        "auroc": roc_auc_score(y, p),
        "auprc": average_precision_score(y, p),
        "brier": brier_score_loss(y, p),
        "precision": tp / max(int(flag.sum()), 1),
        "recall": tp / max(int((y == 1).sum()), 1),
    }
    ci = bootstrap_ci(y, p, threshold)
    extras = {
        "n": int(len(y)),
        "prevalence": float(y.mean()),
        "mean_predicted_risk": float(p.mean()),
        "brier_baseline_base_rate": float(brier_score_loss(y, np.full(len(y), train_rate))),
        "decision_threshold": float(threshold),
        "share_flagged": float(flag.mean()),
        "lift_at_threshold": point["precision"] / float(y.mean()),
    }
    return point, ci, extras


def reliability_table(y, p, bins=10):
    d = pd.DataFrame({"y": y, "p": p})
    d["bin"] = pd.qcut(d["p"], bins, labels=False, duplicates="drop")
    t = d.groupby("bin").agg(n=("y", "size"), mean_predicted=("p", "mean"),
                             actual_rate=("y", "mean")).reset_index(drop=True)
    t.index = [f"decile {i + 1}" for i in range(len(t))]
    return t


def gate_table(d, threshold, high_cut, low_cut):
    """d needs columns: y, p, is_anomaly."""
    low_conf = (d["p"] - threshold).abs() <= low_cut
    anomaly = d["is_anomaly"] == 1
    high = d["p"] >= high_cut
    any_gate = low_conf | anomaly | high
    masks = {
        "Low confidence (near threshold)": low_conf,
        "Anomaly flagged": anomaly,
        "High stakes (top risk)": high,
        "ANY gate -> human review": any_gate,
        "No gate -> auto-handled": ~any_gate,
    }
    correct = ((d["p"] >= threshold).astype(int) == d["y"])
    rows = {}
    for name, m in masks.items():
        sub = d[m]
        rows[name] = {
            "n": int(m.sum()),
            "share": float(m.mean()),
            "readmit_rate": float(sub["y"].mean()) if len(sub) else np.nan,
            "mean_predicted": float(sub["p"].mean()) if len(sub) else np.nan,
            "accuracy_at_threshold": float(correct[m].mean()) if len(sub) else np.nan,
        }
    out = pd.DataFrame(rows).T
    out["n"] = out["n"].astype(int)
    return out


def group_table(d, col, threshold):
    rows = []
    for g, sub in d.groupby(col):
        n, pos = len(sub), int(sub["y"].sum())
        flag = sub["p"] >= threshold
        auroc = None
        if n >= MIN_GROUP and pos >= MIN_POSITIVES and pos < n:
            auroc = roc_auc_score(sub["y"], sub["p"])
        rows.append({
            "group": str(g), "n": n,
            "readmit_rate": sub["y"].mean(),
            "mean_predicted": sub["p"].mean(),
            "flag_rate": flag.mean(),
            "recall": ((flag) & (sub["y"] == 1)).sum() / pos if pos else np.nan,
            "auroc": auroc,
        })
    return pd.DataFrame(rows).set_index("group")


def records(df):
    return df.astype(object).where(df.notna(), None).reset_index().to_dict("records")


def fmt(df, pct_cols=(), dec_cols=()):
    out = df.copy()
    for c in pct_cols:
        out[c] = out[c].map(lambda v: "-" if pd.isna(v) else f"{v:.1%}")
    for c in dec_cols:
        out[c] = out[c].map(
            lambda v: "-" if v is None or pd.isna(v) else f"{v:.3f}")
    return out


def run():
    with open(MODEL_PATH, "rb") as f:
        bundle = pickle.load(f)
    threshold = bundle.get("threshold")

    train = pd.read_csv(f"{DATA_DIR}/train.csv")
    val = pd.read_csv(f"{DATA_DIR}/val.csv")
    test = pd.read_csv(f"{DATA_DIR}/test.csv")

    p_val, p_test = score(val, bundle), score(test, bundle)
    if threshold is None:
        threshold = float(np.quantile(p_val, 0.80))
    y_test = test["readmitted_30d"].to_numpy()

    # Gate cutoffs come from VAL only -- test must not influence them.
    high_cut = float(np.quantile(p_val, HIGH_STAKES_PERCENTILE))
    low_cut = float(np.quantile(
        np.abs(p_val - threshold), LOW_CONF_PERCENTILE))

    print("=" * 64)
    print("FINAL TEST-SET EVALUATION  (run once; do not tune on these numbers)")
    print("=" * 64)

    # 1. Headline metrics
    point, ci, ex = headline(y_test, p_test, threshold,
                             float(train["readmitted_30d"].mean()))
    print(f"\n[1] Headline metrics  (n={ex['n']:,}, prevalence {ex['prevalence']:.2%}, "
          f"threshold {ex['decision_threshold']:.3f}, {N_BOOT} bootstrap resamples)")
    for k, label in [("auroc", "AUROC"), ("auprc", "AUPRC"), ("brier", "Brier score"),
                     ("precision", "Precision @ threshold"), ("recall", "Recall @ threshold")]:
        lo, hi = ci[k]
        print(f"  {label:<22} {point[k]:.3f}   (95% CI {lo:.3f} - {hi:.3f})")
    print(
        f"  {'AUPRC if random':<22} {ex['prevalence']:.3f}   <- AUPRC must beat this")
    print(
        f"  {'Brier, always-base-rate':<22} {ex['brier_baseline_base_rate']:.3f}   <- Brier must beat this")
    print(
        f"  Mean predicted risk {ex['mean_predicted_risk']:.3f} vs actual {ex['prevalence']:.3f}")
    print(f"  Share flagged {ex['share_flagged']:.1%}  |  precision lift over random: "
          f"{ex['lift_at_threshold']:.2f}x")

    # 2. Reliability
    rel = reliability_table(y_test, p_test)
    print(
        "\n[2] Reliability by risk decile  (well calibrated => the two columns match)")
    print(fmt(rel, pct_cols=["mean_predicted", "actual_rate"]).to_string())

    # 3. Gate buckets
    d = pd.DataFrame(
        {"y": y_test, "p": p_test, "is_anomaly": test["is_anomaly"].to_numpy()})
    gate = gate_table(d, threshold, high_cut, low_cut)
    print(f"\n[3] Review-gate buckets  (cutoffs from VAL: high-stakes >= {high_cut:.3f}, "
          f"low-confidence within {low_cut:.3f} of {threshold:.3f})")
    print(fmt(gate, pct_cols=["share", "readmit_rate", "mean_predicted",
                              "accuracy_at_threshold"]).to_string())
    print("  Read this as: is the readmit rate of gated groups higher than the 'No gate' row?\n"
          "  If not, the gate isn't earning its keep. (Accuracy in the High-stakes row equals\n"
          "  precision, since everyone there is flagged, so it is low by construction.)")

    # 4. Subgroups
    d_full = test.copy()
    d_full["y"], d_full["p"] = y_test, p_test
    print("\n[4] Subgroup audit  (AUROC shown only for groups with enough data)")
    sub_out = {}
    for col in SUBGROUP_COLUMNS:
        if col not in d_full.columns:
            continue
        t = group_table(d_full, col, threshold)
        sub_out[col] = records(t)
        print(f"\n  by {col}:")
        print(fmt(t, pct_cols=["readmit_rate", "mean_predicted", "flag_rate", "recall"],
                  dec_cols=["auroc"]).to_string())

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "headline": {"point": point, "ci95": ci, **ex},
        "reliability": records(rel),
        "gate_buckets": records(gate),
        "subgroups": sub_out,
        "gate_cutoffs_from_val": {"high_stakes": high_cut, "low_conf_halfwidth": low_cut},
    }
    with open(OUT_PATH, "w") as f:
        json.dump(report, f, indent=2, default=lambda o: float(o))
    print(f"\nSaved -> {OUT_PATH}")


if __name__ == "__main__":
    run()
