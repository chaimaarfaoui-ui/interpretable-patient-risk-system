"""
Checks that Platt calibration (src/risk_model.py) only changes the
percentages, never the ranking of patients.

Run from the project folder with either:
    python tests/test_calibration.py
    pytest tests/test_calibration.py     (if pytest is installed)
"""
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from risk_model import calibrate, fit_calibration  # noqa: E402


def _squeezed_data(n=20000, seed=0):
    """Fake patients whose predicted risks are squeezed toward the average."""
    rng = np.random.default_rng(seed)
    true_risk = rng.beta(1.2, 11, n)                  # real risks, mean ~ 0.09
    y = (rng.random(n) < true_risk).astype(int)       # outcomes
    raw = 0.09 + 0.5 * (true_risk - 0.09)             # shrunk toward 0.09
    return raw, y


def test_ranking_is_preserved():
    raw, y = _squeezed_data()
    cal = fit_calibration(raw, y)
    out = calibrate(cal, raw)
    order_before = np.argsort(raw, kind="stable")
    assert np.all(np.diff(out[order_before]) >= 0), "calibration reordered patients"


def test_auroc_is_unchanged():
    raw, y = _squeezed_data()
    out = calibrate(fit_calibration(raw, y), raw)
    assert abs(roc_auc_score(y, raw) - roc_auc_score(y, out)) < 1e-9


def test_squeezed_scores_are_stretched():
    raw, y = _squeezed_data()
    cal = fit_calibration(raw, y)
    assert cal["slope"] > 1, "squeezed probabilities should give a slope above 1"
    out = calibrate(cal, raw)
    assert out.max() - out.min() > raw.max() - raw.min()


def test_outputs_stay_valid_probabilities():
    raw, y = _squeezed_data()
    out = calibrate(fit_calibration(raw, y), raw)
    assert out.min() >= 0 and out.max() <= 1


def test_no_calibration_returns_raw():
    raw = np.array([0.01, 0.1, 0.5])
    assert np.allclose(calibrate(None, raw), raw)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"PASS  {name}")
    print("All calibration tests passed.")
