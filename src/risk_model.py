"""
risk_model.py

Small shared helpers so every script turns the Random Forest's raw output into
the SAME calibrated risk. The forest ranks patients well, but its raw
probabilities are squeezed toward the average (the riskiest tenth is
under-predicted, the safest tenth over-predicted). Platt scaling fixes that
with just two numbers (a slope and an intercept) fitted on the validation set.

Because the mapping is strictly increasing, it never changes who ranks above
whom: AUROC and the set of flagged patients stay exactly the same.
"""

import numpy as np

EPS = 1e-6


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def fit_calibration(raw_proba_val, y_val):
    """Fit slope and intercept on the validation set (logistic regression on the logit)."""
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(C=1e6, max_iter=1000)  # effectively no regularization
    lr.fit(_logit(raw_proba_val).reshape(-1, 1), np.asarray(y_val))
    return {"slope": float(lr.coef_[0][0]), "intercept": float(lr.intercept_[0])}


def calibrate(cal, raw):
    """Map raw forest probabilities to calibrated risk. cal=None means no change."""
    raw = np.asarray(raw, dtype=float)
    if cal is None:
        return raw
    return _sigmoid(cal["slope"] * _logit(raw) + cal["intercept"])


def local_slope(cal, raw):
    """d(calibrated risk) / d(raw probability) at each patient. Used to rescale
    SHAP contributions (which are in raw-probability units) to the displayed scale."""
    raw = np.clip(np.asarray(raw, dtype=float), EPS, 1 - EPS)
    if cal is None:
        return np.ones_like(raw)
    c = calibrate(cal, raw)
    return cal["slope"] * c * (1 - c) / (raw * (1 - raw))


def raw_proba(bundle, X):
    return bundle["model"].predict_proba(X)[:, 1]


def predict_risk(bundle, X):
    """Calibrated 30-day readmission risk for the rows of X."""
    return calibrate(bundle.get("calibration"), raw_proba(bundle, X))
