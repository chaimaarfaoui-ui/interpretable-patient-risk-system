# Model Card: Interpretable Patient Risk System

A one-page summary of what this model is, what it is for, how well it works, and where it falls short.

## Model details
- **Task:** estimate the probability that a diabetic inpatient is readmitted to hospital within 30 days of discharge.
- **Model:** Random Forest (300 trees, max depth 8, min 20 patients per leaf, no class weighting), followed by Platt scaling (a slope and an intercept fitted on the validation set) so the output is a calibrated probability.
- **Supporting layers:** an Isolation Forest flags unusual patient profiles without using the outcome; SHAP gives per-patient reasons; a review gate routes cases to a human.
- **Baseline:** logistic regression (validation AUROC 0.632 vs 0.643 for the Random Forest).

## Intended use
- A portfolio and learning project showing how to combine prediction, calibration, explanation, anomaly detection and human review.
- Studying how a risk score and a review gate behave on a public dataset.

## Out-of-scope use
- **Not for clinical decisions.** Do not use it to decide anyone's care, discharge or follow-up.
- Not validated on any hospital other than the 130 in the source data, or on data after 2008.
- Not a substitute for clinician judgment: the simulated reviewer in the demo UI is not a real clinical workflow.

## Data
- UCI "Diabetes 130-US Hospitals for Years 1999-2008".
- 101,766 encounters reduced to 69,973 patients: first encounter per patient only, hospice and death discharges removed.
- Split 70/15/15 (train/validation/test), stratified; readmission rate 8.97% in each split.
- Inputs are administrative only (counts, codes, ages in bands, medications). No clinical notes, lab results or information about home support.

## Performance (held-out test set, 10,496 patients)
| Measure | Result |
|---|---|
| AUROC | 0.638 (95% CI 0.620 to 0.656) |
| AUPRC | 0.156 (random: 0.090) |
| Brier score | 0.080 (always guessing 9%: 0.082) |
| Flag top 20% (threshold 0.122) | recall 36%, precision 16%, 1.8x lift |
| Review gate | 18.9% of patients routed; readmitted at 15.5% vs 7.4% for the rest |

**Calibration:** lowest-risk tenth predicted 3.9% (actual 3.4%); highest-risk tenth predicted 18.3% (actual 19.4%). Some middle tenths are slightly off (for example the 2nd and 3rd tenths are predicted about one point low).

**Interpretation:** the model ranks patients better than chance but is modest. It cannot say much about any single person.

## Fairness findings
Checked on the test set; these are differences to monitor, not explanations.
- **Gender:** AUROC 0.65 for women, 0.62 for men.
- **Race:** recall is 28% for African American patients vs 38% for Caucasian patients (flag rates 15% vs 22%); mean predicted risk is a little low for African American patients (8.3% vs 9.0% actual). Asian, Hispanic, Other and Missing groups are too small to judge.
- **Age:** flags 35% of patients aged 80 to 90 but only 9 to 12% of those aged 40 to 60; risk is slightly under-predicted for ages 70 to 80 (9.9% vs 11.2% actual).
- Recalibration did not change these gaps, because they depend on ranking.

## Limitations
- Modest predictive power; Brier score barely beats the base rate.
- Old, single-condition data (diabetic inpatients, 1999 to 2008).
- SHAP reasons are rescaled to match the calibrated risk, so they are approximate and do not add up exactly.
- The anomaly detector has not yet been tested against injected outliers.
- Review decisions are logged but do not yet feed back into retraining.
- The test set was evaluated twice (before and after adding calibration); the reported numbers are from the second run and no changes were made afterwards.

## Reproduce
```
pip install -r requirements.txt
python src/data_pipeline.py
python src/anomaly_detection.py
python src/train_baseline.py
python src/explainability.py
python src/generate_review_queue.py
python src/evaluate_test.py
```
Run `train_baseline.py` before anything that loads the saved model.
