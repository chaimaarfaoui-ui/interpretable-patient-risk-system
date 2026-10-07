# Interpretable Patient Risk System

**Explainable 30-day hospital readmission risk, with a human in the loop.**

A readmission risk system built on the UCI "Diabetes 130-US Hospitals"
dataset. It doesn't just predict risk. It flags patients whose overall
profile looks statistically unusual, explains every prediction in plain
language, and routes uncertain or high-stakes cases to a human reviewer
instead of acting alone.

**At a glance (held-out test set, 10,496 patients):**
Random Forest with **AUROC 0.64 (95% CI 0.62 to 0.66)** and well-calibrated
average risk (8.9% predicted vs 9.0% actual) · flagging the top 20% of
patients catches **36% of readmissions at 16% precision (1.8x better than
random)** · Isolation Forest flags ~2% of patients, who are readmitted at
~2x the normal rate · SHAP explains every prediction · **18.8% of patients
are routed to human review, and they are readmitted at 15.4% vs 7.5% for
everyone else** · every review decision is logged.

![Human-in-the-loop review UI](assets/review_ui_screenshot.png)
*The review interface: risk against the average patient, why the case was
routed, the top reasons with their size, and an agree/disagree decision.*

## Architecture

```
Raw UCI data (101,766 encounters)
        |
        v
[data_pipeline.py]          keep 1 encounter per patient, drop unusable
                             columns and hospice/death discharges, ICD-9
                             grouping, 70/15/15 stratified split
        |
        v
[anomaly_detection.py]      Isolation Forest trained on training patients
                             only -> anomaly_score + is_anomaly on every row
        |
        v
[train_baseline.py]         Logistic Regression baseline + Random Forest
                             (kept model). No class weighting, so predicted
                             risk is a real probability. Saves the model
                             and a decision threshold (top 20% of risk).
        |
        v
[explainability.py]         SHAP TreeExplainer -> per-patient reasons +
                             global feature importance
        |
        v
[generate_review_queue.py]  Gate (any one triggers review):
                             - risk within the closest 10% to the decision
                               threshold (model is on the fence)
                             - anomaly flagged
                             - risk in the top 8% (high stakes)
                             Builds the review queue with plain-English
                             SHAP reasons and updates the review UI
        |
        v
[app/human_review_ui.html]  Reviewer sees each gated case, agrees or
                             disagrees, decisions logged and exportable
        |
        v
[evaluate_test.py]          One-time final evaluation on the test set:
                             confidence intervals, calibration, gate check,
                             subgroup audit
```

## Results (held-out test set)

| Metric | Value | Reference point |
|---|---|---|
| AUROC | **0.638** (95% CI 0.620 to 0.656) | 0.5 = coin flip |
| AUPRC | 0.156 (95% CI 0.141 to 0.175) | 0.090 for random guessing |
| Brier score | 0.080 (95% CI 0.076 to 0.085) | 0.082 for always predicting the 9% base rate |
| Mean predicted risk | 8.9% | 9.0% actual |
| Flag top 20%: recall | 36% | |
| Flag top 20%: precision | 16% | 1.8x the 9% base rate |

Logistic regression, the baseline, reaches a validation AUROC of 0.632
against 0.643 for the Random Forest.

Predictive power is modest, and that is the honest headline. The inputs are
administrative records (counts, codes, dates). They contain no clinical
notes, labs, or information about home support or follow-up care, which
strongly influence readmission. Published results on this dataset vary
widely, and scores depend heavily on whether the same patient can appear in
both the training and test data. This project keeps only each patient's
first encounter, so no patient is ever in more than one split.

### Calibration

Risk is well calibrated on average but compressed at the extremes. The model
is slightly under-confident: the highest-risk tenth of patients is predicted
at 13.6% but is readmitted at 19.4%, and the lowest tenth is predicted at
5.9% but is readmitted at 3.4%. The probabilities rank patients usefully,
but should not be quoted as exact risks.

### Does the review gate work?

| Group | Share of patients | Readmitted |
|---|---|---|
| Routed to human review (any rule) | 18.8% | **15.4%** |
| Not routed | 81.2% | 7.5% |
| High stakes | 7.8% | 20.4% |
| Anomaly flagged | 2.1% | 18.3% |
| Low confidence (near threshold) | 10.4% | 12.0% |

The gated group is readmitted at twice the rate of everyone else and
contains about a third of all readmissions. The "low confidence" group is
close to the average by design: it holds the cases where the model can't
tell. Gate cutoffs are learned on the validation set and applied unchanged
to the test set.

### The anomaly layer

The Isolation Forest never sees the readmission label, yet flagged patients
are readmitted at 16 to 18% against about 9% for the rest, and they have more
medications, longer stays, and more prior inpatient visits. For those
patients the risk model predicts only 13.2% on average against an actual
18.3%, so the safety net catches patients the main model underestimates.

### Subgroup audit

| Group | Finding |
|---|---|
| Gender | Similar calibration. AUROC 0.65 for women, 0.62 for men. |
| Race | Calibration similar across groups. Recall is lower for African American patients (28%) than for Caucasian patients (38%), with flag rates of 15% vs 22%. Hispanic, Asian, Other, and Missing groups are too small to judge. |
| Age | The model leans heavily on age: it flags 35% of patients aged 80 to 90 but only about 9 to 12% of those aged 40 to 60, and slightly under-predicts risk for ages 70 to 80 (11.2% actual vs 9.5% predicted). |

These are differences to monitor, not explanations. Full tables are in
`outputs/evaluation_report.json`.

## Responsible AI reflection

**Honest probabilities came from fixing a mistake.** My first model used
`class_weight="balanced"` to help it notice the rare readmitted patients.
That pushed every score toward 0.5. The SHAP base rate printed 0.500 instead
of the true 9%, so a "65% risk" was not a 65% chance, and the review gate's
"low confidence" band was centered on a number that only existed because of
the weighting. I removed the weighting, re-centered the gate on the model's
real decision threshold, and checked calibration directly. Ranking quality
did not change (AUROC stayed near 0.64).

**Fixed thresholds failed; percentiles worked.** My first gate used cutoffs
that sounded reasonable (flag 0.40 to 0.60 as uncertain, 0.70 or more as
high-stakes) and it routed 85% of patients, which defeats the point of a
gate. Setting cutoffs from the model's own output distribution fixed it.
Cutoffs that look sensible on paper can be silently wrong for a specific
model, so check the real distribution before trusting them.

**The test set was used once, at the end.** All model and threshold
decisions were made on validation data. The numbers above come from a single
run of `evaluate_test.py`, with bootstrap confidence intervals.

## Limitations

- **Modest predictive power.** AUROC 0.64, and the Brier score barely beats
  always guessing the base rate. The model ranks patients, but cannot say
  much about any single one.
- **Probabilities are compressed at the extremes** (see Calibration).
- **Fairness gaps exist and are unexplained.** Recall differs by race, and
  flag rates differ strongly by age. This dataset has known historical
  quirks in how race was recorded.
- **Small groups were not evaluated.** Several race and age groups have too
  few patients for reliable estimates.
- **Old, single-condition data.** Diabetic inpatients at 130 US hospitals,
  1999 to 2008, administrative features only, first encounter per patient.
- **The review step is simulated.** The interface logs decisions, but they
  do not yet feed back into retraining, and the "reviewers" are the person
  running the demo.
- **The anomaly detector is not yet tested against known outliers.** I have
  shown that flagged patients differ and are riskier, but not a measured
  detection rate on injected anomalies.
- **Not a clinical tool.** This is a portfolio project and must not be used
  to make care decisions.

## Next steps

- Recalibrate the probabilities (isotonic or Platt scaling) and explain the
  calibrated score consistently.
- Try gradient boosting or an Explainable Boosting Machine and use the
  discarded `diag_2` and `diag_3` diagnosis columns.
- Test the anomaly layer by inserting synthetic outliers and measuring the
  detection rate.
- Add automated tests and a one-command pipeline.
- Close the loop: use logged reviewer decisions for retraining.

## Project structure

```
interpretable-patient-risk-system/
├── data/
│   ├── raw/              # not committed, see "Getting the data"
│   └── processed/        # not committed, regenerated by data_pipeline.py
├── src/
│   ├── data_pipeline.py           # Module 1: clean + split
│   ├── train_baseline.py          # Module 2: baseline + Random Forest
│   ├── anomaly_detection.py       # Module 3: Isolation Forest
│   ├── explainability.py          # Module 4: SHAP
│   ├── generate_review_queue.py   # Module 5: gate + review data
│   ├── evaluate_test.py           # final test-set evaluation
│   └── generate_synthetic_data.py # legacy: early practice data, not used
├── app/
│   └── human_review_ui.html       # review UI, open directly in a browser
├── assets/
│   └── review_ui_screenshot.png
├── models/
│   └── rf_model.pkl               # not committed, made by train_baseline.py
├── outputs/
│   ├── global_feature_importance.csv
│   ├── review_queue.json
│   └── evaluation_report.json
└── requirements.txt
```

## Getting the data

Download the "Diabetes 130-US Hospitals for Years 1999-2008" dataset (UCI
Machine Learning Repository, dataset 296) and place `diabetic_data.csv` and
`IDS_mapping.csv` in `data/raw/`.

## Running it

```bash
pip install -r requirements.txt

python src/data_pipeline.py          # clean + split
python src/anomaly_detection.py      # flag outlier patients
python src/train_baseline.py         # train + save model
python src/explainability.py         # SHAP explanations
python src/generate_review_queue.py  # build the review queue + update the UI
python src/evaluate_test.py          # final test-set evaluation (run once)
```

Then open `app/human_review_ui.html` in a browser (on Windows:
`start app/human_review_ui.html`; on macOS: `open app/human_review_ui.html`).

Order matters. `anomaly_detection.py` rewrites the processed CSVs, so if you
re-run `data_pipeline.py` you must re-run `anomaly_detection.py` afterwards,
or the `is_anomaly` column disappears and later steps fail.
`generate_review_queue.py` and `evaluate_test.py` both need it.

## Dataset citation

Strack, B., DeShazo, J.P., Gennings, C., et al. (2014). "Impact of HbA1c
Measurement on Hospital Readmission Rates: Analysis of 70,000 Clinical
Database Patient Records." *BioMed Research International*. Dataset via
UCI Machine Learning Repository.

## License

MIT. See `LICENSE`.
