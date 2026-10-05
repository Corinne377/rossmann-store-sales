# Validation design

The competition test dates are 2015-08-01 through 2015-09-17. The validation option holds out the last 42 calendar days of labeled data, ending 2015-07-31, and trains only on earlier dates. This mirrors the forecast horizon and avoids random-split leakage.

The scoring helper excludes actual zero-sales rows, matching the competition's RMSPE description. The model is trained on log1p(Sales), then transformed back to nonnegative sales. Test rows with Open=0 are explicitly set to zero.

## Measured run

- Validation RMSPE: **0.175309**
- Validation window: 2015-06-20 through 2015-07-31 (42 calendar days)
- CatBoost configuration: depth 9, learning rate 0.055, maximum 450 iterations, early stopping; best iteration 417 (418 trees)
- Full-data fit: 418 trees
- Submission: 41,088 rows; IDs match `sample_submission.csv` in order; closed stores have zero predictions

This score is a single forward holdout estimate. Keep experimenting with time-based validation and compare RMSPE before claiming a model improvement.
