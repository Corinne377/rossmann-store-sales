# Validation notes

Both CatBoost models were evaluated on the same chronological holdout: 2015-06-20 through 2015-07-31 (42 days). Training rows precede the cutoff. RMSPE excludes rows whose actual Sales value is zero.

| Model | RMSPE |
|---|---:|
| Calendar-only baseline | 0.175309 |
| Past-history feature model | 0.161062 |

The history-feature model improved RMSPE by 8.1%. It uses past-only mean log sales grouped by store and weekday, and by store, weekday, and promotion. For training rows the aggregates are expanding and shifted so the current target is excluded. Holdout feature aggregates use only the pre-cutoff training period.

The history model used depth 9, learning rate 0.055, seed 42, and early stopping. The best iteration was 724 (zero-based), so 725 trees were used when fitting on all labeled rows. The target is `log1p(Sales)`; predictions are transformed back with `expm1`.

This is one forward holdout, not a leaderboard result. `reports/metrics-calendar.json`, `reports/metrics-history.json`, and the paired validation prediction CSVs contain the source values for the README comparison.
