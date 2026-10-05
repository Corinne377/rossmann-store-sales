# Rossmann Store Sales

A reproducible CatBoost baseline for the Kaggle Rossmann Store Sales forecasting competition. It predicts the six-week test horizon using store, date, promotion, holiday, and competition features. Customer counts are excluded because they are not available in the test set.

## Evaluation

Kaggle scores Root Mean Square Percentage Error (RMSPE), excluding rows with zero actual sales. The script reports this metric on a chronological 42-day holdout that ends on the final date in the training data, matching the six-week competition horizon. This is a local estimate, not a leaderboard guarantee.

## Setup (macOS)

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

CatBoost runs on CPU by default and uses all available CPU threads. This works on both Intel and Apple Silicon MacBook Pros; no GPU setup is required.

## Data

Download `train.csv`, `test.csv`, `store.csv`, and `sample_submission.csv` from the [Kaggle competition data page](https://www.kaggle.com/competitions/rossmann-store-sales/data), accept the competition rules, and place the files in `data/raw/`. Data files are ignored by Git because the dataset is subject to Kaggle's competition rules.

## Train, validate, and create submission

```bash
python src/train.py --validation
```

The command evaluates on a forward 42-day holdout, then trains on all labeled rows and writes `submission.csv` in the required `Id,Sales` format. The included run used 450 maximum iterations with early stopping and selected 418 trees. It scored **0.175309 RMSPE** on the holdout. For a faster experiment, use `--iterations 250`; increase the cap to explore longer training. Upload `submission.csv` on Kaggle.

## Project layout

- `src/train.py` — feature engineering, forward validation, CatBoost fit, and submission generation
- `data/raw/README.md` — data placement instructions
- `requirements.txt` — Python dependencies

## Notes

- All validation folds are time ordered; random row splits would leak future patterns across the forecast boundary.
- Closed test stores are assigned zero sales as required by the data semantics.
- Keep Kaggle credentials out of this repository. The project creates a local submission; upload it through Kaggle after reviewing the file.
