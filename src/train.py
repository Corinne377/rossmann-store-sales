"""Train and validate a forward-looking Rossmann sales model."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw"
CATEGORICAL = ["Store", "DayOfWeek", "StateHoliday", "StoreType", "Assortment", "PromoInterval", "month", "year", "week", "day", "day_of_year", "week_of_year"]


def read_data(data_dir: Path):
    train = pd.read_csv(data_dir / "train.csv", low_memory=False)
    test = pd.read_csv(data_dir / "test.csv", low_memory=False)
    store = pd.read_csv(data_dir / "store.csv", low_memory=False)
    return train, test, store


def make_features(frame: pd.DataFrame, store: pd.DataFrame) -> pd.DataFrame:
    x = frame.merge(store, on="Store", how="left", validate="many_to_one")
    d = pd.to_datetime(x["Date"])
    x["year"] = d.dt.year.astype(str)
    x["month"] = d.dt.month.astype(str)
    x["week"] = d.dt.isocalendar().week.astype(str)
    x["day"] = d.dt.day.astype(str)
    x["day_of_year"] = d.dt.dayofyear.astype(str)
    x["week_of_year"] = d.dt.isocalendar().week.astype(str)
    x["weekend"] = (d.dt.dayofweek >= 5).astype("int8")
    x["quarter"] = d.dt.quarter.astype("int8")
    x["days_from_start"] = (d - pd.Timestamp("2013-01-01")).dt.days
    x["competition_open_months"] = ((d.dt.year - x["CompetitionOpenSinceYear"].fillna(d.dt.year)) * 12 + d.dt.month - x["CompetitionOpenSinceMonth"].fillna(d.dt.month)).clip(lower=0)
    current_week = d.dt.isocalendar().week.astype(float)
    promo_year = x["Promo2SinceYear"].fillna(9999).astype(float)
    promo_week = x["Promo2SinceWeek"].fillna(9999).astype(float)
    started = (d.dt.year > promo_year) | ((d.dt.year == promo_year) & (current_week >= promo_week))
    x["promo2_active"] = ((x["Promo2"].fillna(0) == 1) & started).astype("int8")
    months = d.dt.strftime("%b")
    x["promo2_month_active"] = [int(bool(interval) and month in str(interval).split(",")) for interval, month in zip(x["PromoInterval"], months)]
    x["StateHoliday"] = x["StateHoliday"].fillna("0").astype(str).replace({"0.0": "0"})
    x["CompetitionDistance"] = x["CompetitionDistance"].fillna(x["CompetitionDistance"].median())
    x["CompetitionDistance_log"] = np.log1p(x["CompetitionDistance"])
    for c in ["StoreType", "Assortment", "PromoInterval"]:
        x[c] = x[c].fillna("missing").astype(str)
    # Customers are deliberately excluded: unavailable for the test set.
    cols = [c for c in CATEGORICAL + ["Promo", "Open", "SchoolHoliday", "Promo2", "weekend", "quarter", "days_from_start", "CompetitionDistance_log", "CompetitionOpenSinceMonth", "CompetitionOpenSinceYear", "Promo2SinceWeek", "Promo2SinceYear", "promo2_active", "promo2_month_active"] if c in x]
    return x[cols]


def rmspe(y, pred):
    y, pred = np.asarray(y), np.asarray(pred)
    mask = y > 0
    return float(np.sqrt(np.mean(((y[mask] - pred[mask]) / y[mask]) ** 2)))


def fit_model(X, y, cat_cols, iterations=2500, verbose=250, eval_set=None):
    model = CatBoostRegressor(iterations=iterations, depth=9, learning_rate=0.055, loss_function="RMSE", eval_metric="RMSE", l2_leaf_reg=5, random_seed=42, verbose=verbose, allow_writing_files=False, thread_count=-1)
    model.fit(X, np.log1p(y), cat_features=cat_cols, eval_set=eval_set, early_stopping_rounds=150 if eval_set is not None else None)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=DATA)
    ap.add_argument("--validation", action="store_true", help="Evaluate on the final 42 training days, then fit all rows and write a submission")
    ap.add_argument("--iterations", type=int, default=450)
    ap.add_argument("--output", type=Path, default=ROOT / "submission.csv")
    args = ap.parse_args()
    train, test, store = read_data(args.data_dir)
    train["Date"] = pd.to_datetime(train["Date"])
    test["Date"] = pd.to_datetime(test["Date"])
    if args.validation:
        cutoff = train.Date.max() - pd.Timedelta(days=41)
        fit = train[train.Date < cutoff].copy()
        valid = train[train.Date >= cutoff].copy()
        X_fit, X_valid = make_features(fit, store), make_features(valid, store)
        cats = [X_fit.columns.get_loc(c) for c in CATEGORICAL if c in X_fit]
        model = fit_model(X_fit, fit.Sales, cats, args.iterations, eval_set=(X_valid, np.log1p(valid.Sales)))
        pred = np.maximum(0, np.expm1(model.predict(X_valid)))
        score = rmspe(valid.Sales, pred)
        print(f"Forward 42-day validation: {score:.6f} RMSPE (zero-sales rows excluded)")
        args.iterations = max(100, model.best_iteration_ + 1)
        print(f"Using {args.iterations} iterations for the full-data fit")
    # Sales are zero whenever the store is closed; retain the competition test row order.
    X = make_features(train, store)
    Xt = make_features(test, store)
    cats = [X.columns.get_loc(c) for c in CATEGORICAL if c in X]
    model = fit_model(X, train.Sales, cats, args.iterations)
    pred = np.maximum(0, np.expm1(model.predict(Xt)))
    pred[test.Open.fillna(0).to_numpy() == 0] = 0
    result = pd.DataFrame({"Id": test.Id, "Sales": np.rint(pred).astype("int64")})
    result.to_csv(args.output, index=False)
    print(f"Wrote {len(result):,} predictions to {args.output}")

if __name__ == "__main__":
    main()
