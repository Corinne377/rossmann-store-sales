"""Train the leakage-safe Rossmann sales model and save validation artifacts."""
from __future__ import annotations

import argparse
import json
import lzma
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data" / "raw"
DEFAULT_MODEL = ROOT / "models" / "rossmann_sales.cbm"
CATEGORICAL = [
    "Store", "DayOfWeek", "StateHoliday", "StoreType", "Assortment",
    "PromoInterval", "month", "year", "week", "day", "day_of_year", "week_of_year",
]
HISTORY_KEYS = ["Store", "DayOfWeek", "Promo"]
HISTORY_COLUMNS = ["hist_store_dow_log_sales", "hist_store_dow_promo_log_sales", "hist_store_dow_promo_count"]


def read_data(data_dir: Path):
    train = pd.read_csv(data_dir / "train.csv", low_memory=False, parse_dates=["Date"])
    store = pd.read_csv(data_dir / "store.csv", low_memory=False)
    return train, store


def base_features(frame: pd.DataFrame, store: pd.DataFrame) -> pd.DataFrame:
    x = frame.merge(store, on="Store", how="left", validate="many_to_one")
    d = pd.to_datetime(x["Date"])
    x["year"] = d.dt.year.astype(str)
    x["month"] = d.dt.month.astype(str)
    x["week"] = d.dt.isocalendar().week.astype(str)
    x["week_of_year"] = x["week"]
    x["day"] = d.dt.day.astype(str)
    x["day_of_year"] = d.dt.dayofyear.astype(str)
    x["weekend"] = (d.dt.dayofweek >= 5).astype("int8")
    x["quarter"] = d.dt.quarter.astype("int8")
    x["days_from_start"] = (d - pd.Timestamp("2013-01-01")).dt.days
    x["competition_open_months"] = (
        (d.dt.year - x["CompetitionOpenSinceYear"].fillna(d.dt.year)) * 12
        + d.dt.month - x["CompetitionOpenSinceMonth"].fillna(d.dt.month)
    ).clip(lower=0)
    week = d.dt.isocalendar().week.astype(float)
    since_year = x["Promo2SinceYear"].fillna(9999).astype(float)
    since_week = x["Promo2SinceWeek"].fillna(9999).astype(float)
    started = (d.dt.year > since_year) | ((d.dt.year == since_year) & (week >= since_week))
    x["promo2_active"] = ((x["Promo2"].fillna(0) == 1) & started).astype("int8")
    months = d.dt.strftime("%b")
    x["promo2_month_active"] = [
        int(month in str(interval).split(","))
        for interval, month in zip(x["PromoInterval"], months)
    ]
    x["StateHoliday"] = x["StateHoliday"].fillna("0").astype(str).replace({"0.0": "0"})
    x["CompetitionDistance"] = x["CompetitionDistance"].fillna(x["CompetitionDistance"].median())
    x["CompetitionDistance_log"] = np.log1p(x["CompetitionDistance"])
    for col in ["StoreType", "Assortment", "PromoInterval"]:
        x[col] = x[col].fillna("missing").astype(str)
    # Customers are excluded because they are not known when making forecasts.
    columns = CATEGORICAL + [
        "Promo", "Open", "SchoolHoliday", "Promo2", "weekend", "quarter",
        "days_from_start", "CompetitionDistance_log",
        "CompetitionOpenSinceMonth", "CompetitionOpenSinceYear", "Promo2SinceWeek",
        "Promo2SinceYear", "promo2_active", "promo2_month_active",
    ]
    return x[columns].copy()


def add_past_history(features: pd.DataFrame, labeled_rows: pd.DataFrame) -> pd.DataFrame:
    """Attach past-only target averages for rows in a chronological training set."""
    z = labeled_rows[["Store", "DayOfWeek", "Promo", "Date", "Sales"]].copy()
    z["_row_index"] = z.index
    z["_log_sales"] = np.log1p(z["Sales"].clip(lower=0))
    z = z.sort_values(["Date", "Store"], kind="stable")
    grouped = z.groupby(HISTORY_KEYS, sort=False, observed=True)["_log_sales"]
    prior_sum = grouped.cumsum() - z["_log_sales"]
    prior_count = grouped.cumcount()
    prior_mean = prior_sum / prior_count.replace(0, np.nan)
    z["hist_store_dow_promo_log_sales"] = prior_mean.to_numpy()
    z["hist_store_dow_promo_count"] = prior_count.to_numpy()

    # A weekday-only fallback stabilizes sparse store/promo combinations.
    z2 = z.sort_values(["Date", "Store"], kind="stable")
    grp = z2.groupby(["Store", "DayOfWeek"], sort=False, observed=True)["_log_sales"]
    z2["hist_store_dow_log_sales"] = (
        (grp.cumsum() - z2["_log_sales"])
        / grp.cumcount().replace(0, np.nan)
    ).to_numpy()
    hist = z2.set_index("_row_index")[HISTORY_COLUMNS].reindex(features.index)
    return pd.concat([features, hist], axis=1)


def add_query_history(features: pd.DataFrame, query_rows: pd.DataFrame, history_rows: pd.DataFrame) -> pd.DataFrame:
    """Use only labeled history available before the query forecast period."""
    h = history_rows[[*HISTORY_KEYS, "Sales"]].copy()
    h["_log_sales"] = np.log1p(h["Sales"].clip(lower=0))
    fine = h.groupby(HISTORY_KEYS, observed=True)["_log_sales"].agg(["mean", "size"])
    fine.columns = ["hist_store_dow_promo_log_sales", "hist_store_dow_promo_count"]
    broad = h.groupby(["Store", "DayOfWeek"], observed=True)["_log_sales"].mean()
    q = query_rows[HISTORY_KEYS].copy()
    q["_row_index"] = q.index
    fine_values = fine.reindex(pd.MultiIndex.from_frame(q[HISTORY_KEYS]))
    broad_values = broad.reindex(pd.MultiIndex.from_frame(q[["Store", "DayOfWeek"]]))
    fine_values.index = q.index
    broad_values.index = q.index
    hist = pd.DataFrame(index=q.index)
    hist["hist_store_dow_promo_log_sales"] = fine_values["hist_store_dow_promo_log_sales"]
    hist["hist_store_dow_promo_count"] = fine_values["hist_store_dow_promo_count"]
    hist["hist_store_dow_log_sales"] = broad_values.to_numpy()
    return pd.concat([features, hist.reindex(features.index)[HISTORY_COLUMNS]], axis=1)


def rmspe(y_true, prediction) -> float:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(prediction, dtype=float)
    mask = y > 0
    return float(np.sqrt(np.mean(((y[mask] - p[mask]) / y[mask]) ** 2)))


def make_model(iterations: int, depth: int, verbose=250) -> CatBoostRegressor:
    return CatBoostRegressor(
        iterations=iterations,
        depth=depth,
        learning_rate=0.055,
        loss_function="RMSE",
        eval_metric="RMSE",
        l2_leaf_reg=5,
        random_seed=42,
        verbose=verbose,
        allow_writing_files=False,
        thread_count=-1,
    )


def fit(X, y, iterations, categorical_indices, depth, eval_set=None):
    model = make_model(iterations, depth)
    model.fit(
        X, np.log1p(y), cat_features=categorical_indices,
        eval_set=eval_set,
        early_stopping_rounds=150 if eval_set is not None else None,
    )
    return model


def compress_model(path: Path) -> Path:
    compressed = Path(str(path) + ".xz")
    with path.open("rb") as source, lzma.open(compressed, "wb", preset=9) as target:
        while chunk := source.read(8 * 1024 * 1024):
            target.write(chunk)
    return compressed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--iterations", type=int, default=900)
    parser.add_argument("--holdout-days", type=int, default=42)
    parser.add_argument("--depth", type=int, default=9)
    parser.add_argument("--history", action="store_true", help="Add leakage-safe historical sales features")
    parser.add_argument("--validate-only", action="store_true", help="Score the holdout without fitting a full-data model")
    parser.add_argument("--fit-only", action="store_true", help="Fit a full-data model using --iterations trees, without rerunning validation")
    parser.add_argument("--compress-model", action="store_true", help="Also write a lossless .xz model suitable for a regular GitHub repository")
    args = parser.parse_args()

    train, store = read_data(args.data_dir)
    # Keep source CSV row order for consistent CatBoost ordered boosting.
    if args.fit_only:
        X_all = base_features(train, store)
        if args.history:
            X_all = add_past_history(X_all, train)
        categories = [X_all.columns.get_loc(c) for c in CATEGORICAL]
        final_model = fit(X_all, train["Sales"], args.iterations, categories, args.depth)
        args.model_path.parent.mkdir(parents=True, exist_ok=True)
        final_model.save_model(str(args.model_path))
        if args.compress_model:
            print(f"Compressed model: {compress_model(args.model_path)}")
        importance = pd.DataFrame({"feature": X_all.columns, "importance": final_model.get_feature_importance()})
        importance.sort_values("importance", ascending=False).to_csv(ROOT / "reports" / "feature-importance.csv", index=False)
        print(f"Saved full-data model: {args.model_path}")
        return
    cutoff = train["Date"].max() - pd.Timedelta(days=args.holdout_days - 1)
    fit_rows = train[train["Date"] < cutoff].copy().reset_index(drop=True)
    valid_rows = train[train["Date"] >= cutoff].copy().reset_index(drop=True)

    X_fit = base_features(fit_rows, store)
    X_valid = base_features(valid_rows, store)
    if args.history:
        X_fit = add_past_history(X_fit, fit_rows)
        X_valid = add_query_history(X_valid, valid_rows, fit_rows)
    categorical_indices = [X_fit.columns.get_loc(c) for c in CATEGORICAL]
    validation_model = fit(
        X_fit, fit_rows["Sales"], args.iterations, categorical_indices, args.depth,
        eval_set=(X_valid, np.log1p(valid_rows["Sales"])),
    )
    validation_prediction = np.maximum(0, np.expm1(validation_model.predict(X_valid)))
    score = rmspe(valid_rows["Sales"], validation_prediction)
    best_trees = max(1, validation_model.best_iteration_ + 1)

    report_dir = ROOT / "reports"
    report_dir.mkdir(exist_ok=True)
    pred_out = pd.DataFrame({
        "Date": valid_rows["Date"].dt.strftime("%Y-%m-%d"),
        "Store": valid_rows["Store"],
        "ActualSales": valid_rows["Sales"],
        "PredictedSales": validation_prediction,
        "Open": valid_rows["Open"],
    })
    feature_name = "history" if args.history else "calendar"
    pred_out.to_csv(report_dir / f"validation-{feature_name}-predictions.csv", index=False)
    metrics = {
        "model": f"CatBoost with {feature_name} features",
        "validation_rmspe": score,
        "validation_start": str(valid_rows["Date"].min().date()),
        "validation_end": str(valid_rows["Date"].max().date()),
        "validation_days": args.holdout_days,
        "validation_rows": int(len(valid_rows)),
        "validation_scored_rows": int((valid_rows["Sales"] > 0).sum()),
        "best_iteration_zero_based": int(validation_model.best_iteration_),
        "selected_trees": int(best_trees),
        "depth": args.depth,
        "learning_rate": 0.055,
        "seed": 42,
    }
    (report_dir / f"metrics-{feature_name}.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"Holdout RMSPE: {score:.6f}; selected {best_trees} trees")
    if args.validate_only:
        return

    # Refit on all labeled data, with past-only features for each training row.
    X_all = base_features(train, store)
    if args.history:
        X_all = add_past_history(X_all, train)
    final_model = fit(X_all, train["Sales"], best_trees, categorical_indices, args.depth)
    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    final_model.save_model(str(args.model_path))
    if args.compress_model:
        print(f"Compressed model: {compress_model(args.model_path)}")
    importance = pd.DataFrame({
        "feature": X_all.columns,
        "importance": final_model.get_feature_importance(),
    }).sort_values("importance", ascending=False)
    importance.to_csv(report_dir / "feature-importance.csv", index=False)
    metrics["model_path"] = str(args.model_path.relative_to(ROOT))
    (report_dir / f"metrics-{feature_name}.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"Saved full-data model: {args.model_path}")


if __name__ == "__main__":
    main()
