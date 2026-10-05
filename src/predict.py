"""Load the saved model and forecast a CSV of future store-day rows."""
from __future__ import annotations
import argparse
import lzma
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from train import add_query_history, base_features, read_data

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "models" / "rossmann_sales.cbm.xz"


def unpack_model(compressed_path: Path) -> Path:
    cache = compressed_path.parent / ".cache" / "rossmann_sales.cbm"
    cache.parent.mkdir(parents=True, exist_ok=True)
    if not cache.exists() or cache.stat().st_mtime < compressed_path.stat().st_mtime:
        with lzma.open(compressed_path, "rb") as source, cache.open("wb") as target:
            shutil.copyfileobj(source, target)
    return cache


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="CSV containing future Store/Date/promotion/open/holiday columns")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=ROOT / "predictions.csv")
    args = parser.parse_args()

    history, store = read_data(args.data_dir)
    future = pd.read_csv(args.input, low_memory=False, parse_dates=["Date"])
    features = base_features(future, store)
    features = add_query_history(features, future, history)
    model = CatBoostRegressor()
    model.load_model(str(unpack_model(args.model)), format="cbm")
    prediction = np.maximum(0, np.expm1(model.predict(features)))
    if "Open" in future:
        prediction[future.Open.fillna(0).to_numpy() == 0] = 0

    result = future[[c for c in ["Id", "Store", "Date"] if c in future]].copy()
    result["SalesPrediction"] = prediction
    result.to_csv(args.output, index=False)
    print(f"Wrote {len(result):,} store-day forecasts to {args.output}")


if __name__ == "__main__":
    main()
