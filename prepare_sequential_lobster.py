"""
Prepare a LEAK-FREE multi-stock LOBSTER dataset for sequence models.

Replaces prepare_balanced_lobster.py for all new experiments. The old script
shuffled rows before train.py built sliding windows, so training "sequences"
were random unrelated ticks and validation leaked through near-duplicate
neighbouring ticks (val F1 ~0.77 vs chance-level out-of-sample F1).

This script keeps every stock in strict temporal order:
1. Per-stock causal rolling Z-score normalization (W=1000), warm-up dropped.
2. Per-stock chronological split: train 70% / val 15% / test 15%.
3. Purge the last max(K_LIST) rows of train and val so no label looks into
   the next split.
4. Smoothed mid returns for several horizons k (return_{k}). For each stock
   and k, 3-class labels (label_k{k}) use 33/66% return quantiles fit on that
   stock's TRAIN rows only. Large-tick stocks (MSFT, INTC) rarely move, so
   both quantiles can be 0 and STATIONARY ("no move") dominates; that is real.
   `label` is a copy of label_k{DEFAULT_K}.
5. NO shuffling and NO undersampling. Windows are built later inside each
   (stock, split) segment; class imbalance is handled by loss weights.

Output: src/data/LOBSTER_sequential.parquet with extra columns `stock`, `split`.
"""

import json
import os

import polars as pl

from src.data.loader import LOBLoader
from src.data.targets import compute_mid_price, compute_smoothed_returns
from src.features.microstructure import extract_all_microstructure_features

STOCKS = ["AAPL", "AMZN", "GOOG", "MSFT", "INTC"]
DATA_DIR = "src/data/"
K_LIST = [5, 10, 20, 50, 100]
DEFAULT_K = 20
ROLLING_WINDOW = 1000
MIN_SAMPLES = 100
CLIP = 10.0
TRAIN_RATIO, VAL_RATIO = 0.70, 0.15
Q_DOWN, Q_UP = 0.33, 0.66
PURGE = max(K_LIST)

RETURN_COLS = [f"return_{k}" for k in K_LIST]
EXCLUDE_FROM_NORM = {"timestamp", "alpha_t", "mid_price", *RETURN_COLS}


def rolling_zscore(df: pl.DataFrame, cols) -> pl.DataFrame:
    exprs = []
    for col in cols:
        mean = pl.col(col).rolling_mean(window_size=ROLLING_WINDOW, min_samples=MIN_SAMPLES)
        std = pl.col(col).rolling_std(window_size=ROLLING_WINDOW, min_samples=MIN_SAMPLES)
        exprs.append(
            pl.when(std > 1e-12)
            .then(((pl.col(col) - mean) / std).clip(-CLIP, CLIP))
            .otherwise(pl.lit(0.0))
            .alias(col)
        )
    return df.with_columns(exprs)


def process_stock(stock: str) -> tuple[pl.DataFrame, dict]:
    print(f"\n{'=' * 60}\n  Processing {stock}...\n{'=' * 60}")
    df = LOBLoader.load_lobster(
        os.path.join(DATA_DIR, f"LOBSTER_{stock}_orderbook.csv"),
        message_path=os.path.join(DATA_DIR, f"LOBSTER_{stock}_message.csv"),
        num_levels=10,
    )
    print(f"  -> Raw ticks: {len(df):,}")

    df = compute_mid_price(df)
    for k in K_LIST:
        df = compute_smoothed_returns(df, k=k)
    df = extract_all_microstructure_features(df, num_levels=10)

    feature_cols = [c for c in df.columns if c not in EXCLUDE_FROM_NORM]
    df = rolling_zscore(df, feature_cols)

    # Drop normalization warm-up and tail rows lacking max(K) future ticks
    df = df.slice(ROLLING_WINDOW).drop_nulls(subset=RETURN_COLS).filter(pl.col(f"return_{PURGE}").is_not_nan())
    n = len(df)
    n_train, n_val = int(n * TRAIN_RATIO), int(n * VAL_RATIO)

    meta = {}
    for k in K_LIST:
        col = f"return_{k}"
        train_ret = df[col][:n_train]
        q_down, q_up = train_ret.quantile(Q_DOWN), train_ret.quantile(Q_UP)
        df = df.with_columns(
            pl.when(pl.col(col) > q_up).then(1)
            .when(pl.col(col) < q_down).then(-1)
            .otherwise(0)
            .cast(pl.Int8)
            .alias(f"label_k{k}")
        )
        meta[f"k{k}"] = {"q_down": q_down, "q_up": q_up}
    df = df.with_columns(pl.col(f"label_k{DEFAULT_K}").alias("label"))

    split = ["train"] * n_train + ["val"] * n_val + ["test"] * (n - n_train - n_val)
    df = df.with_columns(pl.Series("split", split), pl.lit(stock).alias("stock"))

    # Purge rows whose labels are built from the next split's prices
    purge = list(range(n_train - PURGE, n_train)) + list(range(n_train + n_val - PURGE, n_train + n_val))
    df = df.with_row_index("_i").filter(~pl.col("_i").is_in(purge)).drop("_i")

    for s in ["train", "val", "test"]:
        part = df.filter(pl.col("split") == s)
        shares = [
            f"k{k}:" + "/".join(f"{part[f'label_k{k}'].eq(c).mean():.2f}" for c in (-1, 0, 1))
            for k in K_LIST
        ]
        print(f"  -> {s:5s} {len(part):>8,} | D/S/U  " + "  ".join(shares))

    meta["n_rows"] = len(df)
    return df, meta


def main():
    parts, meta = [], {}
    for stock in STOCKS:
        df, meta[stock] = process_stock(stock)
        parts.append(df)

    final = pl.concat(parts, how="vertical")
    out_path = os.path.join(DATA_DIR, "LOBSTER_sequential.parquet")
    LOBLoader.save_parquet(final, out_path)

    config = {
        "k_list": K_LIST, "default_k": DEFAULT_K, "normalization": "rolling_zscore",
        "window": ROLLING_WINDOW, "min_samples": MIN_SAMPLES, "clip": CLIP,
        "train_ratio": TRAIN_RATIO, "val_ratio": VAL_RATIO, "purge_rows": PURGE,
        "label_quantiles_train_fit": [Q_DOWN, Q_UP], "stocks": meta,
    }
    with open(os.path.join(DATA_DIR, "sequential_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n  Saved {len(final):,} rows to {out_path}")


if __name__ == "__main__":
    main()
