"""
Prepare balanced & normalized multi-stock LOBSTER dataset for generalized training.

CRITICAL FIX (Oct 2026): features are now normalized with a causal
rolling-window Z-score (W=1000 ticks). Previously a global per-stock Z-score
was used, which caused a ~2σ covariate shift between train (early 80%) and
test (late 20%) due to intra-stock price drift.

Anti-overfitting measures:
1. Rolling-window Z-score normalization (locally stationary → no drift shift)
2. Quantile-based labeling for balanced classes
3. Undersampling majority class to achieve perfect 33/33/33 balance
4. SHUFFLE after concatenation so train/val splits have all stocks & classes mixed
5. 80/20 chronological split per stock (no future leakage)
"""

import json
import os
import numpy as np
import polars as pl
from src.data.loader import LOBLoader
from src.data.targets import compute_mid_price, compute_smoothed_returns, assign_directional_labels
from src.features.microstructure import extract_all_microstructure_features

stocks = ["AAPL", "AMZN", "GOOG", "MSFT", "INTC"]
data_dir = "src/data/"
k = 5

# Columns that should NOT be z-score normalized (metadata / targets)
EXCLUDE_FROM_NORM = {"timestamp", "label", "return_5", "alpha_t", "mid_price"}

train_dfs = []
test_dfs = []

for stock in stocks:
    print(f"\n{'='*60}")
    print(f"  Processing {stock}...")
    print(f"{'='*60}")
    ob_path = os.path.join(data_dir, f"LOBSTER_{stock}_orderbook.csv")
    msg_path = os.path.join(data_dir, f"LOBSTER_{stock}_message.csv")

    # 1. Load raw LOBSTER data
    df = LOBLoader.load_lobster(ob_path, message_path=msg_path, num_levels=10)
    print(f"  -> Raw ticks loaded: {len(df):,}")

    # 2. Compute returns & balanced labels (quantile-based)
    df = compute_mid_price(df)
    df = compute_smoothed_returns(df, k=k)
    df = assign_directional_labels(df, return_col=f"return_{k}", alpha=-1.0, use_dynamic_volatility=False)

    # 3. Extract Microstructure Features
    df = extract_all_microstructure_features(df, num_levels=10)
    df = df.drop_nulls(subset=["label"])
    print(f"  -> After feature extraction: {len(df):,} ticks")

    # 4. Rolling-Window Z-Score Normalization
    #    Instead of global Z-score (which fails due to price drift), normalize
    #    each tick relative to a trailing window of W ticks. This makes the
    #    features locally stationary — train and test will have the same
    #    distribution regardless of absolute price levels.
    ROLLING_WINDOW = 1000  # ~1000 ticks lookback for rolling mean/std
    MIN_SAMPLES = 100
    CLIP = 10.0  # clip extreme z-scores (near-zero rolling std)

    feature_cols = [c for c in df.columns if c not in EXCLUDE_FROM_NORM]
    print(f"  -> Rolling Z-score normalization (W={ROLLING_WINDOW}) on {len(feature_cols)} features...")

    norm_exprs = []
    for col in feature_cols:
        rolling_mean = pl.col(col).rolling_mean(window_size=ROLLING_WINDOW, min_samples=MIN_SAMPLES)
        rolling_std = pl.col(col).rolling_std(window_size=ROLLING_WINDOW, min_samples=MIN_SAMPLES)
        # Z-score: (x - rolling_mean) / rolling_std, with safeguard against zero std
        norm_expr = (
            pl.when(rolling_std > 1e-12)
            .then(((pl.col(col) - rolling_mean) / rolling_std).clip(-CLIP, CLIP))
            .otherwise(pl.lit(0.0))
        ).alias(col)
        norm_exprs.append(norm_expr)

    df = df.with_columns(norm_exprs)

    # Drop initial rows where rolling window is not yet filled
    df = df.slice(ROLLING_WINDOW, len(df) - ROLLING_WINDOW)
    print(f"  -> After dropping warm-up rows: {len(df):,} ticks")

    # 5. Chronological 80/20 split (AFTER normalization — both splits are
    #    now locally normalized so no covariate shift is possible)
    split_idx = int(len(df) * 0.8)
    train_df = df.slice(0, split_idx)
    test_df = df.slice(split_idx, len(df) - split_idx)
    print(f"  -> Chronological split: Train {len(train_df):,} | Test {len(test_df):,}")

    # Verify: train and test should now have similar means (~0)
    tr_mean = train_df["ask_price_1"].mean()
    te_mean = test_df["ask_price_1"].mean()
    print(f"  -> Verification: ask_price_1 mean — Train: {tr_mean:.4f} | Test: {te_mean:.4f}")

    train_dfs.append(train_df)
    test_dfs.append(test_df)

# 6. Concatenate all stocks
print(f"\n{'='*60}")
print("  Concatenating all stocks...")
final_train = pl.concat(train_dfs)
final_test = pl.concat(test_dfs)

# 7. Shuffle the training data
print("  Shuffling training data for mixed stock/class distribution...")
final_train = final_train.sample(fraction=1.0, shuffle=True, seed=42)

# 8. Undersample majority class for perfect balance
print("  Undersampling for perfect class balance...")
label_counts = final_train["label"].value_counts()
min_count = label_counts["count"].min()
print(f"  -> Minimum class count: {min_count:,}")

balanced_parts = []
for label_val in [-1, 0, 1]:
    subset = final_train.filter(pl.col("label") == label_val)
    if len(subset) > min_count:
        indices = np.random.default_rng(42).choice(len(subset), size=min_count, replace=False)
        subset = subset[indices.tolist()]
    balanced_parts.append(subset)

final_train_balanced = pl.concat(balanced_parts)

# 9. Shuffle AGAIN after undersampling
final_train_balanced = final_train_balanced.sample(fraction=1.0, shuffle=True, seed=123)

# 10. Save to Parquet
print("  Saving to Parquet...")
LOBLoader.save_parquet(final_train_balanced, os.path.join(data_dir, "LOBSTER_balanced_train.parquet"))
LOBLoader.save_parquet(final_test, os.path.join(data_dir, "LOBSTER_balanced_test.parquet"))

# 11. Save normalization config for reproducibility
config_path = os.path.join(data_dir, "normalization_config.json")
with open(config_path, "w") as f:
    json.dump({"method": "rolling_zscore", "window": ROLLING_WINDOW,
               "min_samples": MIN_SAMPLES, "clip": CLIP}, f, indent=2)
print(f"  -> Normalization config saved to {config_path}")

# 12. Final report with covariate shift verification
print(f"\n{'='*60}")
print(f"  FINAL DATASET SUMMARY")
print(f"{'='*60}")
print(f"  Train Set Size: {len(final_train_balanced):,} ticks (SHUFFLED)")
print(f"  Test Set Size:  {len(final_test):,} ticks")
print(f"\n  Train Class Distribution (Perfectly Balanced):")
print(final_train_balanced["label"].value_counts().sort("label"))
print(f"\n  Test Class Distribution:")
print(final_test["label"].value_counts().sort("label"))

# 13. Covariate Shift Verification
print(f"\n  {'='*60}")
print(f"  COVARIATE SHIFT CHECK (should be < 0.3 std devs)")
print(f"  {'='*60}")
check_cols = [c for c in final_test.columns if c not in EXCLUDE_FROM_NORM]
all_ok = True
max_shift = 0.0
for col in check_cols:
    tr_m = final_train_balanced[col].mean()
    te_m = final_test[col].mean()
    tr_s = final_train_balanced[col].std()
    shift = abs(tr_m - te_m) / max(tr_s, 0.001)
    max_shift = max(max_shift, shift)
    status = "✓ OK" if shift < 0.3 else "⚠️ SHIFT"
    if shift >= 0.1:
        print(f"  {col:15s} | Train mean: {tr_m:+.4f} | Test mean: {te_m:+.4f} | Shift: {shift:.3f}σ {status}")
    if shift >= 0.3:
        all_ok = False

print(f"  Max shift across {len(check_cols)} features: {max_shift:.3f}σ")
if all_ok:
    print(f"\n  ✅ ALL FEATURES ALIGNED — No covariate shift detected!")
else:
    print(f"\n  ⚠️  Some features still show shift — investigate further.")

# Sanity check: first 100 rows have mixed labels
first_100_labels = final_train_balanced["label"].head(100).value_counts().sort("label")
print(f"\n  Sanity Check — First 100 rows label mix:")
print(first_100_labels)
print(f"{'='*60}")
