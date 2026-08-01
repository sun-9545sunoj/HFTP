"""
Exploratory Data Analysis (EDA) Script for Real LOB Datasets (FI-2010 & LOBSTER).

Parses real LOB dataset files, computes summary statistics, mid-price volatility,
directional label distributions, bid-ask spread profiles, and order flow imbalances.
"""

import argparse
from pathlib import Path
import polars as pl

from src.data.loader import LOBLoader
from src.data.targets import compute_mid_price, compute_smoothed_returns, assign_directional_labels
from src.features.microstructure import extract_all_microstructure_features


def analyze_lob_dataset(
    file_path: str,
    dataset_format: str = "fi2010",
    orderbook_path: str = None,
    message_path: str = None,
    k: int = 5,
    num_levels: int = 10,
):
    print("=========================================================")
    print("       Real LOB Dataset Exploratory Analysis (EDA)       ")
    print("=========================================================\n")

    # 1. Load Dataset
    print(f"[1/4] Loading {dataset_format.upper()} dataset from disk...")
    if dataset_format == "fi2010":
        df = LOBLoader.load_fi2010(file_path, num_levels=num_levels)
    elif dataset_format == "lobster":
        df = LOBLoader.load_lobster(
            orderbook_path=orderbook_path or file_path,
            message_path=message_path,
            num_levels=num_levels,
        )
    else:
        raise ValueError(f"Unknown format '{dataset_format}'. Choose 'fi2010' or 'lobster'.")

    print(f"-> Ticks Loaded: {len(df):,} rows across {df.width} columns.\n")

    # 2. Mid-Price & Target Calculation
    print(f"[2/4] Computing Mid-Price, {k}-Tick Returns & Directional Labels...")
    df_mid = compute_mid_price(df)
    df_ret = compute_smoothed_returns(df_mid, k=k)
    df_labeled = assign_directional_labels(
        df_ret,
        return_col=f"return_{k}",
        use_dynamic_volatility=True,
    )

    # Label counts
    label_dist = df_labeled["label"].value_counts().to_dict()
    print("-> Directional Label Distribution y_t in {-1: DOWN, 0: STATIONARY, 1: UP}:")
    for lbl, count in zip(label_dist["label"], label_dist["count"]):
        if lbl is not None:
            pct = (count / len(df_labeled)) * 100.0
            name = "UP (+1)" if lbl == 1 else ("DOWN (-1)" if lbl == -1 else "STATIONARY (0)")
            print(f"   - {name}: {count:,} ticks ({pct:.2f}%)")
    print()

    # 3. Microstructure Feature Analysis
    print("[3/4] Extracting Microstructure Features (OFI, Micro-Price, Spreads, VWAP)...")
    df_features = extract_all_microstructure_features(df_labeled, num_levels=num_levels)

    print("\n---------------------------------------------------------")
    print("               MICROSTRUCTURE SUMMARY STATS               ")
    print("---------------------------------------------------------")
    stats = df_features.select([
        "mid_price",
        "micro_price",
        "bid_ask_spread",
        "relative_spread",
        "queue_imbalance_10",
        "ofi_total",
        "vwap_dev_50",
    ]).describe()
    print(stats)
    print("=========================================================\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze real FI-2010 or LOBSTER dataset file.")
    parser.add_argument("--format", type=str, default="fi2010", choices=["fi2010", "lobster"], help="Dataset format.")
    parser.add_argument("--file_path", type=str, default=None, help="Path to FI-2010 txt file or LOBSTER orderbook csv.")
    parser.add_argument("--message_path", type=str, default=None, help="Optional LOBSTER message file path.")
    parser.add_argument("--k", type=int, default=5, help="Return lookahead horizon k ticks.")
    args = parser.parse_args()

    if args.file_path is None:
        print("Note: No --file_path provided. Pass a real FI-2010 or LOBSTER file path to analyze real data.")
        print("Example Usage:")
        print("  python analyze_dataset.py --format fi2010 --file_path data/Train_Dst_NoAmbush_5.txt")
        print("  python analyze_dataset.py --format lobster --file_path data/AAPL_2025-01-01_orderbook_10.csv --message_path data/AAPL_2025-01-01_message_10.csv")
    else:
        analyze_lob_dataset(
            file_path=args.file_path,
            dataset_format=args.format,
            message_path=args.message_path,
            k=args.k,
        )
