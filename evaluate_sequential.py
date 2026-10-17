"""
Leak-free evaluation on the sequential LOBSTER parquet.

Reports raw argmax classification metrics (no confidence threshold) overall
and per stock, for an explicitly given checkpoint, and appends a JSON line to
reports/results.jsonl so runs can be tabulated for the paper.

Usage:
    .venv/bin/python evaluate_sequential.py --checkpoint checkpoints/<file>.ckpt --k 20 --stocks AAPL,AMZN,GOOG
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader

from src.data.dataset import LOBWindowDataset, get_model_feature_cols, map_labels_to_indices, segment_end_indices
from src.models.lightning_module import DeepLOBModule

CLASS_NAMES = ["DOWN", "STAT", "UP"]


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 3) -> dict:
    """Accuracy, macro F1, per-class precision/recall and confusion matrix (rows = actual)."""
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(cm, (y_true, y_pred), 1)
    tp = np.diag(cm).astype(np.float64)
    precision = tp / np.maximum(cm.sum(axis=0), 1)
    recall = tp / np.maximum(cm.sum(axis=1), 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    return {
        "n": int(len(y_true)),
        "accuracy": float(tp.sum() / max(len(y_true), 1)),
        "f1_macro": float(f1.mean()),
        "precision": precision.round(4).tolist(),
        "recall": recall.round(4).tolist(),
        "confusion": cm.tolist(),
    }


def print_metrics(title: str, m: dict) -> None:
    print(f"\n  {title}  (n={m['n']:,})")
    print(f"    accuracy={m['accuracy']:.4f}  macro F1={m['f1_macro']:.4f}")
    for i, name in enumerate(CLASS_NAMES):
        print(f"    {name:4s} precision={m['precision'][i]:.3f} recall={m['recall'][i]:.3f}")


def load_split(parquet: str, stocks: str | None, k: int):
    df = pl.read_parquet(parquet)
    if stocks:
        df = df.filter(pl.col("stock").is_in(stocks.split(",")))
    feature_cols = get_model_feature_cols(df, k=k)
    features = torch.tensor(df.select(feature_cols).to_numpy(), dtype=torch.float32)
    labels = torch.tensor(map_labels_to_indices(df[f"label_k{k}"].to_numpy()), dtype=torch.long)
    return df, feature_cols, features, labels


def append_result(record: dict, path: str = "reports/results.jsonl") -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--parquet", default="src/data/LOBSTER_sequential.parquet")
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--stocks", default=None)
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--seq_len", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=1024)
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    module = DeepLOBModule.load_from_checkpoint(args.checkpoint, map_location=device)
    module.eval().to(device)

    df, feature_cols, features, labels = load_split(args.parquet, args.stocks, args.k)
    if module.hparams.in_features != len(feature_cols):
        raise ValueError(f"Checkpoint expects {module.hparams.in_features} features, data has {len(feature_cols)}.")

    ends = segment_end_indices(df, args.seq_len)[args.split]
    loader = DataLoader(LOBWindowDataset(features, labels, ends, args.seq_len), batch_size=args.batch_size)

    preds = []
    with torch.no_grad():
        for x, _ in loader:
            preds.append(module(x.to(device)).argmax(dim=1).cpu())
    y_pred = torch.cat(preds).numpy()
    y_true = labels[ends].numpy()
    stock_of = df["stock"].to_numpy()[ends]

    overall = classification_metrics(y_true, y_pred)
    print(f"\n{'=' * 60}\n  {Path(args.checkpoint).name}\n  split={args.split} k={args.k} stocks={args.stocks or 'all'}\n{'=' * 60}")
    print_metrics("OVERALL", overall)
    per_stock = {}
    for s in np.unique(stock_of):
        mask = stock_of == s
        per_stock[s] = classification_metrics(y_true[mask], y_pred[mask])
        print_metrics(s, per_stock[s])

    append_result({
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": module.hparams.model_type,
        "pooling": module.hparams.get("pooling", "mean"),
        "checkpoint": Path(args.checkpoint).name,
        "split": args.split, "k": args.k, "stocks": args.stocks or "all",
        "overall": overall, "per_stock": per_stock,
    })


if __name__ == "__main__":
    main()
