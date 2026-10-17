"""
Non-sequence baselines on the leak-free sequential LOBSTER parquet.

- majority:  always predict the most frequent TRAIN class.
- logreg:    multinomial logistic regression on the LAST tick's features only
             (the window end), class-weighted. If a deep sequence model cannot
             beat this, the 100-tick history adds nothing.

Evaluated on the same window end indices as the deep models, so numbers are
directly comparable. Results are appended to reports/results.jsonl.

Usage:
    .venv/bin/python baseline_linear.py --k 20 --stocks AAPL,AMZN,GOOG
"""

import argparse
import time

import numpy as np
import torch
import torch.nn as nn

from evaluate_sequential import append_result, classification_metrics, load_split, print_metrics
from src.data.dataset import segment_end_indices


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", default="src/data/LOBSTER_sequential.parquet")
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--stocks", default=None)
    parser.add_argument("--seq_len", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=4096)
    args = parser.parse_args()
    torch.manual_seed(0)

    df, feature_cols, features, labels = load_split(args.parquet, args.stocks, args.k)
    ends = segment_end_indices(df, args.seq_len)
    stock_arr = df["stock"].to_numpy()
    X = {s: features[e] for s, e in ends.items()}
    y = {s: labels[e] for s, e in ends.items()}

    counts = np.bincount(y["train"].numpy(), minlength=3).astype(np.float32)
    weights = torch.tensor(counts.sum() / (3 * np.maximum(counts, 1)), dtype=torch.float32)

    def report(name: str, split: str, y_pred: np.ndarray) -> dict:
        y_true = y[split].numpy()
        stock_of = stock_arr[ends[split]]
        overall = classification_metrics(y_true, y_pred)
        per_stock = {s: classification_metrics(y_true[stock_of == s], y_pred[stock_of == s]) for s in np.unique(stock_of)}
        return {"overall": overall, "per_stock": per_stock}

    # Majority-class baseline
    majority = int(counts.argmax())
    res = report("majority", "test", np.full(len(y["test"]), majority))
    print_metrics("MAJORITY (test)", res["overall"])
    append_result({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "model": "majority", "split": "test",
                   "k": args.k, "stocks": args.stocks or "all", **res})

    # Logistic regression on last-tick features
    model = nn.Linear(X["train"].shape[1], 3)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    best_f1, best_state = -1.0, None
    n = len(X["train"])
    for epoch in range(args.epochs):
        perm = torch.randperm(n)
        for i in range(0, n, args.batch_size):
            idx = perm[i : i + args.batch_size]
            opt.zero_grad()
            loss_fn(model(X["train"][idx]), y["train"][idx]).backward()
            opt.step()
        with torch.no_grad():
            val_pred = model(X["val"]).argmax(1).numpy()
        val_f1 = classification_metrics(y["val"].numpy(), val_pred)["f1_macro"]
        print(f"  logreg epoch {epoch}: val macro F1 = {val_f1:.4f}")
        if val_f1 > best_f1:
            best_f1, best_state = val_f1, {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    with torch.no_grad():
        test_pred = model(X["test"]).argmax(1).numpy()
    res = report("logreg", "test", test_pred)
    print_metrics("LOGREG last-tick (test)", res["overall"])
    for s, m in res["per_stock"].items():
        print_metrics(s, m)
    append_result({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "model": "logreg_last_tick", "split": "test",
                   "k": args.k, "stocks": args.stocks or "all", "best_val_f1": best_f1, **res})


if __name__ == "__main__":
    main()
