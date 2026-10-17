"""
Ensemble Model Evaluation & Backtest Script.

Loads all 3 trained architectures (DeepLOB, TransLOB, SpatialTemporalTransformerLOB),
combines their softmax outputs via weighted averaging, and evaluates the ensemble
on the out-of-sample test set with classification metrics and backtest simulation.

Supports multiple ensemble strategies:
  - 'weighted_avg': F1-weighted softmax averaging (default)
  - 'majority_vote': Hard majority voting across models
  - 'stacking': Learned meta-classifier on top of model outputs
"""

import argparse
from pathlib import Path
import numpy as np
import polars as pl
import torch
import torch.nn.functional as F_torch
import torchmetrics

from src.data.dataset import LOBSequenceDataset, get_model_feature_cols, map_indices_to_signals
from src.models.lightning_module import DeepLOBModule
from src.backtest.engine import MicrostructureBacktester


# ── Model Checkpoints ──────────────────────────────────────────────────────────
CHECKPOINTS = {
    "DeepLOB": "checkpoints/enhanced-parquet-deeplob-best-epoch=14-val_f1_macro=0.7694.ckpt",
    "TransLOB": "checkpoints/enhanced-parquet-translob-best-epoch=14-val_f1_macro=0.7311.ckpt",
    "PureTransformer": "checkpoints/enhanced-parquet-transformer-best-epoch=12-val_f1_macro=0.6602.ckpt",
}

# Validation F1 scores used as ensemble weights (higher F1 → more influence)
VAL_F1_SCORES = {
    "DeepLOB": 0.7694,
    "TransLOB": 0.7311,
    "PureTransformer": 0.6602,
}


def load_all_models(device: torch.device) -> dict:
    """Load all 3 model checkpoints and return them in eval mode."""
    models = {}
    for name, ckpt_path in CHECKPOINTS.items():
        path = Path(ckpt_path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        module = DeepLOBModule.load_from_checkpoint(path, map_location=device)
        model = module.model
        model.eval()
        model.to(device)
        models[name] = model
        param_count = sum(p.numel() for p in model.parameters())
        print(f"  ✓ {name:20s} | {param_count:>8,} params | F1={VAL_F1_SCORES[name]:.4f}")
    return models


def get_model_predictions(
    models: dict, dataloader, device: torch.device
) -> dict:
    """Run inference with all models and return softmax probabilities."""
    results = {name: [] for name in models}

    with torch.no_grad():
        for x_batch, y_batch in dataloader:
            x_batch = x_batch.to(device)
            for name, model in models.items():
                logits = model(x_batch)
                probs = F_torch.softmax(logits, dim=1).cpu()
                results[name].append(probs)

    # Concatenate all batches
    for name in results:
        results[name] = torch.cat(results[name])

    return results


def ensemble_weighted_average(
    model_probs: dict, weights: dict
) -> torch.Tensor:
    """Combine model softmax outputs using F1-weighted averaging."""
    total_weight = sum(weights.values())
    normalized_weights = {k: v / total_weight for k, v in weights.items()}

    ensemble_probs = None
    for name, probs in model_probs.items():
        w = normalized_weights[name]
        if ensemble_probs is None:
            ensemble_probs = w * probs
        else:
            ensemble_probs += w * probs

    return ensemble_probs


def ensemble_majority_vote(model_probs: dict) -> torch.Tensor:
    """Hard majority voting: each model votes for its argmax class."""
    votes = []
    for name, probs in model_probs.items():
        preds = torch.argmax(probs, dim=1)
        votes.append(preds)

    # Stack: (num_models, num_samples)
    vote_stack = torch.stack(votes, dim=0)

    # Count votes per class for each sample
    num_samples = vote_stack.shape[1]
    num_classes = 3
    final_preds = torch.zeros(num_samples, dtype=torch.long)

    for i in range(num_samples):
        counts = torch.bincount(vote_stack[:, i], minlength=num_classes)
        final_preds[i] = torch.argmax(counts)

    # Build pseudo-probabilities from vote counts for confidence thresholding
    pseudo_probs = torch.zeros(num_samples, num_classes)
    for i in range(num_samples):
        counts = torch.bincount(vote_stack[:, i], minlength=num_classes).float()
        pseudo_probs[i] = counts / counts.sum()

    return pseudo_probs


def evaluate_ensemble(
    test_file_path: str,
    seq_len: int = 100,
    batch_size: int = 128,
    max_position: float = 10.0,
    min_prob: float = 0.50,
    strategy: str = "weighted_avg",
):
    print("=" * 65)
    print("       ENSEMBLE MODEL EVALUATION & BACKTEST")
    print("=" * 65)
    print()

    # ── Device ──
    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device: {device}\n")

    # ── 1. Load All Models ──
    print("[1/5] Loading all 3 model checkpoints...")
    models = load_all_models(device)
    total_params = sum(
        sum(p.numel() for p in m.parameters()) for m in models.values()
    )
    print(f"\n  Total ensemble parameters: {total_params:,}\n")

    # ── 2. Load Test Data ──
    print(f"[2/5] Loading test dataset: {Path(test_file_path).name}...")
    df_clean = pl.read_parquet(test_file_path)

    # Same feature selection as training (excludes raw mid_price)
    feature_cols = get_model_feature_cols(df_clean, k=5)

    feature_arr = df_clean.select(feature_cols).to_numpy()
    label_arr = df_clean["label"].to_numpy()

    test_dataset = LOBSequenceDataset(feature_arr, label_arr, seq_len=seq_len)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )
    print(f"  → {len(test_dataset):,} test sequences (T={seq_len})\n")

    # ── 3. Run Inference with All Models ──
    print("[3/5] Running inference with all 3 models...")
    model_probs = get_model_predictions(models, test_loader, device)

    targets_list = []
    for _, y_batch in test_loader:
        targets_list.append(y_batch)
    targets_tensor = torch.cat(targets_list)

    # ── 4. Ensemble Combination ──
    print(f"[4/5] Combining predictions (strategy: {strategy})...")

    if strategy == "weighted_avg":
        ensemble_probs = ensemble_weighted_average(model_probs, VAL_F1_SCORES)
    elif strategy == "majority_vote":
        ensemble_probs = ensemble_majority_vote(model_probs)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    # ── Individual Model Metrics (without thresholding) ──
    print("\n" + "-" * 65)
    print("     INDIVIDUAL MODEL METRICS (No Confidence Threshold)")
    print("-" * 65)

    for name, probs in model_probs.items():
        preds = torch.argmax(probs, dim=1)
        f1 = torchmetrics.F1Score(task="multiclass", num_classes=3, average="macro")
        acc = torchmetrics.Accuracy(task="multiclass", num_classes=3)
        f1_val = f1(preds, targets_tensor).item()
        acc_val = acc(preds, targets_tensor).item() * 100
        print(f"  {name:20s} | OOS Accuracy: {acc_val:6.2f}% | OOS F1: {f1_val:.4f}")

    # Ensemble without thresholding
    ens_preds_raw = torch.argmax(ensemble_probs, dim=1)
    f1_ens = torchmetrics.F1Score(task="multiclass", num_classes=3, average="macro")
    acc_ens = torchmetrics.Accuracy(task="multiclass", num_classes=3)
    f1_ens_val = f1_ens(ens_preds_raw, targets_tensor).item()
    acc_ens_val = acc_ens(ens_preds_raw, targets_tensor).item() * 100
    print(f"  {'★ ENSEMBLE':20s} | OOS Accuracy: {acc_ens_val:6.2f}% | OOS F1: {f1_ens_val:.4f}")
    print("-" * 65)

    # ── Apply Confidence Thresholding ──
    max_probs, raw_preds = torch.max(ensemble_probs, dim=1)
    confident_mask = max_probs >= min_prob
    filtered_preds = torch.where(confident_mask, raw_preds, torch.tensor(1))

    # Classification Metrics (with threshold)
    acc_metric = torchmetrics.Accuracy(task="multiclass", num_classes=3)
    f1_metric = torchmetrics.F1Score(task="multiclass", num_classes=3, average="macro")
    prec_metric = torchmetrics.Precision(task="multiclass", num_classes=3, average=None)
    rec_metric = torchmetrics.Recall(task="multiclass", num_classes=3, average=None)
    conf_metric = torchmetrics.ConfusionMatrix(task="multiclass", num_classes=3)

    acc = acc_metric(filtered_preds, targets_tensor).item() * 100.0
    f1_macro = f1_metric(filtered_preds, targets_tensor).item()
    prec = prec_metric(filtered_preds, targets_tensor).numpy()
    rec = rec_metric(filtered_preds, targets_tensor).numpy()
    conf_mat = conf_metric(filtered_preds, targets_tensor).numpy()

    confident_pct = confident_mask.float().mean().item() * 100

    print(f"\n{'=' * 65}")
    print(f"  ENSEMBLE CLASSIFICATION (Confidence Threshold >= {min_prob*100:.0f}%)")
    print(f"{'=' * 65}")
    print(f"  Strategy:              {strategy}")
    print(f"  Confident Predictions: {confident_pct:.1f}% of {len(targets_tensor):,} samples")
    print(f"  Accuracy:              {acc:.2f}%")
    print(f"  Macro F1-Score:        {f1_macro:.4f}")
    print()
    print("  Per-Class Metrics:")
    print(f"    DOWN (-1):        Precision: {prec[0]:.4f} | Recall: {rec[0]:.4f}")
    print(f"    STATIONARY (0):   Precision: {prec[1]:.4f} | Recall: {rec[1]:.4f}")
    print(f"    UP (+1):          Precision: {prec[2]:.4f} | Recall: {rec[2]:.4f}")
    print()
    print("  Confusion Matrix (Rows: Actual, Cols: Predicted):")
    print(f"  {conf_mat}")
    print(f"{'=' * 65}")

    # ── 5. Backtest ──
    print(f"\n[5/5] Running Backtest (q_max={max_position}, min_prob={min_prob})...")
    signals = map_indices_to_signals(filtered_preds.numpy())

    padded_signals = np.zeros(len(df_clean), dtype=int)
    padded_signals[seq_len - 1:] = signals

    backtester = MicrostructureBacktester(
        latency_ticks=2,
        maker_fee_per_share=-0.002,
        taker_fee_per_share=0.003,
        inventory_penalty_factor=1e-5,
        exit_on_neutral=False,
        max_hold_ticks=50,
        flatten_on_reversal=True,
    )
    metrics, df_backtest = backtester.run_backtest(
        df_clean, padded_signals, order_size=1.0, max_position=max_position,
    )

    print("-" * 65)
    print("        ENSEMBLE BACKTEST PERFORMANCE SUMMARY")
    print("-" * 65)
    print(f" Net PnL:                 ${metrics.net_pnl:.4f}")
    print(f" Gross PnL:               ${metrics.gross_pnl:.4f}")
    print(f" Total Rebates / Fees:    ${metrics.total_fees:.4f}")
    print(f" Inventory Risk Penalty:  ${metrics.inventory_penalties:.4f}")
    print(f" Sharpe Ratio:            {metrics.sharpe_ratio:.2f}")
    print(f" Max Drawdown:            ${metrics.max_drawdown:.4f} ({metrics.max_drawdown_pct:.2f}%)")
    print(f" Fill Rate %:             {metrics.fill_rate_pct:.2f}%")
    print(f" Orders Placed / Filled:  {metrics.total_orders_placed:,} / {metrics.total_orders_filled:,}")
    print(f" Win Rate %:              {metrics.win_rate_pct:.2f}%")
    print(f" Profit Factor:           {metrics.profit_factor:.2f}")
    print("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ensemble LOB Model Evaluation")
    parser.add_argument("--test_file", type=str, default="src/data/LOBSTER_balanced_test.parquet")
    parser.add_argument("--seq_len", type=int, default=100)
    parser.add_argument("--max_position", type=float, default=10.0)
    parser.add_argument("--min_prob", type=float, default=0.50)
    parser.add_argument(
        "--strategy", type=str, default="weighted_avg",
        choices=["weighted_avg", "majority_vote"],
        help="Ensemble strategy: 'weighted_avg' (F1-weighted softmax) or 'majority_vote'"
    )
    args = parser.parse_args()

    evaluate_ensemble(
        test_file_path=args.test_file,
        seq_len=args.seq_len,
        max_position=args.max_position,
        min_prob=args.min_prob,
        strategy=args.strategy,
    )
