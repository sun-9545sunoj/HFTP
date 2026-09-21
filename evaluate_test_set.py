"""
Out-of-Sample Model Evaluation & Execution Backtest Script.

Evaluates trained DeepLOB / Transformer models on unseen test datasets (FI2010_test.csv).
Computes classification metrics (Accuracy, Macro F1, Precision/Recall, Confusion Matrix)
and runs the Latency & Queue-Aware Microstructure Engine with signal confidence threshold filtering.
"""

import argparse
from pathlib import Path
import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
import torchmetrics

from src.data.dataset import LOBSequenceDataset, map_indices_to_signals
from src.data.loader import LOBLoader
from src.data.targets import assign_directional_labels, compute_mid_price, compute_smoothed_returns
from src.features.microstructure import extract_all_microstructure_features
from src.models.deeplob import DeepLOB, SpatialTemporalTransformerLOB
from src.models.lightning_module import DeepLOBModule
from src.backtest.engine import MicrostructureBacktester


def find_latest_checkpoint(checkpoint_dir: str = "checkpoints") -> Path:
    """Find the best/latest saved model checkpoint file."""
    ckpt_path = Path(checkpoint_dir)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint directory '{checkpoint_dir}' does not exist.")

    ckpt_files = list(ckpt_path.glob("*.ckpt"))
    if not ckpt_files:
        raise FileNotFoundError(f"No model checkpoint (.ckpt) files found in '{checkpoint_dir}'. Train model first.")

    latest_ckpt = max(ckpt_files, key=lambda f: f.stat().st_mtime)
    return latest_ckpt


def evaluate_on_test_set(
    train_file_path: str,
    test_file_path: str,
    dataset_type: str = "fi2010",
    k: int = 5,
    seq_len: int = 100,
    num_levels: int = 10,
    batch_size: int = 128,
    max_position: float = 5.0,
    min_prob: float = 0.50,
):
    print("=========================================================")
    print("      Enhanced DeepLOB Out-of-Sample Evaluation         ")
    print("=========================================================\n")

    # 1. Load Trained Checkpoint
    ckpt_file = find_latest_checkpoint()
    print(f"[1/4] Loading trained model checkpoint: {ckpt_file.name}...")
    
    module = DeepLOBModule.load_from_checkpoint(ckpt_file)
    model = module.model
    model.eval()
    
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(device)
    print(f"-> Model loaded on device: {device}\n")

    # Detect in_features count from model weight shape
    in_features_count = module.hparams.in_features
    print(f"-> Model expected feature count: {in_features_count} features.")

    # 2. Ingest Test Dataset
    print(f"[2/4] Ingesting Out-of-Sample Test Dataset: {Path(test_file_path).name}...")
    
    if dataset_type == "parquet":
        df_clean = pl.read_parquet(test_file_path)
    else:
        test_df_raw = LOBLoader.load_fi2010(test_file_path, num_levels=num_levels)
        print(f"-> Test Ticks Loaded: {len(test_df_raw):,} rows.")

        df_mid = compute_mid_price(test_df_raw)
        df_ret = compute_smoothed_returns(df_mid, k=k)
        df_labeled = assign_directional_labels(
            df_ret,
            return_col=f"return_{k}",
            use_dynamic_volatility=True,
        )

        df_features = extract_all_microstructure_features(df_labeled, num_levels=num_levels)
        df_clean = df_features.drop_nulls(subset=["label"])

    if in_features_count == 40 and dataset_type != "parquet":
        feature_cols = LOBLoader.get_canonical_column_names(num_levels=num_levels)
    else:
        feature_cols = [
            c for c in df_clean.columns 
            if c not in ["timestamp", "label", f"return_{k}", "alpha_t"] 
            and not df_clean[c].is_null().all()
        ][:in_features_count]

    feature_arr = df_clean.select(feature_cols).to_numpy()
    label_arr = df_clean["label"].to_numpy()

    # Create Test Dataset
    test_dataset = LOBSequenceDataset(feature_arr, label_arr, seq_len=seq_len)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    print(f"-> Test Sequence Samples: {len(test_dataset):,} sliding windows (T={seq_len}).\n")

    # 3. Model Inference & Evaluation Metrics
    print("[3/4] Running Model Inference & Probability Confidence Extraction...")
    all_preds = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)
            logits = model(x_batch)
            probs = F.softmax(logits, dim=1).cpu()
            preds = torch.argmax(probs, dim=1)
            
            all_preds.append(preds)
            all_probs.append(probs)
            all_targets.append(y_batch)

    preds_tensor = torch.cat(all_preds)
    probs_tensor = torch.cat(all_probs)
    targets_tensor = torch.cat(all_targets)

    # Apply Signal Confidence Thresholding (min_prob filter)
    max_probs, raw_preds = torch.max(probs_tensor, dim=1)
    confident_mask = max_probs >= min_prob
    
    # Filter predictions: if confidence < min_prob, set prediction to 1 (STATIONARY/HOLD)
    filtered_preds = torch.where(confident_mask, raw_preds, torch.tensor(1))

    # Compute Classification Metrics
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

    print("---------------------------------------------------------")
    print("           OUT-OF-SAMPLE CLASSIFICATION METRICS          ")
    print("---------------------------------------------------------")
    print(f" Signal Confidence Threshold:   >= {min_prob * 100:.1f}%")
    print(f" Out-of-Sample Accuracy:        {acc:.2f}%")
    print(f" Out-of-Sample Macro F1-Score:  {f1_macro:.4f}\n")
    print(" Class Metrics:")
    print(f"  - DOWN (-1):        Precision: {prec[0]:.4f} | Recall: {rec[0]:.4f}")
    print(f"  - STATIONARY (0):   Precision: {prec[1]:.4f} | Recall: {rec[1]:.4f}")
    print(f"  - UP (+1):          Precision: {prec[2]:.4f} | Recall: {rec[2]:.4f}\n")
    print(" 3x3 Confusion Matrix (Rows: Actual, Cols: Predicted):")
    print(conf_mat)
    print("---------------------------------------------------------\n")

    # 4. Latency & Queue-Aware Backtest Evaluation
    print(f"[4/4] Running Backtest with Risk Control (q_max = {max_position}, min_prob = {min_prob})...")
    signals = map_indices_to_signals(filtered_preds.numpy())
    
    padded_signals = np.zeros(len(df_clean), dtype=int)
    padded_signals[seq_len - 1 :] = signals

    backtester = MicrostructureBacktester(
        latency_ticks=2,
        maker_fee_per_share=-0.002,   # $0.002 rebate per share (limit order entry)
        taker_fee_per_share=0.003,    # $0.003 fee per share (market order exit)
        inventory_penalty_factor=1e-5,
        exit_on_neutral=False,    # Let alpha develop; don't close on every neutral tick
        max_hold_ticks=50,        # Force-close after 50 ticks to limit inventory risk
        flatten_on_reversal=True, # Close entire position on opposite signal
    )
    metrics, df_backtest = backtester.run_backtest(
        df_clean,
        padded_signals,
        order_size=1.0,
        max_position=max_position,
    )

    print("---------------------------------------------------------")
    print("        OUT-OF-SAMPLE BACKTEST PERFORMANCE SUMMARY       ")
    print("---------------------------------------------------------")
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
    print("=========================================================\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate DeepLOB model out-of-sample dataset.")
    parser.add_argument("--train_file", type=str, default="/Users/nagasunojr/Documents/naga/HFTPM_LOBAS/src/data/FI2010_train 2.csv")
    parser.add_argument("--test_file", type=str, default="/Users/nagasunojr/Documents/naga/HFTPM_LOBAS/src/data/FI2010_test.csv")
    parser.add_argument("--dataset_type", type=str, default="fi2010", choices=["fi2010", "parquet"])
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--seq_len", type=int, default=100)
    parser.add_argument("--max_position", type=float, default=5.0)
    parser.add_argument("--min_prob", type=float, default=0.50, help="Signal confidence threshold.")
    args = parser.parse_args()

    evaluate_on_test_set(
        train_file_path=args.train_file,
        test_file_path=args.test_file,
        dataset_type=args.dataset_type,
        k=args.k,
        seq_len=args.seq_len,
        max_position=args.max_position,
        min_prob=args.min_prob,
    )
