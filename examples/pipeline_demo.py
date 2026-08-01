"""
End-to-End Pipeline Demonstration Script for High-Frequency Microstructure Research.

Demonstrates:
1. Synthetic FI-2010 / LOBSTER dataset generation & parsing.
2. Target labeling with smoothed k-tick forward returns & dynamic volatility thresholds.
3. Market microstructure feature engineering (OFI, Micro-Price, Queue Imbalance, Spread/Depth decay).
4. DeepLOB model instantiation & forward pass.
5. Latency & Queue-Aware Microstructure Execution Backtest simulation.
"""

from pathlib import Path
import numpy as np
import polars as pl
import torch

from src.data.loader import LOBLoader
from src.data.targets import compute_mid_price, compute_smoothed_returns, assign_directional_labels
from src.features.microstructure import extract_all_microstructure_features
from src.models.deeplob import DeepLOB
from src.backtest.engine import MicrostructureBacktester


def run_pipeline_demo():
    print("=========================================================")
    print("  High-Frequency LOB Mid-Price & Backtest Framework Demo  ")
    print("=========================================================\n")

    # 1. Generate Synthetic 10-level LOB ticks
    n_ticks = 200
    num_levels = 10
    print(f"[Phase 1] Generating synthetic {n_ticks}-tick LOB dataset across {num_levels} levels...")

    dict_data = {"timestamp": list(range(n_ticks))}
    np.random.seed(42)
    base_mid = 100.0 + np.cumsum(np.random.normal(0, 0.05, n_ticks))

    for i in range(1, num_levels + 1):
        dict_data[f"ask_price_{i}"] = base_mid + (i * 0.05)
        dict_data[f"ask_size_{i}"] = np.random.randint(10, 100, n_ticks).astype(float)
        dict_data[f"bid_price_{i}"] = base_mid - (i * 0.05)
        dict_data[f"bid_size_{i}"] = np.random.randint(10, 100, n_ticks).astype(float)

    raw_df = pl.DataFrame(dict_data)
    
    # Compute mid price & target labels
    df_mid = compute_mid_price(raw_df)
    k = 5
    df_ret = compute_smoothed_returns(df_mid, k=k)
    df_labeled = assign_directional_labels(
        df_ret,
        return_col=f"return_{k}",
        use_dynamic_volatility=True,
        volatility_window=20,
        volatility_multiplier=0.5,
    )
    print(f"-> Mid-Price & Target Labeling complete. Columns added: ['mid_price', 'return_{k}', 'alpha_t', 'label']")
    print(f"-> Label Distribution: {df_labeled['label'].value_counts().to_dict()}\n")

    # 2. Extract Microstructure Features
    print("[Phase 2] Computing Market Microstructure Features (OFI, Micro-Price, Deep Queue Imbalance, Spread Decay)...")
    df_features = extract_all_microstructure_features(df_labeled, num_levels=num_levels)
    feature_cols = ["micro_price", "queue_imbalance_10", "ofi_total", "bid_ask_spread", "relative_spread", "depth_decay_ratio"]
    print(f"-> Extracted {len(feature_cols)} microstructure features cleanly.\n")

    # 3. Model Architecture Instantiation
    print("[Phase 3] Instantiating DeepLOB Model (Zhang et al., 2019)...")
    model = DeepLOB(in_features=40, seq_len=100, num_classes=3)
    dummy_input = torch.randn(8, 100, 40)
    logits = model(dummy_input)
    print(f"-> Input Shape: {dummy_input.shape} -> Output Logits Shape: {logits.shape}\n")

    # 4. Latency & Queue-Aware Microstructure Backtest Execution
    print("[Phase 4] Running Microstructure Backtester (Latency=2 ticks, Queue Priority Decay=0.2, Maker Rebate=-1 bps)...")
    signals = np.random.choice([-1, 0, 1], size=n_ticks, p=[0.15, 0.70, 0.15])
    
    backtester = MicrostructureBacktester(
        latency_ticks=2,
        maker_fee=-0.0001,
        taker_fee=0.0004,
        inventory_penalty_factor=1e-5,
    )
    metrics, df_backtest = backtester.run_backtest(df_features, signals, order_size=1.0)

    print("---------------------------------------------------------")
    print("               BACKTEST PERFORMANCE SUMMARY               ")
    print("---------------------------------------------------------")
    print(f" Total Net PnL:             ${metrics.net_pnl:.4f}")
    print(f" Gross PnL:                 ${metrics.gross_pnl:.4f}")
    print(f" Total Fees / Rebates:      ${metrics.total_fees:.4f}")
    print(f" Inventory Risk Penalty:    ${metrics.inventory_penalties:.4f}")
    print(f" Sharpe Ratio:              {metrics.sharpe_ratio:.2f}")
    print(f" Max Drawdown:              ${metrics.max_drawdown:.4f} ({metrics.max_drawdown_pct:.2f}%)")
    print(f" Fill Rate %:               {metrics.fill_rate_pct:.2f}%")
    print(f" Orders Placed / Filled:    {metrics.total_orders_placed} / {metrics.total_orders_filled}")
    print(f" Win Rate %:                {metrics.win_rate_pct:.2f}%")
    print(f" Profit Factor:             {metrics.profit_factor:.2f}")
    print("=========================================================\n")


if __name__ == "__main__":
    run_pipeline_demo()
