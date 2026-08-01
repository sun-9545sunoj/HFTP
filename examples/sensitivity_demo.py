"""
Execution Latency and Parameter Sensitivity Sweeper Demo.

Runs backtest simulations across execution latency delays (delta t), queue priority decay parameters,
and inventory penalties to analyze performance degradation under real-world HFT conditions.
"""

import numpy as np
import polars as pl

from src.backtest.sensitivity import SensitivityAnalyzer


def run_sensitivity_demo():
    print("=========================================================")
    print("      HFT Execution Latency & Sensitivity Sweeper Demo    ")
    print("=========================================================\n")

    # Generate synthetic 500-tick LOB dataset
    n_ticks = 500
    np.random.seed(42)
    base_mid = 100.0 + np.cumsum(np.random.normal(0, 0.05, n_ticks))

    dict_data = {"timestamp": list(range(n_ticks))}
    for i in range(1, 11):
        dict_data[f"ask_price_{i}"] = base_mid + (i * 0.05)
        dict_data[f"ask_size_{i}"] = np.random.randint(10, 100, n_ticks).astype(float)
        dict_data[f"bid_price_{i}"] = base_mid - (i * 0.05)
        dict_data[f"bid_size_{i}"] = np.random.randint(10, 100, n_ticks).astype(float)
    
    dict_data["mid_price"] = base_mid

    df_lob = pl.DataFrame(dict_data)
    signals = np.random.choice([-1, 0, 1], size=n_ticks, p=[0.15, 0.70, 0.15])

    print("Running sensitivity grid sweep across:")
    print(" - Execution Latency Latencies: [1, 2, 5, 10, 20] ticks")
    print(" - Queue Priority Decay Rates:  [0.1, 0.2, 0.5]")
    print(" - Inventory Risk Penalties:    [0.0, 1e-6, 1e-5]\n")

    analyzer = SensitivityAnalyzer(
        latencies=[1, 2, 5, 10, 20],
        queue_decay_rates=[0.1, 0.2, 0.5],
        inventory_penalties=[0.0, 1e-6, 1e-5],
    )

    df_results = analyzer.run_sweep(df_lob, signals, order_size=1.0)

    print("---------------------------------------------------------")
    print("               SENSITIVITY COMPARISON TABLE               ")
    print("---------------------------------------------------------")
    print(df_results.select([
        "latency_ticks",
        "queue_decay_rate",
        "inventory_penalty",
        "net_pnl",
        "fill_rate_pct",
        "sharpe_ratio",
        "win_rate_pct",
    ]).head(10))
    print("\n=========================================================\n")


if __name__ == "__main__":
    run_sensitivity_demo()
