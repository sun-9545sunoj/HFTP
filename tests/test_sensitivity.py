"""
Unit tests for Latency Sensitivity & Backtest Parameter Sweeper.
"""

import numpy as np
import polars as pl
import pytest

from src.backtest.sensitivity import SensitivityAnalyzer


def test_sensitivity_sweep():
    """Verify SensitivityAnalyzer grid execution and output table schema."""
    n_ticks = 20
    ask_prices = [100.1 + (i * 0.01) for i in range(n_ticks)]
    bid_prices = [100.0 + (i * 0.01) for i in range(n_ticks)]
    mid_prices = [(a + b) / 2.0 for a, b in zip(ask_prices, bid_prices)]

    df_lob = pl.DataFrame({
        "timestamp": list(range(n_ticks)),
        "ask_price_1": ask_prices,
        "ask_size_1": [10.0] * n_ticks,
        "bid_price_1": bid_prices,
        "bid_size_1": [15.0] * n_ticks,
        "mid_price": mid_prices,
    })

    signals = np.zeros(n_ticks, dtype=int)
    signals[2] = 1
    signals[10] = -1

    analyzer = SensitivityAnalyzer(
        latencies=[1, 2],
        queue_decay_rates=[0.1],
        inventory_penalties=[0.0],
    )

    df_res = analyzer.run_sweep(df_lob, signals)

    assert len(df_res) == 2  # 2 latency scenarios
    assert "latency_ticks" in df_res.columns
    assert "net_pnl" in df_res.columns
    assert "fill_rate_pct" in df_res.columns
