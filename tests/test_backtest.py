"""
Unit tests for Latency & Queue-Aware Microstructure Backtester Engine.
"""

import numpy as np
import polars as pl
import pytest

from src.backtest.engine import MicrostructureBacktester, OrderSide, OrderStatus
from src.backtest.metrics import compute_performance_metrics


@pytest.fixture
def sample_lob_stream() -> pl.DataFrame:
    """Generate 20 ticks of synthetic LOB data."""
    n_ticks = 20
    ask_prices = [100.1 + (i * 0.01) for i in range(n_ticks)]
    bid_prices = [100.0 + (i * 0.01) for i in range(n_ticks)]
    ask_sizes = [10.0] * n_ticks
    bid_sizes = [15.0] * n_ticks
    mid_prices = [(a + b) / 2.0 for a, b in zip(ask_prices, bid_prices)]

    return pl.DataFrame({
        "timestamp": list(range(n_ticks)),
        "ask_price_1": ask_prices,
        "ask_size_1": ask_sizes,
        "bid_price_1": bid_prices,
        "bid_size_1": bid_sizes,
        "mid_price": mid_prices,
    })


def test_backtest_execution(sample_lob_stream: pl.DataFrame):
    """Test full backtest simulation run."""
    n_ticks = len(sample_lob_stream)
    signals = np.zeros(n_ticks, dtype=int)
    
    # Place BUY signal at tick 2, SELL signal at tick 10
    signals[2] = 1
    signals[10] = -1

    backtester = MicrostructureBacktester(
        latency_ticks=2,
        maker_fee=-0.0001,  # Rebate
        queue_decay_rate=0.5,
    )

    metrics, df_res = backtester.run_backtest(sample_lob_stream, signals, order_size=1.0)

    assert metrics.total_orders_placed == 2
    assert metrics.total_orders_filled >= 1
    assert "equity" in df_res.columns
    assert "position" in df_res.columns
    assert len(df_res) == n_ticks


def test_latency_delay(sample_lob_stream: pl.DataFrame):
    """Verify latency delay prevents order activation before latency_ticks."""
    signals = np.zeros(len(sample_lob_stream), dtype=int)
    signals[1] = 1  # Signal generated at tick 1

    latency = 3
    backtester = MicrostructureBacktester(latency_ticks=latency, queue_decay_rate=0.0)

    metrics, df_res = backtester.run_backtest(sample_lob_stream, signals)
    # Order generated at tick 1 with latency 3 should become active at tick 4
    # At tick 2 or 3, position must still be 0
    positions = df_res["position"].to_numpy()
    assert positions[1] == 0.0
    assert positions[2] == 0.0
    assert positions[3] == 0.0


def test_performance_metrics():
    """Verify performance metrics math."""
    equity = np.array([0.0, 10.0, 20.0, 15.0, 25.0])
    gross_equity = np.array([0.0, 10.0, 20.0, 15.0, 25.0])
    fees = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
    penalties = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

    metrics = compute_performance_metrics(
        equity_curve=equity,
        gross_equity_curve=gross_equity,
        fees_curve=fees,
        inventory_penalties_curve=penalties,
        placed_orders_count=10,
        filled_orders_count=8,
        trade_pnls=[10.0, 10.0, -5.0, 10.0],
    )

    assert metrics.net_pnl == 25.0
    assert metrics.fill_rate_pct == 80.0
    assert metrics.win_rate_pct == 75.0  # 3 wins out of 4 trades
    assert metrics.profit_factor == 30.0 / 5.0  # 6.0
