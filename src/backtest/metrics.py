"""
Performance Metrics Calculation Module for High-Frequency Microstructure Backtesting.
"""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class PerformanceMetrics:
    """Dataclass holding microstructural strategy backtest performance results."""
    total_pnl: float
    net_pnl: float
    gross_pnl: float
    total_fees: float
    inventory_penalties: float
    sharpe_ratio: float
    max_drawdown: float
    max_drawdown_pct: float
    fill_rate_pct: float
    total_orders_placed: int
    total_orders_filled: int
    win_rate_pct: float
    profit_factor: float


def compute_performance_metrics(
    equity_curve: np.ndarray,
    gross_equity_curve: np.ndarray,
    fees_curve: np.ndarray,
    inventory_penalties_curve: np.ndarray,
    placed_orders_count: int,
    filled_orders_count: int,
    trade_pnls: List[float],
    annualization_factor: float = 252.0 * 6.5 * 3600.0,  # Second-level or tick-level factor
) -> PerformanceMetrics:
    """
    Calculate comprehensive performance metrics from backtest executions.

    Args:
        equity_curve: Net PnL curve array over time.
        gross_equity_curve: Gross PnL curve array.
        fees_curve: Cumulative transaction fees array.
        inventory_penalties_curve: Cumulative inventory penalty costs array.
        placed_orders_count: Total orders submitted.
        filled_orders_count: Total orders successfully filled.
        trade_pnls: List of individual trade PnLs.
        annualization_factor: Factor to scale tick/second returns to annual Sharpe.

    Returns:
        PerformanceMetrics object.
    """
    net_pnl = float(equity_curve[-1]) if len(equity_curve) > 0 else 0.0
    gross_pnl = float(gross_equity_curve[-1]) if len(gross_equity_curve) > 0 else 0.0
    total_fees = float(fees_curve[-1]) if len(fees_curve) > 0 else 0.0
    total_penalties = float(inventory_penalties_curve[-1]) if len(inventory_penalties_curve) > 0 else 0.0

    # Calculate returns for Sharpe ratio
    if len(equity_curve) > 1:
        diffs = np.diff(equity_curve)
        std = np.std(diffs)
        mean = np.mean(diffs)
        sharpe = float((mean / (std + 1e-8)) * np.sqrt(annualization_factor)) if std > 0 else 0.0
    else:
        sharpe = 0.0

    # Calculate Maximum Drawdown
    if len(equity_curve) > 0:
        peak = np.maximum.accumulate(equity_curve)
        drawdown = peak - equity_curve
        max_dd = float(np.max(drawdown))
        
        # Max drawdown percentage
        peak_non_zero = np.where(peak == 0, 1e-8, peak)
        max_dd_pct = float(np.max(drawdown / np.abs(peak_non_zero))) * 100.0
    else:
        max_dd = 0.0
        max_dd_pct = 0.0

    # Fill rate %
    fill_rate = (float(filled_orders_count) / float(placed_orders_count) * 100.0) if placed_orders_count > 0 else 0.0

    # Trade statistics
    if len(trade_pnls) > 0:
        pnls_arr = np.array(trade_pnls)
        wins = pnls_arr[pnls_arr > 0]
        losses = pnls_arr[pnls_arr < 0]
        
        win_rate = (len(wins) / len(pnls_arr)) * 100.0
        sum_wins = np.sum(wins) if len(wins) > 0 else 0.0
        sum_losses = np.abs(np.sum(losses)) if len(losses) > 0 else 1e-8
        profit_factor = float(sum_wins / sum_losses)
    else:
        win_rate = 0.0
        profit_factor = 0.0

    return PerformanceMetrics(
        total_pnl=net_pnl,
        net_pnl=net_pnl,
        gross_pnl=gross_pnl,
        total_fees=total_fees,
        inventory_penalties=total_penalties,
        sharpe_ratio=sharpe,
        max_drawdown=max_dd,
        max_drawdown_pct=max_dd_pct,
        fill_rate_pct=fill_rate,
        total_orders_placed=placed_orders_count,
        total_orders_filled=filled_orders_count,
        win_rate_pct=win_rate,
        profit_factor=profit_factor,
    )
