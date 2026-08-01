"""
Execution Latency and Parameter Sensitivity Sweeper for Microstructure Backtesting.

Runs grid experiments across latency delays (delta t), queue priority decay parameters,
and inventory holding penalties to analyze strategy performance degradation.
"""

from typing import List, Optional

import numpy as np
import polars as pl

from src.backtest.engine import MicrostructureBacktester
from src.backtest.metrics import PerformanceMetrics


class SensitivityAnalyzer:
    """Runs parameter sweeps across HFT execution realities to assess strategy robustness."""

    def __init__(
        self,
        latencies: Optional[List[int]] = None,
        queue_decay_rates: Optional[List[float]] = None,
        inventory_penalties: Optional[List[float]] = None,
    ) -> None:
        self.latencies = latencies or [1, 2, 5, 10, 20]
        self.queue_decay_rates = queue_decay_rates or [0.1, 0.2, 0.5]
        self.inventory_penalties = inventory_penalties or [0.0, 1e-6, 1e-5]

    def run_sweep(
        self,
        lob_df: pl.DataFrame,
        signals: np.ndarray,
        order_size: float = 1.0,
    ) -> pl.DataFrame:
        """
        Execute parameter sweep over latency, queue decay, and inventory penalty parameters.

        Args:
            lob_df: Standardized LOB Polars DataFrame.
            signals: Array of directional signals y_t in {-1, 0, 1}.
            order_size: Trade order size.

        Returns:
            Polars DataFrame summarizing comparative performance metrics per scenario.
        """
        records = []

        for lat in self.latencies:
            for q_decay in self.queue_decay_rates:
                for inv_pen in self.inventory_penalties:
                    backtester = MicrostructureBacktester(
                        latency_ticks=lat,
                        queue_decay_rate=q_decay,
                        inventory_penalty_factor=inv_pen,
                    )
                    
                    metrics, _ = backtester.run_backtest(
                        lob_df=lob_df,
                        signals=signals,
                        order_size=order_size,
                    )

                    records.append({
                        "latency_ticks": lat,
                        "queue_decay_rate": q_decay,
                        "inventory_penalty": inv_pen,
                        "net_pnl": metrics.net_pnl,
                        "gross_pnl": metrics.gross_pnl,
                        "total_fees": metrics.total_fees,
                        "sharpe_ratio": metrics.sharpe_ratio,
                        "max_drawdown_pct": metrics.max_drawdown_pct,
                        "fill_rate_pct": metrics.fill_rate_pct,
                        "orders_placed": metrics.total_orders_placed,
                        "orders_filled": metrics.total_orders_filled,
                        "win_rate_pct": metrics.win_rate_pct,
                        "profit_factor": metrics.profit_factor,
                    })

        return pl.DataFrame(records)
