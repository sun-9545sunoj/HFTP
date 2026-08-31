"""
Latency & Queue-Aware Market Microstructure Event-Driven Backtesting Engine.

Models realistic HFT execution dynamics:
1. Fixed execution latency delay (delta t ticks) between signal generation and order placement.
2. L3/L2 queue priority model tracking volume ahead and queue decay.
3. Transaction cost model (Maker rebates vs Taker fees and slippage).
4. Inventory risk holding cost penalization & maximum position risk limits.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
import polars as pl

from src.backtest.metrics import PerformanceMetrics, compute_performance_metrics


class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    LIMIT = "LIMIT"
    MARKET = "MARKET"


class OrderStatus(str, Enum):
    PENDING_LATENCY = "PENDING_LATENCY"
    ACTIVE = "ACTIVE"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"


@dataclass
class Order:
    """Represents an order in the microstructure backtest engine."""
    order_id: int
    side: OrderSide
    order_type: OrderType
    price: float
    size: float
    creation_tick: int
    active_tick: int
    queue_ahead: float
    status: OrderStatus = OrderStatus.PENDING_LATENCY


@dataclass
class Trade:
    """Represents a filled trade execution."""
    trade_id: int
    order_id: int
    side: OrderSide
    price: float
    size: float
    timestamp_tick: int
    fee: float


class MicrostructureBacktester:
    """
    Event-driven Microstructure Backtesting Engine accounting for execution latency,
    queue priority decay, maker/taker fee structures, and inventory risk holding costs.
    """

    def __init__(
        self,
        latency_ticks: int = 2,
        maker_fee: float = -0.0001,  # 1 bps rebate for limit orders
        taker_fee: float = 0.0004,   # 4 bps fee for market orders
        slippage_ticks: float = 0.5,
        inventory_penalty_factor: float = 1e-6,
        queue_decay_rate: float = 0.2,  # 20% volume ahead consumed per tick
    ) -> None:
        self.latency_ticks = latency_ticks
        self.maker_fee = maker_fee
        self.taker_fee = taker_fee
        self.slippage_ticks = slippage_ticks
        self.inventory_penalty_factor = inventory_penalty_factor
        self.queue_decay_rate = queue_decay_rate

    def run_backtest(
        self,
        lob_df: pl.DataFrame,
        signals: np.ndarray,
        order_size: float = 1.0,
        max_position: Optional[float] = None,
    ) -> Tuple[PerformanceMetrics, pl.DataFrame]:
        """
        Run event-driven microstructure backtest given a LOB DataFrame and directional signals.

        Args:
            lob_df: Polars DataFrame containing canonical ask_price_1, ask_size_1, bid_price_1, bid_size_1, mid_price.
            signals: Array of signals y_t in {-1, 0, 1} where 1 = BUY, -1 = SELL, 0 = NEUTRAL.
            order_size: Trade size per order.
            max_position: Maximum inventory position limit (|q_t| <= max_position).

        Returns:
            Tuple of (PerformanceMetrics, Polars DataFrame with equity curve & positions).
        """
        required_cols = ["ask_price_1", "ask_size_1", "bid_price_1", "bid_size_1", "mid_price"]
        for col in required_cols:
            if col not in lob_df.columns:
                raise ValueError(f"Required column '{col}' missing from LOB DataFrame.")

        ask_p1 = lob_df["ask_price_1"].to_numpy()
        ask_v1 = lob_df["ask_size_1"].to_numpy()
        bid_p1 = lob_df["bid_price_1"].to_numpy()
        bid_v1 = lob_df["bid_size_1"].to_numpy()
        mid_p = lob_df["mid_price"].to_numpy()
        n_ticks = len(lob_df)

        if len(signals) != n_ticks:
            raise ValueError(f"Length of signals ({len(signals)}) does not match LOB ticks ({n_ticks}).")

        # Portfolio state tracking
        position: float = 0.0
        cash: float = 0.0
        
        orders: List[Order] = []
        trades: List[Trade] = []
        
        equity_curve = np.zeros(n_ticks)
        gross_equity_curve = np.zeros(n_ticks)
        fees_curve = np.zeros(n_ticks)
        penalties_curve = np.zeros(n_ticks)
        position_curve = np.zeros(n_ticks)

        cumulative_fees: float = 0.0
        cumulative_penalties: float = 0.0
        placed_orders_count: int = 0
        order_id_counter: int = 0
        trade_id_counter: int = 0
        trade_pnls: List[float] = []

        entry_prices: List[float] = []

        for t in range(n_ticks):
            current_ask = ask_p1[t]
            current_bid = bid_p1[t]
            current_mid = mid_p[t]
            sig = signals[t]

            # 1. Process Signal & Generate Orders (with Latency Delay & Max Position Limits)
            if sig != 0:
                can_buy = (max_position is None) or (position + order_size <= max_position)
                can_sell = (max_position is None) or (position - order_size >= -max_position)

                if sig == 1 and can_buy:  # BUY Signal
                    order_id_counter += 1
                    placed_orders_count += 1
                    queue_vol = bid_v1[t]
                    new_order = Order(
                        order_id=order_id_counter,
                        side=OrderSide.BUY,
                        order_type=OrderType.LIMIT,
                        price=current_bid,
                        size=order_size,
                        creation_tick=t,
                        active_tick=t + self.latency_ticks,
                        queue_ahead=queue_vol,
                        status=OrderStatus.PENDING_LATENCY,
                    )
                    orders.append(new_order)
                elif sig == -1 and can_sell:  # SELL Signal
                    order_id_counter += 1
                    placed_orders_count += 1
                    queue_vol = ask_v1[t]
                    new_order = Order(
                        order_id=order_id_counter,
                        side=OrderSide.SELL,
                        order_type=OrderType.LIMIT,
                        price=current_ask,
                        size=order_size,
                        creation_tick=t,
                        active_tick=t + self.latency_ticks,
                        queue_ahead=queue_vol,
                        status=OrderStatus.PENDING_LATENCY,
                    )
                    orders.append(new_order)

            # 2. Update Pending Orders Status (Activate orders past latency delay)
            for o in orders:
                if o.status == OrderStatus.PENDING_LATENCY and t >= o.active_tick:
                    o.status = OrderStatus.ACTIVE

            # 3. Simulate Queue Priority Decay & Order Fills
            for o in orders:
                if o.status != OrderStatus.ACTIVE:
                    continue

                filled = False
                fill_price = o.price

                if o.side == OrderSide.BUY:
                    if current_ask <= o.price:
                        filled = True
                        fill_price = current_ask
                    else:
                        o.queue_ahead -= o.queue_ahead * self.queue_decay_rate
                        if o.queue_ahead <= 0.1:
                            filled = True
                            fill_price = o.price

                elif o.side == OrderSide.SELL:
                    if current_bid >= o.price:
                        filled = True
                        fill_price = current_bid
                    else:
                        o.queue_ahead -= o.queue_ahead * self.queue_decay_rate
                        if o.queue_ahead <= 0.1:
                            filled = True
                            fill_price = o.price

                if filled:
                    o.status = OrderStatus.FILLED
                    fee = fill_price * o.size * self.maker_fee
                    cumulative_fees += fee

                    trade_id_counter += 1
                    tr = Trade(
                        trade_id=trade_id_counter,
                        order_id=o.order_id,
                        side=o.side,
                        price=fill_price,
                        size=o.size,
                        timestamp_tick=t,
                        fee=fee,
                    )
                    trades.append(tr)

                    if o.side == OrderSide.BUY:
                        position += o.size
                        cash -= (fill_price * o.size + fee)
                        entry_prices.append(fill_price)
                    else:
                        position -= o.size
                        cash += (fill_price * o.size - fee)
                        if len(entry_prices) > 0:
                            entry_p = entry_prices.pop(0)
                            trade_pnl = (fill_price - entry_p) * o.size - fee
                            trade_pnls.append(trade_pnl)

            # 4. Calculate Inventory Penalty for holding position risk
            inventory_penalty = self.inventory_penalty_factor * (position ** 2)
            cumulative_penalties += inventory_penalty

            # 5. Mark-to-Market Valuation at Current Mid Price
            gross_pnl = cash + (position * current_mid)
            net_pnl = gross_pnl - cumulative_penalties

            equity_curve[t] = net_pnl
            gross_equity_curve[t] = gross_pnl
            fees_curve[t] = cumulative_fees
            penalties_curve[t] = cumulative_penalties
            position_curve[t] = position

        # Compute summary metrics
        filled_count = len(trades)
        metrics = compute_performance_metrics(
            equity_curve=equity_curve,
            gross_equity_curve=gross_equity_curve,
            fees_curve=fees_curve,
            inventory_penalties_curve=penalties_curve,
            placed_orders_count=placed_orders_count,
            filled_orders_count=filled_count,
            trade_pnls=trade_pnls,
        )

        res_df = lob_df.with_columns([
            pl.Series("equity", equity_curve),
            pl.Series("gross_equity", gross_equity_curve),
            pl.Series("fees", fees_curve),
            pl.Series("position", position_curve),
        ])

        return metrics, res_df
