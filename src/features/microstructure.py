"""
High-Frequency Market Microstructure Feature Engineering Module.

Implements vectorized computations for:
1. Multi-Level Order Flow Imbalance (OFI) based on Cont et al. (2014) across N levels.
2. Micro-Price: P_{micro} = (v^{bid} * p^{ask} + v^{ask} * p^{bid}) / (v^{bid} + v^{ask}).
3. Deep Queue Volume Imbalance across N depth levels: I_t = (sum v^{bid} - sum v^{ask}) / (sum v^{bid} + sum v^{ask}).
4. Dynamic Bid-Ask Spread and multi-level depth decay profiles.
5. Volume-Weighted Average Price (VWAP) and VWAP Price Deviations over rolling window W.
"""

from typing import List, Optional

import numpy as np
import polars as pl


def compute_micro_price(
    df: pl.DataFrame,
    ask_price_col: str = "ask_price_1",
    ask_size_col: str = "ask_size_1",
    bid_price_col: str = "bid_price_1",
    bid_size_col: str = "bid_size_1",
    out_col: str = "micro_price",
) -> pl.DataFrame:
    """
    Compute Level-1 Micro-Price:
        P_{micro} = (v^{bid} * p^{ask} + v^{ask} * p^{bid}) / (v^{bid} + v^{ask})
    """
    num = pl.col(bid_size_col) * pl.col(ask_price_col) + pl.col(ask_size_col) * pl.col(bid_price_col)
    den = pl.col(bid_size_col) + pl.col(ask_size_col)
    
    return df.with_columns((num / den).alias(out_col))


def compute_queue_imbalance(
    df: pl.DataFrame,
    num_levels: int = 10,
    out_col: Optional[str] = None,
) -> pl.DataFrame:
    """
    Compute Deep Queue Volume Imbalance across top N depth levels:
        I_t = (sum_{i=1}^N v_{i,t}^{bid} - sum_{i=1}^N v_{i,t}^{ask}) / 
              (sum_{i=1}^N v_{i,t}^{bid} + sum_{i=1}^N v_{i,t}^{ask})
    """
    if out_col is None:
        out_col = f"queue_imbalance_{num_levels}"

    bid_sum = pl.sum_horizontal([f"bid_size_{i}" for i in range(1, num_levels + 1)])
    ask_sum = pl.sum_horizontal([f"ask_size_{i}" for i in range(1, num_levels + 1)])

    imbalance_expr = (bid_sum - ask_sum) / (bid_sum + ask_sum)
    return df.with_columns(imbalance_expr.alias(out_col))


def compute_order_flow_imbalance(
    df: pl.DataFrame,
    num_levels: int = 10,
    out_col: str = "ofi_total",
) -> pl.DataFrame:
    """
    Compute Multi-Level Order Flow Imbalance (OFI) based on Cont et al. (2014).
    """
    ofi_level_exprs = []
    
    for i in range(1, num_levels + 1):
        p_bid = pl.col(f"bid_price_{i}")
        p_bid_prev = p_bid.shift(1)
        v_bid = pl.col(f"bid_size_{i}")
        v_bid_prev = v_bid.shift(1)

        delta_L_bid = (
            pl.when(p_bid > p_bid_prev)
            .then(v_bid)
            .when(p_bid == p_bid_prev)
            .then(v_bid - v_bid_prev)
            .otherwise(-v_bid_prev)
        )

        p_ask = pl.col(f"ask_price_{i}")
        p_ask_prev = p_ask.shift(1)
        v_ask = pl.col(f"ask_size_{i}")
        v_ask_prev = v_ask.shift(1)

        delta_L_ask = (
            pl.when(p_ask > p_ask_prev)
            .then(-v_ask_prev)
            .when(p_ask == p_ask_prev)
            .then(v_ask - v_ask_prev)
            .otherwise(v_ask)
        )

        ofi_i = (delta_L_bid - delta_L_ask).fill_null(0.0).alias(f"ofi_level_{i}")
        ofi_level_exprs.append(ofi_i)

    df_ofi = df.with_columns(ofi_level_exprs)
    total_ofi = pl.sum_horizontal([f"ofi_level_{i}" for i in range(1, num_levels + 1)]).alias(out_col)
    return df_ofi.with_columns(total_ofi)


def compute_spread_and_depth_features(
    df: pl.DataFrame,
    num_levels: int = 10,
    decay_lambda: float = 0.5,
) -> pl.DataFrame:
    """
    Compute Bid-Ask Spread and Multi-Level Depth Decay Profiles.
    """
    mid = (pl.col("ask_price_1") + pl.col("bid_price_1")) / 2.0
    spread = (pl.col("ask_price_1") - pl.col("bid_price_1")).alias("bid_ask_spread")
    rel_spread = (spread / mid).alias("relative_spread")

    weights = [np.exp(-decay_lambda * (i - 1)) for i in range(1, num_levels + 1)]

    decay_bid_expr = pl.sum_horizontal(
        [pl.col(f"bid_size_{i}") * weights[i - 1] for i in range(1, num_levels + 1)]
    ).alias("depth_decay_bid")

    decay_ask_expr = pl.sum_horizontal(
        [pl.col(f"ask_size_{i}") * weights[i - 1] for i in range(1, num_levels + 1)]
    ).alias("depth_decay_ask")

    df_res = df.with_columns([spread, rel_spread, decay_bid_expr, decay_ask_expr])

    decay_ratio = (
        (pl.col("depth_decay_bid") - pl.col("depth_decay_ask")) /
        (pl.col("depth_decay_bid") + pl.col("depth_decay_ask"))
    ).alias("depth_decay_ratio")

    return df_res.with_columns(decay_ratio)


def compute_vwap_features(
    df: pl.DataFrame,
    window_size: int = 50,
) -> pl.DataFrame:
    """
    Compute Volume-Weighted Average Price (VWAP) and relative price deviation over rolling window W.

    VWAP_W = sum_{i=0}^{W-1} (mid_price * total_vol) / sum_{i=0}^{W-1} (total_vol)
    VWAP_Dev = (mid_price - VWAP_W) / VWAP_W
    """
    if "mid_price" not in df.columns:
        df = df.with_columns(((pl.col("ask_price_1") + pl.col("bid_price_1")) / 2.0).alias("mid_price"))

    total_vol = pl.col("ask_size_1") + pl.col("bid_size_1")
    vol_price = pl.col("mid_price") * total_vol

    rolling_vol_price = vol_price.rolling_sum(window_size=window_size, min_samples=1)
    rolling_vol = total_vol.rolling_sum(window_size=window_size, min_samples=1)

    vwap = (rolling_vol_price / rolling_vol).alias(f"vwap_{window_size}")
    vwap_dev = ((pl.col("mid_price") - vwap) / vwap).alias(f"vwap_dev_{window_size}")

    return df.with_columns([vwap, vwap_dev])


def extract_all_microstructure_features(
    df: pl.DataFrame,
    num_levels: int = 10,
    decay_lambda: float = 0.5,
    vwap_window: int = 50,
) -> pl.DataFrame:
    """
    Convenience function to extract all microstructure features in a single pipeline execution.
    """
    df_out = compute_micro_price(df)
    df_out = compute_queue_imbalance(df_out, num_levels=num_levels)
    df_out = compute_order_flow_imbalance(df_out, num_levels=num_levels)
    df_out = compute_spread_and_depth_features(df_out, num_levels=num_levels, decay_lambda=decay_lambda)
    df_out = compute_vwap_features(df_out, window_size=vwap_window)
    return df_out
