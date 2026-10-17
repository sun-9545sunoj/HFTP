"""
Target Labeling and Smoothed Return Pipeline for Limit Order Book Mid-Price Modeling.

Provides vectorized functions in Polars and NumPy to compute:
1. Mid-price: m_t = (ask_1 + bid_1) / 2
2. Smoothed k-tick ahead forward return: r_k(t) = ((1/k) * sum_{i=1}^k m_{t+i} - m_t) / m_t
3. Directional categorical labels y_t in {-1, 0, 1} with dynamic volatility thresholding.
"""

from typing import Optional, Union

import numpy as np
import polars as pl


def compute_mid_price(
    df: pl.DataFrame,
    ask_col: str = "ask_price_1",
    bid_col: str = "bid_price_1",
    out_col: str = "mid_price",
) -> pl.DataFrame:
    """
    Compute mid-price m_t = (p_t^{ask} + p_t^{bid}) / 2.

    Args:
        df: Input Polars DataFrame containing ask and bid price columns.
        ask_col: Column name for best ask price.
        bid_col: Column name for best bid price.
        out_col: Target column name for mid-price.

    Returns:
        Polars DataFrame with added mid-price column.
    """
    if ask_col not in df.columns or bid_col not in df.columns:
        raise ValueError(f"Columns {ask_col} and {bid_col} must be present in DataFrame.")
        
    return df.with_columns(
        ((pl.col(ask_col) + pl.col(bid_col)) / 2.0).alias(out_col)
    )


def compute_smoothed_returns(
    df: pl.DataFrame,
    k: int,
    mid_price_col: str = "mid_price",
    out_col: Optional[str] = None,
) -> pl.DataFrame:
    """
    Calculate smoothed forward return over future window of k ticks.

    Formula:
        r_k(t) = ((1 / k) * sum_{i=1}^k m_{t+i} - m_t) / m_t

    Args:
        df: Input Polars DataFrame containing mid-price.
        k: Forward lookahead horizon (number of ticks k >= 1).
        mid_price_col: Column name for mid-price.
        out_col: Output column name. Defaults to f"return_{k}".

    Returns:
        Polars DataFrame with added smoothed return column.
    """
    if k < 1:
        raise ValueError(f"Lookahead window k must be >= 1, got {k}")
    if mid_price_col not in df.columns:
        raise ValueError(f"Column '{mid_price_col}' not found in DataFrame.")

    if out_col is None:
        out_col = f"return_{k}"

    mid_prices = df[mid_price_col].to_numpy()
    n = len(mid_prices)

    if n < k + 1:
        # Not enough ticks to compute smoothed returns
        res_returns = np.full(n, np.nan, dtype=np.float64)
    else:
        # Forward mean of m_{t+1..t+k} for t in 0..n-k-1. A sliding window is
        # used instead of differencing a global cumsum: the cumsum over ~1e6
        # prices accumulates rounding error (~1e-10 relative), which turned
        # "no move" ticks into spurious tiny UP/DOWN returns.
        mids = np.asarray(mid_prices, dtype=np.float64)
        forward_means = np.lib.stride_tricks.sliding_window_view(mids[1:], k).mean(axis=1)
        
        m_t = mid_prices[: n - k]
        returns_valid = (forward_means - m_t) / m_t
        # Snap float residue (~1e-16) to exact zero so "no move" stays a tie
        # under quantile labeling. Real moves are >= half a tick / (k * price),
        # i.e. orders of magnitude above this cutoff for any listed price.
        returns_valid[np.abs(returns_valid) < 1e-10] = 0.0

        # Tail elements beyond n-k don't have k future ticks available
        res_returns = np.full(n, np.nan, dtype=np.float64)
        res_returns[: n - k] = returns_valid

    return df.with_columns(pl.Series(out_col, res_returns))


def assign_directional_labels(
    df: pl.DataFrame,
    return_col: str,
    alpha: float = 0.0002,
    use_dynamic_volatility: bool = False,
    volatility_window: int = 100,
    volatility_multiplier: float = 0.5,
    min_alpha: float = 0.00005,
    label_col: str = "label",
) -> pl.DataFrame:
    """
    Assign dynamic categorical label y_t in {-1, 0, 1} based on smoothed forward return.

    Classification Rule:
        y_t =  1 (UP)         if r_k(t) >  alpha_t
        y_t = -1 (DOWN)       if r_k(t) < -alpha_t
        y_t =  0 (STATIONARY) otherwise

    Dynamic Volatility Thresholding:
        If use_dynamic_volatility is True:
            alpha_t = max(volatility_multiplier * std_W(r_1), min_alpha)
        Otherwise:
            alpha_t = alpha (fixed scalar)

    Args:
        df: Input DataFrame containing return_col.
        return_col: Column name of smoothed forward returns r_k(t).
        alpha: Fixed threshold value (used when use_dynamic_volatility is False).
        use_dynamic_volatility: If True, dynamically calculates alpha per tick.
        volatility_window: Rolling window size W for computing returns volatility.
        volatility_multiplier: Scaling factor c for rolling standard deviation.
        min_alpha: Lower bound floor for dynamic threshold alpha_t.
        label_col: Output column name for discrete labels {-1, 0, 1}.

    Returns:
        Polars DataFrame with discrete label column and threshold value column ('alpha_t').
    """
    if return_col not in df.columns:
        raise ValueError(f"Return column '{return_col}' not found in DataFrame.")

    if use_dynamic_volatility:
        # Calculate tick-to-tick return if available or use mid_price rolling std
        if "mid_price" in df.columns:
            r1 = df["mid_price"].pct_change().fill_null(0.0)
        else:
            r1 = df[return_col].fill_null(0.0)

        # Rolling standard deviation over window W
        rolling_std = r1.rolling_std(window_size=volatility_window, min_samples=1).fill_null(0.0)
        
        # Calculate dynamic alpha_t
        alpha_expr = (
            (rolling_std * volatility_multiplier)
            .clip(lower_bound=min_alpha)
            .alias("alpha_t")
        )
        df_with_alpha = df.with_columns(alpha_expr)
    else:
        df_with_alpha = df.with_columns(pl.lit(alpha).alias("alpha_t"))

    # Vectorized label evaluation
    ret = df_with_alpha[return_col]
    
    # If perfect quantile balancing is requested via kwargs (handled dynamically)
    if df_with_alpha.get_column("alpha_t").mean() == -1.0: # signal for quantile balancing
        quant_down = ret.quantile(0.33)
        quant_up = ret.quantile(0.66)
        label_expr = (
            pl.when(ret.is_null() | ret.is_nan()).then(None)
            .when(pl.col(return_col) > quant_up).then(1)
            .when(pl.col(return_col) < quant_down).then(-1)
            .otherwise(0)
            .cast(pl.Int8)
            .alias(label_col)
        )
    else:
        label_expr = (
            pl.when(ret.is_null() | ret.is_nan())
            .then(None)
            .when(pl.col(return_col) > pl.col("alpha_t"))
            .then(1)
            .when(pl.col(return_col) < -pl.col("alpha_t"))
            .then(-1)
            .otherwise(0)
            .cast(pl.Int8)
            .alias(label_col)
        )

    return df_with_alpha.with_columns(label_expr)
