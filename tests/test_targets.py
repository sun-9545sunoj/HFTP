"""
Unit tests for Limit Order Book (LOB) Target Labeling & Smoothed Return Pipeline.
"""

import numpy as np
import polars as pl
import pytest

from src.data.targets import (
    assign_directional_labels,
    compute_mid_price,
    compute_smoothed_returns,
)


@pytest.fixture
def sample_lob_df() -> pl.DataFrame:
    """Generate synthetic LOB dataframe with known prices for mathematical validation."""
    # 10 timestamps with deterministic mid-prices
    # Ask prices: 100.2, 102.2, 104.2, 106.2, 101.2, 98.2, 98.2, 98.2, 100.2, 100.2
    # Bid prices:  99.8, 101.8, 103.8, 105.8, 100.8, 97.8, 97.8, 97.8,  99.8,  99.8
    # Mid-prices: 100.0, 102.0, 104.0, 106.0, 101.0, 98.0, 98.0, 98.0, 100.0, 100.0
    ask_prices = [100.2, 102.2, 104.2, 106.2, 101.2, 98.2, 98.2, 98.2, 100.2, 100.2]
    bid_prices = [99.8, 101.8, 103.8, 105.8, 100.8, 97.8, 97.8, 97.8, 99.8, 99.8]

    return pl.DataFrame({
        "timestamp": list(range(10)),
        "ask_price_1": ask_prices,
        "ask_size_1": [10.0] * 10,
        "bid_price_1": bid_prices,
        "bid_size_1": [10.0] * 10,
    })


def test_compute_mid_price(sample_lob_df: pl.DataFrame):
    """Verify mid-price calculation m_t = (ask_1 + bid_1) / 2."""
    df_mid = compute_mid_price(sample_lob_df)
    assert "mid_price" in df_mid.columns

    expected_mid = [100.0, 102.0, 104.0, 106.0, 101.0, 98.0, 98.0, 98.0, 100.0, 100.0]
    actual_mid = df_mid["mid_price"].to_list()
    np.testing.assert_allclose(actual_mid, expected_mid, rtol=1e-6)


def test_compute_smoothed_returns_mathematical_validation(sample_lob_df: pl.DataFrame):
    """
    Validate smoothed k-tick forward return calculation mathematically.
    Formula: r_k(t) = ((1/k) * sum_{i=1}^k m_{t+i} - m_t) / m_t
    """
    df_mid = compute_mid_price(sample_lob_df)
    k = 2
    df_ret = compute_smoothed_returns(df_mid, k=k)

    returns = df_ret[f"return_{k}"].to_numpy()
    m = df_mid["mid_price"].to_numpy()

    # Manual mathematical computation for k=2:
    # t=0: m_0=100.0, next 2 are m_1=102, m_2=104 -> avg=103.0 -> r_2(0) = (103 - 100)/100 = 0.03
    # t=1: m_1=102.0, next 2 are m_2=104, m_3=106 -> avg=105.0 -> r_2(1) = (105 - 102)/102 = 3/102 ≈ 0.02941176
    # t=2: m_2=104.0, next 2 are m_3=106, m_4=101 -> avg=103.5 -> r_2(2) = (103.5 - 104)/104 = -0.5/104 ≈ -0.00480769
    # t=3: m_3=106.0, next 2 are m_4=101, m_5=98  -> avg=99.5  -> r_2(3) = (99.5 - 106)/106 = -6.5/106
    # t=7: m_7=98.0,  next 2 are m_8=100, m_9=100 -> avg=100.0 -> r_2(7) = (100 - 98)/98 = 2/98
    # t=8, 9: NaN (since N=10 and k=2, last 2 elements do not have k future ticks)

    expected_0 = (np.mean([102.0, 104.0]) - 100.0) / 100.0
    expected_1 = (np.mean([104.0, 106.0]) - 102.0) / 102.0
    expected_2 = (np.mean([106.0, 101.0]) - 104.0) / 104.0
    expected_7 = (np.mean([100.0, 100.0]) - 98.0) / 98.0

    assert np.isclose(returns[0], expected_0)
    assert np.isclose(returns[1], expected_1)
    assert np.isclose(returns[2], expected_2)
    assert np.isclose(returns[7], expected_7)

    # Validate boundary conditions (tail elements should be NaN)
    assert np.isnan(returns[-1])
    assert np.isnan(returns[-2])


def test_assign_directional_labels_fixed_threshold(sample_lob_df: pl.DataFrame):
    """Test discrete directional label assignment y_t in {-1, 0, 1} using fixed threshold alpha."""
    df_mid = compute_mid_price(sample_lob_df)
    k = 2
    df_ret = compute_smoothed_returns(df_mid, k=k)

    # Use alpha = 0.01 (1%)
    # t=0: r_2(0) = 0.03 > 0.01 -> Label 1 (UP)
    # t=1: r_2(1) ≈ 0.0294 > 0.01 -> Label 1 (UP)
    # t=2: r_2(2) ≈ -0.0048 (in [-0.01, 0.01]) -> Label 0 (STATIONARY)
    # t=3: r_2(3) = -6.5/106 ≈ -0.0613 < -0.01 -> Label -1 (DOWN)
    alpha = 0.01
    df_labeled = assign_directional_labels(
        df_ret, return_col=f"return_{k}", alpha=alpha, use_dynamic_volatility=False
    )

    labels = df_labeled["label"].to_list()
    assert labels[0] == 1
    assert labels[1] == 1
    assert labels[2] == 0
    assert labels[3] == -1
    assert labels[-1] is None  # Tail element with NaN return should be None/null


def test_assign_directional_labels_dynamic_volatility(sample_lob_df: pl.DataFrame):
    """Test label assignment with dynamic volatility-adjusted threshold."""
    df_mid = compute_mid_price(sample_lob_df)
    k = 2
    df_ret = compute_smoothed_returns(df_mid, k=k)

    df_labeled = assign_directional_labels(
        df_ret,
        return_col=f"return_{k}",
        use_dynamic_volatility=True,
        volatility_window=5,
        volatility_multiplier=0.5,
        min_alpha=0.001,
    )

    assert "alpha_t" in df_labeled.columns
    assert "label" in df_labeled.columns
    
    alpha_t = df_labeled["alpha_t"].to_numpy()
    assert np.all(alpha_t >= 0.001)
