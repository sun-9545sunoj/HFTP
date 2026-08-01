"""
Unit tests for Market Microstructure Feature Engineering.
"""

import numpy as np
import polars as pl
import pytest

from src.features.microstructure import (
    compute_micro_price,
    compute_order_flow_imbalance,
    compute_queue_imbalance,
    compute_spread_and_depth_features,
    extract_all_microstructure_features,
)


@pytest.fixture
def sample_lob_data() -> pl.DataFrame:
    """Generate synthetic 10-level LOB dataframe across 5 ticks."""
    dict_data = {"timestamp": list(range(5))}
    for i in range(1, 11):
        # Ask prices increase with level
        dict_data[f"ask_price_{i}"] = [100.0 + i * 0.1 + tick * 0.05 for tick in range(5)]
        dict_data[f"ask_size_{i}"] = [10.0 * i] * 5
        # Bid prices decrease with level
        dict_data[f"bid_price_{i}"] = [100.0 - i * 0.1 + tick * 0.05 for tick in range(5)]
        dict_data[f"bid_size_{i}"] = [20.0 * i] * 5
    return pl.DataFrame(dict_data)


def test_compute_micro_price(sample_lob_data: pl.DataFrame):
    """Test Micro-Price P_{micro} = (v^{bid} * p^{ask} + v^{ask} * p^{bid}) / (v^{bid} + v^{ask})."""
    df = compute_micro_price(sample_lob_data)
    assert "micro_price" in df.columns
    
    # Tick 0: ask_1=100.1, ask_size_1=10.0, bid_1=99.9, bid_size_1=20.0
    # P_micro = (20.0 * 100.1 + 10.0 * 99.9) / 30.0 = (2002 + 999) / 30 = 3001 / 30 = 100.033333...
    expected_p0 = (20.0 * 100.1 + 10.0 * 99.9) / 30.0
    actual_p0 = df["micro_price"][0]
    assert np.isclose(actual_p0, expected_p0)


def test_compute_queue_imbalance(sample_lob_data: pl.DataFrame):
    """Test Deep Queue Volume Imbalance I_t = (sum v^{bid} - sum v^{ask}) / (sum v^{bid} + sum v^{ask})."""
    df = compute_queue_imbalance(sample_lob_data, num_levels=10)
    assert "queue_imbalance_10" in df.columns

    # sum v_bid = sum_{i=1}^{10} 20*i = 20 * 55 = 1100
    # sum v_ask = sum_{i=1}^{10} 10*i = 10 * 55 = 550
    # I_t = (1100 - 550) / (1100 + 550) = 550 / 1650 = 1/3 = 0.333333...
    expected_imb = 1.0 / 3.0
    assert np.allclose(df["queue_imbalance_10"].to_numpy(), expected_imb)


def test_compute_order_flow_imbalance():
    """Test Cont et al. (2014) Order Flow Imbalance logic."""
    # Construct a 2-tick scenario for Level 1
    # Tick 0: bid_price=100, bid_size=10, ask_price=101, ask_size=15
    # Tick 1: bid_price=100 (same), bid_size=12 (+2), ask_price=101 (same), ask_size=10 (-5)
    # Delta L_bid = 12 - 10 = 2
    # Delta L_ask = 10 - 15 = -5
    # OFI_1 = 2 - (-5) = 7
    df_lob = pl.DataFrame({
        "timestamp": [0, 1],
        "ask_price_1": [101.0, 101.0],
        "ask_size_1": [15.0, 10.0],
        "bid_price_1": [100.0, 100.0],
        "bid_size_1": [10.0, 12.0],
    })

    df_ofi = compute_order_flow_imbalance(df_lob, num_levels=1)
    ofi_vals = df_ofi["ofi_total"].to_numpy()
    
    assert ofi_vals[0] == 0.0  # First tick has no prev tick, filled with 0
    assert ofi_vals[1] == 7.0  # Second tick OFI = 7.0


def test_compute_spread_and_depth_features(sample_lob_data: pl.DataFrame):
    """Test Spread and Depth decay features."""
    df = compute_spread_and_depth_features(sample_lob_data, num_levels=10, decay_lambda=0.5)
    assert "bid_ask_spread" in df.columns
    assert "relative_spread" in df.columns
    assert "depth_decay_bid" in df.columns
    assert "depth_decay_ask" in df.columns
    assert "depth_decay_ratio" in df.columns


def test_extract_all_microstructure_features(sample_lob_data: pl.DataFrame):
    """Test pipeline wrapper function."""
    df = extract_all_microstructure_features(sample_lob_data, num_levels=10)
    expected_cols = [
        "micro_price",
        "queue_imbalance_10",
        "ofi_total",
        "bid_ask_spread",
        "relative_spread",
        "depth_decay_ratio",
    ]
    for col in expected_cols:
        assert col in df.columns
