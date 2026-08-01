"""
Unit tests for LOB Data Loader (FI-2010, LOBSTER, and NSE stock formats).
"""

from pathlib import Path
import numpy as np
import polars as pl
import pytest

from src.data.loader import LOBLoader


@pytest.fixture
def temp_fi2010_file(tmp_path: Path) -> Path:
    """Generate a synthetic FI-2010 text file (149 rows, 20 ticks)."""
    n_ticks = 20
    data = np.zeros((149, n_ticks))

    data[0, :] = 100.5  # ask_price_1
    data[1, :] = 10.0   # ask_size_1
    data[2, :] = 99.5   # bid_price_1
    data[3, :] = 15.0   # bid_size_1

    file_path = tmp_path / "fi2010_sample.txt"
    np.savetxt(file_path, data)
    return file_path


@pytest.fixture
def temp_lobster_files(tmp_path: Path) -> tuple[Path, Path]:
    """Generate synthetic LOBSTER orderbook and message CSV files."""
    n_rows = 15
    ob_data = np.zeros((n_rows, 40))
    ob_data[:, 0] = 1000000
    ob_data[:, 1] = 50
    ob_data[:, 2] = 990000
    ob_data[:, 3] = 30

    ob_path = tmp_path / "AAPL_2025-01-01_34200000_57600000_orderbook_10.csv"
    np.savetxt(ob_path, ob_data, delimiter=",", fmt="%d")

    msg_data = []
    for i in range(n_rows):
        msg_data.append([34200.0 + i * 0.1, 1, 1000 + i, 10, 1000000, 1])
    
    msg_path = tmp_path / "AAPL_2025-01-01_34200000_57600000_message_10.csv"
    np.savetxt(msg_path, msg_data, delimiter=",", fmt="%.4f,%d,%d,%d,%d,%d")

    return ob_path, msg_path


@pytest.fixture
def temp_nse_file(tmp_path: Path) -> Path:
    """Generate synthetic NSE stock LOB CSV file (e.g. RELIANCE / TCS depth)."""
    nse_df = pl.DataFrame({
        "timestamp": [1600000000 + i for i in range(10)],
        "ap1": [2500.50 + i * 0.05 for i in range(10)],
        "av1": [100.0] * 10,
        "bp1": [2500.00 + i * 0.05 for i in range(10)],
        "bv1": [150.0] * 10,
        "ap2": [2500.55 + i * 0.05 for i in range(10)],
        "av2": [200.0] * 10,
        "bp2": [2499.95 + i * 0.05 for i in range(10)],
        "bv2": [250.0] * 10,
    })

    file_path = tmp_path / "RELIANCE_NSE_LOB.csv"
    nse_df.write_csv(file_path)
    return file_path


def test_load_fi2010(temp_fi2010_file: Path):
    """Test loading FI-2010 format file."""
    df = LOBLoader.load_fi2010(temp_fi2010_file, num_levels=10)
    assert isinstance(df, pl.DataFrame)
    assert len(df) == 20
    assert "ask_price_1" in df.columns
    assert "bid_price_1" in df.columns
    assert "mid_price" in df.columns
    assert np.allclose(df["mid_price"].to_numpy(), 100.0)


def test_load_lobster(temp_lobster_files: tuple[Path, Path]):
    """Test loading LOBSTER format files with price factor scaling."""
    ob_path, msg_path = temp_lobster_files
    df = LOBLoader.load_lobster(
        orderbook_path=ob_path,
        message_path=msg_path,
        num_levels=10,
        price_factor=10000.0,
    )

    assert len(df) == 15
    assert df["ask_price_1"][0] == 100.0
    assert df["bid_price_1"][0] == 99.0
    assert df["mid_price"][0] == 99.5
    assert "timestamp" in df.columns


def test_load_nse(temp_nse_file: Path):
    """Test loading NSE stock LOB CSV with column auto-renaming."""
    df = LOBLoader.load_nse(temp_nse_file, num_levels=2)

    assert len(df) == 10
    assert "ask_price_1" in df.columns
    assert "bid_price_1" in df.columns
    assert "mid_price" in df.columns
    assert df["ask_price_1"][0] == 2500.50
    assert df["bid_price_1"][0] == 2500.00
    assert df["mid_price"][0] == 2500.25


def test_save_parquet(tmp_path: Path, temp_fi2010_file: Path):
    """Test converting LOB DataFrame to Apache Arrow and saving as Parquet."""
    df = LOBLoader.load_fi2010(temp_fi2010_file)
    output_parquet = tmp_path / "lob_data.parquet"

    LOBLoader.save_parquet(df, output_parquet)
    assert output_parquet.exists()

    df_read = pl.read_parquet(output_parquet)
    assert df_read.shape == df.shape
