"""
Data Ingestion Pipeline for Limit Order Book (LOB) Datasets.

Parses FI-2010 benchmark dataset, LOBSTER L2/L3 CSV formats, and NSE Stock LOB tick data
into standardized, memory-efficient Polars and PyArrow DataFrames.
"""

from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq


class LOBDatasetFormat(str, Enum):
    """Supported LOB dataset formats."""
    FI2010 = "fi2010"
    LOBSTER = "lobster"
    NSE = "nse"
    GENERIC = "generic"


class LOBLoader:
    """
    High-performance parser and loader for High-Frequency Limit Order Book datasets.
    
    Supports loading:
    - FI-2010 Benchmark Dataset (Ntakaris et al., 2018)
    - LOBSTER L2/L3 CSV datasets (Orderbook & Message files)
    - NSE (National Stock Exchange of India) Stock Tick LOB data (Zerodha, TrueData, IB, NSE TBT)
    - Generic multi-level LOB CSV/Parquet files
    """

    @staticmethod
    def get_canonical_column_names(num_levels: int = 10) -> List[str]:
        """
        Generate canonical column names for an N-level limit order book.

        Args:
            num_levels: Number of depth levels (default: 10).

        Returns:
            List of standardized column names:
            ['ask_price_1', 'ask_size_1', 'bid_price_1', 'bid_size_1', ...]
        """
        cols = []
        for i in range(1, num_levels + 1):
            cols.extend([
                f"ask_price_{i}",
                f"ask_size_{i}",
                f"bid_price_{i}",
                f"bid_size_{i}",
            ])
        return cols

    @classmethod
    def load_fi2010(
        cls,
        file_path: Union[str, Path],
        num_levels: int = 10,
        has_extracted_features: bool = True,
    ) -> pl.DataFrame:
        """Parse FI-2010 dataset file into a standardized Polars DataFrame."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"FI-2010 file not found at: {path}")

        raw_arr = np.loadtxt(path)
        if raw_arr.shape[0] == 149 or (has_extracted_features and raw_arr.shape[0] > raw_arr.shape[1]):
            if raw_arr.shape[0] == 149:
                raw_arr = raw_arr.T

        num_ticks = raw_arr.shape[0]
        num_raw_cols = num_levels * 4

        if raw_arr.shape[1] < num_raw_cols:
            raise ValueError(
                f"FI-2010 array has {raw_arr.shape[1]} columns, expected at least {num_raw_cols}."
            )

        lob_matrix = raw_arr[:, :num_raw_cols]
        canonical_cols = cls.get_canonical_column_names(num_levels=num_levels)

        df_dict: Dict[str, np.ndarray] = {
            "timestamp": np.arange(num_ticks, dtype=np.int64)
        }
        for idx, col in enumerate(canonical_cols):
            df_dict[col] = lob_matrix[:, idx]

        df = pl.DataFrame(df_dict)
        df = df.with_columns(
            ((pl.col("ask_price_1") + pl.col("bid_price_1")) / 2.0).alias("mid_price")
        )
        return df

    @classmethod
    def load_lobster(
        cls,
        orderbook_path: Union[str, Path],
        message_path: Optional[Union[str, Path]] = None,
        num_levels: int = 10,
        price_factor: float = 10000.0,
    ) -> pl.DataFrame:
        """Parse LOBSTER format orderbook and optional message CSV files."""
        ob_path = Path(orderbook_path)
        if not ob_path.exists():
            raise FileNotFoundError(f"LOBSTER orderbook file not found at: {ob_path}")

        canonical_cols = cls.get_canonical_column_names(num_levels=num_levels)
        df_raw = pl.read_csv(ob_path, has_header=False)
        
        target_cols_count = num_levels * 4
        if df_raw.width < target_cols_count:
            raise ValueError(
                f"LOBSTER orderbook file has {df_raw.width} columns, required {target_cols_count}."
            )
        
        df_raw = df_raw.select(df_raw.columns[:target_cols_count])
        df_raw.columns = canonical_cols

        price_exprs = [
            (pl.col(col) / price_factor).alias(col)
            for col in canonical_cols if "price" in col
        ]
        df = df_raw.with_columns(price_exprs)

        if message_path is not None:
            msg_p = Path(message_path)
            if msg_p.exists():
                msg_df = pl.read_csv(
                    msg_p,
                    has_header=False,
                    new_columns=["timestamp", "event_type", "order_id", "size", "price", "direction"]
                )
                df = df.with_columns(msg_df["timestamp"])
            else:
                df = df.with_columns(pl.Series("timestamp", np.arange(len(df), dtype=np.float64)))
        else:
            df = df.with_columns(pl.Series("timestamp", np.arange(len(df), dtype=np.float64)))

        df = df.with_columns(
            ((pl.col("ask_price_1") + pl.col("bid_price_1")) / 2.0).alias("mid_price")
        )
        return df

    @classmethod
    def load_nse(
        cls,
        file_path: Union[str, Path],
        num_levels: int = 5,
        column_mapping: Optional[Dict[str, str]] = None,
    ) -> pl.DataFrame:
        """
        Parse NSE Stock Limit Order Book CSV or Parquet files (e.g. Zerodha, TrueData, NSE TBT).

        Supports standard 5-level retail broker depth or 10/20 level institutional TBT feeds.

        Args:
            file_path: Path to CSV or Parquet file containing NSE LOB tick data.
            num_levels: Number of depth levels (default: 5 for standard NSE broker depth, 10 for TBT).
            column_mapping: Custom dictionary mapping existing column names to canonical names.

        Returns:
            Standardized Polars DataFrame with canonical LOB columns and mid_price in INR.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"NSE dataset file not found at: {path}")

        if path.suffix == ".parquet":
            df_raw = pl.read_parquet(path)
        else:
            df_raw = pl.read_csv(path)

        if column_mapping:
            df_raw = df_raw.rename(column_mapping)

        # Standardize column naming if columns follow patterns like 'bp1', 'bs1', 'ap1', 'as1' or 'bid_price_1'
        rename_dict = {}
        for col in df_raw.columns:
            lower_col = col.lower()
            if lower_col in ["time", "datetime", "timestamp", "date"]:
                rename_dict[col] = "timestamp"
            
            for lvl in range(1, num_levels + 1):
                if lower_col in [f"ap{lvl}", f"ask_p{lvl}", f"ask_price_{lvl}", f"askprice{lvl}"]:
                    rename_dict[col] = f"ask_price_{lvl}"
                elif lower_col in [f"av{lvl}", f"as{lvl}", f"ask_v{lvl}", f"ask_size_{lvl}", f"askqty{lvl}"]:
                    rename_dict[col] = f"ask_size_{lvl}"
                elif lower_col in [f"bp{lvl}", f"bid_p{lvl}", f"bid_price_{lvl}", f"bidprice{lvl}"]:
                    rename_dict[col] = f"bid_price_{lvl}"
                elif lower_col in [f"bv{lvl}", f"bs{lvl}", f"bid_v{lvl}", f"bid_size_{lvl}", f"bidqty{lvl}"]:
                    rename_dict[col] = f"bid_size_{lvl}"

        if rename_dict:
            df_raw = df_raw.rename(rename_dict)

        if "timestamp" not in df_raw.columns:
            df_raw = df_raw.with_columns(pl.Series("timestamp", np.arange(len(df_raw), dtype=np.int64)))

        # Ensure mid-price column exists
        df = df_raw.with_columns(
            ((pl.col("ask_price_1") + pl.col("bid_price_1")) / 2.0).alias("mid_price")
        )
        return df

    @staticmethod
    def to_pyarrow(df: pl.DataFrame) -> pa.Table:
        """Convert a Polars LOB DataFrame to PyArrow Table."""
        return df.to_arrow()

    @staticmethod
    def save_parquet(df: pl.DataFrame, output_path: Union[str, Path]) -> None:
        """Save LOB DataFrame to Apache Parquet format using PyArrow engine."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrow_table = df.to_arrow()
        pq.write_table(arrow_table, path, compression="snappy")
