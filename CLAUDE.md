# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Tick-level mid-price direction forecasting (y ∈ {-1, 0, 1}, horizon k=5) on Limit Order Book data (LOBSTER 5 stocks: AAPL, AMZN, GOOG, MSFT, INTC; also FI-2010, NSE), plus a latency/queue-aware backtester. Stack: Polars, PyTorch, PyTorch Lightning (MPS on macOS).

## Commands

Always use the project venv: `.venv/bin/python`.

```bash
# Tests
.venv/bin/python -m pytest tests -q
.venv/bin/python -m pytest tests/test_dataset.py::test_name -q   # single test

# --- Leak-free pipeline (use this for all new experiments) ---
# 1. Build src/data/LOBSTER_sequential.parquet (per-stock chronological, stock/split columns, labels for k in 5,10,20,50,100)
.venv/bin/python prepare_sequential_lobster.py

# 2. Baselines (majority + last-tick logistic regression)
.venv/bin/python baseline_linear.py --k 20 --stocks AAPL,AMZN,GOOG

# 3. Train one model (model_type: deeplob | translob | transformer; --pooling last|mean; --augment --aug_strength ...)
.venv/bin/python train.py --model_type translob --dataset_type parquet \
  --file_path src/data/LOBSTER_sequential.parquet --k 20 --stocks AAPL,AMZN,GOOG --pooling last

# 4. Evaluate an explicit checkpoint (raw argmax, overall + per stock; appends to reports/results.jsonl)
.venv/bin/python evaluate_sequential.py --checkpoint checkpoints/<file>.ckpt --k 20 --stocks AAPL,AMZN,GOOG

# Full benchmark (train + eval all 3 models)
nohup ./run_benchmark.sh 20 AAPL,AMZN,GOOG > reports/benchmark.log 2>&1 &

# --- Legacy (shuffled, leaky) pipeline: kept only to reproduce old results as an ablation ---
.venv/bin/python prepare_balanced_lobster.py
.venv/bin/python evaluate_test_set.py ...   # loads most recently modified .ckpt; applies --min_prob filter before F1

.venv/bin/python evaluate_ensemble.py   # multi-model ensemble
.venv/bin/python run_dashboard.py       # web dashboard at http://localhost:8000
```

Training takes ~5 min/epoch on MPS for the 3 small-tick stocks (~17 min/epoch on the full legacy set) — run them in the background with `nohup` and log to `reports/`. Python stdout is block-buffered when redirected; logs may look empty until the process exits (use `python -u` for live logs).

## Architecture

Data flow: `src/data/loader.py` (LOBLoader → canonical Polars table) → `src/data/targets.py` (mid-price, smoothed returns) → `src/features/microstructure.py` (OFI, micro-price, spread, depth decay, VWAP) → `prepare_sequential_lobster.py` (per-stock rolling Z-score W=1000, chronological 70/15/15 split, 100-row purge between splits, per-stock train-fit 33/66% quantile labels `label_k{k}`) → `src/data/dataset.py` (`segment_end_indices` + `LOBWindowDataset`: T=100 windows that never cross a stock/split boundary; `create_segmented_dataloaders`) → `src/models/` → `evaluate_sequential.py`. `train.py` switches to segmented loaders automatically when the parquet has a `split` column and uses `label_k{--k}`.

Data facts: all LOBSTER files are the free sample of ONE day (2012-06-21). AAPL/AMZN/GOOG are small-tick (balanced labels from k≈10–20); MSFT/INTC are large-tick (~$30 price, mid rarely moves; STATIONARY dominates even at k=100) — analyse them separately.

Key invariants that span files:
- **Column layout is interleaved per level**: `ask_price_i, ask_size_i, bid_price_i, bid_size_i` → index `4*(i-1)+{0,1,2,3}` for i=1..10, then microstructure features. `src/data/augmentation.py` index constants depend on this.
- **Feature selection must go through `get_model_feature_cols()`** in `src/data/dataset.py` (used by train.py and both evaluate scripts). It excludes `timestamp, label, return_k, alpha_t` and raw **`mid_price`**, which stays unnormalized in the parquet only for the backtester. Models take 60 features; checkpoints made before this change expect 61 and won't load.
- Labels {-1,0,1} map to class indices {0,1,2} (`map_labels_to_indices` / `map_indices_to_signals`).
- Models (`src/models/deeplob.py`: DeepLOB, TransLOB, SpatialTemporalTransformerLOB) are wrapped by `DeepLOBModule` in `lightning_module.py` (focal / weighted CE loss from `loss.py`). Checkpoints monitor `val_f1_macro`.
- `compute_smoothed_returns` uses a sliding-window mean and snaps |r| < 1e-10 to 0. A global-cumsum version produced ~1e-10 rounding noise that quantile labels split into fake UP/DOWN; don't reintroduce it.
- Transformer models take `pooling` ("last" for new runs; "mean" is the legacy default so old checkpoints load).

## Known issues (as of Oct 2026, branch `model4-augmentation`)

- Legacy `prepare_balanced_lobster.py` shuffles rows before windowing (leak) and its labels were partly float noise. All metrics in `reports/*.md` before Oct 2026 come from it and are invalid. The sequential pipeline above fixes both issues.
- Backtester reports implausible profit for chance-level models (needs investigation).
- 3 tests fail before any of this work was started: `test_backtest.py::test_backtest_execution`, `test_loader.py::test_load_fi2010`, `test_loader.py::test_save_parquet`.
- `RUNBOOK.md` wrongly gives the prep script path as `src/data/prepare_balanced_lobster.py`.
