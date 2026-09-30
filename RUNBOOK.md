# HFTPM_LOBAS: End-to-End Execution Guide

This guide explains the exact flow of scripts required to go from raw Limit Order Book (LOB) data to a fully trained model and backtest evaluation.

---

## Step 1: Data Preparation
**Goal:** Convert raw LOBSTER CSV files into enriched, normalized Parquet datasets for fast loading during training.

**Script:** `src/data/prepare_balanced_lobster.py`

**What it does:**
1. Reads raw message and orderbook CSV files from `src/data/`.
2. Computes mid-price, bid-ask spread, and extracts 61 microstructure features.
3. Calculates forward-looking return labels (UP, DOWN, STATIONARY) using rolling horizons.
4. Z-score normalizes the features (crucially keeping the `mid_price` raw for realistic backtesting).
5. Balances the classes (undersampling the majority class) and saves to `LOBSTER_balanced_train.parquet` and `LOBSTER_balanced_test.parquet`.

**Command to run (if data changes):**
```bash
.venv/bin/python src/data/prepare_balanced_lobster.py
```

---

## Step 2: Model Training
**Goal:** Train one of the three deep learning architectures on the prepared dataset.

**Script:** `train.py`

**What it does:**
1. Loads the `.parquet` dataset.
2. Instantiates the specified model (`deeplob`, `translob`, or `transformer`).
3. Uses PyTorch Lightning to train the model, automatically handling learning rate decay, focal loss (for class imbalance), and checkpointing.
4. Saves the best model checkpoint to the `checkpoints/` directory based on Validation F1 Macro score.

**Commands to run:**

*To train Model 1 (DeepLOB):*
```bash
.venv/bin/python train.py --model_type deeplob --dataset_type parquet --file_path src/data/LOBSTER_balanced_train.parquet --batch_size 256 --epochs 15
```

*To train Model 2 (TransLOB):*
```bash
.venv/bin/python train.py --model_type translob --dataset_type parquet --file_path src/data/LOBSTER_balanced_train.parquet --batch_size 256 --epochs 15
```

*To train Model 3 (Pure Attention):*
```bash
.venv/bin/python train.py --model_type transformer --dataset_type parquet --file_path src/data/LOBSTER_balanced_train.parquet --batch_size 256 --epochs 15
```

---

## Step 3: Out-of-Sample Evaluation & Backtesting
**Goal:** Evaluate the trained model on unseen test data and run a simulated trading backtest.

**Script:** `evaluate_test_set.py`

**What it does:**
1. Automatically finds the most recent `.ckpt` file in the `checkpoints/` directory.
2. Runs inference over `LOBSTER_balanced_test.parquet`.
3. Computes Out-of-Sample metrics (Accuracy, Precision, Recall, F1, Confusion Matrix).
4. Feeds the model's confidence probabilities (filtered by `--min_prob`) into the `MicrostructureBacktester`.
5. The backtester simulates limit order entries, market order exits, flat per-share fees ($0.003 taker, $0.002 maker), and inventory risk limits.

**Command to run:**
```bash
.venv/bin/python evaluate_test_set.py --dataset_type parquet --test_file src/data/LOBSTER_balanced_test.parquet --min_prob 0.65 --max_position 10.0
```
*Note: Adjust `--min_prob` (e.g., 0.65) to control how confident the model must be before placing a trade.*

---

## Project Structure Quick Reference

- **`src/data/`**: Data preprocessing scripts and `.parquet` files.
- **`src/models/deeplob.py`**: Defines all PyTorch architectures (`DeepLOB`, `TransLOB`, `SpatialTemporalTransformerLOB`).
- **`src/models/lightning_module.py`**: PyTorch Lightning wrapper for training/validation steps and loss functions.
- **`src/backtest/engine.py`**: The `MicrostructureBacktester` execution logic and fee structures.
- **`checkpoints/`**: Saved model weights (best epochs).
- **`reports/`**: Markdown reports and analysis generated during the project.
