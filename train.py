"""
PyTorch Lightning Training Script for DeepLOB & Spatial-Temporal Transformer Models.

Supports training on:
1. NSE Stock LOB Data (--dataset_type nse --file_path data/RELIANCE_LOB.csv)
2. Real FI-2010 Benchmark Datasets (--dataset_type fi2010 --file_path data/Train_Dst.txt)
3. Real LOBSTER L2/L3 Datasets (--dataset_type lobster --file_path data/orderbook.csv --message_path data/message.csv)
4. Synthetic Demonstration Datasets (--dataset_type synthetic)
"""

import argparse
from pathlib import Path
import numpy as np
import polars as pl
import pytorch_lightning as pl_trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from src.data.dataset import create_chronological_dataloaders
from src.data.loader import LOBLoader
from src.data.targets import assign_directional_labels, compute_mid_price, compute_smoothed_returns
from src.features.microstructure import extract_all_microstructure_features
from src.models.lightning_module import DeepLOBModule


def generate_synthetic_training_data(n_ticks: int = 1000, num_levels: int = 10) -> pl.DataFrame:
    """Generate synthetic 10-level LOB dataset for training demonstration."""
    dict_data = {"timestamp": list(range(n_ticks))}
    np.random.seed(42)
    base_mid = 100.0 + np.cumsum(np.random.normal(0, 0.05, n_ticks))

    for i in range(1, num_levels + 1):
        dict_data[f"ask_price_{i}"] = base_mid + (i * 0.05)
        dict_data[f"ask_size_{i}"] = np.random.randint(10, 100, n_ticks).astype(float)
        dict_data[f"bid_price_{i}"] = base_mid - (i * 0.05)
        dict_data[f"bid_size_{i}"] = np.random.randint(10, 100, n_ticks).astype(float)

    return pl.DataFrame(dict_data)


def load_dataset(
    dataset_type: str,
    file_path: str = None,
    message_path: str = None,
    num_levels: int = 10,
) -> pl.DataFrame:
    """Load LOB dataset from NSE, FI-2010, LOBSTER, or Synthetic generator."""
    if dataset_type == "nse":
        if not file_path or not Path(file_path).exists():
            raise FileNotFoundError(f"NSE dataset file not found at: {file_path}")
        print(f"-> Ingesting NSE Stock LOB Dataset from: {file_path}")
        return LOBLoader.load_nse(file_path, num_levels=min(num_levels, 5))

    elif dataset_type == "fi2010":
        if not file_path or not Path(file_path).exists():
            raise FileNotFoundError(f"FI-2010 file not found at: {file_path}")
        print(f"-> Ingesting FI-2010 Benchmark Dataset from: {file_path}")
        return LOBLoader.load_fi2010(file_path, num_levels=num_levels)

    elif dataset_type == "lobster":
        if not file_path or not Path(file_path).exists():
            raise FileNotFoundError(f"LOBSTER orderbook file not found at: {file_path}")
        print(f"-> Ingesting LOBSTER L2/L3 Dataset from: {file_path}")
        return LOBLoader.load_lobster(
            orderbook_path=file_path,
            message_path=message_path,
            num_levels=num_levels,
        )

    else:
        print("-> Generating Synthetic Demonstration LOB Dataset (1,000 ticks)...")
        return generate_synthetic_training_data(n_ticks=1000, num_levels=num_levels)


def main():
    parser = argparse.ArgumentParser(description="Train DeepLOB / Transformer models on NSE, FI-2010, or LOBSTER dataset.")
    parser.add_argument("--dataset_type", type=str, default="synthetic", choices=["synthetic", "nse", "fi2010", "lobster"], help="Dataset format.")
    parser.add_argument("--file_path", type=str, default=None, help="Path to NSE/FI-2010 file or LOBSTER orderbook csv.")
    parser.add_argument("--message_path", type=str, default=None, help="Optional path to LOBSTER message csv.")
    parser.add_argument("--model_type", type=str, default="deeplob", choices=["deeplob", "transformer"], help="Model architecture.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")
    parser.add_argument("--seq_len", type=int, default=100, help="Sequence length T.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--k", type=int, default=5, help="Lookahead return window k ticks.")
    parser.add_argument("--loss_type", type=str, default="focal", choices=["focal", "weighted_ce", "ce"], help="Loss function.")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory to save model checkpoints.")
    args = parser.parse_args()

    print("=========================================================")
    print(f"    LOB Model Training ({args.model_type.upper()} on {args.dataset_type.upper()})    ")
    print("=========================================================\n")

    # 1. Prepare Data
    print("[1/4] Ingesting LOB dataset & computing target labels...")
    raw_df = load_dataset(
        dataset_type=args.dataset_type,
        file_path=args.file_path,
        message_path=args.message_path,
        num_levels=10,
    )

    df_mid = compute_mid_price(raw_df)
    df_ret = compute_smoothed_returns(df_mid, k=args.k)
    df_labeled = assign_directional_labels(
        df_ret,
        return_col=f"return_{args.k}",
        use_dynamic_volatility=True,
    )

    print(f"-> Ticks Processed: {len(df_labeled):,} rows")
    label_counts = df_labeled["label"].value_counts().to_dict()
    print(f"-> Directional Class Distribution: {dict(zip(label_counts['label'], label_counts['count']))}\n")

    # 2. Extract Microstructure Features
    num_levels_feat = 5 if args.dataset_type == "nse" else 10
    print(f"[2/4] Engineering microstructure features across {num_levels_feat} levels...")
    df_features = extract_all_microstructure_features(df_labeled, num_levels=num_levels_feat)
    canonical_cols = LOBLoader.get_canonical_column_names(num_levels=num_levels_feat)
    
    # 3. Create PyTorch DataLoaders
    print(f"[3/4] Creating chronological DataLoaders (Sequence Length T={args.seq_len})...")
    train_loader, val_loader, test_loader, class_weights = create_chronological_dataloaders(
        df_features,
        feature_cols=canonical_cols,
        label_col="label",
        seq_len=args.seq_len,
        batch_size=args.batch_size,
    )
    print(f"-> Train batches: {len(train_loader)} | Val batches: {len(val_loader)} | Class Weights: {class_weights.numpy()}\n")

    # 4. Instantiate Lightning Module
    model_module = DeepLOBModule(
        model_type=args.model_type,
        in_features=len(canonical_cols),
        seq_len=args.seq_len,
        num_classes=3,
        learning_rate=args.lr,
        loss_type=args.loss_type,
        class_weights=class_weights,
    )

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename=f"{args.dataset_type}-{args.model_type}-best-{{epoch:02d}}-{{val_f1_macro:.4f}}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
    )

    # 5. Trainer Execution
    print(f"[4/4] Starting PyTorch Lightning Training for {args.epochs} epochs...")
    trainer = pl_trainer.Trainer(
        max_epochs=args.epochs,
        accelerator="auto",
        callbacks=[checkpoint_callback],
        log_every_n_steps=5,
    )

    trainer.fit(model_module, train_dataloaders=train_loader, val_dataloaders=val_loader)
    print(f"\n-> Training Complete! Best model checkpoint saved to: {checkpoint_dir}")


if __name__ == "__main__":
    main()
