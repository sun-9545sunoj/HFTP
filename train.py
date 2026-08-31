"""
PyTorch Lightning Training Script for DeepLOB & Spatial-Temporal Transformer Models.

Supports training on:
1. Enriched Microstructure Features (--use_micro_features)
2. NSE Stock LOB Data (--dataset_type nse --file_path data/RELIANCE_LOB.csv)
3. Real FI-2010 Benchmark Datasets (--dataset_type fi2010 --file_path data/Train_Dst.txt)
4. Real LOBSTER L2/L3 Datasets (--dataset_type lobster --file_path data/orderbook.csv --message_path data/message.csv)
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

    elif dataset_type == "parquet":
        if not file_path or not Path(file_path).exists():
            raise FileNotFoundError(f"Parquet file not found at: {file_path}")
        print(f"-> Ingesting Parquet Dataset from: {file_path}")
        return pl.read_parquet(file_path)

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
    parser = argparse.ArgumentParser(description="Train DeepLOB / Transformer models with performance optimizations.")
    parser.add_argument("--dataset_type", type=str, default="synthetic", choices=["synthetic", "nse", "fi2010", "lobster", "parquet"])
    parser.add_argument("--file_path", type=str, default=None)
    parser.add_argument("--message_path", type=str, default=None)
    parser.add_argument("--model_type", type=str, default="deeplob", choices=["deeplob", "transformer"])
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size.")
    parser.add_argument("--seq_len", type=int, default=100, help="Sequence length T.")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate.")
    parser.add_argument("--k", type=int, default=5, help="Lookahead return window k ticks.")
    parser.add_argument("--loss_type", type=str, default="focal", choices=["focal", "weighted_ce", "ce"])
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="L2 regularization weight decay.")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate for regularization.")
    parser.add_argument("--use_micro_features", action="store_true", help="Include engineered microstructure features (OFI, Micro-price, VWAP).")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--resume_checkpoint", type=str, default=None, help="Path to checkpoint to resume training from.")
    args = parser.parse_args()

    print("=========================================================")
    print(f"   Enhanced Training ({args.model_type.upper()} on {args.dataset_type.upper()})   ")
    print("=========================================================\n")

    # 1. Prepare Data
    print("[1/4] Ingesting LOB dataset & computing target labels...")
    df_raw = load_dataset(
        dataset_type=args.dataset_type,
        file_path=args.file_path,
        message_path=args.message_path,
        num_levels=10,
    )

    if args.dataset_type == "parquet":
        print("-> Dataset is Parquet format. Skipping redundant feature extraction and labeling.")
        df_features = df_raw
        if "label" not in df_features.columns:
            raise ValueError("Parquet file must contain 'label' column.")
    else:
        # 1. Feature Engineering & Target Labeling
        print("[1/4] Computing Labels and Microstructure Features...")
        num_levels_raw = 10 if args.dataset_type in ["fi2010", "lobster"] else 5
        df_mid = compute_mid_price(df_raw, ask_col="ask_price_1", bid_col="bid_price_1")
        df_ret = compute_smoothed_returns(df_mid, k=args.k)
        df_labeled = assign_directional_labels(
            df_ret,
            return_col=f"return_{args.k}",
            use_dynamic_volatility=True,
            volatility_multiplier=0.5,
        )
        
        if args.use_micro_features:
            df_features = extract_all_microstructure_features(df_labeled, num_levels=num_levels_raw)
        else:
            df_features = df_labeled

    if args.use_micro_features or args.dataset_type == "parquet":
        canonical_cols = [
            c for c in df_features.columns 
            if c not in ["timestamp", "label", f"return_{args.k}", "alpha_t"] 
            and not df_features[c].is_null().all()
        ]
        print(f"-> Using Enriched Feature Set: {len(canonical_cols)} features.")
    else:
        canonical_cols = LOBLoader.get_canonical_column_names(num_levels=num_levels_feat)
        print(f"-> Using Standard LOB Feature Set: {len(canonical_cols)} raw features.")
    
    # 3. Create PyTorch DataLoaders
    print(f"[3/4] Creating DataLoaders (Sequence Length T={args.seq_len}, Batch Size={args.batch_size})...")
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
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        loss_type=args.loss_type,
        class_weights=class_weights,
    )

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename=f"enhanced-{args.dataset_type}-{args.model_type}-best-{{epoch:02d}}-{{val_f1_macro:.4f}}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
    )

    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=4,
        mode="min",
    )

    # 5. Trainer Execution
    print(f"[4/4] Starting PyTorch Lightning Training for {args.epochs} epochs...")
    trainer = pl_trainer.Trainer(
        max_epochs=args.epochs,
        accelerator="auto",
        devices=1,
        callbacks=[checkpoint_callback, early_stop_callback],
        enable_progress_bar=True,
    )

    print(f"\n[4/4] Commencing Training Phase (Max Epochs: {args.epochs})...")
    
    if args.resume_checkpoint:
        print(f"-> Resuming from checkpoint: {args.resume_checkpoint}")
        trainer.fit(model_module, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=args.resume_checkpoint)
    else:
        trainer.fit(model_module, train_dataloaders=train_loader, val_dataloaders=val_loader)
        
    print(f"\n-> Training Complete! Best model checkpoint saved to: {checkpoint_dir}")


if __name__ == "__main__":
    main()
