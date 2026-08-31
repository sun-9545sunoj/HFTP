"""
Optuna Bayesian Hyperparameter Tuning Script for DeepLOB
Automatically searches for the mathematically optimal learning rate, batch size, and weight decay.
"""

import argparse
import optuna
from optuna.integration import PyTorchLightningPruningCallback
import pytorch_lightning as pl_trainer
from pytorch_lightning.callbacks import EarlyStopping
import warnings

# Suppress PyTorch Lightning logging noise for cleaner Optuna output
import logging
logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
warnings.filterwarnings("ignore")

from src.data.dataset import create_chronological_dataloaders
from src.data.loader import LOBLoader
from src.data.targets import assign_directional_labels, compute_mid_price, compute_smoothed_returns
from src.features.microstructure import extract_all_microstructure_features
from src.models.lightning_module import DeepLOBModule

# We will load the dataset globally ONCE to avoid reloading it 50 times during tuning.
print("Loading FI-2010 Dataset into Memory for Tuning...")
TRAIN_FILE = "src/data/FI2010_train 2.csv"
raw_df = LOBLoader.load_fi2010(TRAIN_FILE, num_levels=10)
df_mid = compute_mid_price(raw_df)
df_ret = compute_smoothed_returns(df_mid, k=5)
df_labeled = assign_directional_labels(df_ret, return_col="return_5", use_dynamic_volatility=True)
df_features = extract_all_microstructure_features(df_labeled, num_levels=10)

canonical_cols = [
    c for c in df_features.columns 
    if c not in ["timestamp", "label", "return_5", "alpha_t"] 
    and not df_features[c].is_null().all()
]
print(f"Global Dataset Ready! Using {len(canonical_cols)} enriched features.")


def objective(trial: optuna.trial.Trial) -> float:
    # 1. Suggest Hyperparameters for this trial
    lr = trial.suggest_float("learning_rate", 1e-5, 5e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512])
    
    # 2. Create DataLoaders with the suggested Batch Size
    train_loader, val_loader, test_loader, class_weights = create_chronological_dataloaders(
        df_features,
        feature_cols=canonical_cols,
        label_col="label",
        seq_len=100,
        batch_size=batch_size,
    )
    
    # 3. Instantiate the Model
    model = DeepLOBModule(
        model_type="deeplob",
        in_features=len(canonical_cols),
        seq_len=100,
        num_classes=3,
        learning_rate=lr,
        weight_decay=weight_decay,
        loss_type="focal",
        class_weights=class_weights,
    )
    
    # 4. Configure PyTorch Lightning Trainer with Pruning
    # Pruning will stop unpromising trials (e.g. at epoch 2) if validation F1 is terrible
    early_stop_callback = EarlyStopping(monitor="val_f1_macro", mode="max", patience=3)
    pruning_callback = PyTorchLightningPruningCallback(trial, monitor="val_f1_macro")
    
    trainer = pl_trainer.Trainer(
        max_epochs=10,            # Shorter epochs for tuning to run more trials
        accelerator="auto",
        enable_checkpointing=False, # We don't need to save all 50 models
        logger=False,             # Disable tensorboard logging for tuning speed
        callbacks=[early_stop_callback, pruning_callback],
        enable_progress_bar=False # Keep console clean
    )
    
    # 5. Train the model
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    
    # 6. Return the ultimate metric we want to maximize
    return trainer.callback_metrics["val_f1_macro"].item()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optuna Hyperparameter Tuning for DeepLOB")
    parser.add_argument("--trials", type=int, default=20, help="Number of Bayesian optimization trials to run")
    args = parser.parse_args()

    print("=========================================================")
    print(f" Starting Optuna Bayesian Hyperparameter Tuning: {args.trials} Trials")
    print("=========================================================\n")

    # We want to MAXIMIZE the Macro F1-Score
    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2))
    
    # Run the optimization
    study.optimize(objective, n_trials=args.trials)

    print("\n=========================================================")
    print(" Tuning Complete! Optimal Hyperparameters Found:")
    print("=========================================================")
    print(f"  Best Validation Macro F1: {study.best_value:.4f}")
    for key, value in study.best_params.items():
        print(f"  - {key}: {value}")
    
    print("\n(You can now use these parameters in `train.py`)")
