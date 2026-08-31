import pandas as pd
import glob
import json
import os

res = {}
for d in sorted(glob.glob("lightning_logs/version_*")):
    mfile = os.path.join(d, "metrics.csv")
    if os.path.exists(mfile):
        df = pd.read_csv(mfile)
        # We drop the 'step' and 'epoch' where appropriate or just forward fill and take the last row
        # However, to be more precise, let's just get max validation f1 or acc
        val_f1 = df['val_f1_macro'].max() if 'val_f1_macro' in df.columns else None
        val_acc = df['val_accuracy'].max() if 'val_accuracy' in df.columns else None
        test_f1 = df['test_f1_macro'].max() if 'test_f1_macro' in df.columns else None
        test_acc = df['test_accuracy'].max() if 'test_accuracy' in df.columns else None
        
        # also get train metrics
        train_loss = df['train_loss'].dropna().iloc[-1] if 'train_loss' in df.columns and not df['train_loss'].dropna().empty else None
        
        res[d] = {
            "val_f1_macro": val_f1,
            "val_accuracy": val_acc,
            "test_f1_macro": test_f1,
            "test_accuracy": test_acc,
            "final_train_loss": train_loss
        }

print(json.dumps(res, indent=2))
