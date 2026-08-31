"""
FastAPI Dashboard Web Server for High-Frequency LOB Microstructure Analytics.
Serves the HTML frontend and provides REST endpoints for live LOB data streaming and Model predictions.
"""

import math
from pathlib import Path
from fastapi import FastAPI, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
import uvicorn
import torch
import torch.nn.functional as F

from src.data.loader import LOBLoader
from src.data.targets import assign_directional_labels, compute_mid_price, compute_smoothed_returns
from src.features.microstructure import extract_all_microstructure_features
from src.models.lightning_module import DeepLOBModule
from evaluate_test_set import find_latest_checkpoint

app = FastAPI(title="HFT DeepLOB Dashboard API")
DASHBOARD_DIR = Path(__file__).parent / "src" / "dashboard"

# Global state for LOB simulation
state = {
    "df_raw": None,
    "df_features": None,
    "predictions": None,
    "current_tick": 0,
    "total_ticks": 0,
    "model_loaded": False
}

def load_backend_data():
    """Loads the test dataset and runs the model inference once on startup."""
    print("-> Loading Dashboard Data and Model...")
    try:
        # 1. Load Data
        test_file_path = "src/data/FI2010_test.csv"
        test_df_raw = LOBLoader.load_fi2010(test_file_path, num_levels=10)
        df_mid = compute_mid_price(test_df_raw)
        df_ret = compute_smoothed_returns(df_mid, k=5)
        df_labeled = assign_directional_labels(df_ret, return_col="return_5", use_dynamic_volatility=True)
        df_features = extract_all_microstructure_features(df_labeled, num_levels=10).drop_nulls(subset=["label"])
        
        state["df_raw"] = test_df_raw
        state["df_features"] = df_features
        state["total_ticks"] = len(df_features)

        # 2. Load Model
        # We enforce using the Champion DeepLOB model (epoch=02, macro_f1=0.4947)
        ckpt_file = Path("checkpoints/enhanced-fi2010-deeplob-best-epoch=02-val_f1_macro=0.4947.ckpt")
        if not ckpt_file.exists():
             ckpt_file = find_latest_checkpoint()
             
        module = DeepLOBModule.load_from_checkpoint(ckpt_file)
        model = module.model
        model.eval()
        device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
        model.to(device)

        # 3. Pre-compute Predictions
        feature_cols = [
            c for c in df_features.columns 
            if c not in ["timestamp", "label", "return_5", "alpha_t"] 
            and not df_features[c].is_null().all()
        ]
        
        feature_arr = df_features.select(feature_cols).to_numpy()
        tensor_x = torch.tensor(feature_arr, dtype=torch.float32).unsqueeze(1).to(device) # Shape [N, 1, 61]
        
        # We need sequential sliding windows, but for dashboard simplicity and speed, 
        # we will approximate by passing overlapping sequences or padding.
        # Actually, let's just create sequences of T=100
        seq_len = 100
        all_probs = []
        
        print("-> Running fast inference for dashboard...")
        # To avoid OOM, run in batches
        with torch.no_grad():
            for i in range(0, len(tensor_x) - seq_len, 500):
                batch_x = []
                for j in range(i, min(i+500, len(tensor_x) - seq_len)):
                     batch_x.append(tensor_x[j:j+seq_len])
                if not batch_x:
                    break
                b_x = torch.stack(batch_x).squeeze(2) # [B, T, F]
                b_x = b_x.unsqueeze(1) # [B, 1, T, F]
                logits = model(b_x)
                probs = F.softmax(logits, dim=1).cpu().numpy()
                all_probs.extend(probs.tolist())
                
        state["predictions"] = all_probs
        state["model_loaded"] = True
        print("-> Dashboard Backend Ready!")
        
    except Exception as e:
        print(f"Failed to load backend data: {e}")

# Trigger data load on startup
@app.on_event("startup")
async def startup_event():
    import threading
    threading.Thread(target=load_backend_data).start()

@app.get("/api/status")
def get_status():
    return {"ready": state["model_loaded"], "total_ticks": state["total_ticks"]}

@app.get("/api/next_tick")
def get_next_tick():
    """Advances the simulation by 1 tick and returns LOB state and predictions."""
    if not state["model_loaded"]:
        return JSONResponse(status_code=503, content={"detail": "Model still loading..."})
    
    idx = state["current_tick"] + 100 # Offset by seq_len
    if idx >= len(state["predictions"]):
        state["current_tick"] = 0 # loop around
        idx = 100
        
    state["current_tick"] += 1
    
    row = state["df_raw"].row(idx, named=True)
    preds = state["predictions"][state["current_tick"]]
    
    lob_levels = []
    for i in range(1, 11):
        lob_levels.append({
            "level": i,
            "bid_price": row.get(f"bid_price_{i}", 0),
            "bid_vol": row.get(f"bid_size_{i}", 0),
            "ask_price": row.get(f"ask_price_{i}", 0),
            "ask_vol": row.get(f"ask_size_{i}", 0),
        })
        
    return {
        "tick_id": idx,
        "timestamp": row.get("timestamp", idx),
        "lob": lob_levels,
        "predictions": {
            "DOWN": preds[0] * 100,
            "STAT": preds[1] * 100,
            "UP": preds[2] * 100
        }
    }

# Mount static files (must be at the bottom)
app.mount("/", StaticFiles(directory=str(DASHBOARD_DIR), html=True), name="dashboard")

if __name__ == "__main__":
    print("=========================================================")
    print("  FastAPI Microstructure Web Dashboard Launcher          ")
    print("=========================================================\n")
    uvicorn.run("run_dashboard:app", host="0.0.0.0", port=8000, reload=True)
