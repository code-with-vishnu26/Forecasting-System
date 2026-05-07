# -*- coding: utf-8 -*-
"""
Master training script -- trains all 4 models for all 43 US states.
Run from project root: python train.py [--states California Texas] [--val-weeks 16]
"""

import argparse
import os
import sys
import json
import time
import warnings

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_pipeline import build_features, train_val_split, get_state_series
from src.model_selector import evaluate_all_models, save_registry

MODEL_DIR = "models"
DATA_FILE = "Forecasting Case- Study (1).xlsx"


def main(states_filter=None, val_weeks=16):
    print("=" * 60)
    print("  Sales Forecasting System -- Training Pipeline")
    print("=" * 60)

    # Step 1: Build features
    print("\n[1/3] Loading and engineering features...")
    t0 = time.time()
    df = build_features(DATA_FILE)
    print(f"      Done. Shape: {df.shape}  ({time.time()-t0:.1f}s)")

    # Save features
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/features.csv", index=False)
    print("      Features saved -> data/features.csv")

    # Step 2: Train per state
    all_states = sorted(df["State"].unique().tolist())
    if states_filter:
        all_states = [s for s in all_states if s in states_filter]
    print(f"\n[2/3] Training models for {len(all_states)} states...")
    print(f"      Validation window: last {val_weeks} weeks\n")

    registry = {}
    for i, state in enumerate(all_states, 1):
        print(f"\n--- [{i}/{len(all_states)}] {state} ---")
        t1 = time.time()
        state_df = get_state_series(df, state)
        if len(state_df) < val_weeks + 30:
            print(f"  Skipping {state} -- insufficient data ({len(state_df)} rows)")
            continue

        train_df, val_df = train_val_split(state_df, val_weeks=val_weeks)
        print(f"  Train: {len(train_df)} rows, Val: {len(val_df)} rows")

        result = evaluate_all_models(
            state=state,
            train_df=train_df,
            val_df=val_df,
            model_dir=MODEL_DIR,
        )

        # Store in registry
        registry[state] = {
            "best_model": result["best_model"],
            "last_train_date": str(train_df["Date"].max().date()),
            "metrics": {
                model: {k: v for k, v in m.items() if k != "predictions"}
                for model, m in result["metrics"].items()
            },
            "val_actual": result["val_actual"],
            "val_dates": result["val_dates"],
        }
        for model_name, m in result["metrics"].items():
            if "predictions" in m:
                registry[state].setdefault("val_predictions", {})[model_name] = m["predictions"]

        elapsed = time.time() - t1
        best = result["best_model"]
        best_rmse = result["metrics"][best]["rmse"]
        print(f"  [OK] Best: {best}  |  Val RMSE: {best_rmse:,.0f}  |  Time: {elapsed:.1f}s")

    # Step 3: Save registry
    print("\n[3/3] Saving model registry...")
    save_registry(registry, "models/model_registry.json")

    print("\n" + "=" * 60)
    print("  Training complete!")
    print(f"  States trained: {len(registry)}")
    best_counts = {}
    for v in registry.values():
        bm = v["best_model"]
        best_counts[bm] = best_counts.get(bm, 0) + 1
    print(f"  Best model distribution: {best_counts}")
    print("=" * 60)
    print("\nTo start API: uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train forecasting models")
    parser.add_argument(
        "--states", nargs="*", default=None,
        help="Specific states to train (default: all)",
    )
    parser.add_argument(
        "--val-weeks", type=int, default=16,
        help="Validation window size in weeks (default: 16)",
    )
    args = parser.parse_args()
    main(states_filter=args.states, val_weeks=args.val_weeks)
