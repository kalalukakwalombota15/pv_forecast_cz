import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import json

# ── config ────────────────────────────────────────────────
DATA    = Path("data/master_features.parquet")
OUT_DIR = Path("models")
RES_DIR = Path("outputs")

TRAIN_END = "2024-12-31"
VAL_END   = "2025-09-30"
# ─────────────────────────────────────────────────────────

OUT_DIR.mkdir(parents=True, exist_ok=True)
RES_DIR.mkdir(parents=True, exist_ok=True)

print("Loading master features...")
df = pd.read_parquet(DATA)
df.index = pd.to_datetime(df.index, utc=True)

train = df[df.index <= TRAIN_END]
val   = df[(df.index > TRAIN_END) & (df.index <= VAL_END)]
test  = df[df.index > VAL_END]

print(f"Train: {len(train)} rows | Val: {len(val)} rows | Test: {len(test)} rows")

TARGET = 'solar_mw'

def evaluate(y_true, y_pred, name, daylight_flag):
    mae  = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    day  = daylight_flag.astype(bool)
    mape = np.mean(np.abs((y_true[day] - y_pred[day]) / (y_true[day] + 1e-6))) * 100
    bias = np.mean(y_pred - y_true)
    print(f"\n{name}")
    print(f"  MAE:  {mae:.2f} MW")
    print(f"  RMSE: {rmse:.2f} MW")
    print(f"  MAPE: {mape:.2f}% (daylight only)")
    print(f"  Bias: {bias:.2f} MW")
    return {"model": name, "MAE": mae, "RMSE": rmse, "MAPE": mape, "Bias": bias}

# ── Physical Baseline ─────────────────────────────────────
print("\nTraining Physical Baseline...")

INSTALLED_CAPACITY_MW = 4500  # Czech Republic approx installed PV capacity 2024

def physical_baseline(df):
    irradiance_factor = df['shortwave_radiation_mean'] / 1000
    temperature_factor = 1 - 0.004 * (df['temperature_2m_mean'] - 25)
    efficiency_factor = 0.15
    pred = INSTALLED_CAPACITY_MW * irradiance_factor * temperature_factor * efficiency_factor
    return pred.clip(lower=0)

train_pred_phys = physical_baseline(train)
val_pred_phys   = physical_baseline(val)
test_pred_phys  = physical_baseline(test)

results = []
results.append(evaluate(train[TARGET], train_pred_phys, "Physical Baseline (Train)", train['daylight_flag']))
results.append(evaluate(val[TARGET],   val_pred_phys,   "Physical Baseline (Val)",   val['daylight_flag']))
results.append(evaluate(test[TARGET],  test_pred_phys,  "Physical Baseline (Test)",  test['daylight_flag']))

# save predictions
pred_df = pd.DataFrame({
    'actual':     df[TARGET],
    'pred_phys':  pd.concat([train_pred_phys, val_pred_phys, test_pred_phys])
})
pred_df.to_parquet(RES_DIR / "predictions_baseline.parquet")

# save metrics
with open(RES_DIR / "metrics_baseline.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nPhysical baseline complete.")
print(f"Predictions saved to {RES_DIR}/predictions_baseline.parquet")
print(f"Metrics saved to {RES_DIR}/metrics_baseline.json")
