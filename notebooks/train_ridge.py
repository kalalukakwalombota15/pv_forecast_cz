import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import json

# ── config ────────────────────────────────────────────────
DATA    = Path("data/master_features.parquet")
OUT_DIR = Path("models")
RES_DIR = Path("outputs")

TRAIN_END   = "2024-12-31"
VAL_END     = "2025-09-30"
RANDOM_SEED = 42

FEATURES = [
    'shortwave_radiation_mean', 'direct_radiation_mean', 'diffuse_radiation_mean',
    'cloud_cover_mean', 'temperature_2m_mean', 'wind_speed_10m_mean',
    'shortwave_radiation_p10', 'shortwave_radiation_p90',
    'cloud_cover_p10', 'cloud_cover_p90',
    'clearness_index', 'solar_elevation', 'daylight_flag',
    'sin_hour', 'cos_hour', 'sin_doy', 'cos_doy',
    'hour', 'month', 'day_of_year', 'is_weekend',
    'load_mw', 'wind_mw', 'nuclear_mw', 'price_eur_mwh',
    'flow_de_mw', 'flow_at_mw', 'flow_sk_mw', 'flow_pl_mw'
]
TARGET = 'solar_mw'
# ─────────────────────────────────────────────────────────

OUT_DIR.mkdir(parents=True, exist_ok=True)
RES_DIR.mkdir(parents=True, exist_ok=True)

np.random.seed(RANDOM_SEED)

print("Loading master features...")
df = pd.read_parquet(DATA)
df.index = pd.to_datetime(df.index, utc=True)

train = df[df.index <= TRAIN_END]
val   = df[(df.index > TRAIN_END) & (df.index <= VAL_END)]
test  = df[df.index > VAL_END]

print(f"Train: {len(train)} rows | Val: {len(val)} rows | Test: {len(test)} rows")

X_train = train[FEATURES].fillna(0).values
y_train = train[TARGET].fillna(0).values
X_val   = val[FEATURES].fillna(0).values
y_val   = val[TARGET].fillna(0).values
X_test  = test[FEATURES].fillna(0).values
y_test  = test[TARGET].fillna(0).values

mean = X_train.mean(axis=0)
std  = X_train.std(axis=0) + 1e-8
X_train_s = (X_train - mean) / std
X_val_s   = (X_val   - mean) / std
X_test_s  = (X_test  - mean) / std

def fit_ridge(X, y, lam):
    A = np.column_stack([np.ones(len(X)), X])
    ATA = A.T @ A
    ATy = A.T @ y
    n = ATA.shape[0]
    coeffs = np.linalg.solve(ATA + lam * np.eye(n), ATy)
    return coeffs

def predict_ridge(X, coeffs):
    A = np.column_stack([np.ones(len(X)), X])
    return A @ coeffs

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

print("Tuning regularisation parameter lambda on validation set...")
lambdas = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
best_lam, best_val_mae = None, np.inf

for lam in lambdas:
    coeffs = fit_ridge(X_train_s, y_train, lam)
    val_pred = predict_ridge(X_val_s, coeffs).clip(0)
    val_mae = np.mean(np.abs(y_val - val_pred))
    print(f"  lambda={lam:.3f} -> Val MAE={val_mae:.2f} MW")
    if val_mae < best_val_mae:
        best_val_mae = val_mae
        best_lam = lam

print(f"\nBest lambda: {best_lam}")
coeffs = fit_ridge(X_train_s, y_train, best_lam)

train_pred = predict_ridge(X_train_s, coeffs).clip(0)
val_pred   = predict_ridge(X_val_s,   coeffs).clip(0)
test_pred  = predict_ridge(X_test_s,  coeffs).clip(0)

results = []
results.append(evaluate(y_train, train_pred, "Ridge (Train)", train['daylight_flag'].values))
results.append(evaluate(y_val,   val_pred,   "Ridge (Val)",   val['daylight_flag'].values))
results.append(evaluate(y_test,  test_pred,  "Ridge (Test)",  test['daylight_flag'].values))

model_data = {
    'coeffs': coeffs,
    'mean': mean,
    'std': std,
    'best_lambda': best_lam,
    'features': FEATURES
}
joblib.dump(model_data, OUT_DIR / "ridge_model.joblib")

pred_df = pd.DataFrame({
    'actual':     np.concatenate([y_train, y_val, y_test]),
    'pred_ridge': np.concatenate([train_pred, val_pred, test_pred])
}, index=pd.concat([train, val, test]).index)
pred_df.to_parquet(RES_DIR / "predictions_ridge.parquet")

with open(RES_DIR / "metrics_ridge.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nRidge regression complete.")
print(f"Model saved to {OUT_DIR}/ridge_model.joblib")
print(f"Metrics saved to {RES_DIR}/metrics_ridge.json")
