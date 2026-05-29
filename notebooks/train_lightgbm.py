import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import json
import lightgbm as lgb
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── config ────────────────────────────────────────────────
DATA    = Path("data/master_features.parquet")
OUT_DIR = Path("models")
RES_DIR = Path("outputs")

TRAIN_END   = "2024-12-31"
VAL_END     = "2025-09-30"
RANDOM_SEED = 42
N_TRIALS    = 200

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

X_train = train[FEATURES].fillna(0)
y_train = train[TARGET].fillna(0)
X_val   = val[FEATURES].fillna(0)
y_val   = val[TARGET].fillna(0)
X_test  = test[FEATURES].fillna(0)
y_test  = test[TARGET].fillna(0)

def evaluate(y_true, y_pred, name, daylight_flag):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mae  = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    day  = np.array(daylight_flag).astype(bool)
    mape = np.mean(np.abs((y_true[day] - y_pred[day]) / (y_true[day] + 1e-6))) * 100
    bias = np.mean(y_pred - y_true)
    print(f"\n{name}")
    print(f"  MAE:  {mae:.2f} MW")
    print(f"  RMSE: {rmse:.2f} MW")
    print(f"  MAPE: {mape:.2f}% (daylight only)")
    print(f"  Bias: {bias:.2f} MW")
    return {"model": name, "MAE": mae, "RMSE": rmse, "MAPE": mape, "Bias": bias}

print(f"\nRunning Optuna with {N_TRIALS} trials...")

def objective(trial):
    params = {
        'objective':         'regression',
        'metric':            'mae',
        'verbosity':         -1,
        'random_state':      RANDOM_SEED,
        'n_estimators':      trial.suggest_int('n_estimators', 200, 2000),
        'learning_rate':     trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
        'num_leaves':        trial.suggest_int('num_leaves', 20, 300),
        'max_depth':         trial.suggest_int('max_depth', 3, 12),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
        'subsample':         trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree':  trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha':         trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda':        trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
    }
    model = lgb.LGBMRegressor(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)]
    )
    val_pred = model.predict(X_val).clip(0)
    return np.mean(np.abs(y_val - val_pred))

study = optuna.create_study(
    direction='minimize',
    sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED)
)
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)

print(f"\nBest trial MAE: {study.best_value:.2f} MW")
print(f"Best params: {study.best_params}")

print("\nTraining final LightGBM model with best parameters...")
best_params = study.best_params
best_params.update({
    'objective':    'regression',
    'metric':       'mae',
    'verbosity':    -1,
    'random_state': RANDOM_SEED,
})

model = lgb.LGBMRegressor(**best_params)
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)]
)

train_pred = model.predict(X_train).clip(0)
val_pred   = model.predict(X_val).clip(0)
test_pred  = model.predict(X_test).clip(0)

results = []
results.append(evaluate(y_train, train_pred, "LightGBM (Train)", train['daylight_flag'].values))
results.append(evaluate(y_val,   val_pred,   "LightGBM (Val)",   val['daylight_flag'].values))
results.append(evaluate(y_test,  test_pred,  "LightGBM (Test)",  test['daylight_flag'].values))

importance = pd.DataFrame({
    'feature':    FEATURES,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)
print(f"\nTop 10 features:\n{importance.head(10).to_string(index=False)}")

joblib.dump(model, OUT_DIR / "lgbm_model.joblib")
importance.to_csv(RES_DIR / "feature_importance.csv", index=False)

pred_df = pd.DataFrame({
    'actual':    np.concatenate([y_train, y_val, y_test]),
    'pred_lgbm': np.concatenate([train_pred, val_pred, test_pred])
}, index=pd.concat([train, val, test]).index)
pred_df.to_parquet(RES_DIR / "predictions_lgbm.parquet")

with open(RES_DIR / "metrics_lgbm.json", "w") as f:
    json.dump(results, f, indent=2)

with open(RES_DIR / "best_params_lgbm.json", "w") as f:
    json.dump(study.best_params, f, indent=2)

print("\nLightGBM training complete.")
print(f"Model saved to {OUT_DIR}/lgbm_model.joblib")
print(f"Metrics saved to {RES_DIR}/metrics_lgbm.json")
