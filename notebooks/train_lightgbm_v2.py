import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import json
import lightgbm as lgb
import optuna
from scipy import stats
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── config ────────────────────────────────────────────────
DATA    = Path("data/master_features_v2.parquet")
OUT_DIR = Path("models")
RES_DIR = Path("outputs")

TRAIN_END   = "2024-12-31"
VAL_END     = "2025-09-30"
RANDOM_SEED = 42
N_TRIALS    = 200
TARGET      = 'solar_mw'

# ── Day-ahead feature set ─────────────────────────────────
# ONLY information available 24h before delivery.
# Weather vars are Open-Meteo FORECASTS (available day-ahead).
# Day-ahead price and ENTSO-E generation forecast are published pre-delivery.
DAYAHEAD_FEATURES = [
    # weather forecast (mean across CZ)
    'shortwave_radiation_mean', 'direct_radiation_mean', 'diffuse_radiation_mean',
    'direct_normal_irradiance_mean', 'global_tilted_irradiance_mean',
    'cloud_cover_mean', 'temperature_2m_mean', 'wind_speed_10m_mean', 'precipitation_mean',
    # weather spread across country (p10/p90)
    'shortwave_radiation_p10', 'shortwave_radiation_p90',
    'cloud_cover_p10', 'cloud_cover_p90',
    # derived weather / solar geometry
    'clearness_index', 'solar_elevation', 'solar_declination', 'daylight_flag',
    'radiation_roll3h_mean', 'radiation_roll24h_mean', 'cloud_roll3h_mean',
    # time / calendar
    'sin_hour', 'cos_hour', 'sin_doy', 'cos_doy',
    'hour', 'month', 'day_of_year', 'is_weekend',
    # known-in-advance system context
    'installed_capacity_mw',     # known
    'price_eur_mwh',             # day-ahead market price (published pre-delivery)
    'gen_forecast_mw',           # ENTSO-E day-ahead generation forecast
    'solar_lag_168h',            # generation one week ago (known)
]

# EXCLUDED at day-ahead (actual values not known 24h ahead — would be leakage):
#   load_mw, wind_mw, hydro_mw, nuclear_mw, flow_de_mw, flow_at_mw, flow_sk_mw, flow_pl_mw
#   solar_lag_1h, solar_lag_24h, capacity_factor

# Intraday secondary comparison MAY use recent lags:
INTRADAY_EXTRA = ['solar_lag_1h', 'solar_lag_24h']
# ─────────────────────────────────────────────────────────

OUT_DIR.mkdir(parents=True, exist_ok=True)
RES_DIR.mkdir(parents=True, exist_ok=True)
np.random.seed(RANDOM_SEED)

print("Loading master features v2...")
df = pd.read_parquet(DATA)
df.index = pd.to_datetime(df.index, utc=True)
df = df.sort_index()

# day-ahead persistence reference: same hour, 24h earlier (known at forecast time)
df['persistence_24h'] = df[TARGET].shift(24)

train = df[df.index <= TRAIN_END]
val   = df[(df.index > TRAIN_END) & (df.index <= VAL_END)]
test  = df[df.index > VAL_END]
print(f"Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")

# ── metric helpers ────────────────────────────────────────
def metrics(y_true, y_pred, daylight):
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    day = np.asarray(daylight).astype(bool)
    mae  = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    if day.sum() > 0:
        mape = np.mean(np.abs((y_true[day] - y_pred[day]) /
                              (y_true[day] + 1e-6))) * 100
    else:
        mape = np.nan
    bias = np.mean(y_pred - y_true)
    return {"MAE": mae, "RMSE": rmse, "MAPE_daylight": mape, "Bias": bias}

def skill_score(rmse_model, rmse_ref):
    return 1.0 - (rmse_model / rmse_ref)

def picp_pinaw(y_true, lower, upper):
    y_true = np.asarray(y_true, float)
    lower  = np.asarray(lower, float)
    upper  = np.asarray(upper, float)
    inside = (y_true >= lower) & (y_true <= upper)
    picp = inside.mean()
    rng = y_true.max() - y_true.min()
    pinaw = np.mean(upper - lower) / (rng + 1e-9)
    return picp, pinaw

def zero_output_accuracy(y_pred, daylight, tol=10.0):
    # fraction of night hours predicted ~zero
    night = ~np.asarray(daylight).astype(bool)
    if night.sum() == 0:
        return np.nan
    return (np.abs(np.asarray(y_pred, float)[night]) <= tol).mean()

def diebold_mariano(y_true, pred_a, pred_b, loss='squared'):
    # tests whether model A and B differ significantly in accuracy
    y_true = np.asarray(y_true, float)
    e_a = y_true - np.asarray(pred_a, float)
    e_b = y_true - np.asarray(pred_b, float)
    if loss == 'squared':
        d = e_a**2 - e_b**2
    else:
        d = np.abs(e_a) - np.abs(e_b)
    n = len(d)
    d_bar = d.mean()
    # Newey-West variance (lag 1 for robustness)
    gamma0 = np.var(d, ddof=0)
    gamma1 = np.mean((d[1:] - d_bar) * (d[:-1] - d_bar)) if n > 1 else 0.0
    var_d = (gamma0 + 2 * gamma1) / n
    if var_d <= 0:
        return np.nan, np.nan
    dm_stat = d_bar / np.sqrt(var_d)
    p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))
    return dm_stat, p_value

# ── feature matrices (day-ahead) ──────────────────────────
def Xy(frame, feats):
    X = frame[feats].fillna(0)
    y = frame[TARGET].fillna(0)
    return X, y

X_train, y_train = Xy(train, DAYAHEAD_FEATURES)
X_val,   y_val   = Xy(val,   DAYAHEAD_FEATURES)
X_test,  y_test  = Xy(test,  DAYAHEAD_FEATURES)

# ── Optuna tuning (point forecast, MAE) ───────────────────
print(f"\nOptuna day-ahead point model, {N_TRIALS} trials...")
def objective(trial):
    params = {
        'objective':'regression','metric':'mae','verbosity':-1,
        'random_state':RANDOM_SEED,'n_jobs':4,
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
    m = lgb.LGBMRegressor(**params)
    m.fit(X_train, y_train, eval_set=[(X_val, y_val)],
          callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
    return np.mean(np.abs(y_val - m.predict(X_val).clip(0)))

study = optuna.create_study(direction='minimize',
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED))
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)
print(f"Best val MAE: {study.best_value:.2f} MW")

best = study.best_params
best.update({'objective':'regression','metric':'mae','verbosity':-1,'random_state':RANDOM_SEED})

# final point model (p50)
model = lgb.LGBMRegressor(**best)
model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
          callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
p50_test = model.predict(X_test).clip(0)

# ── quantile models p10 / p90 ─────────────────────────────
print("Training quantile models p10 and p90...")
def quantile_model(alpha):
    qp = dict(best); qp.update({'objective':'quantile','alpha':alpha,'metric':'quantile'})
    qm = lgb.LGBMRegressor(**qp)
    qm.fit(X_train, y_train, eval_set=[(X_val, y_val)],
           callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
    return qm

m_p10 = quantile_model(0.1)
m_p90 = quantile_model(0.9)
p10_test = m_p10.predict(X_test).clip(0)
p90_test = m_p90.predict(X_test).clip(0)
# enforce ordering
p10_test = np.minimum(p10_test, p50_test)
p90_test = np.maximum(p90_test, p50_test)

# ── references ────────────────────────────────────────────
persist_test = test['persistence_24h'].fillna(0).values.clip(0)

# physical baseline (same formula as v1)
def physical(frame):
    irr = frame['shortwave_radiation_mean'] / 1000
    temp = 1 - 0.004 * (frame['temperature_2m_mean'] - 25)
    return (frame['installed_capacity_mw'] * irr * temp * 0.15).clip(lower=0)
phys_test = physical(test).values

# ── evaluation ────────────────────────────────────────────
dl_test = test['daylight_flag'].values
results = {}
results['LightGBM_dayahead'] = metrics(y_test, p50_test, dl_test)
results['Persistence_24h']   = metrics(y_test, persist_test, dl_test)
results['Physical_baseline'] = metrics(y_test, phys_test, dl_test)

# skill scores vs persistence
results['LightGBM_dayahead']['SkillScore_vs_persistence'] = skill_score(
    results['LightGBM_dayahead']['RMSE'], results['Persistence_24h']['RMSE'])
results['Physical_baseline']['SkillScore_vs_persistence'] = skill_score(
    results['Physical_baseline']['RMSE'], results['Persistence_24h']['RMSE'])

# interval metrics
picp, pinaw = picp_pinaw(y_test, p10_test, p90_test)
results['LightGBM_dayahead']['PICP_p10_p90'] = picp
results['LightGBM_dayahead']['PINAW_p10_p90'] = pinaw

# zero-output accuracy
results['LightGBM_dayahead']['ZeroOutputAcc'] = zero_output_accuracy(p50_test, dl_test)

# Diebold-Mariano: LightGBM vs persistence, and vs physical
dm1, p1 = diebold_mariano(y_test, p50_test, persist_test, 'squared')
dm2, p2 = diebold_mariano(y_test, p50_test, phys_test, 'squared')
results['DM_LightGBM_vs_persistence'] = {"DM_stat": dm1, "p_value": p1}
results['DM_LightGBM_vs_physical']    = {"DM_stat": dm2, "p_value": p2}

# ── intraday secondary comparison ─────────────────────────
print("\nIntraday secondary comparison (recent lags allowed)...")
INTRA_FEATURES = DAYAHEAD_FEATURES + INTRADAY_EXTRA
Xi_tr, yi_tr = Xy(train, INTRA_FEATURES)
Xi_va, yi_va = Xy(val,   INTRA_FEATURES)
Xi_te, yi_te = Xy(test,  INTRA_FEATURES)
m_intra = lgb.LGBMRegressor(**best)
m_intra.fit(Xi_tr, yi_tr, eval_set=[(Xi_va, yi_va)],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
intra_test = m_intra.predict(Xi_te).clip(0)
results['LightGBM_intraday'] = metrics(yi_te, intra_test, dl_test)
results['LightGBM_intraday']['SkillScore_vs_persistence'] = skill_score(
    results['LightGBM_intraday']['RMSE'], results['Persistence_24h']['RMSE'])

# ── print summary ─────────────────────────────────────────
print("\n===== DAY-AHEAD RESULTS (TEST) =====")
for k, v in results.items():
    print(f"\n{k}")
    for mk, mv in v.items():
        print(f"  {mk}: {mv:.4f}" if isinstance(mv, float) else f"  {mk}: {mv}")

# feature importance (day-ahead point model)
importance = pd.DataFrame({'feature': DAYAHEAD_FEATURES,
                           'importance': model.feature_importances_}
                          ).sort_values('importance', ascending=False)
print("\nTop 10 day-ahead features:")
print(importance.head(10).to_string(index=False))

# ── save ──────────────────────────────────────────────────
joblib.dump(model, OUT_DIR / "lgbm_dayahead_v2.joblib")
joblib.dump(m_p10, OUT_DIR / "lgbm_p10_v2.joblib")
joblib.dump(m_p90, OUT_DIR / "lgbm_p90_v2.joblib")
importance.to_csv(RES_DIR / "feature_importance_v2.csv", index=False)

pred_df = pd.DataFrame({
    'actual': y_test.values,
    'p10': p10_test, 'p50': p50_test, 'p90': p90_test,
    'persistence': persist_test, 'physical': phys_test,
    'intraday': intra_test,
}, index=test.index)
pred_df.to_parquet(RES_DIR / "predictions_v2.parquet")

with open(RES_DIR / "metrics_v2.json", "w") as f:
    json.dump(results, f, indent=2, default=float)
with open(RES_DIR / "best_params_v2.json", "w") as f:
    json.dump(study.best_params, f, indent=2)

print("\nDone. Models, predictions and metrics saved with _v2 suffix.")
