import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import json
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
import optuna
from scipy import stats
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── config ────────────────────────────────────────────────
DATA    = Path("data/master_features_v2.parquet")
OUT_DIR = Path("models")
RES_DIR = Path("outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)
RES_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_END   = "2024-12-31"
VAL_END     = "2025-09-30"
RANDOM_SEED = 42
N_TRIALS    = 200
TARGET      = 'solar_mw'

DAYAHEAD_FEATURES = [
    'shortwave_radiation_mean','direct_radiation_mean','diffuse_radiation_mean',
    'direct_normal_irradiance_mean','global_tilted_irradiance_mean',
    'cloud_cover_mean','temperature_2m_mean','wind_speed_10m_mean','precipitation_mean',
    'shortwave_radiation_p10','shortwave_radiation_p90','cloud_cover_p10','cloud_cover_p90',
    'clearness_index','solar_elevation','solar_declination','daylight_flag',
    'radiation_roll3h_mean','radiation_roll24h_mean','cloud_roll3h_mean',
    'sin_hour','cos_hour','sin_doy','cos_doy','hour','month','day_of_year','is_weekend',
    'installed_capacity_mw','price_eur_mwh','gen_forecast_mw','solar_lag_168h',
]
# ─────────────────────────────────────────────────────────

np.random.seed(RANDOM_SEED)
print("Loading data...")
df = pd.read_parquet(DATA)
df.index = pd.to_datetime(df.index, utc=True)
df = df.sort_index()
df['persistence_24h'] = df[TARGET].shift(24)

train = df[df.index <= TRAIN_END]
val   = df[(df.index > TRAIN_END) & (df.index <= VAL_END)]
test  = df[df.index > VAL_END]
print(f"Train {len(train)} | Val {len(val)} | Test {len(test)}")

X_train, y_train = train[DAYAHEAD_FEATURES].fillna(0), train[TARGET].fillna(0)
X_val,   y_val   = val[DAYAHEAD_FEATURES].fillna(0),   val[TARGET].fillna(0)
X_test,  y_test  = test[DAYAHEAD_FEATURES].fillna(0),  test[TARGET].fillna(0)
dl_test = test['daylight_flag'].values.astype(bool)

def metrics(y_true, y_pred):
    y_true = np.asarray(y_true, float); y_pred = np.asarray(y_pred, float)
    mae  = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    mape = np.mean(np.abs((y_true[dl_test]-y_pred[dl_test])/(y_true[dl_test]+1e-6)))*100
    return mae, rmse, mape

def dm_test(y_true, pa, pb):
    y_true = np.asarray(y_true, float)
    d = (y_true-np.asarray(pa,float))**2 - (y_true-np.asarray(pb,float))**2
    n = len(d); dbar = d.mean()
    g0 = np.var(d, ddof=0)
    g1 = np.mean((d[1:]-dbar)*(d[:-1]-dbar)) if n>1 else 0
    var = (g0 + 2*g1)/n
    if var <= 0: return np.nan, np.nan
    dm = dbar/np.sqrt(var)
    return dm, 2*(1-stats.norm.cdf(abs(dm)))

results = {}
preds = {}

# ── persistence reference ─────────────────────────────────
persist_test = test['persistence_24h'].fillna(0).values.clip(0)
rmse_persist = np.sqrt(np.mean((y_test.values - persist_test)**2))
mae,rmse,mape = metrics(y_test, persist_test)
results['Persistence_24h'] = {'MAE':mae,'RMSE':rmse,'MAPE':mape,'Skill':0.0}
preds['Persistence_24h'] = persist_test

# ── XGBoost ───────────────────────────────────────────────
print(f"\nTuning XGBoost ({N_TRIALS} trials)...")
def xgb_obj(trial):
    p = {
        'objective':'reg:squarederror','random_state':RANDOM_SEED,'n_jobs':4,
        'n_estimators':trial.suggest_int('n_estimators',200,2000),
        'learning_rate':trial.suggest_float('learning_rate',0.005,0.1,log=True),
        'max_depth':trial.suggest_int('max_depth',3,12),
        'min_child_weight':trial.suggest_int('min_child_weight',1,20),
        'subsample':trial.suggest_float('subsample',0.5,1.0),
        'colsample_bytree':trial.suggest_float('colsample_bytree',0.5,1.0),
        'reg_alpha':trial.suggest_float('reg_alpha',1e-8,10,log=True),
        'reg_lambda':trial.suggest_float('reg_lambda',1e-8,10,log=True),
    }
    m = xgb.XGBRegressor(**p, early_stopping_rounds=50, verbosity=0)
    m.fit(X_train,y_train,eval_set=[(X_val,y_val)],verbose=False)
    return np.mean(np.abs(y_val - m.predict(X_val).clip(0)))
st = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED))
st.optimize(xgb_obj, n_trials=N_TRIALS, show_progress_bar=False)
bp = st.best_params; bp.update({'objective':'reg:squarederror','random_state':RANDOM_SEED,'n_jobs':4})
mx = xgb.XGBRegressor(**bp, early_stopping_rounds=50, verbosity=0)
mx.fit(X_train,y_train,eval_set=[(X_val,y_val)],verbose=False)
px = mx.predict(X_test).clip(0)
mae,rmse,mape = metrics(y_test, px)
results['XGBoost'] = {'MAE':mae,'RMSE':rmse,'MAPE':mape,'Skill':1-rmse/rmse_persist}
preds['XGBoost'] = px
joblib.dump(mx, OUT_DIR/"xgboost_v2.joblib")
print(f"XGBoost test MAE {mae:.2f}, skill {1-rmse/rmse_persist:.3f}")

# ── CatBoost ──────────────────────────────────────────────
print(f"\nTuning CatBoost ({N_TRIALS} trials)...")
def cat_obj(trial):
    p = {
        'loss_function':'MAE','random_seed':RANDOM_SEED,'thread_count':4,'verbose':0,
        'iterations':trial.suggest_int('iterations',200,2000),
        'learning_rate':trial.suggest_float('learning_rate',0.005,0.1,log=True),
        'depth':trial.suggest_int('depth',3,10),
        'l2_leaf_reg':trial.suggest_float('l2_leaf_reg',1e-3,10,log=True),
        'subsample':trial.suggest_float('subsample',0.5,1.0),
    }
    m = CatBoostRegressor(**p)
    m.fit(X_train,y_train,eval_set=(X_val,y_val),early_stopping_rounds=50,verbose=0)
    return np.mean(np.abs(y_val - m.predict(X_val).clip(0)))
st = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED))
st.optimize(cat_obj, n_trials=N_TRIALS, show_progress_bar=False)
bp = st.best_params; bp.update({'loss_function':'MAE','random_seed':RANDOM_SEED,'thread_count':4,'verbose':0})
mc = CatBoostRegressor(**bp)
mc.fit(X_train,y_train,eval_set=(X_val,y_val),early_stopping_rounds=50,verbose=0)
pc = mc.predict(X_test).clip(0)
mae,rmse,mape = metrics(y_test, pc)
results['CatBoost'] = {'MAE':mae,'RMSE':rmse,'MAPE':mape,'Skill':1-rmse/rmse_persist}
preds['CatBoost'] = pc
mc.save_model(str(OUT_DIR/"catboost_v2.cbm"))
print(f"CatBoost test MAE {mae:.2f}, skill {1-rmse/rmse_persist:.3f}")

# ── LightGBM (reload existing saved model) ────────────────
print("\nReloading existing LightGBM...")
ml = joblib.load(OUT_DIR/"lgbm_dayahead_v2.joblib")
pl = ml.predict(X_test).clip(0)
mae,rmse,mape = metrics(y_test, pl)
results['LightGBM'] = {'MAE':mae,'RMSE':rmse,'MAPE':mape,'Skill':1-rmse/rmse_persist}
preds['LightGBM'] = pl
print(f"LightGBM test MAE {mae:.2f}, skill {1-rmse/rmse_persist:.3f}")

# ── DM tests between the three GBMs ───────────────────────
print("\nDiebold-Mariano tests (squared-error loss):")
dm_results = {}
for a,b in [('LightGBM','XGBoost'),('LightGBM','CatBoost'),('XGBoost','CatBoost')]:
    dm,p = dm_test(y_test, preds[a], preds[b])
    dm_results[f"{a}_vs_{b}"] = {'DM_stat':dm,'p_value':p}
    print(f"  {a} vs {b}: DM={dm:.3f}, p={p:.4f}")

# ── save comparison table ─────────────────────────────────
table = pd.DataFrame(results).T
table = table[['MAE','RMSE','MAPE','Skill']].round(3)
table = table.sort_values('MAE')
table.to_csv(RES_DIR/"model_comparison.csv")
print("\n===== MODEL COMPARISON (test set) =====")
print(table.to_string())

with open(RES_DIR/"model_comparison_dm.json","w") as f:
    json.dump(dm_results, f, indent=2, default=float)

# ── comparison figure ─────────────────────────────────────
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
order = table.index.tolist()
fig, ax = plt.subplots(figsize=(9,5))
x = np.arange(len(order)); w=0.35
ax.bar(x-w/2, table['MAE'], w, label='MAE', color='#1D9E75')
ax.bar(x+w/2, table['RMSE'], w, label='RMSE', color='#0C447C')
ax.set_xticks(x); ax.set_xticklabels(order, rotation=15)
ax.set_ylabel('Error (MW)'); ax.set_title('Model Comparison - Day-Ahead (Test Set)')
ax.legend()
for i,(m,r) in enumerate(zip(table['MAE'],table['RMSE'])):
    ax.annotate(f'{m:.0f}',(i-w/2,m),ha='center',va='bottom',fontsize=8)
    ax.annotate(f'{r:.0f}',(i+w/2,r),ha='center',va='bottom',fontsize=8)
fig.tight_layout()
fig.savefig(RES_DIR/"figures/fig8_model_comparison_full.png", bbox_inches='tight')
print("\nSaved model_comparison.csv and fig8_model_comparison_full.png")
