import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATA      = Path("data/master_features_v2.parquet")
MODEL_DIR = Path("models")
RES_DIR   = Path("outputs")
FIG_DIR   = Path("outputs/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_END = "2024-12-31"
VAL_END   = "2025-09-30"
TARGET    = 'solar_mw'
ALPHA     = 0.20

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

print("Loading data and quantile models...")
df = pd.read_parquet(DATA)
df.index = pd.to_datetime(df.index, utc=True)
df = df.sort_index()
val  = df[(df.index > TRAIN_END) & (df.index <= VAL_END)]
test = df[df.index > VAL_END]

m_p10 = joblib.load(MODEL_DIR / "lgbm_p10_v2.joblib")
m_p50 = joblib.load(MODEL_DIR / "lgbm_dayahead_v2.joblib")
m_p90 = joblib.load(MODEL_DIR / "lgbm_p90_v2.joblib")

def predict_all(frame):
    X = frame[DAYAHEAD_FEATURES].fillna(0)
    p10 = m_p10.predict(X).clip(0); p50 = m_p50.predict(X).clip(0); p90 = m_p90.predict(X).clip(0)
    q = np.sort(np.column_stack([p10,p50,p90]), axis=1)
    return q[:,0], q[:,1], q[:,2]

v_p10,v_p50,v_p90 = predict_all(val)
t_p10,t_p50,t_p90 = predict_all(test)
y_val  = val[TARGET].fillna(0).values
y_test = test[TARGET].fillna(0).values
dl_val  = val['daylight_flag'].values.astype(bool)
dl_test = test['daylight_flag'].values.astype(bool)
ci_val  = val['clearness_index'].fillna(0).values
ci_test = test['clearness_index'].fillna(0).values

# regime tertile cuts from DAYLIGHT validation clearness
q1, q2 = np.quantile(ci_val[dl_val], [1/3, 2/3])
print(f"Clearness tertile cuts (daylight val): {q1:.3f}, {q2:.3f}")
def regime(ci):
    r = np.empty(len(ci), dtype=object)
    r[ci <= q1] = 'overcast'
    r[(ci > q1) & (ci <= q2)] = 'partly'
    r[ci > q2] = 'clear'
    return r
reg_val  = regime(ci_val)
reg_test = regime(ci_test)

# conformity scores on validation
E_val = np.maximum(v_p10 - y_val, y_val - v_p90)

# per-regime conformal Q (daylight only)
Q = {}
for rg in ['overcast','partly','clear']:
    mask = dl_val & (reg_val == rg)
    scores = E_val[mask]
    n = len(scores)
    k = min(int(np.ceil((n+1)*(1-ALPHA))), n)
    Q[rg] = float(np.sort(scores)[k-1])
    print(f"  Q[{rg}] = {Q[rg]:.2f} MW  (n_cal={n})")

# also global Q for comparison
sc_all = E_val[dl_val]; n=len(sc_all); k=min(int(np.ceil((n+1)*(1-ALPHA))),n)
Q_global = float(np.sort(sc_all)[k-1])

# apply per-regime correction to test
t_p10_cond = t_p10.copy(); t_p90_cond = t_p90.copy()
for rg in ['overcast','partly','clear']:
    mask = reg_test == rg
    t_p10_cond[mask] = np.clip(t_p10[mask] - Q[rg], 0, None)
    t_p90_cond[mask] = t_p90[mask] + Q[rg]

# global correction (for comparison)
t_p10_glob = np.clip(t_p10 - Q_global, 0, None)
t_p90_glob = t_p90 + Q_global

def cw(y, lo, hi, mask):
    inside = ((y>=lo)&(y<=hi))[mask].mean()
    rng = y[mask].max()-y[mask].min()
    width = np.mean((hi-lo)[mask])/(rng+1e-9)
    return inside, width

print("\n===== CONDITIONAL vs GLOBAL CONFORMAL (daylight test) =====")
res = {}
picp_raw, pinaw_raw   = cw(y_test, t_p10, t_p90, dl_test)
picp_glob, pinaw_glob = cw(y_test, t_p10_glob, t_p90_glob, dl_test)
picp_cond, pinaw_cond = cw(y_test, t_p10_cond, t_p90_cond, dl_test)
print(f"  Raw:         PICP {picp_raw*100:5.1f}%  PINAW {pinaw_raw:.3f}")
print(f"  Global CQR:  PICP {picp_glob*100:5.1f}%  PINAW {pinaw_glob:.3f}")
print(f"  Conditional: PICP {picp_cond*100:5.1f}%  PINAW {pinaw_cond:.3f}")

# per-regime coverage of the conditional method
print("\n  Per-regime coverage (conditional, daylight):")
per_regime = {}
for rg in ['overcast','partly','clear']:
    mask = dl_test & (reg_test == rg)
    if mask.sum() > 0:
        inside = ((y_test>=t_p10_cond)&(y_test<=t_p90_cond))[mask].mean()
        per_regime[rg] = {'PICP': float(inside), 'n': int(mask.sum())}
        print(f"    {rg:9s}: {inside*100:5.1f}%  (n={mask.sum()})")

res = {
    "tertile_cuts": [float(q1), float(q2)],
    "Q_per_regime": Q, "Q_global": Q_global,
    "raw":        {"PICP": float(picp_raw),  "PINAW": float(pinaw_raw)},
    "global_cqr": {"PICP": float(picp_glob), "PINAW": float(pinaw_glob)},
    "conditional":{"PICP": float(picp_cond), "PINAW": float(pinaw_cond)},
    "conditional_per_regime": per_regime,
}
with open(RES_DIR/"conformal_conditional_results.json","w") as f:
    json.dump(res, f, indent=2)

out = pd.DataFrame({
    'actual': y_test, 'p50': t_p50,
    'p10_cond': t_p10_cond, 'p90_cond': t_p90_cond,
    'regime': reg_test,
}, index=test.index)
out.to_parquet(RES_DIR/"predictions_conformal_conditional.parquet")

# figure: coverage comparison bar
fig, ax = plt.subplots(figsize=(8,5))
methods = ['Raw','Global CQR','Conditional']
picps = [picp_raw*100, picp_glob*100, picp_cond*100]
widths = [pinaw_raw, pinaw_glob, pinaw_cond]
x = np.arange(len(methods))
ax.bar(x, picps, color=['#C0392B','#E67E22','#1D9E75'])
ax.axhline(80, color='black', ls='--', lw=1.2, label='Target 80%')
ax.set_xticks(x); ax.set_xticklabels(methods)
ax.set_ylabel('Daylight coverage (%)')
ax.set_title('Interval Coverage by Calibration Method (test set)')
for i,p in enumerate(picps):
    ax.annotate(f'{p:.0f}%\nPINAW {widths[i]:.2f}', (i,p), ha='center', va='bottom', fontsize=8)
ax.legend()
fig.tight_layout()
fig.savefig(FIG_DIR/"fig10_conditional_conformal.png", bbox_inches='tight')
print("\nSaved conformal_conditional_results.json, predictions_conformal_conditional.parquet, fig10_conditional_conformal.png")
