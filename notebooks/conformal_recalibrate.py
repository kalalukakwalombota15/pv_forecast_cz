import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── config ────────────────────────────────────────────────
DATA    = Path("data/master_features_v2.parquet")
MODEL_DIR = Path("models")
RES_DIR = Path("outputs")
FIG_DIR = Path("outputs/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_END = "2024-12-31"
VAL_END   = "2025-09-30"
TARGET    = 'solar_mw'
ALPHA     = 0.20   # nominal miscoverage; p10-p90 = 80% interval

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
    p10 = m_p10.predict(X).clip(0)
    p50 = m_p50.predict(X).clip(0)
    p90 = m_p90.predict(X).clip(0)
    # enforce monotonicity
    q = np.sort(np.column_stack([p10, p50, p90]), axis=1)
    return q[:,0], q[:,1], q[:,2]

# ── predictions on validation (for calibration) and test ──
v_p10, v_p50, v_p90 = predict_all(val)
t_p10, t_p50, t_p90 = predict_all(test)
y_val  = val[TARGET].fillna(0).values
y_test = test[TARGET].fillna(0).values
dl_val  = val['daylight_flag'].values.astype(bool)
dl_test = test['daylight_flag'].values.astype(bool)

# ── Conformalized Quantile Regression (CQR) ───────────────
# Conformity score: how far the actual falls outside [p10, p90].
# E_i = max(p10 - y, y - p90). Positive = outside the band.
# Calibrate on DAYLIGHT validation hours only.
E_val = np.maximum(v_p10 - y_val, y_val - v_p90)
E_val_daylight = E_val[dl_val]

n = len(E_val_daylight)
# conformal quantile of the scores at level (1 - alpha)
k = int(np.ceil((n + 1) * (1 - ALPHA)))
k = min(k, n)  # guard
Q = np.sort(E_val_daylight)[k - 1]
print(f"Conformal correction Q (daylight, level {1-ALPHA:.2f}): {Q:.2f} MW  (n_cal={n})")

# ── apply correction to test bands ────────────────────────
t_p10_cal = np.clip(t_p10 - Q, 0, None)
t_p90_cal = t_p90 + Q

def coverage_width(y, lo, hi, mask):
    inside = ((y >= lo) & (y <= hi))[mask].mean()
    rng = y[mask].max() - y[mask].min()
    width = np.mean((hi - lo)[mask]) / (rng + 1e-9)
    return inside, width

# before
picp_before, pinaw_before = coverage_width(y_test, t_p10, t_p90, dl_test)
# after
picp_after,  pinaw_after  = coverage_width(y_test, t_p10_cal, t_p90_cal, dl_test)

# also full (incl night) for reference
picp_before_all = ((y_test >= t_p10) & (y_test <= t_p90)).mean()
picp_after_all  = ((y_test >= t_p10_cal) & (y_test <= t_p90_cal)).mean()

print("\n===== CONFORMAL RECALIBRATION RESULTS (test set) =====")
print(f"  DAYLIGHT coverage before: {picp_before*100:.1f}%  (target 80%)")
print(f"  DAYLIGHT coverage after:  {picp_after*100:.1f}%")
print(f"  DAYLIGHT PINAW before:    {pinaw_before:.4f}")
print(f"  DAYLIGHT PINAW after:     {pinaw_after:.4f}")
print(f"  Full coverage before:     {picp_before_all*100:.1f}%")
print(f"  Full coverage after:      {picp_after_all*100:.1f}%")

results = {
    "conformal_Q_MW": float(Q),
    "n_calibration_daylight": int(n),
    "daylight_PICP_before": float(picp_before),
    "daylight_PICP_after": float(picp_after),
    "daylight_PINAW_before": float(pinaw_before),
    "daylight_PINAW_after": float(pinaw_after),
    "full_PICP_before": float(picp_before_all),
    "full_PICP_after": float(picp_after_all),
}
with open(RES_DIR / "conformal_results.json", "w") as f:
    json.dump(results, f, indent=2)

# save recalibrated predictions
out = pd.DataFrame({
    'actual': y_test,
    'p10_raw': t_p10, 'p50': t_p50, 'p90_raw': t_p90,
    'p10_cal': t_p10_cal, 'p90_cal': t_p90_cal,
}, index=test.index)
out.to_parquet(RES_DIR / "predictions_conformal.parquet")

# ── figure: before vs after coverage on a sample week ─────
sample = out.loc['2025-10-06':'2025-10-12']
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
ax1.fill_between(sample.index, sample['p10_raw'], sample['p90_raw'], color='#C0392B', alpha=0.2, label='Raw band')
ax1.plot(sample.index, sample['p50'], color='#1A1A1A', lw=1.3, label='p50')
ax1.plot(sample.index, sample['actual'], color='#1A1A1A', lw=1.8, ls='--', label='Actual')
ax1.set_title(f'Raw bands - daylight coverage {picp_before*100:.0f}%')
ax1.legend(fontsize=8); ax1.set_ylabel('MW')
ax2.fill_between(sample.index, sample['p10_cal'], sample['p90_cal'], color='#1D9E75', alpha=0.25, label='Conformal band')
ax2.plot(sample.index, sample['p50'], color='#1A1A1A', lw=1.3, label='p50')
ax2.plot(sample.index, sample['actual'], color='#1A1A1A', lw=1.8, ls='--', label='Actual')
ax2.set_title(f'Conformal-recalibrated bands - daylight coverage {picp_after*100:.0f}%')
ax2.legend(fontsize=8); ax2.set_ylabel('MW')
fig.tight_layout()
fig.savefig(FIG_DIR / "fig9_conformal_calibration.png", bbox_inches='tight')
print("\nSaved conformal_results.json, predictions_conformal.parquet, fig9_conformal_calibration.png")
