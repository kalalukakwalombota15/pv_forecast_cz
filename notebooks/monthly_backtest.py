import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATA    = Path("data/master_features_v2.parquet")
MODEL_DIR = Path("models")
RES_DIR = Path("outputs")
FIG_DIR = Path("outputs/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

TARGET = 'solar_mw'
VAL_END = "2025-09-30"  # months up to here were validation; after = test

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

print("Loading data and model...")
df = pd.read_parquet(DATA)
df.index = pd.to_datetime(df.index, utc=True)
model = joblib.load(MODEL_DIR / "lgbm_dayahead_v2.joblib")

# predict across all of 2025
year2025 = df['2025-01-01':'2025-12-31'].copy()
X = year2025[DAYAHEAD_FEATURES].fillna(0)
year2025['pred'] = model.predict(X).clip(0)

# persistence reference (24h)
year2025['persistence'] = year2025[TARGET].shift(24)

print("Computing monthly metrics...")
rows = []
for month, g in year2025.groupby(year2025.index.month):
    actual = g[TARGET].values
    pred   = g['pred'].values
    persist = g['persistence'].fillna(0).values
    dl = g['daylight_flag'].values.astype(bool)
    mae  = np.mean(np.abs(actual - pred))
    rmse = np.sqrt(np.mean((actual - pred)**2))
    mae_persist = np.mean(np.abs(actual - persist))
    skill = 1 - (np.sqrt(np.mean((actual-pred)**2)) / (np.sqrt(np.mean((actual-persist)**2)) + 1e-9))
    if dl.sum() > 0:
        mape = np.mean(np.abs((actual[dl]-pred[dl])/(actual[dl]+1e-6)))*100
    else:
        mape = np.nan
    window = "validation" if month <= 9 else "test"
    rows.append({"month": month, "MAE": mae, "RMSE": rmse,
                 "MAPE_daylight": mape, "MAE_persistence": mae_persist,
                 "SkillScore": skill, "window": window, "n": len(g)})

monthly = pd.DataFrame(rows)
monthly.to_csv(RES_DIR / "monthly_backtest.csv", index=False)
print(monthly.round(2).to_string(index=False))

# ── Figure 6: monthly MAE ─────────────────────────────────
month_names = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
colors = ['#1D9E75' if w == 'validation' else '#C0392B' for w in monthly['window']]

fig, ax = plt.subplots(figsize=(11, 5))
bars = ax.bar(monthly['month'], monthly['MAE'], color=colors)
ax.set_xticks(range(1, 13))
ax.set_xticklabels(month_names)
ax.set_ylabel('MAE (MW)')
ax.set_title('Monthly Day-Ahead Forecast MAE — 2025 (green = validation, red = test)')
ax.grid(axis='x', alpha=0)
for b, v in zip(bars, monthly['MAE']):
    ax.annotate(f'{v:.0f}', xy=(b.get_x()+b.get_width()/2, b.get_height()),
                xytext=(0,3), textcoords='offset points', ha='center', fontsize=8)
fig.tight_layout()
fig.savefig(FIG_DIR / "fig6_monthly_backtest.png", bbox_inches='tight')
plt.close(fig)

print(f"\nSaved monthly_backtest.csv and fig6_monthly_backtest.png")
