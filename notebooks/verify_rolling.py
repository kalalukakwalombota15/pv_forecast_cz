import os
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from entsoe import EntsoePandasClient

API_TOKEN = os.environ.get("ENTSOE_TOKEN")
if not API_TOKEN:
    raise RuntimeError("ENTSOE_TOKEN not set. Check the PBS script loads the token file.")

RES_DIR = Path("outputs")
FIG_DIR = Path("outputs/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

print("Loading rolling forecast...")
fc = pd.read_csv(RES_DIR / "rolling_forecast.csv")
fc['target_time'] = pd.to_datetime(fc['target_time'], utc=True)
fc = fc.sort_values('target_time').drop_duplicates(subset='target_time', keep='last')
fc = fc.set_index('target_time')
print(f"Forecast spans {fc.index.min()} to {fc.index.max()} ({len(fc)} hours)")

print("Fetching ACTUAL solar generation from ENTSO-E...")
ent = EntsoePandasClient(api_key=API_TOKEN)
start = (fc.index.min() - pd.Timedelta(hours=3)).tz_convert("Europe/Prague")
end   = (fc.index.max() + pd.Timedelta(hours=3)).tz_convert("Europe/Prague")

sol = ent.query_generation(country_code="CZ", start=start, end=end, psr_type="B16")
if hasattr(sol, 'ndim') and sol.ndim > 1:
    sol = sol.iloc[:, 0]
sol.index = pd.to_datetime(sol.index, utc=True)
actual = sol.resample('h').mean()

df = fc.copy()
df['actual'] = actual.reindex(df.index, method='nearest')
# keep only hours where actual exists (recent days may not be published yet)
df = df.dropna(subset=['actual'])
print(f"Verifiable hours with published actuals: {len(df)}")
print(f"Verified window: {df.index.min()} to {df.index.max()}")

# metrics
mae  = np.mean(np.abs(df['actual'] - df['p50']))
rmse = np.sqrt(np.mean((df['actual'] - df['p50'])**2))
inside = ((df['actual'] >= df['p10']) & (df['actual'] <= df['p90'])).mean()

# daylight-only (actual > 1 MW)
day = df['actual'] > 1.0
mae_day = np.mean(np.abs(df['actual'][day] - df['p50'][day]))
inside_day = ((df['actual'][day] >= df['p10'][day]) & (df['actual'][day] <= df['p90'][day])).mean()

print("\n===== ROLLING FORECAST VERIFICATION =====")
print(f"  Days verified:        {df.index.normalize().nunique()}")
print(f"  MAE (all hours):      {mae:.2f} MW")
print(f"  RMSE (all hours):     {rmse:.2f} MW")
print(f"  MAE (daylight):       {mae_day:.2f} MW")
print(f"  Coverage p10-p90:     {inside*100:.1f}% (all), {inside_day*100:.1f}% (daylight)")

df.to_csv(RES_DIR / "rolling_forecast_verified.csv")

# ── figure: continuous multi-day forecast vs actual ───────
fig, ax = plt.subplots(figsize=(14, 5))
ax.fill_between(df.index, df['p10'], df['p90'], color='#1D9E75', alpha=0.25, label='Forecast p10-p90')
ax.plot(df.index, df['p50'], color='#1D9E75', lw=1.6, label='Forecast (p50)')
ax.plot(df.index, df['actual'], color='#1A1A1A', lw=1.4, label='Actual')
ax.set_ylabel('Solar generation (MW)')
ax.set_title(f'Rolling Day-Ahead Forecast vs Actual - CZ Solar '
             f'({df.index.normalize().nunique()} days, MAE {mae_day:.0f} MW daylight, {inside_day*100:.0f}% in band)')
ax.legend(loc='upper right')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
ax.xaxis.set_major_locator(mdates.DayLocator())
fig.autofmt_xdate()
fig.tight_layout()
fig.savefig(FIG_DIR / "fig11_rolling_verification.png", bbox_inches='tight')
print("\nSaved rolling_forecast_verified.csv and fig11_rolling_verification.png")
