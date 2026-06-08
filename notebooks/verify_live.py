import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from entsoe import EntsoePandasClient

API_TOKEN = "REPLACE_THIS"
RES_DIR = Path("outputs")
FIG_DIR = Path("outputs/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

print("Loading saved live forecast...")
fc = pd.read_csv(RES_DIR / "live_forecast_48h.csv", index_col=0)
fc.index = pd.to_datetime(fc.index, utc=True)
print(f"Forecast covers {fc.index.min()} to {fc.index.max()} ({len(fc)} hours)")

print("Fetching ACTUAL solar generation from ENTSO-E...")
ent = EntsoePandasClient(api_key=API_TOKEN)
start = (fc.index.min() - pd.Timedelta(hours=2)).tz_convert("Europe/Prague")
end   = (fc.index.max() + pd.Timedelta(hours=2)).tz_convert("Europe/Prague")

sol = ent.query_generation(country_code="CZ", start=start, end=end, psr_type="B16")
if hasattr(sol, 'ndim') and sol.ndim > 1:
    sol = sol.iloc[:, 0]
sol.index = pd.to_datetime(sol.index, utc=True)
actual = sol.resample('h').mean()

# align actual to forecast hours
df = fc.copy()
df['actual'] = actual.reindex(df.index, method='nearest')
df = df.dropna(subset=['actual'])
print(f"Aligned {len(df)} hours with actual data")

# metrics on the real outcome
mae  = np.mean(np.abs(df['actual'] - df['p50']))
rmse = np.sqrt(np.mean((df['actual'] - df['p50'])**2))
inside = ((df['actual'] >= df['p10']) & (df['actual'] <= df['p90'])).mean()

print("\n===== LIVE FORECAST VERIFICATION =====")
print(f"  MAE (actual vs p50):  {mae:.2f} MW")
print(f"  RMSE:                 {rmse:.2f} MW")
print(f"  Actual inside p10-p90: {inside*100:.1f}% (nominal 80%)")
print("\nHour-by-hour:")
print(df[['actual','p10','p50','p90']].round(1).to_string())

# save
df.to_csv(RES_DIR / "live_forecast_verified.csv")

# plot forecast vs reality
fig, ax = plt.subplots(figsize=(11, 4.5))
ax.fill_between(df.index, df['p10'], df['p90'], color='#1D9E75', alpha=0.25, label='Forecast p10-p90')
ax.plot(df.index, df['p50'], color='#1D9E75', lw=2, label='Forecast (p50)')
ax.plot(df.index, df['actual'], color='#1A1A1A', lw=1.8, label='Actual', marker='o', markersize=3)
ax.set_ylabel('Solar generation (MW)')
ax.set_title(f'Live Forecast vs Actual - CZ Solar (MAE {mae:.0f} MW, {inside*100:.0f}% in band)')
ax.legend(loc='upper right')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d %Hh'))
fig.autofmt_xdate()
fig.tight_layout()
fig.savefig(FIG_DIR / "fig7_live_vs_actual.png", bbox_inches='tight')
print(f"\nSaved fig7_live_vs_actual.png and live_forecast_verified.csv")
