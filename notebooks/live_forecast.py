import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import openmeteo_requests
import requests_cache
from retry_requests import retry
from entsoe import EntsoePandasClient

# ── config ────────────────────────────────────────────────
API_TOKEN = "REPLACE_THIS"
MODEL_DIR = Path("models")
RES_DIR   = Path("outputs")
RES_DIR.mkdir(parents=True, exist_ok=True)

POINTS = [
    {"name": "Prague","lat":50.08,"lon":14.44},{"name":"Brno","lat":49.19,"lon":16.61},
    {"name":"Ostrava","lat":49.83,"lon":18.29},{"name":"Plzen","lat":49.74,"lon":13.37},
    {"name":"Liberec","lat":50.76,"lon":15.05},{"name":"Olomouc","lat":49.59,"lon":17.25},
    {"name":"Ceske Budejovice","lat":48.97,"lon":14.47},{"name":"Hradec Kralove","lat":50.21,"lon":15.83},
    {"name":"Zlin","lat":49.22,"lon":17.66},{"name":"Jihlava","lat":49.39,"lon":15.59},
    {"name":"Karlovy Vary","lat":50.23,"lon":12.87},{"name":"Usti nad Labem","lat":50.66,"lon":14.03},
    {"name":"Pardubice","lat":50.04,"lon":15.77},{"name":"Znojmo","lat":48.86,"lon":16.05},
    {"name":"Hodonin","lat":48.85,"lon":17.13},
]
VARIABLES = [
    "shortwave_radiation","direct_radiation","diffuse_radiation",
    "direct_normal_irradiance","global_tilted_irradiance",
    "cloud_cover","temperature_2m","wind_speed_10m","precipitation",
]
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
SOLAR_ELEV_THRESHOLD = 5
# ─────────────────────────────────────────────────────────

print("Fetching live 48h Open-Meteo forecast for 15 CZ cities...")
cache = requests_cache.CachedSession('.cache_live', expire_after=3600)
client = openmeteo_requests.Client(session=retry(cache, retries=5, backoff_factor=0.2))

dfs = []
for pt in POINTS:
    params = {
        "latitude": pt["lat"], "longitude": pt["lon"],
        "hourly": VARIABLES, "timezone": "Europe/Prague",
        "forecast_days": 3,
    }
    r = client.weather_api("https://api.open-meteo.com/v1/forecast", params=params)[0]
    h = r.Hourly()
    d = pd.DataFrame({
        "time": pd.date_range(
            start=pd.to_datetime(h.Time(), unit="s", utc=True),
            end=pd.to_datetime(h.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=h.Interval()), inclusive="left"
        ).tz_convert("Europe/Prague"),
        **{v: h.Variables(i).ValuesAsNumpy() for i, v in enumerate(VARIABLES)}
    })
    d["location"] = pt["name"]
    dfs.append(d)

weather = pd.concat(dfs, ignore_index=True)
weather["time"] = pd.to_datetime(weather["time"], utc=True)

print("Aggregating across cities...")
agg_mean = weather.groupby("time")[VARIABLES].mean()
agg_p10  = weather.groupby("time")[VARIABLES].quantile(0.1)
agg_p90  = weather.groupby("time")[VARIABLES].quantile(0.9)
agg_mean.columns = [f"{c}_mean" for c in agg_mean.columns]
agg_p10.columns  = [f"{c}_p10"  for c in agg_p10.columns]
agg_p90.columns  = [f"{c}_p90"  for c in agg_p90.columns]
m = pd.concat([agg_mean, agg_p10, agg_p90], axis=1)
m.index = pd.to_datetime(m.index, utc=True)

# take the next 48 hours from now
now = pd.Timestamp.now(tz="UTC")
m = m[(m.index >= now) & (m.index < now + pd.Timedelta(hours=48))]
print(f"Forecast window: {m.index.min()} to {m.index.max()} ({len(m)} hours)")

# time + solar geometry features
def solar_elev(idx):
    lat_r = np.radians(49.8)
    doy = idx.dayofyear; hour = idx.hour + idx.minute/60
    decl = np.radians(23.45*np.sin(np.radians(360/365*(doy-81))))
    ha = np.radians(15*(hour-12)+15.5)
    elev = np.degrees(np.arcsin(np.sin(lat_r)*np.sin(decl)+np.cos(lat_r)*np.cos(decl)*np.cos(ha)))
    return elev, np.degrees(decl)

m['hour'] = m.index.hour; m['day_of_year'] = m.index.dayofyear; m['month'] = m.index.month
m['is_weekend'] = (m.index.dayofweek >= 5).astype(int)
m['sin_hour'] = np.sin(2*np.pi*m['hour']/24); m['cos_hour'] = np.cos(2*np.pi*m['hour']/24)
m['sin_doy'] = np.sin(2*np.pi*m['day_of_year']/365); m['cos_doy'] = np.cos(2*np.pi*m['day_of_year']/365)
elev, decl = solar_elev(m.index)
m['solar_elevation'] = elev; m['solar_declination'] = decl
m['daylight_flag'] = (m['solar_elevation'] > SOLAR_ELEV_THRESHOLD).astype(int)

rad = 'shortwave_radiation_mean'
daily_max = m[rad].resample('D').transform('max').replace(0, np.nan)
m['clearness_index'] = (m[rad]/daily_max).clip(0,1).fillna(0)
m['radiation_roll3h_mean'] = m[rad].rolling(3, min_periods=1).mean()
m['radiation_roll24h_mean'] = m[rad].rolling(24, min_periods=1).mean()
m['cloud_roll3h_mean'] = m['cloud_cover_mean'].rolling(3, min_periods=1).mean()
m['installed_capacity_mw'] = 4812  # current CZ capacity

print("Fetching ENTSO-E context (prices, gen forecast, last-week solar)...")
ent = EntsoePandasClient(api_key=API_TOKEN)
start = (now - pd.Timedelta(days=8)).tz_convert("Europe/Prague")
end   = (now + pd.Timedelta(days=2)).tz_convert("Europe/Prague")

try:
    prices = ent.query_day_ahead_prices(country_code="CZ", start=start, end=end)
    prices.index = pd.to_datetime(prices.index, utc=True)
    m['price_eur_mwh'] = prices.reindex(m.index, method='nearest').values
except Exception as e:
    print(f"  prices failed ({e}); filling 0"); m['price_eur_mwh'] = 0.0

try:
    gf = ent.query_generation_forecast(country_code="CZ", start=start, end=end)
    if hasattr(gf, 'to_frame'): gf = gf.iloc[:,0] if hasattr(gf,'iloc') and gf.ndim>1 else gf
    gf.index = pd.to_datetime(gf.index, utc=True)
    m['gen_forecast_mw'] = gf.reindex(m.index, method='nearest').values
except Exception as e:
    print(f"  gen forecast failed ({e}); filling 0"); m['gen_forecast_mw'] = 0.0

try:
    sol = ent.query_generation(country_code="CZ", start=start, end=end, psr_type="B16")
    if hasattr(sol, 'iloc') and getattr(sol, 'ndim', 1) > 1: sol = sol.iloc[:,0]
    sol.index = pd.to_datetime(sol.index, utc=True)
    sol_h = sol.resample('h').mean()
    lag = sol_h.reindex(m.index - pd.Timedelta(hours=168), method='nearest')
    m['solar_lag_168h'] = lag.values
except Exception as e:
    print(f"  last-week solar failed ({e}); filling 0"); m['solar_lag_168h'] = 0.0

m = m.fillna(0)

print("Loading models and predicting...")
model_p50 = joblib.load(MODEL_DIR / "lgbm_dayahead_v2.joblib")
model_p10 = joblib.load(MODEL_DIR / "lgbm_p10_v2.joblib")
model_p90 = joblib.load(MODEL_DIR / "lgbm_p90_v2.joblib")

X = m[DAYAHEAD_FEATURES].fillna(0)
p50 = model_p50.predict(X).clip(0)
p10 = model_p10.predict(X).clip(0)
p90 = model_p90.predict(X).clip(0)
p10 = np.minimum(p10, p50); p90 = np.maximum(p90, p50)

out = pd.DataFrame({'p10':p10,'p50':p50,'p90':p90}, index=m.index)
out.to_csv(RES_DIR / "live_forecast_48h.csv")
print(out.round(1).to_string())
print(f"\nSaved to {RES_DIR}/live_forecast_48h.csv")

# plot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
fig, ax = plt.subplots(figsize=(11,4.5))
ax.fill_between(out.index, out['p10'], out['p90'], color='#1D9E75', alpha=0.25, label='p10-p90')
ax.plot(out.index, out['p50'], color='#1D9E75', lw=2, label='Forecast (p50)')
ax.set_ylabel('Solar generation (MW)')
ax.set_title('Live 48-Hour Day-Ahead PV Forecast — Czech Republic')
ax.legend(); ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d %Hh'))
fig.autofmt_xdate(); fig.tight_layout()
fig.savefig(RES_DIR / "figures/fig5_live_48h_forecast.png", bbox_inches='tight')
print("Figure saved to outputs/figures/fig5_live_48h_forecast.png")
