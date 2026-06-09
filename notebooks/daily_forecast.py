import os
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import openmeteo_requests
import requests_cache
from retry_requests import retry
from entsoe import EntsoePandasClient

# ── token from environment (loaded by PBS from ~/.entsoe_token) ──
API_TOKEN = os.environ.get("ENTSOE_TOKEN")
if not API_TOKEN:
    raise RuntimeError("ENTSOE_TOKEN not set. Check the PBS script loads ~/.entsoe_token.")

MODEL_DIR = Path("models")
RES_DIR   = Path("outputs")
RES_DIR.mkdir(parents=True, exist_ok=True)
ROLLING_CSV = RES_DIR / "rolling_forecast.csv"

POINTS = [
    {"name":"Prague","lat":50.08,"lon":14.44},{"name":"Brno","lat":49.19,"lon":16.61},
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

# ── target = FULL next calendar day (Europe/Prague), as UTC hours ──
now_prague = pd.Timestamp.now(tz="Europe/Prague")
issue_date = now_prague.date()
tomorrow = (now_prague + pd.Timedelta(days=1)).normalize()      # tomorrow 00:00 Prague
target_prague = pd.date_range(tomorrow, periods=24, freq='h', tz='Europe/Prague')
target_utc = target_prague.tz_convert('UTC')
target_date = tomorrow.date()
print(f"Issue date: {issue_date} | Forecasting full day: {target_date}")

print("Fetching Open-Meteo forecast...")
cache = requests_cache.CachedSession('.cache_daily', expire_after=3600)
client = openmeteo_requests.Client(session=retry(cache, retries=5, backoff_factor=0.2))
dfs = []
for pt in POINTS:
    params = {"latitude":pt["lat"],"longitude":pt["lon"],"hourly":VARIABLES,
              "timezone":"Europe/Prague","forecast_days":3}
    r = client.weather_api("https://api.open-meteo.com/v1/forecast", params=params)[0]
    h = r.Hourly()
    d = pd.DataFrame({
        "time": pd.date_range(
            start=pd.to_datetime(h.Time(), unit="s", utc=True),
            end=pd.to_datetime(h.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=h.Interval()), inclusive="left"),
        **{v: h.Variables(i).ValuesAsNumpy() for i, v in enumerate(VARIABLES)}})
    dfs.append(d)
weather = pd.concat(dfs, ignore_index=True)
weather["time"] = pd.to_datetime(weather["time"], utc=True)

agg_mean = weather.groupby("time")[VARIABLES].mean()
agg_p10  = weather.groupby("time")[VARIABLES].quantile(0.1)
agg_p90  = weather.groupby("time")[VARIABLES].quantile(0.9)
agg_mean.columns = [f"{c}_mean" for c in agg_mean.columns]
agg_p10.columns  = [f"{c}_p10"  for c in agg_p10.columns]
agg_p90.columns  = [f"{c}_p90"  for c in agg_p90.columns]
m = pd.concat([agg_mean, agg_p10, agg_p90], axis=1)
m.index = pd.to_datetime(m.index, utc=True)

# keep only the target day's 24 UTC hours
m = m.reindex(target_utc).dropna(how='all')
if len(m) == 0:
    raise RuntimeError("No weather data for target day - is the forecast horizon long enough?")
print(f"Target hours available: {len(m)}")

def solar_elev(idx):
    lat_r = np.radians(49.8); doy = idx.dayofyear; hour = idx.hour + idx.minute/60
    decl = np.radians(23.45*np.sin(np.radians(360/365*(doy-81))))
    ha = np.radians(15*(hour-12)+15.5)
    elev = np.degrees(np.arcsin(np.sin(lat_r)*np.sin(decl)+np.cos(lat_r)*np.cos(decl)*np.cos(ha)))
    return elev, np.degrees(decl)

m['hour']=m.index.hour; m['day_of_year']=m.index.dayofyear; m['month']=m.index.month
m['is_weekend']=(m.index.dayofweek>=5).astype(int)
m['sin_hour']=np.sin(2*np.pi*m['hour']/24); m['cos_hour']=np.cos(2*np.pi*m['hour']/24)
m['sin_doy']=np.sin(2*np.pi*m['day_of_year']/365); m['cos_doy']=np.cos(2*np.pi*m['day_of_year']/365)
elev,decl = solar_elev(m.index); m['solar_elevation']=elev; m['solar_declination']=decl
m['daylight_flag']=(m['solar_elevation']>SOLAR_ELEV_THRESHOLD).astype(int)
rad='shortwave_radiation_mean'
dm_=m[rad].resample('D').transform('max').replace(0,np.nan)
m['clearness_index']=(m[rad]/dm_).clip(0,1).fillna(0)
m['radiation_roll3h_mean']=m[rad].rolling(3,min_periods=1).mean()
m['radiation_roll24h_mean']=m[rad].rolling(24,min_periods=1).mean()
m['cloud_roll3h_mean']=m['cloud_cover_mean'].rolling(3,min_periods=1).mean()
m['installed_capacity_mw']=4812

print("Fetching ENTSO-E context...")
ent = EntsoePandasClient(api_key=API_TOKEN)
start=(now_prague - pd.Timedelta(days=8)); end=(now_prague + pd.Timedelta(days=2))
try:
    pr = ent.query_day_ahead_prices(country_code="CZ", start=start, end=end)
    pr.index = pd.to_datetime(pr.index, utc=True)
    m['price_eur_mwh'] = pr.reindex(m.index, method='nearest').values
except Exception as e:
    print(f"  prices failed ({e})"); m['price_eur_mwh']=0.0
try:
    gf = ent.query_generation_forecast(country_code="CZ", start=start, end=end)
    if hasattr(gf,"ndim") and gf.ndim>1: gf=gf.iloc[:,0]
    gf.index = pd.to_datetime(gf.index, utc=True)
    m['gen_forecast_mw'] = gf.reindex(m.index, method='nearest').values
except Exception as e:
    print(f"  gen forecast failed ({e})"); m['gen_forecast_mw']=0.0
try:
    sol = ent.query_generation(country_code="CZ", start=start, end=end, psr_type="B16")
    if hasattr(sol,"ndim") and sol.ndim>1: sol=sol.iloc[:,0]
    sol.index = pd.to_datetime(sol.index, utc=True)
    sol_h = sol.resample('h').mean()
    m['solar_lag_168h'] = sol_h.reindex(m.index - pd.Timedelta(hours=168), method='nearest').values
except Exception as e:
    print(f"  last-week solar failed ({e})"); m['solar_lag_168h']=0.0
m = m.fillna(0)

print("Predicting...")
mp10=joblib.load(MODEL_DIR/"lgbm_p10_v2.joblib")
mp50=joblib.load(MODEL_DIR/"lgbm_dayahead_v2.joblib")
mp90=joblib.load(MODEL_DIR/"lgbm_p90_v2.joblib")
X=m[DAYAHEAD_FEATURES].fillna(0)
p10=mp10.predict(X).clip(0); p50=mp50.predict(X).clip(0); p90=mp90.predict(X).clip(0)
q=np.sort(np.column_stack([p10,p50,p90]),axis=1); p10,p50,p90=q[:,0],q[:,1],q[:,2]

new = pd.DataFrame({
    'issue_date': str(issue_date),
    'target_time': m.index.tz_convert('UTC'),
    'p10':p10,'p50':p50,'p90':p90,
})

# append to rolling file (drop duplicate target_time, keep latest issue)
if ROLLING_CSV.exists():
    old = pd.read_csv(ROLLING_CSV)
    combined = pd.concat([old, new], ignore_index=True)
    combined = combined.drop_duplicates(subset='target_time', keep='last')
else:
    combined = new
combined.to_csv(ROLLING_CSV, index=False)
print(f"Appended {len(new)} hours for {target_date}. Rolling file now has {len(combined)} rows.")
print(new[['target_time','p10','p50','p90']].round(1).to_string(index=False))
