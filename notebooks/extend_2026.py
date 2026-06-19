import os
import time
import pandas as pd
from pathlib import Path
import openmeteo_requests
import requests_cache
from retry_requests import retry
from entsoe import EntsoePandasClient

# ── token from environment (loaded by PBS from ~/.entsoe_token) ──
API_TOKEN = os.environ.get("ENTSOE_TOKEN")
if not API_TOKEN:
    raise RuntimeError("ENTSOE_TOKEN not set. Check the PBS script loads the token file.")

DATA = Path("data")
COUNTRY = "CZ"
SLEEP = 2.2

# 2026 extension window
START = pd.Timestamp("2026-01-01", tz="Europe/Prague")
END   = pd.Timestamp("2026-06-15", tz="Europe/Prague")
OM_START = "2026-01-01"
OM_END   = "2026-06-15"

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

client = EntsoePandasClient(api_key=API_TOKEN)

def append_series(new_df, filename):
    """Append a single-column ENTSO-E series to existing raw file, dedupe on index."""
    if new_df is None:
        print(f"  SKIP {filename}: download failed")
        return
    path = DATA / filename
    if hasattr(new_df, 'to_frame'):
        new_df = new_df.to_frame()
    new_df.index = pd.to_datetime(new_df.index, utc=True)
    if path.exists():
        old = pd.read_parquet(path)
        old.index = pd.to_datetime(old.index, utc=True)
        combined = pd.concat([old, new_df])
        combined = combined[~combined.index.duplicated(keep='last')].sort_index()
    else:
        combined = new_df.sort_index()
    combined.to_parquet(path)
    print(f"  {filename}: now {len(combined)} rows ({combined.index.min()} -> {combined.index.max()})")

def safe_dl(label, func, **kwargs):
    print(f"\nDownloading: {label} (2026)...")
    try:
        time.sleep(SLEEP)
        r = func(**kwargs)
        print(f"  Success shape: {r.shape}")
        return r
    except Exception as e:
        print(f"  FAILED: {e}")
        return None

# ── ENTSO-E: solar (B16), prices, generation forecast ──
solar = safe_dl("Solar B16", client.query_generation,
                country_code=COUNTRY, start=START, end=END, psr_type="B16")
# query_generation may return a DataFrame; reduce to one column if so
if solar is not None and hasattr(solar, 'ndim') and solar.ndim > 1:
    solar = solar.iloc[:, 0]
append_series(solar, "entsoe_cz_solar_raw.parquet")

prices = safe_dl("Day-Ahead Prices", client.query_day_ahead_prices,
                 country_code=COUNTRY, start=START, end=END)
append_series(prices, "entsoe_cz_prices.parquet")

gen_fc = safe_dl("Generation Forecast", client.query_generation_forecast,
                 country_code=COUNTRY, start=START, end=END)
if gen_fc is not None and hasattr(gen_fc, 'ndim') and gen_fc.ndim > 1:
    gen_fc = gen_fc.iloc[:, 0]
append_series(gen_fc, "entsoe_cz_generation_forecast.parquet")

# ── Open-Meteo weather (long format) ──
print("\nDownloading Open-Meteo weather (2026)...")
cache = requests_cache.CachedSession('.cache_extend', expire_after=-1)
om = openmeteo_requests.Client(session=retry(cache, retries=5, backoff_factor=0.2))

all_dfs = []
for point in POINTS:
    params = {
        "latitude": point["lat"], "longitude": point["lon"],
        "start_date": OM_START, "end_date": OM_END,
        "hourly": VARIABLES, "timezone": "Europe/Prague",
    }
    responses = om.weather_api(
        "https://historical-forecast-api.open-meteo.com/v1/forecast",
        params=params
    )
    r = responses[0]
    hourly = r.Hourly()
    df = pd.DataFrame({
        "time": pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left"
        ).tz_convert("Europe/Prague"),
        **{var: hourly.Variables(i).ValuesAsNumpy() for i, var in enumerate(VARIABLES)}
    })
    df["location"] = point["name"]
    all_dfs.append(df)
    print(f"  {point['name']}: {len(df)} hours")

new_weather = pd.concat(all_dfs, ignore_index=True)

wpath = DATA / "openmeteo_cz_historical.parquet"
if wpath.exists():
    old_w = pd.read_parquet(wpath)
    old_w["time"] = pd.to_datetime(old_w["time"], utc=True)
    new_weather["time"] = pd.to_datetime(new_weather["time"], utc=True)
    combined_w = pd.concat([old_w, new_weather], ignore_index=True)
    combined_w = combined_w.drop_duplicates(subset=["time", "location"], keep="last")
    combined_w = combined_w.sort_values(["location", "time"]).reset_index(drop=True)
else:
    combined_w = new_weather

combined_w.to_parquet(wpath)
print(f"\nWeather: now {len(combined_w)} rows, time range "
      f"{combined_w['time'].min()} -> {combined_w['time'].max()}")
print("\n2026 extension complete. Raw files updated.")
