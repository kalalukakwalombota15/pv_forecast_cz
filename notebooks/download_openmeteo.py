import openmeteo_requests
import requests_cache
import pandas as pd
from retry_requests import retry
from pathlib import Path

# ── config ────────────────────────────────────────────────
START  = "2023-01-01"
END    = "2025-12-31"
OUTPUT = Path("data/openmeteo_cz_historical.parquet")

POINTS = [
    {"name": "Prague",            "lat": 50.08, "lon": 14.44},
    {"name": "Brno",              "lat": 49.19, "lon": 16.61},
    {"name": "Ostrava",           "lat": 49.83, "lon": 18.29},
    {"name": "Plzen",             "lat": 49.74, "lon": 13.37},
    {"name": "Liberec",           "lat": 50.76, "lon": 15.05},
    {"name": "Olomouc",           "lat": 49.59, "lon": 17.25},
    {"name": "Ceske Budejovice",  "lat": 48.97, "lon": 14.47},
    {"name": "Hradec Kralove",    "lat": 50.21, "lon": 15.83},
    {"name": "Zlin",              "lat": 49.22, "lon": 17.66},
    {"name": "Jihlava",           "lat": 49.39, "lon": 15.59},
    {"name": "Karlovy Vary",      "lat": 50.23, "lon": 12.87},
    {"name": "Usti nad Labem",    "lat": 50.66, "lon": 14.03},
    {"name": "Pardubice",         "lat": 50.04, "lon": 15.77},
    {"name": "Znojmo",            "lat": 48.86, "lon": 16.05},
    {"name": "Hodonin",           "lat": 48.85, "lon": 17.13},
]

VARIABLES = [
    "shortwave_radiation",
    "direct_radiation",
    "diffuse_radiation",
    "direct_normal_irradiance",
    "global_tilted_irradiance",
    "cloud_cover",
    "temperature_2m",
    "wind_speed_10m",
    "precipitation",
]
# ─────────────────────────────────────────────────────────

cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
client = openmeteo_requests.Client(session=retry_session)

all_dfs = []

for point in POINTS:
    print(f"Downloading {point['name']}...")
    params = {
        "latitude": point["lat"],
        "longitude": point["lon"],
        "start_date": START,
        "end_date": END,
        "hourly": VARIABLES,
        "timezone": "Europe/Prague",
    }
    responses = client.weather_api(
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

print("Combining all points...")
combined = pd.concat(all_dfs, ignore_index=True)
print(f"Total shape: {combined.shape}")
print(combined.head())

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
combined.to_parquet(OUTPUT)
print(f"Saved to {OUTPUT}")
