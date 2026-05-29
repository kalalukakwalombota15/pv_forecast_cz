import pandas as pd
import numpy as np
from pathlib import Path
from scipy.interpolate import CubicSpline

# ── config ────────────────────────────────────────────────
DATA_DIR   = Path("data")
OUTPUT     = Path("data/master_features.parquet")
SOLAR_ELEV_THRESHOLD = 5
MAX_GAP_FILL = 2
# ─────────────────────────────────────────────────────────

print("Loading ENTSO-E datasets...")
solar    = pd.read_parquet(DATA_DIR / "entsoe_cz_solar_raw.parquet")
load     = pd.read_parquet(DATA_DIR / "entsoe_cz_load.parquet")
wind     = pd.read_parquet(DATA_DIR / "entsoe_cz_wind.parquet")
hydro    = pd.read_parquet(DATA_DIR / "entsoe_cz_hydro.parquet")
nuclear  = pd.read_parquet(DATA_DIR / "entsoe_cz_nuclear.parquet")
prices   = pd.read_parquet(DATA_DIR / "entsoe_cz_prices.parquet")
gen_fc   = pd.read_parquet(DATA_DIR / "entsoe_cz_generation_forecast.parquet")
flow_de  = pd.read_parquet(DATA_DIR / "entsoe_cz_de_flow.parquet")
flow_at  = pd.read_parquet(DATA_DIR / "entsoe_cz_at_flow.parquet")
flow_sk  = pd.read_parquet(DATA_DIR / "entsoe_cz_sk_flow.parquet")
flow_pl  = pd.read_parquet(DATA_DIR / "entsoe_cz_pl_flow.parquet")
weather  = pd.read_parquet(DATA_DIR / "openmeteo_cz_historical.parquet")

def to_series(df, name):
    if isinstance(df, pd.DataFrame):
        s = df.iloc[:, 0]
    else:
        s = df.copy()
    s.name = name
    s.index = pd.to_datetime(s.index, utc=True)
    return s

def cubic_resample(series, target_freq='h'):
    series = series.sort_index()
    series = series[~series.index.duplicated(keep='first')]
    full_index = pd.date_range(
        start=series.index.min().floor(target_freq),
        end=series.index.max().ceil(target_freq),
        freq=target_freq
    )
    resampled = series.resample(target_freq).mean()
    resampled = resampled.reindex(full_index)
    is_nan = resampled.isna()
    gap_groups = (is_nan != is_nan.shift()).cumsum()
    gap_sizes = is_nan.groupby(gap_groups).transform('sum')
    long_gap = is_nan & (gap_sizes > MAX_GAP_FILL)
    valid = resampled[~is_nan]
    if len(valid) > 3:
        x_valid = np.array([t.timestamp() for t in valid.index])
        y_valid = valid.values.astype(float)
        cs = CubicSpline(x_valid, y_valid, bc_type='natural')
        x_all = np.array([t.timestamp() for t in resampled.index])
        interpolated = pd.Series(cs(x_all), index=resampled.index)
        resampled = resampled.where(~is_nan | long_gap, interpolated)
    resampled[long_gap] = np.nan
    return resampled

print("Resampling ENTSO-E data to hourly using cubic spline...")
solar_h   = cubic_resample(to_series(solar,   "solar_mw"))
load_h    = cubic_resample(to_series(load,    "load_mw"))
wind_h    = cubic_resample(to_series(wind,    "wind_mw"))
hydro_h   = cubic_resample(to_series(hydro,   "hydro_mw"))
nuclear_h = cubic_resample(to_series(nuclear, "nuclear_mw"))
prices_h  = cubic_resample(to_series(prices,  "price_eur_mwh"))
gen_fc_h  = cubic_resample(to_series(gen_fc,  "gen_forecast_mw"))
flow_de_h = cubic_resample(to_series(flow_de, "flow_de_mw"))
flow_at_h = cubic_resample(to_series(flow_at, "flow_at_mw"))
flow_sk_h = cubic_resample(to_series(flow_sk, "flow_sk_mw"))
flow_pl_h = cubic_resample(to_series(flow_pl, "flow_pl_mw"))

print("Aggregating Open-Meteo weather across 15 Czech cities...")
weather['time'] = pd.to_datetime(weather['time'], utc=True)
weather_vars = [
    'shortwave_radiation', 'direct_radiation', 'diffuse_radiation',
    'direct_normal_irradiance', 'global_tilted_irradiance',
    'cloud_cover', 'temperature_2m', 'wind_speed_10m', 'precipitation'
]

agg_mean = weather.groupby('time')[weather_vars].mean()
agg_p10  = weather.groupby('time')[weather_vars].quantile(0.1)
agg_p90  = weather.groupby('time')[weather_vars].quantile(0.9)

agg_mean.columns = [f"{c}_mean" for c in agg_mean.columns]
agg_p10.columns  = [f"{c}_p10"  for c in agg_p10.columns]
agg_p90.columns  = [f"{c}_p90"  for c in agg_p90.columns]

agg = pd.concat([agg_mean, agg_p10, agg_p90], axis=1)
agg.index = pd.to_datetime(agg.index, utc=True)

print("Building time and solar features...")
master = pd.DataFrame(index=solar_h.index)
master['solar_mw']        = solar_h
master['load_mw']         = load_h
master['wind_mw']         = wind_h
master['hydro_mw']        = hydro_h
master['nuclear_mw']      = nuclear_h
master['price_eur_mwh']   = prices_h
master['gen_forecast_mw'] = gen_fc_h
master['flow_de_mw']      = flow_de_h
master['flow_at_mw']      = flow_at_h
master['flow_sk_mw']      = flow_sk_h
master['flow_pl_mw']      = flow_pl_h

master = master.join(agg, how='left')

master['hour']        = master.index.hour
master['day_of_year'] = master.index.dayofyear
master['month']       = master.index.month
master['is_weekend']  = master.index.dayofweek >= 5
master['sin_hour']    = np.sin(2 * np.pi * master['hour'] / 24)
master['cos_hour']    = np.cos(2 * np.pi * master['hour'] / 24)
master['sin_doy']     = np.sin(2 * np.pi * master['day_of_year'] / 365)
master['cos_doy']     = np.cos(2 * np.pi * master['day_of_year'] / 365)

lat, lon = 49.8, 15.5
def solar_elevation(dt_index):
    lat_r = np.radians(lat)
    doy   = dt_index.dayofyear
    hour  = dt_index.hour + dt_index.minute / 60
    decl  = np.radians(23.45 * np.sin(np.radians(360 / 365 * (doy - 81))))
    ha    = np.radians(15 * (hour - 12) + lon - 15)
    elev  = np.degrees(np.arcsin(
        np.sin(lat_r) * np.sin(decl) +
        np.cos(lat_r) * np.cos(decl) * np.cos(ha)
    ))
    return elev

master['solar_elevation'] = solar_elevation(master.index)
master['daylight_flag']   = (master['solar_elevation'] > SOLAR_ELEV_THRESHOLD).astype(int)

rad_col = 'shortwave_radiation_mean'
if rad_col in master.columns:
    daily_max = master[rad_col].resample('D').transform('max')
    daily_max = daily_max.replace(0, np.nan)
    master['clearness_index'] = (master[rad_col] / daily_max).clip(0, 1)

print("Filtering to 2023-2025 and cleaning...")
master = master['2023-01-01':'2025-12-31']
master = master[master.index.notna()]

print(f"Master table shape: {master.shape}")
print(f"Columns: {list(master.columns)}")
print(f"Missing values:\n{master.isna().sum()[master.isna().sum() > 0]}")
print(f"Date range: {master.index.min()} to {master.index.max()}")

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
master.to_parquet(OUTPUT)
print(f"\nSaved to {OUTPUT}")
