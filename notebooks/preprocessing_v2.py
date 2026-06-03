import pandas as pd
import numpy as np
from pathlib import Path
from scipy.interpolate import CubicSpline
import warnings
warnings.filterwarnings('ignore')

# ── config ────────────────────────────────────────────────
DATA_DIR   = Path("data")
OUTPUT     = Path("data/master_features_v2.parquet")
REPORT     = Path("outputs/data_quality_report.txt")

SOLAR_ELEV_THRESHOLD = 5
MAX_GAP_FILL         = 2
MAX_RAMP_RATE        = 1000
RANDOM_SEED          = 42

CZ_INSTALLED_CAPACITY = {
    2023: 3500,
    2024: 4640,
    2025: 4812,
}

def get_installed_capacity(dt_index):
    year = dt_index.year
    cap = np.where(year <= 2023, 3500,
          np.where(year == 2024, 4640, 4812))
    month_frac = (dt_index.month - 1) / 12
    prev_cap = np.where(year <= 2023, 2580,
               np.where(year == 2024, 3500, 4640))
    return prev_cap + (cap - prev_cap) * month_frac
# ─────────────────────────────────────────────────────────

Path("outputs").mkdir(parents=True, exist_ok=True)
report_lines = []

def log(msg):
    print(msg)
    report_lines.append(msg)

log("=" * 60)
log("PV FORECAST CZ — DATA QUALITY REPORT")
log("=" * 60)

log("\n[1] Loading ENTSO-E datasets...")
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
log("All datasets loaded.")

def to_series(df, name):
    if isinstance(df, pd.DataFrame):
        s = df.iloc[:, 0]
    else:
        s = df.copy()
    s.name = name
    s.index = pd.to_datetime(s.index, utc=True)
    return s

def solar_elevation_calc(dt_index):
    lat_r = np.radians(49.8)
    doy   = dt_index.dayofyear
    hour  = dt_index.hour + dt_index.minute / 60
    decl  = np.radians(23.45 * np.sin(np.radians(360 / 365 * (doy - 81))))
    decl_deg = np.degrees(decl)
    ha    = np.radians(15 * (hour - 12) + 15.5)
    elev  = np.degrees(np.arcsin(
        np.sin(lat_r) * np.sin(decl) +
        np.cos(lat_r) * np.cos(decl) * np.cos(ha)
    ))
    return elev, decl_deg

def cubic_resample(series, target_freq='h'):
    series = series.sort_index()
    series = series[~series.index.duplicated(keep='first')]
    series.index = series.index.tz_convert('UTC')
    full_index = pd.date_range(
        start=series.index.min().floor(target_freq),
        end=series.index.max().ceil(target_freq),
        freq=target_freq,
        tz='UTC'
    )
    resampled = series.resample(target_freq).mean()
    resampled = resampled.reindex(full_index)
    is_nan = resampled.isna()
    gap_groups = (is_nan != is_nan.shift()).cumsum()
    gap_sizes = is_nan.groupby(gap_groups).transform('sum')
    long_gap = is_nan & (gap_sizes > MAX_GAP_FILL)
    short_gap = is_nan & ~long_gap
    valid = resampled[~is_nan]
    if len(valid) > 3:
        x_valid = np.array([t.timestamp() for t in valid.index])
        y_valid = valid.values.astype(float)
        cs = CubicSpline(x_valid, y_valid, bc_type='natural')
        x_all = np.array([t.timestamp() for t in resampled.index])
        interpolated = pd.Series(cs(x_all), index=resampled.index)
        resampled = resampled.where(~short_gap, interpolated)
    resampled[long_gap] = np.nan
    return resampled, short_gap.sum(), long_gap.sum()

log("\n[2] Resampling ENTSO-E data to hourly using cubic spline...")
datasets = {
    'solar_mw':        solar,
    'load_mw':         load,
    'wind_mw':         wind,
    'hydro_mw':        hydro,
    'nuclear_mw':      nuclear,
    'price_eur_mwh':   prices,
    'gen_forecast_mw': gen_fc,
    'flow_de_mw':      flow_de,
    'flow_at_mw':      flow_at,
    'flow_sk_mw':      flow_sk,
    'flow_pl_mw':      flow_pl,
}

resampled_series = {}
for name, df in datasets.items():
    s = to_series(df, name)
    result, short_filled, long_dropped = cubic_resample(s)
    resampled_series[name] = result
    log(f"  {name}: short gaps filled={short_filled}, long gaps dropped={long_dropped}")

log("\n[3] Physical validation of solar generation...")
solar_h = resampled_series['solar_mw'].copy()

temp_idx = solar_h.index
elev, _ = solar_elevation_calc(temp_idx)
elev_s = pd.Series(elev, index=temp_idx)

night_mask = elev_s <= SOLAR_ELEV_THRESHOLD
night_nonzero = (solar_h[night_mask] > 0).sum()
solar_h[night_mask] = 0
log(f"  Night hours forced to zero: {night_nonzero} values corrected")

neg_count = (solar_h < 0).sum()
solar_h = solar_h.clip(lower=0)
log(f"  Negative values clipped to zero: {neg_count} values corrected")

hourly_diff = solar_h.diff().abs()
ramp_violations = (hourly_diff > MAX_RAMP_RATE).sum()
log(f"  Ramp rate violations (>{MAX_RAMP_RATE} MW/h): {ramp_violations}")
if ramp_violations > 0:
    violation_idx = hourly_diff[hourly_diff > MAX_RAMP_RATE].index
    for idx in violation_idx:
        pos = solar_h.index.get_loc(idx)
        if 0 < pos < len(solar_h) - 1:
            prev_val = solar_h.iloc[pos - 1]
            next_val = solar_h.iloc[pos + 1]
            solar_h.iloc[pos] = (prev_val + next_val) / 2
    log(f"  Ramp violations corrected by linear interpolation")

daylight_solar = solar_h[~night_mask]
mean_s = daylight_solar.mean()
std_s = daylight_solar.std()
outlier_mask = (solar_h > mean_s + 3 * std_s) & (~night_mask)
outlier_count = outlier_mask.sum()
solar_h[outlier_mask] = mean_s + 3 * std_s
log(f"  Outliers (>3 sigma daylight): {outlier_count} values capped")

resampled_series['solar_mw'] = solar_h

log("\n[4] Aggregating Open-Meteo weather across 15 Czech cities...")
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
log(f"  Weather aggregated: {agg.shape[1]} features across {len(agg)} hours")

log("\n[5] Building master feature table...")
master = pd.DataFrame(index=resampled_series['solar_mw'].index)
for name, s in resampled_series.items():
    master[name] = s

master = master.join(agg, how='left')

master['hour']        = master.index.hour
master['day_of_year'] = master.index.dayofyear
master['month']       = master.index.month
master['is_weekend']  = (master.index.dayofweek >= 5).astype(int)
master['sin_hour']    = np.sin(2 * np.pi * master['hour'] / 24)
master['cos_hour']    = np.cos(2 * np.pi * master['hour'] / 24)
master['sin_doy']     = np.sin(2 * np.pi * master['day_of_year'] / 365)
master['cos_doy']     = np.cos(2 * np.pi * master['day_of_year'] / 365)

elev_full, decl_full = solar_elevation_calc(master.index)
master['solar_elevation']   = elev_full
master['solar_declination'] = decl_full
master['daylight_flag']     = (master['solar_elevation'] > SOLAR_ELEV_THRESHOLD).astype(int)

rad_col = 'shortwave_radiation_mean'
if rad_col in master.columns:
    daily_max = master[rad_col].resample('D').transform('max')
    daily_max = daily_max.replace(0, np.nan)
    master['clearness_index'] = (master[rad_col] / daily_max).clip(0, 1)

master['installed_capacity_mw'] = get_installed_capacity(master.index)
master['capacity_factor'] = (
    master['solar_mw'] / master['installed_capacity_mw']
).clip(0, 1)

log("\n[6] Adding lag features...")
master['solar_lag_1h']   = master['solar_mw'].shift(1)
master['solar_lag_24h']  = master['solar_mw'].shift(24)
master['solar_lag_168h'] = master['solar_mw'].shift(168)

log("\n[7] Adding rolling statistics...")
master['radiation_roll3h_mean']  = master[rad_col].rolling(3,  min_periods=1).mean()
master['radiation_roll24h_mean'] = master[rad_col].rolling(24, min_periods=1).mean()
master['cloud_roll3h_mean']      = master['cloud_cover_mean'].rolling(3, min_periods=1).mean()

log("\n[8] Filtering to 2023-2025 and final cleaning...")
master = master['2023-01-01':'2025-12-31']
master = master[master.index.notna()]

log("\n[9] Timestamp alignment verification...")
expected_hours = pd.date_range('2023-01-01', '2025-12-31 23:00', freq='h', tz='UTC')
missing_hours = expected_hours.difference(master.index)
extra_hours = master.index.difference(expected_hours)
log(f"  Expected hours: {len(expected_hours)}")
log(f"  Actual hours:   {len(master)}")
log(f"  Missing hours:  {len(missing_hours)}")
log(f"  Extra hours:    {len(extra_hours)}")

log("\n[10] Final data quality summary...")
log(f"  Master table shape: {master.shape}")
log(f"  Date range: {master.index.min()} to {master.index.max()}")
missing = master.isna().sum()
missing_cols = missing[missing > 0]
if len(missing_cols) > 0:
    log(f"  Columns with missing values:")
    for col, cnt in missing_cols.items():
        log(f"    {col}: {cnt} missing ({cnt/len(master)*100:.1f}%)")
else:
    log("  No missing values in any column.")

log(f"\n  Solar MW statistics:")
log(f"    Min: {master['solar_mw'].min():.2f} MW")
log(f"    Max: {master['solar_mw'].max():.2f} MW")
log(f"    Mean: {master['solar_mw'].mean():.2f} MW")
log(f"    Installed capacity range: {master['installed_capacity_mw'].min():.0f} - {master['installed_capacity_mw'].max():.0f} MW")

log("\n" + "=" * 60)
log("DATA QUALITY REPORT COMPLETE")
log("=" * 60)

master.to_parquet(OUTPUT)
log(f"\nSaved to {OUTPUT}")

with open(REPORT, 'w') as f:
    f.write('\n'.join(report_lines))
log(f"Report saved to {REPORT}")
