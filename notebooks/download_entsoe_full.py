from entsoe import EntsoePandasClient
import pandas as pd
from pathlib import Path
import time

API_TOKEN  = "REPLACE_THIS"
START      = pd.Timestamp("2023-01-01", tz="Europe/Prague")
END        = pd.Timestamp("2025-12-31", tz="Europe/Prague")
COUNTRY    = "CZ"
OUTPUT_DIR = Path("data")
SLEEP      = 2.2

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
client = EntsoePandasClient(api_key=API_TOKEN)

def safe_download(label, func, *args, **kwargs):
    print(f"\nDownloading: {label}...")
    try:
        time.sleep(SLEEP)
        result = func(*args, **kwargs)
        print(f"  Success — shape: {result.shape}")
        return result
    except Exception as e:
        print(f"  FAILED: {e}")
        return None

def save(df, filename):
    if df is not None:
        path = OUTPUT_DIR / filename
        if hasattr(df, 'to_frame'):
            df = df.to_frame()
        df.to_parquet(path)
        print(f"  Saved to {path}")

df = safe_download("Actual Total Load", client.query_load, country_code=COUNTRY, start=START, end=END)
save(df, "entsoe_cz_load.parquet")

df = safe_download("Wind Generation", client.query_generation, country_code=COUNTRY, start=START, end=END, psr_type="B19")
save(df, "entsoe_cz_wind.parquet")

df = safe_download("Hydro Generation", client.query_generation, country_code=COUNTRY, start=START, end=END, psr_type="B11")
save(df, "entsoe_cz_hydro.parquet")

df = safe_download("Nuclear Generation", client.query_generation, country_code=COUNTRY, start=START, end=END, psr_type="B14")
save(df, "entsoe_cz_nuclear.parquet")

df = safe_download("Day-Ahead Prices", client.query_day_ahead_prices, country_code=COUNTRY, start=START, end=END)
save(df, "entsoe_cz_prices.parquet")

df = safe_download("Generation Forecast", client.query_generation_forecast, country_code=COUNTRY, start=START, end=END)
save(df, "entsoe_cz_generation_forecast.parquet")

df = safe_download("Cross-Border Flow CZ to DE", client.query_crossborder_flows, country_code_from="CZ", country_code_to="DE", start=START, end=END)
save(df, "entsoe_cz_de_flow.parquet")

df = safe_download("Cross-Border Flow CZ to AT", client.query_crossborder_flows, country_code_from="CZ", country_code_to="AT", start=START, end=END)
save(df, "entsoe_cz_at_flow.parquet")

df = safe_download("Cross-Border Flow CZ to SK", client.query_crossborder_flows, country_code_from="CZ", country_code_to="SK", start=START, end=END)
save(df, "entsoe_cz_sk_flow.parquet")

df = safe_download("Cross-Border Flow CZ to PL", client.query_crossborder_flows, country_code_from="CZ", country_code_to="PL", start=START, end=END)
save(df, "entsoe_cz_pl_flow.parquet")

print("\nAll downloads complete. Check data/ folder.")
