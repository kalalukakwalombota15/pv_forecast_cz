from entsoe import EntsoePandasClient
import pandas as pd
from pathlib import Path

API_TOKEN = "YOUR_TOKEN_HERE"
START     = pd.Timestamp("2023-01-01", tz="Europe/Prague")
END       = pd.Timestamp("2025-12-31", tz="Europe/Prague")
COUNTRY   = "CZ"
OUTPUT    = Path("data/entsoe_cz_solar_raw.parquet")

client = EntsoePandasClient(api_key=API_TOKEN)

print("Downloading ENTSO-E CZ solar generation 2023-2025...")
ts = client.query_generation(
    country_code=COUNTRY,
    start=START,
    end=END,
    psr_type="B16"
)

print(f"Raw shape: {ts.shape}")
print(ts.head())

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
ts.to_parquet(OUTPUT)
print(f"Saved to {OUTPUT}")
