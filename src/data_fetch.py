from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
import requests
import pandas as pd

# Define the local directory for caching CSV files
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

# Base URL for CoinGecko API
COINGECKO_BASE = "https://api.coingecko.com/api/v3"

# Mapping between display names and CoinGecko API IDs
ASSET_MAP = {
    "Bitcoin (BTC)": "bitcoin",
    "Ethereum (ETH)": "ethereum",
    "Solana (SOL)": "solana",
    "BNB": "binancecoin",
    "XRP": "ripple",
    "Cardano (ADA)": "cardano",
}


def _coingecko_market_chart(
    coin_id: str,
    vs: str = "eur",
    days: int = 30,
    max_age_sec: int = 300,  # Cache duration (default: 5 minutes)
    retries: int = 5,
) -> pd.Series:
    """
    Fetch CoinGecko market_chart data:
    - Checks for local CSV cache in data/ first.
    - Implements retry logic + exponential backoff for temporary network errors or Rate Limits (429).
    - Returns a pandas Series with a DateTime index.
    """
    cache_path = DATA_DIR / f"{coin_id}_{vs}_{days}d.csv"

    # 1) Check for "fresh" cache (file exists and is recent enough)
    if cache_path.exists():
        age = time.time() - cache_path.stat().st_mtime
        if age <= max_age_sec:
            df = pd.read_csv(cache_path, parse_dates=["datetime"], index_col="datetime")
            s = df["price"].astype(float)
            s.name = coin_id
            return s

    # 2) If no cache, prepare API request
    url = f"{COINGECKO_BASE}/coins/{coin_id}/market_chart"
    params = {"vs_currency": vs, "days": days}

    last_err = None
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=20)

            # Handle Rate Limiting (HTTP 429)
            if r.status_code == 429:
                retry_after = r.headers.get("Retry-After")
                # Calculate sleep time: use header if available, else exponential backoff
                sleep_s = float(retry_after) if retry_after else (2 ** attempt)
                time.sleep(min(sleep_s, 30.0)) # Cap wait time at 30s
                last_err = requests.HTTPError(f"429 Too Many Requests (attempt {attempt+1})", response=r)
                continue

            r.raise_for_status()
            data = r.json()
            prices = data.get("prices", [])
            
            if not prices:
                raise RuntimeError(f"No data received for {coin_id}")

            # Process raw data into a DataFrame
            df = pd.DataFrame(prices, columns=["ts_ms", "price"])
            # Convert timestamp (ms) to datetime
            df["datetime"] = pd.to_datetime(df["ts_ms"], unit="ms", utc=True).dt.tz_convert(None)

            # Clean and sort index
            s = df.set_index("datetime")["price"].sort_index()
            
            # Align timestamps to minutes and remove duplicates
            s.index = s.index.floor("1min")
            s = s[~s.index.duplicated(keep="last")]
            s.name = coin_id

            # Save to local cache for next time
            s.to_frame("price").to_csv(cache_path)
            return s

        except Exception as e:
            last_err = e
            # Wait before retrying on general errors (exponential backoff)
            time.sleep(min(2 ** attempt, 10.0))

    # 3) Fallback: If all retries fail, try to load stale cache (even if old)
    if cache_path.exists():
        df = pd.read_csv(cache_path, parse_dates=["datetime"], index_col="datetime")
        s = df["price"].astype(float)
        s.name = coin_id
        return s

    # If no data and no cache, raise the last encountered error
    raise last_err


def fetch_price_series(coin_id: str, vs: str = "eur", days: int = 30) -> pd.Series:
    """
    Public wrapper to get a clean price series (datetime index) for a single asset.
    """
    s = _coingecko_market_chart(coin_id, vs=vs, days=days)
    return s.dropna().astype(float)


def resample_price(s: pd.Series, rule: str) -> pd.Series:
    """
    Robust resampling function (last value + forward fill).
    
    Args:
        s: Input pandas Series (prices).
        rule: Resampling rule (e.g., "5min", "15min", "1H", "4H", "1D").
              Use "raw" to return original data.
    """
    if s.empty:
        raise ValueError("Price series is empty")
    
    if rule == "raw":
        out = s.copy()
    else:
        # Resample logic: take the last price of the bin, then fill forward gaps
        out = s.resample(rule).last().ffill()
    
    out = out.dropna()
    return out