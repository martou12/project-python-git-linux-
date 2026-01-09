from __future__ import annotations

import time
from pathlib import Path
from typing import Optional, Tuple

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

_DEFAULT_HEADERS = {
    "User-Agent": "project-python-git-linux (student dashboard) - contact: none",
    "Accept": "application/json",
}


def _read_cache(path: Path) -> Optional[pd.Series]:
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=["datetime"])
    if df.empty or "price" not in df.columns:
        return None
    s = df.set_index("datetime")["price"].astype(float).sort_index()
    s = s[~s.index.duplicated(keep="last")]
    return s


def _write_cache(path: Path, s: pd.Series) -> None:
    out = s.to_frame("price").reset_index().rename(columns={"index": "datetime"})
    out.to_csv(path, index=False)


def fetch_price_series(
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
    if vs not in {"eur", "usd"}:
        raise ValueError("vs must be 'eur' or 'usd'")
    if days <= 0:
        raise ValueError("days must be > 0")

    cache_path = DATA_DIR / f"{coin_id}_{vs}_{days}d.csv"

    # 1) Check for "fresh" cache (file exists and is recent enough)
    if cache_path.exists():
        age = time.time() - cache_path.stat().st_mtime
        if age <= ttl_sec:
            s = _read_cache(cache_path)
            if s is not None and len(s) > 2:
                meta = {"source": "cache_fresh", "cache_path": str(cache_path), "age_sec": age}
                return (s, meta) if return_meta else s

    sess = session or requests.Session()
    sess.headers.update(_DEFAULT_HEADERS)

    # 2) If no cache, prepare API request
    url = f"{COINGECKO_BASE}/coins/{coin_id}/market_chart"
    params = {"vs_currency": vs, "days": int(days)}

    last_err = None
    for attempt in range(retries):
        try:
            r = sess.get(url, params=params, timeout=25)

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
            s = df.set_index("datetime")["price"].astype(float).sort_index()

            # Clean and sort index
            s = df.set_index("datetime")["price"].sort_index()
            
            # Align timestamps to minutes and remove duplicates
            s.index = s.index.floor("1min")
            s = s[~s.index.duplicated(keep="last")]
            s = s.dropna()
            if len(s) < 5:
                raise RuntimeError("Not enough points returned.")

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


def fetch_price_series(coin_id: str, vs: str = "eur", days: int = 30, return_meta: bool = False, **kwargs):
    """
    Public wrapper to get a clean price series.
    Compatible with both Quant A (needs meta) and Quant B (needs just prices).
    """
    s = _coingecko_market_chart(coin_id, vs=vs, days=days)
    s = s.dropna().astype(float)
    
    
    if return_meta:
        return s, {"source": "CoinGecko", "currency": vs, "days": days}
        
    
    return s


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
