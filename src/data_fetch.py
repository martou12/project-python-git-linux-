from __future__ import annotations

import time
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import requests

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

COINGECKO_BASE = "https://api.coingecko.com/api/v3"

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
    ttl_sec: int = 300,
    retries: int = 6,
    session: Optional[requests.Session] = None,
    return_meta: bool = False,
) -> pd.Series | Tuple[pd.Series, dict]:
    """
    Fetch CoinGecko market_chart (prices) with:
    - cache disk TTL 5 min
    - retry/backoff + 429 handling
    - fallback to stale cache if API down
    """
    if vs not in {"eur", "usd"}:
        raise ValueError("vs must be 'eur' or 'usd'")
    if days <= 0:
        raise ValueError("days must be > 0")

    cache_path = DATA_DIR / f"{coin_id}_{vs}_{days}d.csv"

    # cache fresh
    if cache_path.exists():
        age = time.time() - cache_path.stat().st_mtime
        if age <= ttl_sec:
            s = _read_cache(cache_path)
            if s is not None and len(s) > 2:
                meta = {"source": "cache_fresh", "cache_path": str(cache_path), "age_sec": age}
                return (s, meta) if return_meta else s

    sess = session or requests.Session()
    sess.headers.update(_DEFAULT_HEADERS)

    url = f"{COINGECKO_BASE}/coins/{coin_id}/market_chart"
    params = {"vs_currency": vs, "days": int(days)}

    last_err = None
    for attempt in range(retries):
        try:
            r = sess.get(url, params=params, timeout=25)

            if r.status_code == 429:
                retry_after = r.headers.get("Retry-After")
                sleep_s = float(retry_after) if retry_after else (2 ** attempt)
                time.sleep(min(sleep_s, 30.0))
                last_err = RuntimeError("CoinGecko rate limit (429)")
                continue

            r.raise_for_status()
            data = r.json()
            prices = data.get("prices", [])
            if not prices:
                raise RuntimeError(f"No prices returned for {coin_id}")

            df = pd.DataFrame(prices, columns=["ts_ms", "price"])
            df["datetime"] = pd.to_datetime(df["ts_ms"], unit="ms", utc=True).dt.tz_convert(None)
            s = df.set_index("datetime")["price"].astype(float).sort_index()

            # clean
            s.index = s.index.floor("1min")
            s = s[~s.index.duplicated(keep="last")]
            s = s.dropna()
            if len(s) < 5:
                raise RuntimeError("Not enough points returned.")

            _write_cache(cache_path, s)
            meta = {"source": "api", "cache_path": str(cache_path), "age_sec": 0.0}
            return (s, meta) if return_meta else s

        except Exception as e:
            last_err = e
            time.sleep(min(2 ** attempt, 10.0))

    # fallback stale cache
    s = _read_cache(cache_path)
    if s is not None and len(s) > 2:
        meta = {"source": "cache_stale", "cache_path": str(cache_path), "age_sec": time.time() - cache_path.stat().st_mtime}
        return (s, meta) if return_meta else s

    raise last_err


def resample_price(s: pd.Series, rule: str) -> pd.Series:
    if s is None or s.empty:
        raise ValueError("Empty series")
    s = s.sort_index()
    if rule == "raw":
        out = s.copy()
    else:
        out = s.resample(rule).last().ffill()
    out = out.dropna()
    if len(out) < 5:
        raise RuntimeError("Not enough points after resample")
    return out
