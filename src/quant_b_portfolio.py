from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import requests
import time


DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

# CoinGecko IDs (No API key required)
ASSET_MAP = {
    "Bitcoin (BTC)": "bitcoin",
    "Ethereum (ETH)": "ethereum",
    "Solana (SOL)": "solana",
    "BNB": "binancecoin",
    "XRP": "ripple",
    "Cardano (ADA)": "cardano",
}

COINGECKO_BASE = "https://api.coingecko.com/api/v3"


@dataclass(frozen=True)
class PortfolioResult:
    """
    Container for the results of the portfolio simulation.
    """
    prices: pd.DataFrame          # Individual asset prices
    portfolio: pd.Series          # Portfolio value (Base 100)
    returns: pd.Series            # Portfolio periodic returns
    corr: pd.DataFrame            # Correlation matrix of asset returns


def _coingecko_market_chart(
    coin_id: str,
    vs: str = "eur",
    days: int = 30,
    max_age_sec: int = 300,      # Cache duration (default: 5 min)
    retries: int = 5,
) -> pd.Series:
    """
    Fetch CoinGecko market_chart with:
    - local CSV cache (stored in data/)
    - retry + exponential backoff logic for 429/temporary errors
    - fallback to cached CSV if the API is strictly rate-limited
    """
    cache_path = DATA_DIR / f"{coin_id}_{vs}_{days}d.csv"

    # 1) Use cache if fresh enough
    if cache_path.exists():
        age = time.time() - cache_path.stat().st_mtime
        if age <= max_age_sec:
            df = pd.read_csv(cache_path, parse_dates=["datetime"], index_col="datetime")
            s = df["price"].astype(float)
            s.name = coin_id
            return s

    url = f"{COINGECKO_BASE}/coins/{coin_id}/market_chart"
    params = {"vs_currency": vs, "days": days}

    last_err = None
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=20)

            # Handle Rate Limit (HTTP 429)
            if r.status_code == 429:
                retry_after = r.headers.get("Retry-After")
                sleep_s = float(retry_after) if retry_after else (2 ** attempt)
                time.sleep(min(sleep_s, 30.0))
                last_err = requests.HTTPError(f"429 Too Many Requests (attempt {attempt+1})", response=r)
                continue

            r.raise_for_status()

            data = r.json()
            prices = data.get("prices", [])
            if not prices:
                raise RuntimeError(f"No data received for {coin_id}")

            df = pd.DataFrame(prices, columns=["ts_ms", "price"])
            # Convert timestamp (ms) to datetime
            df["datetime"] = pd.to_datetime(df["ts_ms"], unit="ms", utc=True).dt.tz_convert(None)

            s = df.set_index("datetime")["price"].sort_index()

            # Align index to the minute to reduce timestamp mismatch across assets
            s.index = s.index.floor("1min")
            s = s[~s.index.duplicated(keep="last")]

            s.name = coin_id

            # Save to local cache
            s.to_frame("price").to_csv(cache_path)

            return s

        except Exception as e:
            last_err = e
            time.sleep(min(2 ** attempt, 10.0))

    # 3) Final fallback: use cache even if it is stale (better than crashing)
    if cache_path.exists():
        df = pd.read_csv(cache_path, parse_dates=["datetime"], index_col="datetime")
        s = df["price"].astype(float)
        s.name = coin_id
        return s

    raise last_err


def fetch_prices_multi(coin_ids: list[str], vs: str = "eur", days: int = 30) -> pd.DataFrame:
    """
    Fetch historical prices for multiple assets and align them into a single DataFrame.
    """
    series = []
    for cid in coin_ids:
        s = _coingecko_market_chart(cid, vs=vs, days=days)
        series.append(s)

        # Local backup (useful + professional practice)
        out = DATA_DIR / f"{cid}_{vs}_{days}d.csv"
        s.to_frame("price").to_csv(out)

    prices = pd.concat(series, axis=1).sort_index()

    # Robust alignment: forward fill the last known value to fill gaps,
    # then drop only the initial rows that are still incomplete.
    prices = prices.ffill().dropna()

    return prices


def compute_portfolio(prices: pd.DataFrame, weights: np.ndarray, rebalance: str = "None") -> pd.Series:
    """
    Portfolio Simulation:
    - Initial Capital = 100
    - Constant holdings, with optional rebalancing (Daily/Weekly/Monthly)
    """
    if prices.empty:
        raise ValueError("Price dataframe is empty")

    w = np.array(weights, dtype=float)
    if (w < 0).any():
        raise ValueError("Weights must be >= 0")
    if w.sum() == 0:
        raise ValueError("Sum of weights cannot be 0")
    
    # Normalize weights just in case
    w = w / w.sum()

    capital = 100.0
    # Calculate initial quantity of each asset
    holdings = (capital * w) / prices.iloc[0].values
    values = [capital]

    idx = prices.index
    for t in range(1, len(prices)):
        # Calculate current portfolio value based on holdings
        capital = float(np.sum(holdings * prices.iloc[t].values))
        values.append(capital)

        # Rebalancing logic
        if rebalance != "None" and t < len(prices) - 1:
            if _should_rebalance(idx[t], idx[t + 1], rebalance):
                # Reset holdings to match target weights based on current capital
                holdings = (capital * w) / prices.iloc[t].values

    return pd.Series(values, index=prices.index, name="portfolio_value")


def _should_rebalance(dt_now: pd.Timestamp, dt_next: pd.Timestamp, freq: str) -> bool:
    """
    Helper to detect if a rebalancing event should occur between two timestamps.
    """
    if freq == "Daily":
        return dt_now.date() != dt_next.date()
    if freq == "Weekly":
        return dt_now.isocalendar().week != dt_next.isocalendar().week
    if freq == "Monthly":
        return (dt_now.year, dt_now.month) != (dt_next.year, dt_next.month)
    return False


def compute_metrics(portfolio: pd.Series) -> dict:
    """
    Calculate standard financial metrics: Annualized Return, Volatility, Sharpe, Max Drawdown.
    """
    rets = portfolio.pct_change().dropna()
    if rets.empty:
        return {"ann_return": 0.0, "ann_vol": 0.0, "sharpe": 0.0, "max_dd": 0.0}

    # Annualization based on the median time step of the data
    dt = portfolio.index.to_series().diff().dropna().median()
    seconds = max(dt.total_seconds(), 1.0)
    periods_per_year = (365.0 * 24 * 3600) / seconds

    mu = float(rets.mean())
    sigma = float(rets.std(ddof=1))

    ann_return = (1.0 + mu) ** periods_per_year - 1.0
    ann_vol = sigma * np.sqrt(periods_per_year)
    sharpe = 0.0 if ann_vol == 0 else (mu * periods_per_year) / (sigma * np.sqrt(periods_per_year))

    # Max Drawdown calculation
    roll_max = portfolio.cummax()
    dd = (portfolio / roll_max) - 1.0
    max_dd = float(dd.min())

    return {
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_dd": max_dd,
    }


def build_portfolio_result(asset_ids: list[str], vs: str, days: int, weights: np.ndarray, rebalance: str) -> PortfolioResult:
    """
    Main factory function to build the Portfolio Result object.
    """
    prices = fetch_prices_multi(asset_ids, vs=vs, days=days)
    port = compute_portfolio(prices, weights=weights, rebalance=rebalance)
    
    asset_returns = prices.pct_change().dropna()
    corr = asset_returns.corr()
    
    port_returns = port.pct_change().dropna()
    
    return PortfolioResult(prices=prices, portfolio=port, returns=port_returns, corr=corr)