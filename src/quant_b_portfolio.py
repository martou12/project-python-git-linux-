from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import requests

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

# CoinGecko IDs (no API key needed)
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
    prices: pd.DataFrame
    portfolio: pd.Series
    corr: pd.DataFrame


def _market_chart(coin_id: str, vs: str, days: int) -> pd.Series:
    url = f"{COINGECKO_BASE}/coins/{coin_id}/market_chart"
    params = {"vs_currency": vs, "days": days}

    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()

    prices = r.json().get("prices", [])
    if not prices:
        raise RuntimeError(f"No price data returned for {coin_id}")

    df = pd.DataFrame(prices, columns=["ts_ms", "price"])
    df["datetime"] = pd.to_datetime(df["ts_ms"], unit="ms", utc=True).dt.tz_convert(None)
    s = df.set_index("datetime")["price"].sort_index()
    s.name = coin_id
    return s


def fetch_prices_multi(coin_ids: list[str], vs: str = "eur", days: int = 30) -> pd.DataFrame:
    series = []
    for cid in coin_ids:
        s = _market_chart(cid, vs=vs, days=days)
        series.append(s)

        # local storage (professional + useful for debugging)
        out = DATA_DIR / f"{cid}_{vs}_{days}d.csv"
        s.to_frame("price").to_csv(out)

    prices = pd.concat(series, axis=1).dropna(how="any")
    return prices


def _should_rebalance(dt_now: pd.Timestamp, dt_next: pd.Timestamp, freq: str) -> bool:
    if freq == "Daily":
        return dt_now.date() != dt_next.date()
    if freq == "Weekly":
        return dt_now.isocalendar().week != dt_next.isocalendar().week
    if freq == "Monthly":
        return (dt_now.year, dt_now.month) != (dt_next.year, dt_next.month)
    return False


def compute_portfolio(prices: pd.DataFrame, weights: np.ndarray, rebalance: str = "None") -> pd.Series:
    w = np.array(weights, dtype=float)
    if (w < 0).any():
        raise ValueError("Weights must be >= 0")
    if w.sum() == 0:
        raise ValueError("Sum of weights is 0")
    w = w / w.sum()

    capital = 100.0
    holdings = (capital * w) / prices.iloc[0].values
    values = [capital]

    idx = prices.index
    for t in range(1, len(prices)):
        capital = float(np.sum(holdings * prices.iloc[t].values))
        values.append(capital)

        if rebalance != "None" and t < len(prices) - 1:
            if _should_rebalance(idx[t], idx[t + 1], rebalance):
                holdings = (capital * w) / prices.iloc[t].values

    return pd.Series(values, index=prices.index, name="portfolio_value")


def compute_metrics(portfolio: pd.Series) -> dict:
    rets = portfolio.pct_change().dropna()
    if rets.empty:
        return {"ann_return": 0.0, "ann_vol": 0.0, "sharpe": 0.0, "max_dd": 0.0}

    dt = portfolio.index.to_series().diff().dropna().median()
    seconds = max(dt.total_seconds(), 1.0)
    periods_per_year = (365.0 * 24 * 3600) / seconds

    mu = float(rets.mean())
    sigma = float(rets.std(ddof=1))

    ann_return = (1.0 + mu) ** periods_per_year - 1.0
    ann_vol = sigma * np.sqrt(periods_per_year)
    sharpe = 0.0 if ann_vol == 0 else (mu * periods_per_year) / (sigma * np.sqrt(periods_per_year))

    roll_max = portfolio.cummax()
    dd = (portfolio / roll_max) - 1.0
    max_dd = float(dd.min())

    return {"ann_return": ann_return, "ann_vol": ann_vol, "sharpe": sharpe, "max_dd": max_dd}


def build_portfolio_result(asset_ids: list[str], vs: str, days: int, weights: np.ndarray, rebalance: str) -> PortfolioResult:
    prices = fetch_prices_multi(asset_ids, vs=vs, days=days)
    portfolio = compute_portfolio(prices, weights=weights, rebalance=rebalance)
    corr = prices.pct_change().dropna().corr()
    return PortfolioResult(prices=prices, portfolio=portfolio, corr=corr)
