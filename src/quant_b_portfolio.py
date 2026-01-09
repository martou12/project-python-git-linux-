from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import requests
import time


DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

# CoinGecko IDs (pas besoin de clé API)
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
    prices: pd.DataFrame          # prix des assets
    portfolio: pd.Series          # valeur portefeuille (base 100)
    returns: pd.Series            # rendements portefeuille
    corr: pd.DataFrame            # corrélation des rendements assets


def _coingecko_market_chart(
    coin_id: str,
    vs: str = "eur",
    days: int = 30,
    max_age_sec: int = 300,      # cache 5 min
    retries: int = 5,
) -> pd.Series:
    """
    Fetch CoinGecko market_chart with:
    - local CSV cache (data/)
    - retry + exponential backoff on 429/temporary errors
    - fallback to cached CSV if API is rate-limited
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

            # Rate limit (429)
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
                raise RuntimeError(f"no data received for  {coin_id}")

            df = pd.DataFrame(prices, columns=["ts_ms", "price"])
            df["datetime"] = pd.to_datetime(df["ts_ms"], unit="ms", utc=True).dt.tz_convert(None)

            s = df.set_index("datetime")["price"].sort_index()

            # align index to minute to reduce timestamp mismatch
            s.index = s.index.floor("1min")
            s = s[~s.index.duplicated(keep="last")]

            s.name = coin_id

            # save cache
            s.to_frame("price").to_csv(cache_path)

            return s

        except Exception as e:
            last_err = e
            time.sleep(min(2 ** attempt, 10.0))

    # 3) Final fallback: use cache even if old
    if cache_path.exists():
        df = pd.read_csv(cache_path, parse_dates=["datetime"], index_col="datetime")
        s = df["price"].astype(float)
        s.name = coin_id
        return s

    raise last_err


def fetch_prices_multi(coin_ids: list[str], vs: str = "eur", days: int = 30) -> pd.DataFrame:
    series = []
    for cid in coin_ids:
        s = _coingecko_market_chart(cid, vs=vs, days=days)
        series.append(s)

        # Sauvegarde locale (utile + “pro”)
        out = DATA_DIR / f"{cid}_{vs}_{days}d.csv"
        s.to_frame("price").to_csv(out)

    prices = pd.concat(series, axis=1).sort_index()

    # Alignement robuste : on propage la dernière valeur connue
    # puis on supprime seulement les premières lignes incomplètes
    prices = prices.ffill().dropna()

    return prices


    


def compute_portfolio(prices: pd.DataFrame, weights: np.ndarray, rebalance: str = "None") -> pd.Series:
    """
    Simulation simple :
    - capital initial = 100
    - holdings constants, rebalancing optionnel (Daily/Weekly/Monthly)
    """
    if prices.empty:
        raise ValueError("prices est vide")

    w = np.array(weights, dtype=float)
    if (w < 0).any():
        raise ValueError("Les poids doivent être >= 0")
    if w.sum() == 0:
        raise ValueError("Somme des poids = 0")
    w = w / w.sum()

    capital = 100.0
    holdings = (capital * w) / prices.iloc[0].values  # quantité de chaque asset
    values = [capital]

    idx = prices.index
    for t in range(1, len(prices)):
        capital = float(np.sum(holdings * prices.iloc[t].values))
        values.append(capital)

        if rebalance != "None" and t < len(prices) - 1:
            if _should_rebalance(idx[t], idx[t + 1], rebalance):
                holdings = (capital * w) / prices.iloc[t].values

    return pd.Series(values, index=prices.index, name="portfolio_value")


def _should_rebalance(dt_now: pd.Timestamp, dt_next: pd.Timestamp, freq: str) -> bool:
    if freq == "Daily":
        return dt_now.date() != dt_next.date()
    if freq == "Weekly":
        return dt_now.isocalendar().week != dt_next.isocalendar().week
    if freq == "Monthly":
        return (dt_now.year, dt_now.month) != (dt_next.year, dt_next.month)
    return False


def compute_metrics(portfolio: pd.Series) -> dict:
    rets = portfolio.pct_change().dropna()
    if rets.empty:
        return {"ann_return": 0.0, "ann_vol": 0.0, "sharpe": 0.0, "max_dd": 0.0}

    # annualisation basée sur le pas de temps médian
    dt = portfolio.index.to_series().diff().dropna().median()
    seconds = max(dt.total_seconds(), 1.0)
    periods_per_year = (365.0 * 24 * 3600) / seconds

    mu = float(rets.mean())
    sigma = float(rets.std(ddof=1))

    ann_return = (1.0 + mu) ** periods_per_year - 1.0
    ann_vol = sigma * np.sqrt(periods_per_year)
    sharpe = 0.0 if ann_vol == 0 else (mu * periods_per_year) / (sigma * np.sqrt(periods_per_year))

    # max drawdown
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
    prices = fetch_prices_multi(asset_ids, vs=vs, days=days)
    port = compute_portfolio(prices, weights=weights, rebalance=rebalance)
    asset_returns = prices.pct_change().dropna()
    corr = asset_returns.corr()
    port_returns = port.pct_change().dropna()
    return PortfolioResult(prices=prices, portfolio=port, returns=port_returns, corr=corr)
