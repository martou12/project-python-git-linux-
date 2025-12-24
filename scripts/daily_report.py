import sys
from pathlib import Path

# Ensure project root is on PYTHONPATH (so "import src" works when running from /scripts)
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


import argparse
import json
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from src.quant_b_portfolio import (
    fetch_prices_multi,
    compute_portfolio,
    compute_metrics,
)

REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(exist_ok=True)

LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)


def _window_last_24h(prices: pd.DataFrame) -> pd.DataFrame:
    end = prices.index.max()
    start = end - timedelta(hours=24)
    w = prices.loc[prices.index >= start]
    if len(w) < 5:
        raise RuntimeError("Pas assez de points sur les dernières 24h.")
    return w


def _open_close_and_return_24h(series: pd.Series) -> dict:
    s = series.dropna()
    o = float(s.iloc[0])
    c = float(s.iloc[-1])
    r = (c / o) - 1.0
    return {"open_24h": o, "close_24h": c, "return_24h": r}


def _realized_vol(series: pd.Series) -> dict:
    # Vol réalisée sur la fenêtre (std des log-returns) + annualisation approximative selon la granularité
    s = series.dropna()
    lr = np.log(s).diff().dropna()
    if lr.empty:
        return {"vol_24h": 0.0, "ann_vol_est": 0.0}

    vol_24h = float(lr.std(ddof=1))  # volatilité “par pas de temps”
    # Estimation annualisée en déduisant le pas de temps médian
    dt = s.index.to_series().diff().dropna().median()
    seconds = max(dt.total_seconds(), 1.0)
    periods_per_year = (365.0 * 24 * 3600) / seconds
    ann_vol_est = float(lr.std(ddof=1) * np.sqrt(periods_per_year))
    return {"vol_24h": vol_24h, "ann_vol_est": ann_vol_est}


def main() -> int:
    parser = argparse.ArgumentParser(description="Daily report (CoinGecko) for BTC/ETH/SOL portfolio.")
    parser.add_argument("--assets", nargs="+", default=["bitcoin", "ethereum", "solana"])
    parser.add_argument("--vs", default="eur", choices=["eur", "usd"])
    parser.add_argument("--history_days", type=int, default=30, help="Days of history to fetch (for drawdown/corr).")
    parser.add_argument("--rebalance", default="None", choices=["None", "Daily", "Weekly", "Monthly"])
    args = parser.parse_args()

    assets = args.assets
    vs = args.vs

    # Fetch history (30d by default) – one API call per asset
    prices = fetch_prices_multi(assets, vs=vs, days=args.history_days)

    # Portfolio weights = equal-weight (Quant B baseline)
    w = np.ones(len(assets), dtype=float) / len(assets)
    portfolio = compute_portfolio(prices, weights=w, rebalance=args.rebalance)

    # Last 24h metrics (per asset + portfolio)
    w_prices = _window_last_24h(prices)
    w_port = portfolio.loc[w_prices.index.min():]

    asset_stats = {}
    for col in w_prices.columns:
        oc = _open_close_and_return_24h(w_prices[col])
        vol = _realized_vol(w_prices[col])
        asset_stats[col] = {**oc, **vol}

    port_stats = {
        **_open_close_and_return_24h(w_port),
        **_realized_vol(w_port),
        **compute_metrics(portfolio),  # includes ann_return, ann_vol, sharpe, max_dd (over full history)
    }

    corr = prices.pct_change().dropna().corr()

    report = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "base_currency": vs,
        "assets": assets,
        "weights_equal": (w * 100).round(2).tolist(),
        "rebalance": args.rebalance,
        "asset_last_24h": asset_stats,
        "portfolio": port_stats,
        "corr_matrix": corr.round(4).to_dict(),
    }

    out_name = f"daily_report_{datetime.now().strftime('%Y-%m-%d_%H%M%S')}_{vs}.json"
    out_path = REPORTS_DIR / out_name
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"[OK] Report written: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
