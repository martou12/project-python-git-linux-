from __future__ import annotations

import numpy as np
import pandas as pd


def max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    roll_max = equity.cummax()
    dd = (equity / roll_max) - 1.0
    return float(dd.min())


def _periods_per_year_from_index(index: pd.DatetimeIndex) -> float:
    if len(index) < 3:
        return 0.0
    dt = pd.Series(index).diff().dropna().median()
    seconds = max(float(dt.total_seconds()), 1.0)
    return (365.0 * 24 * 3600) / seconds


def compute_metrics(equity: pd.Series) -> dict:
    """
    equity: valeur cumulée (ex base 100)
    """
    if equity.empty or len(equity) < 3:
        return {"ann_return": 0.0, "ann_vol": 0.0, "sharpe": 0.0, "max_dd": 0.0}

    rets = equity.pct_change().dropna()
    if rets.empty:
        return {"ann_return": 0.0, "ann_vol": 0.0, "sharpe": 0.0, "max_dd": 0.0}

    ppy = _periods_per_year_from_index(equity.index)
    if ppy <= 0:
        return {"ann_return": 0.0, "ann_vol": 0.0, "sharpe": 0.0, "max_dd": max_drawdown(equity)}

    mu = float(rets.mean())
    sigma = float(rets.std(ddof=1))

    ann_return = (1.0 + mu) ** ppy - 1.0
    ann_vol = sigma * np.sqrt(ppy)
    sharpe = 0.0 if ann_vol == 0 else (mu * ppy) / (sigma * np.sqrt(ppy))

    return {
        "ann_return": float(ann_return),
        "ann_vol": float(ann_vol),
        "sharpe": float(sharpe),
        "max_dd": max_drawdown(equity),
    }


def open_close_return_24h(price: pd.Series) -> dict:
    """
    approx: open/close sur dernière fenêtre 24h
    """
    s = price.dropna()
    if s.empty:
        return {"open_24h": 0.0, "close_24h": 0.0, "return_24h": 0.0}

    end = s.index.max()
    start = end - pd.Timedelta(hours=24)
    w = s.loc[s.index >= start]
    if len(w) < 2:
        return {"open_24h": float(s.iloc[0]), "close_24h": float(s.iloc[-1]), "return_24h": float(s.iloc[-1] / s.iloc[0] - 1.0)}

    o = float(w.iloc[0])
    c = float(w.iloc[-1])
    r = (c / o) - 1.0
    return {"open_24h": o, "close_24h": c, "return_24h": float(r)}


def realized_vol(price: pd.Series) -> dict:
    """
    vol réalisée (std log-returns) + annualisation via pas de temps médian
    """
    s = price.dropna()
    lr = np.log(s).diff().dropna()
    if lr.empty:
        return {"vol_step": 0.0, "ann_vol_est": 0.0}

    vol_step = float(lr.std(ddof=1))
    ppy = _periods_per_year_from_index(s.index)
    ann_vol_est = float(vol_step * np.sqrt(ppy)) if ppy > 0 else 0.0

    return {"vol_step": vol_step, "ann_vol_est": ann_vol_est}
