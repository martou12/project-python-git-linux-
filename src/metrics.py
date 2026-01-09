from __future__ import annotations

import numpy as np
import pandas as pd


def _periods_per_year(index: pd.DatetimeIndex) -> float:
    if len(index) < 3:
        return 0.0
    dt = pd.Series(index).diff().dropna().median()
    if pd.isna(dt):
        return 0.0
    seconds = max(float(dt.total_seconds()), 1.0)
    return (365.0 * 24 * 3600) / seconds


def max_drawdown(equity: pd.Series) -> float:
    if equity is None or equity.empty:
        return 0.0
    peak = equity.cummax()
    dd = (equity / peak) - 1.0
    return float(dd.min())


def drawdown_series(equity: pd.Series) -> pd.Series:
    if equity is None or equity.empty:
        return pd.Series(dtype=float)
    peak = equity.cummax()
    dd = (equity / peak) - 1.0
    dd.name = "drawdown"
    return dd


def compute_metrics(equity: pd.Series, rf_annual: float = 0.0) -> dict:
    """
    equity: valeur cumulée (base 100)
    rf_annual: taux sans risque annuel (ex 0.02)
    """
    if equity is None or equity.empty or len(equity) < 5:
        return {
            "total_return": 0.0,
            "cagr": 0.0,
            "ann_vol": 0.0,
            "sharpe": 0.0,
            "sortino": 0.0,
            "calmar": 0.0,
            "max_dd": 0.0,
        }

    equity = equity.sort_index()
    rets = equity.pct_change().dropna()
    if rets.empty:
        mdd = max_drawdown(equity)
        return {
            "total_return": float(equity.iloc[-1] / equity.iloc[0] - 1.0),
            "cagr": 0.0,
            "ann_vol": 0.0,
            "sharpe": 0.0,
            "sortino": 0.0,
            "calmar": 0.0,
            "max_dd": mdd,
        }

    ppy = _periods_per_year(equity.index)
    if ppy <= 0:
        mdd = max_drawdown(equity)
        return {
            "total_return": float(equity.iloc[-1] / equity.iloc[0] - 1.0),
            "cagr": 0.0,
            "ann_vol": 0.0,
            "sharpe": 0.0,
            "sortino": 0.0,
            "calmar": 0.0,
            "max_dd": mdd,
        }

    n = len(rets)
    total_return = float(equity.iloc[-1] / equity.iloc[0] - 1.0)
    cagr = float((equity.iloc[-1] / equity.iloc[0]) ** (ppy / n) - 1.0)

    ann_vol = float(rets.std(ddof=1) * np.sqrt(ppy)) if rets.std(ddof=1) > 0 else 0.0

    # Sharpe (avec rf annuel)
    excess = cagr - float(rf_annual)
    sharpe = float(excess / ann_vol) if ann_vol > 0 else 0.0

    # Sortino
    downside = rets[rets < 0]
    dd_std = float(downside.std(ddof=1) * np.sqrt(ppy)) if len(downside) > 1 else 0.0
    sortino = float(excess / dd_std) if dd_std > 0 else 0.0

    mdd = max_drawdown(equity)
    calmar = float(cagr / abs(mdd)) if mdd < 0 else 0.0

    return {
        "total_return": total_return,
        "cagr": cagr,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "max_dd": mdd,
    }


def open_close_return_24h(price: pd.Series) -> dict:
    s = price.dropna().sort_index()
    if s.empty:
        return {"open_24h": 0.0, "close_24h": 0.0, "return_24h": 0.0}

    end = s.index.max()
    start = end - pd.Timedelta(hours=24)
    w = s.loc[s.index >= start]

    if len(w) < 2:
        o = float(s.iloc[0])
        c = float(s.iloc[-1])
        return {"open_24h": o, "close_24h": c, "return_24h": float(c / o - 1.0)}

    o = float(w.iloc[0])
    c = float(w.iloc[-1])
    return {"open_24h": o, "close_24h": c, "return_24h": float(c / o - 1.0)}


def realized_vol(price: pd.Series) -> dict:
    s = price.dropna().sort_index()
    lr = np.log(s).diff().dropna()
    if lr.empty:
        return {"vol_step": 0.0, "ann_vol_est": 0.0}

    vol_step = float(lr.std(ddof=1))
    ppy = _periods_per_year(s.index)
    ann_vol_est = float(vol_step * np.sqrt(ppy)) if ppy > 0 else 0.0
    return {"vol_step": vol_step, "ann_vol_est": ann_vol_est}


def trade_stats(position: pd.Series) -> dict:
    """
    position: série en {-1,0,1} ou [0,1]
    """
    if position is None or position.empty:
        return {"trades": 0, "turnover": 0.0}

    p = position.fillna(0.0).astype(float)
    dp = p.diff().abs().fillna(0.0)
    trades = int((dp > 0).sum())
    turnover = float(dp.sum())
    return {"trades": trades, "turnover": turnover}
