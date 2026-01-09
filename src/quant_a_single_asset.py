from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

from src.data_fetch import fetch_price_series, resample_price
from src.metrics import trade_stats


@dataclass(frozen=True)
class SingleAssetResult:
    asset_id: str
    vs: str
    prices: pd.Series
    equity: pd.Series
    position: pd.Series
    strategy_name: str
    params: dict
    meta: dict


def _apply_costs_and_build_equity(
    prices: pd.Series,
    position: pd.Series,
    leverage: float = 1.0,
    fee_bps: float = 0.0,
    slippage_bps: float = 0.0,
) -> pd.Series:
    """
    equity_t = equity_{t-1} * (1 + leverage * pos_{t-1} * ret_t - turnover_t * cost)
    cost in bps applied on turnover (abs change in position)
    """
    prices = prices.sort_index()
    rets = prices.pct_change().fillna(0.0)

    pos = position.fillna(0.0).astype(float).clip(-1.0, 1.0)
    turnover = pos.diff().abs().fillna(0.0)

    cost_rate = (fee_bps + slippage_bps) / 10000.0
    strat_rets = leverage * pos.shift(1).fillna(0.0) * rets - turnover * cost_rate

    equity = 100.0 * (1.0 + strat_rets).cumprod()
    equity.name = "equity"
    return equity


def _buy_and_hold(prices: pd.Series) -> pd.Series:
    return pd.Series(1.0, index=prices.index, name="position")


def _momentum(prices: pd.Series, lookback: int, allow_short: bool) -> pd.Series:
    if lookback < 1:
        raise ValueError("lookback must be >= 1")

    ret_lb = prices / prices.shift(lookback) - 1.0

    if allow_short:
        sig = np.sign(ret_lb).astype(float)     # -1 / 0 / +1
        pos = pd.Series(sig, index=prices.index).shift(1).fillna(0.0)
    else:
        pos = (ret_lb > 0).astype(float).shift(1).fillna(0.0)

    pos.name = "position"
    return pos


def _sma_crossover(prices: pd.Series, short: int, long: int, allow_short: bool):
    if short < 1 or long < 2 or short >= long:
        raise ValueError("Need 1 <= short < long")

    sma_s = prices.rolling(short).mean()
    sma_l = prices.rolling(long).mean()

    if allow_short:
        pos = np.where(sma_s > sma_l, 1.0, -1.0)
        pos = pd.Series(pos, index=prices.index).shift(1).fillna(0.0)
    else:
        pos = (sma_s > sma_l).astype(float).shift(1).fillna(0.0)

    pos.name = "position"
    return pos, sma_s, sma_l


def build_single_asset_result(
    asset_id: str,
    vs: str = "eur",
    days: int = 30,
    periodicity: str = "raw",
    strategy: str = "Momentum",
    lookback: int = 20,
    sma_short: int = 10,
    sma_long: int = 30,
    allow_short: bool = False,
    leverage: float = 1.0,
    fee_bps: float = 0.0,
    slippage_bps: float = 0.0,
) -> SingleAssetResult:
    (prices_raw, meta) = fetch_price_series(asset_id, vs=vs, days=days, return_meta=True)
    prices = resample_price(prices_raw, periodicity)

    if len(prices) < 20:
        raise RuntimeError("Not enough points. Increase days or use periodicity=raw")

    params = {
        "days": int(days),
        "periodicity": periodicity,
        "allow_short": bool(allow_short),
        "leverage": float(leverage),
        "fee_bps": float(fee_bps),
        "slippage_bps": float(slippage_bps),
    }

    sma_s = sma_l = None

    if strategy == "Buy & Hold":
        pos = _buy_and_hold(prices)
    elif strategy == "Momentum":
        pos = _momentum(prices, lookback=int(lookback), allow_short=allow_short)
        params["lookback"] = int(lookback)
    elif strategy == "SMA Crossover":
        pos, sma_s, sma_l = _sma_crossover(prices, int(sma_short), int(sma_long), allow_short=allow_short)
        params["sma_short"] = int(sma_short)
        params["sma_long"] = int(sma_long)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    equity = _apply_costs_and_build_equity(
        prices=prices,
        position=pos,
        leverage=float(leverage),
        fee_bps=float(fee_bps),
        slippage_bps=float(slippage_bps),
    )

    # trade stats + keep meta about source/cache
    meta = {**meta, **trade_stats(pos)}

    # SMA overlay for UI (optional)
    if sma_s is not None and sma_l is not None:
        meta = {**meta, "sma_short": sma_s, "sma_long": sma_l}

    return SingleAssetResult(
        asset_id=asset_id,
        vs=vs,
        prices=prices,
        equity=equity,
        position=pos,
        strategy_name=strategy,
        params=params,
        meta=meta,
    )


def simple_linear_forecast(series: pd.Series, horizon: int = 20, fit_last: int = 200) -> pd.DataFrame:
    s = series.dropna().sort_index()
    if len(s) < 20:
        raise ValueError("Series too short for forecast")

    s_fit = s.iloc[-min(len(s), fit_last):]
    y = s_fit.values.astype(float)
    x = np.arange(len(y), dtype=float)

    a, b = np.polyfit(x, y, deg=1)
    y_hat_fit = a * x + b
    resid = y - y_hat_fit
    sigma = float(np.std(resid, ddof=1)) if len(resid) > 2 else 0.0

    dt = s_fit.index.to_series().diff().dropna().median()
    if pd.isna(dt):
        dt = pd.Timedelta(minutes=5)

    x_f = np.arange(len(y), len(y) + horizon, dtype=float)
    yhat = a * x_f + b

    idx_f = [s_fit.index[-1] + (i + 1) * dt for i in range(horizon)]
    lo = yhat - 1.96 * sigma
    hi = yhat + 1.96 * sigma

    return pd.DataFrame({"yhat": yhat, "lo": lo, "hi": hi}, index=pd.DatetimeIndex(idx_f))
