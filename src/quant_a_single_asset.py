from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

from src.data_fetch import fetch_price_series, resample_price
from src.metrics import compute_metrics


@dataclass(frozen=True)
class SingleAssetResult:
    asset_id: str
    vs: str
    prices: pd.Series          # prix brut (devise vs)
    equity: pd.Series          # valeur stratégie (base 100)
    position: pd.Series        # exposition (0/1)
    strategy_name: str
    params: dict


def _buy_and_hold(prices: pd.Series) -> tuple[pd.Series, pd.Series]:
    """
    Buy&Hold: equity base 100, position=1 tout le temps.
    """
    equity = (prices / prices.iloc[0]) * 100.0
    pos = pd.Series(1.0, index=prices.index, name="position")
    return equity, pos


def _momentum(prices: pd.Series, lookback: int = 20) -> tuple[pd.Series, pd.Series]:
    """
    Momentum simple:
    - signal = 1 si return sur lookback > 0, sinon 0 (cash)
    - on shift le signal pour éviter look-ahead
    """
    if lookback < 1:
        raise ValueError("lookback must be >= 1")

    ret_lb = prices / prices.shift(lookback) - 1.0
    signal = (ret_lb > 0).astype(float)
    position = signal.shift(1).fillna(0.0)

    rets = prices.pct_change().fillna(0.0)
    strat_rets = position * rets
    equity = 100.0 * (1.0 + strat_rets).cumprod()
    equity.name = "equity"
    position.name = "position"
    return equity, position


def _sma_crossover(prices: pd.Series, short: int = 10, long: int = 30) -> tuple[pd.Series, pd.Series]:
    """
    SMA crossover:
    - long only : position = 1 si SMA_short > SMA_long sinon 0
    - shift pour éviter look-ahead
    """
    if short < 1 or long < 2 or short >= long:
        raise ValueError("Paramètres SMA not valid (we nedd 1 <= short < long)")

    sma_s = prices.rolling(short).mean()
    sma_l = prices.rolling(long).mean()
    signal = (sma_s > sma_l).astype(float)
    position = signal.shift(1).fillna(0.0)

    rets = prices.pct_change().fillna(0.0)
    strat_rets = position * rets
    equity = 100.0 * (1.0 + strat_rets).cumprod()
    equity.name = "equity"
    position.name = "position"
    return equity, position


def build_single_asset_result(
    asset_id: str,
    vs: str = "eur",
    days: int = 30,
    periodicity: str = "raw",
    strategy: str = "Momentum",
    lookback: int = 20,
    sma_short: int = 10,
    sma_long: int = 30,
) -> SingleAssetResult:
    prices_raw = fetch_price_series(asset_id, vs=vs, days=days)
    prices = resample_price(prices_raw, periodicity)

    if len(prices) < 10:
        raise RuntimeError("not enough points for  backtesting (increase days ou put periodicity=raw).")

    params: dict = {"periodicity": periodicity, "days": days}

    if strategy == "Buy & Hold":
        equity, pos = _buy_and_hold(prices)
        params.update({})
    elif strategy == "Momentum":
        equity, pos = _momentum(prices, lookback=lookback)
        params.update({"lookback": lookback})
    elif strategy == "SMA Crossover":
        equity, pos = _sma_crossover(prices, short=sma_short, long=sma_long)
        params.update({"sma_short": sma_short, "sma_long": sma_long})
    else:
        raise ValueError(f"unknown strategy: {strategy}")

    return SingleAssetResult(
        asset_id=asset_id,
        vs=vs,
        prices=prices,
        equity=equity,
        position=pos,
        strategy_name=strategy,
        params=params,
    )


def simple_linear_forecast(
    series: pd.Series,
    horizon: int = 20,
    fit_last: int = 200,
) -> pd.DataFrame:
    """
    Bonus simple (optionnel):
    - régression linéaire sur le temps (sur les fit_last derniers points)
    - intervalle ± 1.96 * std(residuals)
    Retourne un DF indexé par dates futures avec columns: yhat, lo, hi
    """
    s = series.dropna()
    if len(s) < 10:
        raise ValueError("Série trop courte pour forecast")

    s_fit = s.iloc[-min(len(s), fit_last):]
    y = s_fit.values.astype(float)
    x = np.arange(len(y), dtype=float)

    # y = a*x + b
    a, b = np.polyfit(x, y, deg=1)
    y_hat_fit = a * x + b
    resid = y - y_hat_fit
    sigma = float(np.std(resid, ddof=1)) if len(resid) > 2 else 0.0

    # pas de temps médian
    dt = s_fit.index.to_series().diff().dropna().median()
    if pd.isna(dt):
        dt = pd.Timedelta(minutes=5)

    x_f = np.arange(len(y), len(y) + horizon, dtype=float)
    yhat = a * x_f + b

    idx_f = [s_fit.index[-1] + (i + 1) * dt for i in range(horizon)]
    lo = yhat - 1.96 * sigma
    hi = yhat + 1.96 * sigma

    return pd.DataFrame({"yhat": yhat, "lo": lo, "hi": hi}, index=pd.DatetimeIndex(idx_f))
