import numpy as np
import pandas as pd

def compute_metrics(series: pd.Series, rf_annual: float = 0.0) -> dict:
    """
    Calculates standard financial metrics.
    Returns a dictionary with GUARANTEED keys for the interface to avoid KeyErrors.
    """
    # 1. Safety check: If series is empty or too short
    if series.empty or len(series) < 2:
        return {
            "ann_return": 0.0,
            "cagr": 0.0,  # Alias for compatibility with friend's code
            "ann_vol": 0.0,
            "sharpe": 0.0,
            "max_dd": 0.0,
            "sortino": 0.0
        }

    # 2. Compute returns
    rets = series.pct_change().dropna()
    if rets.empty:
        return {"ann_return": 0.0, "cagr": 0.0, "ann_vol": 0.0, "sharpe": 0.0, "max_dd": 0.0}

    # 3. Annualization (Based on median time step)
    dt = series.index.to_series().diff().dropna().median()
    if pd.isna(dt):
        # Fallback if single data point or irregular time step -> default to 1 day
        seconds = 24 * 3600
    else:
        seconds = max(dt.total_seconds(), 1.0)
    
    periods_per_year = (365.25 * 24 * 3600) / seconds

    # --- CAGR / Annual Return ---
    total_ret = (series.iloc[-1] / series.iloc[0]) - 1.0
    # Duration in years
    duration_years = (series.index[-1] - series.index[0]).total_seconds() / (365.25 * 24 * 3600)
    
    if duration_years > 0:
        ann_return = (series.iloc[-1] / series.iloc[0]) ** (1 / duration_years) - 1.0
    else:
        ann_return = 0.0

    # --- Volatility ---
    ann_vol = rets.std(ddof=1) * np.sqrt(periods_per_year)

    # --- Sharpe Ratio ---
    # rf_period = rf_annual / periods_per_year (approx)
    mu = rets.mean() * periods_per_year
    sigma = ann_vol
    if sigma > 1e-9:
        sharpe = (mu - rf_annual) / sigma
    else:
        sharpe = 0.0

    # --- Max Drawdown ---
    roll_max = series.cummax()
    drawdown = (series / roll_max) - 1.0
    max_dd = float(drawdown.min())

    return {
        "ann_return": ann_return,
        "cagr": ann_return,      # DUPLICATED: Ensures compatibility if friend uses 'cagr'
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "sortino": 0.0           # Placeholder if requested later
    }


def open_close_return_24h(prices: pd.Series) -> dict:
    """
    Calculates statistics for the last 24h.
    """
    if prices.empty:
        return {"open_24h": 0.0, "close_24h": 0.0, "return_24h": 0.0}
    
    last_ts = prices.index[-1]
    cutoff = last_ts - pd.Timedelta(hours=24)
    
    # Current price
    close_val = float(prices.iloc[-1])
    
    # Price 24h ago (or closest data point before that)
    # Get points <= cutoff, take the last available one
    past = prices[prices.index <= cutoff]
    if past.empty:
        # If not enough history, take the very first data point
        open_val = float(prices.iloc[0])
    else:
        open_val = float(past.iloc[-1])
        
    ret_24h = (close_val - open_val) / open_val if open_val != 0 else 0.0
    
    return {
        "open_24h": open_val,
        "close_24h": close_val,
        "return_24h": ret_24h
    }


def realized_vol(prices: pd.Series, window: int = 20) -> dict:
    """
    Simple realized volatility over the last X points.
    """
    rets = prices.pct_change().dropna()
    if len(rets) < window:
        vol_est = rets.std() * np.sqrt(365*24) # Rough approximation
    else:
        vol_est = rets.tail(window).std() * np.sqrt(365*24) # Approx hourly -> yearly
        
    return {"ann_vol_est": 0.0 if pd.isna(vol_est) else vol_est}


def drawdown_series(equity: pd.Series) -> pd.Series:
    """
    Calculates the drawdown series (percentage drop from peak).
    """
    roll_max = equity.cummax()
    dd = (equity / roll_max) - 1.0
    return dd