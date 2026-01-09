import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from streamlit_autorefresh import st_autorefresh

from src.data_fetch import ASSET_MAP
from src.metrics import (
    compute_metrics,
    open_close_return_24h,
    realized_vol,
    drawdown_series,
)
from src.quant_a_single_asset import build_single_asset_result, simple_linear_forecast

# Page config + styling

st.set_page_config(page_title="Single Asset (Quant A)", layout="wide")
st_autorefresh(interval=5 * 60 * 1000, key="single_asset_refresh")  # 5 minutes

st.markdown(
    """
<style>
/* Clean "pro" cards */
.kpi-wrap {display: grid; grid-template-columns: repeat(6, 1fr); gap: 10px; margin-top: 6px;}
.kpi-card {padding: 14px; border-radius: 16px; background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.12);}
.kpi-title {font-size: 12px; opacity: 0.75; margin-bottom: 2px;}
.kpi-value {font-size: 20px; font-weight: 650;}
.kpi-sub {font-size: 12px; opacity: 0.70; margin-top: 2px;}
.small-muted {opacity: 0.7; font-size: 12px;}
hr {opacity: 0.2;}
</style>
""",
    unsafe_allow_html=True,
)

st.title("📈 Quant A — Single Asset (Univariate Backtesting)")
st.caption("Real-time data refresh every 5 minutes • Raw price + Strategy equity on the main chart")


# Helpers
def _fmt_pct(x: float) -> str:
    return f"{x*100:.2f}%"

def _fmt_num(x: float) -> str:
    try:
        return f"{x:,.2f}"
    except Exception:
        return str(x)

def _periods_per_year_from_index(idx: pd.DatetimeIndex) -> float:
    if len(idx) < 3:
        return 0.0
    dt = pd.Series(idx).diff().dropna().median()
    if pd.isna(dt):
        return 0.0
    seconds = max(float(dt.total_seconds()), 1.0)
    return (365.0 * 24 * 3600) / seconds

def _rolling_sharpe(equity: pd.Series, window: int, rf_annual: float = 0.0) -> pd.Series:
    equity = equity.dropna()
    if len(equity) < window + 5:
        return pd.Series(dtype=float)
    rets = equity.pct_change().dropna()
    ppy = _periods_per_year_from_index(equity.index)
    if ppy <= 0:
        return pd.Series(dtype=float)

    # convert annual rf to per-period approx
    rf_per = (1.0 + rf_annual) ** (1.0 / ppy) - 1.0
    ex = rets - rf_per

    m = ex.rolling(window).mean()
    s = rets.rolling(window).std(ddof=1)
    out = (m / s) * np.sqrt(ppy)
    out.name = "rolling_sharpe"
    return out.dropna()

def _monthly_returns(equity: pd.Series) -> pd.DataFrame:
    """
    Build a Year x Month table of monthly returns from equity curve.
    Works on intraday data (resample daily then monthly).
    """
    e = equity.dropna().sort_index()
    if e.empty or len(e) < 10:
        return pd.DataFrame()

    d = e.resample("1D").last().dropna()
    if len(d) < 10:
        return pd.DataFrame()

    m = d.resample("M").last()
    mr = m.pct_change().dropna()

    if mr.empty:
        return pd.DataFrame()

    df = mr.to_frame("ret")
    df["year"] = df.index.year
    df["month"] = df.index.month
    pivot = df.pivot(index="year", columns="month", values="ret").sort_index()
    pivot.columns = [pd.Timestamp(2000, c, 1).strftime("%b") for c in pivot.columns]
    return pivot


# Sidebar controls
with st.sidebar:
    st.header("Controls")

    label = st.selectbox("Asset", options=list(ASSET_MAP.keys()), index=0)
    asset_id = ASSET_MAP[label]

    vs = st.selectbox("Quote currency", ["eur", "usd"], index=0)
    days = st.selectbox("History window (days)", [1, 7, 30, 90, 180, 365], index=2)

    periodicity = st.selectbox(
        "Sampling / Resample",
        options=["raw", "5min", "15min", "1H", "4H", "1D"],
        index=0,
        help="raw = CoinGecko native sampling. Otherwise resample (last + ffill).",
    )

    strategy = st.selectbox("Strategy", ["Buy & Hold", "Momentum", "SMA Crossover"], index=1)

    st.subheader("Backtest settings")
    allow_short = st.checkbox("Allow short (long/short)", value=False)
    leverage = st.slider("Leverage", min_value=0.5, max_value=3.0, value=1.0, step=0.1)

    fee_bps = st.slider("Fees (bps)", 0.0, 50.0, 5.0, 0.5)
    slippage_bps = st.slider("Slippage (bps)", 0.0, 50.0, 5.0, 0.5)

    st.subheader("Strategy parameters")
    lookback = 20
    sma_short, sma_long = 10, 30

    if strategy == "Momentum":
        lookback = st.number_input("Momentum lookback (#points)", min_value=1, max_value=500, value=20, step=1)

    if strategy == "SMA Crossover":
        sma_short = st.number_input("SMA short window (#points)", min_value=1, max_value=500, value=10, step=1)
        sma_long = st.number_input("SMA long window (#points)", min_value=2, max_value=1000, value=30, step=1)

    st.divider()
    st.subheader("Risk / metrics")
    rf = st.number_input("Annual risk-free rate (e.g. 0.02)", min_value=0.0, max_value=0.2, value=0.0, step=0.005)

    st.subheader("Optional bonus: forecasting")
    enable_forecast = st.checkbox("Show linear forecast (price)", value=False)
    horizon = st.number_input("Forecast horizon (#points)", min_value=5, max_value=200, value=20, step=5)
    fit_last = st.number_input("Fit on last N points", min_value=50, max_value=2000, value=200, step=50)

    st.divider()
    st.subheader("Extra analytics")
    show_bh = st.checkbox("Compare with Buy & Hold", value=True)
    roll_window = st.number_input("Rolling Sharpe window (#points)", min_value=10, max_value=500, value=60, step=10)


# Cached compute
@st.cache_data(ttl=300, show_spinner=False)
def cached_res(asset_id, vs, days, periodicity, strategy, lookback, sma_short, sma_long, allow_short, leverage, fee_bps, slippage_bps):
    return build_single_asset_result(
        asset_id=asset_id,
        vs=vs,
        days=int(days),
        periodicity=periodicity,
        strategy=strategy,
        lookback=int(lookback),
        sma_short=int(sma_short),
        sma_long=int(sma_long),
        allow_short=bool(allow_short),
        leverage=float(leverage),
        fee_bps=float(fee_bps),
        slippage_bps=float(slippage_bps),
    )


with st.spinner("Fetching data and running backtest..."):
    try:
        res = cached_res(asset_id, vs, days, periodicity, strategy, lookback, sma_short, sma_long, allow_short, leverage, fee_bps, slippage_bps)
    except Exception as e:
        st.error(f"Data/backtest error: {e}")
        st.stop()


# Core derived values
price_now = float(res.prices.iloc[-1])
oc = open_close_return_24h(res.prices)
vol = realized_vol(res.prices)

kpi = compute_metrics(res.equity, rf_annual=float(rf))
dd = drawdown_series(res.equity)

current_pos = float(res.position.dropna().iloc[-1]) if not res.position.dropna().empty else 0.0
trades = int(res.meta.get("trades", 0))
turnover = float(res.meta.get("turnover", 0.0))
source = res.meta.get("source", "unknown")

# Buy&Hold benchmark
bh_equity = None
bh_kpi = None
if show_bh:
    bh_equity = (res.prices / res.prices.iloc[0]) * 100.0
    bh_equity.name = "bh_equity"
    bh_kpi = compute_metrics(bh_equity, rf_annual=float(rf))


# KPI row (custom cards)
kpi_html = f"""
<div class="kpi-wrap">
  <div class="kpi-card">
    <div class="kpi-title">Current price</div>
    <div class="kpi-value">{_fmt_num(price_now)} {vs.upper()}</div>
    <div class="kpi-sub">Source: {source}</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-title">CAGR</div>
    <div class="kpi-value">{_fmt_pct(kpi.get("cagr", 0.0))}</div>
    <div class="kpi-sub">Total: {_fmt_pct(kpi.get("total_return", 0.0))}</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-title">Annual vol</div>
    <div class="kpi-value">{_fmt_pct(kpi.get("ann_vol", 0.0))}</div>
    <div class="kpi-sub">24h est.: {_fmt_pct(vol.get("ann_vol_est", 0.0))}</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-title">Sharpe</div>
    <div class="kpi-value">{kpi.get("sharpe", 0.0):.2f}</div>
    <div class="kpi-sub">rf={rf:.3f}</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-title">Max drawdown</div>
    <div class="kpi-value">{_fmt_pct(kpi.get("max_dd", 0.0))}</div>
    <div class="kpi-sub">Calmar: {kpi.get("calmar", 0.0):.2f}</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-title">Position / Trading</div>
    <div class="kpi-value">{current_pos:.0f}</div>
    <div class="kpi-sub">Trades: {trades} • Turnover: {turnover:.2f}</div>
  </div>
</div>
"""
st.markdown(kpi_html, unsafe_allow_html=True)

st.caption(
    f"Last refresh: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')} • "
    f"Strategy: {res.strategy_name} • Params: {res.params}"
)

tabs = st.tabs(["📊 Dashboard", "🧪 Diagnostics", "🗓️ Monthly returns", "⬇️ Export"])


# TAB 1 - Dashboard
with tabs[0]:
    # Trade markers (position changes)
    pos = res.position.fillna(0.0)
    chg = pos.diff().fillna(0.0)
    trade_idx = chg[chg != 0].index

    # 3 rows: price+equity, drawdown, position
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.55, 0.25, 0.20],
        specs=[[{"secondary_y": True}], [{"secondary_y": False}], [{"secondary_y": False}]],
    )

    # Price
    fig.add_trace(
        go.Scatter(x=res.prices.index, y=res.prices.values, mode="lines", name=f"Price ({label})"),
        row=1, col=1, secondary_y=False
    )

    # Strategy equity
    fig.add_trace(
        go.Scatter(x=res.equity.index, y=res.equity.values, mode="lines", name="Strategy equity (base 100)"),
        row=1, col=1, secondary_y=True
    )

    # Buy&Hold benchmark equity
    if bh_equity is not None:
        fig.add_trace(
            go.Scatter(x=bh_equity.index, y=bh_equity.values, mode="lines", name="Buy & Hold equity (base 100)", line=dict(dash="dot")),
            row=1, col=1, secondary_y=True
        )

    # SMA overlays when available
    if strategy == "SMA Crossover" and "sma_short" in res.meta and "sma_long" in res.meta:
        sma_s = res.meta["sma_short"]
        sma_l = res.meta["sma_long"]
        fig.add_trace(go.Scatter(x=sma_s.index, y=sma_s.values, mode="lines", name="SMA short", line=dict(dash="dot")), row=1, col=1, secondary_y=False)
        fig.add_trace(go.Scatter(x=sma_l.index, y=sma_l.values, mode="lines", name="SMA long", line=dict(dash="dot")), row=1, col=1, secondary_y=False)

    # Trade markers on price
    if len(trade_idx) > 0:
        y_trade = res.prices.reindex(trade_idx)
        fig.add_trace(
            go.Scatter(
                x=trade_idx,
                y=y_trade.values,
                mode="markers",
                name="Signal change",
                marker=dict(size=8, symbol="circle"),
            ),
            row=1, col=1, secondary_y=False
        )

    #forecast
    if enable_forecast:
        try:
            fc = simple_linear_forecast(res.prices, horizon=int(horizon), fit_last=int(fit_last))
            fig.add_trace(go.Scatter(x=fc.index, y=fc["yhat"], mode="lines", name="Forecast (yhat)", line=dict(dash="dash")), row=1, col=1, secondary_y=False)
            fig.add_trace(go.Scatter(x=fc.index, y=fc["hi"], mode="lines", name="Forecast (high)", line=dict(dash="dot")), row=1, col=1, secondary_y=False)
            fig.add_trace(go.Scatter(x=fc.index, y=fc["lo"], mode="lines", name="Forecast (low)", line=dict(dash="dot")), row=1, col=1, secondary_y=False)
        except Exception as e:
            st.warning(f"Forecast unavailable: {e}")

    # Drawdown
    fig.add_trace(go.Scatter(x=dd.index, y=dd.values, mode="lines", name="Drawdown"), row=2, col=1)

    # Position
    fig.add_trace(go.Scatter(x=res.position.index, y=res.position.values, mode="lines", name="Position"), row=3, col=1)

    fig.update_layout(
        title="Main chart: Raw price + Strategy equity (two curves) • Drawdown • Position",
        legend_title="Series",
        height=860,
    )
    fig.update_yaxes(title_text=f"Price ({vs.upper()})", row=1, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Equity (base 100)", row=1, col=1, secondary_y=True)
    fig.update_yaxes(title_text="Drawdown", row=2, col=1)
    fig.update_yaxes(title_text="Position", row=3, col=1)

    st.plotly_chart(fig, use_container_width=True)

    # 24h stats
    st.subheader("Last 24h (price stats)")
    a1, a2, a3, a4 = st.columns(4)
    a1.metric("Open (24h)", f"{oc['open_24h']:.2f}")
    a2.metric("Close (24h)", f"{oc['close_24h']:.2f}")
    a3.metric("Return (24h)", f"{oc['return_24h']*100:.2f}%")
    a4.metric("Realized vol (ann. est.)", f"{vol['ann_vol_est']*100:.2f}%")

    # Summary table
    st.subheader("Performance summary")
    rows = [
        ("Strategy", "—", kpi.get("total_return", 0.0), kpi.get("cagr", 0.0), kpi.get("ann_vol", 0.0), kpi.get("sharpe", 0.0), kpi.get("sortino", 0.0), kpi.get("max_dd", 0.0)),
    ]
    if bh_kpi is not None:
        rows.append(("Buy & Hold", "—", bh_kpi.get("total_return", 0.0), bh_kpi.get("cagr", 0.0), bh_kpi.get("ann_vol", 0.0), bh_kpi.get("sharpe", 0.0), bh_kpi.get("sortino", 0.0), bh_kpi.get("max_dd", 0.0)))

    summary = pd.DataFrame(
        rows,
        columns=["Model", "Note", "Total return", "CAGR", "Ann. vol", "Sharpe", "Sortino", "Max DD"],
    )
    summary["Total return"] = summary["Total return"].map(_fmt_pct)
    summary["CAGR"] = summary["CAGR"].map(_fmt_pct)
    summary["Ann. vol"] = summary["Ann. vol"].map(_fmt_pct)
    summary["Max DD"] = summary["Max DD"].map(_fmt_pct)
    st.dataframe(summary, use_container_width=True)

    with st.expander("Show latest values"):
        df_tail = pd.DataFrame({"price": res.prices, "equity": res.equity, "position": res.position})
        st.dataframe(df_tail.tail(80), use_container_width=True)


# TAB 2 - Diagnostics
with tabs[1]:
    st.subheader("Diagnostics")

    rets = res.equity.pct_change().dropna()
    if rets.empty:
        st.info("Not enough points to compute return diagnostics.")
    else:
        # Returns histogram
        hist = go.Figure()
        hist.add_trace(go.Histogram(x=rets.values, nbinsx=60, name="Strategy returns"))
        hist.update_layout(title="Return distribution (strategy)", xaxis_title="Return", yaxis_title="Count")
        st.plotly_chart(hist, use_container_width=True)

        # Rolling Sharpe
        rs = _rolling_sharpe(res.equity, window=int(roll_window), rf_annual=float(rf))
        if not rs.empty:
            rs_fig = go.Figure()
            rs_fig.add_trace(go.Scatter(x=rs.index, y=rs.values, mode="lines", name="Rolling Sharpe"))
            rs_fig.update_layout(title=f"Rolling Sharpe (window={int(roll_window)})", xaxis_title="Time", yaxis_title="Sharpe")
            st.plotly_chart(rs_fig, use_container_width=True)

    # Show position stats
    st.subheader("Position / trading info")
    b1, b2, b3 = st.columns(3)
    b1.metric("Current position", f"{current_pos:.0f}")
    b2.metric("Trades (# signal changes)", f"{trades}")
    b3.metric("Turnover (sum |Δpos|)", f"{turnover:.2f}")


# TAB 3 - Monthly returns
with tabs[2]:
    st.subheader("Monthly returns heatmap (strategy equity)")
    piv = _monthly_returns(res.equity)
    if piv.empty:
        st.info("Not enough history to compute monthly returns (try 90/180/365 days).")
    else:
        z = (piv.values * 100.0)
        heat = go.Figure(
            data=go.Heatmap(
                z=z,
                x=list(piv.columns),
                y=[str(i) for i in piv.index],
                text=np.round(z, 2),
                texttemplate="%{text}%",
                hovertemplate="Year=%{y}<br>Month=%{x}<br>Return=%{z:.2f}%<extra></extra>",
            )
        )
        heat.update_layout(title="Monthly returns (%)", xaxis_title="Month", yaxis_title="Year")
        st.plotly_chart(heat, use_container_width=True)


# TAB 4 - Export
with tabs[3]:
    st.subheader("Export / download")

    df = pd.DataFrame({"price": res.prices, "equity": res.equity, "position": res.position})
    st.download_button(
        "⬇️ Download CSV (price/equity/position)",
        data=df.to_csv().encode("utf-8"),
        file_name=f"quant_a_{asset_id}_{vs}.csv",
        mime="text/csv",
    )

    report = {
        "generated_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "asset": asset_id,
        "label": label,
        "vs": vs,
        "strategy": res.strategy_name,
        "params": res.params,
        "meta": {k: (str(v) if k in {"cache_path"} else v) for k, v in res.meta.items() if k not in {"sma_short", "sma_long"}},
        "kpi": kpi,
        "price_now": price_now,
        "last_24h": {**oc, **vol},
    }

    st.download_button(
        "⬇️ Download JSON report",
        data=json.dumps(report, indent=2).encode("utf-8"),
        file_name=f"quant_a_report_{asset_id}_{vs}.json",
        mime="application/json",
    )
