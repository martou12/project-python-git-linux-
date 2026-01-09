import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from streamlit_autorefresh import st_autorefresh

from src.data_fetch import ASSET_MAP
from src.metrics import compute_metrics, open_close_return_24h, realized_vol, drawdown_series
from src.quant_a_single_asset import build_single_asset_result, simple_linear_forecast

# Page Configuration
st.set_page_config(page_title="Single Asset (Quant A)", layout="wide")

# Auto-refresh every 5 minutes
st_autorefresh(interval=5 * 60 * 1000, key="single_asset_refresh")

st.title("Single Asset Analysis (Quant A) — Univariate")

# --- Sidebar Settings ---
with st.sidebar:
    st.header("Settings")

    label = st.selectbox("Asset", options=list(ASSET_MAP.keys()), index=0)
    asset_id = ASSET_MAP[label]

    vs = st.selectbox("Currency", ["eur", "usd"], index=0)
    days = st.selectbox("History (days)", [1, 7, 30, 90, 180, 365], index=2)

    periodicity = st.selectbox(
        "Periodicity / Resample",
        options=["raw", "5min", "15min", "1H", "4H", "1D"],
        index=0,
        help="raw = original CoinGecko data. Otherwise resampled (last + ffill).",
    )

    strategy = st.selectbox("Strategy", ["Buy & Hold", "Momentum", "SMA Crossover"], index=1)

    st.subheader("Backtest (pro)")
    allow_short = st.checkbox("Autoriser short (long/short)", value=False)
    leverage = st.slider("Leverage", min_value=0.5, max_value=3.0, value=1.0, step=0.1)

    fee_bps = st.slider("Frais (bps)", 0.0, 50.0, 5.0, 0.5)
    slippage_bps = st.slider("Slippage (bps)", 0.0, 50.0, 5.0, 0.5)

    st.subheader("Paramètres stratégie")
    lookback = 20
    sma_short, sma_long = 10, 30

    if strategy == "Momentum":
        lookback = st.number_input("Momentum Lookback (points)", min_value=1, max_value=500, value=20, step=1)

    if strategy == "SMA Crossover":
        sma_short = st.number_input("SMA Short Window", min_value=1, max_value=500, value=10, step=1)
        sma_long = st.number_input("SMA Long Window", min_value=2, max_value=1000, value=30, step=1)

    st.divider()
    st.subheader("Extra Features (Optional)")
    enable_forecast = st.checkbox("Show Linear Forecast (Price)", value=False)
    horizon = st.number_input("Forecast Horizon (points)", min_value=5, max_value=200, value=20, step=5)
    fit_last = st.number_input("Fit on last N points", min_value=50, max_value=2000, value=200, step=50)

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

try:
    res = cached_res(asset_id, vs, days, periodicity, strategy, lookback, sma_short, sma_long)
except Exception as e:
    st.error(f"Data/Backtest Error: {e}")
    st.stop()

#  Metrics Calculation 
price_now = float(res.prices.iloc[-1])
kpi = compute_metrics(res.equity, rf_annual=float(rf))
oc = open_close_return_24h(res.prices)
vol = realized_vol(res.prices)
dd = drawdown_series(res.equity)

#  KPIs Section with Deltas 
c1, c2, c3, c4, c5 = st.columns(5)

# Price with 24h variation delta
c1.metric(
    "Current Price", 
    f"{price_now:,.2f} {vs.upper()}", 
    delta=f"{oc['return_24h']*100:.2f}% (24h)"
)
c2.metric("Annualized Perf.", f"{kpi['ann_return']*100:.2f}%")
c3.metric("Annualized Vol.", f"{kpi['ann_vol']*100:.2f}%", delta_color="inverse") # Lower is usually safer
c4.metric("Sharpe Ratio (rf=0)", f"{kpi['sharpe']:.2f}")
c5.metric("Max Drawdown", f"{kpi['max_dd']*100:.2f}%", delta_color="inverse") # Lower (closer to 0) is better

st.caption(
    f"Last update: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')} "
    f"(5 min auto-refresh) — strategy: {res.strategy_name} — params: {res.params}"
)

# --- Layout with Tabs ---
tab1, tab2 = st.tabs([" Visualization & Strategy", " Raw Data & Stats"])

with tab1:
    st.subheader("Price & Strategy Equity")
    
    # Main chart: raw price + strategy equity
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(x=res.prices.index, y=res.prices.values, mode="lines", name=f"Price ({label})"),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(x=res.equity.index, y=res.equity.values, mode="lines", name=f"Strategy Value (Base 100)"),
        secondary_y=True,
    )

    # Optional Linear Forecast
    if enable_forecast:
        try:
            fc = simple_linear_forecast(res.prices, horizon=int(horizon), fit_last=int(fit_last))
            fig.add_trace(go.Scatter(x=fc.index, y=fc["yhat"], mode="lines", name="Forecast (yhat)", line=dict(dash="dash", color="green")), secondary_y=False)
            fig.add_trace(go.Scatter(x=fc.index, y=fc["hi"], mode="lines", name="Forecast Upper", line=dict(dash="dot", width=1, color="green"), showlegend=False), secondary_y=False)
            fig.add_trace(go.Scatter(x=fc.index, y=fc["lo"], mode="lines", name="Forecast Lower", line=dict(dash="dot", width=1, color="green"), showlegend=False), secondary_y=False)
        except Exception as e:
            st.warning(f"Forecast failed: {e}")

    fig.update_layout(
        title="Raw Price vs. Cumulative Strategy Equity",
        xaxis_title="Time",
        legend_title="Series",
        hovermode="x unified"
    )
    fig.update_yaxes(title_text=f"Price ({vs.upper()})", secondary_y=False)
    fig.update_yaxes(title_text="Strategy Value (Base 100)", secondary_y=True)

    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

with tab2:
    # 24h Stats in the Data Tab
    st.subheader("24h Price Statistics")
    c6, c7, c8, c9 = st.columns(4)
    c6.metric("24h Open", f"{oc['open_24h']:.2f}")
    c7.metric("24h Close", f"{oc['close_24h']:.2f}")
    c8.metric("24h Return", f"{oc['return_24h']*100:.2f}%")
    c9.metric("Est. Ann. Volatility", f"{vol['ann_vol_est']*100:.2f}%")

    st.divider()

    st.subheader("Latest Data Points")
    df = pd.DataFrame({"price": res.prices, "equity": res.equity, "position": res.position})
    st.dataframe(df.tail(50), use_container_width=True)
