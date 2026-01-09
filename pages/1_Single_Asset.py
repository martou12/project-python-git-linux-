import time
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from streamlit_autorefresh import st_autorefresh

# Import internal modules
from src.data_fetch import ASSET_MAP
from src.metrics import compute_metrics, open_close_return_24h, realized_vol, drawdown_series
from src.quant_a_single_asset import build_single_asset_result, simple_linear_forecast

# --- Page Configuration ---
st.set_page_config(page_title="Single Asset (Quant A)", layout="wide")

# Auto-refresh every 5 minutes
st_autorefresh(interval=5 * 60 * 1000, key="single_asset_refresh")

st.title("Single Asset Analysis (Quant A) — Univariate Strategies")

# --- Sidebar Settings ---
with st.sidebar:
    st.header("Settings")

    # Asset Selection
    label = st.selectbox("Asset", options=list(ASSET_MAP.keys()), index=0)
    asset_id = ASSET_MAP[label]

    # General Parameters
    vs = st.selectbox("Currency", ["eur", "usd"], index=0)
    days = st.selectbox("History (days)", [30, 90, 180, 365, 720], index=3)

    periodicity = st.selectbox(
        "Periodicity / Resample",
        options=["raw", "5min", "15min", "1H", "4H", "1D"],
        index=0,
        help="raw = original data. Otherwise resampled (last + ffill).",
    )

    strategy = st.selectbox("Strategy", ["Buy & Hold", "Momentum", "SMA Crossover"], index=1)

    # --- Strategy Parameters ---
    st.subheader("Strategy Parameters")
    lookback = 20
    sma_short, sma_long = 10, 30

    if strategy == "Momentum":
        lookback = st.number_input("Momentum Lookback (points)", min_value=1, max_value=500, value=20, step=1)

    if strategy == "SMA Crossover":
        sma_short = st.number_input("SMA Short Window", min_value=1, max_value=500, value=10, step=1)
        sma_long = st.number_input("SMA Long Window", min_value=2, max_value=1000, value=30, step=1)

    # --- Advanced Backtest Settings (Friend's Logic Enabled) ---
    st.divider()
    st.subheader("Advanced Execution")
    allow_short = st.checkbox("Allow Short Selling", value=False)
    leverage = st.slider("Leverage (x)", min_value=0.5, max_value=5.0, value=1.0, step=0.1)
    fee_bps = st.slider("Trading Fees (bps)", 0.0, 100.0, 5.0, 1.0)
    slippage_bps = st.slider("Slippage (bps)", 0.0, 100.0, 5.0, 1.0)

    # --- Extra Features ---
    st.divider()
    st.caption("Extras")
    enable_forecast = st.checkbox("Show Linear Forecast", value=False)


# --- Caching & Computation ---
@st.cache_data(ttl=300, show_spinner=False)
def cached_res(asset_id, vs, days, periodicity, strategy, lookback, sma_short, sma_long, allow_short, leverage, fee_bps, slippage_bps):
    """
    Wrapper calling the backend.
    We pass ALL parameters including friend's advanced execution params.
    """
    return build_single_asset_result(
        asset_id=asset_id,
        vs=vs,
        days=int(days),
        periodicity=periodicity,
        strategy=strategy,
        lookback=int(lookback),
        sma_short=int(sma_short),
        sma_long=int(sma_long),
        # Passing advanced params supported by your friend's backend
        allow_short=bool(allow_short),
        leverage=float(leverage),
        fee_bps=float(fee_bps),
        slippage_bps=float(slippage_bps)
    )

try:
    # Run the backtest with all parameters
    res = cached_res(
        asset_id, vs, days, periodicity, strategy, 
        lookback, sma_short, sma_long, 
        allow_short, leverage, fee_bps, slippage_bps
    )
except Exception as e:
    st.error(f"Backtest Computation Error: {e}")
    st.stop()

# --- Metrics Calculation ---
price_now = float(res.prices.iloc[-1])
kpi = compute_metrics(res.equity) 
oc = open_close_return_24h(res.prices)
vol = realized_vol(res.prices)
dd = drawdown_series(res.equity)

# --- KPI Dashboard (English) ---
st.markdown("### Market & Strategy Performance")
c1, c2, c3, c4, c5 = st.columns(5)

c1.metric(
    "Current Price", 
    f"{price_now:,.2f} {vs.upper()}", 
    delta=f"{oc['return_24h']*100:.2f}% (24h)"
)
c2.metric("Annualized Return", f"{kpi['ann_return']*100:.2f}%")
c3.metric("Annualized Volatility", f"{kpi['ann_vol']*100:.2f}%", delta_color="inverse")
c4.metric("Sharpe Ratio", f"{kpi['sharpe']:.2f}")
c5.metric("Max Drawdown", f"{kpi['max_dd']*100:.2f}%", delta_color="inverse")

st.caption(
    f"Last update: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')} (UTC) | "
    f"Strategy: **{res.strategy_name}** | Trades executed by engine."
)

# --- Main Interface Tabs ---
tab1, tab2 = st.tabs(["📈 Visualization & Strategy", "📊 Raw Data & Stats"])

with tab1:
    st.subheader("Price vs. Strategy Equity Curve")
    
    # Dual Axis Chart
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # 1. Asset Price Line
    fig.add_trace(
        go.Scatter(x=res.prices.index, y=res.prices.values, mode="lines", name=f"{label} Price"),
        secondary_y=False,
    )
    
    # 2. Strategy Equity Line
    fig.add_trace(
        go.Scatter(x=res.equity.index, y=res.equity.values, mode="lines", name="Strategy Equity (Base 100)", line=dict(color="orange")),
        secondary_y=True,
    )

    # 3. Forecast Overlay (Optional)
    if enable_forecast:
        try:
            fc = simple_linear_forecast(res.prices, horizon=20, fit_last=200)
            fig.add_trace(go.Scatter(x=fc.index, y=fc["yhat"], mode="lines", name="Forecast", line=dict(dash="dash", color="green")), secondary_y=False)
        except Exception as e:
            st.warning(f"Forecast unavailable: {e}")

    fig.update_layout(
        title_text="Performance Analysis",
        xaxis_title="Date",
        hovermode="x unified",
        legend=dict(orientation="h", y=1.1)
    )
    fig.update_yaxes(title_text=f"Price ({vs.upper()})", secondary_y=False)
    fig.update_yaxes(title_text="Strategy Value (100 start)", secondary_y=True)

    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Detailed Statistics")
    
    c_a, c_b, c_c = st.columns(3)
    c_a.metric("24h Open", f"{oc['open_24h']:.2f}")
    c_b.metric("24h Close", f"{oc['close_24h']:.2f}")
    c_c.metric("Est. Volatility (24h)", f"{vol['ann_vol_est']*100:.2f}%")

    st.divider()
    st.markdown("#### Recent Data Points")
    
    # Data Table
    df_display = pd.DataFrame({
        "Asset Price": res.prices,
        "Strategy Equity": res.equity,
        "Market Position": res.position
    })
    st.dataframe(df_display.tail(100), use_container_width=True)