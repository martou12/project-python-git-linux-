import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from streamlit_autorefresh import st_autorefresh

from src.quant_b_portfolio import ASSET_MAP, build_portfolio_result, compute_metrics

# Page Configuration
st.set_page_config(page_title="Portfolio (Quant B)", layout="wide")

# Auto-refresh every 5 minutes
st_autorefresh(interval=5 * 60 * 1000, key="portfolio_refresh")

st.title("Portfolio Management (Quant B) — Multi-Assets")

#  Sidebar Settings 
with st.sidebar:
    st.header("Settings")
    vs = st.selectbox("Currency", ["eur", "usd"], index=0)
    days = st.selectbox("History (days)", [1, 7, 30, 90, 180, 365], index=2)
    rebalance = st.selectbox("Rebalancing Frequency", ["None", "Daily", "Weekly", "Monthly"], index=0)

    assets = st.multiselect(
        "Assets (Select at least 3)",
        options=list(ASSET_MAP.keys()),
        default=["Bitcoin (BTC)", "Ethereum (ETH)", "Solana (SOL)"],
    )

    st.caption("Weights (Auto-normalized to 100%)")
    w_inputs = []
    for a in assets:
        w_inputs.append(st.number_input(f"Weight: {a} (%)", min_value=0.0, max_value=100.0, value=33.33))
    
    w = np.array(w_inputs, dtype=float)
    if w.sum() == 0:
        st.warning("Total weight is 0. Please set at least one weight > 0.")
    
    # Normalization logic
    w = w / max(w.sum(), 1e-12)

if len(assets) < 3:
    st.error("Please select at least 3 assets to analyze diversification.")
    st.stop()

asset_ids = [ASSET_MAP[a] for a in assets]

@st.cache_data(ttl=300)
def cached_result(asset_ids, vs, days, weights, rebalance):
    return build_portfolio_result(asset_ids, vs=vs, days=days, weights=weights, rebalance=rebalance)

try:
    res = cached_result(tuple(asset_ids), vs, days, tuple(w), rebalance)
except Exception as e:
    st.error(f"Data Source Error: {e}")
    st.stop()

#  KPI Section with Deltas 
m = compute_metrics(res.portfolio)

# Calculate 24h variation for the portfolio to show a Delta
port_last = res.portfolio.iloc[-1]
port_prev = res.portfolio.iloc[-2] if len(res.portfolio) > 1 else port_last
delta_24h = (port_last / port_prev) - 1.0

c1, c2, c3, c4, c5 = st.columns(5)

# Added a "Current Value" metric to make the Delta meaningful
c1.metric(
    "Current Value (Base 100)", 
    f"{port_last:.2f}", 
    delta=f"{delta_24h*100:.2f}% (24h)"
)
c2.metric("Annualized Perf.", f"{m['ann_return']*100:.2f}%")
c3.metric("Annualized Vol.", f"{m['ann_vol']*100:.2f}%", delta_color="inverse") # Low vol is often better
c4.metric("Sharpe Ratio (rf=0)", f"{m['sharpe']:.2f}")
c5.metric("Max Drawdown", f"{m['max_dd']*100:.2f}%", delta_color="inverse") # Low DD is better

st.caption(f"Last update: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')} (5 min auto-refresh)")

#  Layout with Tabs (Professional UX) 
tab1, tab2 = st.tabs([" Visualization & Analysis", " Raw Data"])

with tab1:
    #  Main chart: Normalized prices + Portfolio performance
    st.subheader("Performance Comparison")
    norm_prices = (res.prices / res.prices.iloc[0]) * 100.0

    fig = go.Figure()
    for col in norm_prices.columns:
        fig.add_trace(go.Scatter(x=norm_prices.index, y=norm_prices[col], mode="lines", name=col, opacity=0.6))

    # Portfolio line is thicker and distinct
    fig.add_trace(go.Scatter(
        x=res.portfolio.index, 
        y=res.portfolio.values, 
        mode="lines", 
        name="Portfolio (Base 100)", 
        line=dict(width=4, dash='dash', color='black')
    ))

    fig.update_layout(
        title="Normalized Prices (Base 100) vs. Portfolio Performance",
        xaxis_title="Date",
        yaxis_title="Normalized Value",
        legend_title="Assets / Portfolio",
        hovermode="x unified"
    )
    # Hide the ugly toolbar
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    #  Correlation Matrix
    st.subheader("Correlation Matrix (Returns)")
    corr_fig = px.imshow(res.corr, aspect="auto", text_auto=True, color_continuous_scale='RdBu_r')
    st.plotly_chart(corr_fig, use_container_width=True, config={"displayModeBar": False})

with tab2:
    #  Raw Data Table
    st.subheader("Latest Computed Values")
    last = res.prices.tail(50).copy()
    last["portfolio_value"] = res.portfolio.tail(50)
    st.dataframe(last, use_container_width=True)