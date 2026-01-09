import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from streamlit_autorefresh import st_autorefresh

# Import internal modules
from src.data_fetch import ASSET_MAP
from src.metrics import compute_metrics, open_close_return_24h, realized_vol, drawdown_series, trade_stats
from src.quant_a_single_asset import build_single_asset_result, simple_linear_forecast

# --- Page Configuration ---
st.set_page_config(page_title="Single Asset (Quant A)", layout="wide")
st_autorefresh(interval=5 * 60 * 1000, key="single_asset_refresh")

st.title("Single Asset (Quant A) — Univariate Strategies")

# --- Sidebar (Identique à celle de ton ami) ---
with st.sidebar:
    st.header("Settings")
    label = st.selectbox("Asset", options=list(ASSET_MAP.keys()), index=0)
    asset_id = ASSET_MAP[label]
    vs = st.selectbox("Currency", ["eur", "usd"], index=0)
    days = st.selectbox("History (days)", [30, 90, 180, 365, 720], index=2)
    periodicity = st.selectbox("Periodicity", ["raw", "5min", "15min", "1H", "4H", "1D"], index=0)
    strategy = st.selectbox("Strategy", ["Buy & Hold", "Momentum", "SMA Crossover"], index=1)

    st.subheader("Paramètres stratégie")
    lookback = st.number_input("Lookback", 20) if strategy == "Momentum" else 20
    sma_short = st.number_input("SMA Short", 10) if strategy == "SMA Crossover" else 10
    sma_long = st.number_input("SMA Long", 30) if strategy == "SMA Crossover" else 30

    st.divider()
    st.subheader("Backtest (pro)")
    allow_short = st.checkbox("Autoriser short", value=False)
    leverage = st.slider("Leverage", 0.5, 5.0, 1.0, 0.1)
    fee_bps = st.slider("Frais (bps)", 0.0, 50.0, 5.0)
    slippage_bps = st.slider("Slippage (bps)", 0.0, 50.0, 5.0)

# --- Calculs ---
@st.cache_data(ttl=300, show_spinner=False)
def cached_res(asset_id, vs, days, periodicity, strategy, lookback, sma_short, sma_long, allow_short, leverage, fee_bps, slippage_bps):
    return build_single_asset_result(
        asset_id, vs, int(days), periodicity, strategy,
        int(lookback), int(sma_short), int(sma_long),
        bool(allow_short), float(leverage), float(fee_bps), float(slippage_bps)
    )

try:
    res = cached_res(asset_id, vs, days, periodicity, strategy, lookback, sma_short, sma_long, allow_short, leverage, fee_bps, slippage_bps)
except Exception as e:
    st.error(f"Erreur : {e}")
    st.stop()

# --- KPIs (Style de ton ami : Prix, CAGR, Sortino...) ---
kpi = compute_metrics(res.equity)
price_now = res.prices.iloc[-1]
t_stats = trade_stats(res.position) # On récupère le nombre de trades

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Prix actuel", f"{price_now:,.2f}")
c2.metric("CAGR", f"{kpi['cagr']*100:.2f}%")
c3.metric("Vol ann.", f"{kpi['ann_vol']*100:.2f}%")
c4.metric("Sharpe", f"{kpi['sharpe']:.2f}")
c5.metric("Sortino", f"{kpi['sortino']:.2f}")
c6.metric("Max DD", f"{kpi['max_dd']*100:.2f}%")

st.caption(f"Dernière mise à jour : {pd.Timestamp.now()} — stratégie: {strategy} — trades: {t_stats['total_trades']}")

# --- Onglets (Le style spécifique de ton ami) ---
tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "🧪 Diagnostics", "⬇️ Export"])

with tab1:
    st.subheader("Prix (gauche) + Equity stratégie (droite) + Drawdown + Position")
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=res.prices.index, y=res.prices.values, name="Prix", line=dict(color="blue")), secondary_y=False)
    fig.add_trace(go.Scatter(x=res.equity.index, y=res.equity.values, name="Equity", line=dict(color="cyan")), secondary_y=True)
    
    # Ajout du Drawdown comme ton ami (probablement une aire rouge en bas ou une ligne)
    dd = drawdown_series(res.equity)
    fig.add_trace(go.Scatter(x=dd.index, y=dd.values, name="Drawdown", line=dict(color="red", width=1), fill='tozeroy'), secondary_y=True)

    fig.update_layout(height=500, hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.write("Statistiques détaillées des trades")
    st.json(t_stats)
    st.write("Dernières données :")
    st.dataframe(res.equity.tail(50))

with tab3:
    st.download_button("Télécharger CSV", res.equity.to_csv(), "backtest.csv")