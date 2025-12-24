import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from streamlit_autorefresh import st_autorefresh

from src.quant_b_portfolio import ASSET_MAP, build_portfolio_result, compute_metrics

st.set_page_config(page_title="Portfolio (Quant B)", layout="wide")

# refresh auto toutes les 5 minutes
st_autorefresh(interval=5 * 60 * 1000, key="portfolio_refresh")

st.title("Portfolio (Quant B) — Multi-Assets")

with st.sidebar:
    st.header("Paramètres")
    vs = st.selectbox("Devise", ["eur", "usd"], index=0)
    days = st.selectbox("Historique", [1, 7, 30, 90, 180, 365], index=2)
    rebalance = st.selectbox("Rebalancing", ["None", "Daily", "Weekly", "Monthly"], index=0)

    assets = st.multiselect(
        "Actifs (>= 3)",
        options=list(ASSET_MAP.keys()),
        default=["Bitcoin (BTC)", "Ethereum (ETH)", "Solana (SOL)"],
    )

    st.caption("Poids (on normalise automatiquement pour que la somme = 100%)")
    w_inputs = []
    for a in assets:
        w_inputs.append(st.number_input(f"Poids {a} (%)", min_value=0.0, max_value=100.0, value=33.33))
    w = np.array(w_inputs, dtype=float)
    if w.sum() == 0:
        st.warning("Somme des poids = 0 → mettez au moins un poids > 0")
    w = w / max(w.sum(), 1e-12)

if len(assets) < 3:
    st.error("Veuillez sélectionner au moins 3 actifs.")
    st.stop()

asset_ids = [ASSET_MAP[a] for a in assets]

@st.cache_data(ttl=300)
def cached_result(asset_ids, vs, days, weights, rebalance):
    return build_portfolio_result(asset_ids, vs=vs, days=days, weights=weights, rebalance=rebalance)

try:
    res = cached_result(tuple(asset_ids), vs, days, tuple(w), rebalance)
except Exception as e:
    st.error(f"Erreur data source : {e}")
    st.stop()

# --- KPIs
m = compute_metrics(res.portfolio)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Perf annualisée", f"{m['ann_return']*100:.2f}%")
c2.metric("Vol annualisée", f"{m['ann_vol']*100:.2f}%")
c3.metric("Sharpe (rf=0)", f"{m['sharpe']:.2f}")
c4.metric("Max Drawdown", f"{m['max_dd']*100:.2f}%")

st.caption(f"Dernière mise à jour : {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')} (refresh auto 5 min)")

# --- Main chart: prix normalisés + portefeuille
norm_prices = (res.prices / res.prices.iloc[0]) * 100.0

fig = go.Figure()
for col in norm_prices.columns:
    fig.add_trace(go.Scatter(x=norm_prices.index, y=norm_prices[col], mode="lines", name=col))

fig.add_trace(go.Scatter(x=res.portfolio.index, y=res.portfolio.values, mode="lines", name="Portfolio (base 100)"))
fig.update_layout(
    title="Prix normalisés (base 100) + Portefeuille",
    xaxis_title="Temps",
    yaxis_title="Base 100",
    legend_title="Séries",
)
st.plotly_chart(fig, use_container_width=True)

# --- Corr matrix
st.subheader("Matrice de corrélation (rendements)")
corr_fig = px.imshow(res.corr, aspect="auto", text_auto=True)
st.plotly_chart(corr_fig, use_container_width=True)

# --- Raw table
with st.expander("Voir les dernières valeurs"):
    last = res.prices.tail(10).copy()
    last["portfolio"] = res.portfolio.tail(10)
    st.dataframe(last)
