import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from streamlit_autorefresh import st_autorefresh

from src.data_fetch import ASSET_MAP
from src.metrics import compute_metrics, open_close_return_24h, realized_vol
from src.quant_a_single_asset import build_single_asset_result, simple_linear_forecast


st.set_page_config(page_title="Single Asset (Quant A)", layout="wide")

# refresh auto 5 min
st_autorefresh(interval=5 * 60 * 1000, key="single_asset_refresh")

st.title("Single Asset (Quant A) — Univariate")

with st.sidebar:
    st.header("Paramètres")

    label = st.selectbox("Actif", options=list(ASSET_MAP.keys()), index=0)
    asset_id = ASSET_MAP[label]

    vs = st.selectbox("Devise", ["eur", "usd"], index=0)
    days = st.selectbox("Historique (jours)", [1, 7, 30, 90, 180, 365], index=2)

    periodicity = st.selectbox(
        "Périodicité / Resample",
        options=["raw", "5min", "15min", "1H", "4H", "1D"],
        index=0,
        help="raw = données CoinGecko brutes. Sinon on resample (last + ffill).",
    )

    strategy = st.selectbox("Stratégie", ["Buy & Hold", "Momentum", "SMA Crossover"], index=1)

    lookback = 20
    sma_short, sma_long = 10, 30

    if strategy == "Momentum":
        lookback = st.number_input("Lookback momentum (nb points)", min_value=1, max_value=500, value=20, step=1)

    if strategy == "SMA Crossover":
        sma_short = st.number_input("SMA short (nb points)", min_value=1, max_value=500, value=10, step=1)
        sma_long = st.number_input("SMA long (nb points)", min_value=2, max_value=1000, value=30, step=1)

    st.divider()
    st.subheader("Bonus (optionnel)")
    enable_forecast = st.checkbox("Afficher forecast linéaire (prix)", value=False)
    horizon = st.number_input("Horizon forecast (nb points)", min_value=5, max_value=200, value=20, step=5)
    fit_last = st.number_input("Fit sur les N derniers points", min_value=50, max_value=2000, value=200, step=50)


@st.cache_data(ttl=300)
def cached_res(asset_id, vs, days, periodicity, strategy, lookback, sma_short, sma_long):
    return build_single_asset_result(
        asset_id=asset_id,
        vs=vs,
        days=int(days),
        periodicity=periodicity,
        strategy=strategy,
        lookback=int(lookback),
        sma_short=int(sma_short),
        sma_long=int(sma_long),
    )


try:
    res = cached_res(asset_id, vs, days, periodicity, strategy, lookback, sma_short, sma_long)
except Exception as e:
    st.error(f"Erreur data/backtest : {e}")
    st.stop()

price_now = float(res.prices.iloc[-1])
kpi = compute_metrics(res.equity)
oc = open_close_return_24h(res.prices)
vol = realized_vol(res.prices)

# --- KPIs
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Prix actuel", f"{price_now:,.2f} {vs.upper()}")
c2.metric("Perf annualisée", f"{kpi['ann_return']*100:.2f}%")
c3.metric("Vol annualisée", f"{kpi['ann_vol']*100:.2f}%")
c4.metric("Sharpe (rf=0)", f"{kpi['sharpe']:.2f}")
c5.metric("Max Drawdown", f"{kpi['max_dd']*100:.2f}%")

st.caption(
    f"Dernière mise à jour : {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')} "
    f"(refresh auto 5 min) — stratégie: {res.strategy_name} — params: {res.params}"
)

# --- Main chart: prix brut + equity stratégie (2 courbes, 2 axes)
fig = make_subplots(specs=[[{"secondary_y": True}]])
fig.add_trace(
    go.Scatter(x=res.prices.index, y=res.prices.values, mode="lines", name=f"Prix ({label})"),
    secondary_y=False,
)
fig.add_trace(
    go.Scatter(x=res.equity.index, y=res.equity.values, mode="lines", name=f"Valeur stratégie (base 100)"),
    secondary_y=True,
)

if enable_forecast:
    try:
        fc = simple_linear_forecast(res.prices, horizon=int(horizon), fit_last=int(fit_last))
        fig.add_trace(go.Scatter(x=fc.index, y=fc["yhat"], mode="lines", name="Forecast (yhat)", line=dict(dash="dash")), secondary_y=False)
        fig.add_trace(go.Scatter(x=fc.index, y=fc["hi"], mode="lines", name="Forecast hi", line=dict(dash="dot")), secondary_y=False)
        fig.add_trace(go.Scatter(x=fc.index, y=fc["lo"], mode="lines", name="Forecast lo", line=dict(dash="dot")), secondary_y=False)
    except Exception as e:
        st.warning(f"Forecast impossible: {e}")

fig.update_layout(
    title="Prix brut + Valeur cumulée de la stratégie",
    xaxis_title="Temps",
    legend_title="Séries",
)
fig.update_yaxes(title_text=f"Prix ({vs.upper()})", secondary_y=False)
fig.update_yaxes(title_text="Valeur stratégie (base 100)", secondary_y=True)

st.plotly_chart(fig, use_container_width=True)

# Infos 24h
st.subheader("Stats 24h (prix)")
c6, c7, c8, c9 = st.columns(4)
c6.metric("Open 24h", f"{oc['open_24h']:.2f}")
c7.metric("Close 24h", f"{oc['close_24h']:.2f}")
c8.metric("Return 24h", f"{oc['return_24h']*100:.2f}%")
c9.metric("Vol ann. estimée", f"{vol['ann_vol_est']*100:.2f}%")

# --- Table
with st.expander("Voir les dernières valeurs"):
    df = pd.DataFrame({"price": res.prices, "equity": res.equity, "position": res.position})
    st.dataframe(df.tail(30))
