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

<<<<<<< HEAD
# Page Configuration
st.set_page_config(page_title="Single Asset (Quant A)", layout="wide")

# Auto-refresh every 5 minutes
st_autorefresh(interval=5 * 60 * 1000, key="single_asset_refresh")

st.title("Single Asset Analysis (Quant A) — Univariate")
=======
st.set_page_config(page_title="Single Asset (Quant A)", layout="wide")

st_autorefresh(interval=5 * 60 * 1000, key="single_asset_refresh")

# --- Small styling
st.markdown(
    """
<style>
.kpi-card {padding: 14px; border-radius: 18px; background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.10);}
.small-muted {opacity: 0.7; font-size: 12px;}
</style>
""",
    unsafe_allow_html=True,
)

st.title("📈 Single Asset (Quant A) — Univariate Strategies")
>>>>>>> origin/feature/quant-a

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
<<<<<<< HEAD
        help="raw = original CoinGecko data. Otherwise resampled (last + ffill).",
=======
>>>>>>> origin/feature/quant-a
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
<<<<<<< HEAD
        lookback = st.number_input("Momentum Lookback (points)", min_value=1, max_value=500, value=20, step=1)

    if strategy == "SMA Crossover":
        sma_short = st.number_input("SMA Short Window", min_value=1, max_value=500, value=10, step=1)
        sma_long = st.number_input("SMA Long Window", min_value=2, max_value=1000, value=30, step=1)

    st.divider()
    st.subheader("Extra Features (Optional)")
    enable_forecast = st.checkbox("Show Linear Forecast (Price)", value=False)
    horizon = st.number_input("Forecast Horizon (points)", min_value=5, max_value=200, value=20, step=5)
    fit_last = st.number_input("Fit on last N points", min_value=50, max_value=2000, value=200, step=50)
=======
        lookback = st.number_input("Lookback momentum (#points)", min_value=1, max_value=500, value=20, step=1)

    if strategy == "SMA Crossover":
        sma_short = st.number_input("SMA short (#points)", min_value=1, max_value=500, value=10, step=1)
        sma_long = st.number_input("SMA long (#points)", min_value=2, max_value=1000, value=30, step=1)

    st.divider()
    st.subheader("Bonus (optionnel)")
    rf = st.number_input("Risk-free annuel (ex: 0.02)", min_value=0.0, max_value=0.2, value=0.0, step=0.005)
    enable_forecast = st.checkbox("Afficher forecast linéaire (prix)", value=False)
    horizon = st.number_input("Horizon forecast (#points)", min_value=5, max_value=200, value=20, step=5)
    fit_last = st.number_input("Fit sur N derniers points", min_value=50, max_value=2000, value=200, step=50)

>>>>>>> origin/feature/quant-a

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

<<<<<<< HEAD
try:
    res = cached_res(asset_id, vs, days, periodicity, strategy, lookback, sma_short, sma_long)
except Exception as e:
    st.error(f"Data/Backtest Error: {e}")
    st.stop()
=======

with st.spinner("Fetching data & running backtest..."):
    try:
        res = cached_res(asset_id, vs, days, periodicity, strategy, lookback, sma_short, sma_long, allow_short, leverage, fee_bps, slippage_bps)
    except Exception as e:
        st.error(f"Erreur data/backtest : {e}")
        st.stop()
>>>>>>> origin/feature/quant-a

# --- Metrics Calculation ---
price_now = float(res.prices.iloc[-1])
kpi = compute_metrics(res.equity, rf_annual=float(rf))
oc = open_close_return_24h(res.prices)
vol = realized_vol(res.prices)
dd = drawdown_series(res.equity)

<<<<<<< HEAD
# --- KPIs Section with Deltas ---
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
tab1, tab2 = st.tabs(["📈 Visualization & Strategy", "📄 Raw Data & Stats"])

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
=======
# --- KPIs header
c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Prix actuel", f"{price_now:,.2f} {vs.upper()}")
c2.metric("CAGR", f"{kpi['cagr']*100:.2f}%")
c3.metric("Vol ann.", f"{kpi['ann_vol']*100:.2f}%")
c4.metric("Sharpe", f"{kpi['sharpe']:.2f}")
c5.metric("Sortino", f"{kpi['sortino']:.2f}")
c6.metric("Max DD", f"{kpi['max_dd']*100:.2f}%")

st.caption(
    f"Dernière mise à jour : {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')} (refresh auto 5 min) — "
    f"stratégie: {res.strategy_name} — params: {res.params} — source: {res.meta.get('source')} — trades: {res.meta.get('trades')}"
)

tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "🧪 Diagnostics", "⬇️ Export"])

# --------- TAB 1
with tab1:
    # 3 rows: price+equity, drawdown, position
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.55, 0.25, 0.20],
        specs=[[{"secondary_y": True}], [{"secondary_y": False}], [{"secondary_y": False}]],
    )

    fig.add_trace(
        go.Scatter(x=res.prices.index, y=res.prices.values, mode="lines", name=f"Prix ({label})"),
        row=1, col=1, secondary_y=False
    )
    fig.add_trace(
        go.Scatter(x=res.equity.index, y=res.equity.values, mode="lines", name="Equity stratégie (base 100)"),
        row=1, col=1, secondary_y=True
    )

    # SMA overlay if available
    if strategy == "SMA Crossover" and "sma_short" in res.meta and "sma_long" in res.meta:
        sma_s = res.meta["sma_short"]
        sma_l = res.meta["sma_long"]
        fig.add_trace(go.Scatter(x=sma_s.index, y=sma_s.values, mode="lines", name="SMA short", line=dict(dash="dot")), row=1, col=1, secondary_y=False)
        fig.add_trace(go.Scatter(x=sma_l.index, y=sma_l.values, mode="lines", name="SMA long", line=dict(dash="dot")), row=1, col=1, secondary_y=False)

    # Forecast (optionnel)
    if enable_forecast:
        try:
            fc = simple_linear_forecast(res.prices, horizon=int(horizon), fit_last=int(fit_last))
            fig.add_trace(go.Scatter(x=fc.index, y=fc["yhat"], mode="lines", name="Forecast", line=dict(dash="dash")), row=1, col=1, secondary_y=False)
            fig.add_trace(go.Scatter(x=fc.index, y=fc["hi"], mode="lines", name="Forecast hi", line=dict(dash="dot")), row=1, col=1, secondary_y=False)
            fig.add_trace(go.Scatter(x=fc.index, y=fc["lo"], mode="lines", name="Forecast lo", line=dict(dash="dot")), row=1, col=1, secondary_y=False)
        except Exception as e:
            st.warning(f"Forecast impossible: {e}")

    # Drawdown
    fig.add_trace(go.Scatter(x=dd.index, y=dd.values, mode="lines", name="Drawdown"), row=2, col=1)

    # Position
    fig.add_trace(go.Scatter(x=res.position.index, y=res.position.values, mode="lines", name="Position"), row=3, col=1)

    fig.update_layout(
        title="Prix (gauche) + Equity stratégie (droite) + Drawdown + Position",
        legend_title="Séries",
        height=850,
    )
    fig.update_yaxes(title_text=f"Prix ({vs.upper()})", row=1, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Equity (base 100)", row=1, col=1, secondary_y=True)
    fig.update_yaxes(title_text="Drawdown", row=2, col=1)
    fig.update_yaxes(title_text="Position", row=3, col=1)

    st.plotly_chart(fig, use_container_width=True)

    # 24h block
    st.subheader("Stats 24h (prix)")
    d1, d2, d3, d4, d5 = st.columns(5)
    d1.metric("Open 24h", f"{oc['open_24h']:.2f}")
    d2.metric("Close 24h", f"{oc['close_24h']:.2f}")
    d3.metric("Return 24h", f"{oc['return_24h']*100:.2f}%")
    d4.metric("Vol ann. estimée", f"{vol['ann_vol_est']*100:.2f}%")
    d5.metric("Turnover", f"{res.meta.get('turnover', 0.0):.2f}")

# --------- TAB 2
with tab2:
    st.subheader("Diagnostics")
    rets = res.equity.pct_change().dropna()
    st.write("Retour stratégie (dernier point):", float(rets.iloc[-1]) if not rets.empty else 0.0)

    # histogram returns
    if not rets.empty:
        hist = go.Figure()
        hist.add_trace(go.Histogram(x=rets.values, nbinsx=50, name="Strategy returns"))
        hist.update_layout(title="Distribution des rendements (stratégie)")
        st.plotly_chart(hist, use_container_width=True)

    st.subheader("Dernières valeurs")
    df = pd.DataFrame({"price": res.prices, "equity": res.equity, "position": res.position})
    st.dataframe(df.tail(80), use_container_width=True)

# --------- TAB 3
with tab3:
    st.subheader("Export / Download")
    df = pd.DataFrame({"price": res.prices, "equity": res.equity, "position": res.position})
    st.download_button(
        "⬇️ Télécharger CSV (price/equity/position)",
        data=df.to_csv().encode("utf-8"),
        file_name=f"quant_a_{asset_id}_{vs}.csv",
        mime="text/csv",
    )

    report = {
        "generated_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "asset": asset_id,
        "vs": vs,
        "strategy": res.strategy_name,
        "params": res.params,
        "meta": {k: (str(v) if not isinstance(v, (int, float, str, bool)) else v) for k, v in res.meta.items() if k not in {"sma_short","sma_long"}},
        "kpi": kpi,
        "price_now": price_now,
        "last_24h": {**oc, **vol},
    }
    st.download_button(
        "⬇️ Télécharger Report JSON",
        data=json.dumps(report, indent=2).encode("utf-8"),
        file_name=f"quant_a_report_{asset_id}_{vs}.json",
        mime="application/json",
    )
>>>>>>> origin/feature/quant-a
