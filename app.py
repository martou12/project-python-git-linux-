import streamlit as st

# Page Configuration
st.set_page_config(page_title="Python Git Linux Finance Dashboard", layout="wide")

# Main Title
st.title("Financial Dashboard – Quant A & Quant B")

# Introduction Section
st.markdown(
    """
    ### Welcome to the Collaborative Financial Platform
    
    This dashboard provides real-time financial analytics and automated reporting.
    
    - **Module A (Single Asset)**: In-depth analysis of a single asset (BTC) including backtesting and risk metrics.
    - **Module B (Portfolio)**: Multi-asset strategy (BTC, ETH, SOL) focusing on correlation and diversification.
    
    **Navigation**: Use the sidebar menu on the left to switch between modules.
    """
)

# Status Information
st.info("🔄 Auto-refresh enabled: Data updates every 5 minutes on Quant A and Quant B pages.")