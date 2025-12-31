import streamlit as st

<<<<<<< HEAD
st.set_page_config(page_title="Quant Dashboard", layout="wide")
st.title("Quant Dashboard")
st.write("Utilisez le menu à gauche pour ouvrir : **Single Asset (Quant A)** ou **Portfolio (Quant B)**.")
=======
st.set_page_config(page_title="Python Git Linux Finance Dashboard", layout="wide")

st.title("Finance Dashboard — Quant A & Quant B")

st.markdown(
    """
Bienvenue.

- **Quant A (Single Asset)** : page *Single Asset (Quant A)*  
- **Quant B (Portfolio)** : page *Portfolio (Quant B)*  

Utilisez le menu Streamlit (à gauche / en haut selon votre UI) pour naviguer.
"""
)

st.info("Refresh auto toutes les 5 minutes sur les pages Quant A et Quant B.")
>>>>>>> feature/quant-a
