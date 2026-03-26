"""Page d'accueil du dashboard Snake RL."""

from __future__ import annotations

import streamlit as st

from utils import inject_css, load_manifest

st.set_page_config(
    page_title="Snake RL Dashboard",
    page_icon="🐍",
    layout="wide",
)

inject_css()

st.title("Snake RL Dashboard")
st.markdown(
    """
    Bienvenue sur le dashboard du projet **Snake RL**.

    Ce projet utilise le **Q-learning** pour entrainer un agent a jouer au jeu Snake.
    Plusieurs configurations d'hyperparametres ont ete testees et peuvent etre comparees ici.

    ---

    **Pages disponibles :**
    - **Comparaison** : comparer les performances de differents entrainements
    - **Visualisation** : regarder un agent entraine jouer une partie
    - **Entrainement** : entrainer un agent avec des hyperparametres personnalises
    """
)

manifest = load_manifest()
if manifest:
    n_runs = len(manifest["runs"])
    st.success(f"{n_runs} configuration(s) d'entrainement disponible(s).")
else:
    st.warning(
        "Aucun resultat de grid search trouve. "
        "Lancez `python -m snake_rl.grid_search` pour entrainer les modeles."
    )
