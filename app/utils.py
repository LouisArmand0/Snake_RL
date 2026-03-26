"""Utilitaires partages entre les pages Streamlit."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st

# Ajouter src/ au path pour les imports snake_rl
_SRC_DIR = str(Path(__file__).resolve().parent.parent / "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from snake_rl.agent import SnakeAgent  # noqa: E402
from snake_rl.env import make_env  # noqa: E402

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts" / "grid_results"


def inject_css() -> None:
    """Injecte le CSS personnalise dans la page."""
    css_path = Path(__file__).parent / "style.css"
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)


@st.cache_data
def load_manifest() -> dict | None:
    """Charge le manifest des resultats de grid search."""
    manifest_path = ARTIFACTS_DIR / "manifest.json"
    if not manifest_path.exists():
        return None
    with open(manifest_path) as f:
        return json.load(f)


@st.cache_data
def load_history(run_id: str) -> dict:
    """Charge l'historique d'entrainement d'un run."""
    path = ARTIFACTS_DIR / run_id / "history.json"
    with open(path) as f:
        return json.load(f)


@st.cache_data
def load_metrics(run_id: str) -> dict:
    """Charge les metriques d'evaluation d'un run."""
    path = ARTIFACTS_DIR / run_id / "metrics.json"
    with open(path) as f:
        return json.load(f)


@st.cache_data
def load_config(run_id: str) -> dict:
    """Charge la configuration d'un run."""
    path = ARTIFACTS_DIR / run_id / "config.json"
    with open(path) as f:
        return json.load(f)


def load_agent(run_id: str, grid_size: int = 10) -> tuple[SnakeAgent, any]:
    """Charge un agent entraine et cree un environnement associe."""
    model_path = ARTIFACTS_DIR / run_id / "model.pkl"
    env = make_env(size=grid_size, record_stats=False)
    agent = SnakeAgent(
        env=env,
        learning_rate=0.01,
        initial_epsilon=0.0,
        epsilon_decay=0.0,
        final_epsilon=0.0,
    )
    agent.load(model_path)
    agent.epsilon = 0.0
    return agent, env
