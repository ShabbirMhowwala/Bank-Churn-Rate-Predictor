"""Global visual system for the premium Streamlit interface."""

from __future__ import annotations

from pathlib import Path

import streamlit as st
from streamlit_extras.metric_cards import style_metric_cards


ROOT_DIR = Path(__file__).resolve().parents[1]
CSS_PATH = ROOT_DIR / "assets" / "css" / "theme.css"


def apply_page_config(title: str = "FinGuard AI") -> None:
    st.set_page_config(
        page_title=title,
        page_icon="FG",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def inject_theme() -> None:
    css = CSS_PATH.read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    style_metric_cards(
        background_color="rgba(15, 23, 42, 0.55)",
        border_left_color="#22d3ee",
        border_color="rgba(148, 163, 184, 0.22)",
        box_shadow="0 18px 60px rgba(0, 0, 0, 0.28)",
    )


def page_bootstrap(title: str = "FinGuard AI") -> None:
    apply_page_config(title)
    inject_theme()
