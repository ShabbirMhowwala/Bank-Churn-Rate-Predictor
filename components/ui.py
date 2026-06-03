"""Small composable UI blocks used across app pages."""

from __future__ import annotations

from pathlib import Path

import streamlit as st
from streamlit_lottie import st_lottie
from streamlit_option_menu import option_menu


ROOT_DIR = Path(__file__).resolve().parents[1]
LOTTIE_PATH = ROOT_DIR / "assets" / "animations" / "ai_pulse.json"


PAGE_ROUTES = {
    "Dashboard": "pages/1_Dashboard.py",
    "Predict": "pages/2_Predict.py",
    "Analytics": "pages/3_Analytics.py",
    "Model Performance": "pages/4_Model_Performance.py",
    "About": "pages/5_About.py",
    "Contact": "pages/6_Contact.py",
}


def app_sidebar(active: str = "Dashboard") -> None:
    with st.sidebar:
        st.markdown(
            """
            <div class="brand-card">
                <div class="brand-orb"></div>
                <div>
                    <p class="eyebrow">FinTech AI Studio</p>
                    <h2>FinGuard AI</h2>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        selected = option_menu(
            menu_title=None,
            options=list(PAGE_ROUTES.keys()),
            icons=["speedometer2", "cpu", "bar-chart-line", "activity", "layers", "person-lines-fill"],
            default_index=list(PAGE_ROUTES.keys()).index(active),
            styles={
                "container": {"padding": "0", "background-color": "transparent"},
                "icon": {"color": "#67e8f9", "font-size": "18px"},
                "nav-link": {
                    "font-size": "14px",
                    "font-weight": "600",
                    "color": "#cbd5e1",
                    "border-radius": "14px",
                    "margin": "6px 0",
                    "padding": "12px 14px",
                    "background": "rgba(255,255,255,0.03)",
                },
                "nav-link-selected": {
                    "background": "linear-gradient(135deg, #2563eb 0%, #7c3aed 48%, #06b6d4 100%)",
                    "color": "#ffffff",
                    "box-shadow": "0 0 24px rgba(34, 211, 238, 0.25)",
                },
            },
        )
        if selected != active:
            st.switch_page(PAGE_ROUTES[selected])

        st.markdown(
            """
            <div class="sidebar-footer">
                <span class="status-dot"></span>
                Model online
            </div>
            """,
            unsafe_allow_html=True,
        )


def hero(title: str, subtitle: str, badge: str, primary: str | None = None) -> None:
    action_html = (
        f'<div class="hero-action">{primary}</div>'
        if primary
        else '<div class="hero-action hero-action-ghost">Explore</div>'
    )
    html = "".join(
        [
            '<section class="hero-panel">',
            '<div class="hero-copy">',
            f'<span class="hero-badge">{badge}</span>',
            f"<h1>{title}</h1>",
            f"<p>{subtitle}</p>",
            action_html,
            "</div>",
            '<div class="hero-visual">',
            '<div class="signal-ring ring-a"></div>',
            '<div class="signal-ring ring-b"></div>',
            '<div class="ai-core">AI</div>',
            "</div>",
            "</section>",
        ]
    )
    st.markdown(html, unsafe_allow_html=True)


def metric_card(label: str, value: str, detail: str, accent: str = "cyan") -> None:
    st.markdown(
        f"""
        <div class="glass-card metric-glow {accent}">
            <p class="metric-label">{label}</p>
            <h3>{value}</h3>
            <span>{detail}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_title(label: str, title: str, body: str | None = None) -> None:
    st.markdown(
        f"""
        <div class="section-title">
            <span>{label}</span>
            <h2>{title}</h2>
            {f'<p>{body}</p>' if body else ''}
        </div>
        """,
        unsafe_allow_html=True,
    )


def glass_container(html: str) -> None:
    st.markdown(f"<div class='glass-card'>{html}</div>", unsafe_allow_html=True)


def lottie_pulse(height: int = 140) -> None:
    if LOTTIE_PATH.exists():
        import json

        st_lottie(json.loads(LOTTIE_PATH.read_text(encoding="utf-8")), height=height, key=f"pulse_{height}")


def footer() -> None:
    st.markdown(
        """
        <footer class="app-footer">
            <span>FinGuard AI</span>
            <span>Deep learning churn intelligence for digital banking teams</span>
            <span><a href="https://github.com/harsh31415926" target="_blank">GitHub: harsh31415926</a></span>
        </footer>
        """,
        unsafe_allow_html=True,
    )
