"""Contact page."""

from __future__ import annotations

from components.theme import page_bootstrap
from components.ui import app_sidebar, footer, hero, section_title

import streamlit as st


page_bootstrap("Contact | FinGuard AI")
app_sidebar("Contact")

hero(
    "Contact and Portfolio",
    (
        "Use this page as a polished placeholder for recruiter demos, GitHub links, LinkedIn, "
        "and future deployment information. The form is intentionally local-only for portfolio safety."
    ),
    "Connect",
)

section_title("Contact form", "Demo inquiry capture")
with st.form("contact_form"):
    c1, c2 = st.columns(2)
    name = c1.text_input("Name")
    email = c2.text_input("Email")
    role = st.selectbox("Interest", ["Recruiter", "Hiring manager", "ML engineer", "Founder", "Student"])
    message = st.text_area("Message", placeholder="Write a short note about the opportunity or feedback.")
    sent = st.form_submit_button("Save demo message")

if sent:
    st.success("Demo message captured locally for the session. Connect this form to email or a database before production use.")

section_title("Links", "Portfolio placeholders")
l1, l2, l3 = st.columns(3)
with l1:
    st.markdown("<div class='glass-card'><h3>GitHub</h3><p>Add repository link here.</p></div>", unsafe_allow_html=True)
with l2:
    st.markdown("<div class='glass-card'><h3>LinkedIn</h3><p>Add profile link here.</p></div>", unsafe_allow_html=True)
with l3:
    st.markdown("<div class='glass-card'><h3>Resume</h3><p>Add resume or portfolio link here.</p></div>", unsafe_allow_html=True)

footer()
