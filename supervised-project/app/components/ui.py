"""Small reusable layout helpers."""

import streamlit as st


def page_header(eyebrow: str, title: str, description: str) -> None:
    st.markdown(
        f'<div class="eyebrow">{eyebrow}</div>'
        f'<div class="hero-title">{title}</div>'
        f'<div class="hero-copy">{description}</div>',
        unsafe_allow_html=True,
    )


def section_title(title: str, subtitle: str | None = None) -> None:
    st.markdown(f'<div class="section-heading">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.caption(subtitle)
