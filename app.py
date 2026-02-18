import streamlit as st
import landing_page
import basis_analysis
import strategies
import stablecoin_dynamics

st.set_page_config(
    page_title="Cross-Currency Dynamics under Stablecoin Regulation",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 900px;
    }

    h1 {
        font-weight: 700;
        letter-spacing: -0.02em;
    }

    h2, h3 {
        font-weight: 600;
        letter-spacing: -0.01em;
        margin-top: 2rem;
    }

    p {
        font-size: 1.05rem;
        line-height: 1.75;
        color: rgba(250, 250, 250, 0.85);
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        border-bottom: 1px solid rgba(250, 250, 250, 0.1);
    }

    .stTabs [data-baseweb="tab"] {
        font-family: 'Inter', sans-serif;
        font-weight: 500;
        font-size: 0.95rem;
        letter-spacing: 0.02em;
        padding-bottom: 0.75rem;
        padding-top: 1rem;
    }

    .stTabs [aria-selected="true"] {
        border-bottom-color: #6C9BF2;
    }

    .stDivider {
        margin-top: 0.5rem;
        margin-bottom: 1.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

tabs = st.tabs([
    "Overview",
    "Cross-Currency Basis Analysis",
    "Cross-Currency Basis Arbitrage",
    "Stablecoin Dynamics",
])

with tabs[0]:
    landing_page.render()

with tabs[1]:
    basis_analysis.render()

with tabs[2]:
    strategies.render()

with tabs[3]:
    stablecoin_dynamics.render()
