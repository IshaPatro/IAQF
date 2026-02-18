import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

@st.cache_data
def load_data():
    btc_usd = pd.read_csv(os.path.join(DATA_DIR, "BTC_USD_coinbase.csv"), parse_dates=["timestamp"])
    btc_usdt = pd.read_csv(os.path.join(DATA_DIR, "BTC_USDT_binance.csv"), parse_dates=["timestamp"])

    btc_usd = btc_usd.rename(columns={"close": "btc_usd"})
    btc_usdt = btc_usdt.rename(columns={"close": "btc_usdt"})

    merged = btc_usd[["timestamp", "btc_usd"]].merge(
        btc_usdt[["timestamp", "btc_usdt"]], on="timestamp", how="inner"
    )
    
    merged["timestamp"] = pd.to_datetime(merged["timestamp"], utc=True)
    merged = merged.sort_values("timestamp").reset_index(drop=True)

    merged["implied_usdt"] = merged["btc_usd"] / merged["btc_usdt"]
    merged["premium_pct"] = (merged["implied_usdt"] - 1.0) * 100
    
    return merged

def _chart_layout(fig, title, yaxis_title, height=420):
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, family="Inter")),
        yaxis_title=yaxis_title,
        template="plotly_dark",
        height=height,
        margin=dict(l=60, r=30, t=50, b=40),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.05)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.05)")
    return fig

def render():
    st.header("**The 'Kimchi Premium' vs. The 'Regulatory Premium'**")
    st.markdown(
        """
        Just as capital controls in South Korea create the **"Kimchi Premium"** (where BTC trades higher 
        in KRW terms due to arbitrage limits), the **GENIUS Act** creates a **"Regulatory Premium."**
        """
    )
    
    st.markdown("#### The Concept")
    st.info(
        """
        **Post-GENIUS Act, the U.S. stablecoin market is walled off from the offshore market.**
        
        *   **Offshore (USDT):** Higher friction to convert to fiat USD, higher censorship resistance, lower regulatory compliance.
        *   **Onshore (USDC/GENIUS-Coin):** Seamless conversion to fiat via FedNow/Visa, high regulatory compliance.
        """
    )

    df = load_data()
    
    st.subheader("Application to March 2023 Data")
    st.markdown(
        """
        We analyze the basis spread between **BTC/USDT (Offshore/Binance)** and **BTC/USD (Onshore/Coinbase)** 
        during the banking crisis.
        """
    )
    
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=df["timestamp"], y=df["btc_usd"],
        name="BTC/USD (Onshore)", line=dict(color="#4ECDC4", width=1.2)
    ))
    fig1.add_trace(go.Scatter(
        x=df["timestamp"], y=df["btc_usdt"],
        name="BTC/USDT (Offshore)", line=dict(color="#FF6B6B", width=1.2)
    ))
    
    svb_start = pd.Timestamp("2023-03-10 00:00:00", tz="UTC")
    svb_end = pd.Timestamp("2023-03-13 23:59:59", tz="UTC")
    
    fig1.add_vrect(
        x0=svb_start, x1=svb_end,
        fillcolor="rgba(255,107,107,0.15)", line_width=0,
        annotation_text="SVB Crisis", annotation_position="top left"
    )
    
    _chart_layout(fig1, "Price Divergence: Onshore vs Offshore", "Price")
    st.plotly_chart(fig1, use_container_width=True)

    st.markdown("#### Observation")
    st.markdown(
        """
        As USDC de-pegged, BTC/USD prices dropped (relative to stablecoin pairs), but BTC/USDT prices surged. 
        This massive divergence implies a **"flight to safety"** where the market paradoxically viewed the 
        unregulated offshore dollar (USDT) as safer than the regulated onshore dollar (USDC) entangled in the banking crisis.
        """
    )
    
    st.divider()
    
    st.subheader("The 'Tether Premium'")
    
    st.latex(r"\text{Implied USDT Price} = \frac{P_{BTC/USD}}{P_{BTC/USDT}}")
    
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=df["timestamp"], y=df["implied_usdt"],
        name="Implied USDT", line=dict(color="#FFD93D", width=1.5),
        fill="tozeroy", fillcolor="rgba(255, 217, 61, 0.1)"
    ))
    fig2.add_hline(y=1.0, line_dash="dash", line_color="white", annotation_text="Parity ($1.00)")
    
    fig2.add_vrect(
        x0=svb_start, x1=svb_end,
        fillcolor="rgba(255,107,107,0.15)", line_width=0,
        annotation_text="SVB Crisis", annotation_position="top left"
    )

    _chart_layout(fig2, "Implied USDT Price (The 'Tether Premium')", "USD")
    st.plotly_chart(fig2, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    max_prem = df[df["timestamp"].between(svb_start, svb_end)]["implied_usdt"].max()
    
    with col1:
       st.metric("Peak Implied USDT Price", f"${max_prem:.4f}")
       
    st.markdown("#### Nuance: Flipping the Script")
    st.markdown(
        """
        This flips the standard **"Kimchi Premium"** logic. Usually, premiums exist where capital is trapped. 
        Here, the premium (high BTC price in USDT? **No!**) signaled that USDT was strong.
        """
    )
    
    st.warning(
        """
        If $P_{BTC/USDT}$ is **lower** than $P_{BTC/USD}$, it implies USDT is worth **more** than USD.
        
        In March 2023, USDT actually traded at a premium to USD ($1.01–$1.03) while USDC traded at a discount. 
        Therefore, $P_{BTC/USDT}$ was actually **lower** than the theoretical parity price (since the denominator, USDT, was strong).
        """
    )
    
    st.markdown(
        """
        **The 'Tether Premium':** Investigating why USDT traded above $1.00$ while the U.S. banking system collapsed 
        is a fascinating angle. It suggests a **"de-coupling"** where the offshore crypto economy was viewed as 
        insulated from U.S. regional banking contagion.
        """
    )
