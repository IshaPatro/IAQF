import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from sklearn.linear_model import LinearRegression

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

@st.cache_data
def load_data():
    btc_usd = pd.read_csv(os.path.join(DATA_DIR, "BTC_USD_coinbase.csv"), parse_dates=["timestamp"])
    btc_usdt = pd.read_csv(os.path.join(DATA_DIR, "BTC_USDT_binance.csv"), parse_dates=["timestamp"])
    btc_usdc = pd.read_csv(os.path.join(DATA_DIR, "BTC_USDC_binance.csv"), parse_dates=["timestamp"])

    btc_usd = btc_usd.rename(columns={"close": "btc_usd"})
    btc_usdt = btc_usdt.rename(columns={"close": "btc_usdt"})
    btc_usdc = btc_usdc.rename(columns={"close": "btc_usdc"})

    merged_2023 = btc_usd[["timestamp", "btc_usd"]].merge(
        btc_usdt[["timestamp", "btc_usdt"]], on="timestamp", how="inner"
    ).merge(
        btc_usdc[["timestamp", "btc_usdc"]], on="timestamp", how="inner"
    )
    
    merged_2023["timestamp"] = pd.to_datetime(merged_2023["timestamp"], utc=True)
    merged_2023 = merged_2023.sort_values("timestamp").reset_index(drop=True)

    merged_2023["implied_usdt"] = merged_2023["btc_usd"] / merged_2023["btc_usdt"]
    
    merged_2023["implied_usdc"] = merged_2023["btc_usd"] / merged_2023["btc_usdc"]
    merged_2023["usdc_premium_bps"] = (merged_2023["btc_usdc"] / merged_2023["btc_usd"] - 1.0) * 10000
    
    ust_path = os.path.join(DATA_DIR, "UST_USD_yfinance.csv")
    luna_path = os.path.join(DATA_DIR, "LUNA_USD_yfinance.csv")
    
    merged_2022 = pd.DataFrame()
    if os.path.exists(ust_path) and os.path.exists(luna_path):
        try:
             cols = ["Date", "ust_price", "High", "Low", "Open", "Volume"]
             ust = pd.read_csv(ust_path, skiprows=3, header=None, names=cols)
             
             cols = ["Date", "luna_price", "High", "Low", "Open", "Volume"]
             luna = pd.read_csv(luna_path, skiprows=3, header=None, names=cols)
             
             ust["Date"] = pd.to_datetime(ust["Date"])
             luna["Date"] = pd.to_datetime(luna["Date"])
             
             merged_2022 = pd.merge(ust[["Date", "ust_price"]], luna[["Date", "luna_price"]], on="Date", how="inner")
             merged_2022 = merged_2022.sort_values("Date")
        except Exception as e:
             st.error(f"Error parse Terra data: {e}")

    return merged_2023, merged_2022

def calculate_half_life(df):
    svb_start = pd.Timestamp("2023-03-10 00:00:00", tz="UTC")
    svb_end = pd.Timestamp("2023-03-13 23:59:59", tz="UTC")
    
    crisis_df = df[(df["timestamp"] >= svb_start) & (df["timestamp"] <= svb_end)].copy()
    col = "usdc_premium_bps"
    
    crisis_df["premium_lag1"] = crisis_df[col].shift(1)
    clean = crisis_df.dropna(subset=[col, "premium_lag1"])
    
    if len(clean) > 100:
        X = clean["premium_lag1"].values.reshape(-1, 1)
        y = clean[col].values
        model = LinearRegression().fit(X, y)
        persistence = float(model.coef_[0])
        
        if 0 < persistence < 1:
            half_life_min = -np.log(2) / np.log(persistence)
            return half_life_min / 60.0 # Return in hours
            
    return np.nan

def calculate_correlations(df_2023, df_2022):
    svb_start = pd.Timestamp("2023-03-10 00:00:00", tz="UTC")
    svb_end = pd.Timestamp("2023-03-13 23:59:59", tz="UTC")
    crisis_2023 = df_2023[(df_2023["timestamp"] >= svb_start) & (df_2023["timestamp"] <= svb_end)]
    corr_usdc_btc = crisis_2023["btc_usdc"].corr(crisis_2023["btc_usd"])
    corr_usdc_price_btc = crisis_2023["implied_usdc"].corr(crisis_2023["btc_usd"])
    
    crash_start = pd.Timestamp("2022-05-07")
    crash_end = pd.Timestamp("2022-05-14")
    
    corr_ust_luna = np.nan
    if not df_2022.empty and "Date" in df_2022.columns:
        crisis_2022 = df_2022[(df_2022["Date"] >= crash_start) & (df_2022["Date"] <= crash_end)]
        if not crisis_2022.empty:
            corr_ust_luna = crisis_2022["ust_price"].corr(crisis_2022["luna_price"])
            
    return corr_usdc_price_btc, corr_ust_luna

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
    st.header("Case Studies")
    
    st.markdown("""
    **Index:**
    1. [The 'Kimchi Premium' vs. The 'Regulatory Premium'](#the-kimchi-premium-vs-the-regulatory-premium)
    2. [Endogenous vs. Exogenous Failures](#endogenous-vs-exogenous-failures)
    3. [Regulatory Compliance vs. Privacy: ZKP Solutions](#regulatory-compliance-vs-privacy-zkp-solutions)
    """)
    
    st.divider()

    st.header("The 'Kimchi Premium' vs. The 'Regulatory Premium'")
    st.subheader("Research Idea")
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

    df_2023, df_2022 = load_data()
    
    st.subheader("Application to March 2023 Data")
    st.markdown(
        """
        We analyze the basis spread between **BTC/USDT (Offshore/Binance)** and **BTC/USD (Onshore/Coinbase)** 
        during the banking crisis.
        """
    )
    
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=df_2023["timestamp"], y=df_2023["btc_usd"],
        name="BTC/USD (Onshore)", line=dict(color="#4ECDC4", width=1.2)
    ))
    fig1.add_trace(go.Scatter(
        x=df_2023["timestamp"], y=df_2023["btc_usdt"],
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
        x=df_2023["timestamp"], y=df_2023["implied_usdt"],
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
    
    max_prem = df_2023[df_2023["timestamp"].between(svb_start, svb_end)]["implied_usdt"].max()
    
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
    

    st.divider()

    st.header("Endogenous vs. Exogenous Failures")
    corr_usdc, corr_ust = calculate_correlations(df_2023, df_2022)
    half_life_hours = calculate_half_life(df_2023)

    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Terra/UST (May 2022): The Endogenous Death Spiral")
        st.error(
            f"""
            **What Happened:**
            UST was an algorithmic stablecoin backed by LUNA. When confidence shook, the mint/burn mechanism 
            forced the protocol to print infinite LUNA to buy back UST.
            
            **What Was Wrong:**
            The backing was *endogenous*. The value of the collateral (LUNA) was derived from the success of 
            the stablecoin itself. This created a positive feedback loop (death spiral).
            
            **Inference:**
            Type I failures are **terminal**. Once the death spiral begins, there is no floor. 
            The correlation between the stablecoin and its collateral approaches +1.0 during the crash 
            (Empirical Correlation: **{corr_ust:.2f}**).
            """
        )

    with col2:
        st.markdown("#### USDC (March 2023): The Exogenous Liquidity Shock")
        st.success(
            f"""
            **What Happened:**
            USDC reserves were held at Silicon Valley Bank (SVB), which failed. \$3.3B of reserves were trapped.
            
            **What Was Wrong:**
            The backing was *exogenous* (US Treasuries/Cash), but the custody was centralized. The failure was 
            due to a counterparty (SVB) in the traditional banking system, not the stablecoin's design.
            
            **Inference:**
            Type II failures are **temporary**. The assets existed but were illiquid. Once access was restored, 
            the peg recovered. The correlation with the broader crypto market remained low/negative 
            (Empirical Correlation: **{corr_usdc:.2f}**).
            """
        )
    
    # Death Spiral Chart
    if not df_2022.empty and "Date" in df_2022.columns:
        st.markdown("#### The Death Spiral Visualization (May 2022)")
        
        crash_start = pd.Timestamp("2022-05-01")
        crash_end = pd.Timestamp("2022-05-31")
        plot_df = df_2022[(df_2022["Date"] >= crash_start) & (df_2022["Date"] <= crash_end)]
        
        fig3 = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig3.add_trace(
            go.Scatter(
                x=plot_df["Date"], y=plot_df["ust_price"],
                name="UST Price", line=dict(color="#EF553B", width=2)
            ),
            secondary_y=False,
        )
        
        fig3.add_trace(
            go.Scatter(
                x=plot_df["Date"], y=plot_df["luna_price"],
                name="LUNA Price", line=dict(color="#00CC96", width=2, dash="dot")
            ),
            secondary_y=True,
        )
        
        fig3.update_yaxes(title_text="UST Price ($)", secondary_y=False, gridcolor="rgba(255,255,255,0.05)")
        fig3.update_yaxes(title_text="LUNA Price ($)", secondary_y=True, showgrid=False)
        
        _chart_layout(fig3, "Terra/UST Death Spiral: Positive Correlation", "")
        st.plotly_chart(fig3, use_container_width=True)

    st.markdown("#### Research Implication")
    
    st.info(
        f"""
        This comparison validates the GENIUS Act's ban on algorithmic stablecoins. The data shows that Type I 
        failures are terminal (absorbing states), while Type II failures are temporary (mean-reverting) if 
        the government intervenes.

        **Empirical Evidence:**
        Our model shows the Mean Reversion Speed (Half-Life) of USDC during the crisis was approximately **{half_life_hours:.2f} hours**.
        This rapid recovery stands in stark contrast to UST's terminal decline.
        """
    )

    st.divider()

    st.header("Regulatory Compliance vs. Privacy: ZKP Solutions")
    
    st.markdown(
        """
        The **GENIUS Act** introduces a fundamental tension: it secures the onshore market by subjecting stablecoin issuers 
        to the **Bank Secrecy Act (BSA)**, but in doing so, it risks "de-pegging" the U.S. from the global, 
        privacy-centric offshore crypto ecosystem.
        """
    )

    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### The Friction: BSA & AML Compliance")
        st.error(
            """
            **Requirement:**
            Stablecoin issuers must implement strict Anti-Money Laundering (AML) and sanctions compliance programs.
            
            **The Problem:**
            Traditional compliance often requires the exchange of sensitive PII (Personally Identifiable Information). 
            Offshore participants—who may prioritize censorship resistance and privacy—may find these "onshore gates" 
            too high, leading to a fragmented global market.
            """
        )

    with col2:
        st.markdown("#### The Solution: ZKP 'Compliance-by-Design'")
        st.success(
            """
            **Zero-Knowledge Proofs (ZKPs):**
            Enable a "proof of validity" without revealing underlying data. An institution can prove a 
            transaction is compliant without sharing sensitive customer details.
            
            **Technology at Work:**
            Initiatives like the BIS's **"Project Mandala"** use ZKPs to verify compliance statements 
            (sanctions screening, capital flow checks) across borders automatically.
            """
        )

    st.info(
        """
        **The Future Architecture: "Compliance-by-Design"**
        
        By automating compliance and attaching cryptographic proofs to digital assets, ZKPs reduce the 
        friction between regulated onshore liquidity and the global market. 
        
        This architecture preserves **transactional privacy** while ensuring **regulatory adherence**, 
        potentially solving the stability vs. control trade-off of the GENIUS Act era.
        """
    )

