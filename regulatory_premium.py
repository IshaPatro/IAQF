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

    # Load Liquidity Evolution Data
    pairs = {
        "btc_usd_cb": ("BTC_USD_coinbase.csv", "btc_usd_cb"),
        "btc_usdt_bn": ("BTC_USDT_binance.csv", "btc_usdt_bn"),
        "btc_usdc_bn": ("BTC_USDC_binance.csv", "btc_usdc_bn"),
        "btc_usdc_cb": ("BTC_USDC_coinbase.csv", "btc_usdc_cb"),
        "btc_usdt_cb": ("BTC_USDT_coinbase.csv", "btc_usdt_cb"),
        "usdt_usd_cb": ("USDT_USD_coinbase.csv", "usdt_usd_cb"),
    }

    frames = {}
    for key, (fname, col_prefix) in pairs.items():
        path = os.path.join(DATA_DIR, fname)
        if os.path.exists(path):
            df = pd.read_csv(path, parse_dates=["timestamp"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            df = df.sort_values("timestamp").reset_index(drop=True)
            df = df.rename(columns={"close": f"{col_prefix}_close", "volume": f"{col_prefix}_vol"})
            frames[key] = df[["timestamp", f"{col_prefix}_close", f"{col_prefix}_vol"]]

    merged_liq = frames["btc_usd_cb"].copy()
    for key in ["btc_usdt_bn", "btc_usdc_bn", "btc_usdc_cb", "btc_usdt_cb", "usdt_usd_cb"]:
        if key in frames:
            merged_liq = merged_liq.merge(frames[key], on="timestamp", how="inner")

    merged_liq = merged_liq.sort_values("timestamp").reset_index(drop=True)

    merged_liq["implied_usdc_cb"] = merged_liq["btc_usd_cb_close"] / merged_liq["btc_usdc_cb_close"]
    merged_liq["implied_usdc_bn"] = merged_liq["btc_usd_cb_close"] / merged_liq["btc_usdc_bn_close"]
    merged_liq["usdc_discount_cb_bps"] = (merged_liq["implied_usdc_cb"] - 1.0) * 10000
    merged_liq["usdc_discount_bn_bps"] = (merged_liq["implied_usdc_bn"] - 1.0) * 10000

    merged_liq["hour"] = merged_liq["timestamp"].dt.floor("h")
    merged_liq["is_weekend"] = merged_liq["timestamp"].dt.dayofweek.isin([5, 6])

    return merged_2023, merged_2022, merged_liq

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
    3. [Liquidity Evolution](#liquidity-evolution)
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

    df_2023, df_2022, df_liq = load_data()
    
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

    st.header("Liquidity Evolution")

    st.markdown("""
    **Sub-Index:**
    1. [Act I: The Fracture](#act-i-the-fracture-march-10-11-2023)
    2. [Act II: The Two Exit Doors](#act-ii-the-two-exit-doors)
    3. [Act III: The Aftermath](#act-iii-the-aftermath-fee-structures-custodial-shifts)
    4. [Act IV: The GENIUS Act](#act-iv-the-genius-act-does-it-solve-the-problem)
    """)

    st.divider()

    # ── ACT I ──────────────────────────────────────────────────────────

    st.subheader("Act I: The Fracture (March 10–11, 2023)")

    st.markdown(
        """
        On **Friday, March 10, 2023**, Silicon Valley Bank collapsed. Circle disclosed that **$3.3 billion** 
        of USDC reserves were trapped at SVB. Within hours, Coinbase — the largest U.S. crypto exchange — 
        **halted USDC-to-USD conversions**, citing the need to wait for the banking system to reopen on Monday.

        The crypto market, which operates 24/7, was now locked out of the traditional financial system 
        on a **weekend**. Panic ensued. But the panic didn't flow in one direction — it **fractured**.
        """
    )

    hourly = df_liq.groupby("hour").agg(
        coinbase_vol=("btc_usdc_cb_vol", "sum"),
        binance_vol=("btc_usdc_bn_vol", "sum"),
        timestamp=("hour", "first"),
    ).reset_index(drop=True)

    fig4 = make_subplots(specs=[[{"secondary_y": True}]])

    fig4.add_trace(
        go.Bar(
            x=hourly["timestamp"], y=hourly["coinbase_vol"],
            name="Coinbase BTC/USDC Volume",
            marker_color="rgba(78, 205, 196, 0.6)",
        ),
        secondary_y=False,
    )
    fig4.add_trace(
        go.Bar(
            x=hourly["timestamp"], y=hourly["binance_vol"],
            name="Binance BTC/USDC Volume",
            marker_color="rgba(255, 107, 107, 0.6)",
        ),
        secondary_y=True,
    )

    fig4.add_vrect(
        x0=svb_start, x1=svb_end,
        fillcolor="rgba(255,107,107,0.15)", line_width=0,
        annotation_text="SVB Weekend", annotation_position="top left",
    )

    fig4.update_yaxes(title_text="Coinbase Volume (BTC)", secondary_y=False)
    fig4.update_yaxes(title_text="Binance Volume (BTC)", secondary_y=True, showgrid=False)
    _chart_layout(fig4, "Volume Fracture: Coinbase vs Binance (BTC/USDC)", "", height=450)
    st.plotly_chart(fig4, use_container_width=True)

    crisis_df_liq = df_liq[(df_liq["timestamp"] >= svb_start) & (df_liq["timestamp"] <= svb_end)]
    pre_crisis_liq = df_liq[df_liq["timestamp"] < svb_start]

    bn_vol_crisis = crisis_df_liq["btc_usdc_bn_vol"].sum()
    bn_vol_pre = pre_crisis_liq["btc_usdc_bn_vol"].sum() / max(1, len(pre_crisis_liq)) * len(crisis_df_liq)
    cb_vol_crisis = crisis_df_liq["btc_usdc_cb_vol"].sum()
    cb_vol_pre = pre_crisis_liq["btc_usdc_cb_vol"].sum() / max(1, len(pre_crisis_liq)) * len(crisis_df_liq)

    c1, c2 = st.columns(2)
    with c1:
        bn_change = ((bn_vol_crisis / max(bn_vol_pre, 1)) - 1) * 100
        st.metric("Binance USDC Volume Change (Crisis vs Pre)", f"+{bn_change:.0f}%")
    with c2:
        cb_change = ((cb_vol_crisis / max(cb_vol_pre, 1)) - 1) * 100
        st.metric("Coinbase USDC Volume Change (Crisis vs Pre)", f"{cb_change:.0f}%")

    st.info(
        """
        **Inference:** When Coinbase halted conversions, liquidity didn't disappear — it **migrated**. 
        Traders who needed to exit USDC flooded to Binance to swap USDC for USDT (the "Flight to Safety"), 
        while those seeking fiat sought alternatives like Kraken. The volume chart shows this fracture in real-time.
        """
    )

    st.divider()

    # ── ACT II ─────────────────────────────────────────────────────────

    st.subheader("Act II: The Two Exit Doors")

    st.markdown(
        """
        The crisis revealed that not all "exits" from USDC are equal. The market spontaneously organized 
        into two distinct corridors:
        """
    )

    col1, col2 = st.columns(2)
    with col1:
        st.error(
            r"""
            **Exit Door 1: "Flight to Fiat"**
            
            USDC $\to$ USD
            
            **Destination:** Kraken, other exchanges with direct fiat rails.
            
            **Who:** Institutional holders needing USD settlement.
            
            **Problem:** Coinbase halted this door. Kraken became the backup, 
            but with wider spreads and less liquidity.
            """
        )
    with col2:
        st.success(
            r"""
            **Exit Door 2: "Flight to Safety"**
            
            USDC $\to$ USDT
            
            **Destination:** Binance (dominant USDC/USDT liquidity).
            
            **Who:** Traders wanting stablecoin exposure without fiat dependency.
            
            **Outcome:** USDT became the "safe haven" — paradoxically, the 
            *unregulated* stablecoin was viewed as safer than the *regulated* one.
            """
        )

    fig5 = go.Figure()
    fig5.add_trace(go.Scatter(
        x=df_liq["timestamp"], y=df_liq["usdc_discount_cb_bps"],
        name="USDC Discount (Coinbase)", line=dict(color="#FF6B6B", width=1.5),
    ))
    fig5.add_trace(go.Scatter(
        x=df_liq["timestamp"], y=df_liq["usdc_discount_bn_bps"],
        name="USDC Discount (Binance)", line=dict(color="#FFD93D", width=1.5),
    ))
    fig5.add_hline(y=0, line_dash="dash", line_color="white", annotation_text="Parity")
    fig5.add_vrect(
        x0=svb_start, x1=svb_end,
        fillcolor="rgba(255,107,107,0.15)", line_width=0,
        annotation_text="SVB Weekend", annotation_position="top left",
    )
    _chart_layout(fig5, "USDC Discount: Coinbase vs Binance (bps)", "Basis Points")
    st.plotly_chart(fig5, use_container_width=True)

    peak_discount_cb = crisis_df_liq["usdc_discount_cb_bps"].min()
    peak_discount_bn = crisis_df_liq["usdc_discount_bn_bps"].min()

    c1, c2 = st.columns(2)
    with c1:
        st.metric("Peak USDC Discount (Coinbase)", f"{peak_discount_cb:.0f} bps")
    with c2:
        st.metric("Peak USDC Discount (Binance)", f"{peak_discount_bn:.0f} bps")

    st.warning(
        """
        **Key Finding:** The USDC discount was **deeper on Binance** than on Coinbase during the crisis. 
        This is counterintuitive — Binance had *more* USDC/USDT liquidity. The explanation is that 
        Binance traders were **dumping USDC en masse** to acquire USDT, creating selling pressure that 
        exceeded what Coinbase (where conversions were halted) experienced.
        """
    )

    st.divider()

    # ── ACT III ────────────────────────────────────────────────────────

    st.subheader("Act III: The Aftermath — Fee Structures & Custodial Shifts")

    st.markdown(
        """
        The SVB crisis forced the stablecoin ecosystem to adapt. Two structural changes emerged 
        that reshape how liquidity flows will behave in the next crisis:
        """
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Coinbase's New Fee Gate")
        st.info(
            """
            **Policy Change:** Coinbase introduced fees for net USDC-to-USD conversions exceeding 
            **$75 million** per rolling 30-day period (with exemptions for Tier 1 liquidity providers).
            
            **Effect:** This creates a soft "circuit breaker." During a panic, large institutional 
            redemptions are disincentivized, slowing capital flight and giving the issuer time to 
            liquidate reserves in an orderly manner.
            
            **Analogy:** This is functionally similar to a mutual fund's **redemption gate** — 
            a mechanism that limits withdrawals during stress to prevent a fire sale.
            """
        )
    with col2:
        st.markdown("#### Circle's Custodial Flight to Quality")
        st.info(
            """
            **Policy Change:** Circle shifted significant reserve custody to **BNY Mellon**, the world's 
            largest custodian bank (with $46.7 trillion in assets under custody).
            
            **Effect:** By diversifying away from regional banks like SVB, Circle reduces single-point-of-failure 
            risk. BNY Mellon is a **G-SIB** (Global Systemically Important Bank), meaning it benefits from 
            enhanced regulatory oversight and implicit government backstops.
            
            **Signal:** This move tells institutional investors: *"Your reserves are held at a bank 
            that is too big to fail."*
            """
        )

    spread = df_liq.copy()
    spread["basis_spread_bps"] = (
        (spread["btc_usd_cb_close"] / spread["btc_usdt_bn_close"]) - 1.0
    ) * 10000

    hourly_spread = spread.groupby("hour").agg(
        spread=("basis_spread_bps", "mean"),
        timestamp=("hour", "first"),
        is_weekend=("is_weekend", "first"),
    ).reset_index(drop=True)

    weekend_spread = hourly_spread[hourly_spread["is_weekend"]]["spread"]
    weekday_spread = hourly_spread[~hourly_spread["is_weekend"]]["spread"]

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Avg Weekday Spread", f"{weekday_spread.mean():.1f} bps")
    with c2:
        st.metric("Avg Weekend Spread", f"{weekend_spread.mean():.1f} bps")
    with c3:
        gap = weekend_spread.std() - weekday_spread.std()
        st.metric("Weekend Volatility Premium", f"+{abs(gap):.1f} bps σ")

    st.divider()

    # ── ACT IV ─────────────────────────────────────────────────────────

    st.subheader("Act IV: The GENIUS Act — Does It Solve the Problem?")

    st.markdown(
        """
        The **GENIUS Act** (Guiding and Establishing National Innovation for U.S. Stablecoins) represents 
        the most comprehensive U.S. regulatory framework for stablecoins. It mandates 1:1 reserve backing, 
        prohibits algorithmic stablecoins, and establishes federal oversight.

        **But does it solve the liquidity friction that caused the Coinbase halt?**

        The data suggests: **No. Not fully.**
        """
    )

    st.error(
        r"""
        **Gap 1: No Federal Reserve Master Accounts**
        
        The GENIUS Act does **not** grant non-bank stablecoin issuers automatic access to 
        Federal Reserve master accounts or the discount window.
        
        **Implication:** Issuers must still rely on commercial banks to interface with the payment system. 
        If those commercial banks close on weekends, or if Fedwire (which is not 24/7) is the only 
        settlement rail, the **"weekend gap" persists**.
        """
    )

    fig6 = go.Figure()

    wkday = hourly_spread[~hourly_spread["is_weekend"]]
    wkend = hourly_spread[hourly_spread["is_weekend"]]

    fig6.add_trace(go.Scatter(
        x=wkday["timestamp"], y=wkday["spread"],
        mode="lines", name="Weekday Spread",
        line=dict(color="#4ECDC4", width=1),
    ))
    fig6.add_trace(go.Scatter(
        x=wkend["timestamp"], y=wkend["spread"],
        mode="lines", name="Weekend Spread",
        line=dict(color="#FF6B6B", width=1.5),
    ))
    fig6.add_hline(y=0, line_dash="dash", line_color="white", annotation_text="Parity")
    fig6.add_vrect(
        x0=svb_start, x1=svb_end,
        fillcolor="rgba(255,107,107,0.15)", line_width=0,
        annotation_text="SVB Weekend", annotation_position="top left",
    )
    _chart_layout(fig6, "The Weekend Gap: Onshore/Offshore Spread by Day Type", "Spread (bps)")
    st.plotly_chart(fig6, use_container_width=True)

    st.warning(
        """
        **Gap 2: Coinbase's Dilemma Remains**
        
        Even under the GENIUS Act, an exchange like Coinbase relies on the banking system to clear USD. 
        If a panic occurs on a **Saturday**, and the issuer cannot liquidate Treasuries or move cash 
        via Fedwire until **Monday**, the exchange may still be forced to **pause conversions** to 
        protect its own liquidity.
        
        The GENIUS Act ensures **solvency** (the reserves exist), but it does not ensure 
        **liquidity** (the reserves can be accessed in real-time, 24/7).
        """
    )

    st.markdown("#### Conclusion")
    st.success(
        """
        The March 2023 crisis exposed a fundamental tension in stablecoin design: **the collision 
        between 24/7 crypto markets and 9-to-5 banking infrastructure**.
        
        The GENIUS Act addresses the *what* (reserves must exist) but not the *when* (reserves 
        must be accessible at any hour). Until stablecoin issuers gain direct access to central bank 
        liquidity facilities — or until real-time payment rails like FedNow are fully integrated — 
        the "weekend gap" will remain a systemic vulnerability.
        
        **The next crisis may not wait for Monday.**
        """
    )

