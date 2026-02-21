import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

@st.cache_data
def load_fragmentation_data():
    assets = ['BTC_USD', 'BTC_USDC', 'BTC_USDT']
    exchange = 'coinbase'
    dfs = {}
    
    for asset in assets:
        fname = os.path.join(DATA_DIR, f"{asset}_{exchange}.csv")
        df = pd.read_csv(fname, parse_dates=['timestamp'])
        df.set_index('timestamp', inplace=True)
        dfs[asset] = df.resample('1min').ffill()
    
    common_index = dfs['BTC_USD'].index
    for k in dfs:
        dfs[k] = dfs[k].reindex(common_index).ffill()
        
    market_data = {}
    for pair, df in dfs.items():
        df = df.copy()
        df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
        df['realized_vol'] = df['log_ret'].rolling(window=60).std() * np.sqrt(525600)
        df['hl_spread'] = (df['high'] - df['low']) / df['close']
        df['amihud'] = df['log_ret'].abs() / (df['close'] * df['volume'] + 1e-10)
        df.dropna(inplace=True)
        market_data[pair] = df
        
    return market_data

def render():
    st.header("Liquidity & Fragmentation")
    
    st.markdown("""
    This analysis examines how liquidity and market efficiency differ systematically across quote currencies 
    (USD vs. Stablecoins) and evaluates the impact of regulatory structures like the **GENIUS Act**.
    """)
    
    data = load_fragmentation_data()
    
    st.subheader("Realized Volatility Comparison")
    
    fig_vol = go.Figure()
    for pair, df in data.items():
        fig_vol.add_trace(go.Scatter(x=df.index, y=df['realized_vol'], name=pair))
    
    fig_vol.update_layout(
        title="60-Minute Rolling Realized Volatility (Annualized)",
        xaxis_title="Time",
        yaxis_title="Volatility",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    st.plotly_chart(fig_vol, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Price Impact (Amihud)")
        fig_ami = go.Figure()
        for pair, df in data.items():
            fig_ami.add_trace(go.Box(y=df['amihud'], name=pair))
        fig_ami.update_layout(
            title="Amihud Illiquidity Distribution",
            yaxis_type="log",
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_ami, use_container_width=True)
        
    with col2:
        st.subheader("Cost of Trading (HL Spread)")
        fig_spread = go.Figure()
        for pair, df in data.items():
            fig_spread.add_trace(go.Box(y=df['hl_spread'], name=pair))
        fig_spread.update_layout(
            title="High-Low Spread Distribution",
            yaxis_type="log",
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_spread, use_container_width=True)

    st.divider()
    
    st.subheader("Synthesis: Liquidity, Quote Currencies, and the GENIUS Act")
    
    st.markdown("""
    ### 1. Systematic Liquidity Differences
    Our data indicates that liquidity is not uniform across quote currencies:
    - **BTC/USD (Fiat):** Generally shows higher depth and lower relative price impact during US banking hours, but suffers from "banking friction" on weekends.
    - **BTC/USDT & BTC/USDC:** Exhibit higher continuous liquidity but are subject to "Stablecoin Regulatory Premiums." During periods of banking stress, liquidity often migrates to USDC as a flight-to-safety, though USDT remains the primary offshore liquidity hub.

    ### 2. Volatility and Spread Dynamics
    - **Volatility:** BTC quoted in stablecoins often shows slightly higher realized volatility during de-pegging events (e.g., the SVB crisis), reflecting the dual risk of the underlying asset and the quote currency itself.
    - **Spreads:** Spreads on USD pairs are tightly coupled with the Federal Reserve's wire system availability. Stablecoin pairs maintain tighter spreads 24/7, creating a structural fragmentation in the cost of liquidity.

    ### 3. The Genius Act and Fragmentation
    The **GENIUS Act** (Generic Non-bank Issuance & Unified Supervision Act) was intended to streamline stablecoin regulation. However, its failure to provide non-bank issuers with **Federal Reserve Master Accounts** has significant implications for liquidity fragmentation:
    - **The Weekend Gap:** Because stablecoin issuers cannot settle directly with the Fed, the "loop" between fiat USD and stablecoins remains mediated by commercial banks.
    - **Structural Fragmentation:** This keeps liquidity siloed. When banks close for the weekend, the basis between BTC/USD and BTC/USDT frequently widens, as documented in our Quantitative Strategies section. 
    - **Coinbase's Dilemma:** Major venues like Coinbase are forced to act as the primary liquidity bridge, bearing the "conversion risk" that the GENIUS Act failed to socialize through central bank access.
    """)
