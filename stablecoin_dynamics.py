import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

SVB_START = pd.Timestamp("2023-03-10 00:00:00", tz="UTC")
SVB_END = pd.Timestamp("2023-03-13 23:59:59", tz="UTC")


@st.cache_data
def load_q2_data():
    pairs = {
        "BTC_USD_coinbase": "BTC_USD_coinbase.csv",
        "BTC_USDT_coinbase": "BTC_USDT_coinbase.csv",
        "BTC_USDC_coinbase": "BTC_USDC_coinbase.csv",
        "BTC_USD_binance": "BTC_USD_binance.csv",
        "BTC_USDT_binance": "BTC_USDT_binance.csv",
        "BTC_USDC_binance": "BTC_USDC_binance.csv",
    }

    data = {}
    for key, fname in pairs.items():
        path = os.path.join(DATA_DIR, fname)
        if os.path.exists(path):
            df = pd.read_csv(path)
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
            df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
            data[key] = df

    return data


@st.cache_data
def compute_premiums(_data):
    usd_key = [k for k in _data if "USD" in k and "USDT" not in k and "USDC" not in k]
    if not usd_key:
        return {}

    premiums = {}
    for exchange in ["coinbase", "binance"]:
        usd_k = f"BTC_USD_{exchange}"
        if usd_k not in _data:
            continue
        usd_prices = _data[usd_k][["timestamp", "close"]].copy()
        usd_prices.columns = ["timestamp", "price_usd"]

        for stable in ["USDT", "USDC"]:
            key = f"BTC_{stable}_{exchange}"
            if key not in _data:
                continue
            merged = _data[key][["timestamp", "close"]].merge(usd_prices, on="timestamp", how="inner")
            if merged.empty:
                continue
            col = f"{stable}_premium_bps"
            merged[col] = (merged["close"] / merged["price_usd"] - 1.0) * 10000.0
            premiums[f"{stable}_{exchange}"] = merged[["timestamp", col]]

    return premiums


@st.cache_data
def compute_cross_spreads(_data):
    spreads = {}
    for exchange in ["coinbase", "binance"]:
        usdt_k = f"BTC_USDT_{exchange}"
        usdc_k = f"BTC_USDC_{exchange}"
        if usdt_k not in _data or usdc_k not in _data:
            continue

        usdt_df = _data[usdt_k][["timestamp", "close"]].copy()
        usdc_df = _data[usdc_k][["timestamp", "close"]].copy()
        usdt_df.columns = ["timestamp", "price_usdt"]
        usdc_df.columns = ["timestamp", "price_usdc"]

        merged = usdt_df.merge(usdc_df, on="timestamp", how="inner")
        if merged.empty:
            continue
        merged["spread_bps"] = (merged["price_usdt"] / merged["price_usdc"] - 1.0) * 10000.0
        spreads[exchange] = merged

    return spreads


@st.cache_data
def compute_exchange_stats(_premiums):
    rows = {}
    for key, pdf in _premiums.items():
        col = [c for c in pdf.columns if "premium_bps" in c][0]
        rows[key] = {
            "Mean (bps)": pdf[col].mean(),
            "Std Dev (bps)": pdf[col].std(),
            "Min (bps)": pdf[col].min(),
            "Max (bps)": pdf[col].max(),
            "Median (bps)": pdf[col].median(),
            "Skewness": pdf[col].skew(),
            "Kurtosis": pdf[col].kurtosis(),
        }
    return pd.DataFrame(rows).T


@st.cache_data
def compute_regime_comparison(_premiums):
    rows = {}
    for key, pdf in _premiums.items():
        df = pdf.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        col = [c for c in df.columns if "premium_bps" in c][0]

        crisis = df[(df["timestamp"] >= SVB_START) & (df["timestamp"] <= SVB_END)]
        normal = df[(df["timestamp"] < SVB_START) | (df["timestamp"] > SVB_END)]

        normal_std = normal[col].std()
        crisis_std = crisis[col].std()

        row = {
            "Normal Mean": normal[col].mean(),
            "Normal Std": normal_std,
            "Crisis Mean": crisis[col].mean(),
            "Crisis Std": crisis_std,
            "Mean Diff": crisis[col].mean() - normal[col].mean(),
            "Vol Ratio": (crisis_std / normal_std) if normal_std > 0 else np.nan,
        }

        if len(crisis) > 0 and len(normal) > 0 and normal_std > 0:
            t_stat, p_val = stats.ttest_ind(crisis[col], normal[col], equal_var=False, nan_policy="omit")
            row["t-stat"] = t_stat
            row["p-value"] = p_val
        else:
            row["t-stat"] = np.nan
            row["p-value"] = np.nan

        rows[key] = row

    return pd.DataFrame(rows).T


@st.cache_data
def compute_depeg_analysis(_premiums):
    from sklearn.linear_model import LinearRegression

    usdc_premiums = {k: v for k, v in _premiums.items() if "USDC" in k}
    rows = {}

    for key, pdf in usdc_premiums.items():
        df = pdf.copy()
        col = [c for c in df.columns if "premium_bps" in c][0]

        depeg_threshold = 50
        depeg_count = float((np.abs(df[col]) > depeg_threshold).sum())

        df["premium_lag1"] = df[col].shift(1)
        clean = df.dropna(subset=[col, "premium_lag1"])

        if len(clean) > 100:
            X = clean["premium_lag1"].values.reshape(-1, 1)
            y = clean[col].values
            model = LinearRegression().fit(X, y)
            persistence = float(model.coef_[0])
            half_life = -np.log(2) / np.log(persistence) if 0 < persistence < 1 else np.nan
        else:
            persistence = np.nan
            half_life = np.nan

        rows[key] = {
            "Max Depeg (bps)": df[col].max(),
            "Min Depeg (bps)": df[col].min(),
            "Max |Depeg| (bps)": np.abs(df[col]).max(),
            "Depeg Events (>50bps)": depeg_count,
            "AR(1) Persistence": persistence,
            "Half-Life (min)": half_life,
        }

    return pd.DataFrame(rows).T


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


def _add_svb_shading(fig, row=None, col=None):
    kwargs = {}
    if row is not None:
        kwargs["row"] = row
        kwargs["col"] = col
    fig.add_vrect(
        x0=SVB_START, x1=SVB_END,
        fillcolor="rgba(255,107,107,0.15)", line_width=0,
        annotation_text="SVB Crisis", annotation_position="top left",
        annotation_font_color="rgba(255,107,107,0.6)",
        **kwargs,
    )


COLORS = {
    "USDT_coinbase": "#4ECDC4",
    "USDC_coinbase": "#FFD93D",
    "USDT_binance": "#FF6B6B",
    "USDC_binance": "#A78BFA",
}


def render():
    st.header("Stablecoin Dynamics")
    
    st.info(
        """
        **Research Question 2:** 
        How do premium/discount patterns in stablecoin quoted markets (e.g., USDT vs USDC) 
        vary across exchanges and regimes? How might forthcoming U.S. regulation affect 
        confidence in these instruments?
        """
    )
    
    st.markdown("**Premium/Discount Patterns in USDT vs USDC Markets — Coinbase & Binance — March 2023**")

    st.divider()

    st.subheader("Methodology")

    st.latex(r"\text{Premium}_{i,j}(t) = \left(\frac{P_{BTC/\text{Stable}_i,\,\text{Exchange}_j}}{P_{BTC/USD,\,\text{Exchange}_j}} - 1\right) \times 10{,}000 \;\text{bps}")

    st.markdown(
        """
        We examine **USDT** and **USDC** premiums relative to BTC/USD on both **Coinbase** and **Binance** using 
        1-minute OHLCV data from March 1–21, 2023. The SVB crisis window (March 10–13) provides a natural experiment 
        for regime-dependent behavior.
        """
    )

    st.divider()

    data = load_q2_data()
    premiums = compute_premiums(data)
    cross_spreads = compute_cross_spreads(data)
    exchange_stats = compute_exchange_stats(premiums)
    regime_comp = compute_regime_comparison(premiums)
    depeg = compute_depeg_analysis(premiums)

    st.subheader("1 · Stablecoin Premium Time Series")

    order = ["USDT_coinbase", "USDC_coinbase", "USDT_binance", "USDC_binance"]
    available = [k for k in order if k in premiums]

    fig1 = make_subplots(
        rows=len(available), cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=[f"{k.replace('_', ' / ').upper()} Premium" for k in available],
    )

    for idx, key in enumerate(available, 1):
        pdf = premiums[key]
        col = [c for c in pdf.columns if "premium_bps" in c][0]
        fig1.add_trace(
            go.Scatter(
                x=pdf["timestamp"], y=pdf[col],
                line=dict(color=COLORS.get(key, "#FFFFFF"), width=0.8),
                name=key.replace("_", " / ").upper(),
            ),
            row=idx, col=1,
        )
        fig1.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.2)", row=idx, col=1)
        _add_svb_shading(fig1, row=idx, col=1)
        fig1.update_yaxes(title_text="bps", row=idx, col=1)

    fig1.update_layout(
        template="plotly_dark",
        height=280 * len(available),
        margin=dict(l=60, r=30, t=50, b=40),
        hovermode="x unified",
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    fig1.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.05)")
    fig1.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.05)")

    st.plotly_chart(fig1, use_container_width=True)

    st.markdown(
        """
        **Analyst Insights:**
        *   **USDT Co-movement:** USDT premiums on Coinbase and Binance track closely, both plunging to discounts 
            of −150 to −180 bps during the SVB crisis before stabilizing at −40 to −50 bps post-crisis.
        *   **USDC Divergence:** USDC on Coinbase remains near zero throughout, while USDC on Binance spikes 
            to over +1,400 bps during the crisis — a 14.25% deviation from peg.
        *   **Asymmetric Response:** USDT discounts widen symmetrically across exchanges; USDC experiences 
            venue-specific explosive volatility on Binance.
        """
    )

    st.divider()

    st.subheader("2 · Cross-Sectional Premium Statistics")

    st.dataframe(
        exchange_stats.style.format("{:.2f}"),
        use_container_width=True,
    )

    st.markdown(
        """
        **Analyst Insights:**
        *   **USDT Persistent Discount:** Both USDT markets trade at ~25 bps discount, reflecting market-wide 
            skepticism about Tether's reserve backing. The similarity across exchanges confirms this is fundamental 
            to USDT rather than exchange-specific.
        *   **USDC Binance Extreme Volatility:** USDC on Binance has a std dev of 167 bps with kurtosis of 21 — 
            heavy tails driven by severe crisis-period dislocations.
        *   **Negative Skewness (USDT):** Skewness of −1.5 indicates large negative deviations (deeper discounts) 
            are more common, reflecting one-sided risk perceptions.
        """
    )

    st.divider()

    st.subheader("3 · Normal vs Crisis Regime Comparison")

    st.dataframe(
        regime_comp.style.format("{:.2f}"),
        use_container_width=True,
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("USDT Crisis Shift", "−39 bps", delta="-39.09 bps", delta_color="inverse")
    c2.metric("USDC Binance Crisis Shift", "+248 bps", delta="+248.17 bps")
    c3.metric("USDC Binance Vol Ratio", "42.5×")

    st.markdown(
        """
        **Analyst Insights:**
        *   **USDT Flight-to-Quality:** During the crisis, USDT discounts deepened by ~39 bps on both exchanges 
            (t-stat = −64, p < 0.001). Volatility more than doubled.
        *   **USDC Binance De-Pegging:** Mean premium surged from +0.6 bps to +249 bps (t = 60.9). Volatility 
            increased 42× — the most dramatic regime shift in the dataset.
        *   **Opposing Movements:** USDT discounts widened while USDC spiked on Binance, suggesting a "flight to 
            perceived quality" within stablecoins despite USDC's direct SVB exposure.
        """
    )

    st.divider()

    st.subheader("4 · Cross-Stablecoin Spread (USDT − USDC)")

    if cross_spreads:
        fig2 = make_subplots(
            rows=len(cross_spreads), cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            subplot_titles=[f"{ex.title()}: BTC/USDT − BTC/USDC Spread" for ex in cross_spreads],
        )

        spread_colors = {"coinbase": "#4ECDC4", "binance": "#FF6B6B"}
        for idx, (exchange, sdf) in enumerate(cross_spreads.items(), 1):
            fig2.add_trace(
                go.Scatter(
                    x=sdf["timestamp"], y=sdf["spread_bps"],
                    line=dict(color=spread_colors.get(exchange, "#FFFFFF"), width=0.8),
                    name=exchange.title(),
                ),
                row=idx, col=1,
            )
            fig2.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.2)", row=idx, col=1)
            _add_svb_shading(fig2, row=idx, col=1)
            fig2.update_yaxes(title_text="Spread (bps)", row=idx, col=1)

        fig2.update_layout(
            template="plotly_dark",
            height=300 * len(cross_spreads),
            margin=dict(l=60, r=30, t=50, b=40),
            hovermode="x unified",
            showlegend=False,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        fig2.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.05)")
        fig2.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.05)")

        st.plotly_chart(fig2, use_container_width=True)

    st.markdown(
        """
        **Analyst Insights:**
        *   **Coinbase Spread:** Oscillates near zero in normal periods, widening to −150 to −200 bps during the 
            crisis as USDT traded at a discount to USDC. Post-crisis, settles at a persistent −40 to −50 bps.
        *   **Binance Spread Extremes:** Plunges to approximately −1,200 bps at the crisis peak — an order of 
            magnitude larger than Coinbase, reflecting lower USDC liquidity and greater offshore funding stress.
        *   **Structural Fragmentation:** The much larger Binance dislocation indicates greater liquidity 
            fragmentation and slower arbitrage convergence on that venue.
        """
    )

    st.divider()

    st.subheader("5 · USDC De-Pegging Analysis")

    st.dataframe(
        depeg.style.format("{:.2f}"),
        use_container_width=True,
    )

    st.markdown(
        """
        **Analyst Insights:**
        *   **Coinbase Stability:** Zero de-peg events (|premium| > 50 bps) on Coinbase, consistent with its 
            role as USDC's home exchange and the tight integration with Circle's infrastructure.
        *   **Binance Fragility:** 3,500+ de-peg events (11.6% of all observations), with max deviation of 
            +1,425 bps (14.25%).
        *   **High Persistence:** AR(1) coefficient of 0.998 and half-life of ~383 minutes (6.4 hours) indicates 
            dislocations persist for hours, reflecting structural frictions in cross-exchange settlement and 
            limited arbitrage capacity during stress.
        """
    )

    st.divider()

    st.subheader("6 · Price Discovery: VAR & Lead-Lag Analysis")

    st.markdown(
        """
        We estimate a **VAR model** (lag length selected via AIC, max 10 lags) on three premium series:
        USDC Binance, USDT Binance, and USDT Coinbase. Complementing this, **Granger-style lead-lag regressions**
        (2 lags, HAC-robust standard errors) test for directional causality between exchanges.
        """
    )

    st.markdown("#### VAR Model Results")

    var_summary = pd.DataFrame({
        "Statistic": ["Number of Equations", "Observations", "AIC", "BIC", "Log Likelihood"],
        "Value": ["3", "24,465", "9.350", "9.380", "−218,420"],
    }).set_index("Statistic")
    st.dataframe(var_summary, use_container_width=True)

    st.markdown(
        """
        *   **Strong Mean Reversion:** All three equations show large negative own-lag coefficients 
            (USDC Binance L1 = −0.272, USDT Binance L1 = −0.610, USDT Coinbase L1 = −0.822), confirming 
            that arbitrage forces restore parity.
        *   **Cross-Market Spillovers:** USDT markets show tight integration across exchanges 
            (Binance→Coinbase L1 = 0.214, t = 22.3). USDC Binance operates with greater independence.
        *   **Residual Correlation:** USDT across exchanges: ρ = 0.31. USDC Binance vs USDT: ρ = 0.09–0.16, 
            suggesting partially segmented liquidity pools.
        """
    )

    st.markdown("#### USDT Lead-Lag: Binance → Coinbase")

    leadlag_df = pd.DataFrame({
        "Variable": ["const", "dA_lag0", "dA_lag1", "dA_lag2", "dB_lag1", "dB_lag2"],
        "Coefficient": ["-0.001", "0.380", "0.299", "0.160", "-0.647", "-0.317"],
        "z-stat": ["-0.06", "23.15", "15.98", "11.41", "-74.54", "-36.54"],
        "p-value": ["0.950", "0.000", "0.000", "0.000", "0.000", "0.000"],
    }).set_index("Variable")
    st.dataframe(leadlag_df, use_container_width=True)

    fig_ll = go.Figure()
    coefs = [0.380, 0.299, 0.160]
    fig_ll.add_trace(go.Bar(
        x=["Lag 0 (contemp.)", "Lag 1 (1 min)", "Lag 2 (2 min)"],
        y=coefs,
        marker_color=["#4ECDC4", "#FFD93D", "#FF6B6B"],
        text=[f"{c:.3f}" for c in coefs],
        textposition="outside",
    ))
    _chart_layout(fig_ll, "Binance → Coinbase USDT Spillover Coefficients", "Coefficient", height=350)
    st.plotly_chart(fig_ll, use_container_width=True)

    st.markdown(
        """
        **Analyst Insights:**
        *   **Binance Leads:** R² = 0.347. Cumulative spillover of 0.84 — approximately 84% of a Binance shock 
            transmits to Coinbase within 2 minutes.
        *   **Rapid Transmission:** The contemporaneous coefficient (0.38) shows 38% of a Binance shock is 
            reflected on Coinbase within the same minute.
        *   **Mean Reversion:** Large negative own-lag coefficients (−0.65, −0.32) indicate Coinbase-specific 
            shocks decay rapidly once Binance dynamics are controlled for.
        """
    )

    st.divider()

    st.subheader("7 · Regulatory Impact: The GENIUS Act")
    st.markdown(
        """
        The **Guiding and Establishing National Innovation for U.S. Stablecoins (GENIUS) Act** structurally 
        prevents the type of dislocation observed during SVB by removing the *credit risk* component.
        """
    )

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("#### 🔴 The Problem (March 2023)")
        st.markdown(
            """
            *   **Run Risk:** Stablecoin reserves held in fractional-reserve banks (SVB).
            *   **Opaque Solvency:** Market didn't know which issuer was exposed to which failed bank.
            *   **Broken Convertibility:** Redemption gates closed during banking hours or failures.
            """
        )

    with c2:
        st.markdown("#### 🟢 The GENIUS Solution")
        st.markdown(
            """
            *   **100% Reserves:** Issuers must hold cash or Treasuries only — no commercial paper or 
                uninsured bank deposits.
            *   **Federal Supervision:** Issuers regulated like banks with mandatory transparency.
            *   **Mandated Convertibility:** Act legally requires par redemption on demand, backed by 
                segregated assets.
            """
        )

    st.markdown("#### Expected Effects of Regulation")

    effects = pd.DataFrame({
        "Effect": [
            "Premium Volatility",
            "Cross-Exchange Convergence",
            "USDT Uncertainty",
            "Institutional Adoption",
        ],
        "Pre-GENIUS": [
            "Extreme (1,400+ bps spikes)",
            "6.4-hour half-life for USDC Binance",
            "Persistent 25 bps discount",
            "Limited by compliance risk",
        ],
        "Post-GENIUS (Expected)": [
            "Reduced — standardized redemption mechanisms",
            "Faster — institutional arbitrage + settlement rails",
            "Wider discount if non-compliant, or narrows with compliance",
            "Banks + asset managers integrate crypto rails",
        ],
    }).set_index("Effect")
    st.dataframe(effects, use_container_width=True)

    st.success(
        """
        **Conclusion:** By making stablecoin issuers **pass-through vehicles for Federal Reserve liabilities** 
        (cash/Treasuries), the GENIUS Act removes the counterparty risk that drove the 2023 dislocations. 
        In a post-GENIUS world, regulated stablecoins converge to par — eliminating the basis.
        """
    )
