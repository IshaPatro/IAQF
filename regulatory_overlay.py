import streamlit as st
import pandas as pd


def render():
    st.header("Regulatory Overlay")
    
    st.info(
        """
        **Research Question 4:** 
        Tie your empirical findings to the broader policy context:
        *   Why might regulated stablecoins alter cross-currency trading patterns?
        *   What implications does the GENIUS Act (and stablecoin settlement adoption by 
            payment systems) have for the structure and efficiency of these markets?
        """
    )

    st.markdown("""
    **Index:**
    1. [Part I — Cross-Currency Basis as Solvency Barometer](#a-the-cross-currency-basis-as-a-solvency-barometer)
    2. [Rail Segmentation: Issuer vs. Venue Fragmentation](#b-rail-segmentation-issuer-vs-venue-fragmentation)
    3. [Cross-Stablecoin Spread & Offshore Fragmentation](#c-the-cross-stablecoin-spread-structural-offshore-fragmentation)
    4. [Bitcoin as Shadow Exchange Rate (SER)](#d-bitcoin-as-shadow-exchange-rate-ser)
    5. [Price Discovery Leadership — VAR & Lead-Lag](#e-price-discovery-leadership-var-lead-lag-analysis)
    6. [Part II — Regulatory Compliance vs. Privacy: ZKP Solutions](#a-regulatory-compliance-vs-privacy-zkp-solutions)
    7. [100% Reserve Mandate](#b-eliminating-the-run-risk-basis-the-100-reserve-mandate)
    8. [The Regulatory Premium](#c-the-regulatory-premium-a-new-structural-fragmentation)
    9. [The Weekend Gap — Master Account Problem](#d-the-unresolved-weekend-gap-the-master-account-problem)
    10. [Liquidity Structure Across Quote Currencies](#e-liquidity-structure-across-quote-currencies)
    11. [Zero-Knowledge Proofs](#f-zero-knowledge-proofs-resolving-the-compliance-privacy-tension)
    12. [Synthesis & Conclusion](#synthesis-conclusion)
    """)

    st.divider()

    st.markdown("## Part I: Why Regulated Stablecoins Alter Cross-Currency Trading Patterns")

    st.divider()

    st.header("A. The Cross-Currency Basis as a Solvency Barometer")

    st.markdown("#### Equilibrium Condition & Basis Definition")

    st.latex(r"P(\text{BTC/USD}) = P(\text{BTC/USDT}) \times P(\text{USDT/USD})")
    st.latex(r"\text{Basis}(t) = \ln\!\big[P(\text{BTC/USD}, t)\big] - \ln\!\big[P(\text{BTC/USDT}, t)\big]")

    st.markdown("Under normal conditions the basis hovers near zero. During the SVB crisis window (March 10–13), it did not.")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Peak Basis", "192 bps")
    with col2:
        st.metric("Round-Trip Arb Cost", "~20 bps")
    with col3:
        st.metric("Net Profit Unexploited", "172 bps")

    st.warning("""
    **Solvency Crisis, Not Pricing Inefficiency**

    Classical arbitrage theory predicts the 172 bps gap would close instantly. It did not. Arbitrageurs
    refused to bridge the basis because they could not trust fiat USD withdrawals through SVB/Signature
    would be honored. The basis functioned as a **real-time, market-implied measure of banking system
    credibility** — something it was never designed to do.
    """)

    st.markdown("#### Empirical Signatures")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.info("""
        **Decoupling Event**

        BTC/USD and BTC/USDT decoupled visibly Mar 10–13. The fiat rail traded at a **discount** as
        fears of USD inaccessibility mounted.
        """)

    with col2:
        st.info("""
        **Volatility Regime Shift**

        Basis volatility exploded **~10×** during the crisis. Decayed rapidly on Mar 13 following the
        Fed/Treasury deposit guarantee — confirming the **government backstop**, not arbitrage, restored stability.
        """)

    with col3:
        st.info("""
        **Rail Dominance**

        BTC/USDT volume dwarfed BTC/USD during the crisis. Stablecoin rails — not fiat rails — carried
        market liquidity under stress. Fiat on-ramps were effectively **closed or too risky** for large flows.
        """)

    st.divider()

    st.header("B. Rail Segmentation: Issuer vs. Venue Fragmentation")

    st.markdown("""
    Premium data reveals fragmentation on **two axes simultaneously** — *which stablecoin* and *which exchange*.
    """)

    st.latex(r"\text{Premium}_{i,j}(t) = \left[\frac{P(\text{BTC/Stable}_i,\; \text{Exchange}_j)}{P(\text{BTC/USD},\; \text{Exchange}_j)} - 1\right] \times 10{,}000 \;\text{bps}")

    st.subheader("USDT — Persistent Structural Discount")

    st.markdown("""
    USDT maintained a **~25 bps discount** to USD on both venues throughout the sample — a fundamental,
    issuer-level discount reflecting chronic skepticism about Tether's reserve opacity. During the SVB crisis,
    discounts deepened by ~39 bps (t = −64, p < 0.001), volatility more than doubled, and negative skewness
    of −1.5 confirms large discounts are **structurally more probable** than equivalent positive deviations.
    """)

    usdt_data = {
        "Metric": ["Mean (bps)", "Std Dev (bps)", "Min (bps)", "Max (bps)", "Skewness", "Kurtosis",
                    "Crisis Mean Shift", "Crisis Vol Ratio"],
        "USDT Coinbase": ["−24.71", "29.41", "−190.39", "31.76", "−1.52", "2.64", "−39.09 bps", "2.38×"],
        "USDT Binance": ["−20.11", "25.94", "−163.17", "30.71", "−1.68", "3.35", "−35.87 bps", "2.40×"]
    }
    st.dataframe(pd.DataFrame(usdt_data).set_index("Metric"), use_container_width=True)

    st.subheader("USDC — Venue-Dependent Fragmentation")

    col1, col2 = st.columns(2)

    with col1:
        st.success("""
        **Coinbase (Home Exchange)**

        - **Zero** de-peg events (|premium| > 50 bps)
        - Tight Circle infrastructure integration
        - Crisis left **no mark** on USDC pricing
        """)

    with col2:
        st.error("""
        **Binance (Offshore)**

        - Mean premium surged +0.6 → **+256.67 bps** (t = 62.18)
        - Max deviation: **+1,430 bps** (14.25%)
        - 3,748 de-peg events (11.6% of observations)
        - AR(1) = 0.998 → half-life **~383 min (6.4 hrs)**
        - Volatility ratio: **36.01×**
        """)

    usdc_data = {
        "Metric": ["Mean (bps)", "Std Dev (bps)", "Max |Depeg| (bps)", "Depeg Events (>50 bps)",
                    "AR(1) Persistence", "Half-Life (min)", "Crisis Mean Shift", "Crisis Vol Ratio"],
        "USDC Coinbase": ["0.00", "0.00", "0.00", "0", "N/A", "N/A", "0.00 bps", "N/A"],
        "USDC Binance": ["51.19", "168.14", "1,430.75", "3,748", "0.998", "~383 (6.4 hrs)", "+253.83 bps", "36.01×"]
    }
    st.dataframe(pd.DataFrame(usdc_data).set_index("Metric"), use_container_width=True)

    st.markdown("#### The Paradox: Better Reserves, Worse Offshore Performance")

    st.warning("""
    USDC had the fundamentally **superior reserve design** (USD + Treasuries, fully audited) yet suffered the
    larger offshore dislocation. USDT — despite chronic reserve opacity — saw a **"flight to perceived quality"**
    as its risks appeared *uncorrelated* with the banking crisis.

    **Insight for regulatory design:** Transparency mandates alone, without settlement certainty, may not prevent
    acute dislocations. The market rewards *opacity* when known exposures are the source of stress.
    """)

    st.divider()

    st.header("C. The Cross-Stablecoin Spread & Structural Offshore Fragmentation")

    st.markdown("The **USDT − USDC** spread provides the clearest signature of onshore/offshore segmentation.")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Coinbase (Crisis Peak)", "−150 to −200 bps", help="Settled at −40 to −50 bps post-crisis")
    with col2:
        st.metric("Binance (Crisis Peak)", "−1,200 bps", help="Order of magnitude larger than Coinbase")

    st.error("""
    **Order-of-Magnitude Divergence**

    The −1,200 bps Binance spread vs. −200 bps on Coinbase is not a marginal difference. It reflects
    **categorically weaker** USDC liquidity depth and slower arbitrage convergence offshore — confirming
    the two markets were **structurally segmented**, not merely temporarily dislocated.
    """)

    st.divider()

    st.header("D. Bitcoin as Shadow Exchange Rate (SER)")

    st.markdown("""
    BTC trades 24/7, so when fiat rails are impaired, the BTC price ratio functions as a **Shadow Exchange Rate**
    — pricing stablecoin premiums/discounts faster than spot stablecoin markets can update.
    """)

    st.latex(r"\hat{P}(\text{USDT}) = \frac{P(\text{BTC/USD})}{P(\text{BTC/USDT})}")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Peak Implied USDT via SER", "$1.0182", help="March 11, 2023")
    with col2:
        st.metric("Interpretation", "USDT > USD", help="BTC channel priced USDT above par before spot markets adjusted")

    st.info("""
    **Lead-Lag Confirmation**

    The SER reached $1.0182 on March 11 — pricing USDT above USD *before* the spot stablecoin market fully
    reflected the USDC de-peg. Crypto liquidity pools are now the **primary venue for price discovery** during
    TradFi disruptions. Convergence of SER and spot post-crisis validates arbitrage efficiency once the
    government backstop removed solvency uncertainty.
    """)

    st.markdown("""
    **Directional Logic:** If P(BTC/USDT) < P(BTC/USD), then USDT is worth *more* than USD. In March 2023, USDT
    traded at a 1.01–1.03 premium through this channel while USDC traded at a discount — **flipping the
    expectation** that the regulated, better-backed stablecoin would command a premium.
    """)

    st.divider()

    st.header("E. Price Discovery Leadership — VAR & Lead-Lag Analysis")

    st.markdown("VAR model (AIC lag selection, max 10) estimated on **24,465 observations** across three premium series.")

    var_data = {
        "Statistic": ["Equations", "Observations", "AIC", "BIC", "Log Likelihood"],
        "Value": ["3", "24,465", "9.350", "9.380", "−218,420"]
    }
    st.dataframe(pd.DataFrame(var_data).set_index("Statistic"), use_container_width=True)

    st.subheader("Finding 1: Strong Mean Reversion")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("USDC Binance (L1)", "−0.272")
    with col2:
        st.metric("USDT Binance (L1)", "−0.610")
    with col3:
        st.metric("USDT Coinbase (L1)", "−0.822")

    st.markdown("Large negative own-lag coefficients confirm arbitrage forces **eventually restore parity** — dislocations are not permanent.")

    st.subheader("Finding 2: Binance Leads Coinbase")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Cumulative Spillover (2 min)", "0.84", help="84% of Binance shock transmits to Coinbase")
    with col2:
        st.metric("Contemporaneous (Same min)", "0.38", help="38% within the same minute")

    st.info("**Binance (offshore) is the price discovery leader** during stress; Coinbase (onshore) follows.")

    st.subheader("Finding 3: Segmented Liquidity Pools")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("USDT Cross-Exchange (ρ)", "0.31", help="Moderate within-stablecoin integration")
    with col2:
        st.metric("USDC vs USDT (ρ)", "0.09 – 0.16", help="Partially segmented cross-stablecoin pools")

    st.markdown("Low cross-stablecoin residual correlation confirms **partially segmented liquidity pools** rather than a unified market.")

    st.subheader("Lead-Lag Regression Coefficients")

    leadlag_data = {
        "Variable": ["const", "dA_lag0 (Binance contemp.)", "dA_lag1 (Binance −1 min)",
                      "dA_lag2 (Binance −2 min)", "dB_lag1 (Coinbase own −1)",
                      "dB_lag2 (Coinbase own −2)", "R²", "Cumulative Spillover"],
        "Coefficient": ["−0.001", "0.380", "0.299", "0.160", "−0.647", "−0.317", "0.347", "0.84"],
        "z-stat": ["−0.06", "23.15", "15.98", "11.41", "−74.54", "−36.54", "—", "—"],
        "p-value": ["0.950", "0.000", "0.000", "0.000", "0.000", "0.000", "—", "—"]
    }
    st.dataframe(pd.DataFrame(leadlag_data).set_index("Variable"), use_container_width=True)

    st.divider()

    st.markdown("## Part II: Implications of the GENIUS Act for Market Structure & Efficiency")

    st.divider()

    st.header("A. Regulatory Compliance vs. Privacy: ZKP Solutions")

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

    st.divider()

    st.header("B. Eliminating the Run Risk Basis — The 100% Reserve Mandate")

    st.markdown("""
    The GENIUS Act's core intervention: issuers must hold **only cash or short-term US Treasuries**.
    Commercial paper, uninsured deposits, and endogenous collateral are explicitly prohibited.
    This targets the credit risk that drove the 192 bps basis blowout.
    """)

    st.latex(r"P(\text{BTC/USD}) \equiv P(\text{BTC/USDT}) \;\;\text{because}\;\; \text{USDT} \equiv \text{USD}")
    st.latex(r"\Rightarrow \;\text{Basis} \to 0 \;\;\text{structurally, not just empirically}")

    st.info("""
    **Mechanism:** The counterparty risk that prevented arbitrageurs from closing the 172 bps net profit
    in March 2023 is neutralized. With segregated, government-backed reserves and mandatory par redemption,
    the **arbitrage corridor is legally and structurally reopened**.
    """)

    effect_data = {
        "Effect": ["Premium Volatility", "Cross-Exchange Convergence", "USDT Uncertainty Discount",
                    "Institutional Adoption", "Offshore/Onshore Basis"],
        "Pre-GENIUS (Observed)": ["Up to 1,430 bps spikes", "6.4-hr half-life (USDC Binance)",
                                   "Persistent ~25 bps discount", "Limited by compliance risk",
                                   "Binance leads Coinbase by 84% in 2 min"],
        "Post-GENIUS (Expected)": ["Reduced — standardized redemption", "Faster — institutional arb + rails",
                                    "Widens if non-compliant; narrows if compliant",
                                    "Banks + asset managers integrate compliant rails",
                                    "Convergence narrows as onshore liquidity deepens"]
    }
    st.dataframe(pd.DataFrame(effect_data).set_index("Effect"), use_container_width=True)

    st.divider()

    st.header("C. The Regulatory Premium — A New Structural Fragmentation")

    st.markdown("""
    The GENIUS Act **does not eliminate** fragmentation — it **restructures** it. By subjecting onshore issuers
    to BSA compliance while offshore venues remain outside US jurisdiction, the Act creates a **Regulatory Premium**
    — structurally analogous to Korea's Kimchi Premium.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.warning("""
        **USDT Binary Outcome**

        - **Widens** materially if Tether remains non-compliant
        and is excluded from US institutional flows
        - **Narrows** if Tether voluntarily seeks compliance
        to access onshore capital
        """)

    with col2:
        st.warning("""
        **Bifurcated Architecture Preview**

        The USDC Coinbase vs. Binance split (0 vs. 3,748
        de-peg events) is the clearest preview of post-GENIUS
        bifurcation at scale
        """)

    st.markdown("""
    **Post-GENIUS market structure:** Onshore compliant stablecoins (BSA-integrated, FedNow/Visa rails) will trade
    in a **fundamentally different liquidity regime** from offshore censorship-resistant alternatives. Institutional
    capital flows strictly into compliant vehicles; retail and privacy-seeking capital accepts higher friction for
    offshore access.
    """)

    st.divider()

    st.header("D. The Unresolved Weekend Gap — The Master Account Problem")

    st.error("""
    **Critical Architectural Gap**

    The GENIUS Act does **not** grant non-bank stablecoin issuers direct access to Federal Reserve Master Accounts.
    Fiat settlement remains mediated by commercial banks — tethering the fiat-to-crypto loop to banking hours.
    """)

    st.markdown("""
    - **BTC/USD** shows higher liquidity during banking hours; spreads widen when wire systems close
    - **BTC/USDT & BTC/USDC** maintain tighter spreads 24/7, creating **structural cost-of-liquidity fragmentation**
    - When banks close for the weekend, the BTC/USD vs. BTC/USDT basis **predictably widens**
    - Exchanges like Coinbase are forced to absorb **conversion risk** as the primary onshore liquidity bridge
    """)

    st.info("""
    **Resolution Path:** Requires either **FedNow integration** or direct Fed account access for licensed issuers
    to close this gap in a second-generation regulatory framework.
    """)

    st.divider()

    st.header("E. Liquidity Structure Across Quote Currencies")

    st.markdown("Realized volatility, Amihud illiquidity, and HL spread analyses reveal systematic differences across quote currencies.")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.info("""
        **BTC/USD (Fiat)**

        Higher depth + lower price impact during
        banking hours. Suffers **weekend friction**
        when wire systems close. HL spreads tighten
        during banking hours, widen outside.
        """)

    with col2:
        st.info("""
        **BTC/USDT (Offshore)**

        Higher continuous 24/7 liquidity. Subject
        to persistent Tether discount + **amplified
        tail volatility** during de-pegs. Amihud
        tails heavier than BTC/USD.
        """)

    with col3:
        st.info("""
        **BTC/USDC (Regulated)**

        Near-identical to BTC/USD on Coinbase
        (zero de-pegs). **Acutely fragile** offshore.
        Realized vol spiked sharpest ~Mar 14
        (delayed USDC depeg transmission).
        """)

    st.markdown("""
    **Policy implication:** USD pair spreads are coupled to Fed wire availability; stablecoin pairs maintain
    tighter 24/7 spreads at the cost of intermittent de-pegging risk. Post-GENIUS, compliant stablecoins narrow
    this fragmentation — **only if the Master Account gap is closed**.
    """)

    st.divider()



    st.divider()

    st.header("Synthesis & Conclusion")

    st.markdown("#### What March 2023 Demonstrated")

    st.markdown("""
    1. The cross-currency basis can function as a **real-time banking solvency indicator**
    2. Stablecoin fragmentation operates on **both issuer and venue axes** — offshore exchanges amplify dislocations by an order of magnitude
    3. Bitcoin markets (24/7, highly liquid) now **lead price discovery** when fiat rails fail
    4. The **government backstop** — not market arbitrage — restores parity when solvency fears dominate
    """)

    st.markdown("#### What the GENIUS Act Solves")

    st.success("""
    The 100% reserve mandate, mandatory par convertibility, and federal supervision **directly address** the
    Type II failure. In a post-GENIUS world, the structural basis approaches zero because USDT ≡ USD by
    legal and reserve design. The credit risk that drove 192 bps basis blowout and 1,430 bps USDC Binance
    de-peg is neutralized.
    """)

    st.markdown("#### What Remains Unresolved")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.error("""
        **Weekend Gap**

        Fiat on-ramps tethered to banking hours.
        Exchanges bear conversion risk as the
        onshore liquidity bridge.
        """)

    with col2:
        st.error("""
        **Regulatory Bifurcation**

        Compliant onshore vs. censorship-resistant
        offshore trade in structurally different
        liquidity regimes.
        """)

    with col3:
        st.error("""
        **Compliance-Privacy Tension**

        Without ZKPs, AML/BSA compliance risks
        routing offshore liquidity permanently
        *around* US markets.
        """)

    st.info("""
    **Final Verdict:** The GENIUS Act solves the Type II failure with precision. It does not — and cannot,
    as currently structured — solve the weekend gap, regulatory bifurcation, or compliance-privacy tension.
    **ZKP-based Compliance-by-Design** is the architectural complement required to make the framework globally
    coherent and prevent the onshore market from becoming a compliance island rather than a liquidity hub.
    """)
