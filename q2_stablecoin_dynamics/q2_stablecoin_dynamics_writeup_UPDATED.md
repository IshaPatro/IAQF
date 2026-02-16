# Question 2: Stablecoin Dynamics

## 2.1 Introduction and Motivation

The pricing relationships between stablecoin-quoted markets provide critical insights into market confidence, liquidity fragmentation, and funding stress in cryptocurrency markets. While stablecoins are designed to maintain parity with the U.S. dollar, deviations from this peg—whether persistent discounts or temporary spikes—reveal underlying frictions in redemption mechanisms, counterparty risk perceptions, and exchange-specific liquidity conditions.

Our analysis focuses on two primary stablecoins: USDT (Tether) and USDC (USD Coin), examining their premium/discount patterns across Binance and Coinbase during March 1-21, 2023. This period is particularly informative as it encompasses the Silicon Valley Bank (SVB) crisis (March 10-13, 2023), which triggered a temporary de-pegging of USDC due to Circle's disclosed exposure to SVB. This event provides a natural experiment for understanding how regulatory uncertainty and counterparty risk manifest in cross-currency cryptocurrency markets.

## 2.2 Methodology

### 2.2.1 Premium Construction

For each stablecoin-exchange pair, we construct a premium measure in basis points relative to a common reference. We designate USDC on Coinbase as our reference price (effectively setting its premium to zero), as Coinbase represents the most regulated U.S. exchange and USDC maintains the tightest peg during normal market conditions.

The premium for asset *i* on exchange *j* is defined as:

Premium\_{i,j} = (P\_{i,j} / P\_{ref} - 1) × 10,000

where P\_{ref} is the BTC/USDC price on Coinbase (our reference), and P\_{i,j} is the corresponding BTc price quoted in stablecoin *i* on exchange *j*.

### 2.2.2 Regime Classification

We classify observations into two regimes based on the SVB crisis timeline:

* **Normal Regime**: March 1-9, 2023 and March 14-21, 2023 (pre- and post-crisis)
* **Crisis Regime**: March 10-13, 2023 (SVB failure and USDC de-pegging period)

This classification allows us to examine how stablecoin dynamics differ under stress versus normal market conditions.

### 2.2.3 Statistical Framework

We employ several complementary approaches:

1. **Descriptive Statistics**: Mean, standard deviation, skewness, and kurtosis of premium distributions across exchanges and regimes
2. **Regime Comparison**: Two-sample t-tests comparing premium levels and volatility ratios between normal and crisis periods
3. **De-pegging Analysis**: Identification of extreme deviations (|premium| > 50 bps), persistence measurement via autoregressive models, and half-life calculations
4. **Vector Autoregression (VAR): VAR model with lag length selected via AIC (maximum lag considered: 10), and we report results using a parsimonious specification consistent with short-horizon dynamics.**
5. **Lead–Lag Relationships: Granger-style lead–lag regressions with 2 lags (minute-level horizon), using HAC-robust standard errors.**

## 2.3 Empirical Results

### 2.3.1 Cross-Sectional Premium Patterns

**Table 1: Summary Statistics of Stablecoin Premiums by Exchange**

|Exchange-Pair|Mean (bps)|Std Dev (bps)|Min (bps)|Max (bps)|Median (bps)|Skewness|Kurtosis|
|-|-|-|-|-|-|-|-|
|USDT\_Coinbase|-24.71|29.41|-190.39|31.76|-23.29|-1.52|2.64|
|USDC\_Coinbase|0.00|0.00|0.00|0.00|0.00|0.00|0.00|
|USDT\_Binance|-23.65|29.15|-179.02|37.15|-21.13|-1.56|2.73|
|USDC\_Binance|48.28|167.27|-94.10|1424.91|1.04|4.40|21.02|

Several striking patterns emerge from Table 1 (visualized in Figure 2.1):

**USDT Persistent Discount**: Both USDT markets trade at substantial discounts averaging 24-25 basis points below the reference USDC/Coinbase price. This persistent negative premium reflects market-wide skepticism about Tether's reserve backing and redemption mechanisms. The similarity of USDT premiums across Binance (-23.65 bps) and Coinbase (-24.71 bps) suggests this discount is fundamental to USDT rather than exchange-specific.

**USDC Binance Extreme Volatility**: While USDC on Coinbase serves as our reference (zero premium by construction), USDC on Binance exhibits extraordinary volatility (std dev = 167.27 bps) with extreme positive skewness (4.40) and kurtosis (21.02). The maximum observed premium of 1,424 basis points (a 14.24% deviation from peg) indicates severe temporary dislocations.

**Negative Skewness in USDT**: Both USDT series display negative skewness (-1.52 and -1.56), indicating that large negative deviations (deeper discounts) are more common than large positive deviations. This asymmetry suggests one-sided risk perceptions regarding USDT.

### 2.3.2 Crisis vs. Normal Regime Dynamics

**Table 2: Regime Comparison (Normal vs. Crisis Period)**

|Exchange-Pair|Normal Mean|Normal Std|Crisis Mean|Crisis Std|Mean Diff|Vol Ratio|t-stat|p-value|
|-|-|-|-|-|-|-|-|-|
|USDT\_Coinbase|-16.70|17.74|-55.79|42.26|-39.09|2.38|-64.08|0.000|
|USDC\_Coinbase|0.00|0.00|0.00|0.00|0.00|-|-|-|
|USDT\_Binance|-15.71|17.63|-57.01|41.67|-41.31|2.36|-73.65|0.000|
|USDC\_Binance|0.58|7.28|248.76|309.23|248.17|42.47|60.88|0.000|

Table 2 reveals dramatic regime-dependent behavior:

**USDT Flight-to-Quality**: During the crisis, USDT discounts widened substantially on both exchanges. On Coinbase, the mean discount deepened from -16.70 bps (normal) to -55.79 bps (crisis)—a statistically significant shift of -39.09 bps (t = -64.08, p < 0.001). Volatility more than doubled (ratio = 2.38), indicating heightened uncertainty about USDT's stability. The pattern is nearly identical on Binance (mean shift = -41.31 bps, vol ratio = 2.36), confirming this is a systemic USDT phenomenon rather than exchange-specific.

**USDC Binance De-Pegging**: The most dramatic finding concerns USDC on Binance. During normal periods, USDC/Binance traded near parity (mean = 0.58 bps), but during the crisis, it surged to a mean premium of +248.76 bps—a shift of 248.17 bps (t = 60.88, p < 0.001). Volatility increased by a factor of 42.47, from 7.28 bps to 309.23 bps. This likely reflects temporary liquidity imbalances and order book dislocations on Binance during the SVB crisis.

**Interpretation**: The opposing movements—USDT discounts widening while USDC premiums spiked on Binance—suggest a "flight to perceived quality" within the stablecoin universe. Despite USDC's direct exposure to SVB, market participants on Binance apparently viewed it as preferable to USDT during the crisis, possibly due to Circle's transparency about reserves and regulatory relationships. Conversely, USDT's opacity and historical controversies made it less attractive under stress.

### 2.3.3 De-Pegging Events and Persistence

**Table 3: De-Pegging Analysis**

|Metric|USDC\_Coinbase|USDC\_Binance|
|-|-|-|
|Max De-peg (bps)|0.00|1,424.91|
|Min De-peg (bps)|0.00|-94.10|
|Max Absolute De-peg (bps)|0.00|1,424.91|
|De-peg Events (|prem|> 50 bps)|
|AR(1) Persistence Coefficient|-|0.9982|
|Half-Life (minutes)|-|382.65|

We define a "de-peg event" as any minute where the absolute premium exceeds 50 basis points. Table 3 documents the stark contrast between USDC on the two exchanges:

**USDC Coinbase Stability**: Zero de-peg events occurred on Coinbase, consistent with its role as the reference market and the tight integration between Circle (USDC issuer) and Coinbase's infrastructure.

**USDC Binance Fragility**: Binance experienced 3,513 de-peg events over the 21-day window (out of approximately 30,240 total minutes), meaning 11.6% of all observations exhibited deviations exceeding 50 bps. The maximum deviation reached an extraordinary 1,424.91 bps (14.25%).

**High Persistence**: The AR(1) coefficient of 0.9982 indicates that USDC premiums on Binance are highly persistent—deviations tend to persist rather than quickly revert. The implied half-life of 382.65 minutes (approximately 6.4 hours) suggests that after a shock, it takes more than six hours for the premium to decay by half. This slow mean reversion reflects structural frictions: limited arbitrage capacity, withdrawal delays, cross-exchange settlement times, and risk aversion during volatile periods.

**Regulatory Implications**: The dramatic difference between exchanges highlights how regulatory status and infrastructure matter. Coinbase, as a U.S.-regulated exchange with direct banking relationships and institutional-grade custody, maintains tight USDC pricing. Binance, operating in a more ambiguous regulatory environment with different settlement mechanisms, experiences extreme dislocations. Under the GENIUS Act framework, requiring reserves, attestations, and banking oversight could reduce such fragmentation by standardizing redemption rights and settlement procedures.

### 2.3.4 Dynamic Interdependencies: VAR Analysis

To understand how stablecoin premiums evolve jointly over time and across exchanges, We estimate a VAR model with lag length selected via AIC, considering up to 10 lags on three premium series: USDC\_Binance, USDT\_Binance, and USDT\_Coinbase. (USDC\_Coinbase is excluded as it is identically zero by construction.)

**Table 4: VAR Model Summary**

|Model Statistics|Value|
|-|-|
|Number of Equations|3|
|Observations|24,465|
|AIC|9.350|
|BIC|9.380|
|Log Likelihood|-218,420|

**Key Findings from VAR Coefficients:**

**USDC Binance Equation** (Dependent Variable: prem\_USDC\_binance):

* Strong negative own-lag effects: L1 coefficient = -0.272 (t = -41.97, p < 0.001), indicating mean reversion
* Significant cross-effects from USDT premiums: USDT\_Binance L1 = -0.058 (t = -2.27, p = 0.023), USDT\_Coinbase L1 = -0.125 (t = -6.98, p < 0.001)
* Interpretation: USDC premiums on Binance revert toward zero but are also influenced by USDT pricing, suggesting interconnected funding dynamics

**USDT Binance Equation** (Dependent Variable: prem\_USDT\_binance):

* Very strong mean reversion: L1 coefficient = -0.610 (t = -89.90, p < 0.001)
* Positive feedback from USDT\_Coinbase: L1 coefficient = 0.058 (t = 12.29, p < 0.001), indicating cross-exchange co-movement
* Weak influence from USDC\_Binance: L1 coefficient = 0.005 (t = 3.06, p = 0.002)
* Interpretation: USDT markets exhibit tight integration across exchanges but operate somewhat independently from USDC

**USDT Coinbase Equation** (Dependent Variable: prem\_USDT\_coinbase):

* Strong mean reversion: L1 coefficient = -0.822 (t = -122.47, p < 0.001)
* Positive spillovers from USDT\_Binance: L1 coefficient = 0.214 (t = 22.26, p < 0.001)
* Weak but significant influence from USDC\_Binance: L1 coefficient = 0.007 (t = 2.85, p = 0.004)

**Residual Correlation Matrix:**

||USDC\_Binance|USDT\_Binance|USDT\_Coinbase|
|-|-|-|-|
|USDC\_Binance|1.000|0.155|0.088|
|USDT\_Binance|0.155|1.000|0.310|
|USDT\_Coinbase|0.088|0.310|1.000|

The correlation structure reveals that USDT markets across Binance and Coinbase are moderately correlated (ρ = 0.31), while USDC\_Binance shows weaker correlation with both USDT markets (ρ = 0.16 and 0.09). This suggests that USDC and USDT operate in partially segmented liquidity pools, with USDT exhibiting tighter cross-exchange integration.

**Economic Interpretation**: The VAR results indicate that stablecoin premiums exhibit strong mean reversion (arbitrage forces work to restore parity), but with significant cross-market spillovers. USDT markets show tight integration across exchanges, consistent with USDT's dominant role as the primary trading pair globally. USDC\_Binance, by contrast, operates with greater independence, likely due to lower USDC liquidity on Binance relative to Coinbase and regional differences in USDC adoption.

### 2.3.5 Lead-Lag Relationships and Price Discovery

To test for directional causality and price discovery leadership between exchanges, we estimate bidirectional lead-lag regressions. For each stablecoin, we regress the price change on one exchange (dependent variable) on contemporaneous and lagged price changes from both exchanges, allowing us to identify cross-market spillovers while controlling for own-market dynamics.

**USDT Lead-Lag Analysis (Coinbase regressed on Binance dynamics):**

We model USDT premium changes on Coinbase as a function of contemporaneous and lagged Binance changes (dA) plus own lagged Coinbase changes (dB):

|Variable|Coefficient|Std Error|z-stat|p-value|Interpretation|
|-|-|-|-|-|-|
|const|-0.0014|0.023|-0.063|0.950|No drift|
|dA\_lag0|0.3796|0.016|23.148|0.000|Contemporaneous spillover from Binance|
|dA\_lag1|0.2987|0.019|15.984|0.000|Binance leads Coinbase (1-min lag)|
|dA\_lag2|0.1604|0.014|11.411|0.000|Binance leads Coinbase (2-min lag)|
|dB\_lag1|-0.6470|0.009|-74.540|0.000|Mean reversion (own lag 1)|
|dB\_lag2|-0.3174|0.009|-36.535|0.000|Mean reversion (own lag 2)|

**Model Fit**: R² = 0.347, F-statistic = 1,250 (p < 0.001), N = 24,473

**Interpretation**:

The model explains 34.7% of USDT premium variation on Coinbase—substantially higher than typical cross-exchange models, indicating tight integration between the two markets. Several key patterns emerge:

1. **Strong Contemporaneous Spillovers**: The coefficient of 0.380 on dA\_lag0 indicates that approximately 38% of a Binance premium shock is immediately transmitted to Coinbase within the same minute. This reflects rapid information diffusion and arbitrage activity linking the two venues.
2. **Persistent Lead-Lag Effects**: Binance shocks continue to influence Coinbase pricing in subsequent periods, with coefficients of 0.299 (lag 1) and 0.160 (lag 2) both highly significant. The cumulative effect (0.380 + 0.299 + 0.160 = 0.839) suggests that approximately 84% of a Binance shock is eventually transmitted to Coinbase, with the majority occurring within two minutes.
3. **Mean Reversion**: The negative own-lag coefficients (-0.647 and -0.317) indicate that Coinbase premiums exhibit strong mean reversion. After controlling for Binance dynamics, Coinbase-specific shocks decay rapidly, consistent with arbitrage forces pulling premiums back toward equilibrium.
4. **Price Discovery Leadership**: The positive and significant Binance lag coefficients, combined with the high R², suggest that Binance leads Coinbase in USDT price discovery. This is consistent with Binance's position as the world's largest cryptocurrency exchange by volume, where global information is likely to be incorporated first before propagating to other venues.

**USDC Lead-Lag Analysis (Binance vs. Coinbase):**

The regression output shows all coefficients as zero with undefined statistics (NaN). This occurs because USDC\_Coinbase has zero variance (it is our reference price by construction), making a standard lead-lag regression degenerate. This result is expected and reinforces that USDC pricing on Coinbase serves as the stable anchor, while Binance exhibits all the variation.

**Price Discovery Summary**:

USDT markets exhibit strong bidirectional integration with Binance leading Coinbase in price discovery. The 34.7% R² represents substantial explanatory power in high-frequency microstructure models, indicating that cross-exchange arbitrage links these markets tightly despite geographic and regulatory differences.

USDC, by contrast, exhibits a hub-and-spoke structure: Coinbase serves as the definitive pricing source (via direct Circle redemptions, U.S. regulatory oversight, and institutional infrastructure), while Binance prices fluctuate around this anchor with slow mean reversion driven by liquidity constraints and settlement frictions.

### 2.3.6 Visual Evidence: Time Series and Spread Dynamics

**Figure 2.1: Premium Time Series by Exchange and Stablecoin**

The premium time series (Figure 2.1 - see q2\_premium\_timeseries.png) provides clear visual confirmation of our quantitative findings:

1. **USDT Co-movement**: USDT premiums on Coinbase and Binance track closely throughout the sample period, moving nearly in lockstep. Both series hover around zero during the pre-crisis period (March 1-9), plunge to discounts of -150 to -180 bps during the SVB crisis (March 10-13), and then stabilize at persistent discounts of approximately -40 to -50 bps in the post-crisis period. This tight co-movement is consistent with the strong cross-exchange integration documented in our lead-lag analysis.
2. **USDC Divergence**: The USDC series exhibit dramatically different behavior across exchanges. USDC on Coinbase (our reference) remains perfectly flat at zero throughout the entire period, reflecting the stability of this venue's pricing infrastructure. In contrast, USDC on Binance exhibits explosive volatility during the crisis, spiking to over 1,400 bps before gradually mean-reverting over several days.
3. **Asymmetric Crisis Response**: The crisis period reveals asymmetric responses. USDT discounts widen moderately and symmetrically across both exchanges, while USDC on Binance experiences an extreme premium spike unique to that venue. This asymmetry highlights venue-specific liquidity and settlement characteristics.

**Figure 2.2: Cross-Currency Spread (USDT - USDC by Exchange)**

The cross-currency spread charts (Figure 2.2 - see q2\_cross\_spread.png) illustrate the relative pricing between USDT and USDC, revealing important differences in market structure:

1. **Coinbase Spread Dynamics**: On Coinbase, the USDT-USDC spread oscillates near zero during normal periods (March 1-9), indicating rough parity between the two stablecoins. During the crisis, the spread widens to approximately -150 to -200 bps, meaning USDT traded at a substantial discount relative to USDC. Post-crisis, the spread settles at a persistent level of -40 to -50 bps, suggesting a permanent reassessment of relative stablecoin quality.
2. **Binance Spread Extremes**: On Binance, the spread behavior is far more dramatic. Starting near zero in normal times, it plunges to approximately -1,200 bps at the crisis peak—an order of magnitude larger than the Coinbase spread. This extreme dislocation reflects Binance's lower USDC liquidity, greater exposure to offshore funding stress, and slower arbitrage convergence. The gradual recovery over 3-4 days demonstrates the persistence of venue-specific frictions.
3. **Structural Interpretation**: The much larger spread on Binance during stress periods indicates that this exchange expnces greater liquidity fragmentation when confidence in stablecoins deteriorates. Traders on Binance apparently faced higher costs or longer delays in arbitraging the USDT-USDC spread, allowing the dislocation to persist. This finding has important implications for institutional participants considering venue selection and execution strategies during volatile periods.

The visual evidence reinforces our statistical findings: USDT markets are tightly integrated across exchanges with rapid information transmission, while USDC exhibits a core-periphery structure with Coinbase as the stable core and Binance as a more volatile periphery subject to liquidity and settlement constraints.

## 2.4 Discussion and Regulatory Implications

### 2.4.1 Stablecoin Heterogeneity

Our findings demonstrate that not all stablecoins behave alike. USDT consistently trades at a discount, reflecting persistent market skepticism about its reserves and governance, despite its dominance by trading volume. USDC, while generally more stable, exhibited extreme volatility on Binance during the SVB crisis, highlighting infrastructure and liquidity fragmentation across exchanges.

This heterogeneity has important implications for traders, risk managers, and regulators:

* **Collateral and Margin Systems**: Using USDT or USDC as margin collateral exposes users to de-pegging risk that varies by exchange and stablecoin issuer
* **Settlement Risk**: Cross-exchange arbitrage and settlement may be impaired during stress, as evidenced by the 6+ hour half-lives for USDC Binance premiums
* **Counterparty Risk Differentiation**: Markets clearly differentiate between stablecoin issuers based on transparency, regulatory status, and reserve quality

### 2.4.2 The GENIUS Act and Future Market Structure

The GENIUS Act represents a fundamental shift toward treating stablecoins as regulated financial instruments rather than experimental tokens. Key provisions—reserve requirements, regular attestations, federal oversight, and clear redemption rights—address many of the frictions observed in our data.

**Expected Effects of Regulation:**

1. **Reduced Premium Volatility**: Standardized redemption mechanisms and banking-grade custody should reduce the extreme deviations observed on Binance. If all USDC issuance is backed by segregated, attested reserves redeemable at par through regulated channels, the basis for large premiums/discounts diminishes.
2. **Convergence Across Exchanges**: Regulatory clarity and institutional infrastructure (e.g., Visa's USDC settlement rails) should facilitate faster arbitrage and tighter cross-exchange pricing. The current 382-minute half-life for USDC Binance premiums may shrink significantly if settlement times and liquidity access improve.
3. **USDT Uncertainty**: Tether's persistent discount reflects concerns about reserve quality and redemption reliability. If USDT cannot or does not comply with GENIUS Act requirements, its discount may widen further, or it may be displaced by compliant alternatives in U.S.-accessible markets. Conversely, successful compliance could narrow the discount.
4. **Institutional Adoption**: Regulated stablecoins enable traditional financial institutions (banks, asset managers, payment processors) to integrate crypto rails into treasury operations. This could increase liquidity, reduce fragmentation, and stabilize pricing—but may also concentrate liquidity in compliant, U.S.-regulated venues at the expense of offshore exchanges.

### 2.4.3 Implications for Market Participants

**For Traders and Arbitrageurs:**

* Stablecoin basis trades (e.g., long USDT/short USDC) may become less profitable as regulation reduces mispricings, but event-driven opportunities (e.g., during regulatory announcements or issuer disclosures) may persist
* Cross-exchange arbitrage requires careful attention to settlement times, withdrawal policies, and regulatory jurisdiction

**For Institutional Treasury Desks:**

* Regulated stablecoins offer a pathway to blockchain-based settlement with reduced operational and compliance risk
* Choice of exchange and stablecoin matters: Coinbase/USDC infrastructure offers tighter pricing and lower de-pegging risk than offshore alternatives

**For Regulators:**

* Stablecoin regulation can reduce systemic risk by ensuring reserve backing and redemption reliability
* However, fragmentation across jurisdictions (U.S. vs. offshore exchanges) may create regulatory arbitrage opportunities and two-tiered market structures

## 2.5 Conclusion

Our analysis of stablecoin premium dynamics reveals substantial heterogeneity across issuers, exchanges, and market regimes. USDT trades at persistent discounts reflecting credibility concerns, while USDC exhibits exchange-dependent behavior: stable on Coinbase but highly volatile on Binance, particularly during the March 2023 SVB crisis. De-pegging events on Binance were frequent (11.6% of observations) and persistent (6+ hour half-lives), indicating structural frictions in cross-exchange arbitrage and settlement.

These findings underscore the importance of regulatory frameworks like the GENIUS Act. By standardizing reserve requirements, redemption procedures, and oversight, such regulation has the potential to reduce volatility, improve arbitrage efficiency, and enhance market resilience. However, the transition period may introduce new dynamics as compliant and non-compliant stablecoins diverge in pricing and adoption.

For market participants operating in this evolving landscape, understanding stablecoin-specific risks, exchange infrastructure differences, and regulatory developments is essential for effective execution, risk management, and strategic positioning.

