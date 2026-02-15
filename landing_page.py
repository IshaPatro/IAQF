import streamlit as st
import pandas as pd
import os

def render():
    st.title("Cross-Currency Dynamics in Cryptocurrencies under Stablecoin Regulation")
    st.subheader("IAQF Student Competition 2026")
    st.divider()

    st.markdown("### The Regime Shift")
    st.markdown(
        """
        The **Guarding Essential Networks and Insuring US Stablecoins (GENIUS) Act of 2025** marks a 
        structural inflection point in digital finance. It integrates stablecoins into the U.S. banking 
        framework, moving the market from a patchwork of state-level oversight to a unified federal regime.
        This project models how this specific regulatory shift alters cross-currency capital flows and 
        stablecoin behavior under stress.
        """
    )

    st.divider()

    col_hero, col_hero_text = st.columns([1, 2])
    with col_hero:
        try:
            image_path = os.path.join(os.path.dirname(__file__), "btc.jpeg")
            st.image(image_path, use_column_width=True)
        except Exception:
            pass
    with col_hero_text:
        st.markdown(
            """
            #### Why This Matters
            Stablecoins now settle **more daily volume than Visa**. A single depeg event in March 2023 
            triggered over **$2 billion in capital rotation** within 48 hours. The GENIUS Act fundamentally 
            changes the rules — this project measures what that means for market stability.
            """
        )

    st.divider()

    st.subheader("The GENIUS Act: Core Regulations")

    st.markdown(
        """
        The Act enforces the "Singleness of Money", ensuring a stablecoin dollar is always interchangeable 
        with a fiat dollar, through five mandatory pillars:
        """
    )

    st.info(
        """
        *   **Issuer Licensing:** Issuance is restricted to **Insured Depository Institutions** (banks) or non-bank entities 
            that meet strict federal solvency standards.
        *   **100% Reserve Backing:** Reserves must be held exclusively in **U.S. Dollars, demand deposits, or short-term Treasuries**. 
            Commercial paper and corporate debt are prohibited.
        *   **Segregation of Funds:** Customer reserves must be **legally segregated** from the issuer's operating capital, 
            granting holders statutory priority in bankruptcy.
        *   **No Algorithmic Stablecoins:** The Act explicitly **bans new algorithmic stablecoins** that rely on endogenous 
            collateral or arbitrage mechanisms to maintain a peg.
        *   **Compliance & Oversight:** All issuers are classified as financial institutions under the **Bank Secrecy Act**, 
            mandating full AML/KYC compliance.
        """
    )

    st.divider()

    st.subheader("Structural Shift: Pre vs. Post-GENIUS")

    st.markdown("A direct comparison of the market structure before and after the 2025 legislation:")

    comparison_data = {
        "Feature": ["Issuer Status", "Reserve Assets", "Redemption Rights", "Algorithmic Stablecoins", "Risk Profile"],
        "Pre-Genius Era (2020-2024)": [
            "State-Licensed / Offshore",
            "Cash, CP, Corp Bonds, Crypto",
            "Terms of Service / Unclear",
            "Permitted (e.g., TerraUSD)",
            "Run Risk / Solvency Uncertainty"
        ],
        "Post-Genius Era (2025+)": [
            "Insured Depository / Federal Trust",
            "100% Cash & Treasuries Only",
            "Statutory Priority Claim",
            "Prohibited / Phased Out",
            "Supervisory / Operational Only"
        ]
    }
    df = pd.DataFrame(comparison_data)
    st.table(df)

    st.divider()

    st.subheader("The 2023 Paradox: Credibility vs. Liquidity")

    st.markdown(
        """
        In March 2023, when regulated USDC depegged due to SVB exposure, capital flowed **into** unregulated USDT.
        This was the **credibility vs. liquidity paradox** — market participants optimized for execution speed 
        over institutional transparency.
        """
    )

    st.warning("**Central Question:** Does the GENIUS Act reverse this flight-to-liquidity dynamic?")

    st.markdown("")
    st.markdown("##### 📅 Day-by-Day Crisis Timeline")
    st.markdown("")

    st.markdown(
        """
        <style>
        .timeline-container {
            position: relative;
            padding-left: 40px;
            margin: 10px 0;
        }
        .timeline-container::before {
            content: '';
            position: absolute;
            left: 15px;
            top: 0;
            bottom: 0;
            width: 3px;
            background: linear-gradient(180deg, #ff4b4b 0%, #ffa534 30%, #4bff88 100%);
            border-radius: 2px;
        }
        .timeline-entry {
            position: relative;
            margin-bottom: 20px;
            padding: 14px 18px;
            background: rgba(255,255,255,0.04);
            border-radius: 10px;
            border-left: 3px solid rgba(255,255,255,0.1);
        }
        .timeline-entry::before {
            content: '';
            position: absolute;
            left: -33px;
            top: 18px;
            width: 11px;
            height: 11px;
            border-radius: 50%;
            border: 2px solid #fff;
        }
        .timeline-entry.crisis::before { background: #ff4b4b; border-color: #ff4b4b; }
        .timeline-entry.peak::before { background: #ff6b35; border-color: #ff6b35; }
        .timeline-entry.recovery::before { background: #ffa534; border-color: #ffa534; }
        .timeline-entry.stable::before { background: #4bff88; border-color: #4bff88; }
        .timeline-date {
            font-weight: 700;
            font-size: 0.95rem;
            margin-bottom: 6px;
            color: rgba(250,250,250,0.95);
        }
        .timeline-entry ul {
            margin: 0;
            padding-left: 18px;
        }
        .timeline-entry li {
            font-size: 0.88rem;
            line-height: 1.6;
            color: rgba(250,250,250,0.75);
        }
        .phase-label {
            display: inline-block;
            font-size: 0.7rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            padding: 2px 8px;
            border-radius: 4px;
            margin-left: 10px;
            vertical-align: middle;
        }
        .phase-crisis { background: rgba(255,75,75,0.2); color: #ff4b4b; }
        .phase-peak { background: rgba(255,107,53,0.2); color: #ff6b35; }
        .phase-recovery { background: rgba(255,165,52,0.2); color: #ffa534; }
        .phase-stable { background: rgba(75,255,136,0.2); color: #4bff88; }
        </style>
        """,
        unsafe_allow_html=True
    )

    timeline_entries = [
        ("crisis", "March 9, 2023", "phase-crisis", "CRISIS ONSET", [
            "Depositors withdraw ~$42 billion from Silicon Valley Bank",
            "SVB liquidity collapses",
            "Crypto markets begin pricing banking counterparty exposure risk",
        ]),
        ("crisis", "March 10, 2023", "phase-crisis", "BANK CLOSURE", [
            "Regulators close SVB; FDIC appointed as receiver",
            "Circle discloses $3.3B of USDC reserves held at SVB",
            "USDC drops below $1.00",
            "Redemption pressure accelerates; arbitrage flows slow",
        ]),
        ("peak", "March 11, 2023", "phase-peak", "PEAK STRESS", [
            "USDC trades as low as $0.87–$0.88",
            "Massive sell-off across exchanges",
            "Coinbase halts USDC → USD conversions (weekend bank closures)",
            "Capital rotates into Tether (USDT)",
            "DAI depegs due to USDC collateral exposure",
            "Cross-currency spreads widen sharply between BTC/USD and BTC/USDT",
        ]),
        ("recovery", "March 12, 2023", "phase-recovery", "INTERVENTION", [
            "U.S. regulators announce all SVB depositors will be made whole",
            "Confidence begins returning to USDC markets",
            "USDC recovers toward $0.95–$0.98",
            "Stablecoin funding spreads begin narrowing",
        ]),
        ("recovery", "March 13, 2023", "phase-recovery", "RECOVERY", [
            "U.S. banking system reopens",
            "Circle confirms access to reserves",
            "USDC returns close to $1.00",
            "Coinbase resumes USDC → USD conversions",
            "Cross-currency basis begins compressing",
        ]),
        ("stable", "March 14, 2023", "phase-stable", "STABILIZATION", [
            "Stablecoin peg effectively restored",
            "Redemption queues normalize",
            "Exchange spreads narrow; liquidity fragmentation decreases",
        ]),
        ("stable", "March 15, 2023", "phase-stable", "ANALYSIS", [
            "On-chain data shows sustained capital rotation into USDT during depeg window",
            "Market analysis frames event as liquidity interruption, not insolvency",
        ]),
        ("stable", "March 16–18, 2023", "phase-stable", "NORMALIZATION", [
            "USDC trades consistently near $1.00",
            "Cross-currency pricing realigns across exchanges",
            "Secondary market liquidity deepens; arbitrage fully restored",
        ]),
        ("stable", "March 19–21, 2023", "phase-stable", "POST-CRISIS", [
            "No further peg stress observed; stablecoin markets operate normally",
            "Commentary labels event as a banking-linked liquidity shock",
            "Market focus shifts from crisis management to regulatory implications",
        ]),
    ]

    html_parts = ['<div class="timeline-container">']
    for css_class, date, phase_class, phase_label, bullets in timeline_entries:
        bullet_html = "".join(f"<li>{b}</li>" for b in bullets)
        html_parts.append(
            f'<div class="timeline-entry {css_class}">'
            f'<div class="timeline-date">{date} <span class="phase-label {phase_class}">{phase_label}</span></div>'
            f'<ul>{bullet_html}</ul>'
            f'</div>'
        )
    html_parts.append('</div>')

    st.markdown("".join(html_parts), unsafe_allow_html=True)

    st.divider()
