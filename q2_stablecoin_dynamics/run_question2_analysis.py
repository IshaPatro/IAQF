"""
Complete End-to-End Analysis for Question 2: Stablecoin Dynamics
IAQF Competition 2026

Loads REAL CSVs from:
- data/coinbase/
- data/binance_clean/

Generates outputs in:
- outputs/
"""

from pathlib import Path
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from stablecoin_dynamics_analysis import StablecoinDynamicsAnalyzer
from q2_models import build_premium_panel, lead_lag_granger_regression, run_var_and_irf


import matplotlib.pyplot as plt


# ---------------------------
# Paths (portable)
# ---------------------------
BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
COINBASE_DIR = DATA / "coinbase"
BINANCE_DIR = DATA / "binance_clean"
OUTPUTS = BASE / "outputs"
OUTPUTS.mkdir(exist_ok=True)


def _load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "timestamp" not in df.columns:
        raise ValueError(f"{path} has no 'timestamp' column.")
    if "close" not in df.columns:
        raise ValueError(f"{path} has no 'close' column.")

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    return df


def load_data_dict():
    """
    Load whichever files exist. Raises error if USD baseline missing on Coinbase.
    """

    expected = {
        "BTC_USD_coinbase": COINBASE_DIR / "BTC_USD_coinbase.csv",
        "BTC_USDT_coinbase": COINBASE_DIR / "BTC_USDT_coinbase.csv",
        "BTC_USDC_coinbase": COINBASE_DIR / "BTC_USDC_coinbase.csv",
        "BTC_USD_binance": BINANCE_DIR / "BTC_USD_binance.csv",
        "BTC_USDT_binance": BINANCE_DIR / "BTC_USDT_binance.csv",
        "BTC_USDC_binance": BINANCE_DIR / "BTC_USDC_binance.csv",
    }

    data_dict = {}
    missing = []

    for k, p in expected.items():
        if p.exists():
            data_dict[k] = _load_csv(p)
        else:
            missing.append(k)

    # Require at least a USD baseline from Coinbase
    if "BTC_USD_coinbase" not in data_dict:
        raise FileNotFoundError(
            f"Missing required USD baseline: {expected['BTC_USD_coinbase']}. "
            "This is required to compute premiums."
        )

    print("Loaded series:")
    for k, df in data_dict.items():
        print(f"  ✓ {k}: {len(df):,} rows, {df['timestamp'].min()} -> {df['timestamp'].max()}")

    if missing:
        print("\nMissing (not fatal unless you need them):")
        for k in missing:
            print(f"  - {k}")

    return data_dict


def save_text(path: Path, text: str):
    path.write_text(text, encoding="utf-8")


def main():
    print("=" * 70)
    print("IAQF COMPETITION 2026 - QUESTION 2: STABLECOIN DYNAMICS ANALYSIS")
    print("=" * 70)

    # 1) Load data
    data_dict = load_data_dict()

    # 2) Run core analyzer
    analyzer = StablecoinDynamicsAnalyzer(data_dict)

    premiums = analyzer.calculate_stablecoin_premium()
    cross_spreads = analyzer.calculate_cross_stablecoin_spread()
    regimes = analyzer.detect_stress_regimes()

    exchange_stats = analyzer.analyze_exchange_differences()
    regime_comparison = analyzer.run_regime_comparison()
    depeg_analysis = analyzer.analyze_depeg_dynamics()

    # Save core tables
    exchange_stats.to_csv(OUTPUTS / "q2_table1_exchange_stats.csv")
    regime_comparison.to_csv(OUTPUTS / "q2_table2_regime_comparison.csv")
    depeg_analysis.to_csv(OUTPUTS / "q2_table3_depeg_analysis.csv")

    # Save core plots
    analyzer.plot_premium_timeseries(save_path=str(OUTPUTS / "q2_premium_timeseries.png"))
    analyzer.plot_cross_stablecoin_spread(save_path=str(OUTPUTS / "q2_cross_spread.png"))

    # 3) Build premium panel for advanced models
    prem_panel = build_premium_panel(analyzer, premiums=premiums)
    prem_panel.to_csv(OUTPUTS / "q2_premium_panel.csv", index=False)

    # 4) Lead–Lag tests (only if both exchanges exist for the stablecoin)
    # USDT
    if ("prem_USDT_binance_bps" in prem_panel.columns) and ("prem_USDT_coinbase_bps" in prem_panel.columns):
        #m_usdt = lead_lag_test(prem_panel, "prem_USDT_binance_bps", "prem_USDT_coinbase_bps", lags=10)
        m_usdt = lead_lag_granger_regression(
           prem_panel,
           "prem_USDT_binance_bps",
           "prem_USDT_coinbase_bps",
           lags=2,
           hac_lags=2
       )
        save_text(OUTPUTS / "leadlag_USDT_binance_vs_coinbase.txt", m_usdt.summary().as_text())
        print("✓ Saved lead-lag USDT (Binance -> Coinbase)")

    # USDC
    if ("prem_USDC_binance_bps" in prem_panel.columns) and ("prem_USDC_coinbase_bps" in prem_panel.columns):
        #m_usdc = lead_lag_test(prem_panel, "prem_USDC_binance_bps", "prem_USDC_coinbase_bps", lags=10)
        m_usdc = lead_lag_granger_regression(
           prem_panel,
           "prem_USDC_binance_bps",
           "prem_USDC_coinbase_bps",
           lags=2,
           hac_lags=2
      )
        save_text(OUTPUTS / "leadlag_USDC_binance_vs_coinbase.txt", m_usdc.summary().as_text())
        print("✓ Saved lead-lag USDC (Binance -> Coinbase)")

    # 5) VAR + IRF plot (choose available columns)
    var_cols = [c for c in prem_panel.columns if c.startswith("prem_") and c.endswith("_bps")]
    # Keep it focused: prefer USDC/USDT for both exchanges if available
    preferred = [
        "prem_USDC_binance_bps", "prem_USDC_coinbase_bps",
        "prem_USDT_binance_bps", "prem_USDT_coinbase_bps",
    ]
    cols = [c for c in preferred if c in var_cols]
    if len(cols) >= 2:
        fit, irf, used_cols = run_var_and_irf(
            prem_panel,
            cols=cols,
            maxlags=10,
            ic="aic",
            irf_steps=120,
            use_diffs=True
        )
        save_text(OUTPUTS / "var_summary.txt", str(fit.summary()))
        save_text(OUTPUTS / "var_series_used.txt", "\n".join(used_cols))

        fig = irf.plot(orth=False)
        plt.tight_layout()
        plt.savefig(OUTPUTS / "var_irf.png", dpi=300, bbox_inches="tight")
        plt.close()

        print(f"✓ Saved VAR summary + IRF plot (series used: {used_cols})")

        plt.tight_layout()
        plt.savefig(OUTPUTS / "var_irf.png", dpi=300, bbox_inches="tight")
        plt.close()

        print("✓ Saved VAR summary + IRF plot")

    print("\nDone. Outputs saved to:", OUTPUTS)


if __name__ == "__main__":
    main()
