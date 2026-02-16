"""
IAQF Competition 2026 - Question 2: Stablecoin Dynamics Analysis
Framework for analyzing premium/discount patterns in stablecoin-quoted markets

Focus: Stablecoin dynamics during March 2023 SVB crisis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings("ignore")


class StablecoinDynamicsAnalyzer:
    """
    Analyzes premium/discount patterns between stablecoin-quoted markets
    """

    def __init__(self, data_dict):
        """
        Initialize with dictionary of DataFrames

        Expected keys may include:
            'BTC_USD_coinbase'
            'BTC_USDT_coinbase'
            'BTC_USDC_coinbase'
            (optionally other exchanges)

        Each df should have columns: timestamp, open, high, low, close, volume
        """
        self.data = data_dict
        self.results = {}

        # Force timestamps to UTC-aware consistently (prevents tz-naive vs tz-aware crashes)
        self._standardize_timestamps_to_utc()

    def _standardize_timestamps_to_utc(self):
        """
        Ensuring every dataframe has a UTC-aware datetime timestamp column.
        """
        for k, df in self.data.items():
            if "timestamp" not in df.columns:
                continue
            # Parse as UTC-aware; if already tz-aware, stays consistent
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
            df.dropna(subset=["timestamp"], inplace=True)
            df.sort_values("timestamp", inplace=True)
            df.reset_index(drop=True, inplace=True)
            self.data[k] = df

    @staticmethod
    def _utc_ts(s: str) -> pd.Timestamp:
        """
        Convenience: create a UTC-aware Timestamp.
        """
        return pd.Timestamp(s, tz="UTC")

    def calculate_stablecoin_premium(self, base_asset="BTC"):
        """
        Calculating premium/discount of stablecoin-quoted prices relative to USD.

        Premium(t) = (P_BTC/STABLE / P_BTC/USD - 1) * 10,000 bps

        Returns:
            dict of DataFrames for each stablecoin market premium time series.
        """
        print("Calculating stablecoin premiums...")

        premiums = {}

        # Get USD baseline (any key containing 'USD' but not 'USDT'/'USDC')
        usd_candidates = [
            k for k in self.data.keys()
            if ("USD" in k) and ("USDT" not in k) and ("USDC" not in k)
        ]
        if not usd_candidates:
            raise KeyError(
                "Could not find a USD baseline series. Expected a key like 'BTC_USD_coinbase'."
            )

        usd_key = usd_candidates[0]
        usd_prices = self.data[usd_key][["timestamp", "close"]].copy()
        usd_prices.columns = ["timestamp", "price_usd"]

        # Calculate premiums for each stablecoin market
        for key, df in self.data.items():
            if ("USDT" in key) or ("USDC" in key):
                stable_type = "USDT" if "USDT" in key else "USDC"
                exchange = key.split("_")[-1]

                merged = df[["timestamp", "close"]].merge(usd_prices, on="timestamp", how="inner")
                if merged.empty:
                    # No overlapping timestamps
                    continue

                merged[f"{stable_type}_premium_bps"] = (merged["close"] / merged["price_usd"] - 1.0) * 10000.0
                premiums[f"{stable_type}_{exchange}"] = merged[["timestamp", f"{stable_type}_premium_bps"]]

        self.results["premiums"] = premiums
        return premiums

    def calculate_cross_stablecoin_spread(self):
        """
        Calculate spread between USDT and USDC quoted prices on the same exchange:

        Spread(t) = (P_BTC/USDT / P_BTC/USDC - 1) * 10,000 bps
        """
        print("Calculating cross-stablecoin spreads...")

        spreads = {}

        exchanges = set([k.split("_")[-1] for k in self.data.keys()])

        for exchange in exchanges:
            usdt_key = [k for k in self.data.keys() if ("USDT" in k and exchange in k)]
            usdc_key = [k for k in self.data.keys() if ("USDC" in k and exchange in k)]

            if usdt_key and usdc_key:
                usdt_df = self.data[usdt_key[0]][["timestamp", "close"]].copy()
                usdc_df = self.data[usdc_key[0]][["timestamp", "close"]].copy()

                usdt_df.columns = ["timestamp", "price_usdt"]
                usdc_df.columns = ["timestamp", "price_usdc"]

                merged = usdt_df.merge(usdc_df, on="timestamp", how="inner")
                if merged.empty:
                    continue

                merged["spread_bps"] = (merged["price_usdt"] / merged["price_usdc"] - 1.0) * 10000.0
                spreads[exchange] = merged

        self.results["cross_spreads"] = spreads
        return spreads

    def detect_stress_regimes(self, threshold_std=2.0):
        """
        Detecting stress vs normal regimes based on premium volatility.
        SVB crisis window: March 10-13, 2023 (UTC).

        """
        print("Detecting stress regimes...")

        # Known SVB crisis period (UTC-aware)
        svb_start = self._utc_ts("2023-03-10 00:00:00")
        svb_end = self._utc_ts("2023-03-13 23:59:59")

        regimes = {}

        if "premiums" not in self.results:
            self.calculate_stablecoin_premium()

        for key, premium_df in self.results["premiums"].items():
            df = premium_df.copy()

            # Ensure UTC-aware timestamps
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
            df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

            col = [c for c in df.columns if "premium_bps" in c][0]

            # 1-hour rolling volatility for 1-minute data
            df["rolling_vol"] = df[col].rolling(window=60).std()

            # Mark SVB period
            df["svb_crisis"] = ((df["timestamp"] >= svb_start) & (df["timestamp"] <= svb_end)).astype(int)

            # Statistical regime detection using high volatility
            mean_vol = df["rolling_vol"].mean()
            std_vol = df["rolling_vol"].std()
            df["high_stress"] = (df["rolling_vol"] > mean_vol + threshold_std * std_vol).astype(int)

            regimes[key] = df

        self.results["regimes"] = regimes
        return regimes

    def analyze_exchange_differences(self):
        """
        Compare premium/discount patterns across markets
        
        """
        print("Analyzing exchange-level differences...")

        if "premiums" not in self.results:
            self.calculate_stablecoin_premium()

        exchange_stats = {}

        for key, premium_df in self.results["premiums"].items():
            col = [c for c in premium_df.columns if "premium_bps" in c][0]

            stats_dict = {
                "mean_premium_bps": premium_df[col].mean(),
                "std_premium_bps": premium_df[col].std(),
                "min_premium_bps": premium_df[col].min(),
                "max_premium_bps": premium_df[col].max(),
                "median_premium_bps": premium_df[col].median(),
                "skewness": premium_df[col].skew(),
                "kurtosis": premium_df[col].kurtosis(),
            }

            exchange_stats[key] = stats_dict

        self.results["exchange_stats"] = pd.DataFrame(exchange_stats).T
        return self.results["exchange_stats"]

    def run_regime_comparison(self):
        """
        Compare stablecoin behavior in normal vs SVB crisis regimes.
        """
        print("Running regime comparison analysis...")

        if "regimes" not in self.results:
            self.detect_stress_regimes()

        regime_comparison = {}

        for key, regime_df in self.results["regimes"].items():
            col = [c for c in regime_df.columns if "premium_bps" in c][0]

            crisis = regime_df[regime_df["svb_crisis"] == 1]
            normal = regime_df[regime_df["svb_crisis"] == 0]

            comparison = {
                "normal_mean": normal[col].mean(),
                "normal_std": normal[col].std(),
                "normal_min": normal[col].min(),
                "normal_max": normal[col].max(),
                "crisis_mean": crisis[col].mean(),
                "crisis_std": crisis[col].std(),
                "crisis_min": crisis[col].min(),
                "crisis_max": crisis[col].max(),
                "mean_diff": crisis[col].mean() - normal[col].mean(),
                "vol_ratio": (crisis[col].std() / normal[col].std()) if normal[col].std() > 0 else np.nan,
            }

            if (len(crisis) > 0) and (len(normal) > 0):
                t_stat, p_value = stats.ttest_ind(crisis[col], normal[col], equal_var=False, nan_policy="omit")
                comparison["t_statistic"] = t_stat
                comparison["p_value"] = p_value
            else:
                comparison["t_statistic"] = np.nan
                comparison["p_value"] = np.nan

            regime_comparison[key] = comparison

        self.results["regime_comparison"] = pd.DataFrame(regime_comparison).T
        return self.results["regime_comparison"]

    def analyze_depeg_dynamics(self, window_hours=24):
        """
        Analyze dynamics of stablecoin depegging (focus USDC).
        """
        print("Analyzing depeg dynamics...")

        if "premiums" not in self.results:
            self.calculate_stablecoin_premium()

        depeg_analysis = {}

        usdc_premiums = {k: v for k, v in self.results["premiums"].items() if "USDC" in k}

        for key, premium_df in usdc_premiums.items():
            df = premium_df.copy()
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
            df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

            col = [c for c in df.columns if "premium_bps" in c][0]

            window_size = window_hours * 60  # 1-min data
            df["rolling_mean"] = df[col].rolling(window=window_size).mean()
            df["rolling_std"] = df[col].rolling(window=window_size).std()
            df["z_score"] = (df[col] - df["rolling_mean"]) / df["rolling_std"]

            depeg_threshold = 50  # 50 bps = 0.5%
            df["depeg_event"] = (np.abs(df[col]) > depeg_threshold).astype(int)

            # AR(1) persistence / half-life
            df["premium_lag1"] = df[col].shift(1)
            clean_df = df.dropna(subset=[col, "premium_lag1"])

            if len(clean_df) > 100:
                X = clean_df["premium_lag1"].values.reshape(-1, 1)
                y = clean_df[col].values
                from sklearn.linear_model import LinearRegression
                model = LinearRegression().fit(X, y)
                persistence = float(model.coef_[0])

                if 0 < persistence < 1:
                    half_life = -np.log(2) / np.log(persistence)  # in minutes
                else:
                    half_life = np.nan
            else:
                persistence = np.nan
                half_life = np.nan

            depeg_analysis[key] = {
                "max_depeg_bps": df[col].max(),
                "min_depeg_bps": df[col].min(),
                "max_abs_depeg_bps": np.abs(df[col]).max(),
                "depeg_events_count": float(df["depeg_event"].sum()),
                "persistence_coef": persistence,
                "half_life_minutes": half_life,
            }

        self.results["depeg_analysis"] = pd.DataFrame(depeg_analysis).T
        return self.results["depeg_analysis"]

    def plot_premium_timeseries(self, save_path="premium_timeseries.png"):
        """
        Plot premium/discount time series with SVB crisis highlighted (UTC-safe)
        """
        if "premiums" not in self.results:
            self.calculate_stablecoin_premium()

        fig, axes = plt.subplots(
            len(self.results["premiums"]),
            1,
            figsize=(14, 4 * max(1, len(self.results["premiums"]))),
        )

        if len(self.results["premiums"]) == 1:
            axes = [axes]

        svb_start = self._utc_ts("2023-03-10 00:00:00")
        svb_end = self._utc_ts("2023-03-13 23:59:59")

        for idx, (key, premium_df) in enumerate(self.results["premiums"].items()):
            df = premium_df.copy()
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
            df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

            col = [c for c in df.columns if "premium_bps" in c][0]

            axes[idx].plot(df["timestamp"], df[col], linewidth=0.6, alpha=0.8)
            axes[idx].axhline(y=0, color="black", linestyle="--", linewidth=1)
            axes[idx].axvspan(svb_start, svb_end, alpha=0.25, color="red", label="SVB Crisis")
            axes[idx].set_title(f"{key} Premium/Discount", fontsize=12, fontweight="bold")
            axes[idx].set_ylabel("Premium (bps)")
            axes[idx].grid(True, alpha=0.3)
            axes[idx].legend()

        axes[-1].set_xlabel("Date (UTC)")
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved plot to {save_path}")
        return fig

    def plot_cross_stablecoin_spread(self, save_path="cross_spread.png"):
        """
        Plot USDT vs USDC spread over time
        """
        if "cross_spreads" not in self.results:
            self.calculate_cross_stablecoin_spread()

        if len(self.results["cross_spreads"]) == 0:
            print("No cross-stablecoin spreads available to plot.")
            return None

        fig, axes = plt.subplots(
            len(self.results["cross_spreads"]),
            1,
            figsize=(14, 4 * max(1, len(self.results["cross_spreads"]))),
        )

        if len(self.results["cross_spreads"]) == 1:
            axes = [axes]

        svb_start = self._utc_ts("2023-03-10 00:00:00")
        svb_end = self._utc_ts("2023-03-13 23:59:59")

        for idx, (exchange, spread_df) in enumerate(self.results["cross_spreads"].items()):
            df = spread_df.copy()
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
            df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

            axes[idx].plot(df["timestamp"], df["spread_bps"], linewidth=0.6, alpha=0.8)
            axes[idx].axhline(y=0, color="black", linestyle="--", linewidth=1)
            axes[idx].axvspan(svb_start, svb_end, alpha=0.25, color="red", label="SVB Crisis")
            axes[idx].set_title(f"{exchange}: BTC/USDT - BTC/USDC Spread", fontsize=12, fontweight="bold")
            axes[idx].set_ylabel("Spread (bps)")
            axes[idx].grid(True, alpha=0.3)
            axes[idx].legend()

        axes[-1].set_xlabel("Date (UTC)")
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved plot to {save_path}")
        return fig

    def generate_summary_report(self):
        """
        Generating comprehensive summary statistics
        """
        print("\n" + "=" * 60)
        print("STABLECOIN DYNAMICS ANALYSIS SUMMARY")
        print("=" * 60 + "\n")

        if "exchange_stats" in self.results:
            print("Exchange-Level Premium Statistics:")
            print(self.results["exchange_stats"].round(2))
            print("\n")

        if "regime_comparison" in self.results:
            print("Normal vs Crisis Regime Comparison:")
            print(self.results["regime_comparison"].round(2))
            print("\n")

        if "depeg_analysis" in self.results:
            print("Depeg Dynamics Analysis:")
            print(self.results["depeg_analysis"].round(2))
            print("\n")

        return self.results

