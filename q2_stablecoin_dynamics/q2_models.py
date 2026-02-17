# q2_models.py
"""
Models/utilities for Question 2 (Stablecoin dynamics / price discovery)

This module provides:
1) build_premium_panel: builds a wide, timestamp-aligned premium panel
2) lead_lag_predictive_regression: simple HAC-robust lead/lag regression (predictive regression)
3) lead_lag_granger_regression: "Granger-style" regression that controls for B's own lags
4) run_var_and_irf: robust VAR fit + impulse responses (IRFs)


"""

from __future__ import annotations

import warnings
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.api import VAR


# -----------------------------
# premium panel
# -----------------------------
def build_premium_panel(
    analyzer,
    premiums: Optional[Dict[str, pd.DataFrame]] = None,
    timestamp_col: str = "timestamp",
) -> pd.DataFrame:
    """
    Builds a single wide panel of premiums aligned on timestamp.

    Parameters
    ----------
    analyzer : object
        Must have method calculate_stablecoin_premium() returning dict[str, DataFrame].
        Keys are expected to look like "USDT_coinbase" or "USDC_binance".
    premiums : dict[str, DataFrame] | None
        If provided, uses these premiums instead of recomputing them.
    timestamp_col : str
        Name of the timestamp column.

    Returns
    -------
    prem_panel : pd.DataFrame
        Columns: timestamp + prem_{stable}_{exchange}_bps for each series.
    """
    if premiums is None:
        premiums = analyzer.calculate_stablecoin_premium()

    if not isinstance(premiums, dict) or len(premiums) == 0:
        raise ValueError("No premium series produced. premiums must be a non-empty dict.")

    frames: List[pd.DataFrame] = []

    for k, df in premiums.items():
        if df is None or df.empty:
            continue
        if timestamp_col not in df.columns:
            raise ValueError(f"Premium df for key '{k}' is missing '{timestamp_col}' column.")

        # find premium column
        prem_cols = [c for c in df.columns if "premium_bps" in c]
        if not prem_cols:
            raise ValueError(f"Premium df for key '{k}' has no column containing 'premium_bps'.")
        prem_col = prem_cols[0]

        # parse key -> stable, exch
        if "_" not in k:
            raise ValueError(f"Premium key '{k}' must contain '_' (e.g., 'USDC_coinbase').")
        stable, exch = k.split("_", 1)

        out_col = f"prem_{stable}_{exch}_bps"

        tmp = df[[timestamp_col, prem_col]].copy()
        tmp[timestamp_col] = pd.to_datetime(tmp[timestamp_col], utc=True, errors="coerce")
        tmp = tmp.dropna(subset=[timestamp_col]).sort_values(timestamp_col)
        tmp = tmp.rename(columns={prem_col: out_col})
        frames.append(tmp)

    if not frames:
        raise ValueError(
            "No usable premium frames after filtering. Check your premium computation and timestamps."
        )

    # outer-merge on timestamp
    prem_panel = frames[0]
    for f in frames[1:]:
        prem_panel = prem_panel.merge(f, on=timestamp_col, how="outer")

    prem_panel = prem_panel.sort_values(timestamp_col).reset_index(drop=True)
    return prem_panel


# -----------------------------
# Helpers
# -----------------------------
def _prepare_two_series_panel(
    prem_panel: pd.DataFrame,
    col_a: str,
    col_b: str,
    timestamp_col: str = "timestamp",
    diff: bool = True,
) -> pd.DataFrame:
    """
    Returns a cleaned dataframe with timestamp index and (optionally) differenced series.
    """
    if timestamp_col not in prem_panel.columns:
        raise ValueError(f"prem_panel must contain '{timestamp_col}' column.")

    df = prem_panel[[timestamp_col, col_a, col_b]].dropna().copy()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], utc=True, errors="coerce")
    df = df.dropna(subset=[timestamp_col]).sort_values(timestamp_col).reset_index(drop=True)

    # set index for consistent shifting
    df = df.set_index(timestamp_col)

    if diff:
        df[col_a] = df[col_a].astype(float).diff()
        df[col_b] = df[col_b].astype(float).diff()
        df = df.dropna(subset=[col_a, col_b])

    return df


# -----------------------------
# 2) predictive lead/lag regression (HAC)
# -----------------------------
def lead_lag_predictive_regression(
    prem_panel: pd.DataFrame,
    col_a: str,
    col_b: str,
    lags: int = 10,
    hac_lags: int = 10,
    timestamp_col: str = "timestamp",
) -> sm.regression.linear_model.RegressionResultsWrapper:
    """
    Predictive lead/lag regression :
        dB_t = alpha + sum_{k=0..lags} beta_k dA_{t-k} + eps_t

    - Uses HAC (Newey-West) robust standard errors.
    - "Lead" evidence: significant beta_k for k>=1.

    Returns
    -------
    statsmodels RegressionResultsWrapper
    """
    if lags < 0:
        raise ValueError("lags must be >= 0")
    if hac_lags < 0:
        raise ValueError("hac_lags must be >= 0")

    df = _prepare_two_series_panel(prem_panel, col_a, col_b, timestamp_col=timestamp_col, diff=True)

    # Build X with contemporaneous and lagged dA
    X = pd.concat(
        [df[col_a].shift(k).rename(f"dA_lag{k}") for k in range(0, lags + 1)],
        axis=1,
    ).dropna()

    y = df.loc[X.index, col_b]
    X = sm.add_constant(X)

    model = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags})
    return model


# -----------------------------
# 3) Granger-style regression
# -----------------------------
def lead_lag_granger_regression(
    prem_panel: pd.DataFrame,
    col_a: str,
    col_b: str,
    lags: int = 10,
    hac_lags: int = 10,
    timestamp_col: str = "timestamp",
    include_contemporaneous_a: bool = True,
) -> sm.regression.linear_model.RegressionResultsWrapper:
    """
     "Granger-style" regression (controls for B's own lags):

        dB_t = alpha
               + sum_{k=1..lags} beta_k dA_{t-k}
               + sum_{k=1..lags} gamma_k dB_{t-k}
               + (optional) beta0 dA_t
               + eps_t

    more acceptable because:
    - It avoids overstating leadership when B is autocorrelated by controlling for lagged dB.
    - Still easy to explain, and keeps HAC robust inference.

    Interpretation:
    - A "leads" B if the beta_k (k>=1) terms are jointly / individually significant.

    Returns
    -------
    statsmodels RegressionResultsWrapper
    """
    if lags < 1:
        raise ValueError("lags must be >= 1 for Granger-style regression")
    if hac_lags < 0:
        raise ValueError("hac_lags must be >= 0")

    df = _prepare_two_series_panel(prem_panel, col_a, col_b, timestamp_col=timestamp_col, diff=True)

    # lagged A terms (k=1..lags) for true "leading" content
    XA = pd.concat(
        [df[col_a].shift(k).rename(f"dA_lag{k}") for k in range(1, lags + 1)],
        axis=1,
    )

    # lagged B terms (k=1..lags) to control for B persistence
    XB = pd.concat(
        [df[col_b].shift(k).rename(f"dB_lag{k}") for k in range(1, lags + 1)],
        axis=1,
    )

    X_parts = [XA, XB]

    if include_contemporaneous_a:
        X_parts.insert(0, df[col_a].rename("dA_lag0"))

    X = pd.concat(X_parts, axis=1).dropna()
    y = df.loc[X.index, col_b]
    X = sm.add_constant(X)

    model = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags})
    return model


# -----------------------------
# 4) Robust VAR + IRF
# -----------------------------
def run_var_and_irf(
    prem_panel: pd.DataFrame,
    cols: Iterable[str],
    maxlags: int = 10,
    ic: str = "aic",
    irf_steps: int = 60,
    use_diffs: bool = True,
    min_std: float = 1e-6,
    timestamp_col: str = "timestamp",
) -> Tuple[object, object, List[str]]:
    """
    Robust VAR:
    - drops near-constant series (std < min_std)
    - optionally differences series to improve stationarity + reduce singular covariance risk
    - tries IC-based lag selection; falls back to lag=2 if selection fails

    Parameters
    ----------
    prem_panel : pd.DataFrame
        Must contain timestamp + cols.
    cols : iterable[str]
        Column names to include in VAR.
    maxlags : int
        Maximum lags considered in selection.
    ic : str
        Information criterion used by statsmodels VAR.fit (e.g., "aic", "bic", "hqic", "fpe").
    irf_steps : int
        Number of steps ahead for impulse responses.
    use_diffs : bool
        If True, VAR runs on first differences (Δ series).
    min_std : float
        Drop columns with std <= min_std.
    timestamp_col : str
        Timestamp column name.

    Returns
    -------
    fit : statsmodels VARResults
    irf : statsmodels IRAnalysis
    used_cols : list[str]
        Columns actually used after filtering.
    """
    cols = list(cols)
    if len(cols) < 2:
        raise ValueError("VAR needs at least 2 series in 'cols'.")

    if timestamp_col not in prem_panel.columns:
        raise ValueError(f"prem_panel must contain '{timestamp_col}' column.")

    df = prem_panel[[timestamp_col] + cols].dropna().copy()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], utc=True, errors="coerce")
    df = df.dropna(subset=[timestamp_col]).sort_values(timestamp_col)
    df = df.set_index(timestamp_col)

    # Drop near-constant columns
    stds = df.std(numeric_only=True)
    keep = [c for c in df.columns if (c in stds.index and float(stds[c]) > min_std)]
    dropped = [c for c in df.columns if c not in keep]
    if dropped:
        print(f"⚠ VAR: Dropping near-constant series: {dropped}")

    df = df[keep]
    if df.shape[1] < 2:
        raise ValueError("VAR needs at least 2 non-constant series after filtering.")

    # Difference to reduce nonstationarity + singularity risk
    if use_diffs:
        df = df.diff().dropna()

    # Fit VAR
    var = VAR(df)

    # Try IC-based lag selection; if it fails, fall back to lag=2
    try:
        fit = var.fit(maxlags=maxlags, ic=ic)
    except Exception as e:
        print(f"⚠ VAR lag selection failed ({type(e).__name__}: {e}). Falling back to lag=2.")
        fit = var.fit(maxlags=2)

    # IRF
    try:
        irf = fit.irf(irf_steps)
    except Exception as e:
        raise RuntimeError(f"IRF computation failed ({type(e).__name__}: {e}).") from e

    return fit, irf, keep
