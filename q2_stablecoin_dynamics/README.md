# IAQF Competition 2026 - Cross-Currency Dynamics Analysis

Team implementation for the IAQF 2026 Student Competition on cryptocurrency cross-currency dynamics during stablecoin regulation.

## Project Overview

This repository contains the analytical framework for studying cross-currency pricing and liquidity patterns in cryptocurrency markets during the March 2023 USDC depeg crisis, with implications for the GENIUS Act stablecoin regulation.

**Competition Period:** March 1-21, 2023  
**Focus:** BTC cross-currency dynamics across USDT, USDC, and USD quote currencies  
**Exchanges:** Binance, Coinbase

---
This project quantifies stablecoin dislocations during the March 2023 SVB episode by measuring BTC price differences across quote currencies (USD, USDT, USDC) and exchanges (Coinbase, Binance).

## What this produces
Running the pipeline generates:
- Premium time series plots (stablecoin vs USD)
- Cross-stablecoin spread plot (USDT vs USDC)
- Regime comparison table (normal vs SVB window)
- Depeg dynamics table (event counts, persistence, half-life)
- Advanced models:
  - Lead–Lag regressions (price discovery across exchanges)
  - VAR + Impulse Response Functions (dynamic spillovers)

All artifacts are saved in `outputs/`.

---



## Repository Structure

```
.
├── README.md                           # This file
├── requirements.txt                    # Python dependencies
├── PROJECT_STRATEGY_GUIDE.md          # Comprehensive project strategy (READ THIS FIRST!)
│
├── data_collection.py                  # Script to fetch historical data from exchanges
├── stablecoin_dynamics_analysis.py    # Core analysis framework for Question 2
├── run_question2_analysis.py          # Complete end-to-end analysis script
│
├── data/                              # Raw data (create with data_collection.py)
│   ├── BTC_USDT_binance.csv
│   ├── BTC_USDC_binance.csv
│   └── ...
│
└── outputs/                           # Generated results, figures, tables
    ├── q2_premium_timeseries.png
    ├── q2_cross_spread.png
    ├── q2_table1_exchange_stats.csv
    └── ...
```

## Quick Start

### 1. Installation

```bash
# Clone repository
git clone <your-repo-url>
cd iaqf-competition-2026

# Creating virtual environment 
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Collection

```bash
# Collects historical data from exchanges
python data_collection.py
```

This will:
- Fetch 1-minute OHLCV candles for March 1-21, 2023
- Download BTC/USDT, BTC/USDC, BTC/USD pairs
- Save to `data/` directory as CSV files

**Note:** Data collection may take 30-60 minutes due to API rate limits.

### 3. Run Analysis (Question 2: Stablecoin Dynamics)

```bash
# Run complete Question 2 analysis
python run_question2_analysis.py
```

This will:
- Load data from `data/` directory
- Calculate stablecoin premiums and cross-spreads
- Detect normal vs crisis regimes
- Analyze depeg dynamics
- Generate all figures and tables
- Save results to `outputs/`

### 4. Review Results

Check the `outputs/` folder for:
- **Figures:** Time series plots, distributions, volatility charts
- **Tables:** Summary statistics, regime comparisons, depeg metrics (CSV format)

## Detailed Usage


------------------------------------------------------------------------

## Models Implemented

-   Stablecoin premium pricing model
-   Lead--lag regression
-   Vector Autoregression (VAR)
-   AR persistence half‑life estimation
-   Stress regime detection
-   HAC‑robust inference

These econometric approaches align with empirical asset pricing
literature.

------------------------------------------------------------------------

## Key Findings Summary

-   USDC exhibited extreme crisis premiums (\>1000bps) on Binance
-   Significant volatility regime shift detected
-   Evidence of exchange price leadership
-   Persistence indicates liquidity fragmentation
-   Supports regulatory diversification arguments

------------------------------------------------------------------------

