# Q2 Deliverable — Stablecoin Dynamics (IAQF 2026)

This folder is a **reproducible** pipeline for Question 2 (Stablecoin Dynamics). It produces **paper-ready tables + figures** in `outputs/` from 1-minute BTC candles on **Binance** and **Coinbase** (March 1–21, 2023).

## Outputs created
After running, `outputs/` should contain:
- `q2_premium_timeseries.png`
- `q2_cross_spread.png`
- `q2_table1_exchange_stats.csv`
- `q2_table2_regime_comparison.csv`
- `q2_table3_depeg_analysis.csv`
- `leadlag_USDT_binance_vs_coinbase.txt`
- `leadlag_USDC_binance_vs_coinbase.txt`
- `var_summary.txt`, `var_irf.png`, `var_series_used.txt`
- `q2_premium_panel.csv`

## Expected structure
```
q2_delivery/
  README.md
  requirements.txt
  run_question2_analysis.py
  stablecoin_dynamics_analysis.py
  q2_models.py
  data/
    coinbase/
      BTC_USD_coinbase.csv
      BTC_USDT_coinbase.csv
      BTC_USDC_coinbase.csv
    binance_clean/
      BTC_USD_binance.csv
      BTC_USDT_binance.csv
      BTC_USDC_binance.csv
  outputs/   (created by script)
```

## Setup + run
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

pip install -r requirements.txt
python run_question2_analysis.py
```

## Paths (avoid hard-coded C:\Users\...)
Inside scripts, prefer:
```python
from pathlib import Path
BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
COINBASE_DIR = DATA / "coinbase"
BINANCE_DIR = DATA / "binance_clean"
OUTPUTS = BASE / "outputs"
OUTPUTS.mkdir(exist_ok=True)
```

## Common issues
- **USDC_coinbase premium is all zeros** → likely a data mix-up (BTC_USDC_coinbase copied from BTC_USD_coinbase) or a merge issue.
- **VAR fails** when a series is constant → the pipeline drops near-constant series and writes `var_series_used.txt`.


