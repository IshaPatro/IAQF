import ccxt
import pandas as pd
import time
import os
import yfinance as yf

class CryptoDataCollector:
    def __init__(self):
        self.coinbase = ccxt.coinbase({
            "enableRateLimit": True,
        })
        self.kraken = ccxt.kraken({
            "enableRateLimit": True,
        })
        self.start_date = "2023-03-01 00:00:00"
        self.end_date = "2023-03-21 23:59:59"

    @staticmethod
    def _to_utc_dt(s: str) -> pd.Timestamp:
        return pd.to_datetime(s, utc=True)

    def fetch_ohlcv(self, exchange, symbol, timeframe="1m", start_date=None, end_date=None):
        if start_date is None:
            start_date = self.start_date
        if end_date is None:
            end_date = self.end_date

        start_dt = self._to_utc_dt(start_date)
        end_dt = self._to_utc_dt(end_date)

        since = int(start_dt.timestamp() * 1000)
        until = int(end_dt.timestamp() * 1000)

        all_candles = []
        current_since = since

        print(f"Fetching {symbol} from {exchange.name}...")
        try:
            exchange.load_markets()
        except Exception as e:
            print(f"  Warning: could not load markets for {exchange.name}: {e}")
        max_loops = 20000
        loops = 0

        while current_since < until and loops < max_loops:
            loops += 1
            try:
                candles = exchange.fetch_ohlcv(
                    symbol=symbol,
                    timeframe=timeframe,
                    since=current_since,
                    limit=1000
                )

                if not candles:
                    break

                candles = [c for c in candles if c[0] <= until]
                if not candles:
                    break

                all_candles.extend(candles)
                last_ts = candles[-1][0]
                next_since = last_ts + 60_000 
                if next_since <= current_since:
                    next_since = current_since + 60_000
                current_since = next_since

                print(f"  Fetched {len(candles)} candles, total: {len(all_candles)}")

                time.sleep(exchange.rateLimit / 1000)
                if last_ts >= until:
                    break

            except Exception as e:
                msg = str(e)
                print(f"  Error: {msg}")

                if "does not have market symbol" in msg:
                    print("  Symbol not available on this exchange. Skipping.\n")
                    break

                time.sleep(5)
                continue

        if loops >= max_loops:
            print("  Warning: hit max_loops; stopping to avoid infinite loop.")

        df = pd.DataFrame(all_candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
        if df.empty:
            print("  Complete: 0 candles (empty)\n")
            return df

        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)

        print(f"  Raw range: {df['timestamp'].min()} to {df['timestamp'].max()} (rows={len(df)})")

        df = df[(df["timestamp"] >= start_dt) & (df["timestamp"] <= end_dt)].copy()

        print(f"  Complete: {len(df)} candles\n")
        return df

    def collect_competition_data(self, base_asset="BTC", timeframe="1m"):
        data_dict = {}

        coinbase_symbols = [
            f"{base_asset}/USD",
            f"{base_asset}/USDC",
            f"{base_asset}/USDT",
            "USDT/USD", 
            "USDC/USD", 
        ]

        kraken_base = base_asset.upper()
        kraken_symbols = [
            f"{kraken_base}/USD",
            f"{kraken_base}/USDT",
            f"{kraken_base}/USDC",
            "USDT/USD", 
            "USDC/USD", 
        ]

        print("=" * 60)
        print("COLLECTING FROM COINBASE")
        print("=" * 60)
        self.coinbase.load_markets()
        for symbol in coinbase_symbols:
            try:
                if symbol in self.coinbase.symbols:
                    df = self.fetch_ohlcv(self.coinbase, symbol, timeframe=timeframe)
                    if not df.empty:
                        key = f"{symbol.replace('/', '_')}_coinbase"
                        data_dict[key] = df
                    else:
                        print(f"  Skipping {symbol} (no data)\n")
                else:
                    print(f"  Skipping {symbol} (not market symbol)\n")
            except Exception as e:
                print(f"Failed to fetch {symbol} from Coinbase: {e}\n")

        print("\n" + "=" * 60)
        print("COLLECTING FROM KRAKEN")
        print("=" * 60)
        try:
            self.kraken.load_markets()
        except Exception as e:
             print(f"Error loading Kraken markets: {e}")

        for symbol in kraken_symbols:
            target_symbol = symbol
            if symbol not in self.kraken.symbols:
                if 'BTC' in symbol:
                    alt_symbol = symbol.replace('BTC', 'XBT')
                    if alt_symbol in self.kraken.symbols:
                        target_symbol = alt_symbol
                        print(f"  Switching {symbol} -> {target_symbol}")
            
            if target_symbol in self.kraken.symbols:
                try:
                    df = self.fetch_ohlcv(self.kraken, target_symbol, timeframe=timeframe)
                    if not df.empty:
                        key_symbol = target_symbol.replace('XBT', 'BTC') 
                        key = f"{key_symbol.replace('/', '_')}_kraken"
                        data_dict[key] = df
                    else:
                        print(f"  Skipping {target_symbol} (no data)\n")
                except Exception as e:
                    print(f"Failed to fetch {target_symbol} from Kraken: {e}\n")
            else:
                 print(f"  Skipping {symbol} (not found in Kraken markets)\n")

        return data_dict


    def save_data(self, data_dict, output_dir="data"):
        os.makedirs(output_dir, exist_ok=True)

        for key, df in data_dict.items():
            filepath = os.path.join(output_dir, f"{key}.csv")
            df.to_csv(filepath, index=False)
            print(f"Saved {key} to {filepath}")

        print(f"\nAll data saved to {output_dir}/")


class YFinanceCryptoCollector:
    """
    Collects daily crypto data from Yahoo Finance for reliable historical analysis.
    Useful when exchange API historical data is limited.
    """
    def __init__(self):
        self.start_date = "2023-03-01"
        self.end_date = "2023-03-22"
        self.tickers = {
            "BTC": "BTC-USD",
            "USDC": "USDC-USD"
        }

    def collect_data(self):
        print(f"\nFetching Crypto data from yfinance ({self.start_date} to {self.end_date})...")
        data_dict = {}
        
        for name, ticker in self.tickers.items():
            print(f"  Fetching {ticker}...")
            df = yf.download(ticker, start=self.start_date, end=self.end_date, interval="1d", progress=False)
            
            if not df.empty:
                # Cleanup MultiIndex if present (yfinance usually returns MultiIndex for >1 ticker, but here 1 by 1)
                if isinstance(df.columns, pd.MultiIndex):
                     df.columns = df.columns.droplevel(1)
                
                # Ensure index is timezone naive or consistent
                df.index = df.index.tz_localize(None)
                data_dict[name] = df
                print(f"    Fetched {len(df)} rows.")
            else:
                print(f"    Warning: No data for {ticker}")
                
        return data_dict

    def save_data(self, data_dict, output_dir="data"):
        os.makedirs(output_dir, exist_ok=True)
        for name, df in data_dict.items():
            filename = f"{name}_yfinance.csv"
            filepath = os.path.join(output_dir, filename)
            df.to_csv(filepath)
            print(f"Saved {name} to {filepath}")


class GFCDataCollector:
    """
    Collects daily adjusted close prices for Gold and S&P 500 during the Global Financial Crisis.
    """
    def __init__(self):
        self.start_date = "2008-09-02"
        self.end_date = "2008-09-30" 
        self.tickers = {
            "Gold": "GC=F",
            "SP500": "^GSPC"
        }

    def collect_data(self):
        print(f"\nFetching GFC data from {self.start_date} to {self.end_date}...")
        
        data = yf.download(
            list(self.tickers.values()), 
            start=self.start_date, 
            end="2008-10-01", 
            interval="1d",
            progress=False
        )
        # Handle MultiIndex columns if present (common in new yfinance)
        if isinstance(data.columns, pd.MultiIndex):
             # Depending on structure, usually (Price, Ticker)
             # We want Adj Close or Close
             try:
                 data = data["Adj Close"]
             except KeyError:
                 data = data["Close"]
        
        # Now data columns should be tickers
        # Rename columns to friendly names
        # Invert tickers dict to map Ticker -> Name
        inv_tickers = {v: k for k, v in self.tickers.items()}
        data = data.rename(columns=inv_tickers)

        business_days = pd.bdate_range(start=self.start_date, end=self.end_date)
        clean_data = data.reindex(business_days).dropna()
        
        print(f"  Fetched {len(clean_data)} rows of aligned data.")
        return clean_data


    def save_data(self, df, filename="gfc_data.csv", output_dir="data"):
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, filename)
        df.to_csv(filepath)
        print(f"Saved GFC data to {filepath}")


def main():
    print("=== Crypto Data Collection (Challenge) ===")
    # Kept for completeness/competition reqs, but analysis will use reliable yfinance data
    try:
        crypto_collector = CryptoDataCollector()
        data_dict = crypto_collector.collect_competition_data()
        crypto_collector.save_data(data_dict)
    except Exception as e:
        print(f"Skipping Challenge Collector due to: {e}")

    print("\n=== Crypto Data Collection (Analysis/YFinance) ===")
    yf_collector = YFinanceCryptoCollector()
    yf_dict = yf_collector.collect_data()
    yf_collector.save_data(yf_dict)

    print("\n=== GFC Data Collection ===")
    gfc_collector = GFCDataCollector()
    gfc_df = gfc_collector.collect_data()
    gfc_collector.save_data(gfc_df)

if __name__ == "__main__":
    main()

