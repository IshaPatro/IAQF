"""
Data Collection Script for IAQF Competition
Fetches historical candle data from Coinbase and Kraken via CCXT

Period: March 1-21, 2023 (UTC)
"""

import ccxt
import pandas as pd
import time
import os


class CryptoDataCollector:
    """
    Collects historical OHLCV data from cryptocurrency exchanges
    """

    def __init__(self):
        self.coinbase = ccxt.coinbase({
            "enableRateLimit": True,
        })

        self.kraken = ccxt.kraken({
            "enableRateLimit": True,
        })

        # Competition date range (UTC)
        self.start_date = "2023-03-01 00:00:00"
        self.end_date = "2023-03-21 23:59:59"

    @staticmethod
    def _to_utc_dt(s: str) -> pd.Timestamp:
        return pd.to_datetime(s, utc=True)

    def fetch_ohlcv(self, exchange, symbol, timeframe="1m", start_date=None, end_date=None):
        """
        Fetch OHLCV data for a specific trading pair.

        Returns DataFrame with columns: timestamp, open, high, low, close, volume
        """

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

        # Load markets once
        try:
            exchange.load_markets()
        except Exception as e:
            print(f"  Warning: could not load markets for {exchange.name}: {e}")

        # Guard against infinite loops
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

                # Keep only candles <= until (some exchanges return beyond)
                candles = [c for c in candles if c[0] <= until]
                if not candles:
                    break

                all_candles.extend(candles)

                # Advance since to just after last candle
                last_ts = candles[-1][0]
                next_since = last_ts + 60_000  # jump by 1 minute
                if next_since <= current_since:
                    # safety
                    next_since = current_since + 60_000
                current_since = next_since

                print(f"  Fetched {len(candles)} candles, total: {len(all_candles)}")

                time.sleep(exchange.rateLimit / 1000)

                # Stop if we've reached the end window
                if last_ts >= until:
                    break

            except Exception as e:
                msg = str(e)
                print(f"  Error: {msg}")

                # If symbol doesn't exist, stop retrying
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

        # Diagnostics BEFORE filtering
        print(f"  Raw range: {df['timestamp'].min()} to {df['timestamp'].max()} (rows={len(df)})")

        # Filter exactly to March 1–21 window
        df = df[(df["timestamp"] >= start_dt) & (df["timestamp"] <= end_dt)].copy()

        print(f"  Complete: {len(df)} candles\n")
        return df

    def collect_competition_data(self, base_asset="BTC", timeframe="1m"):
        """
        Collect data needed for IAQF.
        Coinbase: BTC/USD, BTC/USDC, BTC/USDT
        Kraken:   BTC/USD, BTC/USDT, BTC/USDC (if available)
        Drivers:  USDT/USD, USDC/USD (to check peg)
        """

        data_dict = {}

        # 1. Main Assets
        coinbase_symbols = [
            f"{base_asset}/USD",
            f"{base_asset}/USDC",
            f"{base_asset}/USDT",
            "USDT/USD", # Driver
            "USDC/USD", # Driver
        ]

        kraken_base = base_asset.upper()
        # Kraken sometimes uses XBT instead of BTC, but ccxt usually normalizes.
        # We'll try both standard and potential raw symbols if needed, 
        # but ccxt 'BTC/USD' should work if mapped correctly.
        kraken_symbols = [
            f"{kraken_base}/USD",
            f"{kraken_base}/USDT",
            f"{kraken_base}/USDC",
            "USDT/USD", # Driver
            "USDC/USD", # Driver
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
            # Check for alternative symbols if direct match missing
            target_symbol = symbol
            if symbol not in self.kraken.symbols:
                # Try XBT mapping manually if needed, though ccxt usually handles it.
                if 'BTC' in symbol:
                    alt_symbol = symbol.replace('BTC', 'XBT')
                    if alt_symbol in self.kraken.symbols:
                        target_symbol = alt_symbol
                        print(f"  Switching {symbol} -> {target_symbol}")
            
            if target_symbol in self.kraken.symbols:
                try:
                    df = self.fetch_ohlcv(self.kraken, target_symbol, timeframe=timeframe)
                    if not df.empty:
                        # Normalize key name to standard BTC even if we fetched XBT
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


def main():
    collector = CryptoDataCollector()

    print("Starting data collection for BTC...")
    print(f"Date range: {collector.start_date} to {collector.end_date}\n")

    data_dict = collector.collect_competition_data(base_asset="BTC", timeframe="1m")
    collector.save_data(data_dict, output_dir="data")

    print("\n" + "=" * 60)
    print("DATA COLLECTION SUMMARY")
    print("=" * 60)
    for key, df in data_dict.items():
        if df.empty:
            print(f"{key}: EMPTY")
        else:
            print(f"{key}: {len(df)} candles, {df['timestamp'].min()} to {df['timestamp'].max()}")

    return data_dict


if __name__ == "__main__":
    main()
