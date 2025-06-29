# binance_downloader.py

import os
import argparse
import pandas as pd
from binance.client import Client
from binance.enums import KLINE_INTERVAL_1MINUTE, KLINE_INTERVAL_1HOUR, KLINE_INTERVAL_1DAY

def get_ohlcv_data(symbol: str, interval: str, start_date: str, limit: int = 1000) -> pd.DataFrame:
    """
    Fetch OHLCV data from Binance and return a cleaned pandas DataFrame.

    Args:
        symbol (str): Trading pair symbol, e.g., 'BTCUSDT'
        interval (str): Kline interval (e.g., '1m', '1h', '1d')
        start_date (str): Start date string, e.g., '1 Jan, 2024'
        limit (int): Number of candles to fetch per request (default 1000)

    Returns:
        pd.DataFrame: DataFrame with columns [open_time, open, high, low, close, volume]
    """
    client = Client()  # No API key needed for public endpoints

    print(f"Fetching {symbol} candles from Binance...")
    raw = client.get_historical_klines(symbol, interval, start_date)

    df = pd.DataFrame(raw, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base", "taker_buy_quote", "ignore"
    ])

    # Convert time and numeric columns
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Keep only needed columns
    df = df[["open_time", "open", "high", "low", "close", "volume"]]

    print(f"Fetched {len(df)} candles.")
    return df


def save_to_csv(df: pd.DataFrame, output_path: str):
    """
    Save DataFrame to CSV file.

    Args:
        df (pd.DataFrame): DataFrame to save
        output_path (str): File path to write to
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Download OHLCV from Binance")
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="Trading pair symbol")
    parser.add_argument("--interval", type=str, default="1h", help="Kline interval (e.g., 1m, 1h, 1d)")
    parser.add_argument("--start", type=str, default="1 Jan, 2024", help="Start date (e.g., '1 Jan, 2024')")
    parser.add_argument("--output", type=str, default="./data/btc_ohlcv.csv", help="Output CSV file path")
    args = parser.parse_args()

    df = get_ohlcv_data(args.symbol, args.interval, args.start)
    save_to_csv(df, args.output)


if __name__ == "__main__":
    main()