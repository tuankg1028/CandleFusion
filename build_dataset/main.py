import argparse
import pandas as pd
import os

from binance_downloader import get_ohlcv_data, save_to_csv
from chart_generator import save_chart_images
from dataset_index import create_dataset_index

def main():
    parser = argparse.ArgumentParser(description="Build candlestick dataset")
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="Symbol to fetch (e.g., BTCUSDT)")
    parser.add_argument("--interval", type=str, default="1h", help="Binance interval (e.g., 1h, 15m)")
    parser.add_argument("--start", type=str, default="1 Jan, 2024", help="Start date for data")
    parser.add_argument("--window", type=int, default=30, help="Candles per chart image")
    parser.add_argument("--output_dir", type=str, default="../data", help="Base directory for outputs")
    args = parser.parse_args()

    # === Paths
    ohlcv_path = os.path.join(args.output_dir, f"{args.symbol.lower()}_ohlcv.csv")
    chart_dir = os.path.join(args.output_dir, "charts")
    index_csv = os.path.join(args.output_dir, "dataset_index.csv")
    os.makedirs(chart_dir, exist_ok=True)

    # === Step 1: Download OHLCV
    df = get_ohlcv_data(args.symbol, args.interval, args.start)
    save_to_csv(df, ohlcv_path)

    # === Step 2: Generate Charts
    save_chart_images(df, window=args.window, output_dir=chart_dir)

    # === Step 3: Create Dataset Index
    create_dataset_index(df, image_dir=chart_dir, window=args.window, output_csv=index_csv)
    
    print("✅ Dataset building complete!")

if __name__ == "__main__":
    main()
