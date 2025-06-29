import argparse
import pandas as pd
import os

from binance_downloader import get_ohlcv_data, save_to_csv
from chart_generator import save_chart_images
from dataset_index import create_dataset_index
from hf_uploader import upload_to_hf, create_dataset_card

def main():
    parser = argparse.ArgumentParser(description="Build candlestick dataset")
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="Symbol to fetch (e.g., BTCUSDT)")
    parser.add_argument("--interval", type=str, default="1h", help="Binance interval (e.g., 1h, 15m)")
    parser.add_argument("--start", type=str, default="1 Jan, 2024", help="Start date for data")
    parser.add_argument("--window", type=int, default=30, help="Candles per chart image")
    parser.add_argument("--output_dir", type=str, default="../data", help="Base directory for outputs")
    
    # HuggingFace arguments
    parser.add_argument("--upload_to_hf", action="store_true", help="Upload dataset to Hugging Face Hub")
    parser.add_argument("--hf_repo_id", type=str, help="HF repository ID (e.g., 'username/dataset-name')")
    parser.add_argument("--hf_token", type=str, help="HF token (optional if logged in)")
    parser.add_argument("--hf_private", action="store_true", help="Create private HF repository")
    
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
    
    # === Step 4: Upload to HuggingFace (optional)
    if args.upload_to_hf:
        if not args.hf_repo_id:
            print("❌ Error: --hf_repo_id is required when uploading to HF")
            return
        
        print("🚀 Starting upload to Hugging Face...")
        
        # Upload dataset
        upload_to_hf(
            dataset_index_csv=index_csv,
            chart_dir=chart_dir,
            repo_id=args.hf_repo_id,
            token=args.hf_token,
            private=args.hf_private,
            commit_message=f"Upload {args.symbol} candlestick dataset ({args.interval} interval)"
        )
        
        # Create dataset card
        df_index = pd.read_csv(index_csv)
        create_dataset_card(
            repo_id=args.hf_repo_id,
            symbol=args.symbol,
            interval=args.interval,
            window=args.window,
            total_records=len(df_index),
            token=args.hf_token
        )
        
        print(f"🎉 Dataset uploaded successfully to https://huggingface.co/datasets/{args.hf_repo_id}")
    
    print("✅ Dataset building complete!")

if __name__ == "__main__":
    main()
