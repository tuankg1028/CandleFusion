import argparse
import pandas as pd
import os

from binance_downloader import get_ohlcv_data, save_to_csv
from chart_generator import save_chart_images
from dataset_index import create_dataset_index
from dataset import CandlestickDataset
from model import CrossAttentionModel
from train import train

from torch.utils.data import DataLoader

def main():
    parser = argparse.ArgumentParser(description="Train candlestick classifier using BERT + ViT")
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="Symbol to fetch (e.g., BTCUSDT)")
    parser.add_argument("--interval", type=str, default="1h", help="Binance interval (e.g., 1h, 15m)")
    parser.add_argument("--start", type=str, default="1 Jan, 2024", help="Start date for data")
    parser.add_argument("--window", type=int, default=30, help="Candles per chart image")
    parser.add_argument("--output_dir", type=str, default="./data", help="Base directory for outputs")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    # === Paths
    ohlcv_path = os.path.join(args.output_dir, f"{args.symbol.lower()}_ohlcv.csv")
    chart_dir = os.path.join(args.output_dir, "charts")
    index_csv = os.path.join(args.output_dir, "dataset_index.csv")
    os.makedirs(chart_dir, exist_ok=True)

    # # === Step 1: Download OHLCV
    # df = get_ohlcv_data(args.symbol, args.interval, args.start)
    # save_to_csv(df, ohlcv_path)

    # # === Step 2: Generate Charts
    # save_chart_images(df, window=args.window, output_dir=chart_dir)

    # # === Step 3–5: Text, Label, Index
    # create_dataset_index(df, image_dir=chart_dir, window=args.window, output_csv=index_csv)

    # === Step 6: Dataset & Loader
    dataset = CandlestickDataset(csv_path=index_csv)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # === Step 7: Model
    model = CrossAttentionModel()

    # === Step 8: Train
    train(model, dataloader, epochs=args.epochs, lr=args.lr, device=args.device)


if __name__ == "__main__":
    main()


# Example command to run the script:
# python main.py --symbol BTCUSDT --interval 1h --start "1 Jan, 2024" --window 30 --batch_size 8 --epochs 5 --device cuda
