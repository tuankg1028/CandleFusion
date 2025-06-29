import argparse
import pandas as pd
import os

from dataset import CandlestickDataset
from model import CrossAttentionModel
from train import train

from torch.utils.data import DataLoader

def main():
    parser = argparse.ArgumentParser(description="Train candlestick classifier using BERT + ViT")
    parser.add_argument("--data_dir", type=str, default="../data", help="Directory containing dataset")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    # === Paths
    index_csv = os.path.join(args.data_dir, "dataset_index.csv")
    
    if not os.path.exists(index_csv):
        print(f"❌ Dataset index not found at {index_csv}")
        print("Please run the build_dataset script first.")
        return

    # === Dataset & Loader
    dataset = CandlestickDataset(csv_path=index_csv)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # === Model
    model = CrossAttentionModel()

    # === Train
    train(model, dataloader, epochs=args.epochs, lr=args.lr, device=args.device)

if __name__ == "__main__":
    main()
