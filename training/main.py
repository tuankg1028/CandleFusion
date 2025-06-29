import argparse
import pandas as pd
import os

from dataset import CandlestickDataset
from model import CrossAttentionModel
from train import train

from torch.utils.data import DataLoader, random_split

def main():
    parser = argparse.ArgumentParser(description="Train candlestick classifier using BERT + ViT")
    parser.add_argument("--data_dir", type=str, default="../data", help="Directory containing dataset")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--push_to_hub", action="store_true", help="Push model to Hugging Face Hub")
    parser.add_argument("--hub_model_id", type=str, help="Hugging Face model ID (e.g., 'username/candlefusion')")
    parser.add_argument("--hub_token", type=str, help="Hugging Face token (or set HF_TOKEN env var)")
    parser.add_argument("--early_stopping_patience", type=int, default=5, help="Early stopping patience (epochs)")
    parser.add_argument("--save_every_n_epochs", type=int, default=10, help="Save checkpoint every N epochs")
    args = parser.parse_args()

    # === Paths
    index_csv = os.path.join(args.data_dir, "dataset_index.csv")
    
    if not os.path.exists(index_csv):
        print(f"❌ Dataset index not found at {index_csv}")
        print("Please run the build_dataset script first.")
        return

    # === Create checkpoints directory
    os.makedirs("./checkpoints", exist_ok=True)

    # === Dataset & Loader
    dataset = CandlestickDataset(csv_path=index_csv)
    
    # Split dataset 80/20
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # === Model
    model = CrossAttentionModel()

    # === Get HF token from env if not provided
    hub_token = args.hub_token or os.getenv("HF_TOKEN")

    # === Train
    train(
        model, 
        train_loader, 
        val_loader=val_loader,
        epochs=args.epochs, 
        lr=args.lr, 
        device=args.device,
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id,
        hub_token=hub_token,
        early_stopping_patience=args.early_stopping_patience,
        save_every_n_epochs=args.save_every_n_epochs
    )

if __name__ == "__main__":
    main()
