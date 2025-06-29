# train.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import get_scheduler
from tqdm import tqdm
import os

from dataset import CandlestickDataset
from model import CrossAttentionModel

def train(model, dataloader, val_loader=None, epochs=5, lr=2e-5, alpha=0.5, device="cuda", 
          push_to_hub=False, hub_model_id=None, hub_token=None):
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    total_steps = len(dataloader) * epochs
    scheduler = get_scheduler("linear", optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    loss_fn_cls = nn.CrossEntropyLoss()
    loss_fn_reg = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_cls_loss = 0
        total_reg_loss = 0

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")

        for batch in progress_bar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["label"].to(device)
            target_price = batch["next_close"].to(device)  # shape: (B,)

            optimizer.zero_grad()

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values
            )

            logits = outputs["logits"]
            forecast = outputs["forecast"].squeeze(1)  # shape: (B,)

            loss_cls = loss_fn_cls(logits, labels)
            loss_reg = loss_fn_reg(forecast, target_price)
            loss = loss_cls + alpha * loss_reg

            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            total_cls_loss += loss_cls.item()
            total_reg_loss += loss_reg.item()

            progress_bar.set_postfix(loss=loss.item(), cls=loss_cls.item(), reg=loss_reg.item())

        avg_loss = total_loss / len(dataloader)
        print(f"✅ Epoch {epoch+1} done | Total Loss: {avg_loss:.4f} | CLS: {total_cls_loss/len(dataloader):.4f} | REG: {total_reg_loss/len(dataloader):.4f}")

        if val_loader:
            evaluate(model, val_loader, device)

    torch.save(model.state_dict(), "./checkpoints/candlefusion_model.pt")
    print("✅ Model saved to ./checkpoints/candlefusion_model.pt")
    
    # Push to Hugging Face Hub if requested
    if push_to_hub and hub_model_id:
        try:
            from huggingface_hub import HfApi, Repository
            import json
            
            # Login to HF Hub
            if hub_token:
                from huggingface_hub import login
                login(token=hub_token)
            
            # Create model card and config
            model_card_content = f"""
---
license: apache-2.0
tags:
- pytorch
- candlestick
- financial-analysis
- multimodal
- bert
- vit
- cross-attention
- trading
- forecasting
---

# CandleFusion Model

A multimodal financial analysis model that combines textual market sentiment with visual candlestick patterns for enhanced trading signal prediction and price forecasting.

## Architecture Overview

### Core Components
- **Text Encoder**: BERT (bert-base-uncased) for processing market sentiment and news
- **Vision Encoder**: Vision Transformer (ViT-base-patch16-224) for candlestick pattern recognition
- **Cross-Attention Fusion**: Multi-head attention mechanism (8 heads, 768 dim) for text-image integration
- **Dual Task Heads**: 
  - Classification head for trading signals (buy/sell/hold)
  - Regression head for next closing price prediction

### Data Flow
1. **Text Processing**: Market sentiment → BERT → CLS token (768-dim)
2. **Image Processing**: Candlestick charts → ViT → Patch embeddings (197 tokens, 768-dim each)
3. **Cross-Modal Fusion**: Text CLS as query, Image patches as keys/values → Fused representation
4. **Dual Predictions**: 
   - Fused features → Classification head → Trading signal logits
   - Fused features → Regression head → Price forecast

### Model Specifications
- **Input Text**: Tokenized to max 64 tokens
- **Input Images**: Resized to 224×224 RGB
- **Hidden Dimension**: 768 (consistent across encoders)
- **Output Classes**: {2} (binary: bullish/bearish)
- **Dropout**: 0.3 in both heads

## Training Details
- **Epochs**: {epochs}
- **Learning Rate**: {lr}
- **Loss Function**: CrossEntropy (classification) + MSE (regression)
- **Loss Weight (α)**: {alpha} for regression term
- **Optimizer**: AdamW with linear scheduling

## Usage
```python
from model import CrossAttentionModel
import torch

# Load model
model = CrossAttentionModel()
model.load_state_dict(torch.load("pytorch_model.bin"))
model.eval()

# Inference
outputs = model(input_ids, attention_mask, pixel_values)
trading_signals = outputs["logits"]
price_forecast = outputs["forecast"]
```

## Performance
The model simultaneously optimizes for:
- **Classification Task**: Trading signal accuracy
- **Regression Task**: Price prediction MSE

This dual-task approach enables the model to learn both categorical market direction and continuous price movements.
"""
            
            config = {
                "model_type": "candlefusion",
                "architecture": "bert+vit+cross_attention",
                "num_labels": 3,
                "epochs": epochs,
                "learning_rate": lr,
                "alpha": alpha
            }
            
            # Create repository
            api = HfApi()
            api.create_repo(repo_id=hub_model_id, exist_ok=True)
            
            # Upload files
            api.upload_file(
                path_or_fileobj="./checkpoints/candlefusion_model.pt",
                path_in_repo="pytorch_model.bin",
                repo_id=hub_model_id,
            )
            
            # Upload model card
            with open("./checkpoints/README.md", "w") as f:
                f.write(model_card_content)
            api.upload_file(
                path_or_fileobj="./checkpoints/README.md",
                path_in_repo="README.md",
                repo_id=hub_model_id,
            )
            
            # Upload config
            with open("./checkpoints/config.json", "w") as f:
                json.dump(config, f, indent=2)
            api.upload_file(
                path_or_fileobj="./checkpoints/config.json",
                path_in_repo="config.json",
                repo_id=hub_model_id,
            )
            
            print(f"✅ Model pushed to Hugging Face Hub: https://huggingface.co/{hub_model_id}")
            
        except ImportError:
            print("❌ huggingface_hub not installed. Install with: pip install huggingface_hub")
        except Exception as e:
            print(f"❌ Error pushing to Hub: {e}")

def evaluate(model, dataloader, device="cuda"):
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model.eval()

    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_forecasts = []
    all_targets = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["label"].to(device)
            target_price = batch["next_close"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values
            )

            logits = outputs["logits"]
            forecast = outputs["forecast"].squeeze(1)

            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())
            all_forecasts.extend(forecast.tolist())
            all_targets.extend(target_price.tolist())

    acc = correct / total
    print(f"📊 Evaluation Accuracy: {acc*100:.2f}%")

    # Optional: print forecasting MSE
    forecast_mse = nn.MSELoss()(torch.tensor(all_forecasts), torch.tensor(all_targets)).item()
    print(f"📈 Forecast MSE: {forecast_mse:.4f}")
