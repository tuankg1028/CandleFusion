# train.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import get_scheduler
from tqdm import tqdm
import os
import json

from dataset import CandlestickDataset
from model import CrossAttentionModel

def train(model, dataloader, val_loader=None, epochs=5, lr=2e-5, alpha=0.5, device="cuda", 
          push_to_hub=False, hub_model_id=None, hub_token=None, 
          early_stopping_patience=5, save_every_n_epochs=10):
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    total_steps = len(dataloader) * epochs
    scheduler = get_scheduler("linear", optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    loss_fn_cls = nn.CrossEntropyLoss()
    loss_fn_reg = nn.MSELoss()

    # Early stopping variables
    best_val_loss = float('inf')
    patience_counter = 0
    best_epoch = 0
    
    # Training history
    train_history = {
        'train_loss': [],
        'train_cls_loss': [],
        'train_reg_loss': [],
        'val_loss': [],
        'val_accuracy': [],
        'val_mse': []
    }

    print(f"🚀 Starting training for {epochs} epochs...")
    print(f"📊 Early stopping patience: {early_stopping_patience}")
    print(f"💾 Saving checkpoints every {save_every_n_epochs} epochs")

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
        avg_cls_loss = total_cls_loss / len(dataloader)
        avg_reg_loss = total_reg_loss / len(dataloader)
        
        # Store training metrics
        train_history['train_loss'].append(avg_loss)
        train_history['train_cls_loss'].append(avg_cls_loss)
        train_history['train_reg_loss'].append(avg_reg_loss)

        print(f"✅ Epoch {epoch+1} done | Total Loss: {avg_loss:.4f} | CLS: {avg_cls_loss:.4f} | REG: {avg_reg_loss:.4f}")

        # Validation and early stopping
        if val_loader:
            val_metrics = evaluate(model, val_loader, device, return_metrics=True)
            val_loss = val_metrics['loss']
            val_accuracy = val_metrics['accuracy']
            val_mse = val_metrics['mse']
            
            # Store validation metrics
            train_history['val_loss'].append(val_loss)
            train_history['val_accuracy'].append(val_accuracy)
            train_history['val_mse'].append(val_mse)
            
            print(f"📊 Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy*100:.2f}% | Val MSE: {val_mse:.4f}")
            
            # Check for best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch + 1
                patience_counter = 0
                
                # Save best model
                best_checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_val_loss': best_val_loss,
                    'val_accuracy': val_accuracy,
                    'val_mse': val_mse,
                    'train_history': train_history,
                    'config': {
                        'lr': lr,
                        'alpha': alpha,
                        'epochs': epochs,
                        'early_stopping_patience': early_stopping_patience
                    }
                }
                
                torch.save(best_checkpoint, "./checkpoints/best_model.pt")
                print(f"💾 New best model saved! (Val Loss: {best_val_loss:.4f})")
                
            else:
                patience_counter += 1
                print(f"⏳ No improvement for {patience_counter}/{early_stopping_patience} epochs")
                
                if patience_counter >= early_stopping_patience:
                    print(f"🛑 Early stopping triggered! Best model was at epoch {best_epoch}")
                    print(f"🏆 Best validation loss: {best_val_loss:.4f}")
                    break
        
        # Periodic checkpointing
        if (epoch + 1) % save_every_n_epochs == 0:
            periodic_checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': avg_loss,
                'train_history': train_history,
                'config': {
                    'lr': lr,
                    'alpha': alpha,
                    'epochs': epochs,
                    'early_stopping_patience': early_stopping_patience
                }
            }
            
            if val_loader:
                periodic_checkpoint['val_loss'] = val_loss
                periodic_checkpoint['val_accuracy'] = val_accuracy
                periodic_checkpoint['val_mse'] = val_mse
            
            checkpoint_path = f"./checkpoints/checkpoint_epoch_{epoch+1}.pt"
            torch.save(periodic_checkpoint, checkpoint_path)
            print(f"💾 Checkpoint saved: {checkpoint_path}")

    # Final model save (last epoch)
    final_checkpoint = {
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_history': train_history,
        'best_epoch': best_epoch,
        'best_val_loss': best_val_loss,
        'config': {
            'lr': lr,
            'alpha': alpha,
            'epochs': epochs,
            'early_stopping_patience': early_stopping_patience
        }
    }
    
    torch.save(final_checkpoint, "./checkpoints/candlefusion_model.pt")
    print("✅ Final model saved to ./checkpoints/candlefusion_model.pt")
    
    # Save training history as JSON
    with open("./checkpoints/training_history.json", "w") as f:
        json.dump(train_history, f, indent=2)
    print("📈 Training history saved to ./checkpoints/training_history.json")
    
    # Load and save best model state dict only (for inference)
    if os.path.exists("./checkpoints/best_model.pt"):
        best_checkpoint = torch.load("./checkpoints/best_model.pt")
        torch.save(best_checkpoint['model_state_dict'], "./checkpoints/best_model_state_dict.pt")
        print("🏆 Best model state dict saved to ./checkpoints/best_model_state_dict.pt")
    
    # Push to Hugging Face Hub if requested
    if push_to_hub and hub_model_id:
        try:
            from huggingface_hub import HfApi, Repository
            
            # Login to HF Hub
            if hub_token:
                from huggingface_hub import login
                login(token=hub_token)
            
            # Use best model for upload if available
            model_path = "./checkpoints/best_model_state_dict.pt" if os.path.exists("./checkpoints/best_model_state_dict.pt") else "./checkpoints/candlefusion_model.pt"
            
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

## Links
- 🔗 **GitHub Repository**: https://github.com/tuankg1028/CandleFusion
- 🚀 **Demo on Hugging Face Spaces**: https://huggingface.co/spaces/tuankg1028/candlefusion

## Training Results
- **Best Epoch**: {best_epoch}
- **Best Validation Loss**: {best_val_loss:.4f}
- **Training Epochs**: {len(train_history['train_loss'])}
- **Early Stopping**: {"Yes" if patience_counter >= early_stopping_patience else "No"}

## Architecture Overview

### Core Components
- **Text Encoder**: BERT (bert-base-uncased) for processing market sentiment and news
- **Vision Encoder**: Vision Transformer (ViT-base-patch16-224) for candlestick pattern recognition
- **Cross-Attention Fusion**: Multi-head attention mechanism (8 heads, 768 dim) for text-image integration
- **Dual Task Heads**: 
  - Classification head for trading signals (buy/sell/hold)
  - Regression head for next closing price prediction

### Data Flow
1. **Text Processing**: Market sentiment -> BERT -> CLS token (768-dim)
2. **Image Processing**: Candlestick charts -> ViT -> Patch embeddings (197 tokens, 768-dim each)
3. **Cross-Modal Fusion**: Text CLS as query, Image patches as keys/values -> Fused representation
4. **Dual Predictions**: 
   - Fused features -> Classification head -> Trading signal logits
   - Fused features -> Regression head -> Price forecast

### Model Specifications
- **Input Text**: Tokenized to max 64 tokens
- **Input Images**: Resized to 224x224 RGB
- **Hidden Dimension**: 768 (consistent across encoders)
- **Output Classes**: 3 (buy/sell/hold)
- **Dropout**: 0.3 in both heads

## Training Details
- **Epochs**: {epochs}
- **Learning Rate**: {lr}
- **Loss Function**: CrossEntropy (classification) + MSE (regression)
- **Loss Weight (alpha)**: {alpha} for regression term
- **Optimizer**: AdamW with linear scheduling
- **Early Stopping Patience**: {early_stopping_patience}

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
                "alpha": alpha,
                "best_epoch": best_epoch,
                "best_val_loss": float(best_val_loss),
                "early_stopping_patience": early_stopping_patience
            }
            
            # Create repository
            api = HfApi()
            api.create_repo(repo_id=hub_model_id, exist_ok=True)
            
            # Upload best model
            api.upload_file(
                path_or_fileobj=model_path,
                path_in_repo="pytorch_model.bin",
                repo_id=hub_model_id,
            )
            
            # Upload model card
            with open("./checkpoints/README.md", "w", encoding="utf-8") as f:
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
            
            # Upload training history
            api.upload_file(
                path_or_fileobj="./checkpoints/training_history.json",
                path_in_repo="training_history.json",
                repo_id=hub_model_id,
            )
            
            print(f"✅ Model pushed to Hugging Face Hub: https://huggingface.co/{hub_model_id}")
            
        except ImportError:
            print("❌ huggingface_hub not installed. Install with: pip install huggingface_hub")
        except Exception as e:
            print(f"❌ Error pushing to Hub: {e}")

def evaluate(model, dataloader, device="cuda", return_metrics=False):
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model.eval()

    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_forecasts = []
    all_targets = []
    total_loss = 0
    total_cls_loss = 0
    total_reg_loss = 0
    
    loss_fn_cls = nn.CrossEntropyLoss()
    loss_fn_reg = nn.MSELoss()
    alpha = 0.5  # Default alpha for evaluation

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
            
            # Calculate losses
            loss_cls = loss_fn_cls(logits, labels)
            loss_reg = loss_fn_reg(forecast, target_price)
            loss = loss_cls + alpha * loss_reg
            
            total_loss += loss.item()
            total_cls_loss += loss_cls.item()
            total_reg_loss += loss_reg.item()

            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())
            all_forecasts.extend(forecast.tolist())
            all_targets.extend(target_price.tolist())

    acc = correct / total
    avg_loss = total_loss / len(dataloader)
    forecast_mse = nn.MSELoss()(torch.tensor(all_forecasts), torch.tensor(all_targets)).item()
    
    if not return_metrics:
        print(f"📊 Evaluation Accuracy: {acc*100:.2f}%")
        print(f"📈 Forecast MSE: {forecast_mse:.4f}")
    
    if return_metrics:
        return {
            'accuracy': acc,
            'loss': avg_loss,
            'cls_loss': total_cls_loss / len(dataloader),
            'reg_loss': total_reg_loss / len(dataloader),
            'mse': forecast_mse
        }
