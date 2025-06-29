# train.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import get_scheduler
from tqdm import tqdm

from dataset import CandlestickDataset
from model import CrossAttentionModel

def train(model, dataloader, val_loader=None, epochs=5, lr=2e-5, device="cuda"):
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    total_steps = len(dataloader) * epochs
    scheduler = get_scheduler("linear", optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")

        for batch in progress_bar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()

            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values
            )

            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(dataloader)
        print(f"✅ Epoch {epoch+1} finished — Avg Loss: {avg_loss:.4f}")

        if val_loader:
            evaluate(model, val_loader, device)

    torch.save(model.state_dict(), "./checkpoints/cross_attention_model.pt")
    print("✅ Model saved at ./checkpoints/cross_attention_model.pt")


def evaluate(model, dataloader, device="cuda"):
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["label"].to(device)

            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values
            )

            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    acc = correct / total
    print(f"📊 Evaluation Accuracy: {acc*100:.2f}%")
