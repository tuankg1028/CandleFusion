# dataset.py

import os
import sys
import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
from transformers import BertTokenizer, ViTImageProcessor

class CandlestickDataset(Dataset):
    def __init__(self, csv_path: str, image_size: int = 224):
        """
        Args:
            csv_path (str): Path to CSV with image_path, text, label
            image_size (int): Size to resize chart images to (default 224)
        """
        self.data = pd.read_csv(csv_path)

        # Validate required columns
        required_columns = ["image_path", "text", "label", "next_close"]
        for col in required_columns:
            if col not in self.data.columns:
                raise ValueError(f"Missing required column: {col} in {csv_path}")
            
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
        self.image_size = image_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # === Load and preprocess image ===
        image_path = row["image_path"]
        image = Image.open(image_path).convert("RGB")
        image_inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = image_inputs["pixel_values"].squeeze(0)  # (3, 224, 224)

        # === Tokenize text ===
        text = row["text"]
        text_inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=64  # can be adjusted
        )
        input_ids = text_inputs["input_ids"].squeeze(0)
        attention_mask = text_inputs["attention_mask"].squeeze(0)

        # === Label ===
        label = torch.tensor(row["label"], dtype=torch.long)
        next_close = torch.tensor(row["next_close"], dtype=torch.float)

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label": label,
            "next_close": next_close
        }