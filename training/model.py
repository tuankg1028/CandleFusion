# model.py

import torch
import torch.nn as nn
from transformers import BertModel, ViTModel

class CrossAttentionModel(nn.Module):
    def __init__(self, 
                 text_model_name="bert-base-uncased",
                 image_model_name="google/vit-base-patch16-224", 
                 hidden_dim=768,
                 num_classes=3):
        super().__init__()

        # Encoders
        self.bert = BertModel.from_pretrained(text_model_name)
        self.vit = ViTModel.from_pretrained(image_model_name)

        # Cross-Attention layer
        self.cross_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=8, batch_first=True)

        # Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )

        # Forecasting Head (regression)
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1)  # Predict next closing price
        )

    def forward(self, input_ids, attention_mask, pixel_values):
        # === Text Encoding ===
        text_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        text_cls = text_outputs.last_hidden_state[:, 0:1, :]  # (B, 1, H)

        # === Image Encoding ===
        image_outputs = self.vit(pixel_values=pixel_values)
        image_tokens = image_outputs.last_hidden_state[:, 1:, :]  # skip CLS token

        # === Cross-Attention ===
        fused_cls, _ = self.cross_attention(
            query=text_cls,
            key=image_tokens,
            value=image_tokens
        )  # (B, 1, H)

        fused_cls = fused_cls.squeeze(1)  # (B, H)

        # === Dual Heads ===
        logits = self.classifier(fused_cls)     # Classification
        forecast = self.regressor(fused_cls)    # Regression (next price)

        return {"logits": logits, "forecast": forecast}
