# model.py

import torch
import torch.nn as nn
from transformers import BertModel, ViTModel

class CrossAttentionModel(nn.Module):
    def __init__(self, 
                 text_model_name="bert-base-uncased",
                 image_model_name="google/vit-base-patch16-224", 
                 hidden_dim=768,
                 num_classes=2):
        super().__init__()

        # Load pretrained models
        self.bert = BertModel.from_pretrained(text_model_name)
        self.vit = ViTModel.from_pretrained(image_model_name)

        # Cross-attention layer: BERT CLS attends to image patches
        self.cross_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=8, batch_first=True)

        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, input_ids, attention_mask, pixel_values):
        # === Text Encoder (BERT) ===
        text_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        text_cls = text_outputs.last_hidden_state[:, 0:1, :]  # (B, 1, H) → use [CLS] token only

        # === Image Encoder (ViT) ===
        image_outputs = self.vit(pixel_values=pixel_values)
        image_tokens = image_outputs.last_hidden_state[:, 1:, :]  # (B, N, H), skip [CLS] token

        # === Cross-Attention ===
        fused_cls, _ = self.cross_attention(
            query=text_cls,        # (B, 1, H)
            key=image_tokens,      # (B, N, H)
            value=image_tokens     # (B, N, H)
        )  # → (B, 1, H)

        fused_cls = fused_cls.squeeze(1)  # (B, H)

        # === Classifier ===
        logits = self.classifier(fused_cls)
        return logits
