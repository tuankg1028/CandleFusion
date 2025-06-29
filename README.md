# 📊 CandleFusion

**CandleFusion** is a deep learning pipeline that combines **Vision Transformer (ViT)** and **BERT** using **cross-attention** to classify candlestick charts as **bullish (1)** or **bearish (0)**. It supports end-to-end training directly from Binance OHLCV data.

---

## 🔥 Features

- ✅ Automated data download from Binance (OHLCV)
- ✅ Candlestick chart image generation
- ✅ Textual description encoding using BERT
- ✅ Cross-attention fusion of BERT and ViT embeddings
- ✅ Binary classification: Bullish or Bearish
- ✅ Modular, end-to-end training CLI via `main.py`

---

## 🧠 Architecture Overview

```text
Text (BERT) [CLS] ──┐
                    ▼
           Cross-Attention  →  Classifier → 0 / 1
                    ▲
Chart (ViT) Patches ┘
```
