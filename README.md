# 📊 CandleFusion

**CandleFusion** is a deep learning pipeline that combines **Vision Transformer (ViT)** and **BERT** using **cross-attention** to classify candlestick charts as **bullish (1)** or **bearish (0)**. It supports end-to-end training directly from Binance OHLCV data.

---

## 🔥 Features

- ✅ Automated data download from Binance (OHLCV)
- ✅ Candlestick chart image generation with parallel processing
- ✅ Textual description encoding using BERT
- ✅ Cross-attention fusion of BERT and ViT embeddings
- ✅ Binary classification: Bullish or Bearish
- ✅ Modular pipeline split into dataset building and training phases
- ✅ Multiprocessing support for efficient chart generation

---

## 🧠 Architecture Overview

```text
Text (BERT) [CLS] ──┐
                    ▼
           Cross-Attention  →  Classifier → 0 / 1
                    ▲
Chart (ViT) Patches ┘
```

---

## 📁 Project Structure

```
CandleFusion/
├── build_dataset/          # Dataset creation pipeline
│   ├── binance_downloader.py
│   ├── chart_generator.py
│   ├── dataset_index.py
│   └── main.py
├── training/               # Model training pipeline
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   └── main.py
├── utils/                  # Shared utilities
│   ├── __init__.py
│   ├── text_formatter.py
│   └── label_generator.py
├── data/                   # Generated datasets
├── charts/                 # Generated chart images
├── checkpoints/            # Saved model weights
└── requirements.txt
```

---

## 🚀 Quick Start

### 1. Installation

```bash
git clone <repository-url>
cd CandleFusion
pip install -r requirements.txt
```

### 2. Build Dataset

```bash
cd build_dataset
python main.py --symbol BTCUSDT --interval 1h --start "1 Jan, 2024" --window 30
```

**Options:**

- `--symbol`: Trading pair (e.g., BTCUSDT, ETHUSDT)
- `--interval`: Kline interval (1m, 5m, 15m, 1h, 1d)
- `--start`: Start date for data collection
- `--window`: Number of candles per chart image
- `--output_dir`: Base directory for outputs (default: ../data)

### 3. Train Model

```bash
cd training
python main.py --batch_size 8 --epochs 5 --lr 2e-5
```

**Options:**

- `--data_dir`: Directory containing dataset (default: ../data)
- `--batch_size`: Training batch size
- `--epochs`: Number of training epochs
- `--lr`: Learning rate
- `--device`: Device for training (cuda/cpu)

---

## 📊 Dataset Format

The pipeline generates:

1. **OHLCV CSV**: Raw candlestick data from Binance
2. **Chart Images**: PNG files of candlestick charts with technical indicators
3. **Dataset Index**: CSV linking images to text descriptions and labels

Example dataset index:

```csv
image_path,text,label
../data/charts/candle_0000.png,"Open: 45230.50, High: 45876.20, Low: 44892.10, Close: 45654.30, Volume: 1234567",1
../data/charts/candle_0001.png,"Open: 45654.30, High: 45999.80, Low: 45123.40, Close: 45321.90, Volume: 987654",0
```

---

## 🔧 Model Details

- **Text Encoder**: BERT-base-uncased
- **Image Encoder**: ViT-base-patch16-224
- **Fusion**: Multi-head cross-attention (8 heads)
- **Classification**: Binary (Bullish=1, Bearish=0)
- **Input Text**: OHLCV numerical descriptions
- **Input Images**: 224x224 candlestick charts with volume and moving averages

---

## 📈 Performance Tips

1. **Chart Generation**: Uses multiprocessing for faster image creation
2. **Memory Management**: Processes charts in batches to avoid memory issues
3. **GPU Training**: Automatically detects CUDA availability
4. **Data Augmentation**: Consider adding for better generalization

---

## 🛠️ Customization

### Adding New Features

- Modify `utils/text_formatter.py` for different text representations
- Update `utils/label_generator.py` for multi-class classification
- Adjust `chart_generator.py` for different chart styles or indicators

### Model Architecture

- Change model dimensions in `training/model.py`
- Experiment with different pre-trained models
- Add regularization or additional layers

---

## 📝 Example Usage

```bash
# Build dataset for Ethereum with 4-hour candles
cd build_dataset
python main.py --symbol ETHUSDT --interval 4h --start "1 Jun, 2023" --window 50

# Train with larger batch size and more epochs
cd training
python main.py --batch_size 16 --epochs 10 --lr 1e-5 --device cuda
```

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

## 📄 License

This project is open source. Please check the license file for details.
