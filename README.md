# 📊 CandleFusion

**CandleFusion** is a deep learning pipeline that combines **Vision Transformer (ViT)** and **BERT** using **cross-attention** to classify candlestick charts and forecast prices. It supports end-to-end training directly from Binance OHLCV data and can push both datasets and trained models to Hugging Face Hub.

---

## 🔥 Features

- ✅ Automated data download from Binance (OHLCV)
- ✅ Candlestick chart image generation with parallel processing
- ✅ Textual description encoding using BERT
- ✅ Cross-attention fusion of BERT and ViT embeddings
- ✅ Dual-task learning: Classification + Price forecasting
- ✅ Modular pipeline split into dataset building and training phases
- ✅ Multiprocessing support for efficient chart generation
- ✅ **Hugging Face Hub integration** for both datasets and models
- ✅ **One-click dataset publishing** to HF Hub with automatic splits
- ✅ **One-click model publishing** to HF Hub with model cards

---

## 🧠 Architecture Overview

```text
Text (BERT) [CLS] ──┐
                    ▼
           Cross-Attention  →  Classifier → Bullish/Bearish (0/1/2)
                    ▲       →  Forecaster → Next Close Price
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
│   ├── hf_uploader.py      # 🆕 HuggingFace dataset uploader
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

# One-time Hugging Face login (optional, for uploading)
huggingface-cli login
```

### 2. Build Dataset

```bash
cd build_dataset

# Basic dataset creation
python main.py --symbol BTCUSDT --interval 1h --start "1 Jan, 2024" --window 30

# Build dataset AND upload to Hugging Face Hub
python main.py --symbol BTCUSDT --interval 1h --start "1 Jan, 2024" --window 30 \
               --upload_to_hf --hf_repo_id "your-username/btc-candlestick-dataset"

# Create private dataset on HF
python main.py --symbol ETHUSDT --interval 4h \
               --upload_to_hf --hf_repo_id "your-username/eth-dataset" --hf_private
```

**Dataset Options:**

- `--symbol`: Trading pair (e.g., BTCUSDT, ETHUSDT)
- `--interval`: Kline interval (1m, 5m, 15m, 1h, 1d)
- `--start`: Start date for data collection
- `--window`: Number of candles per chart image
- `--output_dir`: Base directory for outputs (default: ../data)

**🤗 HuggingFace Dataset Options:**

- `--upload_to_hf`: Enable pushing dataset to Hugging Face Hub
- `--hf_repo_id`: Your HF dataset repository (e.g., "username/dataset-name")
- `--hf_token`: HF token (optional if using `huggingface-cli login`)
- `--hf_private`: Create private HF dataset repository

### 3. Train Model

```bash
cd training

# Basic training
python main.py --batch_size 8 --epochs 5 --lr 2e-5

# Train and push model to Hugging Face Hub
python main.py --batch_size 8 --epochs 10 --lr 2e-5 \
               --push_to_hub --hub_model_id "your-username/candlefusion"

# Using environment variable for token
export HF_TOKEN="your_hf_token_here"
python main.py --push_to_hub --hub_model_id "your-username/candlefusion"
```

**Training Options:**

- `--data_dir`: Directory containing dataset (default: ../data)
- `--batch_size`: Training batch size
- `--epochs`: Number of training epochs
- `--lr`: Learning rate
- `--device`: Device for training (cuda/cpu)

**🤗 Hugging Face Model Options:**

- `--push_to_hub`: Enable pushing to Hugging Face Hub
- `--hub_model_id`: Your HF model repository (e.g., "username/candlefusion")
- `--hub_token`: HF token (optional if using `huggingface-cli login`)

---

## 📊 Dataset Format & HF Integration

### Local Dataset Structure

The pipeline generates:

1. **OHLCV CSV**: Raw candlestick data from Binance
2. **Chart Images**: PNG files of candlestick charts with technical indicators
3. **Dataset Index**: CSV linking images to text descriptions and labels

### Hugging Face Dataset Structure

When uploaded to HF Hub, datasets are automatically split into:

- **Train**: 72% of data for training
- **Validation**: 8% of data for validation
- **Test**: 20% of data for testing

Each record contains:
```python
{
    'image': PIL.Image,           # Candlestick chart
    'text': str,                  # OHLCV description
    'label': str,                 # Trading signal
    'next_close': float,          # Next closing price
    'image_path': str             # Original filename
}
```

### Loading HF Datasets

```python
from datasets import load_dataset

# Load your uploaded dataset
dataset = load_dataset("your-username/btc-candlestick-dataset")

# Access different splits
train_data = dataset['train']
test_data = dataset['test']

# Example usage
for example in train_data:
    image = example['image']
    text = example['text']
    label = example['label']
```

---

## 🔧 Model Details

- **Text Encoder**: BERT-base-uncased
- **Image Encoder**: ViT-base-patch16-224
- **Fusion**: Multi-head cross-attention (8 heads)
- **Classification**: 3-class (Bearish=0, Neutral=1, Bullish=2)
- **Regression**: Next closing price prediction
- **Input Text**: OHLCV numerical descriptions
- **Input Images**: 224x224 candlestick charts with volume and moving averages

---

## 🤗 Hugging Face Integration

### Dataset Upload Features

- **Automatic Splits**: Train/validation/test splits created automatically
- **Dataset Cards**: Auto-generated documentation with metadata
- **Image Handling**: Efficient storage and loading of chart images
- **Metadata Preservation**: All OHLCV data and labels maintained

### Model Upload Features

- **Model Card Generation**: Detailed description with training parameters
- **Config File**: Model architecture and hyperparameters
- **Model Weights**: PyTorch state dict as `pytorch_model.bin`

### Loading from Hub

```python
# Load dataset
from datasets import load_dataset
dataset = load_dataset("username/candlestick-dataset")

# Load model
from huggingface_hub import hf_hub_download
import torch
from model import CrossAttentionModel

model_path = hf_hub_download(repo_id="username/candlefusion", filename="pytorch_model.bin")
model = CrossAttentionModel()
model.load_state_dict(torch.load(model_path))
model.eval()
```

---

## 📈 Performance Tips

1. **Chart Generation**: Uses multiprocessing for faster image creation
2. **Memory Management**: Processes charts in batches to avoid memory issues
3. **GPU Training**: Automatically detects CUDA availability
4. **HF Upload**: Large datasets are uploaded efficiently with automatic chunking
5. **Data Versioning**: Use HF Hub for dataset and model versioning

---

## 🛠️ Customization

### Adding New Features

- Modify `utils/text_formatter.py` for different text representations
- Update `utils/label_generator.py` for different classification schemes
- Adjust `chart_generator.py` for different chart styles or indicators
- Customize `hf_uploader.py` for different dataset structures

### Model Architecture

- Change model dimensions in `training/model.py`
- Experiment with different pre-trained models
- Add regularization or additional layers

---

## 📝 Complete Example Workflow

```bash
# 1. Build dataset for Ethereum with 4-hour candles and upload to HF
cd build_dataset
python main.py --symbol ETHUSDT --interval 4h --start "1 Jun, 2023" --window 50 \
               --upload_to_hf --hf_repo_id "myuser/eth-4h-candlestick"

# 2. Train model using the local dataset and push to HF
cd ../training
python main.py --batch_size 16 --epochs 10 --lr 1e-5 --device cuda \
               --push_to_hub --hub_model_id "myuser/candlefusion-eth-4h"

# 3. Use the published dataset in other projects
python -c "
from datasets import load_dataset
dataset = load_dataset('myuser/eth-4h-candlestick')
print(f'Train: {len(dataset[\"train\"])} samples')
print(f'Test: {len(dataset[\"test\"])} samples')
"
```

---

## 🌐 Hub Repositories

Once uploaded to Hugging Face Hub, your assets will be available at:

**Datasets**: `https://huggingface.co/datasets/your-username/dataset-name`
- Automatic data viewer
- Download statistics
- Community discussions

**Models**: `https://huggingface.co/your-username/model-name`
- Model card with performance metrics
- Usage examples
- Integration with Inference API

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test with both local and HF Hub workflows
5. Submit a pull request

---

## 📄 License

This project is open source. Please check the license file for details.
