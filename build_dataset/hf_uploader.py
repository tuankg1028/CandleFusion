import os
import pandas as pd
from huggingface_hub import HfApi, create_repo
from datasets import Dataset, DatasetDict, Features, Value, Image
import shutil
from typing import Optional

def upload_to_hf(
    dataset_index_csv: str,
    chart_dir: str,
    repo_id: str,
    token: Optional[str] = None,
    private: bool = False,
    commit_message: str = "Upload candlestick dataset"
):
    """
    Upload the candlestick dataset to Hugging Face Hub.
    
    Args:
        dataset_index_csv (str): Path to the dataset index CSV
        chart_dir (str): Directory containing chart images
        repo_id (str): HF repository ID (e.g., "username/dataset-name")
        token (str, optional): HF token (if None, uses saved token)
        private (bool): Whether to create a private repository
        commit_message (str): Commit message for the upload
    """
    print(f"🚀 Uploading dataset to Hugging Face: {repo_id}")
    
    # Initialize HF API
    api = HfApi(token=token)
    
    # Create repository if it doesn't exist
    try:
        create_repo(
            repo_id=repo_id,
            token=token,
            private=private,
            repo_type="dataset",
            exist_ok=True
        )
        print(f"✅ Repository {repo_id} created/verified")
    except Exception as e:
        print(f"⚠️ Repository creation warning: {e}")
    
    # Load dataset index
    df = pd.read_csv(dataset_index_csv)
    print(f"📊 Loaded {len(df)} records from dataset index")
    
    # Prepare dataset features
    features = Features({
        'image': Image(),
        'text': Value('string'),
        'label': Value('string'),
        'next_close': Value('float64'),
        'image_path': Value('string')
    })
    
    # Process data for HF format
    def load_image_data():
        data = []
        for _, row in df.iterrows():
            image_path = row['image_path']
            if os.path.exists(image_path):
                data.append({
                    'image': image_path,
                    'text': row['text'],
                    'label': row['label'],
                    'next_close': row['next_close'],
                    'image_path': os.path.basename(image_path)
                })
            else:
                print(f"⚠️ Image not found: {image_path}")
        return data
    
    dataset_data = load_image_data()
    print(f"📦 Prepared {len(dataset_data)} records for upload")
    
    # Create HF dataset
    dataset = Dataset.from_list(dataset_data, features=features)
    
    # Split dataset (optional - you can adjust these ratios)
    train_test = dataset.train_test_split(test_size=0.2, seed=42)
    train_val = train_test['train'].train_test_split(test_size=0.1, seed=42)
    
    dataset_dict = DatasetDict({
        'train': train_val['train'],
        'validation': train_val['test'],
        'test': train_test['test']
    })
    
    print(f"📈 Dataset splits: Train={len(dataset_dict['train'])}, Val={len(dataset_dict['validation'])}, Test={len(dataset_dict['test'])}")
    
    # Push to hub
    try:
        dataset_dict.push_to_hub(
            repo_id=repo_id,
            token=token,
            commit_message=commit_message
        )
        print(f"✅ Dataset successfully uploaded to https://huggingface.co/datasets/{repo_id}")
    except Exception as e:
        print(f"❌ Upload failed: {e}")
        raise

def create_dataset_card(
    repo_id: str,
    symbol: str,
    interval: str,
    window: int,
    total_records: int,
    token: Optional[str] = None
):
    """
    Create a README.md dataset card for the HF repository.
    """
    card_content = f"""---
license: mit
task_categories:
- image-to-text
- text-classification
language:
- en
tags:
- finance
- trading
- candlestick
- time-series
size_categories:
- 1K<n<10K
---

# Candlestick Chart Dataset - {symbol}

## Dataset Description

This dataset contains candlestick chart images paired with textual descriptions and trading labels for the {symbol} trading pair.

### Dataset Summary

- **Symbol**: {symbol}
- **Interval**: {interval}
- **Window Size**: {window} candles per chart
- **Total Records**: {total_records}
- **Image Format**: PNG (candlestick charts with volume)
- **Text Format**: Structured candle data description
- **Labels**: Trading signal classification

### Dataset Structure

Each record contains:
- `image`: Candlestick chart image (PNG)
- `text`: Textual description of the last candle
- `label`: Trading signal/classification
- `next_close`: Next closing price for validation
- `image_path`: Original image filename

### Usage

```python
from datasets import load_dataset

dataset = load_dataset("{repo_id}")
```

### Citation

If you use this dataset, please cite:

```
@dataset{{candlestick_dataset_{symbol.lower()},
  title={{Candlestick Chart Dataset - {symbol}}},
  author={{CandleFusion}},
  year={{2024}},
  url={{https://huggingface.co/datasets/{repo_id}}}
}}
```
"""
    
    api = HfApi(token=token)
    api.upload_file(
        path_or_fileobj=card_content.encode(),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
        token=token
    )
    print(f"📝 Dataset card created for {repo_id}")
