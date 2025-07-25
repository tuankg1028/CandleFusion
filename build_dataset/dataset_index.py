import os
import sys
import pandas as pd

# Add parent directory to path for utils access
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.text_formatter import format_candle_to_text
from utils.label_generator import generate_label

def create_dataset_index(ohlcv_df: pd.DataFrame, image_dir: str, window: int = 30, output_csv: str = "./data/dataset_index.csv"):
    """
    Create an index linking each chart image to its text + label.

    Args:
        ohlcv_df (pd.DataFrame): OHLCV dataframe
        image_dir (str): Directory where images are saved
        window (int): Number of candles per image
        output_csv (str): Path to save the CSV index
    """
    records = []
    total = len(ohlcv_df) - window + 1
    missing_images = 0

    for i in range(total):
        # Check if we have enough future data
        if i + window >= len(ohlcv_df):
            break

        chunk = ohlcv_df.iloc[i:i+window]
        last_candle = chunk.iloc[-1].to_dict()
        next_close = ohlcv_df.iloc[i + window]["close"] if i + window < len(ohlcv_df) else ohlcv_df.iloc[i + window - 1]["close"]

        # Validate image exists
        image_path = os.path.join(image_dir, f"candle_{i:04d}.png")
        if not os.path.exists(image_path):
            missing_images += 1
            continue

        # Validate data quality
        if any(pd.isna([last_candle['open'], last_candle['high'], 
                       last_candle['low'], last_candle['close'], next_close])):
            continue

        record = {
            "image_path": os.path.join(image_dir, f"candle_{i:04d}.png"),
            "text": format_candle_to_text(last_candle),
            "label": generate_label(last_candle),
            "next_close": next_close
        }
        records.append(record)

    df_index = pd.DataFrame(records)
    # Print label distribution
    label_counts = df_index['label'].value_counts().sort_index()
    print(f"📊 Label distribution: {dict(label_counts)}")
    
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df_index.to_csv(output_csv, index=False)
    print(f"✅ Dataset index saved to {output_csv} with {len(df_index)} records.")