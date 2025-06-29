# chart_generator.py

import os
import mplfinance as mpf
import pandas as pd

def save_chart_images(df: pd.DataFrame, window: int = 30, output_dir: str = "./charts"):
    """
    Generate candlestick chart images using rolling OHLCV windows.

    Args:
        df (pd.DataFrame): OHLCV DataFrame with open_time, open, high, low, close, volume
        window (int): Number of candles per chart image
        output_dir (str): Directory to save images
    """
    os.makedirs(output_dir, exist_ok=True)

    df = df.copy()
    df.set_index("open_time", inplace=True)

    total = len(df) - window + 1
    print(f"Generating {total} chart images (window={window})...")

    for i in range(total):
        chunk = df.iloc[i:i+window]

        filename = os.path.join(output_dir, f"candle_{i:04d}.png")
        mpf.plot(chunk,
                 type='candle',
                 style='charles',
                 volume=True,
                 mav=(3, 6),  # Optional: moving averages
                 savefig=dict(fname=filename, dpi=100, bbox_inches='tight', pad_inches=0.1),
                 tight_layout=True)
        
        if i % 100 == 0:
            print(f"Saved {i}/{total} charts...")

    print(f"✅ Finished saving {total} candlestick images to '{output_dir}'")


if __name__ == "__main__":
    # Example usage
    df = pd.read_csv("./data/btc_ohlcv.csv", parse_dates=["open_time"])
    save_chart_images(df, window=30, output_dir="./charts")
