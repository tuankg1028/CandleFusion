# chart_generator.py

import os
import mplfinance as mpf
import pandas as pd
from multiprocessing import Pool, cpu_count
from functools import partial
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for speed

def _generate_single_chart(args):
    """Helper function to generate a single chart (for multiprocessing)"""
    i, chunk, output_dir = args
    filename = os.path.join(output_dir, f"candle_{i:04d}.png")
    
    mpf.plot(chunk,
             type='candle',
             style='charles',
             volume=True,
             mav=(3, 6),
             savefig=dict(fname=filename, dpi=100, bbox_inches='tight', pad_inches=0.1),
             tight_layout=True,
             show_nontrading=False,  # Skip non-trading periods for speed
             scale_padding=dict(left=0.3, top=0.8, right=0.3, bottom=0.8))
    
    return i

def save_chart_images(df: pd.DataFrame, window: int = 30, output_dir: str = "./charts", 
                     n_processes: int = None, batch_size: int = 1000):
    """
    Generate candlestick chart images using rolling OHLCV windows with parallel processing.

    Args:
        df (pd.DataFrame): OHLCV DataFrame with open_time, open, high, low, close, volume
        window (int): Number of candles per chart image
        output_dir (str): Directory to save images
        n_processes (int): Number of parallel processes (default: CPU count)
        batch_size (int): Number of charts to process in each batch
    """
    os.makedirs(output_dir, exist_ok=True)

    df = df.copy()
    df.set_index("open_time", inplace=True)

    total = len(df) - window + 1
    if n_processes is None:
        n_processes = min(cpu_count(), 8)  # Cap at 8 to avoid memory issues
    
    print(f"Generating {total} chart images (window={window}) using {n_processes} processes...")

    # Prepare data chunks for parallel processing
    def generate_batches():
        for batch_start in range(0, total, batch_size):
            batch_end = min(batch_start + batch_size, total)
            batch_args = []
            for i in range(batch_start, batch_end):
                chunk = df.iloc[i:i+window]
                batch_args.append((i, chunk, output_dir))
            yield batch_args, batch_start, batch_end

    completed = 0
    
    # Process in batches to manage memory
    for batch_args, batch_start, batch_end in generate_batches():
        with Pool(processes=n_processes) as pool:
            results = pool.map(_generate_single_chart, batch_args)
        
        completed += len(results)
        print(f"Saved {completed}/{total} charts...")

    print(f"✅ Finished saving {total} candlestick images to '{output_dir}'")

if __name__ == "__main__":
    # Example usage
    df = pd.read_csv("./data/btc_ohlcv.csv", parse_dates=["open_time"])
    save_chart_images(df, window=30, output_dir="./charts")
