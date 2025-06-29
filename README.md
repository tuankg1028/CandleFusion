# Binance OHLCV Data Downloader

A Python tool to download historical OHLCV (Open, High, Low, Close, Volume) candlestick data from Binance exchange.

## Features

- Download historical candlestick data from Binance
- Support for multiple time intervals (1m, 1h, 1d, etc.)
- Clean data output with timestamp conversion
- CSV export functionality
- Command-line interface for easy automation

## Installation

1. Clone or download this repository
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Command Line Interface

Basic usage:

```bash
python binance_downloader.py
```

With custom parameters:

```bash
python binance_downloader.py --symbol ETHUSDT --interval 1d --start "1 Jan, 2023" --output ./data/eth_daily.csv
```

### Parameters

- `--symbol`: Trading pair symbol (default: BTCUSDT)
- `--interval`: Kline interval - 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M (default: 1h)
- `--start`: Start date in human-readable format (default: "1 Jan, 2024")
- `--output`: Output CSV file path (default: ./data/btc_ohlcv.csv)

### Examples

Download Bitcoin hourly data:

```bash
python binance_downloader.py --symbol BTCUSDT --interval 1h --start "1 Jan, 2024"
```

Download Ethereum daily data:

```bash
python binance_downloader.py --symbol ETHUSDT --interval 1d --start "1 Jun, 2023" --output ./data/eth_daily.csv
```

Download Solana 15-minute data:

```bash
python binance_downloader.py --symbol SOLUSDT --interval 15m --start "1 Dec, 2023" --output ./data/sol_15m.csv
```

## Output Format

The downloaded data is saved as a CSV file with the following columns:

- `open_time`: Timestamp of candle open time
- `open`: Opening price
- `high`: Highest price
- `low`: Lowest price
- `close`: Closing price
- `volume`: Trading volume

## Requirements

- Python 3.7+
- pandas
- python-binance

## Notes

- No API key required (uses public endpoints)
- Data availability depends on Binance's historical data retention
- Large date ranges may take some time to download
- The tool automatically creates the output directory if it doesn't exist

## License

This project is open source and available under the MIT License.
