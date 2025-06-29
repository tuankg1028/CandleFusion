# CandleFusion Web Demo

This directory contains a Gradio-based web demo for the CandleFusion model.

## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Make sure you have a trained model checkpoint at:

```
../training/checkpoints/candlefusion_model.pt
```

3. Run the demo:

```bash
python gradio_demo.py
```

## Usage

1. Upload a candlestick chart image
2. Enter market context or analysis text
3. Click "Analyze Chart" to get predictions

The demo will provide:

- Market direction prediction (Bullish/Bearish) with confidence scores
- Next closing price forecast

## Features

- **Real-time Predictions**: Upload images and get instant results
- **Dual Output**: Both classification and regression predictions
- **User-friendly Interface**: Clean Gradio interface with examples
- **Error Handling**: Graceful handling of invalid inputs
- **Mobile Responsive**: Works on both desktop and mobile devices

## Notes

- The demo uses the same preprocessing pipeline as the training code
- Model checkpoint is loaded automatically if available
- If no checkpoint is found, the demo will use an untrained model (for testing interface)
