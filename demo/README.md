---
title: CandleFusion Demo
emoji: 🕯️
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: "4.44.0"
app_file: app.py
pinned: false
---

# CandleFusion Web Demo

This directory contains a Gradio-based web demo for the CandleFusion model.

## Local Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run the demo:

```bash
python gradio_demo.py
```

## Hugging Face Spaces Deployment

This demo is designed to run on Hugging Face Spaces. To deploy:

1. Create a new Space on [Hugging Face Spaces](https://huggingface.co/spaces)
2. Choose "Gradio" as the SDK
3. Upload these files to your Space:
   - `app.py` (entry point for HF Spaces)
   - `requirements.txt`
   - `gradio_demo.py`
   - Any example images

The model will be automatically downloaded from `tuankg1028/candlefusion` on startup.

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
- **Auto Model Loading**: Downloads pre-trained model from Hugging Face Hub

## Notes

- The demo automatically downloads the model from `tuankg1028/candlefusion`
- Model is cached locally after first download
- Compatible with both local development and Hugging Face Spaces
