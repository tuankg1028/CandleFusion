rm -rf candlefusion-demo  # Remove any existing demo directory

# Create a new directory for the demo
mkdir candlefusion-demo

# From your CandleFusion project root
cp demo/app.py candlefusion-demo/
cp demo/requirements.txt candlefusion-demo/
cp demo/gradio_demo.py candlefusion-demo/
cp -r demo/examples candlefusion-demo/examples

# Copy the training module
cp -r training/ candlefusion-demo/training  # Copy the training module