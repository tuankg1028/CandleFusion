import gradio as gr
import torch
import sys
import os
from PIL import Image
import numpy as np
from huggingface_hub import hf_hub_download

# Import spaces for GPU support on Hugging Face Spaces
try:
    import spaces
    HF_SPACES = True
except ImportError:
    HF_SPACES = False
    # Create a dummy decorator if not on Spaces
    def spaces_gpu_decorator(func):
        return func
    spaces = type('spaces', (), {'GPU': spaces_gpu_decorator})()

# Add parent directory to path to import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.model import CrossAttentionModel
from transformers import BertTokenizer, ViTImageProcessor

class CandleFusionDemo:
    def __init__(self, model_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model from Hugging Face
        self.model = CrossAttentionModel()
        
        try:
            # Download model from Hugging Face Hub
            print("📥 Downloading model from Hugging Face...")
            model_file = hf_hub_download(
                repo_id="tuankg1028/candlefusion",
                filename="pytorch_model.bin",
                cache_dir="./model_cache"
            )
            
            # Load the downloaded model
            self.model.load_state_dict(torch.load(model_file, map_location=self.device))
            print(f"✅ Model loaded from Hugging Face: tuankg1028/candlefusion")
            
        except Exception as e:
            print(f"❌ Error loading model from Hugging Face: {str(e)}")
            print("⚠️ Using untrained model instead.")
        
        self.model.to(self.device)
        self.model.eval()
        
        # Initialize processors
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
        
        # Class labels
        self.class_labels = ["Bearish", "Bullish"]
    
    def preprocess_inputs(self, image, text):
        """Preprocess image and text inputs for the model"""
        # Process image
        if image is None:
            raise ValueError("Please upload a candlestick chart image")
        
        image = Image.fromarray(image).convert("RGB")
        image_inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = image_inputs["pixel_values"].to(self.device)
        
        # Process text
        if not text.strip():
            text = "Market analysis"  # Default text if empty
        
        text_inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=64
        )
        input_ids = text_inputs["input_ids"].to(self.device)
        attention_mask = text_inputs["attention_mask"].to(self.device)
        
        return pixel_values, input_ids, attention_mask
    
    @spaces.GPU
    def predict(self, image, text):
        """Make prediction using the model"""
        try:
            # Preprocess inputs
            pixel_values, input_ids, attention_mask = self.preprocess_inputs(image, text)
            
            # Model prediction
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values
                )
                
                logits = outputs["logits"]
                forecast = outputs["forecast"]
                
                # Get classification results
                probabilities = torch.softmax(logits, dim=1)
                predicted_class = torch.argmax(logits, dim=1).item()
                confidence = probabilities[0][predicted_class].item()
                
                # Get price forecast
                predicted_price = forecast.squeeze().item()
                
                # Format results
                classification_result = f"**Prediction:** {self.class_labels[predicted_class]}\n"
                classification_result += f"**Confidence:** {confidence:.2%}\n\n"
                classification_result += "**Class Probabilities:**\n"
                for i, (label, prob) in enumerate(zip(self.class_labels, probabilities[0])):
                    classification_result += f"- {label}: {prob:.2%}\n"
                
                forecast_result = f"**Predicted Next Close Price:** ${predicted_price:.2f}"
                
                return classification_result, forecast_result
                
        except Exception as e:
            error_msg = f"Error during prediction: {str(e)}"
            return error_msg, error_msg

def create_demo():
    """Create and launch the Gradio demo"""
    demo_instance = CandleFusionDemo()
    
    # Create Gradio interface
    with gr.Blocks(title="CandleFusion - Candlestick Chart Analysis", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🕯️ CandleFusion Demo
        
        Upload a candlestick chart image and provide market context to get:
        - **Market Direction Prediction** (Bullish/Bearish)
        - **Next Close Price Forecast**
        
        This model combines visual analysis of candlestick charts with textual market context using BERT + ViT architecture.
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📊 Input")
                
                image_input = gr.Image(
                    label="Candlestick Chart",
                    type="numpy",
                    height=300
                )
                
                text_input = gr.Textbox(
                    label="Market Context",
                    placeholder="Enter market analysis, news, or context (e.g., 'Strong volume with positive earnings report')",
                    lines=3,
                    value="Technical analysis of price action"
                )
                
                predict_btn = gr.Button("🔮 Analyze Chart", variant="primary")
                
                gr.Markdown("""
                ### 💡 Tips:
                - Upload clear candlestick chart images
                - Provide relevant market context
                - Charts should show recent price action
                """)
            
            with gr.Column(scale=1):
                gr.Markdown("### 📈 Results")
                
                classification_output = gr.Markdown(
                    value="Upload an image and click 'Analyze Chart' to see prediction"
                )
                
                forecast_output = gr.Markdown(
                    value=""
                )
        
        # Example section
        gr.Markdown("### 📚 Example")
        gr.Examples(
            examples=[
                ["examples/example_chart.png", "Strong bullish momentum with high volume"],
                ["examples/example_chart2.png", "Bearish reversal pattern forming"]
            ],
            inputs=[image_input, text_input],
            label="Try these examples:"
        )
        
        # Connect the prediction function
        predict_btn.click(
            fn=demo_instance.predict,
            inputs=[image_input, text_input],
            outputs=[classification_output, forecast_output]
        )
        
        gr.Markdown("""
        ---
        **Note:** This is a demo model. For production trading decisions, always consult with financial professionals and use additional analysis tools.
        """)
    
    return demo

def main():
    """Main function to launch the demo"""
    try:
        demo = create_demo()
        # Launch with server_name for compatibility on HF Spaces
        demo.launch(server_name="0.0.0.0")
    except Exception as e:
        print(f"Failed to launch Gradio demo: {e}")
        # Fallback launch with minimal configuration
        demo = create_demo()
        demo.launch(server_name="0.0.0.0")

if __name__ == "__main__":
    main()
