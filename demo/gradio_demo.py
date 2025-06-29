import gradio as gr
import torch
import sys
import os
from PIL import Image
import numpy as np

# Add parent directory to path to import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.model import CrossAttentionModel
from training.dataset import CandlestickDataset
from transformers import BertTokenizer, ViTImageProcessor

class CandleFusionDemo:
    def __init__(self, model_path="./training/checkpoints/candlefusion_model.pt"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model
        self.model = CrossAttentionModel()
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"✅ Model loaded from {model_path}")
        else:
            print(f"⚠️ Model checkpoint not found at {model_path}. Using untrained model.")
        
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
                    label="Market Direction",
                    value="Upload an image and click 'Analyze Chart' to see prediction"
                )
                
                forecast_output = gr.Markdown(
                    label="Price Forecast",
                    value=""
                )
        
        # Example section
        gr.Markdown("### 📚 Example")
        gr.Examples(
            examples=[
                ["example_chart.png", "Strong bullish momentum with high volume"],
                ["example_chart2.png", "Bearish reversal pattern forming"]
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
    demo = create_demo()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,  # Set to False if you don't want a public link
        debug=True
    )

if __name__ == "__main__":
    main()
