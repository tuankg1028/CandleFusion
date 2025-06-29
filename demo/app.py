"""
CandleFusion Demo App for Hugging Face Spaces
Entry point for the Gradio demo
"""

import os
import sys

# Since HF Spaces runs from the demo directory, we need to add the parent directory
# to access the training modules
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Import and run the demo
try:
    from gradio_demo import main
    
    if __name__ == "__main__":
        main()
except Exception as e:
    print(f"Error launching demo: {e}")
    # Fallback: create a simple error page
    import gradio as gr
    
    def error_interface():
        return gr.Interface(
            fn=lambda x: f"Demo temporarily unavailable. Error: {str(e)}",
            inputs=gr.Textbox(label="Input"),
            outputs=gr.Textbox(label="Output"),
            title="CandleFusion Demo - Error"
        )
    
    error_demo = error_interface()
    error_demo.launch(server_name="0.0.0.0", server_port=7860)
