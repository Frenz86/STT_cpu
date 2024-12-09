import gradio as gr
import numpy as np
import torch
from transformers import pipeline
import soundfile as sf
import os
import warnings
warnings.filterwarnings("ignore")

# Configurazione iniziale
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

# Initialize the whisper model
model = pipeline("automatic-speech-recognition",
                model="distil-whisper/distil-large-v3",
                device=DEVICE,
                torch_dtype=DTYPE,
                )

def transcribe_audio(audio):
    """
    Transcribe audio data to text.
    """
    if audio is None:
        return ""
    
    try:
        # Get sample rate and data from audio tuple
        sample_rate, data = audio
        # Save temporary WAV file
        temp_path = "temp_audio.wav"
        sf.write(temp_path, data, sample_rate)
        
        # Perform transcription
        result = model(temp_path)
        transcription = result["text"]
        
        # Clean up temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
        return transcription
        
    except Exception as e:
        print(f"Error during transcription: {str(e)}")
        return ""

def create_interface():
    """
    Create the Gradio interface
    """
    with gr.Blocks() as demo:
        gr.Markdown("# Audio Transcription")
        
        with gr.Row():
            # Audio input component
            audio_input = gr.Audio(
                sources=["microphone", "upload"],
                type="numpy"
            )
            
            # Text output component
            text_output = gr.Textbox(
                label="Transcription",
                placeholder="Transcription will appear here...",
                lines=5
            )
        
        # Button to trigger transcription
        submit_btn = gr.Button("Transcribe")
        
        # Event handler
        submit_btn.click(
                        fn=transcribe_audio,
                        inputs=[audio_input],
                        outputs=[text_output]
                        )
        # Example usage
        gr.Examples(
            examples=[
                ["sample_audio.wav"],  # Aggiungi qui i tuoi file di esempio
            ],
            inputs=[audio_input],
            outputs=[text_output],
        )
        
        gr.Markdown("""
        ## Instructions
        1. Click the microphone button to record audio or upload an audio file
        2. Click 'Transcribe' to convert the audio to text and the transcription will appear in the text box
        
        """)
        
    return demo

if __name__ == "__main__":
    print(f"Using device: {DEVICE}")
    demo = create_interface()
    demo.launch(share=True)