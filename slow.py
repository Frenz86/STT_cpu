import gradio as gr
import numpy as np
import torch
from transformers import pipeline
import soundfile as sf
import os
import warnings
import json
from datetime import datetime
warnings.filterwarnings("ignore")

# Initial configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

# Initialize the whisper model
model = pipeline(
    "automatic-speech-recognition",
    model="distil-whisper/distil-large-v2",
    device=DEVICE,
    torch_dtype=DTYPE
)

def save_transcription(text, timestamps=None):
    """
    Save transcription to a JSON file with metadata
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"transcription_{timestamp}.json"
    
    data = {
        "timestamp": timestamp,
        "text": text,
        "word_timestamps": timestamps,
    }
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    return output_file

def transcribe_audio(audio, return_timestamps=False):
    """
    Transcribe audio data to text with enhanced features.
    """
    if audio is None:
        return "", None
    
    try:
        # Get sample rate and data from audio tuple
        sample_rate, data = audio
        
        # Save temporary WAV file
        temp_path = "temp_audio.wav"
        sf.write(temp_path, data, sample_rate)
        
        # Perform transcription with timestamps
        result = model(
            temp_path,
            chunk_length_s=30,
            batch_size=24,
            return_timestamps=True
        )
        
        transcription = result["text"]
        timestamps = result.get("chunks", []) if return_timestamps else None
        
        # Save transcription with metadata
        output_file = save_transcription(transcription, timestamps)
        
        # Clean up temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        return transcription, output_file
        
    except Exception as e:
        print(f"Error during transcription: {str(e)}")
        return "", None

def create_interface():
    """
    Create the enhanced Gradio interface
    """
    with gr.Blocks() as demo:
        gr.Markdown("# Advanced Audio Transcription")
        
        with gr.Row():
            # Audio input component
            audio_input = gr.Audio(
                sources=["microphone", "upload"],
                type="numpy",
                label="Audio Input"
            )
            
            with gr.Column():
                # Text output component
                text_output = gr.Textbox(
                    label="Transcription",
                    placeholder="Transcription will appear here...",
                    lines=5
                )
                
                # File output component
                file_output = gr.File(
                    label="Download Transcription"
                )
        
        with gr.Row():
            # Advanced options
            timestamp_checkbox = gr.Checkbox(
                label="Include word timestamps",
                value=False
            )
            
            # Button to trigger transcription
            submit_btn = gr.Button("Transcribe", variant="primary")
        
        # Event handler
        submit_btn.click(
            fn=transcribe_audio,
            inputs=[audio_input, timestamp_checkbox],
            outputs=[text_output, file_output]
        )
        
        # Example usage
        gr.Examples(
            examples=[
                ["sample_audio.wav", False],  # Add your example files here
            ],
            inputs=[audio_input, timestamp_checkbox],
            outputs=[text_output, file_output],
        )
        
        gr.Markdown("""
        ## Features
        - Real-time audio transcription
        - Support for file upload and microphone recording
        - Optional word-level timestamps
        - Automatic save of transcriptions with metadata
        - Download transcriptions as JSON files
        
        ## Instructions
        1. Click the microphone button to record audio or upload an audio file
        2. Optionally enable word timestamps
        3. Click 'Transcribe' to convert the audio to text
        4. Download the full transcription with metadata as JSON
        
        Supported audio formats: WAV, MP3, OGG
        """)
        
    return demo

if __name__ == "__main__":
    print(f"Using device: {DEVICE}")
    demo = create_interface()
    demo.launch(share=True)