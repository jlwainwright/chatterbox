#!/usr/bin/env python3
"""
Minimal TTS Demo - Quick test with reduced model loading time
"""

import sys
import time
import torch
import torchaudio as ta

def main():
    print("🎭 Chatterbox TTS Minimal Demo")
    print("=" * 40)
    
    # Show system info
    print(f"🐍 Python: {sys.version.split()[0]}")
    print(f"🔥 PyTorch: {torch.__version__}")
    print(f"📱 Device: {'MPS' if torch.backends.mps.is_available() else 'CPU'}")
    print()
    
    # Import Chatterbox
    print("📦 Importing Chatterbox TTS...")
    try:
        from chatterbox.tts import ChatterboxTTS
        print("✅ Import successful")
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return
    
    # Test device detection
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"🖥️  Selected device: {device}")
    print()
    
    # Show what would happen during model loading
    print("📥 Model loading process:")
    print("   1. Downloading model files from HuggingFace Hub (~2GB)")
    print("   2. Loading T3 (Text-to-Token) model")
    print("   3. Loading S3Gen (Speech Generation) model") 
    print("   4. Loading Voice Encoder")
    print("   5. Loading tokenizer")
    print("   6. Initializing Perth watermarker")
    print()
    
    # Test text processing without full model
    print("📝 Testing text processing...")
    from chatterbox.tts import punc_norm
    
    test_text = "hello world! this is a test of the text processing"
    normalized = punc_norm(test_text)
    print(f"   Input:  '{test_text}'")
    print(f"   Output: '{normalized}'")
    print("   ✅ Text normalization working")
    print()
    
    # Show example usage
    print("💡 Example usage (when model is loaded):")
    print("""
    # Load model (this step takes 30-60 seconds)
    model = ChatterboxTTS.from_pretrained(device=device)
    
    # Generate speech
    text = "Hello! This is Chatterbox TTS speaking."
    wav = model.generate(text)
    ta.save("output.wav", wav, model.sr)
    
    # Voice cloning
    wav = model.generate(
        text="New words in cloned voice",
        audio_prompt_path="reference_voice.wav",
        exaggeration=0.5
    )
    """)
    
    print("🚀 Ready to run full TTS generation!")
    print("   Run: python example_tts.py")
    print("   Or:  python quick_tts_demo.py")
    print()
    print("📚 More examples in the README:")
    print("   https://github.com/jlwainwright/chatterbox")

if __name__ == "__main__":
    main()