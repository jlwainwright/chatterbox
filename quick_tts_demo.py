#!/usr/bin/env python3
"""
Quick TTS Demo - Test Chatterbox TTS installation and basic functionality
"""

import os
import sys
import time
import torch
import torchaudio as ta
from chatterbox.tts import ChatterboxTTS

def main():
    print("🎭 Chatterbox TTS Quick Demo")
    print("=" * 40)
    
    # Device detection
    if torch.cuda.is_available():
        device = "cuda"
        print(f"🚀 Using GPU: {torch.cuda.get_device_name()}")
    elif torch.backends.mps.is_available():
        device = "mps"
        print("🍎 Using Apple Silicon MPS")
    else:
        device = "cpu"
        print("💻 Using CPU")
    
    print(f"📱 Device: {device}")
    print()
    
    # Load model
    print("📥 Loading Chatterbox TTS model...")
    start_time = time.time()
    
    try:
        model = ChatterboxTTS.from_pretrained(device=device)
        load_time = time.time() - start_time
        print(f"✅ Model loaded successfully in {load_time:.1f} seconds")
        print(f"🎵 Sample rate: {model.sr} Hz")
        print()
        
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    # Generate speech samples
    demo_texts = [
        "Hello! This is Chatterbox TTS, your new AI voice companion.",
        "I can speak with natural emotion and amazing clarity.",
        "Thanks for trying the demo! Isn't voice synthesis incredible?"
    ]
    
    print("🎬 Generating speech samples...")
    
    for i, text in enumerate(demo_texts, 1):
        print(f"📝 Sample {i}: {text}")
        
        try:
            start_time = time.time()
            wav = model.generate(text)
            gen_time = time.time() - start_time
            
            # Calculate real-time factor
            audio_duration = len(wav.squeeze()) / model.sr
            rtf = gen_time / audio_duration
            
            # Save audio
            output_file = f"demo_sample_{i}.wav"
            ta.save(output_file, wav, model.sr)
            
            print(f"   ✅ Generated in {gen_time:.2f}s ({rtf:.1f}x realtime)")
            print(f"   🎵 Saved: {output_file} ({audio_duration:.1f}s audio)")
            print()
            
        except Exception as e:
            print(f"   ❌ Generation failed: {e}")
            print()
    
    print("🎉 Demo completed!")
    print("\n📁 Generated files:")
    for i in range(1, len(demo_texts) + 1):
        filename = f"demo_sample_{i}.wav"
        if os.path.exists(filename):
            print(f"   - {filename}")
    
    print("\n💡 Try the voice cloning examples in the README!")
    print("🔗 Visit: https://github.com/jlwainwright/chatterbox")

if __name__ == "__main__":
    main()