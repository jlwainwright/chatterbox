#!/usr/bin/env python3
"""
Chatterbox TTS Installation Test
Tests that all components are properly installed and accessible
"""

def test_imports():
    """Test that all required modules can be imported"""
    print("🔍 Testing imports...")
    
    try:
        import torch
        print(f"   ✅ PyTorch {torch.__version__}")
        
        import torchaudio
        print(f"   ✅ TorchAudio {torchaudio.__version__}")
        
        import transformers
        print(f"   ✅ Transformers {transformers.__version__}")
        
        import librosa
        print(f"   ✅ Librosa {librosa.__version__}")
        
        import numpy as np
        print(f"   ✅ NumPy {np.__version__}")
        
        # Test Chatterbox imports
        from chatterbox.tts import ChatterboxTTS
        from chatterbox.vc import ChatterboxVC
        print("   ✅ Chatterbox TTS/VC classes")
        
        # Test watermarking
        import perth
        print("   ✅ Perth Watermarker")
        
        return True
        
    except ImportError as e:
        print(f"   ❌ Import failed: {e}")
        return False

def test_device_detection():
    """Test device detection and availability"""
    print("\n🖥️  Testing device detection...")
    
    import torch
    
    # Test CUDA
    if torch.cuda.is_available():
        print(f"   ✅ CUDA available: {torch.cuda.get_device_name()}")
    else:
        print("   ℹ️  CUDA not available")
    
    # Test MPS (Apple Silicon)
    if torch.backends.mps.is_available():
        print("   ✅ Apple MPS available")
    else:
        print("   ℹ️  Apple MPS not available")
    
    # CPU is always available
    print("   ✅ CPU available")

def test_model_info():
    """Test model information and repository access"""
    print("\n📦 Testing model repository access...")
    
    try:
        from huggingface_hub import hf_hub_download, list_repo_files
        
        # List available files
        repo_id = "ResembleAI/chatterbox"
        files = list_repo_files(repo_id)
        model_files = [f for f in files if f.endswith(('.safetensors', '.json', '.pt'))]
        
        print(f"   ✅ Repository accessible: {repo_id}")
        print(f"   📁 Model files found: {len(model_files)}")
        for file in model_files:
            print(f"      - {file}")
            
        return True
        
    except Exception as e:
        print(f"   ❌ Repository access failed: {e}")
        return False

def test_audio_processing():
    """Test basic audio processing capabilities"""
    print("\n🎵 Testing audio processing...")
    
    try:
        import librosa
        import numpy as np
        import torch
        
        # Create a simple test signal
        sr = 22050
        duration = 1.0  # 1 second
        t = np.linspace(0, duration, int(sr * duration))
        
        # Generate a simple sine wave
        frequency = 440  # A4 note
        test_signal = 0.3 * np.sin(2 * np.pi * frequency * t)
        
        # Test librosa processing
        mfcc = librosa.feature.mfcc(y=test_signal, sr=sr, n_mfcc=13)
        print(f"   ✅ Librosa MFCC extraction: {mfcc.shape}")
        
        # Test torch audio operations
        test_tensor = torch.from_numpy(test_signal).float()
        print(f"   ✅ PyTorch tensor conversion: {test_tensor.shape}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Audio processing test failed: {e}")
        return False

def test_text_processing():
    """Test text processing utilities"""
    print("\n📝 Testing text processing...")
    
    try:
        from chatterbox.tts import punc_norm
        
        # Test text normalization
        test_texts = [
            "hello world",
            "This is a test... with multiple punctuation!",
            "What about this — dash and 'quotes'?",
            ""
        ]
        
        for text in test_texts:
            normalized = punc_norm(text)
            print(f"   '{text}' → '{normalized}'")
        
        print("   ✅ Text normalization working")
        return True
        
    except Exception as e:
        print(f"   ❌ Text processing test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🎭 Chatterbox TTS Installation Test")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_device_detection,
        test_model_info,
        test_audio_processing,
        test_text_processing
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"   ❌ Test failed with exception: {e}")
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Chatterbox TTS is ready to use.")
        print("\n💡 Next steps:")
        print("   1. Run 'python example_tts.py' for basic TTS")
        print("   2. Try voice cloning with your own audio files")
        print("   3. Explore the comprehensive examples in README.md")
        print("   4. Join the community: https://discord.gg/rJq9cRJBJ6")
    else:
        print("⚠️  Some tests failed. Check the error messages above.")
        print("💬 For help: https://github.com/jlwainwright/chatterbox/issues")

if __name__ == "__main__":
    main()