<img width="1200" alt="cb-big2" src="https://github.com/user-attachments/assets/bd8c5f03-e91d-4ee5-b680-57355da204d1" />

# Chatterbox TTS

[![Alt Text](https://img.shields.io/badge/listen-demo_samples-blue)](https://resemble-ai.github.io/chatterbox_demopage/)
[![Alt Text](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-sm.svg)](https://huggingface.co/spaces/ResembleAI/Chatterbox)
[![Alt Text](https://static-public.podonos.com/badges/insight-on-pdns-sm-dark.svg)](https://podonos.com/resembleai/chatterbox)
[![Discord](https://img.shields.io/discord/1377773249798344776?label=join%20discord&logo=discord&style=flat)](https://discord.gg/rJq9cRJBJ6)

_Made with ♥️ by <a href="https://resemble.ai" target="_blank"><img width="100" alt="resemble-logo-horizontal" src="https://github.com/user-attachments/assets/35cf756b-3506-4943-9c72-c05ddfa4e525" /></a>

We're excited to introduce Chatterbox, [Resemble AI's](https://resemble.ai) first production-grade open source TTS model. Licensed under MIT, Chatterbox has been benchmarked against leading closed-source systems like ElevenLabs, and is consistently preferred in side-by-side evaluations.

Whether you're working on memes, videos, games, or AI agents, Chatterbox brings your content to life. It's also the first open source TTS model to support **emotion exaggeration control**, a powerful feature that makes your voices stand out. Try it now on our [Hugging Face Gradio app.](https://huggingface.co/spaces/ResembleAI/Chatterbox)

If you like the model but need to scale or tune it for higher accuracy, check out our competitively priced TTS service (<a href="https://resemble.ai">link</a>). It delivers reliable performance with ultra-low latency of sub 200ms—ideal for production use in agents, applications, or interactive media.

## 📋 Table of Contents

- [Key Features](#-key-features)
- [Quick Start](#-quick-start)
- [CLI Demo Menu](#-cli-demo-menu)
- [Installation](#-installation)
- [Voice Cloning Guide](#-voice-cloning-guide)  
- [Comprehensive Demos](#-comprehensive-demos)
- [Parameter Tuning Guide](#-parameter-tuning-guide)
- [Voice Conversion](#-voice-conversion)
- [Web Interface](#-web-interface)
- [Performance & Benchmarks](#-performance--benchmarks)
- [Troubleshooting](#-troubleshooting)
- [API Reference](#-api-reference)
- [Supported Languages](#-supported-languages)
- [Watermarking](#-built-in-perth-watermarking-for-responsible-ai)
- [Citation](#-citation)

# 🚀 Key Features

- **State-of-the-Art Zero-Shot TTS**: Clone any voice from just a few seconds of audio
- **0.5B Parameter Llama Backbone**: Powerful transformer-based architecture for high-quality synthesis
- **Unique Emotion Exaggeration Control**: First open source TTS with controllable emotion intensity
- **Ultra-Stable Generation**: Alignment-informed inference prevents common TTS artifacts
- **Massive Training Data**: Trained on 0.5M hours of cleaned, high-quality audio data
- **Built-in Watermarking**: Responsible AI with imperceptible Perth watermarks
- **Voice Conversion**: Transform any audio to match a target voice
- **Multi-Platform Support**: CUDA, Apple Silicon (MPS), and CPU inference
- **Production Ready**: [Outperforms ElevenLabs](https://podonos.com/resembleai/chatterbox) in quality evaluations

# ⚡ Quick Start

Get started with Chatterbox in less than 5 minutes:

```python
import torchaudio as ta
from chatterbox.tts import ChatterboxTTS

# Automatic device detection (CUDA/MPS/CPU)
model = ChatterboxTTS.from_pretrained(device="cuda")

# Generate speech with built-in voice
text = "Welcome to Chatterbox, the future of open source text-to-speech!"
wav = model.generate(text)
ta.save("welcome.wav", wav, model.sr)

print("🎉 Success! Generated welcome.wav")
```

**What just happened?**
1. Downloaded the pre-trained model (~2GB) automatically from HuggingFace
2. Generated high-quality speech using the built-in voice
3. Applied imperceptible watermarking for responsible AI usage
4. Saved 22kHz audio ready for production use

# 🎭 CLI Demo Menu

**New!** Interactive command-line menu for exploring all Chatterbox TTS demos with guided assistance.

## ⚡ Launch the Menu

```bash
# Direct launch
python demo_menu.py

# Platform-specific launchers
./run_demos.sh        # macOS/Linux
run_demos.bat         # Windows
python run_demos.py   # Cross-platform
```

## 🎯 What You Get

The CLI menu provides a professional, guided experience with:

- **🎭 9 Interactive Demos** - From quick tests to full TTS generation
- **🛠️ System Information** - Automatic device detection and compatibility checking
- **📦 Automated Setup** - One-click virtual environment and dependency installation
- **🎨 Colored Interface** - Beautiful terminal UI with status indicators
- **⚡ Smart Execution** - Automatic virtual environment activation
- **🚨 Error Handling** - Helpful guidance when things go wrong

## 📋 Demo Categories

### 🔍 Testing & Setup (No Model Download)
- **📋 Installation Test** (~30 seconds) - Verify all dependencies
- **⚡ Minimal Demo** (~10 seconds) - Quick overview without model loading
- **🌐 Web Interface Test** (~15 seconds) - Test Gradio integration

### 🎵 Audio Generation (Requires Model Download)
- **🎬 Quick TTS Demo** (~2-5 minutes) - Full TTS with actual audio generation
- **📱 Example TTS** (~2-5 minutes) - Official basic example
- **🍎 Example for Mac** (~2-5 minutes) - Mac-optimized with device detection
- **🔄 Voice Conversion** (~2-5 minutes) - Transform audio to match target voice

### 🌐 Web Interfaces (Browser-Based)
- **🌐 Gradio TTS App** - Full browser-based TTS interface
- **🔄 Gradio VC App** - Browser-based voice conversion

## 🎛️ Menu Features

- **Smart Guidance** - Duration estimates and requirement descriptions
- **Confirmation Prompts** - For operations requiring downloads
- **System Detection** - Automatic CUDA/MPS/CPU detection
- **Help System** - Comprehensive help (press `h`)
- **Installation Assistant** - Automated environment setup (press `i`)

## 💡 Perfect For

- **First-time users** exploring Chatterbox capabilities
- **Developers** testing different configurations  
- **Researchers** comparing voice cloning approaches
- **Anyone** who wants guided access to all features

See [CLI_MENU_README.md](CLI_MENU_README.md) for complete documentation.

# 📦 Installation

## Option 1: PyPI Installation (Recommended)
```bash
pip install chatterbox-tts
```

## Option 2: Development Installation
```bash
# Create conda environment (recommended)
conda create -yn chatterbox python=3.11
conda activate chatterbox

# Clone and install
git clone https://github.com/resemble-ai/chatterbox.git
cd chatterbox
pip install -e .
```

## System Requirements
- **Python**: 3.9+ (3.11 recommended)
- **OS**: Linux, macOS, Windows
- **GPU**: NVIDIA GPU with CUDA (optional but recommended)
- **Memory**: 4GB RAM minimum, 8GB+ recommended
- **Storage**: 3GB for model weights

**Platform-Specific Notes:**
- **CUDA**: Full GPU acceleration with automatic mixed precision
- **Apple Silicon**: Native MPS support for M1/M2/M3 Macs (macOS 12.3+)
- **CPU**: Optimized CPU inference with automatic threading

# 🎭 Voice Cloning Guide

## Understanding Voice Cloning

Chatterbox uses zero-shot voice cloning, meaning it can replicate any voice from just a short audio sample without additional training. The system extracts speaker characteristics and applies them to generate new speech.

## Step 1: Prepare Your Reference Audio

### Audio Quality Guidelines
- **Duration**: 5-30 seconds (10-15 seconds optimal)
- **Format**: WAV, MP3, FLAC, or other common formats
- **Sample Rate**: Any (automatically resampled to 22kHz)
- **Quality**: Clear speech, minimal background noise
- **Content**: Natural conversational speech works best

### What Makes Good Reference Audio
✅ **Good Examples:**
- Conversational speech with natural intonation
- Consistent volume throughout
- Single speaker, no overlapping voices
- Emotional content matching desired output style

❌ **Avoid:**
- Heavy background music or noise
- Multiple speakers talking simultaneously
- Extremely monotone or robotic speech
- Very short clips (<3 seconds) or very long clips (>60 seconds)

## Step 2: Basic Voice Cloning

```python
import torchaudio as ta
from chatterbox.tts import ChatterboxTTS

# Load model
model = ChatterboxTTS.from_pretrained(device="cuda")

# Your reference audio
REFERENCE_AUDIO = "path/to/your/reference.wav"

# Text to synthesize
text = "This is my cloned voice speaking new words that were never in the original audio."

# Generate with voice cloning
wav = model.generate(
    text=text,
    audio_prompt_path=REFERENCE_AUDIO,
    exaggeration=0.5,  # Neutral emotion
    cfg_weight=0.5     # Balanced quality/style control
)

ta.save("cloned_voice.wav", wav, model.sr)
```

## Step 3: Advanced Voice Cloning Techniques

### Emotion Control
```python
# Subtle emotion (natural conversation)
wav_subtle = model.generate(text, audio_prompt_path=REFERENCE_AUDIO, exaggeration=0.3)

# Neutral emotion (default)
wav_neutral = model.generate(text, audio_prompt_path=REFERENCE_AUDIO, exaggeration=0.5)

# Enhanced emotion (more expressive)
wav_expressive = model.generate(text, audio_prompt_path=REFERENCE_AUDIO, exaggeration=0.8)

# Dramatic emotion (high intensity)
wav_dramatic = model.generate(text, audio_prompt_path=REFERENCE_AUDIO, exaggeration=1.2)
```

### Style and Pacing Control
```python
# Fast-paced, energetic delivery
wav_energetic = model.generate(
    text, 
    audio_prompt_path=REFERENCE_AUDIO,
    exaggeration=0.7,
    cfg_weight=0.3,    # Lower CFG for faster pacing
    temperature=0.9    # Higher temp for more variation
)

# Slow, deliberate delivery
wav_deliberate = model.generate(
    text,
    audio_prompt_path=REFERENCE_AUDIO, 
    exaggeration=0.4,
    cfg_weight=0.7,    # Higher CFG for slower pacing
    temperature=0.6    # Lower temp for consistency
)
```

### Batch Processing Multiple Voices
```python
reference_voices = [
    "voice1.wav",
    "voice2.wav", 
    "voice3.wav"
]

texts = [
    "Hello, this is voice one speaking.",
    "Greetings from voice number two!",
    "And here's the third voice saying hello."
]

for i, (voice_path, text) in enumerate(zip(reference_voices, texts)):
    wav = model.generate(text, audio_prompt_path=voice_path)
    ta.save(f"cloned_voice_{i+1}.wav", wav, model.sr)
```

## Step 4: Professional Voice Cloning Tips

### Quality Enhancement Techniques
```python
# High-quality professional output
wav_professional = model.generate(
    text,
    audio_prompt_path=REFERENCE_AUDIO,
    exaggeration=0.45,        # Slightly reduced for natural sound
    cfg_weight=0.6,           # Higher CFG for stability
    temperature=0.7,          # Moderate randomness
    repetition_penalty=1.3,   # Prevent repetitive patterns
    min_p=0.05               # Quality filtering
)
```

### Matching Original Speaker Style
```python
def analyze_and_clone(reference_path, text):
    """
    Analyze reference audio characteristics and optimize parameters
    """
    # Load and analyze reference audio
    ref_audio, sr = librosa.load(reference_path, sr=22050)
    
    # Simple energy analysis (you can extend this)
    energy = librosa.feature.rms(y=ref_audio)[0]
    avg_energy = np.mean(energy)
    
    # Adjust parameters based on analysis
    if avg_energy > 0.02:  # High energy speaker
        exaggeration = 0.6
        cfg_weight = 0.4
        temperature = 0.8
    else:  # Lower energy speaker
        exaggeration = 0.4
        cfg_weight = 0.6
        temperature = 0.7
    
    # Generate with optimized settings
    wav = model.generate(
        text,
        audio_prompt_path=reference_path,
        exaggeration=exaggeration,
        cfg_weight=cfg_weight,
        temperature=temperature
    )
    
    return wav
```

# 🎬 Comprehensive Demos

## Demo 1: Basic Text-to-Speech
```python
import torchaudio as ta
from chatterbox.tts import ChatterboxTTS
import torch

# Automatic device detection
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"  # Apple Silicon
else:
    device = "cpu"

print(f"Using device: {device}")

# Load model once for multiple generations
model = ChatterboxTTS.from_pretrained(device=device)

# Sample texts for different scenarios
demo_texts = {
    "news": "Breaking news: Scientists have discovered a new exoplanet located in the habitable zone of its star system.",
    "storytelling": "Once upon a time, in a kingdom far away, there lived a brave knight who embarked on an epic quest.",
    "technical": "The API endpoint accepts JSON payloads with authentication headers and returns structured data responses.",
    "conversational": "Hey there! How's your day going? I hope you're having a fantastic time exploring voice synthesis!",
    "dramatic": "The storm was approaching rapidly, with thunder echoing across the mountains and lightning illuminating the dark sky!"
}

# Generate each demo
for style, text in demo_texts.items():
    wav = model.generate(text)
    ta.save(f"demo_{style}.wav", wav, model.sr)
    print(f"✅ Generated demo_{style}.wav")
```

## Demo 2: Voice Cloning Comparison
```python
def voice_cloning_comparison():
    """Compare original voice with cloned versions"""
    
    # Reference audio (your input)
    REFERENCE = "your_voice_sample.wav"
    
    # Text not present in reference audio
    NEW_TEXT = "This text was never spoken by the original speaker, but will sound like them."
    
    # Generate variations
    variations = {
        "neutral": {"exaggeration": 0.5, "cfg_weight": 0.5},
        "expressive": {"exaggeration": 0.8, "cfg_weight": 0.4},
        "calm": {"exaggeration": 0.3, "cfg_weight": 0.7},
        "energetic": {"exaggeration": 0.7, "cfg_weight": 0.3}
    }
    
    for style, params in variations.items():
        wav = model.generate(
            NEW_TEXT,
            audio_prompt_path=REFERENCE,
            **params
        )
        ta.save(f"cloned_{style}.wav", wav, model.sr)
        print(f"✅ Generated cloned_{style}.wav")

voice_cloning_comparison()
```

## Demo 3: Emotion and Style Control
```python
def emotion_demo():
    """Demonstrate emotion exaggeration control"""
    
    REFERENCE = "emotional_reference.wav"
    TEXT = "I can't believe what just happened! This is absolutely incredible!"
    
    emotion_levels = {
        "subtle": 0.25,
        "natural": 0.5, 
        "enhanced": 0.75,
        "intense": 1.0,
        "extreme": 1.5
    }
    
    for emotion, level in emotion_levels.items():
        wav = model.generate(
            TEXT,
            audio_prompt_path=REFERENCE,
            exaggeration=level,
            cfg_weight=0.5 - (level - 0.5) * 0.2  # Adjust CFG based on exaggeration
        )
        ta.save(f"emotion_{emotion}.wav", wav, model.sr)
        print(f"✅ Generated emotion_{emotion}.wav (level: {level})")

emotion_demo()
```

## Demo 4: Long-Form Content Generation
```python
def long_form_demo():
    """Generate longer content with consistent voice"""
    
    REFERENCE = "narrator_voice.wav"
    
    # Chapter from a story
    long_text = """
    In the year 2045, artificial intelligence had become seamlessly integrated into daily life. 
    Dr. Sarah Chen stood in her laboratory, observing the latest breakthrough in neural networks. 
    The implications were staggering - this technology could revolutionize how humans communicate.
    
    As she reviewed the data, she realized that the boundary between human and artificial 
    intelligence was becoming increasingly blurred. The future she had once imagined 
    was now becoming reality, one algorithm at a time.
    """
    
    wav = model.generate(
        long_text,
        audio_prompt_path=REFERENCE,
        exaggeration=0.4,      # Subtle for narration
        cfg_weight=0.6,        # Stable for long content
        temperature=0.7,       # Consistent but natural
        repetition_penalty=1.4 # Prevent repetitive patterns
    )
    
    ta.save("long_form_narration.wav", wav, model.sr)
    print("✅ Generated long_form_narration.wav")

long_form_demo()
```

## Demo 5: Multi-Speaker Conversation
```python
def conversation_demo():
    """Simulate a conversation between different speakers"""
    
    speakers = {
        "alice": "alice_voice.wav",
        "bob": "bob_voice.wav", 
        "charlie": "charlie_voice.wav"
    }
    
    conversation = [
        ("alice", "Hi everyone! Thanks for joining today's meeting."),
        ("bob", "Happy to be here, Alice. What's on the agenda?"),
        ("charlie", "I'm excited to discuss the new project developments."),
        ("alice", "Great! Let's start with the quarterly review."),
        ("bob", "The numbers look very promising this quarter."),
        ("charlie", "I agree. The team has done excellent work.")
    ]
    
    all_audio = []
    
    for speaker, text in conversation:
        wav = model.generate(
            text,
            audio_prompt_path=speakers[speaker],
            exaggeration=0.5,
            cfg_weight=0.5
        )
        all_audio.append(wav.squeeze(0))
        ta.save(f"conversation_{speaker}_{len(all_audio)}.wav", wav, model.sr)
    
    # Combine all audio with brief pauses
    silence = torch.zeros(int(0.5 * model.sr))  # 0.5 second pause
    combined = torch.cat([item for pair in zip(all_audio, [silence] * len(all_audio)) for item in pair][:-1])
    ta.save("full_conversation.wav", combined.unsqueeze(0), model.sr)
    
    print("✅ Generated conversation with multiple speakers")

conversation_demo()
```

# 🎛️ Parameter Tuning Guide

## Core Parameters

### Exaggeration (0.25 - 2.0)
Controls emotional intensity and expressiveness.

```python
# Parameter exploration
exaggeration_guide = {
    0.25: "Very subtle, almost monotone",
    0.4:  "Natural conversation, slight emotion", 
    0.5:  "Balanced default, good for most content",
    0.7:  "Expressive, good for storytelling",
    1.0:  "Highly expressive, dramatic content",
    1.5:  "Very dramatic, use sparingly",
    2.0:  "Extreme expression, may cause artifacts"
}

for level, description in exaggeration_guide.items():
    print(f"Exaggeration {level}: {description}")
```

### CFG Weight (0.0 - 1.0)
Controls generation guidance and pacing.

```python
cfg_guide = {
    0.0: "No guidance - fastest, most random",
    0.2: "Minimal guidance - fast paced, varied",
    0.5: "Balanced - good default",
    0.7: "Strong guidance - slower, more controlled", 
    1.0: "Maximum guidance - slowest, most stable"
}
```

### Temperature (0.05 - 5.0)
Controls randomness in generation.

```python
temperature_guide = {
    0.1: "Very deterministic, may sound robotic",
    0.5: "Conservative, consistent output",
    0.8: "Balanced randomness (default)",
    1.2: "More varied, creative output",
    2.0: "High variation, may be unstable"
}
```

## Optimization Recipes

### Recipe 1: Professional Narration
```python
professional_params = {
    "exaggeration": 0.4,
    "cfg_weight": 0.6,
    "temperature": 0.7,
    "repetition_penalty": 1.3,
    "min_p": 0.05
}
```

### Recipe 2: Energetic Presentation
```python
energetic_params = {
    "exaggeration": 0.8,
    "cfg_weight": 0.3,
    "temperature": 0.9,
    "repetition_penalty": 1.2,
    "min_p": 0.03
}
```

### Recipe 3: Calm Meditation
```python
calm_params = {
    "exaggeration": 0.3,
    "cfg_weight": 0.7,
    "temperature": 0.6,
    "repetition_penalty": 1.4,
    "min_p": 0.08
}
```

### Recipe 4: Character Voice Acting
```python
character_params = {
    "exaggeration": 1.0,
    "cfg_weight": 0.4,
    "temperature": 1.0,
    "repetition_penalty": 1.1,
    "min_p": 0.02
}
```

## Parameter Interaction Guide

```python
def optimize_for_content(content_type, speaker_energy="medium"):
    """
    Automatically optimize parameters based on content type and speaker energy
    """
    base_configs = {
        "news": {"exaggeration": 0.4, "cfg_weight": 0.6, "temperature": 0.7},
        "story": {"exaggeration": 0.7, "cfg_weight": 0.4, "temperature": 0.8},
        "technical": {"exaggeration": 0.3, "cfg_weight": 0.7, "temperature": 0.6},
        "casual": {"exaggeration": 0.5, "cfg_weight": 0.5, "temperature": 0.8},
        "dramatic": {"exaggeration": 1.0, "cfg_weight": 0.3, "temperature": 0.9}
    }
    
    config = base_configs.get(content_type, base_configs["casual"])
    
    # Adjust for speaker energy
    energy_adjustments = {
        "low": {"exaggeration": -0.1, "cfg_weight": +0.1, "temperature": -0.1},
        "medium": {"exaggeration": 0.0, "cfg_weight": 0.0, "temperature": 0.0},
        "high": {"exaggeration": +0.1, "cfg_weight": -0.1, "temperature": +0.1}
    }
    
    adjustments = energy_adjustments.get(speaker_energy, energy_adjustments["medium"])
    
    for key, adjustment in adjustments.items():
        config[key] += adjustment
        # Clamp to valid ranges
        if key == "exaggeration":
            config[key] = max(0.25, min(2.0, config[key]))
        elif key == "cfg_weight":
            config[key] = max(0.0, min(1.0, config[key]))
        elif key == "temperature":
            config[key] = max(0.05, min(5.0, config[key]))
    
    return config

# Example usage
news_config = optimize_for_content("news", "low")
story_config = optimize_for_content("story", "high")
```

# 🔄 Voice Conversion

Transform existing audio to match a target voice while preserving the original content and timing.

## Basic Voice Conversion
```python
from chatterbox.vc import ChatterboxVC
import torchaudio as ta

# Load voice conversion model
vc_model = ChatterboxVC.from_pretrained(device="cuda")

# Convert audio to target voice
source_audio = "original_speech.wav"      # Audio to convert
target_voice = "target_speaker.wav"      # Voice to match

converted_wav = vc_model.generate(
    audio=source_audio,
    target_voice_path=target_voice
)

ta.save("voice_converted.wav", converted_wav, vc_model.sr)
```

## Advanced Voice Conversion Workflow
```python
def voice_conversion_pipeline():
    """Complete voice conversion workflow with quality control"""
    
    vc_model = ChatterboxVC.from_pretrained(device="cuda")
    
    # Multiple source files to convert
    source_files = ["speech1.wav", "speech2.wav", "speech3.wav"]
    target_voice = "celebrity_voice.wav"
    
    for i, source in enumerate(source_files):
        try:
            # Convert voice
            converted = vc_model.generate(
                audio=source,
                target_voice_path=target_voice
            )
            
            output_path = f"converted_{i+1}.wav"
            ta.save(output_path, converted, vc_model.sr)
            print(f"✅ Converted {source} -> {output_path}")
            
        except Exception as e:
            print(f"❌ Failed to convert {source}: {e}")

voice_conversion_pipeline()
```

## Voice Conversion Use Cases
- **Content Localization**: Convert narrator voice across different content
- **Character Consistency**: Maintain character voices across multiple recordings  
- **Privacy Protection**: Replace speaker identity while preserving speech content
- **Audio Restoration**: Upgrade old recordings with modern voice quality

# 🌐 Web Interface

## Gradio TTS Interface
Launch an interactive web interface for text-to-speech:

```bash
python gradio_tts_app.py
```

Features:
- Real-time text-to-speech generation
- Voice cloning with file upload
- Parameter adjustment sliders
- Audio playback and download
- Seed control for reproducible results

## Gradio Voice Conversion Interface
```bash
python gradio_vc_app.py
```

Features:
- Upload source and target audio files
- Real-time voice conversion
- Quality preview before download
- Batch processing support

## Custom Web Integration
```python
import gradio as gr
from chatterbox.tts import ChatterboxTTS

def create_custom_interface():
    model = ChatterboxTTS.from_pretrained("cuda")
    
    def tts_generate(text, ref_audio, exaggeration, cfg_weight):
        wav = model.generate(
            text=text,
            audio_prompt_path=ref_audio,
            exaggeration=exaggeration,
            cfg_weight=cfg_weight
        )
        return (model.sr, wav.squeeze(0).numpy())
    
    interface = gr.Interface(
        fn=tts_generate,
        inputs=[
            gr.Textbox(label="Text to speak", lines=3),
            gr.Audio(label="Reference voice", type="filepath"),
            gr.Slider(0.25, 2.0, value=0.5, label="Exaggeration"),
            gr.Slider(0.0, 1.0, value=0.5, label="CFG Weight")
        ],
        outputs=gr.Audio(label="Generated speech"),
        title="Custom Chatterbox Interface"
    )
    
    return interface

# Launch custom interface
if __name__ == "__main__":
    interface = create_custom_interface()
    interface.launch(share=True)
```

# 📊 Performance & Benchmarks

## Quality Comparisons
- **vs ElevenLabs**: Consistently preferred in A/B testing
- **vs Other Open Source**: Superior naturalness and stability
- **Emotion Control**: Unique capability among open source models

## Speed Benchmarks

| Hardware | Generation Speed | Model Load Time |
|----------|------------------|-----------------|
| RTX 4090 | ~15x realtime | 8-12 seconds |
| RTX 3080 | ~10x realtime | 10-15 seconds |
| M2 Max | ~6x realtime | 15-20 seconds |
| CPU (16 cores) | ~2x realtime | 25-30 seconds |

## Memory Usage
- **GPU Memory**: 2-4GB VRAM
- **System RAM**: 4-8GB recommended  
- **Model Size**: ~2GB download
- **Generation**: Scales with audio length

## Optimization Tips

### GPU Optimization
```python
# Enable mixed precision for RTX cards
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

# Batch processing for efficiency
texts = ["Text 1", "Text 2", "Text 3"]
reference = "reference.wav"

# Process in batches
batch_outputs = []
for text in texts:
    wav = model.generate(text, audio_prompt_path=reference)
    batch_outputs.append(wav)
```

### Memory Optimization
```python
import gc
import torch

# Clear cache between generations
def generate_with_cleanup(model, text, **kwargs):
    wav = model.generate(text, **kwargs)
    
    # Clear GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Force garbage collection
    gc.collect()
    
    return wav
```

# 🔧 Troubleshooting

## Common Issues and Solutions

### 1. Model Loading Issues
```python
Problem: "CUDA out of memory" during model loading
Solution: 
- Reduce batch size or use CPU: device="cpu"
- Close other GPU applications
- Use torch.cuda.empty_cache() before loading

Problem: "MPS not available" on Mac
Solution:
- Update to macOS 12.3+
- Fallback to CPU will be automatic
- Check PyTorch MPS installation
```

### 2. Audio Quality Issues
```python
Problem: Robotic or unnatural speech
Solutions:
- Increase temperature (0.8-1.0)
- Adjust exaggeration (try 0.4-0.6)
- Use higher quality reference audio
- Check reference audio length (10-15 seconds optimal)

Problem: Speech too fast or slow
Solutions:
- Fast speech: increase cfg_weight (0.6-0.8)
- Slow speech: decrease cfg_weight (0.3-0.4)
- Adjust exaggeration inversely
```

### 3. Voice Cloning Problems
```python
Problem: Cloned voice doesn't match reference
Solutions:
- Use longer reference audio (10-15 seconds)
- Ensure reference is clean, single speaker
- Try different exaggeration values
- Check reference audio quality

Problem: Artifacts or distortion
Solutions:
- Lower exaggeration (<0.5)
- Increase cfg_weight (>0.5)
- Use lower temperature (<0.8)
- Check input audio format
```

### 4. Installation Issues
```bash
# PyTorch compatibility
pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118

# CUDA version mismatch
nvidia-smi  # Check CUDA version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Apple Silicon issues
pip install torch torchvision torchaudio
# Ensure MPS support: torch.backends.mps.is_available()
```

## Debug Mode
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable verbose model output
model = ChatterboxTTS.from_pretrained(device="cuda")
# Will show detailed loading and generation logs
```

## Performance Profiling
```python
import time
import torch

def profile_generation():
    model = ChatterboxTTS.from_pretrained("cuda")
    text = "This is a test for performance profiling."
    
    # Warmup
    _ = model.generate(text)
    
    # Profile multiple runs
    times = []
    for i in range(10):
        start = time.time()
        with torch.inference_mode():
            wav = model.generate(text)
        end = time.time()
        times.append(end - start)
    
    print(f"Average: {np.mean(times):.2f}s")
    print(f"Min: {np.min(times):.2f}s") 
    print(f"Max: {np.max(times):.2f}s")

profile_generation()
```

# 📚 API Reference

## ChatterboxTTS Class

### Methods

#### `from_pretrained(device: str) -> ChatterboxTTS`
Load pre-trained model from HuggingFace Hub.

**Parameters:**
- `device`: "cuda", "mps", or "cpu"

#### `from_local(ckpt_dir: str, device: str) -> ChatterboxTTS`
Load model from local checkpoint directory.

#### `generate(text: str, **kwargs) -> torch.Tensor`
Generate speech from text.

**Parameters:**
- `text`: Input text to synthesize
- `audio_prompt_path`: Path to reference audio (optional)
- `exaggeration`: Emotion intensity (0.25-2.0, default: 0.5)
- `cfg_weight`: Generation guidance (0.0-1.0, default: 0.5)  
- `temperature`: Sampling randomness (0.05-5.0, default: 0.8)
- `repetition_penalty`: Prevent repetition (1.0-2.0, default: 1.2)
- `min_p`: Minimum probability filtering (0.0-1.0, default: 0.05)
- `top_p`: Top-p sampling (0.0-1.0, default: 1.0)

**Returns:**
- `torch.Tensor`: Generated audio waveform at 22kHz

#### `prepare_conditionals(wav_fpath: str, exaggeration: float)`
Pre-compute conditioning from reference audio.

## ChatterboxVC Class

#### `generate(audio: str, target_voice_path: str) -> torch.Tensor`
Convert audio to match target voice.

**Parameters:**
- `audio`: Path to source audio
- `target_voice_path`: Path to target voice reference

**Returns:**
- `torch.Tensor`: Voice-converted audio

## Utility Functions

#### `punc_norm(text: str) -> str`
Normalize punctuation and formatting for optimal TTS output.

# 🌍 Supported Languages

**Currently Supported:**
- English (Native support, trained on 500k hours)

**Coming Soon:**
- Spanish (Q2 2025)
- French (Q2 2025) 
- German (Q3 2025)
- Mandarin Chinese (Q3 2025)

**Community Contributions:**
We welcome community contributions for additional language support. See our [contributing guidelines](CONTRIBUTING.md) for details on training data requirements and model adaptation.

# 🛡️ Built-in Perth Watermarking for Responsible AI

Every audio file generated by Chatterbox includes [Resemble AI's Perth (Perceptual Threshold) Watermarker](https://github.com/resemble-ai/perth) - imperceptible neural watermarks that survive MP3 compression, audio editing, and common manipulations while maintaining nearly 100% detection accuracy.

## Why Watermarking Matters
- **Content Authentication**: Verify AI-generated content
- **Misuse Prevention**: Detect unauthorized synthetic media
- **Transparency**: Clear labeling of AI-generated audio
- **Industry Standard**: Following best practices for responsible AI

## Watermark Detection

### Basic Detection
```python
import perth
import librosa

AUDIO_PATH = "generated_audio.wav"

# Load the watermarked audio
watermarked_audio, sr = librosa.load(AUDIO_PATH, sr=None)

# Initialize watermarker (same as used for embedding)
watermarker = perth.PerthImplicitWatermarker()

# Extract watermark
watermark = watermarker.get_watermark(watermarked_audio, sample_rate=sr)
print(f"Extracted watermark: {watermark}")
# Output: 0.0 (no watermark) or 1.0 (watermarked)
```

### Batch Watermark Detection
```python
def detect_watermarks_batch(audio_files):
    """Detect watermarks in multiple audio files"""
    watermarker = perth.PerthImplicitWatermarker()
    results = {}
    
    for audio_file in audio_files:
        try:
            audio, sr = librosa.load(audio_file, sr=None)
            watermark = watermarker.get_watermark(audio, sample_rate=sr)
            results[audio_file] = {
                'watermarked': watermark > 0.5,
                'confidence': watermark
            }
        except Exception as e:
            results[audio_file] = {'error': str(e)}
    
    return results

# Usage
files = ["audio1.wav", "audio2.wav", "audio3.wav"]
detection_results = detect_watermarks_batch(files)
for file, result in detection_results.items():
    print(f"{file}: {result}")
```

### Watermark Survival Testing
```python
import librosa
import soundfile as sf

def test_watermark_survival():
    """Test watermark persistence through common audio manipulations"""
    
    # Generate watermarked audio
    model = ChatterboxTTS.from_pretrained("cuda")
    wav = model.generate("Testing watermark survival through audio processing.")
    
    # Save original
    sf.write("original.wav", wav.squeeze().numpy(), model.sr)
    
    # Test different manipulations
    manipulations = {
        "mp3_128": lambda x, sr: apply_mp3_compression(x, sr, 128),
        "mp3_64": lambda x, sr: apply_mp3_compression(x, sr, 64),
        "volume_50": lambda x, sr: (x * 0.5, sr),
        "speed_110": lambda x, sr: librosa.effects.time_stretch(x, rate=1.1),
        "pitch_shift": lambda x, sr: librosa.effects.pitch_shift(x, sr=sr, n_steps=2)
    }
    
    watermarker = perth.PerthImplicitWatermarker()
    
    for name, manipulation in manipulations.items():
        try:
            # Apply manipulation  
            modified_audio, sr = manipulation(wav.squeeze().numpy(), model.sr)
            
            # Save manipulated version
            sf.write(f"manipulated_{name}.wav", modified_audio, sr)
            
            # Check watermark survival
            watermark = watermarker.get_watermark(modified_audio, sample_rate=sr)
            survival = "SURVIVED" if watermark > 0.5 else "LOST"
            
            print(f"{name}: {survival} (confidence: {watermark:.3f})")
            
        except Exception as e:
            print(f"{name}: ERROR - {e}")

test_watermark_survival()
```

## Watermark Ethics and Usage
- Watermarks are embedded for transparency and responsible use
- Detection tools are publicly available to verify content authenticity
- Watermarks do not affect audio quality or listening experience
- Removal of watermarks may violate terms of service

# 🏆 Citation

If you find this model useful in your research or applications, please consider citing:

```bibtex
@misc{chatterboxtts2025,
  author       = {{Resemble AI}},
  title        = {{Chatterbox-TTS: Production-Grade Open Source Text-to-Speech with Emotion Control}},
  year         = {2025},
  howpublished = {\url{https://github.com/resemble-ai/chatterbox}},
  note         = {GitHub repository},
  version      = {0.1.2}
}
```

For academic papers, you may also reference our technical approach:
```bibtex
@misc{chatterbox_technical2025,
  title={Zero-Shot Voice Cloning with Emotion Exaggeration Control using Transformer-based Speech Synthesis},
  author={{Resemble AI Research Team}},
  year={2025},
  note={Technical implementation of Chatterbox TTS system}
}
```

# 🤝 Contributing

We welcome contributions from the community! Areas where help is needed:

- **Language Support**: Training data curation and model adaptation
- **Performance Optimization**: GPU memory efficiency and speed improvements
- **Quality Improvements**: Better parameter tuning and artifact reduction
- **Documentation**: Tutorials, examples, and use case guides
- **Bug Reports**: Issue identification and reproduction steps

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

# 📞 Support & Community

## Official Discord
👋 Join us on [Discord](https://discord.gg/rJq9cRJBJ6) and let's build something awesome together!

## Resources
- **Demo Page**: [Listen to samples](https://resemble-ai.github.io/chatterbox_demopage/)
- **Hugging Face Space**: [Try it online](https://huggingface.co/spaces/ResembleAI/Chatterbox)
- **Benchmarks**: [Quality comparisons](https://podonos.com/resembleai/chatterbox)
- **Commercial API**: [Resemble AI Platform](https://resemble.ai)

## Getting Help
1. Check this README and troubleshooting section
2. Search existing [GitHub Issues](https://github.com/resemble-ai/chatterbox/issues)
3. Join our Discord for community support
4. Create a new issue with detailed reproduction steps

# ⚖️ License & Disclaimer

**License**: MIT License - Use freely in commercial and non-commercial projects

**Responsible Use**: Don't use this model to do bad things. This includes but is not limited to:
- Creating misleading or deceptive content
- Impersonating individuals without consent
- Generating harmful or offensive speech
- Violating privacy or intellectual property rights

**Training Data**: Prompts and voices are sourced from freely available data on the internet, processed and cleaned for responsible AI development.

**Watermarking**: All generated audio includes imperceptible watermarks for content authenticity and responsible AI practices.

---

<div align="center">

**Made with ♥️ by [Resemble AI](https://resemble.ai)**

*Bringing the future of voice synthesis to everyone*

</div>