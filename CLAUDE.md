# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

Chatterbox TTS is a production-grade open source Text-to-Speech (TTS) and Voice Conversion (VC) system developed by Resemble AI. It features state-of-the-art zero-shot voice cloning with emotion exaggeration control, built on a 0.5B parameter Llama backbone and trained on 0.5M hours of cleaned data.

## Core Architecture

The system consists of three main neural components working in pipeline:

### 1. T3 (Token-To-Token) Model (`src/chatterbox/models/t3/`)
- **Purpose**: Converts text tokens to speech tokens using Llama transformer backbone
- **Key Files**: `t3.py`, `modules/t3_config.py`, `llama_configs.py`
- **Features**: Supports conditional generation with speaker embeddings, emotion control, and CFG
- **Inference**: Autoregressive generation with sampling controls (temperature, top-p, min-p, repetition penalty)

### 2. S3Gen (Speech Synthesis Generator) (`src/chatterbox/models/s3gen/`)
- **Purpose**: Converts speech tokens to mel-spectrograms using flow matching
- **Key Files**: `s3gen.py`, `flow_matching.py`, `decoder.py`
- **Components**: Matcha-TTS based decoder with HiFiGAN vocoder for final audio generation
- **Architecture**: Flow-based generative model with reference voice conditioning

### 3. Voice Encoder (`src/chatterbox/models/voice_encoder/`)
- **Purpose**: Extracts speaker embeddings from reference audio
- **Implementation**: Extracts speaker characteristics for zero-shot voice cloning
- **Sample Rate**: Operates at 16kHz for speaker embedding extraction

### Supporting Components
- **S3Tokenizer** (`src/chatterbox/models/s3tokenizer/`): Speech-to-token conversion for voice conversion
- **Tokenizers** (`src/chatterbox/models/tokenizers/`): Text tokenization and processing
- **Watermarking**: Built-in Perth watermarking for responsible AI usage

## Common Development Commands

### Installation and Setup
```bash
# Install from PyPI
pip install chatterbox-tts

# Development installation
git clone https://github.com/resemble-ai/chatterbox.git
cd chatterbox
pip install -e .
```

### Running Examples
```bash
# Basic TTS example with automatic device detection
python example_tts.py

# Voice conversion example
python example_vc.py

# macOS-specific example (handles MPS backend)
python example_for_mac.py

# Gradio web interface for TTS
python gradio_tts_app.py

# Gradio web interface for voice conversion
python gradio_vc_app.py
```

### Testing the Installation
```bash
# Quick test - should generate test-1.wav and test-2.wav
python -c "
import torchaudio as ta
from chatterbox.tts import ChatterboxTTS
model = ChatterboxTTS.from_pretrained('cuda' if torch.cuda.is_available() else 'cpu')
wav = model.generate('Hello world, this is a test.')
ta.save('test.wav', wav, model.sr)
print('Success! Generated test.wav')
"
```

## Key Usage Patterns

### Device Handling
The codebase has sophisticated device handling for different platforms:
- **CUDA**: Full GPU acceleration with automatic detection
- **MPS** (Apple Silicon): Fallback with compatibility checks for macOS 12.3+
- **CPU**: Universal fallback with optimized loading for non-CUDA checkpoints

### Model Loading Patterns
```python
# From pretrained (downloads models automatically)
model = ChatterboxTTS.from_pretrained(device="cuda")

# From local checkpoint directory
model = ChatterboxTTS.from_local("/path/to/checkpoints", device="cuda")
```

### Audio Processing Pipeline
1. **Reference Audio**: Processed at both 16kHz (S3_SR) and 22kHz (S3GEN_SR)
2. **Conditioning**: 6-second reference for encoder, 10-second for decoder
3. **Generation**: Produces 22kHz output with built-in watermarking
4. **Post-processing**: Automatic watermarking via Perth watermarker

### Parameter Tuning Guidelines
- **exaggeration** (0.25-2.0): Controls emotion intensity, default 0.5
- **cfg_weight** (0.0-1.0): Controls CFG strength and pacing, default 0.5
- **temperature** (0.05-5.0): Sampling randomness, default 0.8
- **min_p/top_p**: Advanced sampling parameters for quality control

## Model Configuration

### Hyperparameters (`T3Config`)
Key configuration is centralized in `src/chatterbox/models/t3/modules/t3_config.py`:
- Llama backbone configuration selection
- Speech conditioning prompt length
- Token vocabulary management
- Positional encoding settings

### Checkpoints and Files
Models are loaded from HuggingFace Hub (`ResembleAI/chatterbox`):
- `ve.safetensors`: Voice encoder weights  
- `t3_cfg.safetensors`: T3 model weights
- `s3gen.safetensors`: S3Gen model weights
- `tokenizer.json`: Text tokenizer configuration
- `conds.pt`: Pre-computed conditioning for built-in voice

## Development Notes

### Code Organization
- **Main APIs**: `tts.py` (TTS) and `vc.py` (Voice Conversion) provide clean interfaces
- **Model Implementations**: Each model has its own subdirectory with modular components
- **Utilities**: Shared utilities in `models/utils.py` and individual model utils
- **Examples**: Root-level example scripts demonstrate different usage patterns

### Key Classes and Methods
- `ChatterboxTTS.generate()`: Main TTS inference method with full parameter control
- `ChatterboxVC.generate()`: Voice conversion with target voice specification  
- `Conditionals`: Handles conditioning data for both T3 and S3Gen models
- `punc_norm()`: Text preprocessing and punctuation normalization

### Memory and Performance
- Models automatically handle CUDA vs CPU loading for compatibility
- Supports inference mode context managers for memory efficiency
- Built-in device migration for mixed-device workflows
- Optimized for both interactive use and batch processing

### Responsible AI Features
- Automatic watermarking via Perth watermarker on all generated audio
- Watermark detection utilities included in examples
- MIT license with responsible use guidelines in README

## Dependencies and Requirements

### Core Dependencies
- PyTorch 2.6.0 with torchaudio (specific version pinned)
- Transformers 4.46.3 (for Llama backbone)
- librosa 0.11.0 (for audio processing)
- diffusers 0.29.0 (for flow matching)
- safetensors 0.5.3 (for model loading)

### Optional Dependencies
- gradio (for web interfaces)
- CUDA toolkit (for GPU acceleration)
- MPS support (for Apple Silicon)

### Python Version
- Requires Python 3.9+
- Developed and tested on Python 3.11 with Debian 11
- Dependencies are pinned in pyproject.toml for reproducibility