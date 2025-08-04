# 🎭 Chatterbox TTS CLI Menu

Interactive command-line interface for running all Chatterbox TTS demos and tests.

## 🚀 Quick Start

### Option 1: Direct Launch
```bash
python demo_menu.py
```

### Option 2: Platform-Specific Launchers

**macOS/Linux:**
```bash
./run_demos.sh
```

**Windows:**
```cmd
run_demos.bat
```

**Cross-platform Python:**
```bash
python run_demos.py
```

## 📋 Available Demos

The CLI menu provides access to all demos with detailed information:

### 🔍 Testing & Setup (No Model Download)
- **📋 Installation Test** - Verify all dependencies and system compatibility
- **⚡ Minimal Demo** - Quick overview without model loading
- **🌐 Web Interface Test** - Test Gradio integration

### 🎵 Audio Generation (Requires Model Download)
- **🎬 Quick TTS Demo** - Full TTS with actual audio generation
- **📱 Example TTS** - Official basic TTS example
- **🍎 Example for Mac** - Mac-optimized with device detection
- **🔄 Voice Conversion** - Transform audio to match target voice

### 🌐 Web Interfaces
- **🌐 Gradio TTS App** - Full browser-based TTS interface
- **🔄 Gradio VC App** - Browser-based voice conversion

## 🛠️ Menu Features

### System Information
- **Automatic device detection** (CUDA/MPS/CPU)
- **Virtual environment status** checking
- **Dependency verification**
- **File system validation**

### Smart Execution
- **Automatic virtual environment activation**
- **Platform-specific command handling**
- **Progress tracking and status updates**
- **Error handling with helpful messages**

### User Guidance
- **Duration estimates** for each demo
- **Requirement descriptions** (model downloads, etc.)
- **Confirmation prompts** for long-running operations
- **Installation assistance**

## 🎯 Usage Examples

### First-Time Setup
1. Run the menu: `python demo_menu.py`
2. Select option `i` for installation instructions
3. Follow the automated setup if needed
4. Start with option `1` (Installation Test)

### Quick Demo Flow
1. Option `1` - Installation Test (30 seconds)
2. Option `2` - Minimal Demo (10 seconds) 
3. Option `4` - Quick TTS Demo (2-5 minutes, includes model download)

### Web Interface Testing
1. Option `3` - Web Interface Test (15 seconds)
2. Option `8` - Full Gradio TTS App (opens in browser)

## 🎛️ Menu Options

### Demo Selection (1-9)
Each demo includes:
- **Status indicator** (✅/❌) showing file availability
- **Duration estimate** for planning your time
- **Requirements description** (downloads, dependencies)
- **Confirmation prompts** for operations requiring downloads

### Utility Options
- **`s`** - 📊 Detailed system information
- **`i`** - ℹ️ Installation instructions and automated setup
- **`h`** - ❓ Comprehensive help and usage tips
- **`q`** - 🚪 Quit the menu

## 🔧 Technical Features

### Virtual Environment Management
- **Automatic detection** of existing virtual environments
- **Smart activation** for demo execution
- **Cross-platform compatibility** (Windows/macOS/Linux)
- **Package verification** (Chatterbox TTS, Gradio)

### Error Handling
- **Graceful degradation** when files are missing
- **Clear error messages** with suggested solutions
- **Interrupt handling** (Ctrl+C) with cleanup
- **Network timeout handling** for model downloads

### Performance Optimization
- **Colored output** for better readability
- **Screen clearing** for clean interface
- **Progress indicators** for long operations
- **Memory-efficient execution**

## 📊 System Requirements

### Minimum Requirements
- **Python 3.9+** (3.11+ recommended)
- **4GB RAM** (8GB+ recommended)
- **3GB free disk space** (for model files)

### Optional Hardware Acceleration
- **NVIDIA GPU** with CUDA for fastest performance
- **Apple Silicon** (M1/M2/M3) with MPS support
- **CPU fallback** supported on all platforms

## 🎭 Demo Categories Explained

### Testing Demos (Quick)
Perfect for:
- Verifying installation
- System compatibility checking
- Understanding capabilities
- No time commitment

### Audio Generation Demos (Longer)
Ideal for:
- Actual TTS generation
- Voice cloning experiments
- Quality evaluation
- Production testing

### Web Interface Demos (Interactive)
Great for:
- Browser-based usage
- Parameter experimentation
- File upload testing
- Sharing with others

## 🚨 Important Notes

### First Run Considerations
- **Model download**: ~2GB (5-10 minutes depending on connection)
- **Cached afterwards**: Subsequent runs are much faster
- **Internet required**: Only for initial model download

### Platform-Specific Notes
- **macOS**: Native MPS acceleration on Apple Silicon
- **Windows**: Full compatibility with proper Python installation
- **Linux**: Optimized for server and desktop environments

## 🔗 Integration Examples

### Automated Testing
```bash
# Run installation test programmatically
python demo_menu.py <<< "1"
```

### CI/CD Integration
```bash
# Test system compatibility in CI
python installation_test.py && echo "✅ System ready"
```

### Development Workflow
```bash
# Quick development test cycle
python demo_menu.py <<< "2"  # Minimal demo
python demo_menu.py <<< "4"  # Full TTS test
```

## 🎯 Troubleshooting

### Common Issues

**Virtual Environment Not Found:**
- Use option `i` for automated setup
- Manually create: `python3 -m venv chatterbox_venv`

**Missing Dependencies:**
- Run installation test (option `1`)
- Use automated installer (option `i`)

**Slow Performance:**
- First run requires model download
- Check system information (option `s`)
- Consider hardware acceleration

**Network Issues:**
- Ensure internet connectivity for first run
- Check firewall settings for HuggingFace Hub

## 📚 Additional Resources

- **Full Documentation**: [README.md](README.md)
- **Voice Cloning Guide**: Comprehensive examples in main README
- **API Reference**: Complete parameter documentation
- **Community Discord**: https://discord.gg/rJq9cRJBJ6

---

*The CLI menu provides the fastest way to explore all Chatterbox TTS capabilities with guided assistance and automated setup.*