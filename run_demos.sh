#!/bin/bash
# Chatterbox TTS Demo Launcher (Unix/Linux/macOS)

echo "🎭 Starting Chatterbox TTS Demo Menu..."

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found. Please install Python 3."
    exit 1
fi

# Check if demo_menu.py exists
if [ ! -f "demo_menu.py" ]; then
    echo "❌ demo_menu.py not found. Please ensure the file is in the current directory."
    exit 1
fi

# Run the demo menu
python3 demo_menu.py