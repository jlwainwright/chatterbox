#!/usr/bin/env python3
"""
Claude Code Pre-Task Hook with TTS Feedback
Provides audio feedback when tasks are started
"""

import os
import sys
import json
import subprocess
from pathlib import Path

def get_project_root():
    """Find the project root directory"""
    current = Path(__file__).parent.parent.parent  # Go up from .claude/hooks/
    return current

def setup_tts_environment():
    """Setup TTS environment and check if available"""
    project_root = get_project_root()
    venv_path = project_root / "chatterbox_venv"
    
    if not venv_path.exists():
        return None, "Virtual environment not found"
    
    # Check if in Unix/Windows
    if os.name == 'nt':
        python_exe = venv_path / "Scripts" / "python.exe"
    else:
        python_exe = venv_path / "bin" / "python"
    
    if not python_exe.exists():
        return None, "Python executable not found in venv"
    
    return str(python_exe), None

def generate_quick_tts(message, python_exe):
    """Generate quick TTS feedback for task start"""
    try:
        # Simple, fast TTS script
        tts_script = f'''
import sys
try:
    import torch
    import torchaudio as ta
    from chatterbox.tts import ChatterboxTTS
    
    # Quick device detection
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model (should be fast if cached)
    model = ChatterboxTTS.from_pretrained(device=device)
    
    # Generate short audio with fast settings
    text = "{message}"
    wav = model.generate(text, exaggeration=0.5, cfg_weight=0.4, temperature=0.6)
    
    # Save with timestamp
    import time
    timestamp = int(time.time())
    filename = f"task_start_{{timestamp}}.wav"
    ta.save(filename, wav, model.sr)
    
    print(f"🔊 TTS saved: {{filename}}")
    
    # Quick play attempt (non-blocking)
    try:
        import subprocess
        import platform
        
        if platform.system() == "Darwin":  # macOS
            subprocess.Popen(["afplay", filename])
        elif platform.system() == "Linux":
            subprocess.Popen(["aplay", filename])
        elif platform.system() == "Windows":
            import winsound
            winsound.PlaySound(filename, winsound.SND_FILENAME | winsound.SND_ASYNC)
            
    except Exception:
        pass  # Silent fail for audio playback
        
except Exception as e:
    print(f"⚠️  Quick TTS failed: {{e}}")
    sys.exit(1)
'''
        
        # Run TTS with shorter timeout for pre-task
        result = subprocess.run([python_exe, "-c", tts_script], 
                              capture_output=True, text=True, timeout=15)
        
        if result.returncode == 0:
            print("✅ Quick TTS feedback generated")
        else:
            print("⚠️  Quick TTS failed (continuing anyway)")
            
    except subprocess.TimeoutExpired:
        print("⚠️  Quick TTS timed out (continuing)")
    except Exception as e:
        print(f"⚠️  Quick TTS error: {e} (continuing)")

def main():
    """Main pre-task hook function"""
    print("🎭 Claude Code Pre-Task Hook with TTS")
    
    # Get task information
    task_description = os.environ.get('CLAUDE_TASK_DESCRIPTION', 'Starting new task')
    
    # Alternative: get from command line arguments
    if len(sys.argv) > 1:
        try:
            task_data = json.loads(sys.argv[1])
            task_description = task_data.get('description', task_description)
        except (json.JSONDecodeError, KeyError):
            task_description = sys.argv[1]
    
    print(f"🚀 Starting: {task_description}")
    
    # Setup TTS (optional for pre-task)
    python_exe, error = setup_tts_environment()
    if not python_exe:
        print(f"⚠️  TTS not available: {error}")
        return
    
    # Generate short start message
    start_messages = [
        "Starting task",
        "Beginning work",
        "Task initiated",
        "Getting started"
    ]
    
    message = start_messages[0]  # Could randomize
    
    print(f"🔊 Quick TTS: {message}")
    generate_quick_tts(message, python_exe)

if __name__ == "__main__":
    main()