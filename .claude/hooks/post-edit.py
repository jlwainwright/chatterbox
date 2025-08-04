#!/usr/bin/env python3
"""
Claude Code Post-Edit Hook with TTS Feedback
Provides audio feedback after file edits
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

def should_announce_edit(file_path, edit_type="edit"):
    """Determine if this edit should trigger TTS"""
    
    # Skip certain file types
    skip_extensions = {'.log', '.tmp', '.cache', '.pyc', '.pyo', '.wav', '.mp3', '.mp4'}
    file_path = Path(file_path)
    
    if file_path.suffix.lower() in skip_extensions:
        return False
    
    # Skip hidden files and directories
    if any(part.startswith('.') for part in file_path.parts):
        # Allow some important hidden files
        important_hidden = {'.gitignore', '.env.example', '.claude'}
        if file_path.name not in important_hidden:
            return False
    
    # Skip very frequent edits (basic heuristic)
    # Could implement more sophisticated rate limiting
    return True

def generate_edit_tts(file_path, edit_type, python_exe):
    """Generate TTS feedback for file edit"""
    try:
        file_name = Path(file_path).name
        
        # Create appropriate message based on edit type and file
        if edit_type == "create":
            message = f"Created {file_name}"
        elif edit_type == "delete":
            message = f"Deleted {file_name}"
        else:
            message = f"Updated {file_name}"
        
        # Keep it short for frequent edits
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
    
    # Generate very short audio
    text = "{message}"
    wav = model.generate(text, exaggeration=0.4, cfg_weight=0.3, temperature=0.5)
    
    # Save with timestamp
    import time
    timestamp = int(time.time())
    filename = f"edit_{{timestamp}}.wav"
    ta.save(filename, wav, model.sr)
    
    # Non-blocking audio play
    try:
        import subprocess
        import platform
        
        if platform.system() == "Darwin":  # macOS
            subprocess.Popen(["afplay", filename], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        elif platform.system() == "Linux":
            subprocess.Popen(["aplay", filename], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        elif platform.system() == "Windows":
            import winsound
            winsound.PlaySound(filename, winsound.SND_FILENAME | winsound.SND_ASYNC)
            
    except Exception:
        pass  # Silent fail
        
except Exception:
    sys.exit(1)  # Silent fail
'''
        
        # Run with very short timeout for edit feedback
        subprocess.run([python_exe, "-c", tts_script], 
                      capture_output=True, timeout=10)
        
    except Exception:
        # Silent fail for edit hooks to not interrupt workflow
        pass

def main():
    """Main post-edit hook function"""
    
    # Get edit information
    file_path = os.environ.get('CLAUDE_EDIT_FILE', '')
    edit_type = os.environ.get('CLAUDE_EDIT_TYPE', 'edit')  # edit, create, delete
    
    # Alternative: get from command line arguments
    if len(sys.argv) > 1:
        try:
            edit_data = json.loads(sys.argv[1])
            file_path = edit_data.get('file', file_path)
            edit_type = edit_data.get('type', edit_type)
        except (json.JSONDecodeError, KeyError):
            file_path = sys.argv[1]
    
    if not file_path:
        return  # No file specified
    
    # Check if we should announce this edit
    if not should_announce_edit(file_path, edit_type):
        return
    
    # Setup TTS
    python_exe, error = setup_tts_environment()
    if not python_exe:
        return  # Silent fail for edit hooks
    
    # Generate edit feedback
    generate_edit_tts(file_path, edit_type, python_exe)

if __name__ == "__main__":
    main()