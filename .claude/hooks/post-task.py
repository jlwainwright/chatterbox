#!/usr/bin/env python3
"""
Claude Code Post-Task Hook with TTS Feedback
Provides audio feedback when tasks are completed
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
        activate_cmd = f'"{venv_path}/Scripts/activate.bat"'
    else:
        python_exe = venv_path / "bin" / "python"
        activate_cmd = f"source {venv_path}/bin/activate"
    
    if not python_exe.exists():
        return None, "Python executable not found in venv"
    
    return str(python_exe), activate_cmd

def generate_tts_feedback(message, python_exe, success=True):
    """Generate TTS feedback for task completion"""
    try:
        # Create a simple TTS script
        tts_script = f'''
import sys
import os
try:
    import torch
    import torchaudio as ta
    from chatterbox.tts import ChatterboxTTS
    
    # Quick device detection
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model (will be fast if cached)
    model = ChatterboxTTS.from_pretrained(device=device)
    
    # Generate audio
    text = "{message}"
    wav = model.generate(text, exaggeration=0.6, cfg_weight=0.5, temperature=0.7)
    
    # Save with timestamp
    import time
    timestamp = int(time.time())
    filename = f"task_complete_{{timestamp}}.wav"
    ta.save(filename, wav, model.sr)
    
    print(f"🔊 TTS saved: {{filename}}")
    
    # Play audio if possible (optional)
    try:
        import subprocess
        import platform
        
        if platform.system() == "Darwin":  # macOS
            subprocess.run(["afplay", filename], check=False)
        elif platform.system() == "Linux":
            subprocess.run(["aplay", filename], check=False)  
        elif platform.system() == "Windows":
            import winsound
            winsound.PlaySound(filename, winsound.SND_FILENAME)
        
        print("🎵 Audio played successfully")
    except Exception as e:
        print(f"⚠️  Could not play audio: {{e}}")
        
except ImportError as e:
    print(f"❌ TTS not available: {{e}}")
    sys.exit(1)
except Exception as e:
    print(f"❌ TTS generation failed: {{e}}")
    sys.exit(1)
'''
        
        # Run the TTS script
        result = subprocess.run([python_exe, "-c", tts_script], 
                              capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("✅ TTS feedback generated successfully")
            print(result.stdout)
        else:
            print("⚠️  TTS feedback failed:")
            print(result.stderr)
            
    except subprocess.TimeoutExpired:
        print("⚠️  TTS generation timed out")
    except Exception as e:
        print(f"⚠️  TTS error: {e}")

def main():
    """Main hook function"""
    print("🎭 Claude Code Post-Task Hook with TTS")
    
    # Get task information from environment or arguments
    task_description = os.environ.get('CLAUDE_TASK_DESCRIPTION', 'Task completed')
    task_success = os.environ.get('CLAUDE_TASK_SUCCESS', 'true').lower() == 'true'
    
    # Alternative: get from command line arguments
    if len(sys.argv) > 1:
        try:
            # Expect JSON input with task details
            task_data = json.loads(sys.argv[1])
            task_description = task_data.get('description', task_description)
            task_success = task_data.get('success', task_success)
        except (json.JSONDecodeError, KeyError):
            # Fallback to simple string
            task_description = sys.argv[1]
    
    print(f"📋 Task: {task_description}")
    print(f"✅ Success: {task_success}")
    
    # Setup TTS
    python_exe, error = setup_tts_environment()
    if not python_exe:
        print(f"⚠️  TTS not available: {error}")
        return
    
    # Generate appropriate message
    if task_success:
        messages = [
            f"Task completed successfully: {task_description}",
            f"Great job! {task_description} is done.",
            f"Task finished: {task_description}. Well done!",
            f"Success! {task_description} completed."
        ]
    else:
        messages = [
            f"Task encountered issues: {task_description}",
            f"Task needs attention: {task_description}",
            f"Check required for: {task_description}"
        ]
    
    # Use first message (could randomize)
    message = messages[0]
    
    # Keep message reasonable length for TTS
    if len(message) > 100:
        if task_success:
            message = "Task completed successfully!"
        else:
            message = "Task needs attention."
    
    print(f"🔊 Generating TTS: {message}")
    generate_tts_feedback(message, python_exe, task_success)

if __name__ == "__main__":
    main()