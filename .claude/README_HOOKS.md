# 🎭 Claude Code TTS Hooks

Audio feedback system for Claude Code using Chatterbox TTS.

## 🚀 Quick Setup

### 1. Enable Post-Task TTS (Recommended)
```bash
# Already configured in .claude/settings.json
# Post-task hook is enabled by default
```

### 2. Test the Hook
```bash
# Trigger a post-task hook manually
python .claude/hooks/post-task.py "Test task completed successfully"
```

### 3. Optional: Enable Pre-Task TTS
Edit `.claude/settings.json`:
```json
{
  "hooks": {
    "pre-task": {
      "enabled": true
    }
  }
}
```

## 🎯 Available Hooks

### Post-Task Hook (Enabled)
- **Triggers**: When Claude Code completes tasks
- **Feedback**: "Task completed successfully" or "Task needs attention"
- **Audio**: Saved as `task_complete_[timestamp].wav`
- **Auto-play**: Yes (macOS/Linux/Windows)

### Pre-Task Hook (Optional)
- **Triggers**: When Claude Code starts tasks
- **Feedback**: "Starting task" or "Beginning work"
- **Audio**: Saved as `task_start_[timestamp].wav`
- **Auto-play**: Yes (background)

### Post-Edit Hook (Optional)
- **Triggers**: After file edits
- **Feedback**: "Created file.py" or "Updated file.py"
- **Audio**: Saved as `edit_[timestamp].wav`
- **Warning**: Can be noisy for frequent edits

## 🎛️ Configuration

### Voice Settings
Edit `.claude/settings.json`:
```json
{
  "tts": {
    "voice_settings": {
      "exaggeration": 0.5,  // 0.25-2.0 (emotion intensity)
      "cfg_weight": 0.5,    // 0.0-1.0 (quality/speed)
      "temperature": 0.7    // 0.05-5.0 (randomness)
    }
  }
}
```

### Audio Settings
```json
{
  "tts": {
    "audio_settings": {
      "auto_play": true,           // Play audio automatically
      "save_files": true,          // Save WAV files
      "cleanup_old_files": true,   // Clean old files
      "max_message_length": 100    // Truncate long messages
    }
  }
}
```

## 🎵 Usage Examples

### Manual Testing
```bash
# Test post-task hook
python .claude/hooks/post-task.py "File upload completed"

# Test pre-task hook  
python .claude/hooks/pre-task.py "Starting code review"

# Test post-edit hook
CLAUDE_EDIT_FILE="test.py" python .claude/hooks/post-edit.py
```

### With JSON Data
```bash
# Structured task data
python .claude/hooks/post-task.py '{"description": "API integration complete", "success": true}'

# Structured edit data
python .claude/hooks/post-edit.py '{"file": "src/main.py", "type": "create"}'
```

## 🔧 How It Works

### Architecture
1. **Claude Code** triggers hooks after tasks/edits
2. **Hook scripts** detect Chatterbox TTS environment
3. **TTS generation** uses cached models for speed
4. **Audio playback** uses system audio (afplay/aplay/winsound)
5. **File cleanup** removes old audio files

### Environment Detection
```python
# Automatic detection of:
- Virtual environment: chatterbox_venv/
- Python executable: venv/bin/python or venv/Scripts/python.exe
- Device acceleration: MPS > CUDA > CPU
- Audio system: macOS/Linux/Windows compatible
```

### Error Handling
- **Silent failures** for edit hooks (non-disruptive)
- **Timeout protection** (10-30 seconds max)
- **Graceful degradation** when TTS unavailable
- **Continue on errors** (doesn't break Claude Code workflow)

## 🎯 Customization

### Custom Messages
Edit hook files to customize messages:
```python
# In post-task.py
success_messages = [
    "Task completed successfully!",
    "Great job! Task finished.",
    "Mission accomplished!",
    "Task done perfectly!"
]

error_messages = [
    "Task needs attention",
    "Check required",
    "Issue detected"
]
```

### Voice Cloning
Add reference audio for custom voice:
```python
# In hook scripts, add:
REFERENCE_VOICE = "reference_voice.wav"
wav = model.generate(text, audio_prompt_path=REFERENCE_VOICE)
```

### Smart Filtering
Customize which edits trigger TTS:
```python
# In post-edit.py
def should_announce_edit(file_path, edit_type):
    # Skip test files
    if "test_" in file_path:
        return False
    
    # Only announce Python files
    if not file_path.endswith('.py'):
        return False
    
    return True
```

## 🚨 Troubleshooting

### No Audio Output
```bash
# Check TTS environment
python .claude/hooks/post-task.py "test"

# Check audio system
afplay demo_sample_1.wav  # macOS
aplay demo_sample_1.wav   # Linux
```

### Slow Performance
```bash
# Check model cache
ls ~/.cache/huggingface/hub/models--ResembleAI--chatterbox/

# Reduce message length in settings.json
"max_message_length": 50
```

### Hook Not Triggering
```bash
# Check Claude Code settings
cat .claude/settings.json

# Test hook manually
python .claude/hooks/post-task.py "manual test"
```

## 📊 Performance

### Timing (with cached models)
- **Post-task**: ~2-5 seconds generation + playback
- **Pre-task**: ~1-3 seconds (optimized for speed)
- **Post-edit**: ~1-2 seconds (minimal settings)

### Resource Usage
- **Memory**: ~500MB during generation
- **CPU**: Moderate usage for 2-5 seconds
- **Storage**: ~50KB per audio file

## 🎉 Benefits

### Developer Experience
- **Immediate feedback** without looking at screen
- **Audio confirmation** of successful tasks
- **Background awareness** of Claude Code activity
- **Customizable voice** and messages

### Workflow Integration
- **Non-disruptive** (continues on errors)
- **Configurable** (enable/disable individual hooks)
- **Extensible** (easy to add custom logic)
- **Cross-platform** (macOS/Linux/Windows)

---

Enjoy your audio-enhanced Claude Code experience! 🎭🔊