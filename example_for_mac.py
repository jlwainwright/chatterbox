import os
import torch
import torchaudio as ta
from chatterbox.tts import ChatterboxTTS

# Detect device (Mac with M1/M2/M3/M4)
device = "mps" if torch.backends.mps.is_available() else "cpu"
map_location = torch.device(device)

torch_load_original = torch.load
def patched_torch_load(*args, **kwargs):
    if 'map_location' not in kwargs:
        kwargs['map_location'] = map_location
    return torch_load_original(*args, **kwargs)

torch.load = patched_torch_load

model = ChatterboxTTS.from_pretrained(device=device)
text = "Today is the day. I want to move like a titan at dawn, sweat like a god forging lightning. No more excuses. From now on, my mornings will be temples of discipline. I am going to work out like the gods… every damn day."

# If you want to synthesize with a different voice, specify the audio prompt
AUDIO_PROMPT_PATH = "reference_voice.wav"
if os.path.exists(AUDIO_PROMPT_PATH):
    print(f"🎤 Using reference voice: {AUDIO_PROMPT_PATH}")
    wav = model.generate(
        text, 
        audio_prompt_path=AUDIO_PROMPT_PATH,
        exaggeration=2.0,
        cfg_weight=0.5
    )
else:
    print("🎵 Using built-in voice (no reference audio found)")
    wav = model.generate(text, exaggeration=2.0, cfg_weight=0.5)
ta.save("test-2.wav", wav, model.sr)
