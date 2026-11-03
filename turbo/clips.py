#!/usr/bin/env python3
"""The ▶︎ preview clip of each Turbo voice (Listen's voice picker plays these, never live synthesis):
bookmuse_ios GMReader Resources/VoiceClips/vi_<key>.m4a, 24 kHz mono AAC like the others.

    uv run --with onnxruntime==1.22.1 python clips.py <VoiceClips folder>
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path
import numpy as np
import soundfile as sf
from vieneu import Vieneu
WORK = Path(__file__).resolve().parent.parent / "work/turbo"
out = Path(sys.argv[1])
voices = json.loads((WORK / "app/vi-turbo/voices.json").read_text())["voices"]
tts = Vieneu(mode="v3turbo")
for v in voices:
    np.random.seed(7)
    wav = tts.infer(f"Xin chào, mình là {v['name']}. Cùng đọc sách với mình nhé!", voice=v["name"], apply_watermark=False)
    with tempfile.TemporaryDirectory() as d:
        src = Path(d) / "c.wav"
        sf.write(src, wav, tts.sample_rate)
        subprocess.run(["afconvert", "-f", "m4af", "-d", "aac@24000", "-c", "1", str(src), str(out / f"vi_{v['key']}.m4a")],
                       check=True)
    print(v["key"], round(len(wav) / tts.sample_rate, 1), flush=True)
