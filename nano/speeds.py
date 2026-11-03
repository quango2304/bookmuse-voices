#!/usr/bin/env python3
"""How fast each voice speaks by itself (Nano's duration predictor, ONNX Runtime), and the speed factor that brings it to
one pace (3.4 syllables a second, a little above the voices' median of 3.26), within 0.75–1.3 so no voice is stretched
far: syllables (words, in Vietnamese) per predicted second over a long mixed text. Prints JSON for
export_assets.py.

    uv run python speeds.py <text.txt>...
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
from vieneu import Vieneu
from vieneu_utils.phonemize_text import normalize_to_chunks_v3_with_gaps, phonemize_text_with_emotions

from catalog import VOICES

WORK = Path(__file__).resolve().parent.parent / "work/nano"
eng = Vieneu(mode="v3nano").engine
text = "\n".join(open(p).read() for p in sys.argv[1:])
chunks, _ = normalize_to_chunks_v3_with_gaps(text, max_chars=180)
phones = [phonemize_text_with_emotions(c) for c in chunks]
syllables = sum(len(c.split()) for c in chunks)
rates = {}
for key, src in VOICES.items():
    v = json.load(open(WORK / "voices" / f"{src}.json"))
    spk = np.array(v["speaker_emb"], np.float32).reshape(1, 192)
    style = np.array(v["style"], np.float32).reshape(1, 50, 256)
    seconds = 0.0
    for ph in phones:
        ids = eng.encode_phones(ph)
        ctx = eng.s_text.run(None, {"ids": ids, "style": style})[0]
        seconds += math.exp(float(eng.s_dur.run(None, {"ctx": ctx, "ctx_mask": ids != eng.pad, "spk": spk})[0][0]))
    rates[key] = syllables / seconds
target = 3.4
speeds = {k: round(min(1.3, max(0.75, target / r)), 3) for k, r in rates.items()}
for k in rates:
    print(f"{k:16s} {rates[k]:.2f} syl/s  speed {speeds[k]}", file=sys.stderr)
print(f"target {target:.2f} syl/s", file=sys.stderr)
print(json.dumps(speeds))
