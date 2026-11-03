#!/usr/bin/env python3
"""Renders a text with Nano (the vieneu package, ONNX Runtime) and a voice from voice.py, at given settings.

    ../turbo/.venv/bin/python render.py <voice.json> <text.txt> <out.wav> steps=8 cfg=3 sway=-1 seed=1234
"""
import json
import sys
import time

import numpy as np
import soundfile as sf
from vieneu import Vieneu

voice = json.loads(open(sys.argv[1]).read())
text = open(sys.argv[2]).read()
opts = {k: float(v) for k, v in (a.split("=") for a in sys.argv[4:])}
tts = Vieneu(mode="v3nano")
t0 = time.time()
wav = tts.infer(text, voice={"speaker_emb": np.array(voice["speaker_emb"], np.float32), "style": np.array(voice["style"], np.float32)},
                steps=int(opts.get("steps", 8)), cfg=opts.get("cfg", 3), sway=opts.get("sway", -1), speed=opts.get("speed", 1),
                seed=int(opts.get("seed", 1234)), apply_watermark=False)
sf.write(sys.argv[3], wav, 24000)
print(sys.argv[3], "%.1f s audio in %.1f s" % (len(wav) / 24000, time.time() - t0), flush=True)
