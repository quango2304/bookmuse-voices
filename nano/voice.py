#!/usr/bin/env python3
"""A Nano voice from a Turbo voice: the Turbo voice's own reference (98 frames of MOSS codes) decoded to 48 kHz audio,
then Nano's `encode_reference` (192-d x-vector + 50 × 256 style tokens). Or any clip (a Vbee one from vbee_ref.py).
Writes work/nano/voices/<key>.json and, for a Turbo voice, the reference WAV beside it.

    ../turbo/.venv/bin/python voice.py thien_tam_duc [--denoise]
    ../turbo/.venv/bin/python voice.py ../work/nano/vbee/<code>.wav
"""
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from transformers import AutoModel
from vieneu import Vieneu

WORK = Path(__file__).resolve().parent.parent / "work"
OUT = WORK / "nano/voices"
OUT.mkdir(parents=True, exist_ok=True)
key = sys.argv[1]
denoise = "--denoise" in sys.argv
tag = key + ("-dn" if denoise else "")

if key.endswith(".wav"):
    ref = Path(key)
    key = tag = ref.stem
    turbo = {"name": key}
else:
    turbo = {v["key"]: v for v in json.loads((WORK / "turbo/app/vi-turbo/voices.json").read_text())["voices"]}[key]
    codes = torch.tensor(turbo["codes"]).T                               # (16, 98)
    moss = AutoModel.from_pretrained(str(WORK / "turbo/moss"), trust_remote_code=True).eval().float()
    with torch.no_grad():
        audio = moss._decode_frame(codes.unsqueeze(1)).audio[0].mean(0).numpy()
    ref = OUT / f"{key}-ref.wav"
    sf.write(ref, audio, 48000)
    print("reference", ref, "%.1f s" % (len(audio) / 48000))

tts = Vieneu(mode="v3nano")
spk, style = tts.encode_reference(str(ref), denoise=denoise)
(OUT / f"{tag}.json").write_text(json.dumps({
    "key": key, "name": turbo["name"], "gender": turbo.get("gender", ""), "region": turbo.get("region", ""),
    "denoise": denoise, "speaker_emb": np.asarray(spk).reshape(-1).tolist(), "style": np.asarray(style).tolist(),
}, ensure_ascii=False))
print("wrote", OUT / f"{tag}.json", np.asarray(style).shape)
