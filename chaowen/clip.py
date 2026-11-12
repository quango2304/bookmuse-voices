"""Chao Wen's ▶︎ preview clip for the app (`zh_chaowen.m4a`, ~3 s, 24 kHz AAC like the others), made as the app makes speech
(the compiled models in the iOS pack, the pack's scale, the decoder in windows).
    uv run python clip.py OUT.m4a"""
import json
import subprocess
import sys

import coremltools as ct
import numpy as np
import soundfile as sf

from check import TOKENS, windowed
from frontend import Frontend
from graph import WORK

TEXT = "你好，我是超文，很高兴陪你一起读书。"
PACK = WORK / "pack/zh"
enc, dur, dec = [ct.models.CompiledMLModel(str(PACK / f"Chaowen{n}.mlmodelc"), compute_units=ct.ComputeUnit.CPU_ONLY)
                 for n in ("Encoder", "Durations", "Decoder")]
scale = json.loads((PACK / "config.json").read_text())["scale"]
ids, _ = Frontend().encode(TEXT); T = len(ids)
padded = np.zeros((1, TOKENS), np.int32); padded[0, :T] = ids
o = enc.predict({"ids": padded, "lengths": np.array([T], np.int32)})
mask = np.zeros((1, 1, TOKENS), np.float32); mask[..., :T] = 1
frames = dur.predict({"x": o["x"], "mask": mask, "scale": np.array([scale], np.float32)})["frames"].reshape(-1)[:T].astype(np.int64)
audio = windowed(lambda w, wm: dec.predict({"z": w, "mask": wm})["audio"].reshape(-1), np.repeat(o["m"][:, :, :T], frames, axis=2))
# trim the edges to 40 ms of silence, as the app does
loud = np.where(np.abs(audio) > 10 ** (-45 / 20))[0]
audio = audio[max(0, loud[0] - 882):loud[-1] + 882]
wav = WORK / "clip.wav"
sf.write(wav, audio, 22050)
subprocess.run(["afconvert", "-f", "m4af", "-d", "aac@24000", "-b", "64000", "-c", "1", str(wav), sys.argv[1]], check=True)
print(sys.argv[1], len(audio) / 22050, "s")
