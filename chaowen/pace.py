"""Natural pace of a Chinese Piper voice: syllables a second over the 55 Chinese-only lines of corpus_zh.txt (noise 0 / 0,
length scale 1; pauses inside a sentence count, silence at the ends doesn't).
    python pace.py <model.onnx> [length_scale]"""
import re, sys
from pathlib import Path
import numpy as np, onnxruntime as ort
from frontend import Frontend
f = Frontend(); tones = {f.ids[t][0] for t in "12345"}
s = ort.InferenceSession(sys.argv[1], providers=["CPUExecutionProvider"]); ls = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
lines = [l.strip() for l in (Path(__file__).resolve().parent / "corpus_zh.txt").read_text().splitlines()
         if l.strip() and not re.search(r"[0-9A-Za-z]", l)]
syl = sec = 0
for t in lines:
    ids, _ = f.encode(t); ids = np.array([ids], np.int64)
    a = s.run(None, {"input": ids, "input_lengths": np.array([ids.shape[1]], np.int64), "scales": np.array([0, ls, 0], np.float32)})[0].ravel()
    loud = np.flatnonzero(np.abs(a) > 10 ** (-40 / 20) * np.abs(a).max())
    sec += (loud[-1] - loud[0]) / 22050; syl += sum(i in tones for i in ids[0])
print(f"{len(lines)} lines, {syl} syllables, {sec:.1f} s: {syl / sec:.2f} syllables a second at length scale {ls}")
