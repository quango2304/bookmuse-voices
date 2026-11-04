"""Pause-compressed copy of a reference clip, to clone with voice.py: every quiet gap longer than KEEP seconds (default 0.12)
is cut to KEEP, leading and trailing silence to 50 ms. Nano copies its reference clip's pauses, so this stops a voice pausing
where nothing is written.

    ../turbo/.venv/bin/python pc.py <in.wav> <out.wav> [keep]
"""
import sys, numpy as np, soundfile as sf
src, dst = sys.argv[1], sys.argv[2]; KEEP = float(sys.argv[3]) if len(sys.argv) > 3 else 0.12
w, sr = sf.read(src); w = w if w.ndim == 1 else w.mean(1)
win = int(sr * 0.01); n = len(w) // win
env = np.abs(w[: n * win]).reshape(n, win).mean(1); quiet = env < 10 ** (-40 / 20)
out, i = [], 0
while i < n:
    j = i
    while j < n and quiet[j] == quiet[i]: j += 1
    seg = w[i * win : j * win]
    if quiet[i]:
        keep = int(sr * (0.05 if (i == 0 or j == n) else KEEP))
        if len(seg) > keep: seg = seg[:keep]
    out.append(seg); i = j
res = np.concatenate(out); sf.write(dst, res, sr)
print(src.split("/")[-1], round(len(w) / sr, 1), "->", round(len(res) / sr, 1), "s")
