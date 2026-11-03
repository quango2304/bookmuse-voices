"""How long Nano predicts each 180-character chunk to last at each voice's speed, against the 15 s cap."""
import json, math, sys
from pathlib import Path
import numpy as np
from vieneu import Vieneu
from vieneu_utils.phonemize_text import normalize_to_chunks_v3_with_gaps, phonemize_text_with_emotions
from catalog import SPEEDS, VOICES
WORK = Path(__file__).resolve().parent.parent / "work/nano"
eng = Vieneu(mode="v3nano").engine
text = "\n".join(open(p).read() for p in sys.argv[1:])
chunks, _ = normalize_to_chunks_v3_with_gaps(text, max_chars=180)
ph = [phonemize_text_with_emotions(c) for c in chunks]
print(len(chunks), "chunks, longest", max(len(c) for c in chunks), "chars")
for key, src in VOICES.items():
    v = json.load(open(WORK / "voices" / f"{src}.json"))
    spk = np.array(v["speaker_emb"], np.float32).reshape(1, 192); style = np.array(v["style"], np.float32).reshape(1, 50, 256)
    secs = []
    for p in ph:
        ids = eng.encode_phones(p)
        ctx = eng.s_text.run(None, {"ids": ids, "style": style})[0]
        secs.append(math.exp(float(eng.s_dur.run(None, {"ctx": ctx, "ctx_mask": ids != eng.pad, "spk": spk})[0][0])) / SPEEDS[key])
    print(f"{key:14s} max {max(secs):5.1f} s, over 15 s: {sum(s > 15 for s in secs)}, over 13 s: {sum(s > 13 for s in secs)}")
