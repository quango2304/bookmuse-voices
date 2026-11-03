"""How long Nano pauses between sentences said in one chunk, for different ways of joining them (phoneme level)."""
import json
import numpy as np
from vieneu import Vieneu
from vieneu_utils.phonemize_text import phonemize_text_with_emotions

tts = Vieneu(mode="v3nano"); eng = tts.engine
v = json.load(open("../work/nano/voices/s_vinhlong_male_hanhquang20260316220828233_story_vc.json"))
spk, style = np.array(v["speaker_emb"], np.float32), np.array(v["style"], np.float32)
sents = ["Ngôi chùa nhỏ nằm lưng chừng núi, nơi mây trắng vẫn thường dừng lại mỗi sớm mai.",
         "Vị sư già ngồi bên hiên, tay chậm rãi pha một ấm trà nóng.",
         "Chú tiểu hỏi thầy làm sao để lòng mình được bình yên."]
ph = [phonemize_text_with_emotions(s).rstrip(" .") for s in sents]

def silences(w, sr=24000, win=240, db=-40, min_ms=60):
    env = np.abs(w[: len(w) // win * win]).reshape(-1, win).mean(1)
    quiet = env < 10 ** (db / 20)
    out, i = [], 0
    while i < len(quiet):
        if quiet[i]:
            j = i
            while j < len(quiet) and quiet[j]: j += 1
            if (j - i) * 10 >= min_ms and i > 5 and j < len(quiet) - 5: out.append(((i * 10) / 1000, (j - i) * 10))
            i = j
        else: i += 1
    return out

from catalog import SPEEDS, VOICES
for key in ["thien_minh", "duc_huy", "thai_son", "mach_vi", "kim_thanh"]:
    v = json.load(open(f"../work/nano/voices/{VOICES[key]}.json"))
    spk, style = np.array(v["speaker_emb"], np.float32), np.array(v["style"], np.float32)
    text = ". ".join(ph) + "."
    res = []
    for seed in [1, 2]:
        w = eng.infer(text, spk, style, steps=16, cfg=3.0, sway=-1.0, seed=seed, speed=SPEEDS[key])
        s2 = sorted(silences(w), key=lambda x: -x[1])[:2]
        res.append((round(len(w) / 24000, 2), sorted(d for _, d in s2)))
    print(f"{key:12s}", res, flush=True)
