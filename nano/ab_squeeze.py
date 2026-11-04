"""A/B clips for T66: each voice faster than its own pace, as before (the model squeezed to the speed) and after (the model at
its own pace, the rest on the audio with the app's time-pitch unit, at most 1.1). Same seed per chunk in both.

    swiftc -O tempo.swift -o ../work/nano/tempo
    uv run python ab_squeeze.py [voice,...] [paragraphs=6]   → ../work/nano/ab_t66/<voice>_cu.m4a, <voice>_moi.m4a
"""
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
from vieneu_utils.core_utils import gaps_to_silence, join_audio_chunks, trim_and_fade
from vieneu_utils.phonemize_text import normalize_to_chunks_v3_with_gaps, phonemize_text_with_emotions

import nano_coreml as nc
from catalog import SPEEDS, VOICES

WORK = Path(__file__).resolve().parent.parent / "work/nano"
OUT = WORK / "ab_t66"
args = [a for a in sys.argv[1:] if "=" not in a]
opts = dict(a.split("=") for a in sys.argv[1:] if "=" in a)
voices = args[0].split(",") if args else [k for k, s in SPEEDS.items() if s > 1]
text = "\n\n".join(open(WORK / "cer/corpus.txt").read().split("\n\n")[: int(opts.get("paragraphs", 6))])
chunks, gaps = normalize_to_chunks_v3_with_gaps(text, max_chars=180)
OUT.mkdir(exist_ok=True)
nano = nc.Nano()


def save(name, wavs):
    out = join_audio_chunks(wavs, sr=24000, silence_ps=gaps_to_silence(gaps))
    wav = OUT / f"{name}.wav"
    sf.write(wav, out, 24000)
    subprocess.run(["afconvert", "-f", "m4af", "-d", "aac", str(wav), str(wav.with_suffix(".m4a"))], check=True)
    wav.unlink()


for key in voices:
    spk, style = nc.voice(WORK / "voices" / f"{VOICES[key]}.json")
    speed = SPEEDS[key]
    old, new = [], []
    with tempfile.TemporaryDirectory() as tmp:
        for i, c in enumerate(chunks):
            ph = phonemize_text_with_emotions(c)
            noise = np.random.default_rng(4242 + i).standard_normal((1, 144, 240))
            for s, bucket in ((speed, old), (min(speed, 1), new)):
                log_s = nano.encode(nano.ids(ph), style, spk)[2]
                T = max(2, int(round(min(np.exp(log_s) / s, 15.0) * nc.CFG["flow_fps"])))
                bucket.append(nano.chunk(ph, spk, style, speed=s, noise=noise[..., :T])[0])
        tempo = min(speed / min(speed, 1), 1.1)
        pairs = []
        for i, w in enumerate(new):
            a, b = f"{tmp}/{i}.wav", f"{tmp}/{i}s.wav"
            sf.write(a, w, 24000, subtype="FLOAT")
            pairs += [a, b]
        subprocess.run([str(WORK / "tempo"), str(tempo), *pairs], check=True)
        new = [sf.read(f"{tmp}/{i}s.wav", dtype="float32")[0] for i in range(len(new))]
    save(f"{key}_cu", [trim_and_fade(w, 24000) for w in old])
    save(f"{key}_moi", [trim_and_fade(w, 24000) for w in new])
    print(key, "speed", speed, "tempo", tempo, flush=True)
