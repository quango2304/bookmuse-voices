#!/usr/bin/env python3
"""The voice picker's ▶︎ preview clips of the ten Japanese voices, `ja_<key>.m4a` (bookmuse_ios GMReader/Resources/VoiceClips),
made as the app makes a unit: the iOS pack (dist/ios/ja, after export.sh) through ptts's own C interface on Core ML (the engine the
app links, built from work/xn-ptts-0.6.0), the app's sample line prepared as `JapaneseText` prepares it, trimmed and faded as the
apps trim a unit (40 ms of each end's silence kept, a 15 ms cosine fade), AAC 64 kb/s. Writes dist/clips/ja/.

    uv run python clips.py
"""
import array
import math
import re
import subprocess
import wave
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
PTTS = ROOT / "work/xn-ptts-0.6.0"
PACK = ROOT / "dist/ios/ja"
OUT = ROOT / "dist/clips/ja"
WORK = ROOT / "work/pocket_ja/clips"
# ListeningSamples' Japanese line (bookmuse_ios), what Apple's voices say too.
TEXT = "これは、あなたの本を朗読するときの私の声です。"
VOICES = [f"{g}_0{i}" for g in ("female", "male") for i in range(1, 6)]


def prepared(text: str) -> str:
    """`JapaneseText.prepared`: the model's character rules, spaces collapsed, a full stop at the end."""
    rules = yaml.safe_load((ROOT / "work/pocket_ja/hf/japanese.yaml").read_text())["replace_characters"]
    out = " ".join(text.strip().translate(str.maketrans(rules)).split())
    out = re.sub(r"([.!?…])\s*[,;:]", r"\1", out).replace(";", ",")
    return out if out[-1] in ".!?…" else out + "."


def trim_and_fade(x, rate):
    """`VieNeuAudio.trimAndFade`: edge silence is 10 ms windows under −40 dB."""
    win = rate // 100
    loud = [20 * math.log10(math.sqrt(sum(v * v for v in x[i:i + win]) / win) + 1e-9) > -40 for i in range(0, len(x) - win + 1, win)]
    lead = loud.index(True) * win
    tail = (len(loud) - 1 - loud[::-1].index(True)) * win + win
    keep = int(0.04 * rate)
    out = x[max(0, lead - keep):min(len(x), tail + keep)]
    n = min(int(0.015 * rate), len(out) // 2)
    for i in range(n):
        ramp = 0.5 - 0.5 * math.cos(math.pi * i / max(n - 1, 1))
        out[i] *= ramp
        out[-1 - i] *= ramp
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    subprocess.run(["cargo", "build", "--release", "-q", "-p", "ptts-ffi"], cwd=PTTS, check=True)
    speak = WORK / "speak"
    subprocess.run(["cc", str(PTTS / "ptts-ffi/examples/speak.c"), "-I", str(PTTS / "ptts-ffi/include"), "-L", str(PTTS / "target/release"),
                    "-lptts_ffi", f"-Wl,-rpath,{PTTS / 'target/release'}", "-o", str(speak)], check=True)
    text = prepared(TEXT)
    for key in VOICES:
        made = WORK / f"ja_{key}.wav"
        subprocess.run([str(speak), str(PACK), "none", text, str(made), key], check=True, capture_output=True)
        with wave.open(str(made)) as w:
            rate, raw = w.getframerate(), w.readframes(w.getnframes())
        trimmed = trim_and_fade([v / 32768 for v in array.array("h", raw)], rate)
        pcm = WORK / f"ja_{key}.trim.wav"
        with wave.open(str(pcm), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(rate)
            w.writeframes(array.array("h", [max(-32767, min(32767, int(v * 32767))) for v in trimmed]).tobytes())
        subprocess.run(["afconvert", "-f", "m4af", "-d", "aac", "-b", "64000", str(pcm), str(OUT / f"ja_{key}.m4a")], check=True)
        print(f"ja_{key}  {len(trimmed) / rate:.2f} s")


if __name__ == "__main__":
    main()
