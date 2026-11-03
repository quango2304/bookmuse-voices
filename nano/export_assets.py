#!/usr/bin/env python3
"""The app's Vietnamese folder (bookmuse_ios GMReader/OnDeviceVoices/vi) from convert.py and voice.py, in work/nano/app/vi:
the three compiled models, config.json (vocab, ids, frame rate, buckets, settings, each voice's speed from speeds.py),
null.f32 (the guidance pass's speaker and style) and voices/<key>.f32 (x-vector 192 + style 50 × 256, float32).
sea_g2p.bin is copied by hand.

    uv run python export_assets.py
"""
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np

from catalog import SETTINGS, SPEEDS, VOICES
from convert import BUCKETS, TOKENS
from graphs import SNAPSHOT

WORK = Path(__file__).resolve().parent.parent / "work/nano"
OUT = WORK / "app/vi"


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "voices").mkdir(parents=True)
    for name in ["NanoText", "NanoStep", "NanoCodec"]:
        subprocess.run(["xcrun", "coremlcompiler", "compile", str(WORK / "coreml" / f"{name}.mlpackage"), str(OUT)], check=True,
                       stdout=subprocess.DEVNULL)
    cfg = json.load(open(SNAPSHOT + "config.json"))
    (OUT / "config.json").write_text(json.dumps({
        "vocab": cfg["vocab"], "bos_id": cfg["bos_id"], "eos_id": cfg["eos_id"], "pad_id": cfg["pad_id"],
        "emotion_tags": cfg.get("emotion_tags", {}), "flow_fps": cfg["flow_fps"], "hop": 1536, "tokens": TOKENS,
        "buckets": BUCKETS, "max_seconds": 15.0, "speeds": SPEEDS, **SETTINGS,
    }, ensure_ascii=False))
    c = np.load(SNAPSHOT + "constants.npz")
    np.concatenate([c["null_spk"].reshape(-1), c["null_style"].reshape(-1)]).astype(np.float32).tofile(OUT / "null.f32")
    for key, src in VOICES.items():
        v = json.load(open(WORK / "voices" / f"{src}.json"))
        np.concatenate([np.asarray(v["speaker_emb"], np.float32).reshape(-1),
                        np.asarray(v["style"], np.float32).reshape(-1)]).tofile(OUT / "voices" / f"{key}.f32")
    print("wrote", OUT, len(VOICES), "voices")


if __name__ == "__main__":
    main()
