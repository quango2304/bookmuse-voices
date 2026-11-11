#!/usr/bin/env python3
"""The pack folders for Chinese (`zh`), with the front end's tables (frontend.py build), config.json and Suxin's licence:
- iOS: the three Core ML models compiled for the phone (8-bit encoder, fp32 durations and decoder, all on the CPU).
  Then `python3 -I ../ios/pack.py <folder> zh <n>`.
- Android: `suxin.onnx`, the published fp32 graph made deterministic (graph.py: noise 0, each phoneme's frames as an extra
  output `durations`), keeping its `scales` input (noise, length, noise_w) for ONNX Runtime.
    uv run python convert.py --fp32 && uv run python quantize.py SuxinEncoder && uv run python frontend.py build
    uv run python export.py ios            # → work/suxin/pack/zh
    uv run python export.py android        # → work/suxin/pack-android/zh
"""
import json
import shutil
import subprocess
import sys

import numpy as np
import onnx
from onnx import helper

from graph import WORK, deterministic

TARGET = sys.argv[1] if len(sys.argv) > 1 else "ios"
OUT = WORK / ("pack/zh" if TARGET == "ios" else "pack-android/zh")
# Xiao Yu says 4.71 syllables a second (55 sentences); books read at about 4.2. As for Phonon and Nano the factor is capped at
# ±10 %: 0.9, i.e. every duration × 1/0.9, which gives 4.24 a second. Whisper reads it as well as at 1 (listening.md).
SCALE = round(1 / 0.9, 4)
MODELS = {"SuxinEncoder": "SuxinEncoder-q8", "SuxinDurations": "SuxinDurations-fp32", "SuxinDecoder": "SuxinDecoder-fp32"}

if __name__ == "__main__":
    shutil.rmtree(OUT, ignore_errors=True)
    OUT.mkdir(parents=True)
    if TARGET == "ios":
        for name, source in MODELS.items():
            tmp = WORK / "compiled"
            shutil.rmtree(tmp, ignore_errors=True)
            subprocess.run(["xcrun", "coremlcompiler", "compile", str(WORK / f"coreml/{source}.mlpackage"), str(tmp)], check=True,
                           capture_output=True)
            shutil.move(str(tmp / f"{source}.mlmodelc"), str(OUT / f"{name}.mlmodelc"))
    else:
        m = deterministic()
        # keep the scales input: the app passes [0, SCALE, 0]
        g = m.graph
        g.initializer.remove(next(t for t in g.initializer if t.name == "scales"))
        g.input.append(helper.make_tensor_value_info("scales", onnx.TensorProto.FLOAT, [3]))
        onnx.save(m, str(OUT / "suxin.onnx"))
    for f in ("phrases.tsv", "chars.tsv", "t2s.tsv", "english.tsv", "phonemes.json"):
        shutil.copy(WORK / "frontend" / f, OUT / f)
    (OUT / "config.json").write_text(json.dumps({
        "model": "hzp2000itb/suxin-light", "sampleRate": 22050, "hop": 256, "tokens": 384, "window": 512, "context": 128,
        "scale": SCALE, "voices": ["xiaoyu"]}, indent=1))
    licence = WORK / "LICENSE"
    shutil.copy(licence, OUT / "LICENSE")
    total = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file())
    print(f"{OUT}: {total:,} B")
