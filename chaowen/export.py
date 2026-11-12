#!/usr/bin/env python3
"""The pack folders for Chinese (`zh`): Chao Wen with the front end's tables (frontend.py build), config.json and his licence notice.
- iOS: the three Core ML models compiled for the phone (all fp32, on the CPU). Then `python3 -I ../ios/pack.py <folder> zh <n>`.
- Android: `chaowen.onnx`, the published fp32 graph unchanged (the app passes scales [0, scale, 0]: noise 0 / 0).
    uv run python convert.py --fp32 && uv run python frontend.py build && uv run python check.py coreml
    uv run python export.py ios            # → work/chaowen/pack/zh
    uv run python export.py android        # → work/chaowen/pack-android/zh
"""
import json
import shutil
import subprocess
import sys

from graph import FP32, WORK

TARGET = sys.argv[1] if len(sys.argv) > 1 else "ios"
OUT = WORK / ("pack/zh" if TARGET == "ios" else "pack-android/zh")
# Chao Wen says about 4.03 syllables a second (pace.py, on Xiao Yu's count); books read at about 4.2: the factor 1.04 (inside the
# ±10 % cap), in the model: every duration × 0.96 (listening.md).
SCALE = 0.96
MODELS = {f"Chaowen{n}": f"Chaowen{n}-fp32" for n in ("Encoder", "Durations", "Decoder")}

NOTICE = """Chao Wen (超文) is Piper's zh_CN-chaowen-medium voice (https://huggingface.co/rhasspy/piper-voices, Piper by the Open Home
Foundation, MIT). Dataset: OHF-Voice voice-datasets "chaowen", CC0. The model was fine-tuned from Piper's en_US hifi_tts_92 voice,
trained on Hi-Fi TTS (speaker 92, Cori Samuel; Bakhturina et al., NVIDIA), licensed CC BY 4.0
(https://creativecommons.org/licenses/by/4.0/). Used in Bookmuse unchanged."""

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
        shutil.copy(FP32, OUT / "chaowen.onnx")
    for f in ("phrases.tsv", "chars.tsv", "t2s.tsv", "english.tsv", "phonemes.json"):
        shutil.copy(WORK / "frontend" / f, OUT / f)
    (OUT / "config.json").write_text(json.dumps({
        "model": "rhasspy/piper-voices zh_CN-chaowen-medium", "sampleRate": 22050, "hop": 256, "tokens": 384, "window": 512,
        "context": 128, "scale": SCALE, "voices": ["chaowen"]}, indent=1))
    (OUT / "LICENSE").write_text(NOTICE + "\n\n--- MODEL_CARD ---\n" + (WORK / "MODEL_CARD").read_text())
    total = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file())
    print(f"{OUT}: {total:,} B")
