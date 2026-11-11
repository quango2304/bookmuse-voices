"""Xiao Yu's ▶︎ preview clip for the app (`zh_xiaoyu.m4a`, ~3 s, 24 kHz AAC like the others), made as the app makes speech.
    uv run python clip.py OUT.m4a"""
import subprocess
import sys

import coremltools as ct
import numpy as np
import soundfile as sf

from check_coreml import run
from frontend import Frontend
from graph import WORK

TEXT = "你好，我是小雨，很高兴陪你一起读书。"
models = [ct.models.CompiledMLModel(str(WORK / f"pack/zh/{n}.mlmodelc"), compute_units=ct.ComputeUnit.CPU_ONLY)
          for n in ("SuxinEncoder", "SuxinDurations", "SuxinDecoder")]
ids, _ = Frontend().encode(TEXT)
_, audio = run(models, ids)
# trim the edges to 40 ms of silence, as the app does
loud = np.where(np.abs(audio) > 10 ** (-45 / 20))[0]
audio = audio[max(0, loud[0] - 882):loud[-1] + 882]
wav = WORK / "clip.wav"
sf.write(wav, audio, 22050)
subprocess.run(["afconvert", "-f", "m4af", "-d", "aac@24000", "-b", "64000", "-c", "1", str(wav), sys.argv[1]], check=True)
print(sys.argv[1], len(audio) / 22050, "s")
