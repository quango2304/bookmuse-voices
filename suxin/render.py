"""Renders the eval passages with the Core ML models (Suxin's own sentence split and joins) into dist/zh_eval/suxin_coreml/.
    uv run python render.py q8,fp32,fp32 [tag]"""
import json
import sys
from pathlib import Path

import coremltools as ct
import numpy as np
import soundfile as sf

from check_coreml import run
from graph import WORK

sys.path.insert(0, str(WORK))
from suxin_tts.synthesizer import SuxinTTS  # noqa: E402

parts = sys.argv[1].split(","); tag = sys.argv[2] if len(sys.argv) > 2 else sys.argv[1].replace(",", "_")
models = [ct.models.MLModel(str(WORK / f"coreml/{n}{'-' + v if v else ''}.mlpackage"), compute_units=ct.ComputeUnit.CPU_ONLY)
          for n, v in zip(("SuxinEncoder", "SuxinDurations", "SuxinDecoder"), parts)]
tts = SuxinTTS.from_pretrained(WORK)
tts.synthesize = lambda text: run(models, tts.frontend.prepare(text)[0])[1]
ZH = Path(__file__).resolve().parent.parent / "dist/zh_eval"
texts = json.load(open(ZH / "work/suxin_run/tn.json"))
out = ZH / "suxin_coreml"; out.mkdir(exist_ok=True)
for k, t in texts.items():
    sf.write(out / f"suxin_coreml_{tag}_{k}.wav", tts.synthesize_long(t), 22050)
    print(k)
