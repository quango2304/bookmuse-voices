#!/usr/bin/env python3
"""VieNeu-TTS v3 Turbo → Core ML for the Neural Engine (no state): ../work/turbo/coreml-io/
TurboBackbone.mlpackage (functions `prefill`, BLOCK rows, and `step`, one row; cache of CACHE rows in, new rows out),
TurboAcoustic.mlpackage (one step), tables.npz.

    uv run python convert_io.py
"""
from pathlib import Path

import coremltools as ct
import numpy as np
import torch
from vieneu._v3_turbo_engine.hub_load_v3_turbo import load_v3_turbo_checkpoint

import modules as M

WORK = Path(__file__).resolve().parent.parent / "work/turbo"
OUT = WORK / "coreml-io"
CACHE = 384
BLOCK = 64
TARGET = ct.target.iOS18
OUT.mkdir(parents=True, exist_ok=True)
model = load_v3_turbo_checkpoint(str(WORK / "vieneu"), subfolder="update", device=torch.device("cpu")).eval()
f16 = np.float16


def backbone(S, name):
    m = M.BackboneIO(model).eval()
    ex = (torch.zeros(1, S, M.H), torch.zeros(S, M.HD), torch.zeros(S, M.HD), torch.zeros(M.L * M.NKV, CACHE, M.HD),
          torch.zeros(M.L * M.NKV, CACHE, M.HD), torch.zeros(1, 1, S, CACHE + S))
    with torch.no_grad():
        tr = torch.jit.trace(m, ex)
    names = ["x", "cos", "sin", "kc", "vc", "mask"]
    ml = ct.convert(tr, inputs=[ct.TensorType(name=n, shape=e.shape, dtype=f16) for n, e in zip(names, ex)],
                    outputs=[ct.TensorType(name=n, dtype=f16) for n in ["hidden", "k_new", "v_new"]],
                    minimum_deployment_target=TARGET, compute_precision=ct.precision.FLOAT16)
    p = OUT / f"_{name}.mlpackage"
    ml.save(str(p))
    return p


import sys
if "--acoustic-only" not in sys.argv:
    pre, step = backbone(BLOCK, "prefill"), backbone(1, "step")
if "--acoustic-only" not in sys.argv:
  d = ct.utils.MultiFunctionDescriptor()
  d.add_function(str(pre), src_function_name="main", target_function_name="prefill")
  d.add_function(str(step), src_function_name="main", target_function_name="step")
  d.default_function_name = "step"
  ct.utils.save_multifunction(d, str(OUT / "TurboBackbone.mlpackage"))
  print("wrote TurboBackbone")
  # int8 weights, per output channel: half the size; the Neural Engine still computes in fp16. Each function is
  # quantized on its own (the tool refuses a multifunction model), then merged the same way.
  import coremltools.optimize as cto
  cfg = cto.coreml.OptimizationConfig(global_config=cto.coreml.OpLinearQuantizerConfig(
      mode="linear_symmetric", dtype="int8", granularity="per_channel"))
  d8 = ct.utils.MultiFunctionDescriptor()
  for name in ["prefill", "step"]:
      m8 = cto.coreml.linear_quantize_weights(ct.models.MLModel(str(OUT / f"_{name}.mlpackage"), skip_model_load=True), config=cfg)
      m8.save(str(OUT / f"_{name}8.mlpackage"))
      d8.add_function(str(OUT / f"_{name}8.mlpackage"), src_function_name="main", target_function_name=name)
  d8.default_function_name = "step"
  ct.utils.save_multifunction(d8, str(OUT / "TurboBackbone8.mlpackage"))
  print("wrote TurboBackbone8")

if "--no-acoustic" in sys.argv:
    raise SystemExit
a = M.AcousticIO(model).eval()
c = torch.zeros(len(model.acoustic_decoder.layers) * M.NH_LOC, M.SLOTS, M.HD_LOC)
ex = (torch.zeros(1, 1, M.H), c, c, torch.zeros(1, 1, 1, M.SLOTS + 1))
with torch.no_grad():
    tr = torch.jit.trace(a, ex)
ml = ct.convert(tr, inputs=[ct.TensorType(name=n, shape=e.shape, dtype=f16) for n, e in zip(["x", "kc", "vc", "mask"], ex)],
                outputs=[ct.TensorType(name=n, dtype=f16) for n in ["hidden", "k_new", "v_new"]],
                minimum_deployment_target=TARGET, compute_precision=ct.precision.FLOAT16)
ml.save(str(OUT / "TurboAcoustic.mlpackage"))
print("wrote TurboAcoustic")
