#!/usr/bin/env python3
"""VieNeu-TTS v3 Turbo → Core ML (fp16, iOS 18+), into ../work/turbo/coreml:

- TurboBackbone.mlpackage: one model, two functions sharing weights and the KV-cache state (`kc`, `vc`):
  `prefill` takes BLOCK rows, `step` takes one.
- TurboAcoustic.mlpackage: one acoustic-decoder step.
- tables.npz: what the app does in Swift (embeddings, slot embedding, speaker projection), stored fp16.

    uv run python convert.py
"""
from pathlib import Path

import coremltools as ct
import numpy as np
import torch
from vieneu._v3_turbo_engine.hub_load_v3_turbo import load_v3_turbo_checkpoint

import modules as M

WORK = Path(__file__).resolve().parent.parent / "work/turbo"
OUT = WORK / "coreml"
MAXLEN = 1024
BLOCK = 64
TARGET = ct.target.iOS18

OUT.mkdir(parents=True, exist_ok=True)
model = load_v3_turbo_checkpoint(str(WORK / "vieneu"), subfolder="update", device=torch.device("cpu")).eval()


def backbone(S: int, name: str):
    bb = M.Backbone(model, MAXLEN).eval()
    ex = (torch.zeros(1, S, M.H), torch.zeros(S, M.HD), torch.zeros(S, M.HD), torch.zeros(1, 1, S, MAXLEN),
          torch.zeros(MAXLEN, S), torch.ones(MAXLEN, 1))
    traced = torch.jit.trace(bb, ex)
    ml = ct.convert(
        traced,
        inputs=[ct.TensorType(name="x", shape=(1, S, M.H), dtype=np.float16),
                ct.TensorType(name="cos", shape=(S, M.HD), dtype=np.float16),
                ct.TensorType(name="sin", shape=(S, M.HD), dtype=np.float16),
                ct.TensorType(name="mask", shape=(1, 1, S, MAXLEN), dtype=np.float16),
                ct.TensorType(name="place", shape=(MAXLEN, S), dtype=np.float16),
                ct.TensorType(name="keep", shape=(MAXLEN, 1), dtype=np.float16)],
        outputs=[ct.TensorType(name="hidden", dtype=np.float16)],
        states=[ct.StateType(wrapped_type=ct.TensorType(shape=(M.L, M.NKV, MAXLEN, M.HD), dtype=np.float16), name="kc"),
                ct.StateType(wrapped_type=ct.TensorType(shape=(M.L, M.NKV, MAXLEN, M.HD), dtype=np.float16), name="vc")],
        minimum_deployment_target=TARGET,
        compute_precision=ct.precision.FLOAT16,
    )
    path = OUT / f"_{name}.mlpackage"
    ml.save(str(path))
    return path


with torch.no_grad():
    pre = backbone(BLOCK, "prefill")
    step = backbone(1, "step")
desc = ct.utils.MultiFunctionDescriptor()
desc.add_function(str(pre), src_function_name="main", target_function_name="prefill")
desc.add_function(str(step), src_function_name="main", target_function_name="step")
desc.default_function_name = "step"
ct.utils.save_multifunction(desc, str(OUT / "TurboBackbone.mlpackage"))
print("wrote TurboBackbone")

with torch.no_grad():
    ac = M.Acoustic(model).eval()
    ex = (torch.zeros(1, 1, M.H), torch.zeros(M.L_LOC, M.NH_LOC, M.SLOTS, M.HD_LOC),
          torch.zeros(M.L_LOC, M.NH_LOC, M.SLOTS, M.HD_LOC), torch.zeros(M.SLOTS, 1), torch.zeros(1, 1, 1, M.SLOTS))
    traced = torch.jit.trace(ac, ex)
ml = ct.convert(
    traced,
    inputs=[ct.TensorType(name="x", shape=ex[0].shape, dtype=np.float16),
            ct.TensorType(name="kc", shape=ex[1].shape, dtype=np.float16),
            ct.TensorType(name="vc", shape=ex[2].shape, dtype=np.float16),
            ct.TensorType(name="place", shape=ex[3].shape, dtype=np.float16),
            ct.TensorType(name="mask", shape=ex[4].shape, dtype=np.float16)],
    outputs=[ct.TensorType(name="hidden", dtype=np.float16), ct.TensorType(name="kc_out", dtype=np.float16),
             ct.TensorType(name="vc_out", dtype=np.float16)],
    minimum_deployment_target=TARGET,
    compute_precision=ct.precision.FLOAT16,
)
ml.save(str(OUT / "TurboAcoustic.mlpackage"))
print("wrote TurboAcoustic")

sd = model.state_dict()
np.savez(
    OUT / "tables.npz",
    text_emb=sd["text_embeddings.weight"].numpy().astype(np.float16),
    audio_emb=torch.stack([sd[f"audio_embeddings.{i}.weight"] for i in range(16)]).numpy().astype(np.float16),
    slot_emb=sd["acoustic_decoder.slot_pos_emb.weight"].numpy().astype(np.float16),
    xvec_w=sd["xvec_proj.0.weight"].numpy(), xvec_b=sd["xvec_proj.0.bias"].numpy(),
    xvec_ln_w=sd["xvec_proj.1.weight"].numpy(), xvec_ln_b=sd["xvec_proj.1.bias"].numpy(),
)
print("wrote tables.npz")
