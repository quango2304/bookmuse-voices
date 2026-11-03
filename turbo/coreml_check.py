#!/usr/bin/env python3
"""The Core ML models against the PyTorch reference: greedy codes of the first 40 frames, and how long each call takes
on this Mac per compute unit.

    uv run python coreml_check.py [ALL|CPU_AND_NE|CPU_AND_GPU|CPU_ONLY]
"""
import json
import sys
import time
from pathlib import Path

import coremltools as ct
import numpy as np
import torch
from tokenizers import Tokenizer
from vieneu._v3_turbo_engine.hub_load_v3_turbo import load_v3_turbo_checkpoint
from vieneu_utils.phonemize_text import phonemize_text_with_emotions

import modules as M

WORK = Path(__file__).resolve().parent.parent / "work/turbo"
CM = WORK / "coreml"
MAXLEN, BLOCK = 1024, 64
units = getattr(ct.ComputeUnit, sys.argv[1] if len(sys.argv) > 1 else "CPU_AND_NE")

model = load_v3_turbo_checkpoint(str(WORK / "vieneu"), subfolder="update", device=torch.device("cpu")).eval()
cfg = model.config
tok = Tokenizer.from_file(str(WORK / "vieneu/update/tokenizer.json"))
import vieneu
voice = json.loads((Path(vieneu.__file__).parent / "assets/voices_v3_turbo.json").read_text())["presets"]["Thiền Tâm Đức"]
ref_codes = torch.tensor(voice["codes"], dtype=torch.long)
spk = torch.tensor(voice["speaker_emb"], dtype=torch.float32)[None]
phon = phonemize_text_with_emotions("Ngôi chùa nhỏ nằm lưng chừng núi, nơi mây trắng vẫn thường dừng lại mỗi sớm mai.")
ids = [16, cfg.text_prompt_start_token_id] + tok.encode(phon, add_special_tokens=False).ids + [cfg.text_prompt_end_token_id]
rows = torch.full((len(ids) + len(ref_codes), 17), cfg.audio_pad_token_id, dtype=torch.long)
rows[: len(ids), 0] = torch.tensor(ids)
rows[len(ids):, 0] = cfg.audio_ref_slot_token_id
rows[len(ids):, 1:] = ref_codes
T = rows.shape[0]

text_emb = model.text_embeddings.weight.detach()
audio_emb = torch.stack([e.weight for e in model.audio_embeddings]).detach()
slot_emb = model.acoustic_decoder.slot_pos_emb.weight.detach()

with torch.no_grad():
    emb = model._build_inputs_embeds(rows[None], speaker_emb=spk)
    out = model.semantic_backbone(inputs_embeds=emb, use_cache=True, return_dict=True)
    past, h = out.past_key_values, out.last_hidden_state[:, -1]
    ref_frames = []
    for t in range(40):
        codes, _ = model.decode_one_frame(h, text_token_id=torch.tensor([5]), temperature=1.0, top_k=1, audio_top_p=1.0)
        ref_frames.append(codes.tolist())
        slot = torch.full((1, 1, 17), 1024, dtype=torch.long)
        slot[0, 0, 0] = 5
        slot[0, 0, 1:] = codes
        o = model.semantic_backbone(inputs_embeds=model._build_inputs_embeds(slot, speaker_emb=spk),
                                    past_key_values=past, use_cache=True, return_dict=True)
        past, h = o.past_key_values, o.last_hidden_state[:, -1]

t0 = time.time()
pre = ct.models.MLModel(str(CM / "TurboBackbone.mlpackage"), compute_units=units, function_name="prefill")
step = ct.models.MLModel(str(CM / "TurboBackbone.mlpackage"), compute_units=units, function_name="step")
ac = ct.models.MLModel(str(CM / "TurboAcoustic.mlpackage"), compute_units=units)
print("load %.1fs" % (time.time() - t0))
state = pre.make_state()
times = {"prefill": [], "step": [], "acoustic": []}


def bb(fn, x, start, name):
    S = x.shape[1]
    pos = torch.arange(start, start + S)
    cos, sin = M.rope(pos)
    mask = np.full((1, 1, S, MAXLEN), -1e4, np.float16)
    for i in range(S):
        mask[0, 0, i, : start + i + 1] = 0
    place = np.zeros((MAXLEN, S), np.float16)
    place[pos.numpy(), np.arange(S)] = 1
    keep = (1 - place.sum(1, keepdims=True)).astype(np.float16)
    t = time.time()
    r = fn.predict({"x": x.numpy().astype(np.float16), "cos": cos.numpy().astype(np.float16),
                    "sin": sin.numpy().astype(np.float16), "mask": mask, "place": place, "keep": keep}, state=state)
    times[name].append(time.time() - t)
    return torch.from_numpy(r["hidden"].astype(np.float32))


def frame(hb):
    kc = np.zeros((M.L_LOC, M.NH_LOC, M.SLOTS, M.HD_LOC), np.float16)
    vc = np.zeros_like(kc)
    seq = [hb, text_emb[5]]
    codes, slot0 = [], None
    for s in range(M.SLOTS):
        if s >= 2:
            seq.append(audio_emb[s - 2][codes[-1]])
        x = (seq[s] + slot_emb[s]).view(1, 1, M.H).numpy().astype(np.float16)
        place = np.zeros((M.SLOTS, 1), np.float16)
        place[s] = 1
        mask = np.full((1, 1, 1, M.SLOTS), -1e4, np.float16)
        mask[..., : s + 1] = 0
        t = time.time()
        r = ac.predict({"x": x, "kc": kc, "vc": vc, "place": place, "mask": mask})
        times["acoustic"].append(time.time() - t)
        kc, vc = r["kc_out"], r["vc_out"]
        hid = torch.from_numpy(r["hidden"].astype(np.float32))[0, 0]
        if s == 0:
            slot0 = hid
        else:
            codes.append(int((hid @ audio_emb[s - 1].T).argmax()))
            if len(codes) == 16:
                break
    return codes, int((slot0 @ text_emb.T).argmax()) == 6


with torch.no_grad():
    hs = []
    for s in range(0, T, BLOCK):
        blk = emb[:, s: s + BLOCK]
        n = blk.shape[1]
        if n < BLOCK:
            blk = torch.cat([blk, torch.zeros(1, BLOCK - n, M.H)], 1)
        hs.append(bb(pre, blk, s, "prefill")[:, :n])
    h = torch.cat(hs, 1)[0, -1]
    mine = []
    for t in range(40):
        codes, eos = frame(h)
        mine.append(codes)
        se = model._build_inputs_embeds(torch.tensor([[[5] + codes]]), speaker_emb=spk)
        h = bb(step, se, T + t, "step")[0, -1]

ref, mine = np.array(ref_frames), np.array(mine)
same = (ref == mine).all(1)
print(units, "frames equal %d/40, first difference %s, codes equal %.3f" % (
    same.sum(), None if same.all() else int(np.argmin(same)), (ref == mine).mean()))
for k, v in times.items():
    v = v[1:] if len(v) > 1 else v
    print("  %s: median %.2f ms over %d calls" % (k, 1000 * np.median(v), len(v)))
ms = 1000 * (np.median(times["step"][1:]) + 17 * np.median(times["acoustic"][1:]))
print("  per 80 ms frame ≈ %.1f ms → %.1fx real time (models only)" % (ms, 80 / ms))
