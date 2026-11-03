#!/usr/bin/env python3
"""The split modules (modules.py) against the checkpoint's own (PyTorch, fp32, greedy): backbone hidden after the prompt,
then the codes of the first frames, sampled with top-k 1 so both sides are deterministic.

    uv run python check.py
"""
import json
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer
from vieneu._v3_turbo_engine.hub_load_v3_turbo import load_v3_turbo_checkpoint
from vieneu._v3_turbo_engine.modeling_v3_turbo import _sample_token
from vieneu_utils.phonemize_text import phonemize_text_with_emotions

import modules as M

WORK = Path(__file__).resolve().parent.parent / "work/turbo"
MAXLEN = 1024
BLOCK = 64

torch.manual_seed(0)
model = load_v3_turbo_checkpoint(str(WORK / "vieneu"), subfolder="update", device=torch.device("cpu")).eval()
cfg = model.config
tok = Tokenizer.from_file(str(WORK / "vieneu/update/tokenizer.json"))
import vieneu
presets = json.loads((Path(vieneu.__file__).parent / "assets/voices_v3_turbo.json").read_text())["presets"]
voice = presets["Thiền Tâm Đức"]
ref_codes = torch.tensor(voice["codes"], dtype=torch.long)
spk = torch.tensor(voice["speaker_emb"], dtype=torch.float32)[None]

phon = phonemize_text_with_emotions("Ngôi chùa nhỏ nằm lưng chừng núi, nơi mây trắng vẫn thường dừng lại mỗi sớm mai.")
ids = [cfg.default_style_token_id if hasattr(cfg, "default_style_token_id") else 16, cfg.text_prompt_start_token_id] + \
      tok.encode(phon, add_special_tokens=False).ids + [cfg.text_prompt_end_token_id]
rows = torch.full((len(ids) + len(ref_codes), cfg.n_vq + 1), cfg.audio_pad_token_id, dtype=torch.long)
rows[: len(ids), 0] = torch.tensor(ids)
rows[len(ids):, 0] = cfg.audio_ref_slot_token_id
rows[len(ids):, 1:] = ref_codes
T = rows.shape[0]
print("prompt rows", T, "phonemes", len(phon))

with torch.no_grad():
    emb = model._build_inputs_embeds(rows[None], speaker_emb=spk)                    # (1, T, H)

    # Reference: HF backbone + decode_one_frame, greedy.
    out = model.semantic_backbone(inputs_embeds=emb, use_cache=True, return_dict=True)
    ref_h = out.last_hidden_state[:, -1]
    past = out.past_key_values
    ref_frames = []
    h = ref_h
    for t in range(40):
        codes, local = model.decode_one_frame(h, text_token_id=torch.tensor([cfg.speech_generation_start_token_id]),
                                              temperature=1.0, top_k=1, audio_top_p=1.0)
        ref_frames.append(codes.tolist())
        slot = torch.full((1, 1, cfg.n_vq + 1), cfg.audio_pad_token_id, dtype=torch.long)
        slot[0, 0, 0] = cfg.speech_generation_start_token_id
        slot[0, 0, 1:] = codes
        se = model._build_inputs_embeds(slot, speaker_emb=spk)
        o = model.semantic_backbone(inputs_embeds=se, past_key_values=past, use_cache=True, return_dict=True)
        past = o.past_key_values
        h = o.last_hidden_state[:, -1]

    # Split modules.
    bb = M.Backbone(model, MAXLEN)
    ac = M.Acoustic(model)

    def run(x, start):
        S = x.shape[1]
        pos = torch.arange(start, start + S)
        cos, sin = M.rope(pos)
        mask = torch.full((1, 1, S, MAXLEN), -1e4)
        for i in range(S):
            mask[0, 0, i, : start + i + 1] = 0
        place = torch.zeros(MAXLEN, S)
        place[pos, torch.arange(S)] = 1
        keep = 1 - place.sum(1, keepdim=True)
        return bb(x, cos, sin, mask, place, keep)

    # Prefill in blocks of BLOCK rows; the last block is padded and its padding rows are written past the prompt, where
    # the next step overwrites them (they are masked until then).
    hs = []
    for s in range(0, T, BLOCK):
        blk = emb[:, s: s + BLOCK]
        n = blk.shape[1]
        if n < BLOCK:
            blk = torch.cat([blk, torch.zeros(1, BLOCK - n, M.H)], 1)
        hs.append(run(blk, s)[:, :n])
    my_h = torch.cat(hs, 1)[:, -1]
    print("prefill hidden: max abs diff %.4f (scale %.2f)" % ((my_h - ref_h).abs().max(), ref_h.abs().max()))

    text_emb = model.text_embeddings.weight
    audio_emb = torch.stack([e.weight for e in model.audio_embeddings])
    slot_emb = model.acoustic_decoder.slot_pos_emb.weight

    def frame(hb):
        kc = torch.zeros(M.L_LOC, M.NH_LOC, M.SLOTS, M.HD_LOC)
        vc = torch.zeros_like(kc)
        seq = [hb[0], text_emb[cfg.speech_generation_start_token_id]]
        codes = []
        slot0 = None
        for s in range(M.SLOTS - 1 + 1):
            if s >= 2:
                seq.append(audio_emb[s - 2][codes[-1]])
            if s == M.SLOTS:
                break
            x = (seq[s] + slot_emb[s]).view(1, 1, M.H)
            place = torch.zeros(M.SLOTS, 1)
            place[s] = 1
            mask = torch.full((1, 1, 1, M.SLOTS), -1e4)
            mask[..., : s + 1] = 0
            out, kc, vc = ac(x, kc, vc, place, mask)
            if s == 0:
                slot0 = out[0, 0]
            if s >= 1:
                ch = s - 1
                codes.append(int((out[0, 0] @ audio_emb[ch].T).argmax()))
                if len(codes) == cfg.n_vq:
                    break
        eos = int((slot0 @ text_emb.T).argmax()) == cfg.speech_generation_end_token_id
        return codes, eos

    my_frames = []
    h = my_h
    for t in range(40):
        codes, eos = frame(h)
        my_frames.append(codes)
        se = model._build_inputs_embeds(
            torch.tensor([[[cfg.speech_generation_start_token_id] + codes]]), speaker_emb=spk)
        h = run(se, T + t)[:, -1]

ref = np.array(ref_frames)
mine = np.array(my_frames)
same = (ref == mine).all(1)
print("frames equal:", same.sum(), "/", len(same), "first difference at", int(np.argmin(same)) if not same.all() else None)
print("codes equal overall: %.3f" % (ref == mine).mean())
