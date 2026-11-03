#!/usr/bin/env python3
"""Everything the app needs for VieNeu Turbo, into ../work/turbo/app/vi-turbo (copied into the app as a folder):

- TurboBackbone.mlmodelc, TurboAcoustic.mlmodelc, TurboCodec.mlmodelc (compiled)
- text_emb.f32 (419 × 768), slot_emb.f32 (17 × 768): little-endian float32; audio_emb.f16 (16 × 1024 × 768); the heads
  are these same tables (tied)
- acoustic.f16 + acoustic.json: the acoustic decoder's one layer, run in Swift
- codec_z.f16 (16 × 1024 × 512): each codebook row already through its projection; the app sums 16 rows per frame
- voices.json: the 25 presets with their reference codes and speaker anchor (xvec_proj applied here)
- tokenizer.json: phoneme character → token ids, and the 256 byte tokens for anything else (checked here against the
  real tokenizer on the sample text)
- config.json: token ids, cache and bucket sizes

    uv run python export_assets.py
"""
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer
import vieneu
from vieneu._v3_turbo_engine.hub_load_v3_turbo import load_v3_turbo_checkpoint
from vieneu_utils.phonemize_text import phonemize_text_with_emotions

HERE = Path(__file__).resolve().parent
WORK = HERE.parent / "work/turbo"
CM = WORK / "coreml-io"
OUT = WORK / "app/vi-turbo"
SAMPLE = HERE.parents[1] / "bookmuse_ios/spikes/vieneu-nano/App/Sample.txt"
if OUT.exists():
    shutil.rmtree(OUT)
OUT.mkdir(parents=True)

for src, name in [("TurboBackbone8", "TurboBackbone"), ("TurboCodec", "TurboCodec")]:
    subprocess.run(["xcrun", "coremlcompiler", "compile", str(CM / f"{src}.mlpackage"), str(OUT)], check=True,
                   stdout=subprocess.DEVNULL)
    if src != name:
        (OUT / f"{src}.mlmodelc").rename(OUT / f"{name}.mlmodelc")

model = load_v3_turbo_checkpoint(str(WORK / "vieneu"), subfolder="update", device=torch.device("cpu")).eval()
sd = model.state_dict()
sd["text_embeddings.weight"].numpy().astype("<f4").tofile(OUT / "text_emb.f32")
torch.stack([sd[f"audio_embeddings.{i}.weight"] for i in range(16)]).numpy().astype("<f2").tofile(OUT / "audio_emb.f16")
# The acoustic decoder (one layer) runs in Swift: its weights in this order, float16, row-major. The app's check
# (acoustic_check.py) shows its outputs within ~1e-5 of PyTorch's.
ad = model.acoustic_decoder
assert len(ad.layers) == 1
l = ad.layers[0]
parts = [("norm1", l.norm1.weight), ("qkv", l.attn.qkv.weight), ("q_norm", l.attn.q_norm.weight),
         ("k_norm", l.attn.k_norm.weight), ("o_proj", l.attn.o_proj.weight), ("norm2", l.norm2.weight),
         ("ff_gate", l.ff_gate.weight), ("ff_up", l.ff_up.weight), ("ff_down", l.ff_down.weight), ("norm", ad.norm.weight)]
with open(OUT / "acoustic.f16", "wb") as f:
    for _, w in parts:
        a = w.detach().numpy()
        f.write(a.astype("<f2").tobytes())
(OUT / "acoustic.json").write_text(json.dumps({"order": [[n, list(w.shape)] for n, w in parts], "eps": model.config.rms_norm_eps,
                                               "heads": 8, "head_dim": 96, "ffn": 2048}))
sd["acoustic_decoder.slot_pos_emb.weight"].numpy().astype("<f4").tofile(OUT / "slot_emb.f32")
np.load(CM / "codec_tables.npz")["z"].astype("<f2").tofile(OUT / "codec_z.f16")

presets = json.loads((Path(vieneu.__file__).parent / "assets/voices_v3_turbo.json").read_text())["presets"]
def voice_key(name):
    """The ASCII key of a voice's settings id (bookmuse.vi.<key>): "Thiền Tâm Đức" → thien_tam_duc."""
    import unicodedata
    plain = unicodedata.normalize("NFD", name.replace("Đ", "D").replace("đ", "d"))
    return "_".join("".join(c for c in plain if c.isascii()).lower().split())


voices = []
with torch.no_grad():
    for name, v in presets.items():
        anchor = model.xvec_proj(torch.tensor(v["speaker_emb"], dtype=torch.float32)[None])[0]
        voices.append({"key": voice_key(name), "name": name, "gender": v.get("gender", ""), "region": v.get("region", ""),
                       "style": v.get("style", ""), "description": v.get("description", ""),
                       "featured": v.get("featured"), "codes": np.asarray(v["codes"]).tolist(),
                       "anchor": [round(float(x), 6) for x in anchor]})
(OUT / "voices.json").write_text(json.dumps({"voices": voices}, ensure_ascii=False))

tok = Tokenizer.from_file(str(WORK / "vieneu/update/tokenizer.json"))
spec = json.loads((WORK / "vieneu/update/tokenizer.json").read_text())
pattern = spec["pre_tokenizer"]["pretokenizers"][0]["pattern"]["Regex"]
merges = [m if isinstance(m, list) else m.split(" ") for m in spec["model"]["merges"]]
vocab = {t: i for t, i in spec["model"]["vocab"].items() if not t.startswith("<|")}
# ByteLevel: each byte is a printable stand-in character (GPT-2's table).
bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
cs, n = bs[:], 0
for b in range(256):
    if b not in bs:
        bs.append(b)
        cs.append(256 + n)
        n += 1
byte_char = [""] * 256
for b, c in zip(bs, cs):
    byte_char[b] = chr(c)
rank = {(a, b): i for i, (a, b) in enumerate(merges)}
import regex


def encode(text):
    """What the app does (TurboTokenizer.swift): split by the pattern, bytes to stand-ins, merge by rank."""
    ids = []
    for piece in regex.findall(pattern, text):
        sym = [byte_char[b] for b in piece.encode()]
        while len(sym) > 1:
            best = min(range(len(sym) - 1), key=lambda i: rank.get((sym[i], sym[i + 1]), 1 << 30))
            if (sym[best], sym[best + 1]) not in rank:
                break
            sym[best:best + 2] = [sym[best] + sym[best + 1]]
        ids += [vocab[s] for s in sym]
    return ids


texts = [phonemize_text_with_emotions(line) for line in SAMPLE.read_text().splitlines() if line.strip()]
bad = [t for t in texts if encode(t) != tok.encode(t, add_special_tokens=False).ids]
assert not bad, f"the BPE port differs from the tokenizer: {bad[:1]}"
(OUT / "tokenizer.json").write_text(json.dumps({"pattern": pattern, "merges": merges, "vocab": vocab,
                                                 "byte_chars": byte_char}, ensure_ascii=False))

c = model.config
(OUT / "config.json").write_text(json.dumps({
    "style_id": 16, "text_prompt_start": c.text_prompt_start_token_id, "text_prompt_end": c.text_prompt_end_token_id,
    "speech_start": c.speech_generation_start_token_id, "speech_end": c.speech_generation_end_token_id,
    "ref_slot": c.audio_ref_slot_token_id, "audio_pad": c.audio_pad_token_id, "n_vq": c.n_vq, "hidden": c.hidden_size,
    "text_vocab": c.text_vocab_size, "audio_vocab": c.audio_vocab_size, "kv_rows": 48, "head_dim": 64, "cache": 384,
    "block": 64, "acoustic_rows": len(model.acoustic_decoder.layers) * 8, "acoustic_head_dim": 96, "slots": 17,
    "codec_buckets": [32, 64, 96, 128], "samples_per_frame": 3840, "sample_rate": 48000,
}))
print("wrote", OUT, sorted(p.name for p in OUT.iterdir()))
