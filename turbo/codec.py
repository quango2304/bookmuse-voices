#!/usr/bin/env python3
"""MOSS-Audio-Tokenizer-Nano decoder → Core ML, rewritten for fixed shapes from the checkpoint's weights (its own code
is too generic to trace). The app sums the 16 codebook rows itself (`codec_tables.npz`: each quantizer's codebook
already through its weight-normed 8 → 512 projection), so the model takes z (1, 512, T) and returns 48 kHz mono audio
(1, T·3840), the mean of the two interleaved channels, as vieneu does.

Stages: output_proj 512 → 768, patch ×4, transformer (4 layers, context 500), patch ×2, transformer (2, 800), patch ×2,
transformer (2, 1200), patch ×2, transformer (4, 1600), patch ×240. Causal, so a chunk is padded to its bucket and the
tail cut off. One function per bucket of frames.

    uv run python codec.py [buckets...]
"""
import math
import sys
from pathlib import Path

import coremltools as ct
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

WORK = Path(__file__).resolve().parent.parent / "work/turbo"
OUT = WORK / "coreml-io"
BUCKETS = [int(b) for b in sys.argv[1:]] or [32, 64, 96, 128, 192, 256]


def conv_weight(conv):
    """The effective weight of a weight-normed 1×1 Conv1d as a Linear (out, in), and its bias."""
    return conv.weight.detach()[:, :, 0].clone(), conv.bias.detach().clone()


class Stage(nn.Module):
    def __init__(self, pt, T):
        super().__init__()
        self.inp = pt.input_proj
        self.out = pt.output_proj
        self.layers = pt.transformer.layers
        a = self.layers[0].self_attn
        self.nh = a.num_heads
        self.d = a.embed_dim
        hd = self.d // self.nh
        self.context = a.context
        pos = torch.arange(T, dtype=torch.float32)
        freqs = torch.exp(torch.arange(hd // 2, dtype=torch.float32) * (-math.log(a.rope.max_period) * 2 / hd))
        ang = pos[:, None] * freqs[None]                                    # (T, hd/2)
        self.register_buffer("cos", ang.cos()[None, None])
        self.register_buffer("sin", ang.sin()[None, None])

    def rope(self, x):                                                      # (1, nh, T, hd), interleaved pairs
        B, H, T, D = x.shape
        x = x.reshape(B * H, T, D // 2, 2)
        r, i = x[..., 0].view(B, H, T, D // 2), x[..., 1].view(B, H, T, D // 2)
        ro = r * self.cos - i * self.sin
        io = r * self.sin + i * self.cos
        return torch.stack([ro.view(B * H, T, D // 2), io.view(B * H, T, D // 2)], -1).view(B, H, T, D)

    def forward(self, x, bias):                                             # (1, C, T) → (1, C', T)
        x = self.inp(x.transpose(1, 2))
        T = x.shape[1]
        hd = self.d // self.nh
        for l in self.layers:
            h = l.norm1(x)
            qkv = l.self_attn.in_proj(h)
            q = qkv[..., : self.d].view(1, T, self.nh, hd).transpose(1, 2)
            k = qkv[..., self.d: 2 * self.d].view(1, T, self.nh, hd).transpose(1, 2)
            v = qkv[..., 2 * self.d:].view(1, T, self.nh, hd).transpose(1, 2)
            q, k = self.rope(q), self.rope(k)
            att = torch.matmul(q, k.transpose(-1, -2)) * (hd ** -0.5) + bias
            o = torch.matmul(att.softmax(-1), v).transpose(1, 2).reshape(1, T, self.d)
            x = x + l.layer_scale_1.scale * l.self_attn.out_proj(o)
            x = x + l.layer_scale_2.scale * l.ffn(l.norm2(x))
        return self.out(x).transpose(1, 2)


def unpatch(x, h):                                                          # (1, D·h, L) → (1, D, L·h)
    _, dh, L = x.shape
    return x.reshape(1, dh // h, h, L).permute(0, 1, 3, 2).reshape(1, dh // h, L * h)


class Decoder(nn.Module):
    def __init__(self, moss, T):
        super().__init__()
        w, b = conv_weight(moss.quantizer.output_proj)
        self.proj = nn.Linear(512, 768)
        self.proj.weight.data, self.proj.bias.data = w, b
        mods, patches, t = [], [], T
        for d in moss.decoder:
            if hasattr(d, "transformer"):
                mods.append(Stage(d, t))
            else:
                patches.append(d.patch_size)
                t *= d.patch_size
        self.stages = nn.ModuleList(mods)
        self.patches = patches                                              # 4, 2, 2, 2, 240

    def forward(self, z, b1, b2, b3, b4):                                   # (1, 512, T); the stages' masks
        x = self.proj(z.transpose(1, 2)).transpose(1, 2)
        x = unpatch(x, self.patches[0])
        for stage, p, b in zip(self.stages, self.patches[1:], [b1, b2, b3, b4]):
            x = unpatch(stage(x, b), p)                                     # last: (1, 1, T·7680) interleaved L R
        x = x.reshape(1, -1, 2)
        return x.mean(-1)


def masks(dec, T):
    """Each stage's additive mask (1, 1, n, n): causal within its context. The app builds the same; as constants they
    made the model hundreds of MB."""
    out, n = [], T * dec.patches[0]
    for stage, p in zip(dec.stages, dec.patches[1:]):
        pos = torch.arange(n)
        d = pos[:, None] - pos[None]
        out.append(torch.where((d >= 0) & (d < stage.context), 0.0, -1e4)[None, None])
        n *= p
    return out


def tables(moss):
    out = []
    for qz in moss.quantizer.quantizers:
        w, b = conv_weight(qz.out_proj)                                     # (512, 8)
        out.append(qz.codebook.weight.detach() @ w.T + b)                   # (1024, 512)
    return torch.stack(out)                                                 # (16, 1024, 512)


if __name__ == "__main__":
    moss = AutoModel.from_pretrained(str(WORK / "moss"), trust_remote_code=True).eval().float()
    tab = tables(moss)
    np.savez(OUT / "codec_tables.npz", z=tab.numpy().astype(np.float16))
    paths = []
    for T in BUCKETS:
        codes = torch.randint(0, 1024, (16, T))
        with torch.no_grad():
            ref = moss._decode_frame(codes.unsqueeze(1)).audio[0].mean(0)
            z = tab[torch.arange(16)[:, None], codes].sum(0).T[None]        # (1, 512, T)
            dec = Decoder(moss, T).eval()
            bs = masks(dec, T)
            mine = dec(z, *bs)[0]
            print(T, "torch: corr %.5f max diff %.5f" % (np.corrcoef(ref, mine)[0, 1], (ref - mine).abs().max()), flush=True)
            tr = torch.jit.trace(dec, (z, *bs))
        ml = ct.convert(tr, inputs=[ct.TensorType(name="z", shape=(1, 512, T), dtype=np.float16)] +
                        [ct.TensorType(name=f"b{i + 1}", shape=b.shape, dtype=np.float16) for i, b in enumerate(bs)],
                        outputs=[ct.TensorType(name="audio", dtype=np.float16)],
                        minimum_deployment_target=ct.target.iOS18, compute_precision=ct.precision.FLOAT16)
        p = OUT / f"_codec{T}.mlpackage"
        ml.save(str(p))
        feed = {"z": z.numpy().astype(np.float16)} | {f"b{i + 1}": b.numpy().astype(np.float16) for i, b in enumerate(bs)}
        got = ml.predict(feed)["audio"][0].astype(np.float32)
        print(T, "core ml: corr %.5f max diff %.5f" % (np.corrcoef(ref, got)[0, 1], np.abs(ref.numpy() - got).max()), flush=True)
        paths.append((T, p))
    d = ct.utils.MultiFunctionDescriptor()
    for T, p in paths:
        d.add_function(str(p), src_function_name="main", target_function_name=f"f{T}")
    d.default_function_name = f"f{BUCKETS[-1]}"
    ct.utils.save_multifunction(d, str(OUT / "TurboCodec.mlpackage"))
    print("wrote TurboCodec", BUCKETS)
