"""VieNeu-TTS v3 Turbo split into the parts Core ML runs, written for fixed shapes and the Neural Engine.

The checkpoint's own modules (vieneu 3.8.3, `_v3_turbo_engine/modeling_v3_turbo.py`) are the reference; `check.py`
compares against them. What stays in the app (Swift): the embedding tables, the speaker anchor, the 16 audio heads and the
text head (plain matrix products), sampling, and the codec's input packing.

- `Backbone`: the 12 Qwen3 layers + final norm over S new rows, with the KV cache as Core ML state. New rows are written
  by a placement matrix (one-hot, positions × S) instead of a dynamic slice, so every shape is fixed.
- `Acoustic`: one step of the 2-layer acoustic decoder; its 17-slot cache goes in and out as plain tensors.

RMSNorm is computed as LayerNorm over [x, -x] (mean 0, so it is x / rms): fp16 on the Neural Engine overflows a plain
sum of squares, its LayerNorm doesn't.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

H = 768
L = 12
NQ = 12
NKV = 4
HD = 64
L_LOC = 2
NH_LOC = 8
HD_LOC = H // NH_LOC
SLOTS = 17


def rms(x, w, eps):
    d = x.shape[-1]
    y = F.layer_norm(torch.cat([x, -x], dim=-1), (2 * d,), eps=eps)
    return y[..., :d] * w


def rotate_half(x):
    a, b = x.chunk(2, dim=-1)
    return torch.cat([-b, a], dim=-1)


class Backbone(nn.Module):
    """x (1, S, H) new rows · cos, sin (S, HD) for their positions · mask (1, 1, S, M) additive · place (M, S) one-hot,
    keep (M, 1) = 1 - (row written). Returns the final-norm hidden of the S rows, (1, S, H)."""

    def __init__(self, model, max_len: int):
        super().__init__()
        bb = model.semantic_backbone
        self.eps = bb.config.rms_norm_eps
        self.layers = bb.layers
        self.norm_w = bb.norm.weight
        self.register_buffer("kc", torch.zeros(L, NKV, max_len, HD, dtype=torch.float16))
        self.register_buffer("vc", torch.zeros(L, NKV, max_len, HD, dtype=torch.float16))

    def forward(self, x, cos, sin, mask, place, keep):
        S = x.shape[1]
        for i, layer in enumerate(self.layers):
            a = layer.self_attn
            h = rms(x, layer.input_layernorm.weight, self.eps)
            q = a.q_proj(h).view(1, S, NQ, HD).transpose(1, 2)
            k = a.k_proj(h).view(1, S, NKV, HD).transpose(1, 2)
            v = a.v_proj(h).view(1, S, NKV, HD).transpose(1, 2)
            q = rms(q, a.q_norm.weight, self.eps)
            k = rms(k, a.k_norm.weight, self.eps)
            q = q * cos + rotate_half(q) * sin
            k = k * cos + rotate_half(k) * sin
            # Write the new rows into the cache: keep what's elsewhere, add the new rows where `place` puts them.
            kf = self.kc[i].float() * keep + torch.matmul(place, k[0])          # (NKV, M, HD)
            vf = self.vc[i].float() * keep + torch.matmul(place, v[0])
            self.kc[i] = kf.half()
            self.vc[i] = vf.half()
            M_ = kf.shape[1]
            kr = kf.unsqueeze(1).expand(NKV, NQ // NKV, M_, HD).reshape(1, NQ, M_, HD)   # (1, NQ, M, HD)
            vr = vf.unsqueeze(1).expand(NKV, NQ // NKV, M_, HD).reshape(1, NQ, M_, HD)
            att = torch.matmul(q, kr.transpose(-1, -2)) * (HD ** -0.5) + mask
            o = torch.matmul(att.softmax(-1), vr).transpose(1, 2).reshape(1, S, H)
            x = x + a.o_proj(o)
            h = rms(x, layer.post_attention_layernorm.weight, self.eps)
            m = layer.mlp
            x = x + m.down_proj(F.silu(m.gate_proj(h)) * m.up_proj(h))
        return rms(x, self.norm_w, self.eps)


class Acoustic(nn.Module):
    """One step: x (1, 1, H) the token embedding plus its slot embedding · kc, vc (L_LOC, NH_LOC, SLOTS, HD_LOC) the
    cache · place (SLOTS, 1) one-hot of this slot · mask (1, 1, 1, SLOTS) additive (0 up to this slot).
    Returns the final-norm hidden (1, 1, H) and the cache with this slot written."""

    def __init__(self, model):
        super().__init__()
        ad = model.acoustic_decoder
        self.layers = ad.layers
        self.norm_w = ad.norm.weight
        self.eps = model.config.rms_norm_eps

    def forward(self, x, kc, vc, place, mask):
        keep = 1 - place
        nk, nv = [], []
        for i, layer in enumerate(self.layers):
            a = layer.attn
            h = rms(x, layer.norm1.weight, self.eps)
            q, k, v = a.qkv(h).view(1, 1, 3, NH_LOC, HD_LOC).unbind(2)
            q = rms(q.transpose(1, 2), a.q_norm.weight, self.eps)                # (1, NH, 1, HD)
            k = rms(k.transpose(1, 2), a.k_norm.weight, self.eps)
            v = v.transpose(1, 2)
            kf = kc[i] * keep + place * k[0]                                     # (NH, SLOTS, HD)
            vf = vc[i] * keep + place * v[0]
            nk.append(kf)
            nv.append(vf)
            att = torch.matmul(q, kf.unsqueeze(0).transpose(-1, -2)) * (HD_LOC ** -0.5) + mask
            o = torch.matmul(att.softmax(-1), vf.unsqueeze(0)).transpose(1, 2).reshape(1, 1, H)
            x = x + a.o_proj(o)
            h = rms(x, layer.norm2.weight, self.eps)
            x = x + layer.ff_down(F.silu(layer.ff_gate(h)) * layer.ff_up(h))
        return rms(x, self.norm_w, self.eps), torch.stack(nk), torch.stack(nv)


def rope(positions, theta=10000.0):
    """cos, sin (S, HD) for absolute positions, as Qwen3's default RoPE."""
    inv = 1.0 / (theta ** (torch.arange(0, HD, 2, dtype=torch.float32) / HD))
    f = positions.float()[:, None] * inv[None]
    e = torch.cat([f, f], dim=-1)
    return e.cos(), e.sin()


class BackboneIO(nn.Module):
    """The backbone without Core ML state, for the Neural Engine (which has no state support): the cache comes in, the
    new rows' keys and values go out, and the app writes them into its cache.

    x (1, S, H) · cos, sin (S, HD) · kc, vc (L·NKV, M, HD) the cache (rows at or past `start` don't matter) ·
    mask (1, 1, S, M + S) additive: cache columns up to `start`, then causal over the new rows.
    Returns hidden (1, S, H), k_new and v_new (L·NKV, S, HD)."""

    def __init__(self, model):
        super().__init__()
        bb = model.semantic_backbone
        self.eps = bb.config.rms_norm_eps
        self.layers = bb.layers
        self.norm_w = bb.norm.weight

    def forward(self, x, cos, sin, kc, vc, mask):
        S = x.shape[1]
        Mc = kc.shape[1]
        nk, nv = [], []
        for i, layer in enumerate(self.layers):
            a = layer.self_attn
            h = rms(x, layer.input_layernorm.weight, self.eps)
            q = a.q_proj(h).view(1, S, NQ, HD).transpose(1, 2)
            k = a.k_proj(h).view(1, S, NKV, HD).transpose(1, 2)
            v = a.v_proj(h).view(1, S, NKV, HD).transpose(1, 2)
            q = rms(q, a.q_norm.weight, self.eps)
            k = rms(k, a.k_norm.weight, self.eps)
            q = q * cos + rotate_half(q) * sin
            k = k * cos + rotate_half(k) * sin
            nk.append(k[0])
            nv.append(v[0])
            kf = torch.cat([kc[i * NKV:(i + 1) * NKV], k[0]], dim=1)                # (NKV, M + S, HD)
            vf = torch.cat([vc[i * NKV:(i + 1) * NKV], v[0]], dim=1)
            kr = kf.unsqueeze(1).expand(NKV, NQ // NKV, Mc + S, HD).reshape(1, NQ, Mc + S, HD)
            vr = vf.unsqueeze(1).expand(NKV, NQ // NKV, Mc + S, HD).reshape(1, NQ, Mc + S, HD)
            att = torch.matmul(q, kr.transpose(-1, -2)) * (HD ** -0.5) + mask
            o = torch.matmul(att.softmax(-1), vr).transpose(1, 2).reshape(1, S, H)
            x = x + a.o_proj(o)
            h = rms(x, layer.post_attention_layernorm.weight, self.eps)
            m = layer.mlp
            x = x + m.down_proj(F.silu(m.gate_proj(h)) * m.up_proj(h))
        return rms(x, self.norm_w, self.eps), torch.cat(nk, 0), torch.cat(nv, 0)


class AcousticHeads(nn.Module):
    """`AcousticIO` plus the heads, so the Neural Engine does them too: the logits of all 16 codebooks (the caller reads
    the one it is sampling) and of the text head (the EOS check). Returns hidden-free outputs: audio_logits (16, 1024),
    text_logits (1, 419), k_new, v_new."""

    def __init__(self, model):
        super().__init__()
        self.io = AcousticIO(model)
        self.audio_w = nn.Parameter(torch.stack([e.weight for e in model.audio_embeddings]).detach(), requires_grad=False)
        self.text_w = nn.Parameter(model.text_embeddings.weight.detach().clone(), requires_grad=False)

    def forward(self, x, kc, vc, mask):
        h, k, v = self.io(x, kc, vc, mask)
        hv = h.reshape(1, H, 1)
        audio = torch.matmul(self.audio_w, hv).reshape(16, -1)              # (16, 1024)
        text = torch.matmul(self.text_w, hv.reshape(H, 1)).reshape(1, -1)   # (1, 419)
        return audio, text, k, v


class AcousticIO(nn.Module):
    """`Acoustic` without the cache coming back whole: the new slot's keys and values go out and the app writes them, as
    for the backbone. x (1, 1, H) · kc, vc (layers·NH_LOC, SLOTS, HD_LOC) · mask (1, 1, 1, SLOTS + 1): the cached slots
    before this one, then this one. Returns hidden (1, 1, H), k_new, v_new (layers·NH_LOC, 1, HD_LOC). The checkpoint
    (`update`) has one acoustic layer."""

    def __init__(self, model):
        super().__init__()
        ad = model.acoustic_decoder
        self.layers = ad.layers
        self.norm_w = ad.norm.weight
        self.eps = model.config.rms_norm_eps

    def forward(self, x, kc, vc, mask):
        nk, nv = [], []
        for i, layer in enumerate(self.layers):
            a = layer.attn
            h = rms(x, layer.norm1.weight, self.eps)
            qkv = a.qkv(h)                                                         # 4-D at most: the Neural Engine's limit
            q = qkv[..., :H].view(1, 1, NH_LOC, HD_LOC).transpose(1, 2)
            k = qkv[..., H:2 * H].view(1, 1, NH_LOC, HD_LOC).transpose(1, 2)
            v = qkv[..., 2 * H:].view(1, 1, NH_LOC, HD_LOC).transpose(1, 2)
            q = rms(q, a.q_norm.weight, self.eps)
            k = rms(k, a.k_norm.weight, self.eps)
            nk.append(k[0])
            nv.append(v[0])
            kf = torch.cat([kc[i * NH_LOC:(i + 1) * NH_LOC], k[0]], dim=1).unsqueeze(0)   # (1, NH, SLOTS + 1, HD)
            vf = torch.cat([vc[i * NH_LOC:(i + 1) * NH_LOC], v[0]], dim=1).unsqueeze(0)
            att = torch.matmul(q, kf.transpose(-1, -2)) * (HD_LOC ** -0.5) + mask
            o = torch.matmul(att.softmax(-1), vf).transpose(1, 2).reshape(1, 1, H)
            x = x + a.o_proj(o)
            h = rms(x, layer.norm2.weight, self.eps)
            x = x + layer.ff_down(F.silu(layer.ff_gate(h)) * layer.ff_up(h))
        return rms(x, self.norm_w, self.eps), torch.cat(nk, 0), torch.cat(nv, 0)
