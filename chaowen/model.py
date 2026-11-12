"""Chao Wen (Piper's zh_CN-chaowen-medium, a VITS) as three PyTorch modules that convert to Core ML, each free of the graph's dynamic ops:

- Encoder: phoneme ids and their mask → the text encoder's hidden state x (1, 192, T) and the prior mean m (1, 192, T).
  Cut from the ONNX graph (onnx2torch).
- Durations: x and the mask → each phoneme's frame count, ceil(exp(logw)). VITS's stochastic duration predictor run in
  reverse from zero noise, written here from its weights: the ONNX graph's spline uses NonZero / GatherND / ScatterND,
  which neither onnx2torch nor Core ML take, so the spline is computed for every element and picked with masks.
- Decoder: m spread over the frames (each phoneme's column repeated for its frames) and the frame mask → 22.05 kHz audio
  (256 samples a frame). The flow in reverse and the HiFiGAN decoder, cut from the ONNX graph.

The spreading between Durations and Decoder is a few lines in the app. With noise 0 / 0 (the card's settings) the prior's
sample is its mean, so the log-variance is never needed."""
import math
import re

import numpy as np
import onnx
import torch
import torch.nn as nn
import torch.nn.functional as F
from onnx import numpy_helper, utils
from onnx2torch import convert

from graph import FP32, WORK, tidy

ENC_OUT = ["/enc_p/encoder/Mul_2_output_0", "/enc_p/Split_output_0"]
DEC_IN = ["/Add_output_0", "/Cast_2_output_0"]


def _extract(inputs, outputs, name):
    path = WORK / f"{name}.onnx"
    utils.extract_model(str(FP32), str(path), inputs, outputs)
    return onnx.load(str(path))


# The rate (samples per frame) a decoder conv's input runs at: HiFiGAN upsamples ×8, ×8, ×4 (256 a frame); each group of three
# resblocks follows one upsampling.
RATES = (1, 8, 64, 256)


def _rate(name: str) -> int:
    if name.startswith("/dec/conv_pre") or name.startswith("/dec/ups.0/"):
        return 1
    if name.startswith("/dec/conv_post"):
        return 256
    m = re.match(r"/dec/ups\.(\d+)/", name)
    if m:
        return RATES[int(m.group(1))]
    m = re.match(r"/dec/resblocks\.(\d+)/", name)
    assert m, name
    return RATES[int(m.group(1)) // 3 + 1]


def _masked_decoder(m: onnx.ModelProto) -> onnx.ModelProto:
    """Zeros the padded frames before every conv of the decoder, at the rate that conv runs at, so a fixed window that runs
    past the end of the sentence gives the audio the whole sentence gives: there the original's convs see their own zero
    padding. (The flow before it takes the mask itself.)"""
    mask = DEC_IN[1]
    m.graph.initializer.append(numpy_helper.from_array(np.array([-1], np.int64), "mask_last"))
    nodes = []
    for r in RATES[1:]:
        m.graph.initializer.append(numpy_helper.from_array(np.array([1, 1, 1, r], np.int64), f"mask_rep{r}"))
        m.graph.initializer.append(numpy_helper.from_array(np.array([1, 1, -1], np.int64), f"mask_shape{r}"))
        nodes += [onnx.helper.make_node("Unsqueeze", [mask, "mask_last"], [f"mask_u{r}"]),
                  onnx.helper.make_node("Tile", [f"mask_u{r}", f"mask_rep{r}"], [f"mask_t{r}"]),
                  onnx.helper.make_node("Reshape", [f"mask_t{r}", f"mask_shape{r}"], [f"mask_{r}"])]
    prod = {o: n for n in m.graph.node for o in n.output}
    done = set()  # convs that share an input (the resblocks after one upsampling) share its masked copy
    for n in m.graph.node:
        if n.op_type in ("Conv", "ConvTranspose") and n.name.startswith("/dec/") and n.input[0] in prod:
            r = _rate(n.name)
            masked = n.input[0] + "_masked"
            if masked not in done:
                nodes.append(onnx.helper.make_node("Mul", [n.input[0], mask if r == 1 else f"mask_{r}"], [masked], name=n.name + "_mask"))
                done.add(masked)
            n.input[0] = masked
        nodes.append(n)
    del m.graph.node[:]
    m.graph.node.extend(nodes)
    return m


class Encoder(nn.Module):
    """ids (1, T) int, lengths (1,) int → x, m (1, 192, T), already multiplied by the mask."""

    def __init__(self):
        super().__init__()
        self.g = convert(tidy(_extract(["input", "input_lengths"], ENC_OUT, "encoder"))).eval()

    def forward(self, ids, lengths):
        x, m = self.g(ids, lengths)
        # onnx2torch hands a Split's output back as the tuple of all its parts
        return x, m[0] if isinstance(m, (tuple, list)) else m


class _LayerNorm(nn.Module):
    """VITS's channel LayerNorm (over dim 1)."""

    def __init__(self, gamma, beta):
        super().__init__()
        self.gamma, self.beta = nn.Parameter(gamma), nn.Parameter(beta)

    def forward(self, x):
        return F.layer_norm(x.transpose(1, -1), (x.shape[1],), self.gamma, self.beta, 1e-5).transpose(1, -1)


class _DDSConv(nn.Module):
    """Dilated depth-separable convs (kernel 3, dilations 1, 3, 9), as VITS's modules.DDSConv."""

    def __init__(self, w, p, layers=3, k=3):
        super().__init__()
        self.sep, self.pw, self.n1, self.n2 = nn.ModuleList(), nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        for i in range(layers):
            d = k ** i
            sep = nn.Conv1d(192, 192, k, groups=192, dilation=d, padding=(k * d - d) // 2)
            sep.weight.data, sep.bias.data = w[f"{p}.convs_sep.{i}.weight"], w[f"{p}.convs_sep.{i}.bias"]
            pw = nn.Conv1d(192, 192, 1)
            pw.weight.data, pw.bias.data = w[f"{p}.convs_1x1.{i}.weight"], w[f"{p}.convs_1x1.{i}.bias"]
            self.sep.append(sep), self.pw.append(pw)
            self.n1.append(_LayerNorm(w[f"{p}.norms_1.{i}.gamma"], w[f"{p}.norms_1.{i}.beta"]))
            self.n2.append(_LayerNorm(w[f"{p}.norms_2.{i}.gamma"], w[f"{p}.norms_2.{i}.beta"]))

    def forward(self, x, mask, g=None):
        if g is not None:
            x = x + g
        for sep, pw, n1, n2 in zip(self.sep, self.pw, self.n1, self.n2):
            y = F.gelu(n1(sep(x * mask)))
            y = F.gelu(n2(pw(y)))
            x = x + y
        return x * mask


def _conv(w, p, cin, cout):
    c = nn.Conv1d(cin, cout, 1)
    c.weight.data, c.bias.data = w[p + ".weight"], w[p + ".bias"]
    return c


BINS, TAIL = 10, 5.0
MIN_W = MIN_H = MIN_D = 1e-3


def _spline_inverse(x, uw, uh, ud):
    """The inverse of nflows' unconstrained rational-quadratic spline with linear tails (VITS transforms.py), for every
    element at once: values outside [-5, 5] pass through, the bin is picked with a one-hot instead of gather."""
    inside = (x >= -TAIL) & (x <= TAIL)
    xc = x.clamp(-TAIL, TAIL)
    const = math.log(math.exp(1 - MIN_D) - 1)
    ud = F.pad(ud, (1, 1), value=const)

    widths = MIN_W + (1 - MIN_W * BINS) * F.softmax(uw, dim=-1)
    cumw = F.pad(torch.cumsum(widths, -1), (1, 0)) * (2 * TAIL) - TAIL
    cumw = torch.cat([torch.full_like(cumw[..., :1], -TAIL), cumw[..., 1:-1], torch.full_like(cumw[..., :1], TAIL)], -1)
    widths = cumw[..., 1:] - cumw[..., :-1]
    derivs = MIN_D + F.softplus(ud)
    heights = MIN_H + (1 - MIN_H * BINS) * F.softmax(uh, dim=-1)
    cumh = F.pad(torch.cumsum(heights, -1), (1, 0)) * (2 * TAIL) - TAIL
    cumh = torch.cat([torch.full_like(cumh[..., :1], -TAIL), cumh[..., 1:-1], torch.full_like(cumh[..., :1], TAIL)], -1)
    heights = cumh[..., 1:] - cumh[..., :-1]

    # searchsorted over the heights (inverse): the last edge nudged up by 1e-6, as nflows does
    edges = torch.cat([cumh[..., :-1], cumh[..., -1:] + 1e-6], -1)
    idx = (xc[..., None] >= edges).sum(-1, keepdim=True) - 1  # (..., 1)
    onehot = (idx == torch.arange(BINS, device=x.device)).to(x.dtype)  # (..., BINS)
    pick = lambda t: (t[..., :BINS] * onehot).sum(-1)
    in_cumw, in_w = pick(cumw), pick(widths)
    in_cumh, in_h = pick(cumh), pick(heights)
    in_delta = pick(heights / widths)
    in_d, in_d1 = pick(derivs), pick(derivs[..., 1:])

    dy = xc - in_cumh
    s = in_d + in_d1 - 2 * in_delta
    a = dy * s + in_h * (in_delta - in_d)
    b = in_h * in_d - dy * s
    c = -in_delta * dy
    disc = (b * b - 4 * a * c).clamp(min=0)
    root = (2 * c) / (-b - torch.sqrt(disc))
    y = root * in_w + in_cumw
    return torch.where(inside, y, x)


class _ConvFlow(nn.Module):
    def __init__(self, w, p):
        super().__init__()
        self.pre = _conv(w, p + ".pre", 1, 192)
        self.convs = _DDSConv(w, p + ".convs")
        self.proj = _conv(w, p + ".proj", 192, 3 * BINS - 1)

    def inverse(self, z, mask, g):
        x0, x1 = z[:, :1], z[:, 1:]
        h = self.proj(self.convs(self.pre(x0), mask, g)) * mask  # (1, 29, T)
        h = h.reshape(1, 1, 3 * BINS - 1, -1).permute(0, 1, 3, 2)  # (1, 1, T, 29)
        uw = h[..., :BINS] / math.sqrt(192)
        uh = h[..., BINS:2 * BINS] / math.sqrt(192)
        ud = h[..., 2 * BINS:]
        x1 = _spline_inverse(x1, uw, uh, ud)
        return torch.cat([x0, x1], 1) * mask


class Durations(nn.Module):
    """x (1, 192, T), mask (1, 1, T), scale (1,) → frames per phoneme (1, 1, T), float, 0 where masked. `scale` stretches every
    duration (VITS's length scale: above 1 slower); 1 is the model's own pace."""

    def __init__(self):
        super().__init__()
        m = onnx.load(str(FP32))
        w = {t.name: torch.from_numpy(numpy_helper.to_array(t).copy()) for t in m.graph.initializer}
        consts = {o: n for n in m.graph.node for o in n.output}
        self.pre = _conv(w, "dp.pre", 192, 192)
        self.convs = _DDSConv(w, "dp.convs")
        self.proj = _conv(w, "dp.proj", 192, 192)
        # Reverse order with the unused flow dropped, as VITS: flips around flows 7, 5, 3, then the elementwise affine.
        self.flows = nn.ModuleList([_ConvFlow(w, f"dp.flows.{i}") for i in (7, 5, 3)])
        # exp(-logs) of the elementwise affine: the second input of its Mul, already folded to a constant in this graph
        scale_in = next(n for n in m.graph.node if n.name == "/dp/flows.0/Mul").input[1]
        scale = w[scale_in] if scale_in in w else torch.from_numpy(numpy_helper.to_array(consts[scale_in].attribute[0].t).copy())
        self.register_buffer("ea_m", w["dp.flows.0.m"][None])  # (1, 2, 1)
        self.register_buffer("ea_scale", scale.reshape(1, 2, 1))

    def forward(self, x, mask, scale=None):
        g = self.proj(self.convs(self.pre(x), mask)) * mask
        z = torch.zeros_like(mask).repeat(1, 2, 1)
        for flow in self.flows:
            z = torch.flip(z, [1])
            z = flow.inverse(z, mask, g)
        z = torch.flip(z, [1])
        z = (z - self.ea_m) * self.ea_scale * mask
        w = torch.exp(z[:, :1]) * mask
        return torch.ceil(w if scale is None else w * scale.reshape(1, 1, 1))


class Decoder(nn.Module):
    """z (1, 192, F): the prior mean spread over the frames; mask (1, 1, F) → audio (1, 1, 256 F)."""

    def __init__(self):
        super().__init__()
        self.g = convert(tidy(_masked_decoder(_extract(DEC_IN, ["output"], "decoder")))).eval()

    def forward(self, z, mask):
        return self.g(z, mask).reshape(1, 1, -1)


def spread(m, frames):
    """m (1, 192, T), frames (T,) ints → (1, 192, sum(frames)): each column repeated for its frames."""
    return torch.repeat_interleave(m, frames.long(), dim=2)
