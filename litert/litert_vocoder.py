"""LiteRT feasibility: convert Kokoro's heaviest stage (the vocoder, ~all of the compute) to .tflite at one fixed shape,
check parity against PyTorch, and time the stages on the host.

Reuses the stage modules of the iOS Core ML converter (FluidInference/mobius 26f5fa8,
models/tts/kokoro/laishere-coreml/convert-coreml.py) so the split is the one iOS ships. Static shape = 300 frames
(T_a, 7.5 s of audio at 24 kHz / 600-sample hop), the iOS default frame bucket.

    python -I litert_vocoder.py <mobius laishere-coreml dir> <kokoro en dir: config.json, kokoro-v1_0.pth, voices/af_heart.pt> <out dir>
    (venv: litert-torch==0.10.0 (torch 2.14.1, ai-edge-litert 2.3.0), kokoro==0.9.4, coremltools==9.0 for the import)
"""
import importlib.util, pathlib, sys, time
import numpy as np, torch

mobius, src, out = map(pathlib.Path, sys.argv[1:4])
out.mkdir(parents=True, exist_ok=True)
spec = importlib.util.spec_from_file_location("conv", mobius / "convert-coreml.py")
conv = importlib.util.module_from_spec(spec); spec.loader.exec_module(conv)

from kokoro import KModel
from kokoro.istftnet import AdaINResBlock1, UpSample1d
import litert_torch


def _upsample_repeat(self, x):
    """Nearest x2 as stack + reshape: F.interpolate lowers to GATHER_ND, which the LiteRT GPU delegate doesn't support
    (it split the vocoder into GPU and CPU partitions, 434 of 1208 ops on the CPU)."""
    if self.layer_type == "none":
        return x
    b, c, t = x.shape
    return torch.stack([x, x], -1).reshape(b, c, t * 2)


UpSample1d.forward = _upsample_repeat

class OverlapAddISTFT(torch.nn.Module):
    """The tail's iSTFT is two ConvTranspose1d (11 bins -> 1, kernel 20, stride 5), which litert-torch 0.10.0 also
    mis-lowers (corr 0.92). Same maths: each frame's 20 samples by a matmul, then overlap-add the 4 hop-sized pieces."""
    def __init__(self, stft):
        super().__init__()
        self.center, self.n_fft, self.hop = stft.center, stft.n_fft, stft.hop_length
        assert self.n_fft % self.hop == 0
        self.register_buffer("wr", stft.deconv_real.weight.detach()[:, 0, :].clone())  # (bins, n_fft)
        self.register_buffer("wi", stft.deconv_imag.weight.detach()[:, 0, :].clone())

    def inverse(self, magnitude, phase):
        re, im = magnitude * torch.cos(phase), magnitude * torch.sin(phase)              # (1, bins, T)
        frames = re.transpose(1, 2) @ self.wr - im.transpose(1, 2) @ self.wi             # (1, T, n_fft)
        t, k = frames.shape[1], self.n_fft // self.hop
        out = 0
        for j in range(k):  # piece j of every frame lands j hops later
            piece = frames[:, :, j * self.hop:(j + 1) * self.hop].reshape(1, t * self.hop)
            out = out + torch.nn.functional.pad(piece, (j * self.hop, (k - 1 - j) * self.hop))
        out = out[:, None, :]
        if self.center:
            out = out[..., self.n_fft // 2:-(self.n_fft // 2)]
        return out


T_A = 300


class DWTransposeAsConv(torch.nn.Module):
    """litert-torch 0.10.0 mis-lowers the depthwise ConvTranspose1d (k=3, stride 2) in the decoder's upsampling block
    (output corr 0.49 vs PyTorch). Same maths: zero-insert x2, then a depthwise Conv1d with the flipped kernel."""
    def __init__(self, ct):
        super().__init__()
        k, p, op = ct.kernel_size[0], ct.padding[0], ct.output_padding[0]
        self.groups = ct.groups
        self.register_buffer("w", ct.weight.detach().flip(-1).clone())
        self.register_buffer("b", ct.bias.detach().clone() if ct.bias is not None else torch.zeros(ct.weight.shape[0]))
        self.pad = (k - 1 - p, k - 1 - p + op)

    def forward(self, x):
        b, c, t = x.shape
        z = torch.stack([x, torch.zeros_like(x)], -1).reshape(b, c, t * 2)[:, :, : t * 2 - 1]
        return torch.nn.functional.conv1d(torch.nn.functional.pad(z, self.pad), self.w, self.b, groups=self.groups)

torch.manual_seed(0)
model = KModel(repo_id="hexgrad/Kokoro-82M", config=str(src / "config.json"), model=str(src / "kokoro-v1_0.pth")).eval()
pack = torch.load(src / "voices" / "af_heart.pt", weights_only=True)

# Real intermediates for a sentence, then stretched/padded to the 300-frame bucket.
phonemes = "ðə kwˈɪk bɹˈWn fˈɑks ʤˈʌmps ˈOvəɹ ðə lˈAzi dˈɔɡ, ænd ðˈɛn ɪt ɹˈæn əwˈA ɪntu ðə fˈɔɹəst."
ids = [0] + [model.vocab[p] for p in phonemes if p in model.vocab] + [0]
input_ids = torch.LongTensor([ids]); ref_s = pack[len(ids) - 3]
with torch.no_grad():
    L = torch.LongTensor([input_ids.shape[1]]); mask = torch.zeros(1, input_ids.shape[1], dtype=torch.bool)
    bert = model.bert(input_ids, attention_mask=(~mask).int()); d_en = model.bert_encoder(bert).transpose(-1, -2)
    s, timbre = ref_s[:, 128:], ref_s[:, :128]
    d = model.predictor.text_encoder(d_en, s, L, mask); x, _ = model.predictor.lstm(d)
    dur = torch.round(torch.sigmoid(model.predictor.duration_proj(x)).sum(-1)).clamp(min=1).long().squeeze()
    scale = T_A / dur.sum().item()
    dur = torch.clamp(torch.round(dur * scale), min=1).long(); dur[-1] += T_A - dur.sum()
    idx = torch.repeat_interleave(torch.arange(input_ids.shape[1]), dur)
    aln = torch.zeros(input_ids.shape[1], T_A); aln[idx, torch.arange(T_A)] = 1; aln = aln[None]
    en = d.transpose(-1, -2) @ aln
    F0, N = model.predictor.F0Ntrain(en, s)
    asr = model.text_encoder(input_ids, L, mask) @ aln
    t = time.perf_counter(); ref_audio = model.decoder(asr, F0, N, timbre).squeeze().numpy(); dec_s = time.perf_counter() - t
print(f"T_a={T_A} audio={len(ref_audio) / 24000:.2f}s  torch decoder (vocoder+noise+tail) {dec_s * 1000:.0f} ms on host")

gen = model.decoder.generator
noise = conv.precompute_noise_sources(gen, F0, timbre)
conv._remove_weight_norm(model.decoder)
AdaINResBlock1.forward = conv._cos_resblock1_forward
for blk in model.decoder.decode:
    if isinstance(getattr(blk, "pool", None), torch.nn.ConvTranspose1d):
        blk.pool = DWTransposeAsConv(blk.pool)
voc = conv.CoreMLVocoderDualOutput(model.decoder).eval()
tail = conv.CoreMLTailModel(gen).eval()
with torch.no_grad():
    _, x_chk = voc(*(asr, F0, N, *conv.precompute_noise_sources(gen, F0, timbre), timbre))
    before = tail(x_chk)
    tail.stft = OverlapAddISTFT(tail.stft)
    print("overlap-add iSTFT == ConvTranspose1d in torch:", float((tail(x_chk) - before).abs().max()))
noise_model = conv.CoreMLFullNoiseModel(gen).eval()
voc_args = (asr, F0, N, noise[0], noise[1], timbre)
with torch.no_grad():
    _, x_pre = voc(*voc_args)
    audio_split = tail(x_pre).squeeze().numpy()

results = {}
for name, mod, args in (("vocoder", voc, voc_args), ("tail", tail, (x_pre,)), ("noise", noise_model, (F0, timbre))):
    path = out / f"kokoro_{name}_T{T_A}.tflite"
    try:
        t = time.perf_counter()
        edge = litert_torch.convert(mod, args)
        edge.export(str(path))
        got = edge(*[a.numpy() for a in args])
        want = mod(*args)
        got = got if isinstance(got, (list, tuple)) else [got]
        want = want if isinstance(want, (list, tuple)) else [want]
        g, w = (np.asarray(got[-1]).ravel(), want[-1].detach().numpy().ravel())
        corr = float(np.corrcoef(g, w)[0, 1])  # noise: SineGen draws random noise, so not comparable run to run
        results[name] = f"ok {path.stat().st_size / 1e6:.1f} MB, output corr vs torch {corr:.6f}, convert {time.perf_counter() - t:.0f}s"
    except Exception as e:  # report, don't hide
        results[name] = f"FAILED {type(e).__name__}: {str(e)[:300]}"
    print(name, results[name])

# Inputs/expected output for the on-device run.
np.savez(out / f"vocoder_T{T_A}_io.npz", *[a.numpy() for a in voc_args], x_pre=x_pre.numpy())
print("split vs decoder audio corr", float(np.corrcoef(audio_split[:len(ref_audio)], ref_audio[:len(audio_split)])[0, 1]))
