"""The windowed Core ML decoder against the PyTorch decoder on the whole sequence, on the same z (exact frames)."""
import sys
import coremltools as ct
import numpy as np
import torch

from check import TEXTS
from check_coreml import LONG, WINDOW
from graph import WORK
from model import Decoder, Durations, Encoder, spread

sys.path.insert(0, str(WORK))
from suxin_tts.mixed_frontend import MixedFrontend  # noqa: E402


BLOCK = 128


def windowed(dec, z, context, window=None):
    """Windows of `window` frames starting at multiples of 128 (the attention's blocks); each keeps its middle, all but
    `context` frames on each side (the first keeps its start, the last its end)."""
    window = window or WINDOW
    F_ = z.shape[2]; step = window - 2 * context; out = np.zeros(F_ * 256, np.float32)
    s = 0
    while True:
        win = np.zeros((1, 192, window), np.float32); wm = np.zeros((1, 1, window), np.float32)
        hi = min(s + window, F_)
        win[:, :, :hi - s] = z[:, :, s:hi]; wm[..., :hi - s] = 1
        audio = dec.predict({"z": win, "mask": wm})["audio"].reshape(-1)
        k0 = 0 if s == 0 else context
        k1 = hi - s if hi == F_ else window - context
        out[(s + k0) * 256:(s + k1) * 256] = audio[k0 * 256:k1 * 256]
        if hi == F_:
            return out
        s += step


def snr(ref, x):
    return 10 * np.log10(np.sum(ref ** 2) / max(np.sum((ref - x) ** 2), 1e-12))


if __name__ == "__main__":
    f = MixedFrontend(WORK, "zh_en-suxin-light")
    enc, dur, tdec = Encoder(), Durations(), Decoder()
    decs = {s: ct.models.MLModel(str(WORK / f"coreml/SuxinDecoder{s}.mlpackage"), compute_units=ct.ComputeUnit.CPU_ONLY)
            for s in sys.argv[1:] or [""]}
    for t in TEXTS + [LONG]:
        ids = f.prepare(t)[0]
        with torch.no_grad():
            h, m = enc(torch.tensor([ids]), torch.tensor([len(ids)]))
            z = spread(m, dur(h, torch.ones(1, 1, len(ids)))[0, 0])
            ref = tdec(z, torch.ones(1, 1, z.shape[2])).numpy().reshape(-1)
            # the torch decoder itself, windowed: the cost of windowing alone
            class W:  # noqa
                def predict(self, d):
                    return {"audio": tdec(torch.from_numpy(d["z"]), torch.from_numpy(d["mask"])).detach().numpy()}
        zn = z.numpy()
        row = [f"{z.shape[2]:5d} frames"]
        for w, c in ((256, 64), (512, 128), (768, 128)):
            row.append(f"torch w{w}/c{c} {snr(ref, windowed(W(), zn, c, w)):5.1f}")
        for s, d in decs.items():
            row.append(f"coreml{s or '-fp16'} w512/c128 {snr(ref, windowed(d, zn, 128)):5.1f}")
        print("  ".join(row), "dB")
