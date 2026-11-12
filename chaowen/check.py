"""Chao Wen's three modules (PyTorch, then Core ML) against the ONNX graph with noise 0 / 0, on the same ids from our front end:
frames per phoneme and the audio (SNR), and Core ML's speed on the Mac's CPU.
    uv run python check.py torch          # the PyTorch modules, the decoder on the whole sentence and in windows
    uv run python check.py coreml [-fp32|""] [DIR]   # the Core ML models (work/chaowen/coreml/Chaowen*<suffix>.mlpackage); DIR saves both audios"""
import sys
import time

import numpy as np
import onnx
import onnxruntime as ort
import torch

from frontend import Frontend
from graph import WORK, deterministic

TEXTS = ["清晨的湖面笼罩着一层薄薄的雾，远处的山影若隐若现。", "“你到底去过哪里？”她压低声音问，眼睛却一直盯着门口。",
         "这样的早晨，他已经经历过几十年，却从来没有觉得厌倦。", "一位银行行长笑着说，这个行业的路还很长。",
         "我昨天买了一部iPhone，用Google搜了很久，NASA和USB都是缩写。",
         "太阳渐渐升高，雾气一点点散开，湖对岸的村庄露出了青灰色的屋顶，几缕炊烟正慢慢升向天空。"
         "他看着她，认真地说：“对不起，这一次我一定说话算数！”记者问他明年的打算，他想了想。"]  # the last: 4 decoder windows
TOKENS, WINDOW, CONTEXT, SCALE = 384, 512, 128, 0.96


def windowed(decode, z):
    """Windows of WINDOW frames starting at multiples of 128; each keeps its middle (all but CONTEXT frames on each side; the
    first keeps its start, the last its end). decode(z_window, mask_window) → audio of 256 × WINDOW samples."""
    F_ = z.shape[2]; step = WINDOW - 2 * CONTEXT; out = np.zeros(F_ * 256, np.float32); s = 0
    while True:
        win = np.zeros((1, 192, WINDOW), np.float32); wm = np.zeros((1, 1, WINDOW), np.float32)
        hi = min(s + WINDOW, F_)
        win[:, :, :hi - s] = z[:, :, s:hi]; wm[..., :hi - s] = 1
        audio = decode(win, wm)
        k0 = 0 if s == 0 else CONTEXT
        k1 = hi - s if hi == F_ else WINDOW - CONTEXT
        out[(s + k0) * 256:(s + k1) * 256] = audio[k0 * 256:k1 * 256]
        if hi == F_:
            return out
        s += step


def snr(ref, x):
    n = min(len(ref), len(x)); ref, x = ref[:n], x[:n]
    return 10 * np.log10(np.sum(ref ** 2) / max(np.sum((ref - x) ** 2), 1e-12))


def reference():
    path = WORK / "det.onnx"
    onnx.save(deterministic(length=SCALE), str(path))
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    f = Frontend()
    for t in TEXTS:
        ids, _ = f.encode(t)
        audio, frames = sess.run(None, {"input": np.array([ids]), "input_lengths": np.array([len(ids)])})
        yield ids, frames.reshape(-1).astype(np.int64), audio.reshape(-1)


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "torch"
    if what == "torch":
        from model import Decoder, Durations, Encoder, spread
        enc, dur, dec = Encoder().eval(), Durations().eval(), Decoder().eval()
        with torch.no_grad():
            for ids, rframes, ref in reference():
                T = len(ids)
                x, m = enc(torch.tensor([ids]), torch.tensor([T]))
                frames = dur(x, torch.ones(1, 1, T), torch.tensor([SCALE])).reshape(-1).long()
                z = spread(m, frames)
                whole = dec(z, torch.ones(1, 1, z.shape[2])).reshape(-1).numpy()
                win = windowed(lambda w, wm: dec(torch.from_numpy(w), torch.from_numpy(wm)).reshape(-1).numpy(), z.numpy())
                print(f"{T:4d} ids  frames off {int((frames.numpy() - rframes).__abs__().sum()):3d}  whole {snr(ref, whole):5.1f} dB  "
                      f"windows {snr(ref, win):5.1f} dB  windows vs whole {snr(whole, win):5.1f} dB")
    else:
        import coremltools as ct
        import soundfile as sf
        suffix = sys.argv[2] if len(sys.argv) > 2 else "-fp32"
        save = sys.argv[3] if len(sys.argv) > 3 else None
        t0 = time.perf_counter()
        enc, dur, dec = [ct.models.MLModel(str(WORK / f"coreml/Chaowen{n}{suffix}.mlpackage"), compute_units=ct.ComputeUnit.CPU_ONLY)
                         for n in ("Encoder", "Durations", "Decoder")]
        print(f"load {time.perf_counter() - t0:.1f} s")
        audio_s = comp_s = 0.0
        for i, (ids, rframes, ref) in enumerate(reference()):
            w0 = time.perf_counter(); T = len(ids)
            padded = np.zeros((1, TOKENS), np.int32); padded[0, :T] = ids
            o = enc.predict({"ids": padded, "lengths": np.array([T], np.int32)})
            mask = np.zeros((1, 1, TOKENS), np.float32); mask[..., :T] = 1
            frames = dur.predict({"x": o["x"], "mask": mask, "scale": np.array([SCALE], np.float32)})["frames"].reshape(-1)[:T].astype(np.int64)
            z = np.repeat(o["m"][:, :, :T], frames, axis=2).astype(np.float32)
            audio = windowed(lambda w, wm: dec.predict({"z": w, "mask": wm})["audio"].reshape(-1), z)
            dt = time.perf_counter() - w0
            if i: audio_s += len(audio) / 22050; comp_s += dt
            print(f"{T:4d} ids  frames off {int(np.abs(frames - rframes).sum()):3d}  SNR {snr(ref, audio):5.1f} dB  {dt * 1000:6.0f} ms")
            if save:
                sf.write(f"{save}/coreml_{i}.wav", audio, 22050); sf.write(f"{save}/onnx_{i}.wav", ref, 22050)
        print(f"{audio_s / comp_s:.1f}x real time on the Mac CPU (after the first)")
