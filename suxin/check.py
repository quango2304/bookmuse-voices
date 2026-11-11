"""Checks the three modules against the ONNX graph (noise 0 / 0) on the eval passages: frames per phoneme and audio."""
import sys
import numpy as np
import onnxruntime as ort
import torch

from graph import WORK, deterministic
from model import Decoder, Durations, Encoder, spread

sys.path.insert(0, str(WORK))
from suxin_tts.mixed_frontend import MixedFrontend  # noqa: E402

TEXTS = ["清晨的湖面笼罩着一层薄薄的雾，远处的山影若隐若现。", "“你到底去过哪里？”她压低声音问，眼睛却一直盯着门口。",
         "这样的早晨，他已经经历过几十年，却从来没有觉得厌倦。", "一位银行行长笑着说，这个行业的路还很长。"]

if __name__ == "__main__":
    f = MixedFrontend(WORK, "zh_en-suxin-light")
    det = WORK / "det.onnx"
    import onnx
    onnx.save(deterministic(), str(det))
    sess = ort.InferenceSession(str(det))
    enc, dur, dec = Encoder(), Durations(), Decoder()
    for t in TEXTS:
        ids, _ = f.prepare(t)
        x = torch.tensor([ids]); L = torch.tensor([len(ids)])
        ref, rdur = sess.run(None, {"input": x.numpy(), "input_lengths": L.numpy()})
        with torch.no_grad():
            h, m = enc(x, L)
            mask = torch.ones(1, 1, len(ids))
            frames = dur(h, mask)
            z = spread(m, frames[0, 0])
            audio = dec(z, torch.ones(1, 1, z.shape[2]))
        fd = np.abs(frames.numpy().reshape(-1) - rdur.reshape(-1)).max()
        n = min(audio.shape[-1], ref.size)
        ad = np.abs(audio.numpy().reshape(-1)[:n] - ref.reshape(-1)[:n]).max() if fd == 0 else float("nan")
        print(f"{len(ids):4d} phonemes  frames diff {fd}  audio {audio.shape[-1]} vs {ref.size}  max diff {ad:.2e}")
