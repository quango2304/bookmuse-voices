"""Runs the Core ML models on the Mac against the ONNX graph: frames, audio difference and speed per compute unit.
    uv run python check_coreml.py [--fp32] [--save DIR]"""
import sys
import time

import coremltools as ct
import numpy as np
import onnxruntime as ort
import soundfile as sf

from check import TEXTS
from graph import WORK

sys.path.insert(0, str(WORK))
from suxin_tts.mixed_frontend import MixedFrontend  # noqa: E402

SUFFIX = "-fp32" if "--fp32" in sys.argv else ""
# --parts=q8,fp32,fp32 picks each model's variant (encoder, durations, decoder): "" (fp16), fp32 or q8
PARTS = next((a.split("=")[1].split(",") for a in sys.argv if a.startswith("--parts=")), None)
SAVE = sys.argv[sys.argv.index("--save") + 1] if "--save" in sys.argv else None
LONG = ("太阳渐渐升高，雾气一点点散开，湖对岸的村庄露出了青灰色的屋顶，几缕炊烟正慢慢升向天空。"
        "他看着她，认真地说：“对不起，这一次我一定说话算数！”记者问他明年的打算，他想了想，说先把眼前这三百多家分行的账算清楚，再谈别的。")


TOKENS, WINDOW = 384, 512
CONTEXT = int(next((a.split("=")[1] for a in sys.argv if a.startswith("--context=")), 32))


def run(models, ids):
    """ids padded to TOKENS; the frames decoded in WINDOW-frame windows aligned to the attention's blocks."""
    from check_decoder import windowed
    enc, dur, dec = models
    T = len(ids)
    padded = np.zeros((1, TOKENS), np.int32); padded[0, :T] = ids
    o = enc.predict({"ids": padded, "lengths": np.array([T], np.int32)})
    mask = np.zeros((1, 1, TOKENS), np.float32); mask[..., :T] = 1
    frames = dur.predict({"x": o["x"], "mask": mask, "scale": np.ones(1, np.float32)})["frames"].reshape(-1)[:T].astype(np.int64)
    z = np.repeat(o["m"][:, :, :T], frames, axis=2).astype(np.float32)
    return frames, windowed(dec, z, 128)


if __name__ == "__main__":
    f = MixedFrontend(WORK, "zh_en-suxin-light")
    sess = ort.InferenceSession(str(WORK / "det.onnx"))
    texts = TEXTS
    for units in (ct.ComputeUnit.CPU_ONLY,):
        t0 = time.perf_counter()
        variants = [("-" + v if v else "") for v in PARTS] if PARTS else [SUFFIX] * 3
        models = [ct.models.MLModel(str(WORK / f"coreml/{n}{v}.mlpackage"), compute_units=units)
                  for n, v in zip(("SuxinEncoder", "SuxinDurations", "SuxinDecoder"), variants)]
        load = time.perf_counter() - t0
        run(models, f.prepare(texts[0])[0])  # warm-up
        total_audio = total_time = 0.0
        for i, t in enumerate(texts):
            ids = f.prepare(t)[0]
            ref, rdur = sess.run(None, {"input": np.array([ids]), "input_lengths": np.array([len(ids)])})
            w0 = time.perf_counter()
            frames, audio = run(models, ids)
            dt = time.perf_counter() - w0
            total_audio += len(audio) / 22050; total_time += dt
            fd = int(np.abs(frames - rdur.reshape(-1)).sum())
            n = min(len(audio), ref.size)
            err = np.abs(audio[:n] - ref.reshape(-1)[:n])
            snr = 10 * np.log10(np.sum(ref.reshape(-1)[:n] ** 2) / max(np.sum((audio[:n] - ref.reshape(-1)[:n]) ** 2), 1e-12))
            print(f"{units.name:11s} {len(ids):4d} ph  frames off {fd:3d}  len {len(audio)} vs {ref.size}  max {err.max():.3f}  SNR {snr:5.1f} dB  {dt*1000:6.1f} ms")
            if SAVE and units == ct.ComputeUnit.CPU_ONLY:
                sf.write(f"{SAVE}/coreml{SUFFIX}_{i}.wav", audio, 22050)
                sf.write(f"{SAVE}/onnx_{i}.wav", ref.reshape(-1), 22050)
        print(f"{units.name}: load {load:.1f} s, {total_audio / total_time:.1f}x real time")
