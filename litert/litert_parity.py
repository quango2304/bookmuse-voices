"""Parity + host timing of the LiteRT vocoder/tail at 300 frames (after litert_vocoder.py).

    python -I litert_parity.py <models/litert dir> <threads>
"""
import pathlib, sys, time
import numpy as np
from ai_edge_litert.interpreter import Interpreter

d, threads = pathlib.Path(sys.argv[1]), int(sys.argv[2])
io = np.load(d / "vocoder_T300_io.npz")
args = [io[f"arr_{i}"].astype(np.float32) for i in range(6)]
x_pre_ref = io["x_pre"]

def run(path, inputs, reps=3):
    it = Interpreter(model_path=str(path), num_threads=threads); it.allocate_tensors()
    ins = sorted(it.get_input_details(), key=lambda x: x["index"])
    for det, a in zip(ins, inputs): it.set_tensor(det["index"], a)
    it.invoke()
    t = time.perf_counter()
    for _ in range(reps): it.invoke()
    ms = (time.perf_counter() - t) / reps * 1000
    outs = [it.get_tensor(o["index"]) for o in it.get_output_details()]
    return outs, ms, [(det["name"], det["shape"].tolist()) for det in ins]

outs, voc_ms, ins = run(d / "kokoro_vocoder_T300.tflite", args)
print("vocoder inputs", ins)
x_pre = max(outs, key=lambda o: o.size)
c = np.corrcoef(x_pre.ravel(), x_pre_ref.ravel())[0, 1]
rel = np.linalg.norm(x_pre - x_pre_ref) / np.linalg.norm(x_pre_ref)
print(f"vocoder x_pre corr {c:.5f} rel L2 err {rel:.2e}  {voc_ms:.0f} ms ({threads} threads)")
(audio_t,), tail_ms, _ = run(d / "kokoro_tail_T300.tflite", [x_pre_ref])
(audio_c,), _, _ = run(d / "kokoro_tail_T300.tflite", [x_pre])
import torch  # reference tail in torch is the same graph; compare the chain against tail(x_pre_ref)
print(f"tail {tail_ms:.0f} ms; chain audio vs tflite-tail(ref x_pre) corr {np.corrcoef(audio_c.ravel(), audio_t.ravel())[0, 1]:.5f}")
audio_s = audio_t.size / 24000
print(f"audio {audio_s:.2f}s -> vocoder+tail rtf {audio_s / ((voc_ms + tail_ms) / 1000):.2f}x on host CPU (XNNPACK, {threads} threads)")
