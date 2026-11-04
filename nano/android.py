#!/usr/bin/env python3
"""VieNeu-TTS v3 Nano for Android (ONNX Runtime, CPU): the Vietnamese pack and its release zip, from the package's ONNX graphs and
the voices, settings and dictionary the iOS app ships (bookmuse_ios GMReader/OnDeviceVoices/vi, made by export_assets.py).

- step.onnx: one Euler step with classifier-free guidance, by ONNX surgery on vector_estimator.onnx (no torch round trip): x and t
  are doubled to a batch of 2, so the conditional and the null pass are one run (ctx, ctx_mask, spk and style come as a pair, the
  null text padded to the chunk's length and masked), then x_next = x + dt · (v_null + cfg · (v − v_null)). Dynamic T and L.
  MatMul and Gemm weights in int8 (dynamic quantization, per tensor): 40 MB and ~1.9× the speed of fp32 on ARM CPUs; F0 moves
  0.65 st from fp32 on the same noise, where two fp32 seeds differ by 2–4 st (T52).
- text_encoder.onnx, codec_decoder.onnx: weights stored in fp16 and cast to fp32 when the session loads (half the size, the same
  output: codec SNR 74 dB against fp32). duration_predictor.onnx as it is.
- config.json (vocab, ids, frame rate, hop, steps, cfg, sway, each voice's speed), null.bin and voices/<key>.bin (x-vector 192 +
  style 50 × 256, float32 little-endian), sea_g2p.bin.

Writes dist/android/vi-nano/ and dist/bookmuse-voices-android-vi-<ZIP_VERSION>.zip (deterministic: sorted entries, fixed timestamps) and
prints the pin for bookmuse_android voices/build.gradle.kts.

    uv run --no-project --python 3.12 --with onnx==1.19.0 --with onnxruntime==1.31.0 --with numpy python -I nano/android.py
"""
import glob
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto as TP, helper, numpy_helper as nh
from onnxruntime.quantization import QuantType, quantize_dynamic
from onnxruntime.quantization.shape_inference import quant_pre_process

ROOT = Path(__file__).resolve().parent.parent
SNAPSHOT = glob.glob(str(Path.home() / ".cache/huggingface/hub/models--pnnbao-ump--VieNeu-TTS-v3-Nano/snapshots/*/"))[0] + "/"
IOS = ROOT.parent / "bookmuse_ios/Packages/GlossmarkKit/Sources/GMReader/OnDeviceVoices/vi"
WORK = ROOT / "work/nano/android"
OUT = ROOT / "dist/android/vi-nano"
ZIP_VERSION = 5  # bookmuse-voices-android-vi-<n>.zip (1 and 2 were an older model)


def step_graph() -> onnx.ModelProto:
    m = onnx.load(SNAPSHOT + "vector_estimator.onnx")
    g = m.graph
    renamed = {"x": "ve_x", "t": "ve_t"}
    for n in g.node:
        for i, name in enumerate(n.input):
            if name in renamed:
                n.input[i] = renamed[name]
    estimate = g.output[0].name
    pairs = [i for i in g.input if i.name not in renamed]
    for i in pairs:
        dim = i.type.tensor_type.shape.dim[0]
        dim.ClearField("dim_value")
        dim.dim_param = "B"
    del g.input[:]
    g.input.extend([helper.make_tensor_value_info("x", TP.FLOAT, [1, 144, "T"]), helper.make_tensor_value_info("t", TP.FLOAT, [1]),
                    helper.make_tensor_value_info("dt", TP.FLOAT, [1]), helper.make_tensor_value_info("cfg", TP.FLOAT, [1])] + pairs)
    for name, value in [("s0", 0), ("s1", 1), ("s2", 2), ("axis0", 0)]:
        g.initializer.append(nh.from_array(np.array([value], np.int64), name))
    before = [helper.make_node("Concat", ["x", "x"], ["ve_x"], axis=0), helper.make_node("Concat", ["t", "t"], ["ve_t"], axis=0)]
    after = [
        helper.make_node("Slice", [estimate, "s0", "s1", "axis0"], ["guided_cond"]),
        helper.make_node("Slice", [estimate, "s1", "s2", "axis0"], ["guided_null"]),
        helper.make_node("Sub", ["guided_cond", "guided_null"], ["guided_diff"]),
        helper.make_node("Mul", ["guided_diff", "cfg"], ["guided_scaled"]),
        helper.make_node("Add", ["guided_null", "guided_scaled"], ["guided"]),
        helper.make_node("Mul", ["guided", "dt"], ["guided_dx"]),
        helper.make_node("Add", ["x", "guided_dx"], ["x_next"]),
    ]
    nodes = before + list(g.node) + after
    del g.node[:]
    g.node.extend(nodes)
    del g.output[:]
    g.output.extend([helper.make_tensor_value_info("x_next", TP.FLOAT, [1, 144, "T"])])
    del g.value_info[:]  # shapes inferred for a batch of 1
    onnx.checker.check_model(m)
    return m


def fp16_storage(m: onnx.ModelProto) -> onnx.ModelProto:
    """Float initializers of 1024 values or more kept as fp16, each cast back to fp32 (ONNX Runtime folds the cast at load)."""
    g = m.graph
    kept, casts = [], []
    for i in g.initializer:
        values = nh.to_array(i) if i.data_type == TP.FLOAT and int(np.prod(i.dims)) >= 1024 else None
        # Only what fp16 holds: a value past its range (a -1e9 mask fill) would become inf.
        if values is not None and np.abs(values).max() <= 60000:
            kept.append(nh.from_array(values.astype(np.float16), i.name + "_f16"))
            casts.append(helper.make_node("Cast", [i.name + "_f16"], [i.name], to=TP.FLOAT, name=i.name + "_cast"))
        else:
            kept.append(i)
    del g.initializer[:]
    g.initializer.extend(kept)
    nodes = casts + list(g.node)
    del g.node[:]
    g.node.extend(nodes)
    return m


def main():
    WORK.mkdir(parents=True, exist_ok=True)
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "voices").mkdir(parents=True)

    onnx.save(step_graph(), WORK / "step_fp32.onnx")
    quant_pre_process(str(WORK / "step_fp32.onnx"), str(WORK / "step_pre.onnx"), skip_symbolic_shape=True)
    quantize_dynamic(str(WORK / "step_pre.onnx"), str(OUT / "step.onnx"), weight_type=QuantType.QInt8,
                     op_types_to_quantize=["MatMul", "Gemm"], extra_options={"MatMulConstBOnly": True})
    for name in ["text_encoder", "codec_decoder"]:
        onnx.save(fp16_storage(onnx.load(SNAPSHOT + name + ".onnx")), OUT / f"{name}.onnx")
    shutil.copyfile(SNAPSHOT + "duration_predictor.onnx", OUT / "duration_predictor.onnx")

    ios = json.loads((IOS / "config.json").read_text())
    config = {k: ios[k] for k in ["vocab", "bos_id", "eos_id", "pad_id", "emotion_tags", "flow_fps", "hop", "max_seconds", "steps",
                                  "cfg", "sway", "speeds"]}
    (OUT / "config.json").write_text(json.dumps(config, ensure_ascii=False))
    shutil.copyfile(IOS / "null.f32", OUT / "null.bin")
    for key in sorted(config["speeds"]):
        shutil.copyfile(IOS / "voices" / f"{key}.f32", OUT / "voices" / f"{key}.bin")
    shutil.copyfile(IOS / "sea_g2p.bin", OUT / "sea_g2p.bin")

    files = sorted(p.relative_to(OUT).as_posix() for p in OUT.rglob("*") if p.is_file())
    for rel in files:
        print(rel, (OUT / rel).stat().st_size)
    zip_path = ROOT / "dist" / f"bookmuse-voices-android-vi-{ZIP_VERSION}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for rel in files:
            info = zipfile.ZipInfo(rel, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            info.file_size = (OUT / rel).stat().st_size
            with open(OUT / rel, "rb") as f, z.open(info, "w") as w:
                shutil.copyfileobj(f, w, 1 << 20)
    digest = hashlib.sha256(zip_path.read_bytes()).hexdigest()
    size = zip_path.stat().st_size
    print(f"{zip_path.name}: {size} bytes, sha256 {digest}")
    print(f'  voices/build.gradle.kts: VoicePack("vi", "{zip_path.name}", {size}L, "{digest}")')


if __name__ == "__main__":
    main()
