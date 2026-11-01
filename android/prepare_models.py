"""The Android model (`int8convsafe`) from the iOS ONNX packs in ../work/ios-* (fetch_sources.sh), into ../work/int8convsafe-<lang>/.

    python -I prepare_models.py          (venv: onnx==1.19.0, onnxruntime==1.31.0, numpy)

Dynamic int8 for MatMul, Gemm and Conv (ConvInteger: Kokoro's time is in the vocoder convs, so int8 only pays when they are
quantized), with the pitch path left fp32 (F0/N predictor, F0_proj/N_proj, conv_post: quantizing them drops the pitch ~0.4 st).
The packs compute conv weights from weight norm at run time, so ORT's basic optimizer constant-folds them first.
From the Android P6a spike (bookmuse_android `spikes/p6a-tts/kokoro/tools/prepare_models.py`, which also makes the bench variants).
"""
import pathlib
import numpy as np, onnx
from onnx import numpy_helper
from onnxruntime.quantization import quantize_dynamic, QuantType

W = pathlib.Path(__file__).resolve().parent.parent / "work"
PACK = {"en": "ios-en-v1", "vi": "ios-vi-v1"}


def fold_fp16_casts(src, dst):
    """The packs store weights as fp16 initializers + Cast to fp32; the quantizer only sees initializers, so fold them."""
    m = onnx.load(src)
    inits = {i.name: i for i in m.graph.initializer}
    keep = []
    for n in m.graph.node:
        src_init = inits.get(n.input[0]) if n.op_type == "Cast" else None
        to = next((a.i for a in n.attribute if a.name == "to"), None)
        if src_init is not None and src_init.data_type == onnx.TensorProto.FLOAT16 and to == onnx.TensorProto.FLOAT:
            m.graph.initializer.append(numpy_helper.from_array(numpy_helper.to_array(src_init).astype(np.float32), n.output[0]))
        else:
            keep.append(n)
    del m.graph.node[:]; m.graph.node.extend(keep)
    onnx.save(m, dst)


def fold_constants(src, dst):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC  # constant folding, no fused contrib ops
    so.optimized_model_filepath = str(dst)
    ort.InferenceSession(str(src), so, providers=["CPUExecutionProvider"])


for lang, src in PACK.items():
    out = W / f"int8convsafe-{lang}"; out.mkdir(exist_ok=True)
    if not (out / "model.onnx").exists():
        fold_fp16_casts(W / src / "model.onnx", out / "fp32.onnx")
        fold_constants(out / "fp32.onnx", out / "folded.onnx"); (out / "folded.onnx").replace(out / "fp32.onnx")
        # (noise_convs stay int8: excluding them makes ORT 1.31's quantizer emit a graph that outputs NaN)
        pitch = ("/F0.", "/N.", "/F0_proj/", "/N_proj/", "conv_post")
        exclude = [n.name for n in onnx.load(out / "fp32.onnx").graph.node
                   if n.op_type == "Conv" and any(k in n.name for k in pitch)]
        quantize_dynamic(str(out / "fp32.onnx"), str(out / "model.onnx"), op_types_to_quantize=["MatMul", "Gemm", "Conv"],
                         weight_type=QuantType.QInt8, nodes_to_exclude=exclude)
        (out / "fp32.onnx").unlink()
    print(f"int8convsafe-{lang}", round((out / "model.onnx").stat().st_size / 1e6, 1), "MB")
