import glob, onnx, numpy as np, onnxruntime as ort, json
from onnx import shape_inference, helper
d = glob.glob('/Users/leongo/.cache/huggingface/hub/models--pnnbao-ump--VieNeu-TTS-v3-Nano/snapshots/*/')[0]
m = onnx.load(d + 'vector_estimator.onnx')
# expose every Softmax and Conv output + ReduceSum to read shapes at runtime
names = [n.output[0] for n in m.graph.node if n.op_type in ('Softmax', 'Conv')]
info = {n.output[0]: (n.op_type, n.name, [a for a in n.attribute if a.name in ('group', 'kernel_shape', 'dilations', 'pads', 'axes')]) for n in m.graph.node if n.op_type in ('Softmax','ReduceSum','Conv')}
for nm in names: m.graph.output.append(helper.make_tensor_value_info(nm, onnx.TensorProto.FLOAT, None))
s = ort.InferenceSession(m.SerializeToString(), providers=['CPUExecutionProvider'])
T, L = 80, 60
r = np.random.default_rng(0)
out = s.run(None, {'x': r.standard_normal((1,144,T)).astype(np.float32), 't': np.array([.3],np.float32), 'ctx': r.standard_normal((1,L,256)).astype(np.float32),
  'ctx_mask': np.ones((1,L),bool), 'spk': r.standard_normal((1,192)).astype(np.float32), 'style': r.standard_normal((1,50,256)).astype(np.float32)})
for nm, o in zip(names, out[1:]):
    op, n, attrs = info[nm]
    print(op, n[:60], o.shape, [(a.name, list(a.ints) or a.i) for a in attrs])
