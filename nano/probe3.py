"""Frame mask: zero the padded frames before every depthwise conv of the vector estimator; same output as unpadded?"""
import glob, onnx, numpy as np, onnxruntime as ort, json
from onnx import helper, TensorProto
d = glob.glob('/Users/leongo/.cache/huggingface/hub/models--pnnbao-ump--VieNeu-TTS-v3-Nano/snapshots/*/')[0]
m = onnx.load(d + 'vector_estimator.onnx')
m.graph.input.append(helper.make_tensor_value_info('x_mask', TensorProto.FLOAT, [1, 1, 'T']))
m.graph.initializer.append(onnx.numpy_helper.from_array(np.array([1], np.int64), 'mask_axes'))
new = []
cos = next(n for n in m.graph.node if n.name == '/ConstantOfShape')
dtype = cos.attribute[0].t.data_type if cos.attribute else TensorProto.FLOAT
print('users of ConstantOfShape:', [n.name for n in m.graph.node if cos.output[0] in n.input])
for n in m.graph.node:
    if n is cos:
        # the frame mask the model builds as all ones: the real one, so its length is the real frame count
        new.append(helper.make_node('Squeeze', ['x_mask', 'mask_axes'], ['x_mask_2d']))
        new.append(helper.make_node('Cast', ['x_mask_2d'], [cos.output[0]], to=dtype))
        continue
    if n.op_type == 'Conv' and 'dw' in n.name:
        masked = n.input[0] + '_masked'
        new.append(helper.make_node('Mul', [n.input[0], 'x_mask'], [masked]))
        n.input[0] = masked
    new.append(n)
del m.graph.node[:]
m.graph.node.extend(new)
s = ort.InferenceSession(m.SerializeToString(), providers=['CPUExecutionProvider'])
ve = ort.InferenceSession(d + 'vector_estimator.onnx', providers=['CPUExecutionProvider'])
r = np.random.default_rng(0)
L, ctx = 60, r.standard_normal((1, 60, 256)).astype(np.float32)
base = dict(t=np.array([.3], np.float32), ctx=ctx, ctx_mask=np.ones((1, L), bool), spk=r.standard_normal((1, 192)).astype(np.float32),
            style=r.standard_normal((1, 50, 256)).astype(np.float32))
for T, P in [(80, 16), (37, 91), (150, 10)]:
    x = r.standard_normal((1, 144, T)).astype(np.float32)
    a = ve.run(None, dict(base, x=x))[0]
    xp = np.concatenate([x, r.standard_normal((1, 144, P)).astype(np.float32)], 2)
    mask = np.concatenate([np.ones((1, 1, T)), np.zeros((1, 1, P))], 2).astype(np.float32)
    b = s.run(None, dict(base, x=xp, x_mask=mask))[0]
    print(T, '+', P, 'max diff', np.abs(b[..., :T] - a).max(), 'of', np.abs(a).max())
