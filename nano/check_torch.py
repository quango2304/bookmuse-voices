"""onnx2torch modules against ONNX Runtime, padded inputs and batch 2 included."""
import json
import numpy as np, onnxruntime as ort, torch
from graphs import SNAPSHOT, torch_graph
S = lambda f: ort.InferenceSession(SNAPSHOT + f + '.onnx', providers=['CPUExecutionProvider'])
r = np.random.default_rng(0)
v = json.load(open('../work/nano/voices/thien_tam_duc.json'))
spk = np.array(v['speaker_emb'], np.float32)[None]; style = np.array(v['style'], np.float32)[None]
L, Lp, T, Tp = 60, 256, 70, 96
ids = r.integers(10, 80, (1, L)); ids[0, 0], ids[0, -1] = 1, 2
idsp = np.concatenate([ids, np.zeros((1, Lp - L), np.int64)], 1)
te = torch_graph('text_encoder')
with torch.no_grad():
    ctx_t = te(torch.tensor(idsp), torch.tensor(style)).numpy()
ctx = S('text_encoder').run(None, {'ids': ids, 'style': style})[0]
print('text', np.abs(ctx_t[:, :L] - ctx).max())
mask = np.zeros((1, Lp), bool); mask[:, :L] = True
dp = torch_graph('duration_predictor')
with torch.no_grad():
    d_t = dp(torch.tensor(ctx_t), torch.tensor(mask), torch.tensor(spk)).numpy()
print('dur', d_t, S('duration_predictor').run(None, {'ctx': ctx, 'ctx_mask': np.ones((1, L), bool), 'spk': spk})[0])
x = r.standard_normal((1, 144, T)).astype(np.float32); t = np.array([0.4], np.float32)
ref = S('vector_estimator').run(None, {'x': x, 't': t, 'ctx': ctx, 'ctx_mask': np.ones((1, L), bool), 'spk': spk, 'style': style})[0]
ve = torch_graph('vector_estimator')
xp = np.concatenate([x, np.zeros((1, 144, Tp - T), np.float32)], 2); xm = np.zeros((1, 1, Tp), np.float32); xm[..., :T] = 1
with torch.no_grad():
    a = ve(torch.tensor(xp), torch.tensor(t), torch.tensor(ctx_t), torch.tensor(mask), torch.tensor(spk), torch.tensor(style), torch.tensor(xm)).numpy()
    print('ve padded', np.abs(a[..., :T] - ref).max())
    b = ve(torch.tensor(xp).repeat(2, 1, 1), torch.tensor(t).repeat(2), torch.tensor(ctx_t).repeat(2, 1, 1), torch.tensor(mask).repeat(2, 1),
           torch.tensor(spk).repeat(2, 1), torch.tensor(style).repeat(2, 1, 1), torch.tensor(xm).repeat(2, 1, 1)).numpy()
    print('ve batch 2', b.shape, np.abs(b[0:1, :, :T] - ref).max(), np.abs(b[1:2, :, :T] - ref).max())
dec = torch_graph('codec_decoder')
with torch.no_grad():
    w = dec(torch.tensor(xp)).numpy()
wr = S('codec_decoder').run(None, {'x': x})[0]
print('codec', w.shape, np.abs(w[..., :wr.shape[-1]] - wr).max())
