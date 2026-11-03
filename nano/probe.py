"""Does padding change Nano's outputs? ctx padded with a false mask; x (frames) padded."""
import glob, numpy as np, onnxruntime as ort, json
d = glob.glob('/Users/leongo/.cache/huggingface/hub/models--pnnbao-ump--VieNeu-TTS-v3-Nano/snapshots/*/')[0]
S = lambda f: ort.InferenceSession(d + f + '.onnx', providers=['CPUExecutionProvider'])
te, dp, ve, dec = S('text_encoder'), S('duration_predictor'), S('vector_estimator'), S('codec_decoder')
cfg = json.load(open(d + 'config.json'))
v = json.load(open('../work/nano/voices/thien_tam_duc.json'))
spk = np.array(v['speaker_emb'], np.float32)[None]; style = np.array(v['style'], np.float32)[None]
rng = np.random.default_rng(0)
L = 60
ids = rng.integers(10, 80, (1, L)); ids[0, 0] = cfg['bos_id']; ids[0, -1] = cfg['eos_id']
ctx = te.run(None, {'ids': ids, 'style': style})[0]
idsp = np.concatenate([ids, np.full((1, 20), cfg['pad_id'])], 1)
ctxp = te.run(None, {'ids': idsp, 'style': style})[0]
print('text enc padded ids: max diff on real rows', np.abs(ctxp[:, :L] - ctx).max())
m = np.ones((1, L), bool); mp = np.concatenate([m, np.zeros((1, 20), bool)], 1)
print('dur', dp.run(None, {'ctx': ctx, 'ctx_mask': m, 'spk': spk})[0], dp.run(None, {'ctx': ctxp, 'ctx_mask': mp, 'spk': spk})[0])
T = 80
x = rng.standard_normal((1, 144, T)).astype(np.float32); t = np.array([0.3], np.float32)
a = ve.run(None, {'x': x, 't': t, 'ctx': ctx, 'ctx_mask': m, 'spk': spk, 'style': style})[0]
b = ve.run(None, {'x': x, 't': t, 'ctx': ctxp, 'ctx_mask': mp, 'spk': spk, 'style': style})[0]
print('ve ctx padded: max diff', np.abs(a - b).max(), 'scale', np.abs(a).max())
xp = np.concatenate([x, np.zeros((1, 144, 8), np.float32)], 2)
c = ve.run(None, {'x': xp, 't': t, 'ctx': ctx, 'ctx_mask': m, 'spk': spk, 'style': style})[0]
print('ve x padded 8 zero frames: max diff on real frames', np.abs(c[..., :T] - a).max(), 'per-frame', np.abs(c[..., :T] - a).max(1).round(2)[0][-12:])
w = dec.run(None, {'x': x})[0]; wp = dec.run(None, {'x': xp})[0]
print('dec samples per frame', w.shape[-1] / T, 'padded: max diff on real part', np.abs(wp[..., :w.shape[-1]] - w).max())
