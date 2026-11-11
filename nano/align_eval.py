"""Word times for Nano (bookmuse docs/listening.md, Nano "Word times"): renders a few prose chunks per voice with every block's text
cross-attention exposed (`ctext` softmax), finds each word's first frame from each block (last step or the mean over steps; a monotonic
path or the expected token position) and scores the word starts against Whisper large-v3-turbo's word times, next to the audio
estimate the apps used before (`NanoTimings`, ported). Speed 1, as the apps render.

    uv run --with mlx-whisper --with scipy python -I align_eval.py render [--android]
    uv run --with mlx-whisper --with scipy python -I align_eval.py score [--android]

`--android` renders on the Android pack's int8 step graph (dist/android/vi-nano/step.onnx, 8 steps) instead of the package's fp32 graphs
(16 steps). Results in work/nano/align[_android]/ (Whisper's words are cached in whisper.json).

2026-11-11, 6 voices × 5 chunks (906 words): block 1, last step, path: 0.037 s mean off (90 % under 0.08 s), int8 0.036 s;
NanoTimings 0.16 s. Blocks 0 and 3 0.05–0.06 s; block 2 doesn't follow the text.
"""
import glob, json, math, os, pickle, re, sys, unicodedata, difflib
from pathlib import Path
import numpy as np

S = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--pnnbao-ump--VieNeu-TTS-v3-Nano/snapshots/*/"))[0]
IOS = Path(__file__).resolve().parent.parent.parent / "bookmuse_ios/Packages/GlossmarkKit/Sources/GMReader/OnDeviceVoices/vi"
OUT = Path(__file__).resolve().parent.parent / "work/nano/align"
CFG = json.load(open(S + "config.json"))
RATE, HOP = 24000, 1536
VOICES = ["tam_an", "hoang_long", "thu_ha", "ba_phuc", "dieu_linh", "hai_dang"]
TEXT = """Buổi sáng hôm ấy, trời mưa nhẹ. Cô bé ngồi bên cửa sổ, nhìn những giọt nước chảy dài trên kính, và nghĩ về người bà đã đi xa.
Ngày xưa, bà thường kể cho cô nghe những câu chuyện cổ tích bên bếp lửa. Giọng bà trầm và ấm, như tiếng gió thổi qua rặng tre đầu làng.
Cuốn sách mà anh đọc tuần trước nói rằng thói quen nhỏ mỗi ngày có thể thay đổi cả cuộc đời. Nhưng thay đổi không đến ngay, nó cần thời gian và sự kiên nhẫn.
Khi thành phố lên đèn, dòng người vội vã trở về nhà. Tiếng còi xe hòa lẫn với tiếng rao của những người bán hàng rong trên vỉa hè.
Ông lão đánh cá ra khơi từ lúc trời còn tối. Biển lặng, sóng vỗ nhè nhẹ vào mạn thuyền, và những vì sao cuối cùng đang tắt dần ở phía chân trời."""


def ve_with_attn():
    import onnx
    from onnx import helper, TensorProto as TP
    m = onnx.load(S + "vector_estimator.onnx")
    names = []
    for n in m.graph.node:
        if n.op_type == "Softmax" and "ctext" in n.name:
            names.append(n.output[0])
            m.graph.output.append(helper.make_tensor_value_info(n.output[0], TP.FLOAT, None))
    p = OUT / "ve_attn.onnx"
    onnx.save(m, str(p))
    return str(p), names


ANDROID = "--android" in sys.argv
if ANDROID:
    OUT = OUT.parent / "align_android"
APK = Path(__file__).resolve().parent.parent / "dist/android/vi-nano"


def android_step():
    import onnx
    from onnx import helper, TensorProto as TP
    m = onnx.load(str(APK / "step.onnx"))
    names = [n.output[0] for n in m.graph.node if n.op_type == "Softmax" and "ctext" in n.name]
    for nm in names:
        m.graph.output.append(helper.make_tensor_value_info(nm, TP.FLOAT, None))
    p = OUT / "step_attn.onnx"
    onnx.save(m, str(p))
    return str(p)


def render():
    import onnxruntime as ort
    from vieneu_utils.phonemize_text import phonemize_text_with_emotions, normalize_to_chunks_v3_with_gaps
    from vieneu_utils.core_utils import edge_silence
    OUT.mkdir(parents=True, exist_ok=True)
    so = lambda p: ort.InferenceSession(p, providers=["CPUExecutionProvider"])
    te, dp, cd = so(S + "text_encoder.onnx"), so(S + "duration_predictor.onnx"), so(S + "codec_decoder.onnx")
    if ANDROID:
        st = so(android_step())
    else:
        vep, attn_names = ve_with_attn()
        ve = so(vep)
    C = np.load(S + "constants.npz")
    nspk, nsty = C["null_spk"].astype(np.float32).reshape(1, 192), C["null_style"].astype(np.float32).reshape(1, 50, 256)
    nctx = te.run(None, {"ids": np.array([[CFG["bos_id"], CFG["eos_id"]]], np.int64), "style": nsty})[0]
    nmask = np.ones((1, 2), bool)
    ios = json.loads((IOS / "config.json").read_text())
    chunks, _ = normalize_to_chunks_v3_with_gaps(TEXT, max_chars=180)
    rows = []
    for vk in VOICES:
        v = np.fromfile(IOS / "voices" / f"{vk}.f32", np.float32)
        spk, style = v[:192].reshape(1, 192), v[192:].reshape(1, 50, 256)
        speed = 1.0  # the app renders at 1 and paces the audio (AudioTempo)
        for ci, text in enumerate(chunks):
            ph = phonemize_text_with_emotions(text)
            ids, word_of = [CFG["bos_id"]], [-1]
            w = 0
            for c in ph:
                if c == " ":
                    w += 1
                if c in CFG["vocab"]:
                    ids.append(CFG["vocab"][c]); word_of.append(w if c != " " else -1)
            ids.append(CFG["eos_id"]); word_of.append(-1)
            ids_a = np.array([ids], np.int64)
            ctx = te.run(None, {"ids": ids_a, "style": style})[0]
            m = np.ones((1, len(ids)), bool)
            secs = math.exp(float(dp.run(None, {"ctx": ctx, "ctx_mask": m, "spk": spk})[0].reshape(-1)[0])) / speed
            T = max(2, int(round(min(secs, 15.0) * CFG["flow_fps"])))
            x = np.random.default_rng(ci * 31 + len(vk)).standard_normal((1, 144, T)).astype(np.float32)
            NS = 8 if ANDROID else 16
            u = np.linspace(0, 1, NS + 1); tg = u + -1 * (np.cos(np.pi / 2 * u) - 1 + u)
            acc = None
            if ANDROID:
                L = len(ids)
                pair = np.zeros((2, L, 256), np.float32); pair[0], pair[1, :2] = ctx[0], nctx[0]
                pm = np.zeros((2, L), bool); pm[0], pm[1, :2] = True, True
                for i in range(NS):
                    outs = st.run(None, {"x": x, "t": np.array([tg[i]], np.float32), "dt": np.array([tg[i + 1] - tg[i]], np.float32),
                                         "cfg": np.array([3.0], np.float32), "ctx": pair, "ctx_mask": pm,
                                         "spk": np.concatenate([spk, nspk]), "style": np.concatenate([style, nsty])})
                    x = outs[0]
                    att = [o for o, out in zip(outs, st.get_outputs()) if "ctext" in out.name]  # the pack's own `align` aside
                    cur = np.stack([a[0].reshape(-1, a.shape[-2], a.shape[-1]) for a in att])
                    acc = cur / NS if acc is None else acc + cur / NS
            for i in range(0 if ANDROID else 16):
                t = np.array([tg[i]], np.float32)
                outs = ve.run(None, {"x": x, "t": t, "ctx": ctx, "ctx_mask": m, "spk": spk, "style": style})
                vc, att = outs[0], outs[1:]
                vu = ve.run(None, {"x": x, "t": t, "ctx": nctx, "ctx_mask": nmask, "spk": nspk, "style": nsty})[0]
                x = x + np.float32(tg[i + 1] - tg[i]) * (vu + 3.0 * (vc - vu))
                cur = np.stack([a.reshape(-1, a.shape[-2], a.shape[-1]) for a in att])  # [blk, H, T, L]
                acc = cur / 16 if acc is None else acc + cur / 16
            wav = np.clip(cd.run(None, {"x": x})[0].reshape(-1)[: T * HOP], -1, 1)
            lead, tail = edge_silence(wav, RATE)
            keep = int(0.04 * RATE); a = max(0, lead - keep)
            rows.append(dict(voice=vk, chunk=ci, text=text, ph=ph, word_of=word_of, T=T, attn=np.stack([cur, acc]).astype(np.float16), wav=wav, trim=a,
                             speed=speed))
            print(vk, ci, T, len(ids), cur.shape, flush=True)
    pickle.dump(rows, open(OUT / "rows.pkl", "wb"))


# ---------- scoring ----------
TONES = {"̀": "2", "́": "1", "̃": "3", "̉": "4", "̣": "5"}


def canon(w):
    w = unicodedata.normalize("NFC", w).lower()
    w = "".join(re.findall(r"[^\W_]+", w))
    d = unicodedata.normalize("NFD", w)
    tone = "".join(TONES[c] for c in d if c in TONES)
    return unicodedata.normalize("NFC", "".join(c for c in d if c not in TONES)) + tone


def whisper_words(wav):
    import mlx_whisper
    from scipy.signal import resample_poly
    res = mlx_whisper.transcribe(resample_poly(wav, 2, 3).astype(np.float32), path_or_hf_repo="mlx-community/whisper-large-v3-turbo",
                                 language="vi", temperature=0.0, condition_on_previous_text=False, word_timestamps=True)
    out = []
    for s in res["segments"]:
        for w in s.get("words", []):
            for k, part in enumerate(w["word"].split()):  # whisper rarely joins; keep first part's time
                out.append((canon(part), w["start"]))
    return out


def baseline(words_text, wav):
    """NanoTimings.pieceStarts at word granularity (piece = word + trailing space), in seconds."""
    total = len(wav); win = RATE // 100; thr = 10 ** (-40 / 20)
    n = total // win
    q = np.abs(wav[: n * win]).reshape(n, win).mean(1) < thr
    pauses, s = [], None
    for i in range(n):
        if q[i] and s is None: s = i
        if not q[i] and s is not None:
            if s > 0 and i - s >= 15: pauses.append((s * win, i * win))
            s = None
    lens = [len(w) + 1 for w in words_text]
    sent = [i + 1 for i, w in enumerate(words_text[:-1]) if w.rstrip("\"'”’)»]")[-1:] in list(".!?…")]
    anchors = [(0, 0)]; used = -1
    long = [(k, p) for k, p in enumerate(pauses) if p[1] - p[0] >= RATE // 4]
    for piece in sent:
        lp, lf = anchors[-1]
        before = sum(lens[lp:piece]); after = sum(lens[lp:])
        exp = lf + int((total - lf) * before / max(after, 1))
        cand = [(k, p) for k, p in long if k > used]
        if cand:
            k, p = min(cand, key=lambda kp: abs(kp[1][1] - exp))
            if abs(p[1] - exp) <= RATE * 3 // 2:
                used = k; anchors.append((piece, p[1]))
    speech = lambda f: f - sum(max(0, min(b, f) - a) for a, b in pauses)

    def real(sf):
        f = sf
        for a, b in pauses:
            if f >= a: f += b - a
        return f
    starts = [0] * len(words_text)
    for i, (ap, af) in enumerate(anchors):
        ep, ef = anchors[i + 1] if i + 1 < len(anchors) else (len(words_text), total)
        lo, hi = speech(af), speech(ef)
        ws = lens[ap:ep]; sm = max(sum(ws), 1); done = 0
        for o, wt in enumerate(ws):
            starts[ap + o] = min(real(lo + (hi - lo) * done // sm), total); done += wt
    for i in range(1, len(starts)): starts[i] = max(starts[i], starts[i - 1])
    return [s / RATE for s in starts]


def word_matrix(A, word_of, W):
    """[T, L] attention → [T, W] mass per word."""
    M = np.zeros((A.shape[0], W))
    for l, w in enumerate(word_of):
        if w >= 0: M[:, w] += A[:, l]
    return M


def dp_starts(M):
    """Monotonic path frame→word (stay or +1), maximizing sum log mass; first frame of each word. Words get ≥1 frame."""
    T, W = M.shape
    if W > T: return None
    L = np.log(M + 1e-6)
    D = np.full((T, W), -np.inf); B = np.zeros((T, W), int)
    D[0, 0] = L[0, 0]
    for t in range(1, T):
        stay = D[t - 1]; move = np.concatenate([[-np.inf], D[t - 1, :-1]])
        B[t] = move > stay
        D[t] = np.maximum(stay, move) + L[t]
    w = W - 1; starts = [0] * W
    for t in range(T - 1, 0, -1):
        if B[t, w]: starts[w] = t; w -= 1
    starts[0] = 0
    return starts


def centroid_starts(A, word_of, W):
    """Expected token position per frame (cumulative max); a word starts at the first frame reaching its first token."""
    pos = np.maximum.accumulate((A * np.arange(A.shape[1])).sum(1))
    first = [word_of.index(w) for w in range(W)]
    return [int(np.searchsorted(pos, f - 0.5)) for f in first]


def score():
    rows = pickle.load(open(OUT / "rows.pkl", "rb"))
    cache = OUT / "whisper.json"
    wcache = json.loads(cache.read_text()) if cache.exists() else {}
    methods = {}
    fr = 1 / CFG["flow_fps"]
    for r in rows:
        key = f"{r['voice']}/{r['chunk']}"
        trimmed = r["wav"][r["trim"]:]
        if key not in wcache:
            wcache[key] = whisper_words(trimmed); cache.write_text(json.dumps(wcache, ensure_ascii=False))
        ww = wcache[key]
        words_text = r["text"].split()
        W = max(r["word_of"]) + 1
        assert W == len(words_text), (W, len(words_text), r["text"], r["ph"])
        ref = [canon(w) for w in words_text]
        sm = difflib.SequenceMatcher(a=ref, b=[w for w, _ in ww], autojunk=False)
        pairs = [(b.a + k, ww[b.b + k][1]) for b in sm.get_matching_blocks() for k in range(b.size)]
        sh = r["trim"] / RATE
        cands = {"baseline": baseline(words_text, trimmed)}
        A_all = r["attn"].astype(np.float32)  # [last|mean, blk, H, T, L]
        for blk in range(A_all.shape[1]):
            for tag, A in [("last", A_all[0, blk].mean(0)), ("mean", A_all[1, blk].mean(0))]:
                M = word_matrix(A, r["word_of"], W)
                d = dp_starts(M)
                if d: cands[f"b{blk}-{tag}-dp"] = [f * fr - sh for f in d]
                cands[f"b{blk}-{tag}-cen"] = [f * fr - sh for f in centroid_starts(A / A.sum(1, keepdims=True), r["word_of"], W)]
        A = A_all[0].mean((0, 1))
        M = word_matrix(A, r["word_of"], W); d = dp_starts(M)
        if d: cands["allblk-last-dp"] = [f * fr - sh for f in d]
        A = A_all[1].mean((0, 1))
        M = word_matrix(A, r["word_of"], W); d = dp_starts(M)
        if d: cands["all-dp"] = [f * fr - sh for f in d]
        for name, st in cands.items():
            for i, ts in pairs:
                if i == 0: continue  # first word: both trivially at the start
                methods.setdefault(name, {}).setdefault(r["voice"], []).append(st[i] - ts)
    print(f"{'method':22s} {'mean|e|':>8s} {'p90':>6s} {'bias':>6s}  per-voice mean|e|")
    order = sorted(methods, key=lambda n: np.mean(np.abs(np.concatenate([np.array(v) for v in methods[n].values()]))))
    for name in order:
        e = np.concatenate([np.array(v) for v in methods[name].values()])
        pv = " ".join(f"{np.mean(np.abs(v)):.3f}" for v in methods[name].values())
        print(f"{name:22s} {np.mean(np.abs(e)):8.3f} {np.percentile(np.abs(e), 90):6.3f} {np.median(e):+6.3f}  {pv}  (n={len(e)})")
    print("voices:", list(next(iter(methods.values())).keys()))
    for name in ["baseline", "b1-last-dp", "allblk-last-dp"]:
        e = np.abs(np.concatenate([np.array(v) for v in methods[name].values()]))
        print(f"{name}: max {e.max():.3f}  >100ms {np.mean(e > 0.1) * 100:.1f}%  >150ms {np.mean(e > 0.15) * 100:.1f}%  >250ms {np.mean(e > 0.25) * 100:.1f}%")
    total = sum(len(r["text"].split()) - 1 for r in rows)
    print(f"words matched to Whisper: {len(np.concatenate([np.array(v) for v in methods['baseline'].values()]))} of {total}")


if __name__ == "__main__":
    {"render": render, "score": score}[sys.argv[1]]()
