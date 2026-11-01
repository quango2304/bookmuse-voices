"""Core ML 7-stage chain vs PyTorch for the Vietnamese Kokoro checkpoint.

Runs the chain the way the iPhone will (dtypes and compute units as in Bookmuse), then compares
durations, the pitch curve (F0, which carries the tones), and the audio, and writes WAV pairs.

Usage: python parity.py <models dir> <ckpt dir> <out dir> [stage-routing: ane|cpu]
"""
import json, sys, pathlib, numpy as np, torch, soundfile as sf, coremltools as ct, librosa, vig2p
from sea_g2p import SEAPipeline
from kokoro import KModel

models, ckpt, out = map(pathlib.Path, sys.argv[1:4])
routing = sys.argv[4] if len(sys.argv) > 4 else "ane"
out.mkdir(parents=True, exist_ok=True)
NE = ct.ComputeUnit.CPU_AND_NE if routing == "ane" else ct.ComputeUnit.CPU_ONLY
CPU = ct.ComputeUnit.CPU_ONLY

def load(name, cu):
    return ct.models.MLModel(str(models / f"{name}.mlpackage"), compute_units=cu)

albert, post, align = load("KokoroAlbert", NE), load("KokoroPostAlbert", NE), load("KokoroAlignment", NE)
pros, noise, voc, tail = load("KokoroProsody", NE), load("KokoroNoise", CPU), load("KokoroVocoder", NE), load("KokoroTail", CPU)

EN = __import__("os").environ.get("LANG_CODE") == "en"
if EN:
    model = KModel(repo_id="hexgrad/Kokoro-82M").eval(); vocab = model.vocab
    from misaki import en
    g2p_en = en.G2P(trf=False, british=False)
else:
    model = KModel(repo_id="hexgrad/Kokoro-82M", config=str(ckpt / "config.json"), model=str(pathlib.Path(__import__("os").environ.get("VI_PTH", ckpt / "kokoro_vi.pth")))).eval()
    vocab = json.load(open(ckpt / "config.json"))["vocab"]
    sea = SEAPipeline(lang="vi")

SENTENCES = [
    "Ba bà đi bán bò, bả bá bã bạ.",
    "Hôm nay trời đẹp, chúng tôi đi dạo quanh hồ Gươm.",
    "Mẹ tôi thường kể rằng ngày xưa, cả làng chỉ có một chiếc đài cũ để nghe tin tức.",
    "Cuốn sách này đã thay đổi cách tôi nhìn nhận về lịch sử, về con người và về chính bản thân mình.",
    "Anh ấy hỏi: “Em có chắc không?” Cô ấy chỉ mỉm cười, rồi lặng lẽ bước đi trong mưa.",
    "Năm 1945, Việt Nam tuyên bố độc lập; đó là một bước ngoặt lớn trong lịch sử dân tộc.",
]
VOICES = ["mai_linh", "duc_an", "ngoc_huyen"]
if EN:
    SENTENCES = ["The harbour was quiet that morning, and the boats rocked gently against the pier.",
                 "He opened the old notebook carefully; inside were the names of forty-two birds.",
                 "\u201cAre you sure?\u201d she asked, and then walked away into the rain without another word.",
                 "In 1945, the war ended, and the long, difficult work of rebuilding the city began.",
                 "Learning to sail, he discovered, was mostly learning to notice the colour of the water.",
                 "Telemachus sailed from Ithaca to find news of his father."]
    VOICES = ["af_heart", "am_michael", "bf_emma"]

RETRIES = []
def predict(model, feed, stage):
    for attempt in range(3):
        try: return model.predict(feed)
        except Exception as e:
            RETRIES.append(stage); last = e
    raise last

def chain(ids, ref, forced=None):
    T = ids.shape[1]
    mask = np.ones((1, T), np.int32)
    s, timbre = ref[:, 128:], ref[:, :128]
    stage = "albert"
    try:
        o1 = predict(albert, {"input_ids": ids.astype(np.int32), "attention_mask": mask}, "albert")
        stage = "post_albert"
        o2 = predict(post, {"bert_dur": o1["bert_dur"].astype(np.float16), "input_ids": ids.astype(np.int32),
                           "style_s": s.astype(np.float16), "speed": np.array([1], np.float16), "attention_mask": mask}, "post")
        dur = np.maximum(np.round(o2["duration"].astype(np.float32)), 1).astype(np.int32)
        if forced is not None: dur = forced.astype(np.int32).reshape(dur.shape)
        stage = "alignment"
        o3 = predict(align, {"pred_dur": dur.reshape(1, -1), "d": o2["d"].astype(np.float16), "t_en": o2["t_en"].astype(np.float16)}, "align")
        stage = "prosody"
        o4 = predict(pros, {"en": o3["en"].astype(np.float16), "style_s": s.astype(np.float16)}, "pros")
        stage = "noise"
        o5 = predict(noise, {"F0_curve": o4["F0"].astype(np.float32), "style_timbre": timbre.astype(np.float32)}, "noise")
        stage = "vocoder"
        o6 = predict(voc, {"asr": o3["asr"].astype(np.float16), "F0_curve": o4["F0"].astype(np.float16),
                          "N_pred": o4["N"].astype(np.float16), "x_source_0": o5["x_source_0"].astype(np.float16),
                          "x_source_1": o5["x_source_1"].astype(np.float16), "style_timbre": timbre.astype(np.float16)}, "voc")
        stage = "tail"
        o7 = predict(tail, {"x_pre": o6["x_pre"].astype(np.float32)}, "tail")
    except Exception as e:
        raise RuntimeError(f"stage {stage}: {str(e)[:300]}")
    return o7["audio"].reshape(-1), dur.reshape(-1), o4["F0"].reshape(-1)

def torch_ref(ids, ref):
    with torch.no_grad():
        audio, dur = model.forward_with_tokens(torch.LongTensor(ids), torch.FloatTensor(ref), 1.0)
        # F0 from the same predictor, for a direct curve comparison.
        L = torch.LongTensor([ids.shape[1]]); m = torch.arange(ids.shape[1]).unsqueeze(0) + 1 > L.unsqueeze(1)
        bert = model.bert(torch.LongTensor(ids), attention_mask=(~m).int())
        d_en = model.bert_encoder(bert).transpose(-1, -2)
        s = torch.FloatTensor(ref[:, 128:])
        d = model.predictor.text_encoder(d_en, s, L, m)
        idx = torch.repeat_interleave(torch.arange(ids.shape[1]), dur)
        aln = torch.zeros(ids.shape[1], int(dur.sum())); aln[idx, torch.arange(int(dur.sum()))] = 1
        en = d.transpose(-1, -2) @ aln.unsqueeze(0)
        f0, _ = model.predictor.F0Ntrain(en, s)
    return audio.numpy().reshape(-1), dur.numpy().reshape(-1), f0.numpy().reshape(-1)

def pitch(y):
    f0, voiced, _ = librosa.pyin(y, fmin=60, fmax=500, sr=24000, frame_length=1200, hop_length=300)
    return f0, voiced

def mel_corr(a, b):
    n = min(len(a), len(b))
    A = librosa.power_to_db(librosa.feature.melspectrogram(y=a[:n], sr=24000, n_mels=80))
    B = librosa.power_to_db(librosa.feature.melspectrogram(y=b[:n], sr=24000, n_mels=80))
    return float(np.corrcoef(A.ravel(), B.ravel())[0, 1])

rows = []
for vi, voice in enumerate(VOICES):
    pack = torch.load(ckpt / ("" if EN else "voicepacks") / f"{voice}.pt", weights_only=True).numpy().reshape(510, 256)
    for si, text in enumerate(SENTENCES):
        ph = g2p_en(text)[0] if EN else vig2p.phonemize_text(sea.normalizer.normalize(text))
        tok = [vocab[c] for c in ph if c in vocab]
        ids = np.array([[0] + tok + [0]], np.int64)
        ref = pack[len(tok) - 1:len(tok)]
        try:
            cm_audio, cm_dur, cm_f0 = chain(ids, ref)
        except RuntimeError as e:
            print(f"{voice} s{si}: FAILED {e}"); rows.append(None); continue
        pt_audio, pt_dur, pt_f0 = torch_ref(ids, ref)
        name = f"{voice}_s{si}"
        sf.write(out / f"{name}_coreml.wav", cm_audio, 24000)
        sf.write(out / f"{name}_pytorch.wav", pt_audio, 24000)
        row = {"name": name, "tokens": len(tok), "pt_frames": int(pt_dur.sum()), "cm_frames": int(cm_dur.sum()),
               "frame_ratio": round(float(cm_dur.sum() / pt_dur.sum()), 3),
               "tokens_off": int((cm_dur != pt_dur).sum()), "finite": bool(np.isfinite(cm_audio).all())}
        # Pitch and audio with PyTorch's durations, so only the acoustic stages differ.
        tf_audio, _, tf_f0 = chain(ids, ref, forced=pt_dur)
        n = min(len(tf_f0), len(pt_f0))
        row["F0_corr"] = round(float(np.corrcoef(tf_f0[:n], pt_f0[:n])[0, 1]), 4)
        row["F0_rel_err_p95"] = round(float(np.percentile(np.abs(tf_f0[:n] - pt_f0[:n]) / np.maximum(np.abs(pt_f0[:n]), 1e-3), 95)), 4)
        row["mel_corr"] = round(mel_corr(tf_audio, pt_audio), 4)
        a, av = pitch(tf_audio); b, bv = pitch(pt_audio); n = min(len(a), len(b)); both = av[:n] & bv[:n]
        if both.sum() > 5:
            c = 1200 * np.abs(np.log2(a[:n][both] / b[:n][both]))
            row["audio_pitch_cents_median"] = round(float(np.median(c)), 1)
            row["audio_pitch_cents_p95"] = round(float(np.percentile(c, 95)), 1)
            row["voicing_agree"] = round(float((av[:n] == bv[:n]).mean()), 3)
        row["peak_cm"], row["peak_pt"] = round(float(np.abs(tf_audio).max()), 3), round(float(np.abs(pt_audio).max()), 3)
        rows.append(row)
        print(json.dumps(row))
json.dump(rows, open(out / "parity.json", "w"), indent=1)
print("RETRIES", len(RETRIES), __import__("collections").Counter(RETRIES))
