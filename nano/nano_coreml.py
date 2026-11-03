"""The app's Nano loop in Python on the Core ML models (the reference for NanoRuntime.swift), and a check against the
vieneu package on the same noise.

    uv run python nano_coreml.py check            # one chunk: Core ML vs ONNX Runtime
    uv run python nano_coreml.py render <voice.json> <text.txt> <out.wav> [steps=16 cfg=3 sway=-1]
"""
import json
import math
import sys
import time
from pathlib import Path

import coremltools as ct
import numpy as np

from graphs import SNAPSHOT

ML = Path(__file__).resolve().parent.parent / "work/nano/coreml"
TOKENS, BUCKETS, HOP = 320, [32, 64, 96, 128, 160, 192, 240], 1536
CFG = json.load(open(SNAPSHOT + "config.json"))
C = np.load(SNAPSHOT + "constants.npz")
NULL_SPK, NULL_STYLE = C["null_spk"].astype(np.float32).reshape(1, 192), C["null_style"].astype(np.float32).reshape(1, 50, 256)


class Nano:
    def __init__(self, units=ct.ComputeUnit.CPU_AND_NE):
        self.text = ct.models.MLModel(str(ML / "NanoText.mlpackage"), compute_units=units)
        self.step = {T: ct.models.MLModel(str(ML / "NanoStep.mlpackage"), compute_units=units, function_name=f"f{T}") for T in BUCKETS}
        self.codec = {T: ct.models.MLModel(str(ML / "NanoCodec.mlpackage"), compute_units=ct.ComputeUnit.CPU_AND_NE, function_name=f"f{T}") for T in BUCKETS}
        self.null_ctx, self.null_mask = self.encode([CFG["bos_id"], CFG["eos_id"]], NULL_STYLE, NULL_SPK)[:2]

    def encode(self, ids, style, spk):
        padded = np.zeros((1, TOKENS), np.int32); padded[0, :len(ids)] = ids
        mask = (padded != CFG["pad_id"]).astype(np.float32)
        out = self.text.predict({"ids": padded, "mask": mask, "style": style, "spk": spk})
        return out["ctx"].astype(np.float32), mask, float(out["log_seconds"].reshape(-1)[0])

    def ids(self, phonemes):
        s = phonemes
        for tag, ch in CFG.get("emotion_tags", {}).items():
            s = s.replace(tag, ch)
        return [CFG["bos_id"]] + [CFG["vocab"][c] for c in s if c in CFG["vocab"]] + [CFG["eos_id"]]

    def chunk(self, phonemes, spk, style, steps=16, cfg=3.0, sway=-1.0, noise=None, seed=0, speed=1.0):
        ctx, mask, log_s = self.encode(self.ids(phonemes), style, spk)
        T = max(2, int(round(min(math.exp(log_s) / speed, 15.0) * CFG["flow_fps"])))
        B = next(b for b in BUCKETS if b >= T)
        x = np.zeros((1, 144, B), np.float32)
        x[..., :T] = noise if noise is not None else np.random.default_rng(seed).standard_normal((1, 144, T))
        x_mask = np.zeros((1, 1, B), np.float32); x_mask[..., :T] = 1
        u = np.linspace(0, 1, steps + 1)
        tg = u + sway * (np.cos(np.pi / 2 * u) - 1 + u)
        feed = {"x_mask": x_mask, "cfg": np.array([cfg], np.float32), "ctx": np.concatenate([ctx, self.null_ctx]),
                "ctx_mask": np.concatenate([mask, self.null_mask]), "spk": np.concatenate([spk, NULL_SPK]),
                "style": np.concatenate([style, NULL_STYLE])}
        for i in range(steps):
            feed |= {"x": x, "t": np.array([tg[i]], np.float32), "dt": np.array([tg[i + 1] - tg[i]], np.float32)}
            x = self.step[B].predict(feed)["x_next"].astype(np.float32)
        x[..., T:] = 0
        wav = self.codec[B].predict({"x": x})["wav"].astype(np.float32).reshape(-1)[: T * HOP]
        return np.clip(wav, -1, 1), x[..., :T], T


def voice(path):
    v = json.load(open(path))
    return np.array(v["speaker_emb"], np.float32).reshape(1, 192), np.array(v["style"], np.float32).reshape(1, 50, 256)


if __name__ == "__main__":
    if sys.argv[1] == "check":
        import onnxruntime as ort
        from vieneu import Vieneu
        tts = Vieneu(mode="v3nano")
        eng = tts.engine
        spk, style = voice("../work/nano/voices/thien_tam_duc.json")
        ph = "tʃˈyəŋ mˈo6t̪, mˈo6t̪ bˈuə4j sˈaːɜŋ ˈəː4 hˈaː2 nˈo6j, mˈyə ʐˈɤj lˈaːɜt̪ ɗˈaːɜt̪ tʃˈen fˈo4 kˈo4."
        nano = Nano()
        t0 = time.time()
        wav, lat, T = nano.chunk(ph, spk, style, noise=np.random.default_rng(5).standard_normal((1, 144, 1)) * 0 + 0, seed=5)
        # the same noise through the package's loop (ONNX Runtime, fp32)
        noise = np.random.default_rng(7).standard_normal((1, 144, T)).astype(np.float32)
        wav, lat, T = nano.chunk(ph, spk, style, noise=noise)
        print("core ml: %d frames in %.2f s" % (T, time.time() - t0))
        ids = eng.encode_phones(ph); m = ids != eng.pad
        ctx = eng.s_text.run(None, {"ids": ids, "style": style})[0]
        log_s = float(eng.s_dur.run(None, {"ctx": ctx, "ctx_mask": m, "spk": spk})[0][0])
        print("onnx frames", int(round(math.exp(log_s) * eng.fps)))
        x = noise.copy(); u = np.linspace(0, 1, 17); tg = u - (np.cos(np.pi / 2 * u) - 1 + u)
        for i in range(16):
            t = np.array([tg[i]], np.float32)
            v = eng.s_ve.run(None, {"x": x, "t": t, "ctx": ctx, "ctx_mask": m, "spk": spk, "style": style})[0]
            vu = eng.s_ve.run(None, {"x": x, "t": t, "ctx": eng._null_ctx, "ctx_mask": eng._null_mask, "spk": eng.null_spk, "style": eng.null_style})[0]
            x = x + np.float32(tg[i + 1] - tg[i]) * (vu + 3 * (v - vu))
        ref = np.clip(eng.s_dec.run(None, {"x": x})[0].reshape(-1), -1, 1)
        print("latents: corr %.5f max diff %.4f (scale %.2f)" % (np.corrcoef(lat.ravel(), x.ravel())[0, 1], np.abs(lat - x).max(), np.abs(x).max()))
        print("audio: corr %.5f" % np.corrcoef(wav, ref[: len(wav)])[0, 1])
    else:
        import soundfile as sf
        sys.path.insert(0, str(Path(__file__).parent))
        from vieneu_utils.phonemize_text import normalize_to_chunks_v3_with_gaps, phonemize_text_with_emotions
        from vieneu_utils.core_utils import join_audio_chunks, gaps_to_silence, trim_and_fade
        opts = {k: float(v) for k, v in (a.split("=") for a in sys.argv[5:])}
        spk, style = voice(sys.argv[2])
        nano = Nano()
        chunks, gaps = normalize_to_chunks_v3_with_gaps(open(sys.argv[3]).read(), max_chars=180)
        t0 = time.time()
        wavs = [trim_and_fade(nano.chunk(phonemize_text_with_emotions(c), spk, style, steps=int(opts.get("steps", 16)), cfg=opts.get("cfg", 3),
                                         sway=opts.get("sway", -1), seed=1234 + i, speed=opts.get("speed", 1))[0], 24000) for i, c in enumerate(chunks)]
        print("synth %.1f s" % (time.time() - t0))
        import inspect
        out = join_audio_chunks(wavs, sr=24000, silences=gaps_to_silence(gaps)) if "silences" in inspect.signature(join_audio_chunks).parameters else np.concatenate(wavs)
        sf.write(sys.argv[4], out, 24000)
        print(sys.argv[4], "%.1f s" % (len(out) / 24000))
