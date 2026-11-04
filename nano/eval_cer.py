"""How often Nano misreads or skips words, by voice and setting: renders a corpus chunk by chunk as the app does (Core ML,
the catalog's voices and speeds, trim and fade), transcribes each chunk with Whisper and aligns it word by word.

    uv run --with mlx-whisper --with jiwer --with scipy python eval_cer.py run <name> [voices=a,b] [seeds=8] [chunks=N]
        [max_chars=180] [steps=16] [cfg=3] [sway=-1] [speedcap=9] [speed=x] [speedmul=1] [noise=1] [backend=coreml|onnx]
    (a voice is a catalog key, or key@file for another clone of it)
    uv run --with jiwer python eval_cer.py report <name> [<name> ...]

Each (chunk text, seed) gets the same noise in every run, so runs compare pair by pair. A word Whisper gets wrong in every
seed of a (voice, chunk) is Whisper's or the normaliser's error, not the model's: the report leaves it out ("stochastic"
errors are the ones that come and go with the seed, which is what a listener hears as a misread that a re-read fixes).
Results: work/nano/cer/<name>.jsonl, audio: work/nano/cer/wav/<name>/.
"""
import json
import math
import re
import sys
import unicodedata
import zlib
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from catalog import SETTINGS, SPEEDS, VOICES  # noqa: E402

WORK = Path(__file__).resolve().parent.parent / "work/nano"
OUT = WORK / "cer"
WHISPER = "mlx-community/whisper-large-v3-turbo"
TONES = {"̀": "2", "́": "1", "̃": "3", "̉": "4", "̣": "5"}
EN = {"laptop", "email", "video", "singapore", "marketing"}


def canon(text):
    """Lower case, no punctuation or <en> tags, NFC, and each syllable's tone moved to its end (hoà = hòa)."""
    text = re.sub(r"</?en>", " ", unicodedata.normalize("NFC", text).lower())
    words = []
    for w in re.findall(r"[^\W_]+", text):
        d = unicodedata.normalize("NFD", w)
        tone = "".join(TONES[c] for c in d if c in TONES)
        words.append(unicodedata.normalize("NFC", "".join(c for c in d if c not in TONES)) + tone)
    return words


def opts_of(args):
    return {k: v for k, v in (a.split("=", 1) for a in args)}


def run(name, opts):
    import mlx_whisper
    import soundfile as sf
    from scipy.signal import resample_poly
    from vieneu_utils.core_utils import trim_and_fade
    from vieneu_utils.phonemize_text import _get_normalizer, normalize_to_chunks_v3_with_gaps, phonemize_text_with_emotions
    import nano_coreml as nc

    voices = opts.get("voices", ",".join(VOICES)).split(",")
    seeds = int(opts.get("seeds", 8))
    steps, cfg = int(opts.get("steps", SETTINGS["steps"])), float(opts.get("cfg", SETTINGS["cfg"]))
    sway, cap, scale = float(opts.get("sway", SETTINGS["sway"])), float(opts.get("speedcap", 9)), float(opts.get("noise", 1))
    backend = opts.get("backend", "coreml")
    chunks, _ = normalize_to_chunks_v3_with_gaps(open(OUT / "corpus.txt").read(), max_chars=int(opts.get("max_chars", 180)))
    chunks = chunks[: int(opts.get("chunks", len(chunks)))]
    normalizer = _get_normalizer()

    path = OUT / f"{name}.jsonl"
    done = set()
    if path.exists():
        done = {(r["voice"], r["chunk"], r["seed"]) for r in map(json.loads, path.open())}
    log = path.open("a")

    nano = nc.Nano()
    if backend == "onnx":
        from vieneu import Vieneu
        eng = Vieneu(mode="v3nano").engine

    for spec in voices:
        # "key" is the catalog's voice; "key@file" another clone of it, at the speed that brings it to the app's pace.
        key, _, file = spec.partition("@")
        spk, style = nc.voice(WORK / "voices" / f"{file or VOICES[key]}.json")
        speed = min(pace(nano, nc, chunks, phonemize_text_with_emotions, spk, style) if file else SPEEDS[key], cap)
        speed = float(opts.get("speed", speed)) * float(opts.get("speedmul", 1))
        key = spec
        for ci, text in enumerate(chunks):
            ph = phonemize_text_with_emotions(text)
            dropped = "".join(sorted({c for c in ph if c not in nc.CFG["vocab"]}))
            for seed in range(seeds):
                if (key, ci, seed) in done:
                    continue
                full = np.random.default_rng(zlib.crc32(text.encode()) * 100 + seed).standard_normal((1, 144, 240)) * scale
                if backend == "onnx":
                    wav, T, secs = onnx_chunk(eng, ph, spk, style, steps, cfg, sway, speed, full)
                else:
                    log_s = nano.encode(nano.ids(ph), style, spk)[2]
                    secs = math.exp(log_s) / speed
                    T = max(2, int(round(min(secs, 15.0) * nc.CFG["flow_fps"])))
                    wav = nano.chunk(ph, spk, style, steps=steps, cfg=cfg, sway=sway, speed=speed, noise=full[..., :T])[0]
                wav = trim_and_fade(wav, 24000)
                folder = OUT / "wav" / name / key.replace("@", "_")
                folder.mkdir(parents=True, exist_ok=True)
                sf.write(folder / f"c{ci:02d}_s{seed}.flac", wav, 24000)
                res = mlx_whisper.transcribe(resample_poly(wav, 2, 3).astype(np.float32), path_or_hf_repo=WHISPER, language="vi",
                                             temperature=0.0, condition_on_previous_text=False, word_timestamps=True)
                hyp = res["text"].strip()
                words = [w for s in res["segments"] for w in s.get("words", [])]
                short = [w["word"].strip() for w in words if w["end"] - w["start"] < 0.08]
                row = {"voice": key, "chunk": ci, "seed": seed, "ref": text, "hyp": hyp,
                       "hyp_norm": normalizer.normalize(hyp, punc_norm=False), "seconds": round(secs, 2), "capped": secs > 15.0,
                       "speed": speed, "dropped": dropped, "short": short}
                log.write(json.dumps(row, ensure_ascii=False) + "\n")
                log.flush()
                print(key, ci, seed, hyp[:70], flush=True)


def pace(nano, nc, chunks, phonemize, spk, style):
    """speeds.py's factor (3.4 syllables a second, within 0.75–1.3) on this corpus, from the Core ML duration predictor."""
    seconds = sum(math.exp(nano.encode(nano.ids(phonemize(c)), style, spk)[2]) for c in chunks)
    return round(min(1.3, max(0.75, 3.4 / (sum(len(c.split()) for c in chunks) / seconds))), 3)


def onnx_chunk(eng, ph, spk, style, steps, cfg, sway, speed, full):
    """The package's loop on ONNX Runtime (fp32), on the same noise: tells fp16 Core ML's part apart."""
    ids = eng.encode_phones(ph)
    m = ids != eng.pad
    ctx = eng.s_text.run(None, {"ids": ids, "style": style})[0]
    secs = math.exp(float(eng.s_dur.run(None, {"ctx": ctx, "ctx_mask": m, "spk": spk})[0][0])) / speed
    T = max(2, int(round(min(secs, 15.0) * eng.fps)))
    x = full[..., :T].astype(np.float32)
    u = np.linspace(0, 1, steps + 1)
    tg = u + sway * (np.cos(np.pi / 2 * u) - 1 + u)
    for i in range(steps):
        t = np.array([tg[i]], np.float32)
        v = eng.s_ve.run(None, {"x": x, "t": t, "ctx": ctx, "ctx_mask": m, "spk": spk, "style": style})[0]
        vu = eng.s_ve.run(None, {"x": x, "t": t, "ctx": eng._null_ctx, "ctx_mask": eng._null_mask, "spk": eng.null_spk,
                                 "style": eng.null_style})[0]
        x = x + np.float32(tg[i + 1] - tg[i]) * (vu + cfg * (v - vu))
    return np.clip(eng.s_dec.run(None, {"x": x})[0].reshape(-1), -1, 1), T, secs


def report(names):
    import jiwer

    for name in names:
        rows = [json.loads(line) for line in (OUT / f"{name}.jsonl").open()]
        # Per (voice, chunk): which reference words each seed got wrong.
        groups = defaultdict(list)
        for r in rows:
            ref, hyp = canon(r["ref"]), canon(r["hyp_norm"])
            out = jiwer.process_words(" ".join(ref), " ".join(hyp) or "∅")
            wrong, dels, ins = {}, 0, 0
            for a in out.alignments[0]:
                if a.type in ("delete", "substitute"):
                    for i in range(a.ref_start_idx, a.ref_end_idx):
                        wrong[i] = a.type
                    dels += (a.ref_end_idx - a.ref_start_idx) if a.type == "delete" else 0
                elif a.type == "insert":
                    ins += a.hyp_end_idx - a.hyp_start_idx
            groups[(r["voice"], r["chunk"])].append((r, ref, wrong, ins))

        per_voice = defaultdict(lambda: defaultdict(float))
        examples = []
        for (voice, ci), seeds in groups.items():
            n = len(seeds)
            always = set.intersection(*(set(w) for _, _, w, _ in seeds)) if n > 1 else set()
            for r, ref, wrong, ins in seeds:
                ref_vi = [i for i, w in enumerate(ref) if w not in EN]
                stoch = {i: t for i, t in wrong.items() if i not in always and i in ref_vi}
                v = per_voice[voice]
                v["renders"] += 1
                v["words"] += len(ref_vi)
                v["bad"] += 1 if stoch else 0
                v["del"] += sum(1 for t in stoch.values() if t == "delete")
                v["sub"] += sum(1 for t in stoch.values() if t == "substitute")
                v["ins"] += ins
                v["sys"] += len(always)
                v["en_wrong"] += sum(1 for i in wrong if ref[i] in EN)
                v["short"] += len(r["short"])
                v["capped"] += r["capped"]
                if stoch:
                    examples.append((voice, ci, r["seed"], [ref[i] for i in sorted(stoch)], r["hyp"]))

        print(f"\n== {name}")
        print(f"{'voice':12} {'renders':>7} {'bad %':>6} {'del/100w':>8} {'sub/100w':>8} {'ins':>4} {'sys':>4} {'EN✗':>4} {'short':>5} {'cap':>3}")
        tot = defaultdict(float)
        for voice, v in sorted(per_voice.items(), key=lambda kv: -kv[1]["bad"] / kv[1]["renders"]):
            for k, x in v.items():
                tot[k] += x
            print(line(voice, v))
        print(line("ALL", tot))
        if "-v" in sys.argv:
            for e in examples[:40]:
                print("  ", e)


def line(label, v):
    return (f"{label:12} {int(v['renders']):7d} {100 * v['bad'] / v['renders']:6.1f} {100 * v['del'] / v['words']:8.2f} "
            f"{100 * v['sub'] / v['words']:8.2f} {int(v['ins']):4d} {int(v['sys']):4d} {int(v['en_wrong']):4d} {int(v['short']):5d} {int(v['capped']):3d}")


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(sys.argv[2], opts_of(sys.argv[3:]))
    else:
        report([a for a in sys.argv[2:] if not a.startswith("-")])
