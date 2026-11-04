#!/usr/bin/env python3
"""Assembles the Android Phonon pack (bookmuse_android T61: Phonon 40M multilingual, five languages) and zips it for a release of this repo.

    hf auth login        # the model is gated: accept its terms (CC BY-NC 4.0) on the model page first
    hf download gradium/phonon-40m-multilingual --local-dir phonon40m
    hf download gradium/phonon-voices --include "voices/*" --include voices.jsonl --local-dir phonon40m/phonon-voices
    python3 android/phonon_pack.py

The pack is a Phonon checkpoint folder as `ptts::synth` reads it on the CPU (bookmuse_android core/ondevice `PhononEngine`):
  config.json, tokenizer.json, model.q8.gguf   gradium/phonon-40m-multilingual (CC BY-NC 4.0)
  voices/<Key>.safetensors                     every human voice of gradium/phonon-voices (the 13 characters are left out): en 25, fr 15,
                                               de 13, es 25, pt 12. The key is the voice's name without accents (the file name; the
                                               catalogues of both apps use it in the ids `bookmuse.<lang>.<Key>`).
Writes dist/android/phonon/ and dist/bookmuse-voices-android-phonon-<ZIP_VERSION>.zip (files at the root, sorted, fixed timestamps),
then prints the line for bookmuse_android `voices/build.gradle.kts`.
"""
import hashlib, json, pathlib, shutil, sys, unicodedata, zipfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = ROOT / "phonon40m"
VOICES = SRC / "phonon-voices"
OUT = ROOT / "dist/android/phonon"
ZIPS = ROOT / "dist"
ZIP_VERSION = 1  # android-en-3 was Phonon 31m; bump with any change to the files
MODEL_FILES = ["config.json", "tokenizer.json", "model.q8.gguf"]


def key(name: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", name) if unicodedata.category(c) != "Mn")


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


if OUT.exists():
    shutil.rmtree(OUT)
files = []
for name in MODEL_FILES:
    src = SRC / name
    if not src.exists():
        sys.exit(f"missing {src} (see the commands at the top)")
    (OUT).mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, OUT / name)
    files.append(name)

rows = [json.loads(line) for line in open(VOICES / "voices.jsonl")]
seen = set()
for r in rows:
    if r["locale"] == "characters":
        continue
    k = key(r["name"])
    if k in seen:
        sys.exit(f"two voices share the key {k}")
    seen.add(k)
    src = VOICES / r["filename"]
    if not src.exists():
        sys.exit(f"missing {src}")
    dst = OUT / "voices" / f"{k}.safetensors"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    files.append(f"voices/{k}.safetensors")
print(len(seen), "voices")

ZIPS.mkdir(parents=True, exist_ok=True)
zip_path = ZIPS / f"bookmuse-voices-android-phonon-{ZIP_VERSION}.zip"
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
    for rel in sorted(files):
        info = zipfile.ZipInfo(rel, date_time=(1980, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_DEFLATED
        info.external_attr = 0o644 << 16
        info.file_size = (OUT / rel).stat().st_size
        with open(OUT / rel, "rb") as f, z.open(info, "w") as w:
            shutil.copyfileobj(f, w, 1 << 20)
digest = sha256(zip_path)
print(f"{zip_path.name}: {zip_path.stat().st_size} bytes, sha256 {digest}")
print(f'  voices/build.gradle.kts: VoicePack("phonon", "{zip_path.name}", {zip_path.stat().st_size}L, "{digest}")')
