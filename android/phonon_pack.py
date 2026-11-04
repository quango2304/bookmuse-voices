#!/usr/bin/env python3
"""Assembles the Android English pack for Phonon 31m (bookmuse_android T58) and zips it for a release of this repo.

    python3 android/phonon_pack.py

The pack is a Phonon checkpoint folder as `ptts::synth` reads it on the CPU (bookmuse_android core/ondevice `PhononEngine`):
  config.json, tokenizer.json, model.q8.gguf   gradium/phonon-31m-en at MODEL_REV (Apache 2.0); the weights are the same as the
                                               revision iOS exports from, this config is the one the CPU runtime was measured with
  voices/<Name>.safetensors                    the 25 human voices iOS ships (bookmuse_ios GMReader/OnDeviceVoices/en/voices: the
                                               `emb` tensors of gradium/phonon-voices, which the CPU loader also takes)
Writes dist/android/en-phonon/ and dist/bookmuse-voices-android-en-<ZIP_VERSION>.zip (files at the root, sorted, fixed timestamps),
then prints the line for bookmuse_android `voices/build.gradle.kts`.
"""
import hashlib, pathlib, shutil, sys, urllib.request, zipfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
IOS_VOICES = ROOT.parent / "bookmuse_ios/Packages/GlossmarkKit/Sources/GMReader/OnDeviceVoices/en/voices"
OUT = ROOT / "dist/android/en-phonon"
ZIPS = ROOT / "dist"
ZIP_VERSION = 3  # en-1 was Kokoro; bump with any change to the files
MODEL_REV = "92741c5b07d085ff840e26fea8ce43110b87852f"
# The weights' sha256 (the HF LFS oid); the two small files are checked by the runtime when it loads them.
MODEL_FILES = {"config.json": None, "tokenizer.json": None, "model.q8.gguf": "a7d9aad2cd6dc66473f24a08bc6d2939030ef4d751ef82349361fbef3272dbae"}

VOICES = ["Zoey", "Juniper", "Everly", "Harper", "Hadley", "Delia", "Easton", "Micah", "Trevor", "Russell", "Preston", "Miles",
          "Maeve", "Freya", "Pippa", "Elodie-Rose", "Imogen", "Lottie", "Bryony", "Alfie", "Hugh", "Edwin", "Spencer", "Toby", "Teddy"]


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


files = []
for name, prefix in MODEL_FILES.items():
    dst = OUT / name
    if not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://huggingface.co/gradium/phonon-31m-en/resolve/{MODEL_REV}/{name}"
        print("downloading", url)
        with urllib.request.urlopen(url) as r, open(dst.with_suffix(".part"), "wb") as w:
            shutil.copyfileobj(r, w, 1 << 20)
        dst.with_suffix(".part").rename(dst)
    if prefix and sha256(dst) != prefix:
        sys.exit(f"{dst} does not match the pinned sha256 {prefix}")
    files.append(name)
for v in VOICES:
    src = IOS_VOICES / f"{v}.safetensors"
    if not src.exists():
        sys.exit(f"missing {src} (bookmuse_ios checked out beside this repo?)")
    dst = OUT / "voices" / src.name
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    files.append(f"voices/{src.name}")

ZIPS.mkdir(parents=True, exist_ok=True)
zip_path = ZIPS / f"bookmuse-voices-android-en-{ZIP_VERSION}.zip"
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
print(f'  voices/build.gradle.kts: VoicePack("en", "{zip_path.name}", {zip_path.stat().st_size}L, "{digest}")')
