#!/usr/bin/env python3
"""Assembles the Android voice packs (Kokoro, int8convsafe) and zips each language for a release of this repo.

    python3 android/make_packs.py

Reads ../work/ (fetch_sources.sh, then prepare_models.py):
  en: int8convsafe-en/model.onnx + ios-en-v1/{config.json, lexicon/, g2p/, voices/<the 4 catalog voices>}
  vi: int8convsafe-vi/model.onnx + ios-vi-v2/{config.json, sea_g2p.bin, voices/<the 13 catalog voices>}
Writes dist/android/<lang>/ (the pack's files) and dist/bookmuse-voices-android-<lang>-<n>.zip (the files at the zip's root,
sorted entries and fixed timestamps, so the same files give the same bytes).

bookmuse_android's `:voices:fetchVoices` downloads the zips at build time and unpacks them into its :voices asset pack. Publishing a
new version: upload the zips to a new release here, then set the tag and each pack's file name, size and sha256 (printed below) in
bookmuse_android `voices/build.gradle.kts`. The voice lists match the app's `BookmuseCatalog`.
"""
import hashlib, pathlib, shutil, sys, zipfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
W = ROOT / "work"
OUT = ROOT / "dist/android"
ZIPS = ROOT / "dist"
ZIP_VERSION = 1  # bump with any change to the files of a pack; the file name is bookmuse-voices-android-<lang>-<ZIP_VERSION>.zip

EN_VOICES = ["af_heart", "af_bella", "af_nicole", "bf_emma"]
VI_VOICES = ["duc_duy", "diem_trinh", "hung_thinh", "mai_linh", "mai_loan", "manh_dung", "my_yen", "ngoc_huyen", "phat_tai",
             "thanh_dat", "thuc_trinh", "tuan_ngoc", "duc_an"]

PACKS = {
    "en": [(W / "int8convsafe-en/model.onnx", "model.onnx"), (W / "ios-en-v1/config.json", "config.json")]
    + [(W / f"ios-en-v1/lexicon/{n}", f"lexicon/{n}") for n in ("us_gold.json", "us_silver.json", "gb_gold.json", "gb_silver.json")]
    + [(W / f"ios-en-v1/g2p/{n}", f"g2p/{n}") for n in ("encoder.onnx", "decoder.onnx", "vocab.json")]
    + [(W / f"ios-en-v1/voices/{v}.bin", f"voices/{v}.bin") for v in EN_VOICES],
    "vi": [(W / "int8convsafe-vi/model.onnx", "model.onnx"), (W / "ios-vi-v2/config.json", "config.json"),
           (W / "ios-vi-v2/sea_g2p.bin", "sea_g2p.bin")]
    + [(W / f"ios-vi-v2/voices/{v}.bin", f"voices/{v}.bin") for v in VI_VOICES],
}


for lang, files in PACKS.items():
    total = 0
    for src, rel in files:
        if not src.exists():
            sys.exit(f"missing {src} (run android/fetch_sources.sh and android/prepare_models.py)")
        dst = OUT / lang / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists() or dst.stat().st_size != src.stat().st_size:
            shutil.copyfile(src, dst)
        total += src.stat().st_size
        print(lang, rel, src.stat().st_size)
    print(lang, "total", total)

    # The release zip: the language's folder contents at the root, deterministic (sorted entries, fixed timestamps, fixed compression level).
    ZIPS.mkdir(parents=True, exist_ok=True)
    zip_path = ZIPS / f"bookmuse-voices-android-{lang}-{ZIP_VERSION}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for _, rel in sorted(files, key=lambda f: f[1]):
            info = zipfile.ZipInfo(rel, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            info.file_size = (OUT / lang / rel).stat().st_size  # lets the writer skip ZIP64 for files under 2 GB
            with open(OUT / lang / rel, "rb") as f, z.open(info, "w") as w:
                shutil.copyfileobj(f, w, 1 << 20)
    digest = hashlib.sha256()
    with open(zip_path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    print(f'{zip_path.name}: {zip_path.stat().st_size} bytes, sha256 {digest.hexdigest()}')
    print(f'  voices/build.gradle.kts: VoicePack("{lang}", "{zip_path.name}", {zip_path.stat().st_size}L, "{digest.hexdigest()}")')
