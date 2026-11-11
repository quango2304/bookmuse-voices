"""Zips an iOS voice pack folder for the apps to download from R2 (bookmuse docs/listening.md, "Delivery").

The iOS app downloads its packs; nothing is bundled. A pack is one folder:
  phonon  the Core ML export `ptts`'s `export_coreml` writes for Phonon 40M multilingual (three .mlpackage, host.safetensors,
          tokenizer.json, bundle.json, source.json, voices/)
  vi      VieNeu Nano for Core ML (nano/export_assets.py: three .mlmodelc, config.json, null.f32, voices/*.f32) plus sea_g2p.bin

Usage:
  python3 -I ios/pack.py <folder> <pack> <n>      e.g. ios/pack.py ~/voices/ios/vi vi 1

Writes dist/bookmuse-voices-ios-<pack>-<n>.zip (files at the root, sorted, fixed timestamps, deflated, so the same folder always
gives the same bytes) and prints the pin for bookmuse_ios `VoicePacks.swift` and the command that uploads it. Never upload over a
published file: a changed pack is a new <n>.
"""
import hashlib, os, pathlib, sys, zipfile

def main():
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    folder, pack, n = pathlib.Path(sys.argv[1]).expanduser().resolve(), sys.argv[2], int(sys.argv[3])
    if not folder.is_dir():
        sys.exit(f"{folder} is not a folder")
    out = pathlib.Path(__file__).resolve().parent.parent / "dist" / f"bookmuse-voices-ios-{pack}-{n}.zip"
    out.parent.mkdir(exist_ok=True)
    files = sorted(p for p in folder.rglob("*") if p.is_file() and p.name != ".DS_Store")
    unpacked = 0
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for path in files:
            info = zipfile.ZipInfo(str(path.relative_to(folder)), date_time=(1980, 1, 1, 0, 0, 0))
            info.external_attr = 0o644 << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            with open(path, "rb") as f, z.open(info, "w") as dst:
                while chunk := f.read(8 << 20):
                    dst.write(chunk)
            unpacked += path.stat().st_size
    digest = hashlib.sha256()
    with open(out, "rb") as f:
        while chunk := f.read(8 << 20):
            digest.update(chunk)
    size = out.stat().st_size
    print(f"{out.name}: {len(files)} files, {size:,} B zipped, {unpacked:,} B unpacked")
    print("Pin (VoicePacks.swift):")
    print(f'    .init(id: "{pack}", version: {n}, size: {size}, unpackedSize: {unpacked}, sha256: "{digest.hexdigest()}"),')
    print("Upload (from wanngo_web, wrangler logged in to the wanngo account):")
    print(f"npx wrangler r2 object put wanngo-media/voices/ios/{out.name} --file {out} --content-type application/zip "
          "--cache-control 'public, max-age=31536000, immutable' --remote")

if __name__ == "__main__":
    main()
