#!/usr/bin/env bash
# The iOS Phonon bundle (bookmuse_ios GMReader/OnDeviceVoices/phonon): Phonon 40M multilingual as Core ML, through `export_coreml` of
# gradium-ai/xn-ptts at the version the app's `ptts` package pins (0.6.0), with each voice's pace fixed in it (listening.md, Pace):
#   - every voice speaks at `duration_delta` 0.15 (the model's own slower pace), the default in host.safetensors;
#   - the voices at or under 3.1 words a second at their own pace (phonon_pace/pace.json) carry their own `conditions` at 0.
#
#     hf download gradium/phonon-40m-multilingual --local-dir phonon40m     (gated: accept its terms first)
#     python3 android/phonon_pack.py                                        (the 90 voices, dist/android/phonon/voices)
#     ios/phonon_export.sh                                                  → dist/ios/phonon, then copy it over the app's folder
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"
PTTS="$ROOT/work/xn-ptts-0.6.0"
REV=95bf24d1ca5e94f0f62abe588b1ccb6f093bcc9c  # tag v0.6.0's merge: the PhononCore.xcframework the app links
SLOWER=0.15
OWN_PACE="Helene Azucena Debora Clarice Flavia Otavio Marcelo"
OUT="$ROOT/dist/ios/phonon"
VOICES="$ROOT/dist/android/phonon/voices"

[ -d "$PTTS" ] || git clone -q https://github.com/gradium-ai/xn-ptts "$PTTS"
git -C "$PTTS" checkout -q "$REV"
(cd "$PTTS" && cargo build --release -q -p ptts --example export_coreml --features cli)
EXPORT="$PTTS/target/release/examples/export_coreml"

export_at() {  # <duration_delta> <out>
  rm -rf "$2"
  "$EXPORT" --model "$ROOT/phonon40m" --weights model.safetensors --voices "$VOICES" --max-tokens 48 \
    --condition "duration_delta=$1" "$2"
}
export_at "$SLOWER" "$OUT"
# Only its host.safetensors is used: the conditioning at 0, which the own-pace voices are given as their own.
export_at 0 "$ROOT/work/phonon-own-pace"

python3 - "$OUT" "$ROOT/work/phonon-own-pace" $OWN_PACE <<'EOF'
import hashlib, json, pathlib, struct, sys

out, own, names = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), sys.argv[3:]

def read(path):
    raw = path.read_bytes()
    n = struct.unpack("<Q", raw[:8])[0]
    header = json.loads(raw[8:8 + n])
    body = raw[8 + n:]
    return {k: (v, body[v["data_offsets"][0]:v["data_offsets"][1]]) for k, v in header.items() if k != "__metadata__"}

def write(path, tensors):
    header, body = {}, b""
    for name, (meta, data) in tensors.items():
        header[name] = {"dtype": meta["dtype"], "shape": meta["shape"], "data_offsets": [len(body), len(body) + len(data)]}
        body += data
    h = json.dumps(header, separators=(",", ":")).encode()
    h += b" " * (-len(h) % 8)
    path.write_bytes(struct.pack("<Q", len(h)) + h + body)

meta, data = read(own / "host.safetensors")["flow_lm.conditions"]
manifest = json.loads((out / "bundle.json").read_text())
files = {f["path"]: f for f in manifest["files"]}
for name in names:
    path = out / "voices" / f"{name}.safetensors"
    tensors = read(path)
    assert "conditions" not in tensors, name
    tensors["conditions"] = ({"dtype": meta["dtype"], "shape": meta["shape"]}, data)
    write(path, tensors)
    entry = files[f"voices/{name}.safetensors"]
    entry["size"] = path.stat().st_size
    entry["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
(out / "bundle.json").write_text(json.dumps(manifest, indent=2) + "\n")
print("own pace:", ", ".join(names))
EOF
cat > "$OUT/source.json" <<EOF
{
  "model": "hf://gradium/phonon-40m-multilingual",
  "weights": "model.safetensors",
  "maxTokens": 48,
  "durationDelta": "$SLOWER; 0 for $OWN_PACE (their own conditions)",
  "voices": "the 90 human voices of hf://gradium/phonon-voices, file names without accents (bookmuse_voices/android/phonon_pack.py)",
  "made": "bookmuse_voices/ios/phonon_export.sh"
}
EOF
du -sh "$OUT"
