#!/usr/bin/env bash
# The Android Japanese pack (`ja`): Pocket TTS Japanese 6L for the app's CPU Phonon runtime (bookmuse_android core/ondevice/phonon,
# patched gradium-ai/xn-ptts e97a3de), which opens a checkpoint folder: config.json, tokenizer.json, model.q8.gguf, voices/.
#
#     uv run python convert.py         (the checkpoint → work/pocket_ja/ptts: weights, config, tokenizer, voices)
#     ./android.sh                     → dist/android/ja
#     python3 -I ../ios/pack.py dist/android/ja ja <n> android     (zip, pin line, upload command)
#
# The flow model's linear weights in q8_0 (what the runtime runs), the Mimi encoder left out: the voices are precomputed.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"
PTTS="$ROOT/../bookmuse_android/core/ondevice/phonon/work/xn-ptts"
SRC="$ROOT/work/pocket_ja/ptts"
OUT="$ROOT/dist/android/ja"

[ -d "$PTTS" ] || { echo "run bookmuse_android/core/ondevice/phonon/build.sh first (it clones and patches xn-ptts)"; exit 1; }
rm -rf "$OUT"; mkdir -p "$OUT"
(cd "$PTTS" && cargo run --release -q -p ptts --example quantize -- "$SRC/model.safetensors" "$OUT/model.q8.gguf" --quant q8_0 --no-mimi-encoder)
cp "$SRC/config.json" "$SRC/tokenizer.json" "$OUT/"
cp -R "$SRC/voices" "$OUT/voices"
sed -e 's/exported to Core ML with ptts.s export_coreml (gradium-ai\/xn-ptts 0.6.0), the flow model.s weights$/converted to GGUF with ptts'"'"'s quantize (gradium-ai\/xn-ptts e97a3de), the flow model'"'"'s weights/' \
    -e 's/^stored in 8 bits, and/in q8_0 and the Mimi encoder left out, and/' "$ROOT/dist/ios/ja/LICENSE" > "$OUT/LICENSE"
du -sh "$OUT"
