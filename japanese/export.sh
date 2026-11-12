#!/usr/bin/env bash
# The iOS Japanese pack (`ja`): Pocket TTS Japanese 6L as Core ML for the app's Phonon engine (listening.md, "Pocket TTS Japanese").
#
#     uv run python convert.py         (the checkpoint → work/pocket_ja/ptts: weights, config, tokenizer, voices)
#     ./export.sh                      → dist/ios/ja
#     python3 -I ../ios/pack.py dist/ios/ja ja <n>     (zip, pin line, upload command)
#
# export_coreml of gradium-ai/xn-ptts at the version the app's `ptts` package pins (0.6.0), with --max-tokens 48 as Phonon's;
# then finish.py: 8-bit flow graphs, every graph compiled on the Mac, LICENSE.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"
PTTS="$ROOT/work/xn-ptts-0.6.0"
REV=95bf24d1ca5e94f0f62abe588b1ccb6f093bcc9c  # tag v0.6.0's merge: the PhononCore.xcframework the app links
SRC="$ROOT/work/pocket_ja/ptts"
EXPORT_DIR="$ROOT/work/pocket_ja/coreml"
OUT="$ROOT/dist/ios/ja"

[ -d "$PTTS" ] || git clone -q https://github.com/gradium-ai/xn-ptts "$PTTS"
git -C "$PTTS" checkout -q "$REV"
(cd "$PTTS" && cargo build --release -q -p ptts --example export_coreml --features cli)

rm -rf "$EXPORT_DIR"
"$PTTS/target/release/examples/export_coreml" --model "$SRC" --weights model.safetensors --voices "$SRC/voices" --max-tokens 48 \
  "$EXPORT_DIR"
(cd "$ROOT/japanese" && uv run python finish.py "$EXPORT_DIR" "$OUT")
