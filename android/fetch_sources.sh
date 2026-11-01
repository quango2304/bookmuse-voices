#!/usr/bin/env bash
# Downloads the iOS ONNX packs the Android models are made from (this repo's releases v1 and v2) into ../work (gitignored) and
# unpacks them. They are Apple Archives, so this runs on a Mac (`aa`). Then: prepare_models.py, make_packs.py.
set -euo pipefail
mkdir -p "$(dirname "$0")/../work/dl"
cd "$(dirname "$0")/../work"
gh release download v1 -R quango2304/bookmuse-voices -p 'kokoro-en-v1.aar' -p 'kokoro-vi-v1.aar' -D dl --skip-existing
gh release download v2 -R quango2304/bookmuse-voices -p 'kokoro-vi-v2.aar' -D dl --skip-existing
for p in en-v1 vi-v1 vi-v2; do [[ -d ios-$p ]] || { mkdir -p ios-$p && aa extract -i dl/kokoro-$p.aar -d ios-$p; }; done
echo "sources ready in $PWD"
