#!/bin/bash
# Compiles the app's ChineseText.swift with the checker and runs it on the golden files (default: the corpus and the extra lines).
# usage: run.sh [pack folder] [golden.json ...]
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
root="$(cd "$here/../../.." && pwd)"
pack="${1:-$root/bookmuse_voices/work/chaowen/pack/zh}"
[ $# -gt 0 ] && shift
goldens=("$@")
[ ${#goldens[@]} -eq 0 ] && goldens=("$here/../corpus_zh.txt.golden.json" "$here/../swift_extra.txt.golden.json")
bin="$(mktemp -d)/swift_check"
swiftc -O -swift-version 6 -o "$bin" \
  "$root/bookmuse_ios/Packages/GlossmarkKit/Sources/GMReader/Speech/ChineseText.swift" "$here/main.swift"
"$bin" "$pack" "${goldens[@]}"
