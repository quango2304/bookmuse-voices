#!/usr/bin/env python3
"""The iOS `ja` pack folder from `export_coreml`'s bundle (export.sh runs it):
- the flow LM's two graphs with their weights in 8 bits (coremltools linear symmetric, per output channel; activations stay fp16):
  the export writes the same weights into both, 161 MB each in fp16, 81 MB in 8 bits; Whisper reads both the same and the owner
  approved by ear (listening.md, Decisions);
- every graph compiled on the Mac to `.mlmodelc`: the ptts driver loads a compiled model when it is beside the package's name and
  never compiles, so the phone doesn't, and the `.mlpackage`s aren't shipped;
- host.safetensors, tokenizer.json, voices/, bundle.json (its file list rewritten to what the folder holds), LICENSE.

    uv run python finish.py <export_coreml bundle> <out>
"""
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import coremltools as ct
import coremltools.optimize.coreml as cto

NOTICE = """Pocket TTS Japanese (CC BY), the 6-layer `japanese` model, by takuma104: https://huggingface.co/takuma104/pocket-tts-ja-cc
(revision 6bfc869a67f0a58fef45349d7f24c741c7ebb58c), licensed CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/).
It derives from Kyutai's Pocket TTS (https://huggingface.co/kyutai/pocket-tts, CC BY 4.0; Kyutai, arXiv:2509.06926).
Training data: YODAS v3 (CC BY 3.0; Chen et al., Interspeech 2026), Emilia-YODAS from Emilia (CC BY 4.0; He et al.,
arXiv:2501.15907) and speech synthesized with Irodori-TTS (MIT). Its ten voices (voices/) are synthetic, made with
Irodori-TTS-v4.1-Small (Aratako, MIT) from text descriptions.
Changes made for Bookmuse: exported to Core ML with ptts's export_coreml (gradium-ai/xn-ptts 0.6.0), the flow model's weights
stored in 8 bits, and the voice prompts precomputed with the model's bos_before_voice token in front."""


def main() -> None:
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    config = cto.OptimizationConfig(global_config=cto.OpLinearQuantizerConfig(
        mode="linear_symmetric", dtype="int8", granularity="per_channel", weight_threshold=2048))
    with tempfile.TemporaryDirectory() as tmp:
        for package in sorted(src.glob("*.mlpackage")):
            if package.name.startswith("flow-"):
                # The export's packages name their weights folder in a way coremltools doesn't find on its own.
                spec = ct.utils.load_spec(str(package))
                model = ct.models.MLModel(spec, weights_dir=str(package / "Data/com.apple.CoreML/weights"), skip_model_load=True)
                quantized = Path(tmp) / package.name
                cto.linear_quantize_weights(model, config=config).save(str(quantized))
                package = quantized
            subprocess.run(["xcrun", "coremlcompiler", "compile", str(package), str(out)], check=True, capture_output=True)
            print(package.name, "compiled")
    for name in ("host.safetensors", "tokenizer.json"):
        shutil.copy(src / name, out / name)
    shutil.copytree(src / "voices", out / "voices")
    (out / "LICENSE").write_text(NOTICE + "\n")
    bundle = json.loads((src / "bundle.json").read_text())
    bundle["files"] = [
        {"path": str(p.relative_to(out)), "size": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
        for p in sorted(out.rglob("*")) if p.is_file()
    ]
    (out / "bundle.json").write_text(json.dumps(bundle, indent=1))
    print(f"{out}: {sum(f['size'] for f in bundle['files']) / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
