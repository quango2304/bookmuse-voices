# Bookmuse Voices on Core ML

How the iOS on-device voices in bookmuse_ios `Packages/GlossmarkKit/Sources/GMReader/OnDeviceVoices/{en,vi}` were made. Both run Kokoro as seven Core ML models (`KokoroCoreML.swift`): five on the Neural Engine, Noise and Tail on the CPU, never the GPU. They ship inside the app, committed to bookmuse_ios (not a release here).

## Converter

FluidInference/mobius at `26f5fa8`, `models/tts/kokoro/laishere-coreml/convert-coreml.py`, with `convert-coreml.patch` applied. The patch adds:

- **FluidAudio fixes, still open upstream:**
  - fp32 Prosody (mobius #108)
  - COLA-normalized Tail (mobius #83)
- **`--config`, `--model`, `--voice` and `--phonemes`:** for a checkpoint other than hexgrad's English one.
- **Default input shapes no request has:** 2 tokens; 300 frames, which the app shifts by one frame when a run would hit it. On iOS/macOS 27 a model fails every prediction after the first at its default shape. Tiny frame defaults put most of the vocoder on the CPU, so the frame default stays at a typical sentence.
- **No GPU in the benchmark and the end-to-end check:** MPSGraph aborts on OS 27.

Set it up with Python 3.11 (3.13 can't build a misaki dependency):

```sh
uv sync --python 3.11
uv pip install --reinstall coremltools==9.0
uv pip install scikit-learn
```

## Vietnamese (contextboxai/Kokoro-Vietnamese)

`kokoro_vi.pth` stores weight norm the new way (`parametrizations.weight.original0/1`). hexgrad's `KModel` expects `weight_g/weight_v`. On a mismatch it falls back to `strict=False` and silently leaves 178 tensors random, so first rename the keys:

```sh
python remap_weight_norm.py kokoro_vi.pth kokoro_vi_wn.pth config.json
python convert-coreml.py --output-dir out/vi --config config.json --model kokoro_vi_wn.pth \
    --voice voicepacks/mai_linh.pt --phonemes "hˈom nˈaj ʈʂˈəː↘j dˈɛʔ↓p, ..."
```

## English (hexgrad/Kokoro-82M)

```sh
python convert-coreml.py --output-dir out/en --voice af_heart.pt
python convert_g2p.py <PeterReid/graphemes_to_phonemes_en_us> out/g2p   # the fallback G2P
```

## Check, compile, bundle

`parity.py` compares the chain with PyTorch: durations, the predicted F0 curve, mel correlation, and pitch in the audio. At release: durations within 1%, F0 correlation 1.0, mel correlation 0.983–0.995, and median pitch error ≤ 10 cents.

```sh
xcrun coremlcompiler compile <stage>.mlpackage OnDeviceVoices/<lang>
```

In bookmuse_ios, bump `BookmuseEngine.outputVersion` when a model changes, so kept audio is made again.

A language's folder holds:

- the seven `Kokoro*.mlmodelc`
- `config.json` (the vocab)
- `voices/*.bin` (510 × 256 float32), only for the voices `BookmuseCatalog` lists
- English: `lexicon/` and `g2p/` (`G2PEncoder.mlmodelc`, `G2PDecoder.mlmodelc`, `vocab.json`)
- Vietnamese: `sea_g2p.bin`
