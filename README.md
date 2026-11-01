# Bookmuse Voices

On-device neural voices for the Bookmuse reading app (Kokoro, English and Vietnamese): the scripts that make the models for each
platform, and the releases the apps' builds download them from.

| Folder | What it makes | Used by |
|---|---|---|
| `coreml/` | iOS: seven Core ML models per language, on the Neural Engine | bookmuse_ios (the compiled models are committed there) |
| `android/` | Android: one ONNX model per language (`int8convsafe`, ONNX Runtime on the CPU) and the release zips | bookmuse_android (`:voices:fetchVoices` downloads the zips) |
| `litert/` | Android GPU (LiteRT): work in progress | nothing yet |

Every folder starts from the same checkpoints: [hexgrad/Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) (English) and
[contextboxai/Kokoro-Vietnamese](https://huggingface.co/contextboxai/Kokoro-Vietnamese). The Vietnamese one needs
`coreml/remap_weight_norm.py` before any conversion.

## Releases

Each release is pinned by sha256 in the app that downloads it; never replace a file in a published release, make a new one.

| Release | File | Contents | Used by |
|---|---|---|---|
| v1 | `kokoro-en-v1.aar` (162 MB) | English ONNX pack: Kokoro-82M, 10 voices, the Misaki dictionaries, the fallback G2P | `android/` (source) |
| v1 | `kokoro-vi-v1.aar` (172 MB) | Vietnamese ONNX pack: Kokoro-Vietnamese, 13 voices, the sea-g2p dictionary | `android/` (source) |
| v1 | `SeaG2P.xcframework.zip` (7 MB) | sea-g2p (Vietnamese text to phonemes) compiled for iOS | **bookmuse_ios** (`Package.swift` binary target) |
| v2 | `kokoro-vi-v2.aar` (173 MB) | Vietnamese ONNX pack, 69 of 90 convs in fp16 | `android/` (its config and voices) |
| v2 | `bookmuse-voices-android-en-1.zip` (97 MB) | Android English pack | **bookmuse_android** release builds |
| v2 | `bookmuse-voices-android-vi-1.zip` (104 MB) | Android Vietnamese pack | **bookmuse_android** release builds |

`.aar` files are Apple Archives (LZFSE), opened with `aa` or the AppleArchive framework. The ONNX export of the v1 packs (English
with kokoro's `KModelForONNX`, Vietnamese from contextboxai's `kokoro_vi.onnx`) wasn't kept as a script.

## What's inside an ONNX pack

- `model.onnx`: the acoustic model. Inputs `input_ids` int64 `[1, n]` (0 at both ends), `ref_s` float32 `[1, 256]`, `speed` float32 `[1]`. Outputs `waveform` float32 (24 kHz) and `duration` int64 `[n]`, the frames each token lasts (600 samples per frame). Weights are stored as fp16 and computed in fp32.
- `voices/<name>.bin`: a voice, raw little-endian float32 `[510, 256]`. Use the row for the phoneme count minus one.
- `config.json`: the phoneme vocabulary (the same in both languages).
- English only: `lexicon/{us,gb}_{gold,silver}.json` and `g2p/{encoder,decoder}.onnx` with `g2p/vocab.json`.
- Vietnamese only: `sea_g2p.bin`, the sea-g2p dictionary with the Thai and Indonesian sections removed.

## Android packs

```sh
android/fetch_sources.sh                 # the v1/v2 packs into work/ (a Mac: they are Apple Archives)
uv run --no-project --python 3.12 --with onnx==1.19.0 --with onnxruntime==1.31.0 --with numpy python -I android/prepare_models.py
python3 android/make_packs.py            # dist/android/<lang>/ and dist/bookmuse-voices-android-<lang>-<n>.zip, with their pins
```

Checked 2026-11-01: the scripts rebuild the shipped `model.onnx` files byte for byte, and `dist/android/` equals the files the app
ships. The zips aren't byte-identical to the published v2 ones (a different compressor), so publishing means uploading the new
zips and pasting the printed pins into bookmuse_android `voices/build.gradle.kts`.

## Credits and licences

Everything here is Apache License 2.0, as are the sources:

- [hexgrad/Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M): the English model and voices, exported to ONNX.
- [contextboxai/Kokoro-Vietnamese](https://huggingface.co/contextboxai/Kokoro-Vietnamese) ([iamdinhthuan/Kokoro-Vietnamese](https://github.com/iamdinhthuan/Kokoro-Vietnamese)): the Vietnamese model and voices.
- [hexgrad/misaki](https://github.com/hexgrad/misaki): the English pronunciation dictionaries.
- [PeterReid/graphemes_to_phonemes_en_us](https://huggingface.co/PeterReid/graphemes_to_phonemes_en_us): the English fallback model, exported to ONNX; its vocabulary file is from [FluidInference/FluidAudio](https://github.com/FluidInference/FluidAudio).
- [pnnbao97/sea-g2p](https://github.com/pnnbao97/sea-g2p): Vietnamese text normalisation and phonemes.
- [FluidInference/mobius](https://github.com/FluidInference/mobius): the Core ML converter `coreml/convert-coreml.patch` applies to.
