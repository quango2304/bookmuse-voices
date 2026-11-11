# Bookmuse Voices

On-device voices for the Bookmuse reading app: the scripts that make them for each platform, and the releases the apps' builds
download them from. English, French, German, Spanish and Portuguese are
[Phonon 40M multilingual](https://huggingface.co/gradium/phonon-40m-multilingual) (Gradium; Phonon 31m English until then),
Vietnamese is [VieNeu-TTS v3 Nano](https://huggingface.co/pnnbao-ump/VieNeu-TTS-v3-Nano).

| Folder | What it makes | Used by |
|---|---|---|
| `android/phonon_pack.py` | Android Phonon: the Phonon 40M checkpoint folder (q8 GGUF, tokenizer, config, 90 human voices: en 25, fr 15, de 13, es 25, pt 12) and its release zip | bookmuse_android (`:voices:fetchVoices`) |
| `phonon_pace/` | The words-a-second of every Phonon voice on one passage per language (`pace.py` → `pace.json`): the numbers behind both apps' `PhononPace` tables | bookmuse_ios, bookmuse_android |
| `nano/` | Vietnamese: VieNeu-TTS v3 Nano for Core ML on iOS (`convert.py`, `export_assets.py`) and ONNX Runtime on Android (`android.py`), and the cloned voices (`voice.py`, `catalog.py`) | bookmuse_ios, bookmuse_android |
| `turbo/` | VieNeu-TTS v3 Turbo: the voices Nano's are cloned from, and the Python environment the Nano scripts run in | `nano/voice.py`, `render.py`, `vbee_ref.py` |

iOS Phonon needs nothing from here: the iOS app bundles the Core ML export of Phonon that `ptts` makes (`bookmuse_ios`
GMReader `OnDeviceVoices`). The Android pack takes its voices from `gradium/phonon-voices` (see the script's header).

## Releases

Each release is pinned by sha256 in the app that downloads it; never replace a file in a published release, make a new one.

**Pinned by bookmuse_android now: v6** (`bookmuse-voices-android-phonon-1.zip`, `bookmuse-voices-android-vi-7.zip`). The table below stops at v4; the full list of releases is in the parent repo's `docs/listening.md`.

| Release | File | Contents | Used by |
|---|---|---|---|
| v4 | `bookmuse-voices-android-phonon-1.zip` (86 MB) | Phonon 40M multilingual: `model.q8.gguf`, `tokenizer.json`, `config.json`, `voices/<Key>.safetensors` (90 voices) | **bookmuse_android** (pinned; **not uploaded to v4 yet**, 2026-11-04) |
| v4 | `bookmuse-voices-android-vi-5.zip` (102 MB) | VieNeu-TTS v3 Nano for ONNX Runtime (`nano/android.py`), 13 voices | **bookmuse_android** |
| v4, v3 | `bookmuse-voices-android-en-3.zip` (60 MB) | Phonon 31m English (phonon-31m-en `92741c5`) | earlier Android builds |
| v3 | `bookmuse-voices-android-vi-3.zip` (102 MB) | VieNeu-TTS v3 Nano, the first voices | earlier Android builds |

v1 and v2 hold the earlier voices (Kokoro); no app build uses them now. They stay published, as every release does.

## Android Phonon pack (`android/phonon_pack.py`)

A checkpoint folder as `ptts::synth` reads it on the CPU (bookmuse_android `core/ondevice/phonon`, the runtime with the Pixel 3
patches). The model and voices are downloaded from Hugging Face first (`hf download`, the commands are at the top of the script;
the model is gated); every human voice of `gradium/phonon-voices` goes in, the 13 characters are left out. Writes
`dist/android/phonon/` and `dist/bookmuse-voices-android-phonon-<n>.zip`, and prints the line for bookmuse_android
`voices/build.gradle.kts`.

## Android Nano pack (`nano/android.py`)

`step.onnx` is Nano's vector estimator with the guidance pair as one batch and the Euler update inside, int8 MatMul (40 MB);
`text_encoder.onnx` and `codec_decoder.onnx` keep fp32 compute with fp16-stored weights; `duration_predictor.onnx` as published;
`config.json`, `null.bin`, `voices/<key>.bin` and `sea_g2p.bin` are the files the iOS app ships. Measured choices are in
bookmuse `docs/listening.md`.

## Credits and licences

The scripts are Apache License 2.0. The sources:

- [gradium/phonon-40m-multilingual](https://huggingface.co/gradium/phonon-40m-multilingual): the five-language model (**CC BY-NC 4.0**, gated: ask Gradium before a store release); [gradium/phonon-voices](https://huggingface.co/gradium/phonon-voices): the voices; [gradium/phonon-31m-en](https://huggingface.co/gradium/phonon-31m-en): the earlier English model (Apache 2.0); the runtime [gradium-ai/xn-ptts](https://github.com/gradium-ai/xn-ptts) (Apache 2.0 / MIT).
- [pnnbao-ump/VieNeu-TTS-v3-Nano](https://huggingface.co/pnnbao-ump/VieNeu-TTS-v3-Nano): the Vietnamese model (code and weights Apache-2.0;
  its training data is CC BY-NC 4.0, so ask the author before a store release).
- [pnnbao97/sea-g2p](https://github.com/pnnbao97/sea-g2p): Vietnamese text normalisation and phonemes.
