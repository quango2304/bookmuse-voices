# Bookmuse Voices

On-device voices for the Bookmuse reading app: the scripts that make them for each platform, and the releases the apps' builds
download them from. English is [Phonon 31m](https://huggingface.co/gradium/phonon-31m-en) (Gradium), Vietnamese is
[VieNeu-TTS v3 Nano](https://huggingface.co/pnnbao-ump/VieNeu-TTS-v3-Nano).

| Folder | What it makes | Used by |
|---|---|---|
| `android/phonon_pack.py` | Android English: the Phonon 31m checkpoint folder (q8 GGUF, tokenizer, config, 25 voices) and its release zip | bookmuse_android (`:voices:fetchVoices`) |
| `nano/` | Vietnamese: VieNeu-TTS v3 Nano for Core ML on iOS (`convert.py`, `export_assets.py`) and ONNX Runtime on Android (`android.py`), and the cloned voices (`voice.py`, `catalog.py`) | bookmuse_ios, bookmuse_android |
| `turbo/` | VieNeu-TTS v3 Turbo: the voices Nano's are cloned from, and the Python environment the Nano scripts run in | `nano/voice.py`, `render.py`, `vbee_ref.py` |

iOS English needs nothing from here: the iOS app bundles the Core ML export of Phonon that `ptts` makes (`bookmuse_ios`
GMReader `OnDeviceVoices/en`), and the Android pack takes its 25 voice files from there.

## Releases

Each release is pinned by sha256 in the app that downloads it; never replace a file in a published release, make a new one.

| Release | File | Contents | Used by |
|---|---|---|---|
| v3 | `bookmuse-voices-android-en-3.zip` (60 MB) | Phonon 31m: `model.q8.gguf`, `tokenizer.json`, `config.json` (phonon-31m-en `92741c5`), `voices/<Name>.safetensors` | **bookmuse_android** |
| v3 | `bookmuse-voices-android-vi-3.zip` (102 MB) | VieNeu-TTS v3 Nano for ONNX Runtime (`nano/android.py`) | **bookmuse_android** |

v1 and v2 hold the earlier voices (Kokoro); no app build uses them now. They stay published, as every release does.

## Android Phonon pack (`android/phonon_pack.py`)

A checkpoint folder as `ptts::synth` reads it on the CPU (bookmuse_android `core/ondevice/phonon`, the runtime with the Pixel 3
patches). The weights are downloaded from Hugging Face at the pinned revision and checked against their sha256; the voices are the
`emb` tensors iOS ships, which the CPU loader also takes. Writes `dist/android/en-phonon/` and the zip, and prints the line for
bookmuse_android `voices/build.gradle.kts`.

## Android Nano pack (`nano/android.py`)

`step.onnx` is Nano's vector estimator with the guidance pair as one batch and the Euler update inside, int8 MatMul (40 MB);
`text_encoder.onnx` and `codec_decoder.onnx` keep fp32 compute with fp16-stored weights; `duration_predictor.onnx` as published;
`config.json`, `null.bin`, `voices/<key>.bin` and `sea_g2p.bin` are the files the iOS app ships. Measured choices are in
bookmuse `specs/tasks/T52-android-vieneu-nano.md`.

## Credits and licences

The scripts are Apache License 2.0. The sources:

- [gradium/phonon-31m-en](https://huggingface.co/gradium/phonon-31m-en) and [gradium/phonon-voices](https://huggingface.co/gradium/phonon-voices): the English model and voices (Apache 2.0); its runtime [gradium-ai/xn-ptts](https://github.com/gradium-ai/xn-ptts) (Apache 2.0 / MIT).
- [pnnbao-ump/VieNeu-TTS-v3-Nano](https://huggingface.co/pnnbao-ump/VieNeu-TTS-v3-Nano): the Vietnamese model (code and weights Apache-2.0;
  its training data is CC BY-NC 4.0, so ask the author before a store release).
- [pnnbao97/sea-g2p](https://github.com/pnnbao97/sea-g2p): Vietnamese text normalisation and phonemes.
