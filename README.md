# Bookmuse Voices

On-device neural voices for the Bookmuse reading app, published as release files so the app can download one language at a time.

| File | Contents | Download |
|---|---|---|
| `kokoro-en-v1.aar` | English (US and UK): Kokoro-82M as ONNX, 10 voices, the Misaki pronunciation dictionaries, and a small grapheme-to-phoneme model for words the dictionaries miss | 162 MB |
| `kokoro-vi-v1.aar` | Vietnamese: Kokoro-Vietnamese as ONNX, 13 voices, and the sea-g2p dictionary | 172 MB |
| `SeaG2P.xcframework.zip` | sea-g2p (Vietnamese text to phonemes) compiled for iOS | 7 MB |

`.aar` files are Apple Archives (LZFSE), opened with the AppleArchive framework.

## What's inside a pack

- `model.onnx`: the acoustic model. Inputs `input_ids` int64 `[1, n]` (0 at both ends), `ref_s` float32 `[1, 256]`, `speed` float32 `[1]`. Outputs `waveform` float32 (24 kHz) and `duration` int64 `[n]`, the frames each token lasts (600 samples per frame). Weights are stored as fp16 and computed in fp32.
- `voices/<name>.bin`: a voice, raw little-endian float32 `[510, 256]`. Use the row for the phoneme count minus one.
- `config.json`: the phoneme vocabulary (the same in both languages).
- English only: `lexicon/{us,gb}_{gold,silver}.json` and `g2p/{encoder,decoder}.onnx` with `g2p/vocab.json`.
- Vietnamese only: `sea_g2p.bin`, the sea-g2p dictionary with the Thai and Indonesian sections removed.

## Credits and licences

Everything here is Apache License 2.0, as are the sources:

- [hexgrad/Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M): the English model and voices, exported to ONNX.
- [contextboxai/Kokoro-Vietnamese](https://huggingface.co/contextboxai/Kokoro-Vietnamese) ([iamdinhthuan/Kokoro-Vietnamese](https://github.com/iamdinhthuan/Kokoro-Vietnamese)): the Vietnamese model and voices.
- [hexgrad/misaki](https://github.com/hexgrad/misaki): the English pronunciation dictionaries.
- [PeterReid/graphemes_to_phonemes_en_us](https://huggingface.co/PeterReid/graphemes_to_phonemes_en_us): the English fallback model, exported to ONNX; its vocabulary file is from [FluidInference/FluidAudio](https://github.com/FluidInference/FluidAudio).
- [pnnbao97/sea-g2p](https://github.com/pnnbao97/sea-g2p): Vietnamese text normalisation and phonemes.
