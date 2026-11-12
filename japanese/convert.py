#!/usr/bin/env python3
"""Pocket TTS Japanese (takuma104/pocket-tts-ja-cc, the 6-layer `japanese` model, CC BY 4.0) → the files `ptts` reads, in
work/pocket_ja/ptts (listening.md, "Pocket TTS Japanese"):

- model.safetensors: the weights in f32, without `flow_lm.bos_before_voice` (it goes into each voice instead);
- config.json: Pocket 6L's shape (no `model_id`, so ptts uses tanh GELU and the `rms_var` time norm, as pocket-tts does), Mimi's
  32-wide encoder latents (`inner_dim`);
- tokenizer.json: the SentencePiece model as a Hugging Face Unigram tokenizer (dummy prefix, byte fallback, no normalisation, as
  Phonon's), checked id for id against SentencePiece on the eval passages and a line with spaces, emoji and Latin letters;
- voices/<key>.safetensors: `emb` [1, 1 + T, 1024] = `bos_before_voice` + the clip's projected Mimi latents, the prompt pocket-tts builds
  in `get_state_for_audio_prompt` when `insert_bos_before_voice` is set, so ptts needs no change.

    uv run python convert.py        then ./export.sh
"""
import json
import re
from pathlib import Path

import torch
import yaml
from huggingface_hub import snapshot_download
from safetensors.torch import load_file, save_file

REPO, REV = "takuma104/pocket-tts-ja-cc", "6bfc869a67f0a58fef45349d7f24c741c7ebb58c"
WORK = Path(__file__).resolve().parent.parent / "work/pocket_ja"
OUT = WORK / "ptts"
VOICES = [f"{g}_0{i}" for g in ("female", "male") for i in range(1, 6)]
# Checked lines: the eval's narrative, dialogue, numbers and literary passages (dist/ja_eval) in short, plus the odd cases.
CHECK = [
    "古い港町の外れに、小さな本屋があった。店主の老人は、毎朝六時に店を開け、窓辺の椅子に座って静かに本を読んでいた。",
    "「今日は何の本を読んでいるの？」と少女が尋ねた。「これはね、星を数える男の物語だよ」と老人は笑って答えた。",
    "2026年11月12日の朝、気温は-2℃、湿度は45%だった。iPhoneの電池は残り8%しかなかった。",
    "時雨の夕暮れ、提灯を提げた老人が橋の袂に佇んでいた。草鞋は泥にまみれ、足袋も濡れている。",
    "吾輩は猫である。 名前は  まだ無い✨ ABC 3.5%",
]


def tokenizer(src: Path) -> None:
    import sentencepiece as spm
    from tokenizers import Tokenizer
    from transformers.convert_slow_tokenizer import SpmConverter

    class Slow:
        vocab_file, legacy, add_prefix_space = str(src / "tokenizer.model"), True, None

    j = json.loads(SpmConverter(Slow()).converted().to_str())
    meta = {"type": "Metaspace", "replacement": "▁", "prepend_scheme": "always", "split": True}
    j["normalizer"] = {"type": "Prepend", "prepend": "▁"}
    j["pre_tokenizer"] = meta
    j["decoder"] = {"type": "Sequence", "decoders": [meta, {"type": "ByteFallback"}, {"type": "Fuse"}]}
    j["model"]["byte_fallback"] = True
    (OUT / "tokenizer.json").write_text(json.dumps(j, ensure_ascii=False))
    sp, tok = spm.SentencePieceProcessor(str(src / "tokenizer.model")), Tokenizer.from_file(str(OUT / "tokenizer.json"))
    rules = yaml.safe_load((src / "japanese.yaml").read_text())["replace_characters"]
    for line in CHECK:
        text = re.sub(r"([.!?…])\s*[,;:]", r"\1", " ".join(line.translate(str.maketrans(rules)).split()))
        a, b = sp.encode(text, out_type=int), tok.encode(text, add_special_tokens=False).ids
        assert a == b, f"tokenizer differs on {text!r}:\n{a}\n{b}"
    print("tokenizer.json: the same ids as SentencePiece on", len(CHECK), "lines")


def main() -> None:
    src = Path(snapshot_download(REPO, revision=REV, local_dir=WORK / "hf",
                                 allow_patterns=["japanese/*", "japanese.yaml", "tokenizer.model", "voices/*", "README.md"]))
    (OUT / "voices").mkdir(parents=True, exist_ok=True)
    weights = load_file(src / "japanese/model.safetensors")
    bos = weights.pop("flow_lm.bos_before_voice").float()
    save_file({k: v.float().contiguous() for k, v in weights.items()}, OUT / "model.safetensors")
    cfg = {
        "flow_lm": {"d_model": 1024, "num_heads": 16, "num_layers": 6, "dim_feedforward": 4096, "max_period": 10000.0,
                    "n_bins": 4000, "lut_dim": 1024, "flow_dim": 512, "flow_depth": 6, "ldim": 32},
        "mimi": {"channels": 1, "sample_rate": 24000, "frame_rate": 12.5, "dimension": 512, "quantizer_dimension": 32,
                 "quantizer_output_dimension": 512, "n_filters": 64, "n_residual_layers": 1, "ratios": [6, 5, 4], "kernel_size": 7,
                 "last_kernel_size": 3, "residual_kernel_size": 3, "dilation_base": 2, "compress": 2, "transformer_d_model": 512,
                 "transformer_num_heads": 8, "transformer_num_layers": 2, "transformer_layer_scale": 0.01,
                 "transformer_context": 250, "transformer_max_period": 10000.0, "transformer_dim_feedforward": 2048,
                 "downsample_channel_wise": False, "inner_dim": 32},
        "lsd_decode_steps": 1, "eos_threshold": -4.0, "model_id": None, "audio_prompt_min_duration": 2.5,
        "audio_prompt_max_duration": 10.0, "cfg_null_audio_empty": False, "speaker_mimi": None, "conditioners": [], "fuser": None,
        "voices": [],
    }
    (OUT / "config.json").write_text(json.dumps(cfg, indent=2))
    tokenizer(src)

    from pocket_tts import TTSModel
    from pocket_tts.data.audio import audio_read
    from pocket_tts.data.audio_utils import convert_audio, end_on_pause

    config = yaml.safe_load((src / "japanese.yaml").read_text())
    config["weights_path"] = str(src / "japanese/model.safetensors")
    config["flow_lm"]["lookup_table"]["tokenizer_path"] = str(src / "tokenizer.model")
    local = WORK / "japanese.local.yaml"
    local.write_text(yaml.safe_dump(config, allow_unicode=True))
    model = TTSModel.load_model(config=str(local))
    assert torch.allclose(model.flow_lm.bos_before_voice.float().cpu(), bos, atol=1e-3)
    rate = model.config.mimi.sample_rate
    for key in VOICES:
        audio, sr = audio_read(src / f"voices/{key}.wav")
        audio = end_on_pause(convert_audio(audio, sr, rate, 1), rate)
        with torch.no_grad():
            prompt = model._encode_audio(audio.unsqueeze(0))
            emb = torch.cat([model.flow_lm.bos_before_voice, prompt], dim=1).float().contiguous()
        save_file({"emb": emb}, OUT / f"voices/{key}.safetensors")
        print(key, tuple(emb.shape))


if __name__ == "__main__":
    main()
