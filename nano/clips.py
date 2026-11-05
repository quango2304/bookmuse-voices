#!/usr/bin/env python3
"""The voice picker's preview clips (bookmuse_ios GMReader/Resources/VoiceClips/vi_<key>.m4a) with the Core ML models,
as the app makes them (T66: the model at its own length, the voice's pace on the audio within 0.9–1.1, by the app's
time-pitch unit in tempo.swift): "Xin chào, mình là <name>. Cùng đọc sách với mình nhé!" With `android`, only the voices
catalog.ANDROID gives their own clone, for bookmuse_android core/ondevice assets voiceclips/ (the other clips are iOS's).

    swiftc -O tempo.swift -o ../work/nano/tempo && uv run python clips.py <out dir> [android]
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

from catalog import ANDROID, SPEEDS
from nano_coreml import Nano, voice
from vieneu_utils.phonemize_text import phonemize_text_with_emotions
from vieneu_utils.core_utils import trim_and_fade

APP = Path(__file__).resolve().parent.parent / "work/nano/app/vi"
TEMPO = APP.parent.parent / "tempo"
ONLY_ANDROID = sys.argv[2:] == ["android"]
NAMES = {"tam_an": "Tâm An", "dang_khoa": "Đăng Khoa", "hoang_long": "Hoàng Long", "van_thanh": "Văn Thành", "ba_phuc": "Bá Phúc", "ngoc_han": "Ngọc Hân", "trung_hieu": "Trung Hiếu", "bao_tran": "Bảo Trân", "hai_dang": "Hải Đăng", "thu_ha": "Thu Hà", "dieu_linh": "Diệu Linh", "mai_chi": "Mai Chi"}
out = Path(sys.argv[1])
nano = Nano()
for key, name in NAMES.items():
    if ONLY_ANDROID and key not in ANDROID:
        continue
    speed = ANDROID[key][1] if ONLY_ANDROID else SPEEDS[key]
    if ONLY_ANDROID:
        spk, style = voice(APP.parent.parent / "voices" / f"{ANDROID[key][0]}.json")
    else:
        v = np.fromfile(APP / "voices" / f"{key}.f32", np.float32)
        spk, style = v[:192].reshape(1, 192), v[192:].reshape(1, 50, 256)
    text = f"Xin chào, mình là {name}. Cùng đọc sách với mình nhé!"
    made = out / f"vi_{key}-made.wav"
    sf.write(made, nano.chunk(phonemize_text_with_emotions(text), spk, style, seed=7)[0], 24000, subtype="FLOAT")
    paced = made.with_name(f"vi_{key}-paced.wav")
    subprocess.run([str(TEMPO), str(min(max(speed, 0.9), 1.1)), str(made), str(paced)], check=True)
    wav = trim_and_fade(sf.read(paced, dtype="float32")[0], 24000)
    made.unlink()
    paced.unlink()
    tmp = out / f"vi_{key}.wav"
    sf.write(tmp, np.concatenate([wav, np.zeros(2400, np.float32)]), 24000)
    subprocess.run(["afconvert", "-f", "m4af", "-d", "aac", "-b", "64000", str(tmp), str(out / f"vi_{key}.m4a")], check=True)
    tmp.unlink()
    print(key, "%.1f s" % (len(wav) / 24000), flush=True)
