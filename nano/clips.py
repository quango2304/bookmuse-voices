#!/usr/bin/env python3
"""The voice picker's preview clips (bookmuse_ios GMReader/Resources/VoiceClips/vi_<key>.m4a) with the Core ML models,
as the app makes them: "Xin chào, mình là <name>. Cùng đọc sách với mình nhé!"

    uv run python clips.py <out dir>
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

from catalog import SPEEDS
from nano_coreml import Nano
from vieneu_utils.phonemize_text import phonemize_text_with_emotions
from vieneu_utils.core_utils import trim_and_fade

APP = Path(__file__).resolve().parent.parent / "work/nano/app/vi"
NAMES = {"tam_an": "Tâm An", "dang_khoa": "Đăng Khoa", "hoang_long": "Hoàng Long", "van_thanh": "Văn Thành", "ba_phuc": "Bá Phúc", "ngoc_han": "Ngọc Hân", "minh_khang": "Minh Khang", "trung_hieu": "Trung Hiếu", "bao_tran": "Bảo Trân", "hai_dang": "Hải Đăng", "thu_ha": "Thu Hà", "dieu_linh": "Diệu Linh", "mai_chi": "Mai Chi"}
out = Path(sys.argv[1])
nano = Nano()
for key, name in NAMES.items():
    v = np.fromfile(APP / "voices" / f"{key}.f32", np.float32)
    spk, style = v[:192].reshape(1, 192), v[192:].reshape(1, 50, 256)
    text = f"Xin chào, mình là {name}. Cùng đọc sách với mình nhé!"
    wav = trim_and_fade(nano.chunk(phonemize_text_with_emotions(text), spk, style, seed=7, speed=SPEEDS[key])[0], 24000)
    tmp = out / f"vi_{key}.wav"
    sf.write(tmp, np.concatenate([wav, np.zeros(2400, np.float32)]), 24000)
    subprocess.run(["afconvert", "-f", "m4af", "-d", "aac", "-b", "64000", str(tmp), str(out / f"vi_{key}.m4a")], check=True)
    tmp.unlink()
    print(key, "%.1f s" % (len(wav) / 24000), flush=True)
