#!/usr/bin/env python3
"""Reference clips from Vbee voices, to clone with voice.py: each voice reads the same short story passage (WAV).
Credentials come from the environment (VBEE_APP_ID, VBEE_KEY), never a file here; VBEE_TEXT (a file) and VBEE_SUFFIX
read another passage into another file.

    VBEE_APP_ID=… VBEE_KEY=… ../turbo/.venv/bin/python vbee_ref.py <voiceCode>...
"""
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "work/nano/vbee"
OUT.mkdir(parents=True, exist_ok=True)
TEXT = ("Ngày ấy, làng tôi còn nghèo lắm. Mỗi sáng, mẹ dậy từ lúc gà gáy, nhóm bếp nấu nồi cơm độn khoai. "
        "Tôi ngồi bên bậc cửa, nghe tiếng mưa rơi lộp độp trên mái lá, và tự hỏi: rồi mai này mình sẽ đi đâu?")
HEAD = {"Authorization": "Bearer " + os.environ["VBEE_KEY"], "App-Id": os.environ["VBEE_APP_ID"], "Content-Type": "application/json"}


def call(url, body=None):
    req = urllib.request.Request(url, data=json.dumps(body).encode() if body else None, headers=HEAD, method="POST" if body else "GET")
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


if os.environ.get("VBEE_TEXT"):
    TEXT = open(os.environ["VBEE_TEXT"]).read().strip()
SUFFIX = os.environ.get("VBEE_SUFFIX", "")
for code in sys.argv[1:]:
    made = call("https://api.vbee.vn/v1/tts", {"text": TEXT, "voiceCode": code, "mode": "async", "outputFormat": "wav",
                                               "webhookUrl": "https://bookmuse.invalid/vbee"})
    rid = made.get("requestId") or made
    for _ in range(90):
        time.sleep(1)
        st = call(f"https://api.vbee.vn/v1/tts/requests/{rid}")
        status = str(st.get("status", "")).upper()
        if status in ("COMPLETED", "SUCCESS") and st.get("audioLink"):
            urllib.request.urlretrieve(st["audioLink"], OUT / f"{code}{SUFFIX}.wav")
            print("wrote", code, flush=True)
            break
        if status in ("FAILED", "FAILURE", "ERROR"):
            print("failed", code, st, flush=True)
            break
    else:
        print("timeout", code, flush=True)
