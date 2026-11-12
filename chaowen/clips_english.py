"""Clips of Chao Wen reading Chinese with English words, for the owner's ear: one per tone scheme (stressed / other syllables).
    python clips_english.py <out dir>"""
import sys, subprocess
from pathlib import Path
import numpy as np, onnxruntime as ort, soundfile as sf
sys.path.insert(0, str(Path(__file__).resolve().parent))
import english as e
from frontend import Frontend
SENT = ["我昨天买了一部iPhone，用Google搜了很久。", "他最喜欢的书是Harry Potter，作者住在London。", "请把文件通过email发给我，OK？",
        "NASA和USB都是缩写，CEO也是。", "她在Starbucks遇见了Steve Jobs的老朋友Michael。", "这家公司叫Microsoft，那家叫Amazon，还有Netflix。"]
out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
s = ort.InferenceSession(str(Path(__file__).resolve().parent.parent / "work/chaowen/zh_CN-chaowen-medium.onnx"), providers=["CPUExecutionProvider"])
for name, tones in {"A_stressed4_other1": ("4", "1"), "B_all1": ("1", "1"), "C_stressed1_other5": ("1", "5")}.items():
    f = Frontend(); t = e.table(tones)
    f.english = {k: [x for syl in v for x in syl] for k, v in t.items()}
    parts = []
    for line in SENT:
        ids, _ = f.encode(line); ids = np.array([ids], np.int64)
        a = s.run(None, {"input": ids, "input_lengths": np.array([ids.shape[1]], np.int64), "scales": np.array([0, 0.96, 0], np.float32)})[0].ravel()
        parts += [a, np.zeros(int(0.4 * 22050), np.float32)]
    w = out / f"chaowen_english_{name}.wav"; sf.write(w, np.concatenate(parts), 22050)
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(w), "-af", "loudnorm=I=-18", "-c:a", "aac", "-b:a", "128k", str(w.with_suffix(".m4a"))], check=True)
    print(w.with_suffix(".m4a"))
