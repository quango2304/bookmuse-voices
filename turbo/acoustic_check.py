"""17 slot inputs of one frame and the PyTorch acoustic decoder's outputs for them, for the app's check (Swift port)."""
import json
from pathlib import Path
import torch
from vieneu._v3_turbo_engine.hub_load_v3_turbo import load_v3_turbo_checkpoint
WORK = Path(__file__).resolve().parent.parent / "work/turbo"
model = load_v3_turbo_checkpoint(str(WORK / "vieneu"), subfolder="update", device=torch.device("cpu")).eval()
torch.manual_seed(1)
x = torch.randn(1, 17, 768) * 0.5
with torch.no_grad():
    out = []
    pk, pv = [None], [None]
    for s in range(17):
        h, pk, pv = model.acoustic_decoder.cached_step(x[:, s:s + 1], torch.tensor([s]), pk, pv)
        out.append(h[0, 0].tolist())
# cached_step adds the slot embedding itself; the app adds it before step(): give the app x + slot_emb.
xin = (x[0] + model.acoustic_decoder.slot_pos_emb.weight[:17]).tolist()
(WORK / "app/vi-turbo/acoustic_check.json").write_text(json.dumps({"x": xin, "out": out}))
print("wrote")
