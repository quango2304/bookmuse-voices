"""Teacher-forced comparison: the reference's codes drive both sides; at each frame and codebook, does the IO module
pick the reference's argmax, and how close are the logits? Runs the IO modules in PyTorch fp32, and Core ML if given."""
import sys, json
from pathlib import Path
import numpy as np, torch
import coremltools as ct
from tokenizers import Tokenizer
from vieneu._v3_turbo_engine.hub_load_v3_turbo import load_v3_turbo_checkpoint
from vieneu_utils.phonemize_text import phonemize_text_with_emotions
import modules as M
WORK = Path("../work/turbo"); CACHE, BLOCK = 384, 64
model = load_v3_turbo_checkpoint(str(WORK / "vieneu"), subfolder="update", device=torch.device("cpu")).eval()
cfg = model.config
tok = Tokenizer.from_file(str(WORK / "vieneu/update/tokenizer.json"))
import vieneu
voice = json.loads((Path(vieneu.__file__).parent / "assets/voices_v3_turbo.json").read_text())["presets"]["Thiền Tâm Đức"]
ref_codes = torch.tensor(voice["codes"]); spk = torch.tensor(voice["speaker_emb"], dtype=torch.float32)[None]
phon = phonemize_text_with_emotions("Ngôi chùa nhỏ nằm lưng chừng núi, nơi mây trắng vẫn thường dừng lại mỗi sớm mai.")
ids = [16, 3] + tok.encode(phon, add_special_tokens=False).ids + [4]
rows = torch.full((len(ids) + len(ref_codes), 17), 1024); rows[: len(ids), 0] = torch.tensor(ids)
rows[len(ids):, 0] = 7; rows[len(ids):, 1:] = ref_codes; T = rows.shape[0]
text_emb = model.text_embeddings.weight.detach(); audio_emb = torch.stack([e.weight for e in model.audio_embeddings]).detach()
slot_emb = model.acoustic_decoder.slot_pos_emb.weight.detach()
NL = len(model.acoustic_decoder.layers)
mode = sys.argv[1] if len(sys.argv) > 1 else "torch"
if mode == "torch":
    BB, AC = M.BackboneIO(model).eval(), M.AcousticIO(model).eval()
    def bbf(fn, d): return [t.numpy() for t in BB(*[torch.from_numpy(d[k].astype(np.float32)) for k in ["x","cos","sin","kc","vc","mask"]])]
    def acf(d): return [t.numpy() for t in AC(*[torch.from_numpy(d[k].astype(np.float32)) for k in ["x","kc","vc","mask"]])]
else:
    u = getattr(ct.ComputeUnit, mode)
    P = {f: ct.models.MLModel(str(WORK / f"coreml-io/{sys.argv[2] if len(sys.argv) > 2 else 'TurboBackbone'}.mlpackage"), compute_units=u, function_name=f) for f in ["prefill","step"]}
    A = ct.models.MLModel(str(WORK / "coreml-io/TurboAcoustic.mlpackage"), compute_units=u)
    def bbf(fn, d): r = P[fn].predict(d); return [r["hidden"], r["k_new"], r["v_new"]]
    def acf(d): r = A.predict(d); return [r["hidden"], r["k_new"], r["v_new"]]
kc = np.zeros((48, CACHE, 64), np.float16); vc = np.zeros_like(kc)
def bb(fn, x, start):
    S = x.shape[1]; cos, sin = M.rope(torch.arange(start, start + S))
    mask = np.full((1, 1, S, CACHE + S), -1e4, np.float16); mask[0, 0, :, :start] = 0
    for i in range(S): mask[0, 0, i, CACHE: CACHE + i + 1] = 0
    h, kn, vn = bbf(fn, {"x": x.numpy().astype(np.float16), "cos": cos.numpy().astype(np.float16), "sin": sin.numpy().astype(np.float16), "kc": kc, "vc": vc, "mask": mask})
    kc[:, start:start + S] = kn; vc[:, start:start + S] = vn
    return torch.from_numpy(np.asarray(h, np.float32))
with torch.no_grad():
    emb = model._build_inputs_embeds(rows[None], speaker_emb=spk)
    out = model.semantic_backbone(inputs_embeds=emb, use_cache=True, return_dict=True); past, rh = out.past_key_values, out.last_hidden_state[:, -1]
    hs = []
    for s in range(0, T, BLOCK):
        blk = emb[:, s:s+BLOCK]; n = blk.shape[1]
        if n < BLOCK: blk = torch.cat([blk, torch.zeros(1, BLOCK - n, M.H)], 1)
        hs.append(bb("prefill", blk, s)[:, :n])
    mh = torch.cat(hs, 1)[0, -1]
    agree = []; cosines = []
    for t in range(30):
        codes, _ = model.decode_one_frame(rh, text_token_id=torch.tensor([5]), temperature=1.0, top_k=1, audio_top_p=1.0)
        codes = codes.tolist()
        # mine, teacher-forced with the reference codes
        akc = np.zeros((NL * 8, 17, 96), np.float16); avc = np.zeros_like(akc)
        seq = [mh, text_emb[5]] + [audio_emb[c][codes[c]] for c in range(15)]
        for s in range(17):
            mask = np.full((1, 1, 1, 18), -1e4, np.float16); mask[..., :s] = 0; mask[..., 17] = 0
            h, kn, vn = acf({"x": (seq[s] + slot_emb[s]).view(1, 1, 768).numpy().astype(np.float16), "kc": akc, "vc": avc, "mask": mask})
            akc[:, s] = np.asarray(kn)[:, 0]; avc[:, s] = np.asarray(vn)[:, 0]
            if s >= 1:
                lg = torch.from_numpy(np.asarray(h, np.float32))[0, 0] @ audio_emb[s - 1].T
                agree.append(int(lg.argmax()) == codes[s - 1])
        cosines.append(float(torch.nn.functional.cosine_similarity(mh, rh[0], dim=0)))
        slot = torch.tensor([[[5] + codes]])
        se = model._build_inputs_embeds(slot, speaker_emb=spk)
        o = model.semantic_backbone(inputs_embeds=se, past_key_values=past, use_cache=True, return_dict=True); past, rh = o.past_key_values, o.last_hidden_state[:, -1]
        mh = bb("step", se, T + t)[0, -1]
print(mode, "argmax agreement %.3f" % np.mean(agree), "backbone hidden cosine min %.4f mean %.4f" % (min(cosines), np.mean(cosines)))
