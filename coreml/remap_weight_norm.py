"""Renames new-style weight-norm keys (parametrizations.weight.original0/1) to the weight_g/weight_v that
hexgrad's kokoro expects, strips the DataParallel 'module.' prefix, and checks a strict load."""
import sys, torch
from kokoro import KModel
src, dst, config = sys.argv[1:4]
sd = torch.load(src, map_location="cpu", weights_only=True)
fixed = {}
for part, tensors in sd.items():
    out = {}
    for k, v in tensors.items():
        k = k[7:] if k.startswith("module.") else k
        k = k.replace(".parametrizations.weight.original0", ".weight_g").replace(".parametrizations.weight.original1", ".weight_v")
        out[k] = v
    fixed[part] = out
# Strict check against hexgrad's modules.
import json
from kokoro.model import KModel as K
m = K.__new__(K); torch.nn.Module.__init__(m)
cfg = json.load(open(config))
from kokoro.modules import CustomAlbert, ProsodyPredictor, TextEncoder
from kokoro.istftnet import Decoder
from transformers import AlbertConfig
m.bert = CustomAlbert(AlbertConfig(vocab_size=cfg['n_token'], **cfg['plbert']))
m.bert_encoder = torch.nn.Linear(m.bert.config.hidden_size, cfg['hidden_dim'])
m.predictor = ProsodyPredictor(style_dim=cfg['style_dim'], d_hid=cfg['hidden_dim'], nlayers=cfg['n_layer'], max_dur=cfg['max_dur'], dropout=cfg['dropout'])
m.text_encoder = TextEncoder(channels=cfg['hidden_dim'], kernel_size=cfg['text_encoder_kernel_size'], depth=cfg['n_layer'], n_symbols=cfg['n_token'])
m.decoder = Decoder(dim_in=cfg['hidden_dim'], style_dim=cfg['style_dim'], dim_out=cfg['n_mels'], **cfg['istftnet'])
for part, tensors in fixed.items():
    result = getattr(m, part).load_state_dict(tensors, strict=False)
    # Instance-norm scale/shift: not saved by StyleTTS2 training (affine=False there); hexgrad's
    # defaults, 1 and 0, are the same identity.
    bad = [k for k in result.missing_keys if not (k.endswith(".norm.weight") or k.endswith(".norm.bias"))]
    assert not bad and not result.unexpected_keys, (part, bad[:5], result.unexpected_keys[:5])
    print(part, "ok", len(tensors), "loaded;", len(result.missing_keys), "norm affine left at 1/0")
torch.save(fixed, dst)
