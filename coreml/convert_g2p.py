"""misaki's English G2P fallback (PeterReid/graphemes_to_phonemes_en_us, a 1+1-layer BART) as two Core ML
models, from a plain-PyTorch forward pass over the same weights (transformers' masking can't be traced).

Encoder: input_ids [1, n] -> hidden [1, n, d]. Decoder: decoder_input_ids [1, t], encoder_hidden_states
[1, n, d] -> logits [1, t, vocab]. Float32, CPU. Default shapes are ones Bookmuse never sends (a word is
at least one letter, so n >= 3): on OS 27 a model fails every prediction after the first at its default
shape.
"""
import math, sys, numpy as np, torch, torch.nn.functional as F, coremltools as ct
from transformers import BartForConditionalGeneration
src, out = sys.argv[1:3]
hf = BartForConditionalGeneration.from_pretrained(src, attn_implementation="eager").eval()
cfg = hf.config
MAX = cfg.max_position_embeddings

def attention(a, x, kv, heads, causal):
    D = cfg.d_model; h = D // heads
    q = (a.q_proj(x) * h ** -0.5).reshape(1, -1, heads, h).transpose(1, 2)
    k = a.k_proj(kv).reshape(1, -1, heads, h).transpose(1, 2)
    v = a.v_proj(kv).reshape(1, -1, heads, h).transpose(1, 2)
    w = q @ k.transpose(-1, -2)
    if causal:
        w = w + torch.ones_like(w[0, 0]).triu(1) * -1e9
    return a.out_proj((w.softmax(-1) @ v).transpose(1, 2).reshape(1, -1, D))

def act(x): return F.gelu(x)

def embed(m, ids):
    pos = torch.ones_like(ids).cumsum(1) + 1  # 0-based positions, plus BART's offset of 2
    return m.layernorm_embedding(m.embed_tokens(ids) + F.embedding(pos, m.embed_positions.weight))

class Enc(torch.nn.Module):
    def __init__(s, hf): super().__init__(); s.e = hf.model.encoder
    def forward(s, input_ids):
        x = embed(s.e, input_ids)
        for l in s.e.layers:
            x = l.self_attn_layer_norm(x + attention(l.self_attn, x, x, cfg.encoder_attention_heads, False))
            x = l.final_layer_norm(x + l.fc2(act(l.fc1(x))))
        return x

class Dec(torch.nn.Module):
    def __init__(s, hf): super().__init__(); s.d = hf.model.decoder; s.head = hf.lm_head; s.register_buffer("bias", hf.final_logits_bias)
    def forward(s, decoder_input_ids, encoder_hidden_states):
        x = embed(s.d, decoder_input_ids)
        for l in s.d.layers:
            x = l.self_attn_layer_norm(x + attention(l.self_attn, x, x, cfg.decoder_attention_heads, True))
            x = l.encoder_attn_layer_norm(x + attention(l.encoder_attn, x, encoder_hidden_states, cfg.decoder_attention_heads, False))
            x = l.final_layer_norm(x + l.fc2(act(l.fc1(x))))
        return s.head(x) + s.bias

assert cfg.activation_function == "gelu" and not cfg.scale_embedding and not getattr(cfg, "normalize_before", False)
enc, dec = Enc(hf).eval(), Dec(hf).eval()
with torch.no_grad():
    for ids, d in [([1, 9, 10, 11, 12, 2], [1, 5, 6]), ([1, 20, 30, 40, 2], [1]), ([1] + list(range(4, 40)) + [2], [1, 7, 8, 9, 10, 11])]:
        ids, d = torch.tensor([ids]), torch.tensor([d])
        ref = hf(input_ids=ids, decoder_input_ids=d).logits
        mine = dec(d, enc(ids))
        print("max |diff| vs transformers:", float((ref - mine).abs().max()))
    ids = torch.tensor([[1, 9, 10, 11, 12, 2]])
    te = torch.jit.trace(enc, (ids,)); td = torch.jit.trace(dec, (torch.tensor([[1, 5, 6]]), enc(ids)))
common = dict(convert_to="mlprogram", minimum_deployment_target=ct.target.iOS17,
              compute_precision=ct.precision.FLOAT32, compute_units=ct.ComputeUnit.CPU_ONLY)
ct.convert(te, inputs=[ct.TensorType(name="input_ids", shape=(1, ct.RangeDim(2, MAX, default=2)), dtype=np.int32)],
           outputs=[ct.TensorType(name="hidden")], **common).save(f"{out}/G2PEncoder.mlpackage")
ct.convert(td, inputs=[ct.TensorType(name="decoder_input_ids", shape=(1, ct.RangeDim(1, MAX, default=1)), dtype=np.int32),
                       ct.TensorType(name="encoder_hidden_states", shape=(1, ct.RangeDim(2, MAX, default=2), cfg.d_model), dtype=np.float32)],
           outputs=[ct.TensorType(name="logits")], **common).save(f"{out}/G2PDecoder.mlpackage")
print("saved")
