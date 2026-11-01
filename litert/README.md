# Kokoro on LiteRT (Android GPU): work in progress

The aim: run Kokoro's vocoder (~80% of the work) on the phone's GPU with LiteRT, so Android voices keep up on phones whose CPU
can't, such as the Pixel 3. Today's Android model is ONNX Runtime on the CPU (`../android/`). Nothing here ships yet.

From the Android P6a spike (bookmuse_android `spikes/p6a-tts/kokoro/README.md`, Findings 5–6):

- `litert_vocoder.py` converts the vocoder, the tail and the noise stage at one fixed shape (300 frames, 7.5 s of audio) with
  litert-torch 0.10.0, reusing the stage split of the iOS Core ML converter (`../coreml/`, FluidInference/mobius `26f5fa8`).
  Two rewrites make it convert exactly (corr 1.000000 vs PyTorch): the depthwise `ConvTranspose1d` in the upsampling block becomes
  zero-insert + depthwise conv, and nearest `F.interpolate` becomes stack + reshape (its GATHER_ND kept 434 of 1208 ops off the
  GPU). The whole vocoder is then GPU-delegable.
- The tail (conv_post + iSTFT, 0.3% of the compute) still converts wrong (corr 0.92) even with the overlap-add rewrite: do it in
  Kotlin or ORT.
- `litert_parity.py` checks the `.tflite` files against PyTorch's intermediates and times them on the host CPU.
- Not measured yet: the GPU on a phone (the emulator has no OpenCL and no GLES compute buffers).

Left to build: frame buckets (e.g. 150/300/600 frames, one `.tflite` with several signatures sharing weights), fp16 weights
(197 → ~100 MB), the text stages and noise on the CPU, and the Vietnamese checkpoint (`../coreml/remap_weight_norm.py` first).

```sh
python -I litert_vocoder.py <mobius laishere-coreml dir> <kokoro en dir: config.json, kokoro-v1_0.pth, voices/af_heart.pt> <out dir>
python -I litert_parity.py <out dir> <threads>
# venv: litert-torch==0.10.0 (torch 2.14.1, ai-edge-litert 2.3.0), kokoro==0.9.4, coremltools==9.0 for the import
```
