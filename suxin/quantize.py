"""8-bit weights (per output channel, linear) for the fp32 models: computed in fp32, stored a quarter the size.
    uv run python quantize.py SuxinEncoder SuxinDurations [SuxinDecoder]"""
import sys

import coremltools as ct
import coremltools.optimize.coreml as cto

from graph import WORK

for name in sys.argv[1:]:
    m = ct.models.MLModel(str(WORK / f"coreml/{name}-fp32.mlpackage"), compute_units=ct.ComputeUnit.CPU_ONLY)
    cfg = cto.OptimizationConfig(global_config=cto.OpLinearQuantizerConfig(mode="linear_symmetric", dtype="int8",
                                                                            granularity="per_channel", weight_threshold=2048))
    q = cto.linear_quantize_weights(m, cfg)
    q.save(str(WORK / f"coreml/{name}-q8.mlpackage"))
    print("saved", name)
