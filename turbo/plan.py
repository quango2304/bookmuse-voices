"""Which compute device Core ML picks for each op of a model (MLComputePlan), counted per op type."""
import sys, collections
import coremltools as ct
from coremltools.models.compute_plan import MLComputePlan
from coremltools.models.compute_device import MLNeuralEngineComputeDevice, MLGPUComputeDevice, MLCPUComputeDevice
path, fn = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else "main")
units = getattr(ct.ComputeUnit, sys.argv[3] if len(sys.argv) > 3 else "ALL")
compiled = ct.models.utils.compile_model(path) if path.endswith(".mlpackage") else path
plan = MLComputePlan.load_from_path(path=compiled, compute_units=units)
prog = plan.model_structure.program
f = prog.functions[fn]
count = collections.Counter(); where = collections.defaultdict(collections.Counter)
for op in f.block.operations:
    u = plan.get_compute_device_usage_for_mlprogram_operation(op)
    d = "none" if u is None else type(u.preferred_compute_device).__name__.replace("ML","").replace("ComputeDevice","")
    count[d] += 1; where[d][op.operator_name] += 1
print(dict(count))
for d, c in where.items(): print(d, dict(c.most_common(12)))
