# Export best_single.h5 (SingleNet state_dict) to best_single.onnx
# and verify torch vs ONNX Runtime (CPU + CUDA) output parity.
#
# Run from the learn/ directory with the project .venv:
#   cd ~/Documents/cpp_animal_shogi/learn
#   ../.venv/bin/python export_onnx.py
#
# Notes:
# - learn/game.py imports gamelibs, which is built for CPython 3.8 only,
#   so importing single_network as-is fails on Python 3.12. SingleNet does
#   not use any symbol from game at class-definition time (State is only
#   used inside print_network()), so we stub the `game` module before import.
# - Model contract: input "input"  float32 (N, 42, 3, 4), dynamic batch axis 0
#                   output "value" float32 (N, 1), opset 17

import sys
import types

# Stub out `game` so `from game import *` in single_network.py succeeds
# without importing the cp38-only gamelibs extension.
sys.modules.setdefault("game", types.ModuleType("game"))

import numpy as np
import torch

from single_network import SingleNet

MODEL_IN = "./model/best_single.h5"
MODEL_OUT = "./model/best_single.onnx"
OPSET = 17


def main():
    model = SingleNet()  # same defaults as conv_jit(): blocks=3, channels=192, fcl=256
    state = torch.load(MODEL_IN, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()

    dummy = torch.zeros(1, 42, 3, 4, dtype=torch.float32)
    with torch.no_grad():
        ref_dummy = model(dummy).numpy()

    torch.onnx.export(
        model,
        dummy,
        MODEL_OUT,
        export_params=True,
        opset_version=OPSET,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["value"],
        dynamic_axes={"input": {0: "batch"}, "value": {0: "batch"}},
    )
    print(f"exported: {MODEL_OUT}")

    import onnx

    onnx_model = onnx.load(MODEL_OUT)
    onnx.checker.check_model(onnx_model)
    print("onnx checker: OK")

    import onnxruntime as ort

    rng = np.random.default_rng(0)
    test = rng.integers(0, 2, size=(8, 42, 3, 4)).astype(np.float32)
    with torch.no_grad():
        ref = model(torch.from_numpy(test)).numpy()

    for provider in ("CPUExecutionProvider", "CUDAExecutionProvider"):
        sess = ort.InferenceSession(MODEL_OUT, providers=[provider])
        used = sess.get_providers()[0]
        out = sess.run(None, {"input": test})[0]
        diff = float(np.max(np.abs(out - ref)))
        print(f"{provider}: session used={used}  max|diff| vs torch = {diff:.3e}")
        inp = sess.get_inputs()[0]
        out_meta = sess.get_outputs()[0]
        print(f"  input:  {inp.name} {inp.shape} {inp.type}")
        print(f"  output: {out_meta.name} {out_meta.shape} {out_meta.type}")

    with torch.no_grad():
        out1 = model(torch.from_numpy(test[:1])).numpy()
    sess = ort.InferenceSession(MODEL_OUT, providers=["CPUExecutionProvider"])
    diff1 = float(np.max(np.abs(sess.run(None, {"input": test[:1]})[0] - out1)))
    print(f"batch=1 parity: max|diff| = {diff1:.3e}")
    print(f"zero-input torch value (reference): {ref_dummy[0][0]:.6f}")


if __name__ == "__main__":
    main()
