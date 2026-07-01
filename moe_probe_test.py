import numpy as np, mlx.core as mx
import mlx_lm.models.olmoe as olmoe
from mlx_lm import load

CAP = []
_orig = olmoe.OlmoeSparseMoeBlock.__call__
def patched(self, x):
    xf = x.reshape(-1, x.shape[-1])
    rw = mx.softmax(self.gate(xf), axis=1, precise=True)
    inds = mx.argpartition(-rw, kth=self.top_k - 1, axis=-1)[..., :self.top_k]
    mx.eval(inds)
    CAP.append((getattr(self, "_lid", -1), np.array(inds)))
    return _orig(self, x)
olmoe.OlmoeSparseMoeBlock.__call__ = patched

print("loading OLMoE-1B-7B 4-bit (downloads ~4GB first time)...", flush=True)
model, tok = load("mlx-community/OLMoE-1B-7B-0125-Instruct-4bit")
layers = getattr(model.model, "layers", None) or model.layers
n = 0
for i, layer in enumerate(layers):
    mlp = getattr(layer, "mlp", None)
    if isinstance(mlp, olmoe.OlmoeSparseMoeBlock):
        mlp._lid = i; n += 1
blk = next(l.mlp for l in layers if isinstance(getattr(l, "mlp", None), olmoe.OlmoeSparseMoeBlock))
print(f"MoE layers={n}  experts={blk.num_experts}  top_k={blk.top_k}", flush=True)

CAP.clear()
ids = tok.encode("def add(a, b):\n    return a + b")
mx.eval(model(mx.array([ids])))
print(f"captured {len(CAP)} MoE-layer routing tensors; layer-0 indices shape={CAP[0][1].shape}", flush=True)
print("first 5 tokens' top-k experts (layer 0):", CAP[0][1][:5].tolist(), flush=True)
print("FEASIBLE: real per-token expert routing extracted.", flush=True)
