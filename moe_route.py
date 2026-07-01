"""
Extract and aggregate genuine MoE routing from OLMoE-1B-7B (16 layers, 64 experts,
top-8). We run a category-labeled corpus, record which experts each token selects
at each layer, and build:
  - counts[layer, expert, category]  (selections)
  - per-(layer,expert) distinctive tokens (by lift over global frequency)
  - per-(layer,expert) specialization score + dominant category
Saves moe_data.npz and moe_tokens.json for figures and plain-language profiling.
"""
import json
import re
from collections import defaultdict, Counter

import numpy as np
import mlx.core as mx
import mlx_lm.models.olmoe as olmoe
from mlx_lm import load

MODEL = "mlx-community/OLMoE-1B-7B-0125-Instruct-4bit"

# ---- routing capture hook ------------------------------------------------- #
CAP = []
_orig = olmoe.OlmoeSparseMoeBlock.__call__
def patched(self, x):
    xf = x.reshape(-1, x.shape[-1])
    rw = mx.softmax(self.gate(xf), axis=1, precise=True)
    inds = mx.argpartition(-rw, kth=self.top_k - 1, axis=-1)[..., :self.top_k]
    mx.eval(inds)
    CAP.append((self._lid, np.array(inds)))
    return _orig(self, x)
olmoe.OlmoeSparseMoeBlock.__call__ = patched

# ---- category-labeled corpus --------------------------------------------- #
CORPUS = {
    "python": [
        "def quicksort(arr):\n    if len(arr) <= 1: return arr\n    pivot = arr[0]",
        "for i in range(n):\n    total += weights[i] * x[i]\n    print(total)",
        "class Node:\n    def __init__(self, val):\n        self.val = val\n        self.next = None",
        "import numpy as np\narr = np.zeros((3, 4))\nreturn arr.sum(axis=1)",
    ],
    "math": [
        "Let f(x) = 3x^2 + 2x - 5. Then f'(x) = 6x + 2 and the root is x = 5/3.",
        "The integral of 1/x from 1 to e equals ln(e) - ln(1) = 1.",
        "If a^2 + b^2 = c^2 and a = 3, b = 4, then c = 5 by the theorem.",
        "Solve 2x + 7 = 19, so 2x = 12 and therefore x = 6.",
    ],
    "english": [
        "The old lighthouse stood against the storm, its lamp sweeping the dark water.",
        "She walked slowly through the quiet morning, thinking about the long year ahead.",
        "Few people remember the village as it once was, before the railway came.",
        "The argument, though elegant, rested on an assumption no one had examined.",
    ],
    "french": [
        "Le vieux phare resistait a la tempete, sa lumiere balayant l'eau sombre.",
        "Elle marchait lentement dans le matin tranquille, pensant a l'annee a venir.",
        "Peu de gens se souviennent du village tel qu'il etait autrefois.",
        "L'argument, bien qu'elegant, reposait sur une hypothese jamais examinee.",
    ],
    "spanish": [
        "El viejo faro resistia la tormenta, su luz barriendo el agua oscura.",
        "Ella caminaba despacio por la manana tranquila, pensando en el ano que venia.",
        "Pocas personas recuerdan el pueblo como era antes de que llegara el tren.",
        "El argumento, aunque elegante, se apoyaba en una suposicion no examinada.",
    ],
    "json": [
        '{"name": "Alice", "age": 30, "roles": ["admin", "user"], "active": true}',
        '{"id": 42, "items": [{"sku": "A1", "qty": 3}, {"sku": "B2", "qty": 7}]}',
        '{"status": "ok", "data": {"count": 128, "next": null}, "errors": []}',
        '{"config": {"lr": 0.001, "epochs": 50, "layers": [64, 32, 16]}}',
    ],
    "dialogue": [
        '"Are you coming tonight?" she asked. "I might," he said, "if the rain stops."',
        '"Did you finish it?" - "Almost. Give me ten more minutes." - "Okay, hurry."',
        '"I told you this would happen," he muttered. "You always say that," she replied.',
        '"Where did you put the keys?" "On the table, like always." "They are not here."',
    ],
    "chinese": [
        "老灯塔在风暴中屹立，灯光扫过黑暗的水面。",
        "她在安静的清晨慢慢走着，思考着漫长的一年。",
        "很少有人记得火车来之前那个村庄的样子。",
        "这个论点虽然优雅，却建立在未经检验的假设上。",
        "今天的天气很好，我们一起去公园散步吧。",
        "他打开窗户，看着远处的山和河流。",
        "学习一门新的语言需要时间和耐心。",
        "这本书讲述了一个关于勇气和友谊的故事。",
    ],
}
# enlarge every category so distinctive tokens are statistically meaningful
CORPUS["python"] += [
    "def fib(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a",
    "result = [x * 2 for x in data if x > 0]\nprint(len(result))",
    "with open(path) as f:\n    lines = f.readlines()\n    return [l.strip() for l in lines]",
    "try:\n    value = int(s)\nexcept ValueError:\n    value = 0\nreturn value",
]
CORPUS["math"] += [
    "The derivative of sin(x) is cos(x), and the integral of cos(x) is sin(x) + C.",
    "By induction, the sum 1 + 2 + ... + n equals n(n+1)/2 for all n >= 1.",
    "Given the matrix A with det(A) = 0, the system Ax = b has no unique solution.",
    "The limit of (1 + 1/n)^n as n approaches infinity is the constant e = 2.718.",
]
CORPUS["english"] += [
    "By the time the letter arrived, the season had changed and the harvest was in.",
    "He never spoke of the journey again, though it had marked him deeply.",
    "The committee debated for hours but reached no decision before nightfall.",
    "A thin light fell across the room, and somewhere a clock began to chime.",
]
CORPUS["json"] += [
    '{"users": [{"id": 1, "name": "Bob"}, {"id": 2, "name": "Eve"}], "total": 2}',
    '{"meta": {"page": 3, "size": 20}, "results": [], "error": null, "ok": false}',
    '{"point": {"x": 1.5, "y": -2.0, "z": 0.0}, "label": "origin", "tags": ["a"]}',
    '{"order": {"id": "X9", "lines": [{"sku": "Q", "qty": 5, "price": 9.99}]}}',
]
CORPUS["dialogue"] += [
    '"You came back," she whispered. "Of course I did," he answered, smiling.',
    '"Is it true?" "I think so." "Then we have to tell them." "Not yet, please."',
    '"What time is it?" he asked. "Almost midnight," she said. "We should go."',
    '"Promise me," she said quietly. "I promise," he replied, and meant it.',
]
CORPUS["french"] += [
    "Quand la lettre arriva, la saison avait change et la recolte etait faite.",
    "Il ne parla plus jamais du voyage, bien qu'il l'eut profondement marque.",
    "Le comite debattit pendant des heures sans prendre aucune decision.",
    "Une lumiere mince traversait la piece, et quelque part une horloge sonna.",
]
CORPUS["spanish"] += [
    "Cuando llego la carta, la estacion habia cambiado y la cosecha estaba hecha.",
    "Nunca volvio a hablar del viaje, aunque lo habia marcado profundamente.",
    "El comite debatio durante horas sin tomar ninguna decision esa noche.",
    "Una luz delgada cruzaba la habitacion, y en algun lugar sono un reloj.",
]

print(f"loading {MODEL} ...", flush=True)
model, tok = load(MODEL)
layers = getattr(model.model, "layers", None) or model.layers
for i, layer in enumerate(layers):
    mlp = getattr(layer, "mlp", None)
    if isinstance(mlp, olmoe.OlmoeSparseMoeBlock):
        mlp._lid = i
blk = next(l.mlp for l in layers if isinstance(getattr(l, "mlp", None), olmoe.OlmoeSparseMoeBlock))
NL = sum(1 for l in layers if isinstance(getattr(l, "mlp", None), olmoe.OlmoeSparseMoeBlock))
NE, TOPK = blk.num_experts, blk.top_k
cats = list(CORPUS)
NC = len(cats)
print(f"layers={NL} experts={NE} top_k={TOPK} categories={cats}", flush=True)

counts = np.zeros((NL, NE, NC), dtype=np.int64)         # selections per (layer,expert,cat)
tok_count = np.zeros(NC, dtype=np.int64)
global_tok = Counter()                                   # token -> total occurrences
le_tok = defaultdict(Counter)                            # (layer,expert) -> token Counter

for ci, cat in enumerate(cats):
    for text in CORPUS[cat]:
        ids = tok.encode(text)
        toks = [tok.decode([t]) for t in ids]
        tok_count[ci] += len(ids)
        for t in toks:
            global_tok[t] += 1
        CAP.clear()
        mx.eval(model(mx.array([ids])))
        for lid, inds in CAP:                            # inds: (seq_len, top_k)
            for p in range(inds.shape[0]):
                tstr = toks[p]
                for e in inds[p]:
                    counts[lid, int(e), ci] += 1
                    le_tok[(lid, int(e))][tstr] += 1
    print(f"  done {cat} ({tok_count[ci]} tokens)", flush=True)

# ---- specialization + distinctive tokens --------------------------------- #
prop = counts / np.maximum(tok_count[None, None, :], 1)  # P(expert selected | category)
qn = prop / np.maximum(prop.sum(axis=2, keepdims=True), 1e-9)
ent = -(qn * np.log(np.maximum(qn, 1e-12))).sum(axis=2)
spec = 1 - ent / np.log(NC)                              # 0=uniform, 1=one category
dom = prop.argmax(axis=2)                                # dominant category index

total_tokens = sum(global_tok.values())
def distinctive(lid, e, topn=8):
    sel = le_tok[(lid, e)]
    tot = sum(sel.values()) or 1
    scored = []
    for t, c in sel.items():
        if c < 2 or not t.strip():
            continue
        lift = (c / tot) / (global_tok[t] / total_tokens)
        scored.append((lift, c, t))
    scored.sort(reverse=True)
    return [t for _, _, t in scored[:topn]]

tokens_out = {}
for lid in range(NL):
    for e in range(NE):
        tokens_out[f"{lid}_{e}"] = distinctive(lid, e)

np.savez("moe_data.npz", counts=counts, prop=prop, spec=spec, dom=dom,
         tok_count=tok_count, cats=np.array(cats))
json.dump(tokens_out, open("moe_tokens.json", "w"))

# ---- console summary: most specialized experts --------------------------- #
print("\n=== most specialized (layer, expert) -> dominant category + tokens ===", flush=True)
flat = [(spec[l, e], l, e) for l in range(NL) for e in range(NE)]
flat.sort(reverse=True)
for s, l, e in flat[:12]:
    print(f"  L{l:2d} E{e:2d}  spec={s:.2f}  cat={cats[dom[l,e]]:8s}  "
          f"tokens={distinctive(l, e, 6)}", flush=True)
print(f"\nmean specialization={spec.mean():.3f}  (0=balanced, 1=fully specialized)")
print("saved moe_data.npz, moe_tokens.json")
