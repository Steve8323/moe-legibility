"""
Plain-language profiling of MoE experts. For the most specialized (layer, expert)
pairs, we hand their distinctive tokens + dominant category to a small trusted
model (llama3.2:3b) and ask for a one-line, plain-English description of what the
expert appears to specialize in -- the legibility step (cf. the paraphrase paper).
Saves moe_profiles.json.
"""
import json
import urllib.request
import numpy as np

HOST = "http://localhost:11434"
MODEL = "llama3.2:3b"

d = np.load("moe_data.npz", allow_pickle=True)
spec, dom, cats = d["spec"], d["dom"], list(d["cats"])
tokens = json.load(open("moe_tokens.json"))

# rank specialized experts that have interpretable distinctive tokens
cands = []
for l in range(spec.shape[0]):
    for e in range(spec.shape[1]):
        toks = tokens.get(f"{l}_{e}", [])
        if len(toks) >= 3:
            cands.append((float(spec[l, e]), l, e, cats[int(dom[l, e])], toks))
cands.sort(reverse=True)
top = cands[:16]

SYS = ("You describe what KIND OF TEXT a neural-network expert activates on, from its "
       "most frequent tokens. Reply with ONE short noun phrase (<= 8 words) naming the "
       "TOKEN TYPE or TEXT PATTERN itself -- e.g. 'arithmetic symbols and numbers', "
       "'speech verbs (said, asked, replied) and quotation marks', 'Chinese characters', "
       "'Python syntax: def, return, brackets'. Describe the tokens, NOT an application, "
       "product, or research field. No preamble, just the phrase.")


def describe(cat, toks):
    body = {"model": MODEL, "system": SYS, "stream": False,
            "prompt": f"Dominant category: {cat}\nTop tokens: {toks}\nSpecialty:",
            "options": {"temperature": 0.3, "num_predict": 24}}
    req = urllib.request.Request(HOST + "/api/generate",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())["response"].strip().strip('".').replace("\n", " ")


profiles = []
print("plain-language expert profiles (genuine routing -> 3B description):\n", flush=True)
for s, l, e, cat, toks in top:
    desc = describe(cat, toks)
    profiles.append({"layer": l, "expert": e, "spec": round(s, 2),
                     "category": cat, "tokens": toks, "description": desc})
    print(f"  L{l:2d} E{e:2d} (spec {s:.2f}, {cat:8s}): {desc}", flush=True)
    print(f"          evidence tokens: {toks[:6]}", flush=True)

json.dump(profiles, open("moe_profiles.json", "w"), indent=2)
print(f"\nsaved {len(profiles)} profiles to moe_profiles.json", flush=True)
