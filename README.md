# Reading the Router: plain-language profiles of Mixture-of-Experts specialization

Can we make a Mixture-of-Experts router **legible**? This applies a legibility
agenda — rendering an opaque internal into plain English — to the *routing* of
an MoE model. We extract genuine per-token expert assignments from
**OLMoE-1B-7B** (16 layers, 64 experts, top-8) on a category-labeled corpus,
quantify how specialized each expert is, and hand each specialized expert's
distinctive tokens to a small trusted model that writes a one-line, plain-English
description of what it does.

📄 **Paper: [`paper/paper2.pdf`](paper/paper2.pdf)** (source: [`paper/paper2.tex`](paper/paper2.tex))

## Findings (OLMoE-1B-7B, 1,541-token category-labeled corpus)

- **Most experts are balanced** — mean specialization 0.19 — but a **legible
  specialized tail exists**: 2.1% of layer×expert units score > 0.5.
- **Every category has a strongly dedicated expert**: peak routing propensity
  **0.88–0.99** vs a 0.125 uniform baseline.
- **Specialization concentrates in late layers.**
- The **plain-language profiles are frequently accurate and human-readable** —
  e.g. a *"calculus and trigonometric notation"* expert, a *"Python syntax"*
  expert, a *"speech verbs and quotation marks"* expert, and shared
  Romance-language experts.

The argument: router-level plain-language profiling is a **cheap, model-agnostic
legibility tool**. It is also honestly limited — specialization is modest,
category-defined, and correlational, and the descriptions are only as good as
the labeller (see the paper's Limitations).

## Run

```bash
# 1. Extract genuine routing from OLMoE (MLX; captures top-8 expert selections)
python moe_route.py         # -> moe_data.npz, moe_tokens.json

# 2. Plain-language profiling of the most specialized (layer, expert) pairs
#    via a small trusted local model (llama3.2:3b through Ollama at :11434)
python moe_profile.py       # -> moe_profiles.json

# 3. Figures for the paper
python paper/moe_figs.py
```

## Layout

```
moe_route.py       extract + aggregate OLMoE routing -> moe_data.npz, moe_tokens.json
moe_profile.py     plain-language profiling of specialized experts -> moe_profiles.json
moe_probe_test.py  MLX hook that captures OLMoE top-k expert selections
moe_profiles.json  the generated one-line expert descriptions
paper/paper2.tex   the paper (compile with `tectonic paper/paper2.tex`)
paper/moe_figs.py  figure generation
```

## Context

Companion to work on *paraphrase bottlenecks* for oversight — applying the same
"make the internal legible in plain English" idea to MoE routing rather than to
a chain of thought.

Released under the MIT License.
