# Reproducing the paper

`run_all.sh` reproduces Tables 1–4 (PDF numbering) and Figures 3–5 of the paper from a
fresh clone:

```bash
bash repro/run_all.sh               # 3 training runs (GPU) + probes + figures
bash repro/run_all.sh --skip-train  # probes + figures on existing checkpoints
```

## What it runs

| stage | artifact | cell | hardware |
|---|---|---|---|
| train `..._s128_dsv4m.yaml` | fixed-structure model (Table 2, Fig 5 zero-shot arm) | seed 42, 4000 steps | ~5 h on one RTX 3090 |
| train `..._s128struct_dsv4w.yaml` | policy model (Tables 1/3, Figs 3–5) | seed 42; paper uses step 3000 | ~5 h |
| train `..._s128struct_dsv4w_s43.yaml` | policy model, replication (Tables 1/4) | seed 43, 4000 steps | ~5 h |
| `analysis/ttt_demo.py` | Table 2 (bank / TTT / ICL / ablate; held, train, subtraction) | dsv4m final | GPU if available, else CPU |
| `analysis/ttt_demo_act2.py` | Table 3 (replacement vs sequential TTT) | dsv4w@3000 | GPU if available, else CPU |
| `analysis/switch_probe_k2.py` (+`--sweep --dump`) | Tables 1/4 switch rows, Figure 5 | all three models | GPU if available, else CPU |
| `analysis/superposition_probe.py` | Figure 4 | both policy seeds | GPU if available, else CPU |
| `paper/figures/make_fig{3,4,5}.py` | Figures 3–5 (png+pdf) | — | CPU |

Outputs land in `repro/out/`. Figures 1–2 are hand-drawn SVG masters
(`paper/figures/fig{1,2}_*.svg`), re-rendered with
`paper/figures/make_fig{1,2}.py` (needs `svglib reportlab pymupdf`).

## Determinism

Data generators are seeded from the configs (seeds 42/43); every probe
sets `torch.manual_seed(0)` and builds its conversations with the CPU
RNG, so the evaluation data is identical on CPU and GPU. The bank's
random seed slots are drawn on the model's device, whose RNG stream
differs between CPU and CUDA — probe numbers therefore shift by a few
tenths of a point across devices (e.g. Table 2 held bank 0.799 CPU vs
0.793 CUDA). Training on GPU is deterministic up to
cuDNN/atomics noise: expect the paper's numbers to within a few points,
with the documented seed-level bifurcation (§9) — the *selectivity of
replacement* is basin-dependent, and a re-run of either seed may land in
either attractor. Checkpoints are saved every 100 steps; the paper's
probes read `dsv4m/final.pt`, `dsv4w/step_3000.pt`, `dsv4w_s43/final.pt`.

## Checkpoint integrity

**This is an integrity check on our artifacts, not a validation of your re-run.** Training
is not bit-reproducible across machines (cuDNN/atomics noise, and the seed-level
bifurcation of §9), so a re-run legitimately produces a *different* SHA. What this table
catches is our own copies going missing or getting corrupted — which is exactly what
happened: as of 2026-08-02 the `dsv4m` and `dsv4w` (seed 42) checkpoints and their
`runs/*/metrics.jsonl` had been lost, making Figure 3 unreproducible from stored artifacts
until they were regenerated.

| artifact | step | keys | params | sha256 (first 16) |
|---|---|---|---|---|
| `dsv4w_s43/step_4000.pt` | 4000 | 221 | 3 076 385 | `0b18b1c84aacf029` |
| `dsv4m/final.pt` | — | — | — | *(regenerating)* |
| `dsv4w/step_3000.pt` | — | — | — | *(regenerating)* |

`params` is the *model's* parameter count, the number the training banner prints. Note that
naively summing `numel()` over the state_dict keys overcounts it by 17 408 here: `embed` and
`lm_head` are tied (`lm_head.weight = embed.weight`), so the same tensor is stored under two
keys. Deduplicate by storage:

```bash
sha256sum <ckpt>.pt | cut -c1-16
python -c "import torch,sys; c=torch.load(sys.argv[1],map_location='cpu',weights_only=False); \
s=c['model']; u={v.data_ptr(): v.numel() for v in s.values()}; \
print(c.get('step'), len(s), sum(u.values()))" <ckpt>.pt
```

## Environment

`setup_environment.sh` creates the conda env (python 3.10, torch, yaml,
matplotlib). Override the interpreter with `PY=... bash repro/run_all.sh`.
