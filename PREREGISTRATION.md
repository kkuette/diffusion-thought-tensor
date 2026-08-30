# Pre-registration: Scaling the Effective Context of a Self-Written Memory Bank

*A pre-registered three-scale ON/OFF study at fixed window W = 512.*

**Pre-registered: 2026-08-17, version 1.0, prior to any training run of the
campaign.**

This document fixes the object, the hypotheses, the experimental arms, the
data, the seeds, the adjudication rules, and the fallback tree of the
upcoming scaling campaign, before any campaign training run is launched.
The campaign configs are published alongside it (§9). Any change after
publication produces a new dated version; deviations are reported as
deviations (§9).

---

## 1. Pre-registered object and hypotheses

**Primary object:** the scaling law of the effective context bought by a
self-written memory bank — the deliverable this campaign exists to
produce, positive or null. At a
fixed attention window W = 512 tokens, we measure the *effective context*
of a transformer that writes and reads its own memory bank, at three model
scales, against the same backbone without the bank. "Effective context" is
never asserted; it is *measured* (§4, requirement 1; estimator in §5.2):
performance is a curve against the distance between a probe and its
evicted source, and the distance at which the memory model falls back to
the memory-free model *is* its effective context. The scaling law — how
that measured quantity moves across three scales — is the primary
deliverable, reported whatever it shows.

**Demarcation.** "Self-written memory bank" is used in a precise sense:
the memory content is selected and written by the model's own policy
during its ordinary forward pass (no re-encoding forward, no gradient at
write time), and read back through trained projections into attention.
This excludes non-parametric retrieval caches (kNN-style), compression
schemes that require a re-encoding forward per compression cycle
(gisting/ICAE), fixed-size recurrent state passed between segments, and
test-time weight updates. To our knowledge no scaling study of this
object exists; that novelty claim is confined to this sentence and will
be defended in the paper's related-work section, never in a title.

**Two pre-registered hypotheses**, each operationalized in §5.2, each able
to fail independently, with §8 applying to both:

- **H1 — the bank buys context.** At each scale, the memory model
  outperforms the memory-free model (lower NLL, paired) beyond the window:
  the effective-context estimator returns a distance strictly greater
  than W.
- **H2 — the advantage does not shrink with scale.** The bank's aggregate
  beyond-window advantage at the largest scale is not smaller than at the
  smallest scale, beyond the measured inter-run variance (§5.2, rule R4).

**Explicitly out of scope of this pre-registration:** the *overhead bar* —
whether the bank's effective context exceeds W + ΔW, where ΔW is the
bank's overhead converted into window tokens against a long-context twin
trained at W + ΔW. The overhead accounting is published in full below
(§1.1) and the twin arm is fully specified as a conditional extension
(§2, its configs are published with the others) but deliberately left out
of this campaign's plan: the overhead-bar comparison will be its own
pre-registration, run as a follow-up.

### 1.1 Overhead accounting (published here, not left to reviewers)

The memory model (ON) carries overhead its baseline (OFF) does not. We
publish the full accounting on all three axes, so that no reader has to
reconstruct it:

- **Information:** the bank exposes at most **S = 2048** key/value lines
  to attention — its nominal KV capacity (8 write events × 256 lines cap
  per write). Each line is the layer-0 embedding of exactly one token
  (RMS-normalized), so the natural unit conversion is 1 line = 1 window
  token. The bank never exposes more than 1792 readable lines by
  construction (the last write is never re-read), and in practice its
  admission controller admits far fewer lines than the cap (admission
  target `mem_write_rho_star: 16`, frozen in the published configs; the
  measured-exposure curve will be published as a descriptive figure,
  never as a conversion). Under this conversion the bank's capacity
  equals ΔW = 2048 window tokens; the conditional twin (§2) is therefore
  specified at seq_len 2560 = 5·W. Signal at distances beyond W + 2048
  cannot be explained by the bank's KV budget; the conversion exists so
  that statement is checkable.
- **Compute:** the read path costs 18–25% of the memory model's forward
  FLOPs across the three scales (paid on all 2048 slots even when empty,
  a fixed-shape requirement). A strict compute-matched conversion yields
  only **ΔW ≈ 114–171 tokens** depending on scale. The ×12–18 gap between
  compute-matched (~114–171) and information-matched (2048) ΔW is the
  quantified sub-linearity argument — the bank's cost per token of
  effective context is O(S) constant while window KV cost grows with W.
  **This same figure quantifies the ON/OFF compute asymmetry**: OFF pays
  no read path, so the trained contrast of this campaign is matched in
  data, steps and forwards (§2), not in FLOPs — ON receives 18–25% more
  forward compute. This asymmetry is published rather than hidden; it is
  one reason the overhead-bar claim is deferred to the twin follow-up
  rather than adjudicated here.
- **Parameters:** the bank's only learned parameters are per-layer KV
  read projections: +0.92 / 1.31 / 1.84 M across the three scales, i.e.
  **0.60–0.90% of active parameters** (0.14–0.23% of total). Parameters
  do not convert cleanly into window tokens (any params→tokens equation
  would be a scaling-law artifact), so this excess is published raw and
  uncompensated.

## 2. Experimental arms, scales, seeds

All arms share one trainer, one data pipeline, matched data order and
seeds. The write happens **inside the ordinary forward pass and adds no
additional forward**; ON and OFF therefore see identical token streams,
identical optimizer steps, and identical forward counts *by construction*.
The remaining asymmetry — the read path's 18–25% extra FLOPs, ON-only —
is published in §1.1.

| Arm | Status | Description |
|---|---|---|
| **ON** | base | Backbone + memory bank. The bank is written online by the model's own eviction-time write policy, inside the ordinary forward pass. |
| **OFF** | base | Same backbone with no bank modules built at all (trunk-only; its parameter count is exactly ON minus the read projections), trained at W = 512 on matched data with the same seed. |
| **Long-context twin** | deferred | Same trunk-only backbone, trained from scratch at seq_len 2560 = W + ΔW (§1.1), matched to ON in total training tokens. **Fully specified here — its configs are published with the others — but not part of this campaign's plan**: it belongs to the follow-up pre-registration of the overhead-bar claim (§1) and would be evaluated at its native window there. |
| **Gisting / ICAE-style compression** | exploratory | Evaluation-only compression baselines on frozen checkpoints. **Explicitly not pre-registered**: no protocol is frozen here, and no conclusion of this campaign rests on them. Reported, if run, as exploratory. |

**Scales:** three mixture-of-experts backbones of **399M / 697M / 1345M
total parameters** (102M / 166M / 304M active per token). Token budgets
are fixed at **30 tokens per total parameter** — ≈ 12.0 / 20.9 / 40.3 B
tokens — identical for ON and OFF at each scale; the conditional twin is
matched in total training tokens. The data mix is frozen and published in
full in §2.1 and in the configs (§9); it is identical across arms at each
scale.

**Seeds (pre-committed values):** the base plan is
**n = 2 seeds at the smallest scale, n = 1 at the two larger** — eight
runs: ON/OFF pairs at seeds 0 and 1 at 399M, plus the ON/OFF pairs at
seed 0 at 697M and at 1345M. The seed drives everything — model init,
data order, data sharding — and is identical across the arms of a pair,
so each ON/OFF pair shares its data order. The seed values are named here
so that no run can be selected after the fact. The trained inter-run
variance measured on the replicated scale is the gate for every
cross-run claim (§5.2, §6.1).

**Surplus priority (pre-declared):** remaining compute goes to seed
replicates only, in this order: (1) a seed-2 ON/OFF pair at 399M; (2) a
seed-1 ON/OFF pair at 697M. The twin is not in this queue (see its arm
entry). The order is declared now precisely so that the decision to fund
each item is not result-driven; any surplus item not run is reported as
not run.

**Incident policy:** a run that NaNs or diverges resumes from its last
checkpoint under the run's frozen guards; if it must restart from
scratch, it restarts with its assigned seed. No seed is ever replaced by
another. A run abandoned for infrastructure reasons is reported as
abandoned, never silently re-rolled.

### 2.1 Frozen data mix

Frozen in the configs on 2026-08-09 and published with this document
(§9); identical at all three scales and in every arm. Training follows a
two-stage WSD schedule: a stable ("cruise") stage covering **66%** of the
token budget, then a decay stage covering the remaining 34% (boundary
computed as round(0.66·T) by the schedule resolver, frozen per scale).

**Cruise mix** (11 sources, weights sum to 1.000):

| Source | Weight | Dataset |
|---|---|---|
| fineweb_edu | 0.30 | HuggingFaceFW/fineweb-edu (sample-100BT) |
| fineweb | 0.10 | HuggingFaceFW/fineweb (sample-100BT) |
| dclm | 0.10 | mlfoundations/dclm-baseline-1.0 |
| starcoder_python | 0.10 | bigcode/starcoderdata (python) |
| starcoder_c | 0.04 | bigcode/starcoderdata (c) |
| starcoder_rust | 0.04 | bigcode/starcoderdata (rust) |
| starcoder_js | 0.04 | bigcode/starcoderdata (javascript) |
| finemath | 0.09 | HuggingFaceTB/finemath (finemath-4plus) |
| infiwebmath | 0.05 | HuggingFaceTB/finemath (infiwebmath-4plus) |
| wikipedia | 0.06 | wikimedia/wikipedia (20231101.en) |
| cosmopedia_v2 | 0.08 | HuggingFaceTB/smollm-corpus (cosmopedia-v2) |

**Decay mix** (13 sources, weights sum to 1.000):

| Source | Weight | Dataset |
|---|---|---|
| fineweb_edu | 0.30 | HuggingFaceFW/fineweb-edu (sample-100BT) |
| starcoder_python | 0.07 | bigcode/starcoderdata (python) |
| starcoder_c | 0.03 | bigcode/starcoderdata (c) |
| starcoder_js | 0.03 | bigcode/starcoderdata (javascript) |
| finemath | 0.08 | HuggingFaceTB/finemath (finemath-4plus) |
| infiwebmath | 0.04 | HuggingFaceTB/finemath (infiwebmath-4plus) |
| wikipedia | 0.05 | wikimedia/wikipedia (20231101.en) |
| cosmopedia_v2 | 0.07 | HuggingFaceTB/smollm-corpus (cosmopedia-v2) |
| metamathqa | 0.05 | HuggingFaceTB/smoltalk (metamathqa-50k) |
| numina_cot | 0.06 | HuggingFaceTB/smoltalk (numina-cot-100k) |
| smol_smoltalk | 0.13 | HuggingFaceTB/smol-smoltalk |
| apigen | 0.06 | HuggingFaceTB/smoltalk (apigen-80k) |
| systemchats | 0.03 | HuggingFaceTB/smoltalk (systemchats-30k) |

**Programmatic components of the decay stage** (identical in every arm;
weights are fractions of the sequence stream, not renormalized token
counts): **37.5%** of decay-stage sequences are synthetic recall lives
(flat over the stage) — bounded-window lives with hard evictions and
probes in the §3 format; this is where the RESET regime is trained. The
remaining 62.5% draw from the decay-mix table; within that natural share,
a note-prefix transform applies to 1.5% of lives (compact "notes from
earlier" lines restating evicted values at window heads — the substrate
of future compaction baselines), drawn so that the recall exposure is
unchanged.

In the deferred twin's geometry (windows of 2560), the same recall
strand yields lives in which **~85% of beyond-window probes are
unlearnable without a bank** (measured on the life geometry, published
with the eval spec). This asymmetry is the contrast the twin exists to
embody at evaluation; its training-time cost to the twin — decay-budget
spent on targets it cannot learn — is acknowledged, is one of the
reasons the twin is deferred out of this campaign, and will be treated
head-on in the follow-up pre-registration of the overhead-bar claim.

Decontamination is applied at the data build wherever an audit found
evaluation leaks (~0.2% of documents dropped). Any change to this mix
after this document is a deviation (§9).

## 3. Task, probes, and evaluation material

Evaluation material consists of long sequences processed through the
bounded window W with hard evictions (RESET events). Probes query
information (spans, values, keys) whose source tokens have left the
window.

- **Lives and distance.** A life is one sequence processed as successive
  W-token windows with a write at each eviction. Distance is measured
  from a probe to its evicted source and banded at window granularity:
  bands 0–7 (up to ~7 windows ≈ 3584 tokens beyond W); the sparsely
  populated far bands 6–7 are aggregated into one far band. Claim-bearing
  evaluations use **≥240 lives per condition** (§6.1).
- **Primary battery (in-format).** Recall lives drawn by the published
  generator from **held-out shards** (per-source held splits, excluded
  from training at the data build). The generator template is the same as
  the training strand's (§2.1) — the primary battery measures the
  *trained* regime; this scoping is stated again in §7.
- **Secondary battery (natural, pre-registered).** Evicted-span probes
  drawn from held-out *natural documents* — no synthetic template: the
  same life geometry, distance protocol and estimator applied to spans
  that occur in real text. Evaluation-only, on the frozen checkpoints.
  Its role is to measure transfer of the trained regime outside its drill
  format; it is reported at the same rank as the primary battery in the
  paper, and H1/H2 are additionally reported *as computed on it* — but
  adjudication (§5.2) binds to the primary battery.
- **Automation.** Probe exclusion (the re-mention check of §4.2) and the
  crossing estimator (§5.2) are computed by scripts published with the
  harness — no adjudication step in this document is performed by visual
  inspection.

## 4. Protocol requirements (binding)

1. **The x-axis is distance beyond W, as a curve — never a binary ON/OFF
   comparison.** Performance is plotted against the distance between the
   probe and its evicted source. The distance at which ON falls back to
   OFF — computed by the §5.2 estimator — *is* the measured effective
   context.
2. **Probes are structurally evicted** at test time, with an automated
   check that no later re-mention has reintroduced the probed content
   into the window. Probes failing the check are excluded by the scripted
   pipeline before any ON/OFF contrast is computed.
3. **The bank under test is the one written by the run itself.** No
   teacher-forced or oracle bank content ever enters a claim-bearing
   evaluation. (§7 defines the write regime this scopes to.)
4. **Multi-RESET:** evaluation covers multiple successive evictions per
   life, not a single eviction event.
5. **OFF is matched in data, optimizer steps and forward count** — exact
   by construction, since the write adds no forward (§2).
6. **The overhead accounting is published** (§1.1), with the
   compute-matched figure and the raw parameter excess alongside it.

Additionally:

- **Spurious copying (pointer false positives) is a first-class metric**,
  not a sanity check: a retrieval pointer is judged by its false
  positives. Reported at the same rank as the headline metrics.

## 5. Adjudication

### 5.1 Metrics

- **Primary metric:** paired difference in live NLL on value spans,
  **Δ(d) = NLL_OFF − NLL_ON** per distance band d (positive = bank
  advantage), evaluated with the model's own running context and
  self-written bank, no teacher forcing. Confidence intervals are
  bootstrap CIs **paired by life** (same lives under ON and OFF).
- **Secondary metrics:** exact-match grade, retrieval rank-1 accuracy
  (r@1), and the spurious-copy rate (§4) — reported alongside every
  recall figure, never adjudicating alone.
- **Within-model ΔNLL (bank-ablated on the same checkpoint) is never a
  judge** — it is deprecated in this program: it rewards a model for
  being bad without the bank as much as for being good with it. It
  appears only as the descriptive load-bearing figure (F1/F3, §5.3).
- **Checkpoint rule:** the final checkpoint of the frozen schedule
  adjudicates. Token-milestone checkpoints are descriptive (the
  params × tokens surface of F1).

### 5.2 Estimator and decision rules

- **R1 — crossing estimator.** For a given scale, let b\* be the first
  distance band such that the paired 95% CI of Δ(d) contains 0 for b\*
  *and every farther band* (far aggregate included). The measured
  effective context is D_eff = W + (lower edge of b\*). If no such band
  exists in the probed range, D_eff is **censored** and reported as
  D_eff > W + 3584 — censoring is a reported outcome, never a claim of
  infinity.
- **R2 — H1 rule.** H1 is met at a scale if the paired 95% CI of Δ
  excludes 0 in the bank's favor at band 1 (the first fully evicted
  band). Where seed replicates exist, the rule applies to the
  seed-pooled estimate — H1 cannot be met by one favorable seed.
- **R3 — aggregate advantage.** A(s) = mean of Δ(d) over the evicted
  bands (equal band weights, far aggregate as one band), per scale, with
  paired CI; the trained inter-run standard deviation σ_run of A is
  estimated from the seed replicates at the smallest scale (pooled with
  any surplus replicates that run). Its limitation is stated in advance:
  σ_run is measured where replication is affordable and *assumed*
  comparable at the larger scales; every use of σ_run in the paper
  carries this caveat.
- **R4 — H2 rule.** H2 is met if A(largest) ≥ A(smallest) − σ_run. With
  three scale points this is an estimand with a guard, not a fitted law;
  the slope of A against log total parameters is additionally reported
  with the same guard, and no stronger scaling language than the rule
  supports will be used.
- **R5 — hierarchy (multiplicity control).** Hypotheses are tested in
  fixed order: H1 at 399M → H1 at 697M → H1 at 1345M → H2; each is
  interpreted confirmatorily only if every earlier test was met
  (gatekeeping). Everything downstream of a failed gate is reported
  descriptively.
- **R6 — direction and units.** All figures state Δ = NLL_OFF − NLL_ON
  explicitly; "the bank wins" always means Δ > 0. Model size axes state
  **total parameters**, with active parameters tabulated alongside.

### 5.3 Pre-registered figures

1. **F1 — the scaling law on one graph.** x = model size (total
   parameters; active tabulated), y = NLL on the held-out evaluation
   batteries, three curves on a single figure: **ON**, **OFF**, and **ON
   with the bank ablated at evaluation** (same ON checkpoint, bank
   masked). Each pairwise gap carries one meaning, fixed here: ON−OFF is
   the trained contrast (the law's headline); ON-ablated−ON is the
   within-model bank delta — *purely descriptive*, never a judge (§5.1):
   its role is to show the bank is **load-bearing** for the trained
   model, which genuinely depends on what it wrote; ON-ablated−OFF is
   what carrying the bank changes in the trunk itself. Final checkpoints
   adjudicate; token milestones shown as the descriptive params × tokens
   surface.
2. **F2 — effective-context curves**, one per scale: Δ(d) vs distance
   beyond W, with the paired CI band and the evaluation noise floor
   (§6.1). The D_eff values feeding H1 are computed by R1 and marked on
   this figure.
3. **F3 — bank-ablation delta vs distance**, per scale: the ON-ablated −
   ON gap as a function of distance beyond W. Purely descriptive, and
   adjudicates nothing: its role is to show that the bank is *imperative*
   for the trained model — masking it collapses the ON model's
   performance at exactly the distances where the window cannot help —
   and to locate where in distance that reliance lives.

The measured-exposure figure (§1.1), the natural-battery curves (§3), and
the closed-loop divergence measurement (§7) complete the pre-registered
figure set.

## 6. Analysis commitments

1. **Two variance sources, two roles — neither substitutes for the
   other.** (a) The **evaluation noise floor** was measured on the
   campaign hardware before this pre-registration: the null distribution
   (5 seeds × 24 evaluation lives on an untrained checkpoint) gives
   sd 0.018–0.029 NLL per distance band on populated bands, 0.06–0.09 on
   the sparse far bands, plus a negative bias of the null at long
   distance. Frozen consequences, already enforced in the evaluation
   harness: claim-bearing evaluations use ≥240 lives per condition, far
   bands are aggregated, and only paired contrasts are interpreted at
   distance — never the absolute sign of a single arm. This floor gates
   *per-band resolution within a run pair*. (b) The **trained inter-run
   variance** σ_run, measured from the base-plan seed replicates (§2),
   gates every *cross-run* claim (R3/R4). No effect below the applicable
   gate will be claimed.
2. **RESET-evaluation cadence frozen in the published configs**, and
   never reduced mid-flight. If throughput pressure forced cadence cuts,
   the distance curve would degenerate into an ON/OFF cross-entropy
   comparison unrelated to the hypotheses; the cadence therefore
   outranks throughput.
3. **Pre-declared fallback tree** if throughput disappoints, in strict
   order:
   1. surplus items (the extra seed replicates) are dropped first, in
      reverse priority order (§2);
   2. then tokens per scaling point are reduced from ×30 to ×20 (tokens
      per total parameter), applied identically to every arm;
   3. the largest scale point is dropped only as a last resort.

   The invariant that survives every fallback: **three model sizes ×
   ON/OFF, matched, with n = 2 seeds at the smallest scale.**

## 7. Scope and pre-declared limitations

**Write regime.** All hypotheses are scoped to **persistence under
procedural (non-autonomous) writing across resets**: the bank is written
by the model's trained eviction-time policy while it processes given
text — not while it consumes its own sampled outputs. In fully
closed-loop operation, generation currently diverges after **~10–17
turns** — an exposure-bias effect of the write policy, invisible to
teacher-forced evaluation. Pre-registered as a *quantified limitation*:
turns-before-divergence, ON vs OFF, measured on the frozen final
checkpoints of all three scales — pure evaluation, no further training.
If it grows with scale, it is a figure of this paper; if not, it defines
the follow-up work. Either way it is reported.

**Evaluation format.** The primary battery is in-format with the trained
recall strand (§3): the adjudicated quantity is the scaling of the
*trained* capability. The natural-document battery (§3) measures
transfer beyond the drill format and is reported at equal rank; the
paper's conclusions will not extrapolate past what that battery shows.

## 8. What counts as a result

The design is symmetric with respect to outcomes. A flat or negative
scaling law — the bank's advantage failing to appear (H1), or shrinking
with scale (H2) — **is a publishable result of this pre-registered
protocol**, and will be published with the same figures, the same
baselines, and the same error bars as a positive one. The scaling law is
the deliverable; the hypotheses are bars laid against it, and each is
reported as met or not met by the §5.2 rules, independently. Censored
outcomes (R1) are reported as censored.

## 9. Verifiability, reporting, governance

- **Published with this document:** the 18 campaign configs
  (`prereg/configs/`, comment-stripped, parse-identical to the frozen
  originals) — every training hyperparameter, the full data mix, the
  admission target (`mem_write_rho_star: 16`), the schedule, the seeds,
  and the conditional twin configs.
- **Code:** the training and evaluation harness will be released in full
  with the paper, at the exact campaign commit. Bugfixes to the trainer
  after this date are permitted and will be listed; the frozen objects
  are the configs, the mix, the seeds, and the adjudication rules of
  this document.
- **Versioning:** this is version 1.0. Any post-publication change
  produces a new version with a dated changelog; analyses follow the
  latest version published before the first campaign run; anything else
  is a deviation, listed in the paper's deviations section.
- **Reporting:** all arms, all pre-registered metrics and figures
  (§5.3), both variance gates (§6.1), the overhead accounting (§1.1),
  the measured-exposure figure, the natural-battery results, and the
  closed-loop divergence measurements appear in the paper regardless of
  outcome.
