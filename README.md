![Fractale](assets/fractale-banner.png)

# Thought Bank

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21225721.svg)](https://doi.org/10.5281/zenodo.21225721)

Research on **persistent thought memory** for language models: a **bank** the
model writes to in-line, *outside* its context window, and reads back by
attention — session state that survives context resets at zero window cost.
This work is the science behind the **FRACTALE** model series.

This repository is the **public record** of that research: the published
paper, its frozen reproduction snapshot, and the pre-registration of the
ongoing scaling campaign. The active development of the training stack
continues privately (see [What is open, what is not](#what-is-open-what-is-not)).

## 📄 Paper

**A Trained Fast-Weight Memory: Continual Rule Binding at Inference
Without Backward** — [PDF](paper/paper.pdf) ·
[DOI 10.5281/zenodo.21225721](https://doi.org/10.5281/zenodo.21225721)
(this version; all versions: [10.5281/zenodo.21222901](https://doi.org/10.5281/zenodo.21222901))

Three claims, all on a 3.08M-parameter DeepSeek-style trunk with an
8-slot bank, two seeds:

1. **A functional, generalizing memory**: a single 13-token presentation
   installs a *never-trained* rule binding at **0.79–1.00** accuracy on
   unseen queries (chance 0.008), retained past physical slot eviction,
   replaced mid-conversation in one forward pass (old-rule persistence
   0.000).
2. **The only functional adaptation pathway**: on the same conversations,
   test-time training fits its adaptation examples (0.99) and transfers
   **exactly nothing**, at **138×** the cost per update and −62%
   catastrophic interference on a concurrent rule (bank: −14%, by
   eviction); in-window ICL is at chance.
3. **Memory policy is a trained behaviour, not an architectural
   property**: the identical architecture trained on fixed-structure
   conversations perseverates totally on a rule switch, zero-shot
   (old-rule persistence 1.000, unreadable dirty-bank writes);
   randomizing conversation *structure* at training time installs the
   full policy.

### Reproducing the paper

The complete code, configs and instructions for reproducing Tables 1–4 and
Figures 3–5 are frozen at the tag
[`V0.2.2-preprint`](https://github.com/kkuette/thought-bank/tree/V0.2.2-preprint),
archived at the DOI above:

```bash
git clone --branch V0.2.2-preprint https://github.com/kkuette/thought-bank.git
cd thought-bank
bash repro/run_all.sh               # 3 training runs (~5 h each, one RTX 3090) + probes + figures
bash repro/run_all.sh --skip-train  # probes + figures on existing checkpoints
```

## 🔬 Pre-registered scaling campaign

The next step is pre-registered **before any campaign training run**:
[PREREGISTRATION.md](PREREGISTRATION.md) (v1.0, 2026-08-17) fixes the
object — *the scaling law of the effective context bought by a self-written
memory bank*, measured ON vs OFF at three model scales at a fixed attention
window — along with the hypotheses, arms, data, seeds, adjudication rules
and fallback tree. The 18 frozen campaign configs are published alongside it
in [prereg/configs/](prereg/configs/). Any change after publication produces
a new dated version; deviations will be reported as deviations.

Results will be reported whatever they show, positive or null.

## What is open, what is not

- **Open**: this paper and its frozen reproduction snapshot
  (`V0.2.2-preprint`), the campaign pre-registration and its configs, the
  campaign's resulting model weights (to be released on Hugging Face, ON and
  OFF arms), the evaluation battery reproducing the claims, and the
  [`fractale-sdk`](https://github.com/fractale-lm/fractale-sdk) inference
  kit.
- **Proprietary**: the training stack and recipe of the in-model memory
  (losses, curriculum, data mix, write path). This repository's history was
  rewritten on 2026-08-30 to reflect that boundary; the paper's reproduction
  snapshot is unaffected.

## 📚 Links

- Inference kit: [fractale-lm/fractale-sdk](https://github.com/fractale-lm/fractale-sdk)
- Models: [huggingface.co/fractale-lm](https://huggingface.co/fractale-lm)

## License

[MIT](LICENSE) — © 2026 Tony Denion. The archived version of record of
the paper is the Zenodo deposit above.
