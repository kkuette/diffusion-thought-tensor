from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml
from .paths import expand


@dataclass
class ThoughtBankConfig:
    # ── Core dimensions ──────────────────────────────────────────────────────
    vocab_size: int = 32000
    d_model: int = 256
    n_layers: int = 6          # total blocks; even → CSA, odd → HCA
    n_heads: int = 4           # query heads per attention layer
    d_head: int = 64           # compressed KV head dimension
    dropout: float = 0.0
    max_seq_len: int = 2048

    # ── mHC: Manifold-Constrained Hyper-Connections (§2.2) ───────────────────
    n_hc: int = 2              # residual stream expansion factor
    sinkhorn_iters: int = 20   # Sinkhorn-Knopp iterations (paper uses t_max=20)
    # À n_hc == 2, la projection de Birkhoff a une forme CLOSE : la boucle de 20
    # itérations approche un point fixe qu'on sait écrire exactement (voir
    # mhc._sinkhorn). Elle coûte 26 % des ops du forward pour un résultat qui
    # n'est doublement stochastique qu'à 2.5e-2 près. OFF par défaut : le contrat
    # de non-régression du dépôt veut qu'une config existante reproduise à
    # l'identique. À poser dans toute config neuve (n_hc vaut 2 partout).
    sinkhorn_closed_form: bool = False

    # ── Décodage : réduire le NOMBRE de dispatches ────────────────────────────
    # Le décodage token-par-token est LAUNCH-BOUND (FINDINGS 2026-07-27 : 12,9k
    # ops aten par forward AVEC cache KV, ~137 ms/token sur 3090 pour ~10 ms de
    # calcul utile). Ces flags coupent des lancements SANS changer un bit des
    # sorties (preuve : A/B float64 torch.equal dans les self-tests attention/
    # model/moe/decode). OFF par défaut — le contrat de non-régression du dépôt
    # veut qu'une config existante reproduise à l'identique.
    #
    # decode_fuse : (1) table RoPE mémoïsée — `_rope_at` recréait arange+outer+
    # cos+sin par couche et par token ; (2) hoisting fw_A/fw_B — la banque est
    # CONSTANTE entre deux writes (l'invariant du projet : seule modif = append
    # au `<think>`), or le read les reprojetait à chaque token ; mémo par
    # identité d'objet + compteur _version, donc un write (nouveau tenseur via
    # cat) ou une mutation en place invalident seuls ; (3) balance_loss MoE
    # coupée sous no_grad — de la télémétrie d'entraînement calculée à chaque
    # token décodé.
    decode_fuse: bool = False
    # decode_dense_moe : à BT == 1 (décodage caché non batché) et sous no_grad,
    # évalue les n_experts DENSÉMENT et pondère par les poids top-k, au lieu de
    # la double boucle Python (top_k × n_experts = 96 itérations/token à 12
    # couches, chacune avec un .any() et un masquage booléen ⇒ ~190 syncs GPU
    # par token). ~5 dispatches et 0 sync contre ~1 500 ops, pour 2× les FLOPs
    # d'une quantité négligeable. BT == 1 STRICT : à BT ≥ 2 la boucle évalue
    # chaque expert sur un sous-ensemble des lignes et un GEMM M=1 vs M=2 ne
    # rend pas les mêmes bits (mesuré : rel ~1.2e-16 même en float64 CPU) — à
    # BT=1 le sous-ensemble élu EST le batch, mêmes bits garantis (torch.equal,
    # self-test moe). Le décodage BATCHÉ (rollouts RL) garde la boucle.
    # Prérequis de la capture CUDA graphs (shapes fixes, plus de data-dependent).
    decode_dense_moe: bool = False
    # decode_static_cache : le `comp` de chaque AttnCache devient un buffer
    # PRÉALLOUÉ à capacité fixe (ceil(max_seq_len / m) blocs par couche, zéros
    # au-delà des blocs fermés) écrit en place — plus un seul cat qui
    # réalloue en grossissant. Les LARGEURS DE CALCUL restent les largeurs
    # historiques (narrow du buffer) : c'est mesuré, pas prudent — présenter
    # le buffer plein au top-k CSA change ses ex æquo (relu ⇒ 0.0 exacts en
    # régime permanent, départagés par une règle qui dépend de la largeur ⇒
    # décodage divergent O(1)), et élargir le softmax HCA change l'arbre de
    # sommation (~5e-15). Bit-identique au cache historique (torch.equal
    # float64, self-tests attention + decode). Les shapes PLEINES fixes — la
    # condition de capture CUDA graph — sont l'affaire du runner decode_graphs
    # (opt-in, classe ULP, comme decode_cache), qui s'appuie sur ces buffers.
    decode_static_cache: bool = False

    # ── Attention (§2.3) ──────────────────────────────────────────────────────
    csa_m: int = 4             # CSA: compress every m tokens (overlapping)
    hca_m: int = 16            # HCA: compress every m' tokens (non-overlapping, m' >> m)
    top_k_csa: int = 8         # CSA: top-k compressed blocks to attend
    n_win: int = 16            # sliding window tokens for local fine-grained attention
    d_latent_q: int = 64       # low-rank query compression dimension
    n_groups: int = 2          # grouped output projection groups

    # ── MoE (§2.1, DeepSeekMoE) ───────────────────────────────────────────────
    n_experts: int = 8         # routed expert count
    n_shared: int = 1          # shared experts (always active)
    top_k_experts: int = 2     # routed experts activated per token
    d_ff: int = 512            # FFN hidden dim per expert

    # ── Fast-weight thought bank ──────────────────────────────────────────────
    # The bank is a rolling FIFO buffer of M thought vectors (dim = mem_dim). It is
    # READ as fast WEIGHTS: each slot parametrises a low-rank MLP layer applied in
    # sequence to the text stream (see DualModalBlock._cross_modal). It is WRITTEN by
    # a gated head that appends one vector per forward. There is no separate
    # thought-stream transformer — the text model writes vectors and reuses them.
    mem_dim: int = 64          # thought-vector dimension (= fast-weight code size)
    max_mem: int = 32          # max bank size (FIFO-evict the oldest beyond this)
    mem_seed_slots: int = 4    # random-uniform[0,1] slots seeding a fresh bank
    mem_seed_init: str = "uniform01"  # "uniform01" | "pm1" (uniform[-1,1]) | "zeros" — ablation idée E
    mem_read_rank: int = 16    # bottleneck rank of each per-slot fast-weight layer
    mem_read_dropout: float = 0.0  # dropout inside the fast-weight MLP layers

    # ── Model selection ───────────────────────────────────────────────────────
    # True  → ThoughtBankLM (text stream + fast-weight thought bank).
    # False → TrunkLM (legacy text-only, optional bolt-on memory below).
    use_dual_stream: bool = True
    use_thought_memory: bool = False    # legacy TrunkLM bolt-on memory only

    # ── Training ──────────────────────────────────────────────────────────────
    balance_loss_weight: float = 1e-4   # MoE balance loss weight (paper §4.2.2: 0.0001)
    # Sparsity budget on the write-decision α: adds cost · E[-log(1-α)] to the loss
    # so writing has an opportunity cost. 0.0 = no penalty (α free to saturate at 1).
    # The budget form keeps a live gradient even when α≈1 (unlike an L1 on α).
    mem_write_cost: float = 0.0
    # Novelty-gated write: penalise the cosine of each new write to the closest
    # existing bank slot (adds weight · E[max_j cos(m_new, slot_j)] to the loss).
    # Trains the write head to store DIVERSE thoughts instead of near-duplicates —
    # without it the bank collapses to ~1 effective vector. 0.0 = off.
    mem_write_diversity: float = 0.0
    # Code-space augmentation: std of Gaussian noise added to each newly written
    # vector during TRAINING only. Forces the read to decode a neighbourhood of
    # each stored code rather than the exact trained points — the lever for
    # held-out-rule generalization (the read snaps to trained codes without it).
    # Scale is relative to the RMSNorm'd write (per-dim RMS ≈ 1). 0.0 = off.
    # NEGATIVE RESULT (2026-07-03, σ=0.1, K=2 interleaved): train 0.995 / held
    # 0.000 exact — noise teaches local robustness but actively REINFORCES
    # snapping (a midpoint code falls in a neighbour's cloud, where supervision
    # says "behave like the neighbour"). Kept as a regularizer knob only.
    mem_write_noise: float = 0.0
    # Code-space mixup: with probability p per presentation lane, replace the
    # stored rule code by the MIDPOINT of the EMA codes of the two neighbouring
    # rules (s-1, s+1) while keeping the labels of s — supervises the read at
    # points BETWEEN trained codes so interpolation is learned as a law of the
    # manifold. Only fires when s-1, s+1 (and s) are all trained shifts → no
    # held-out leakage. Requires mem_teacher_forcing plumbing (rule ids).
    mem_code_mixup_p: float = 0.0
    # EMA momentum for the per-shift code dictionary the mixup draws from.
    mem_code_mixup_momentum: float = 0.99
    # Step at which midpoint injection starts (the EMA dictionary builds from
    # step 0 regardless). Early injection feeds immature-EMA midpoints into 30%
    # of presentations and stalls base learning (v2 stalled at 0.14@600 with
    # start=0); inject only once the model writes real codes.
    mem_code_mixup_start: int = 0
    # STRUCTURAL smoothness for the read: spectral normalization on the fw_A /
    # fw_B hypernet maps (slot code → low-rank layer weights). Caps the read's
    # Lipschitz constant wrt the CODE, making razor decision boundaries between
    # neighbouring codes inexpressible — the lever against snapping that data
    # pressure (noise, mixup) could not provide.
    mem_read_spectral_norm: bool = False
    # Gated (SwiGLU-style) fast-weight read: z = clamp(silu(A_g·y) ⊙ (A_v·y)).
    # The code then gates the token stream MULTIPLICATIVELY (half-FiLM) instead
    # of only shaping an additive residual — the functional form "apply a rule"
    # wants. Clamped (DSv4 "SwiGLU clamping") to keep the product stable.
    mem_read_swiglu: bool = False
    # Which blocks read the bank as fast weights. Empty list = all blocks
    # (historical behaviour). Reading at every block composes code-dependent
    # transforms in depth, so code sensitivity escalates polynomially even
    # with SN per matrix; a single late read makes the slope cap global.
    mem_read_layers: list = field(default_factory=list)
    # Target-rate objective on the write gate: adds weight · (E[α] - target)² to the
    # loss. Unlike mem_write_cost (monotone, only pushes α→0), this has a stable
    # minimum at α=target, so it curbs BOTH α→1 (over-write, duplicate pollution)
    # and α→0 (never write) — the two ways the bank collapses. weight 0.0 = off.
    mem_write_target: float = 0.0        # target E[α] (e.g. 0.5); 0.0 disables via weight
    mem_write_target_weight: float = 0.0

    # ── Write gate ────────────────────────────────────────────────────────────
    # When False the write head is UNGATED: the appended slot is the pure normalised
    # thought (no scalar α, no per-dim content gate p). The gate otherwise attenuates
    # the written vector, diminishing the code it carries; ablating it removes a
    # coordination variable during bootstrap. Keep OFF while bootstrapping the
    # fast-weight transport, re-enable for streaming write selectivity.
    mem_write_gate: bool = True
    # Novelty gate (gate v2, no learned parameters): the written vector is scaled
    # by g = clamp(1 − max_j cos(m, slot_j), 0, 1) against the stop-grad bank —
    # near-duplicates (cos→1) are skimmed to zero, novel thoughts pass whole.
    # No scalar α, no per-dim p (their heads stay untrained). Takes precedence
    # over mem_write_gate. Differentiable through m, so the write head still
    # trains via TBPTT.
    mem_write_gate_novelty: bool = False
    # Delta write (gate v2b, no learned parameters): instead of scaling the whole
    # vector, SUBTRACT the component already stored — m' = m − max(⟨m, ŝ*⟩, 0)·ŝ*
    # with ŝ* the closest (cosine) bank slot, stop-grad. Exact duplicates cancel
    # to zero, partial copies keep their novel correction (delta-rule twin,
    # Schlag et al. 2021). Takes precedence over mem_write_gate_novelty.
    mem_write_gate_delta: bool = False
    # Dedup-refresh (gate v2c, GDN-2 lesson: decouple "is it new content?" from
    # "which slot pays for it"): the write is NEVER attenuated; when the new
    # thought's closest bank slot exceeds mem_write_merge_tau in cosine, the
    # write REPLACES that twin slot (recency refresh) instead of FIFO-evicting
    # the oldest distinct memory. Only active once the bank is at capacity.
    mem_write_gate_merge: bool = False
    mem_write_merge_tau: float = 0.85

    # ── Teacher-forced bank bootstrap (multiturn_rule) ────────────────────────
    # Breaks the "ignore-bank" fixed point: during bootstrap the read consumes a
    # CLEAN teacher code correlated with the latent rule id (a meta-training signal —
    # the id is latent at inference), while a distill loss pulls the WRITTEN slot
    # toward that teacher. The read's code is annealed teacher→written (β: 1→0).
    # Only wired for multiturn_rule (needs the ground-truth rule id per conversation).
    mem_teacher_forcing: bool = False
    # Fixed Fourier teacher codes instead of a learned embedding: code[s] =
    # [cos(2πks/S), sin(2πks/S)]_k (torus product for the affine family). The
    # circle geometry is IMPOSED — held rules sit literally between trained
    # codes — and "apply s" becomes composing a rotation (what grokking nets
    # discover by themselves on modular arithmetic; Nanda et al.). No teacher
    # optimizer: the codes are buffers.
    mem_teacher_fourier: bool = False
    # Cap Fourier frequencies at k ≤ kmax (cycled to fill mem_dim). Full-spectrum
    # codes ask the write head for high-frequency trig of s (k up to mem_dim/2 —
    # sign flips every few symbols): expensive for a 3M model. Low-k codes keep
    # the circle geometry (and interpolation) with a smooth, cheap s → code map.
    # 0 = full spectrum (k = 1..mem_dim/2).
    mem_teacher_fourier_kmax: int = 0
    # Distill on DIRECTION only: 1 − cos(w0, teacher[s]) instead of MSE. The MSE
    # against (near) zero-mean fixed targets has a rule-free descent path — shrink
    # ‖w0‖ toward 0 and collect MSE → 1.0 without encoding anything (dsv4f@1000:
    # RMS(w) 0.674, cos(w, target) −0.004, distill 1.46 = the constant-writer
    # value exactly). Cosine closes that loophole: only alignment pays.
    mem_teacher_distill_cosine: bool = False
    mem_teacher_anneal_start: int = 300      # β=1 until this optimiser step
    mem_teacher_anneal_end: int = 500        # β linear→0 by here, then 0 (teacher gone)
    # Adaptive anneal trigger — PULL-EARLIER ONLY. "ce_below" starts the anneal at
    # the FIXED window above, or earlier if the train-CE EMA drops under
    # ln(n_symbols) − margin first (teacher code demonstrably in use → something to
    # hand over; each β=1 step past that is dead time — distill alone never teaches
    # the write to identify s while the teacher hands the answer over). The fixed
    # window MUST stay as fallback: in s256L v2 CE sat at ln S through ALL of β=1
    # yet the anneal cracked at ~1000 — read organization is silent below CE, so a
    # CE gate can never be the only path to annealing. "" = fixed window only.
    mem_teacher_anneal_trigger: str = ""     # "" | "ce_below"
    mem_teacher_anneal_margin: float = 0.5   # trigger threshold: ln(n_symbols) − margin
    mem_teacher_anneal_len: int = 500        # anneal duration once triggered
    mem_teacher_distill_weight: float = 2.0  # weight on MSE(w0, teacher[s]); scaled by β

    # ── Factory helpers ───────────────────────────────────────────────────────
    @classmethod
    def tiny(cls) -> "ThoughtBankConfig":
        """~8M params – fast CPU/single-GPU experimentation."""
        return cls(
            d_model=128, n_layers=4, n_heads=2, d_head=32,
            csa_m=4, hca_m=16, top_k_csa=4, n_win=8,
            d_latent_q=32, n_groups=1, n_experts=4,
            top_k_experts=1, d_ff=256, sinkhorn_iters=3,
            mem_dim=32, max_mem=16, mem_seed_slots=4, mem_read_rank=16,
        )

    @classmethod
    def small(cls) -> "ThoughtBankConfig":
        """~50M params – single RTX 3090 training target."""
        return cls(
            d_model=256, n_layers=6, n_heads=4, d_head=64,
            csa_m=4, hca_m=32, top_k_csa=8, n_win=16,
            d_latent_q=64, n_groups=2, n_experts=8,
            top_k_experts=2, d_ff=512,
            mem_dim=64, max_mem=32, mem_seed_slots=4, mem_read_rank=16,
        )

    @classmethod
    def from_yaml(cls, path: Path) -> "ThoughtBankConfig":
        """Config au format PLAT (arc dsv4mini) : les clés modèle sont à la racine.

        Le filtrage silencieux des clés inconnues est volontaire — une config
        plate porte aussi ses clés d'entraînement à la racine, et `repro/run_all.sh`
        en dépend. Ce qui n'est PAS acceptable, c'est de rendre un config
        entièrement par défaut sans le dire : appelée sur une config IMBRIQUÉE
        (format actif, bloc `model:`), cette méthode ne matchait aucune clé et
        renvoyait les valeurs d'usine — un modèle silencieusement faux. Les deux
        cas se lèvent maintenant.
        """
        with open(path) as f:
            data = expand(yaml.safe_load(f)) or {}
        if "model" in data:
            raise ValueError(
                f"{path}: config imbriquée (bloc `model:`) — from_yaml lit le "
                f"format PLAT et rendrait un config tout par défaut. "
                f"Utiliser ThoughtBankConfig(**raw['model']).")
        fields = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        if not fields:
            raise ValueError(
                f"{path}: aucune clé modèle reconnue à la racine — from_yaml "
                f"rendrait un config tout par défaut.")
        return cls(**fields)


# Legacy alias (pre-rename scripts; the project is Thought Bank, the class
# carried the name of the DeepSeek-V4 architecture it borrows its trunk from)
DeepSeekV4MiniConfig = ThoughtBankConfig
