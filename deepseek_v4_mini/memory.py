"""
Fast-weight thought bank (single-stream).

The bank is a rolling FIFO buffer of M thought vectors, dim = mem_dim:

    bank : [B, M, mem_dim]     slot 0 = oldest surviving, slot M-1 = newest

It is READ as fast WEIGHTS — each slot parametrises a low-rank MLP layer that the
text stream is passed through (see DualModalBlock._cross_modal in model.py). There
is NO separate thought-stream transformer: the text model writes vectors and reuses
them directly. This module owns only the WRITE side.

Seeding
───────
A fresh bank is seeded with `mem_seed_slots` random-uniform[0,1] vectors (see
ThoughtStream.seed_bank). So the fast-weight "layers" are non-zero from the first
forward, and later writes append on top of that random scaffold.

Write as a modality choice (gated)
──────────────────────────────────
`_new_thought` produces, alongside a per-dim content gate, a scalar write-decision
α = sigmoid(write_decision(ctx)) ∈ [0,1] that multiplicatively scales the whole new
vector. α≈0 means "nothing worth committing this pass" (a near-zero slot, soon
evicted by FIFO); α≈1 means "commit this thought". Fully differentiable and driven
by the LM loss, so the model learns when to write.

Eviction (FIFO)
───────────────
Each pass appends one vector; once the bank exceeds max_mem the OLDEST slot is
dropped (`bank[:, -max_mem:]`). No learned consolidation — the bank's usefulness is
attributable to its content, not to an auxiliary compressor.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mhc import RMSNorm
from .config import ThoughtBankConfig


class ThoughtStream(nn.Module):
    """Write head for the fast-weight thought bank.

    Public surface used elsewhere:
      seed_bank(B, device, dtype)        → [B, mem_seed_slots, mem_dim] random[0,1]
      _new_thought(H_text, bank, pad)    → [B, 1, mem_dim] gated new thought
      _write(H_text, bank, pad)          → [B, M', mem_dim] bank with the thought appended
    """

    def __init__(self, cfg: ThoughtBankConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d = cfg.mem_dim

        # Write head: attention-pooled text context → per-dim gate + thought vector.
        # write_ctx_q scores each text token for the pool; write_gate is per-dim so
        # each feature can be written independently (LSTM-input-gate analogue).
        self.write_ctx_q  = nn.Linear(cfg.d_model, 1, bias=False)
        self.write_gate   = nn.Linear(cfg.d_model, d, bias=True)
        self.thought_head = nn.Linear(cfg.d_model, d, bias=False)
        self.norm_write   = RMSNorm(d)

        # Write-decision head: scalar α = sigmoid(.) ∈ [0,1] scaling the whole new
        # vector — the "write or skip" choice. bias=0 → α≈0.5 at init (neutral).
        self.write_decision = nn.Linear(cfg.d_model, 1, bias=True)

        # ── Telemetry / differentiable regularisers (read by train.py) ──────────
        # Batch-mean write probability α (detached), for logging write/skip rate.
        self.last_write_alpha: Optional[torch.Tensor] = None
        # Differentiable write budget E[-log(1-α)] = E[softplus(z)]; weighted by
        # mem_write_cost. Budget form keeps a live gradient even when α≈1.
        self.last_write_penalty: Optional[torch.Tensor] = None
        # Differentiable E[α] for the target-rate objective (mem_write_target_weight).
        self.last_write_alpha_mean: Optional[torch.Tensor] = None
        # Differentiable novelty: E[max_j cos(m_new, slot_j)] vs the stop-grad bank;
        # weighted by mem_write_diversity to push writes away from stored duplicates.
        self.last_write_redundancy: Optional[torch.Tensor] = None

    # ── Bank seeding ───────────────────────────────────────────────────────────

    def seed_bank(
        self, batch: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """Return a fresh bank of `mem_seed_slots` vectors (init per cfg.mem_seed_init)."""
        n = max(1, int(self.cfg.mem_seed_slots))
        init = getattr(self.cfg, "mem_seed_init", "uniform01")
        if init == "zeros":
            return torch.zeros(batch, n, self.cfg.mem_dim, device=device, dtype=dtype)
        if init == "pm1":
            return torch.rand(batch, n, self.cfg.mem_dim, device=device, dtype=dtype) * 2.0 - 1.0
        if init != "uniform01":
            raise ValueError(f"mem_seed_init inconnu: {init!r}")
        return torch.rand(batch, n, self.cfg.mem_dim, device=device, dtype=dtype)

    # ── Write ──────────────────────────────────────────────────────────────────

    def _write(
        self,
        H_text: torch.Tensor,            # [B, T, d_model]
        mem_bank: torch.Tensor,          # [B, M, mem_dim]
        pad_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Append a new gated thought vector and FIFO-evict the oldest slot."""
        m_new    = self._new_thought(H_text, mem_bank, pad_mask)   # [B, 1, mem_dim]
        if self.training and self.cfg.mem_write_noise > 0.0:
            m_new = m_new + self.cfg.mem_write_noise * torch.randn_like(m_new)
        if (getattr(self.cfg, "mem_write_gate_merge", False)
                and mem_bank.size(1) >= self.cfg.max_mem):
            # Gate v2c (dedup-refresh): a near-duplicate write replaces its twin
            # slot instead of costing the oldest distinct memory its FIFO slot.
            # Novel writes keep plain FIFO semantics (drop slot 0). Write content
            # is never attenuated — the read sees full-norm codes either way.
            B, M, D = mem_bank.shape
            mn  = F.normalize(m_new.squeeze(1), dim=-1)
            bn  = F.normalize(mem_bank.detach().float(), dim=-1).to(m_new.dtype)
            cos = torch.einsum("bd,bmd->bm", mn, bn)               # [B, M]
            max_cos, twin = cos.max(dim=1)
            merge = max_cos > float(getattr(self.cfg, "mem_write_merge_tau", 0.85))
            drop  = torch.where(merge, twin, torch.zeros_like(twin))   # else oldest
            self.last_merge_rate = merge.float().mean().detach()
            keep = torch.arange(M, device=mem_bank.device).unsqueeze(0) != drop.unsqueeze(1)
            kept = mem_bank[keep].view(B, M - 1, D)                # order preserved
            return torch.cat([kept, m_new], dim=1)
        mem_bank = torch.cat([mem_bank, m_new], dim=1)
        if mem_bank.size(1) > self.cfg.max_mem:
            mem_bank = mem_bank[:, -self.cfg.max_mem:, :]
        return mem_bank

    def _new_thought(
        self,
        H_text: torch.Tensor,
        bank: Optional[torch.Tensor] = None,
        pad_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Produce one new thought vector [B, 1, mem_dim] from the current text.

        Context is a soft attention pool over positions (pad-masked). The content
        gate (per-dim sigmoid) and the write-decision α (scalar sigmoid) modulate
        the thought; both are learned end-to-end via the LM loss.
        """
        # Attention-pooled summary over real (non-pad) positions.
        scores = self.write_ctx_q(H_text).squeeze(-1)              # [B, T]
        if pad_mask is not None:
            m    = pad_mask.bool()
            safe = m | (~m.any(dim=1, keepdim=True))               # keep all-pad rows finite
            scores = scores.masked_fill(~safe, float("-inf"))
        weights = torch.softmax(scores, dim=-1)                    # [B, T]
        h_ctx   = (weights.unsqueeze(-1) * H_text).sum(dim=1)      # [B, d_model]

        p     = torch.sigmoid(self.write_gate(h_ctx))             # [B, mem_dim] content gate
        m     = self.norm_write(self.thought_head(h_ctx))         # [B, mem_dim] thought
        z     = self.write_decision(h_ctx)                        # [B, 1] decision logit
        alpha = torch.sigmoid(z)                                  # [B, 1] write/skip

        self.last_write_alpha      = alpha.detach().mean()
        self.last_write_penalty    = F.softplus(z).mean()         # E[-log(1-α)]
        self.last_write_alpha_mean = alpha.mean()                 # E[α]
        if bank is not None and bank.size(1) > 0:
            mn  = F.normalize(m, dim=-1)                          # [B, mem_dim]
            bn  = F.normalize(bank.detach().float(), dim=-1).to(m.dtype)  # [B, M, mem_dim]
            cos = torch.einsum("bd,bmd->bm", mn, bn)             # [B, M]
            max_cos, nn_idx = cos.max(dim=1)                      # [B] closest neighbour
            self.last_write_redundancy = max_cos.mean()
        else:
            max_cos = nn_idx = None
            self.last_write_redundancy = torch.zeros((), device=H_text.device)

        if getattr(self.cfg, "mem_write_gate_delta", False):
            # Gate v2b: delta write — subtract the stored component (projection on
            # the closest slot, positive part only) so only the NOVEL part is
            # written. Duplicates cancel; partial copies keep their correction.
            if nn_idx is None:
                return m.unsqueeze(1)
            s_hat = bn.gather(1, nn_idx.view(-1, 1, 1).expand(-1, 1, bn.size(-1))).squeeze(1)  # [B, mem_dim]
            coef  = torch.einsum("bd,bd->b", m, s_hat).clamp_min(0.0).unsqueeze(-1)  # [B, 1]
            m_new = m - coef * s_hat
            # telemetry: write_alpha = E[|m'|/|m|] (fraction of the write that survives)
            ratio = m_new.norm(dim=-1) / m.norm(dim=-1).clamp_min(1e-8)
            self.last_write_alpha      = ratio.detach().mean()
            self.last_write_alpha_mean = ratio.mean()
            return m_new.unsqueeze(1)

        if getattr(self.cfg, "mem_write_gate_novelty", False):
            # Gate v2: parameter-free novelty gate — duplicates are skimmed to
            # zero, novel writes pass whole. Differentiable through m only (the
            # bank is stop-grad, so the model can't game the gate by trashing
            # stored slots). α/p heads above stay unused.
            if max_cos is None:
                g = torch.ones_like(m[:, :1])                     # empty bank: all novel
            else:
                g = (1.0 - max_cos).clamp(0.0, 1.0).unsqueeze(-1) # [B, 1]
            self.last_write_alpha      = g.detach().mean()        # telemetry: write_alpha = E[g]
            self.last_write_alpha_mean = g.mean()
            return (g * m).unsqueeze(1)                           # [B, 1, mem_dim]

        if not self.cfg.mem_write_gate:
            return m.unsqueeze(1)               # ungated: pure normalised thought
        return (alpha * p * m).unsqueeze(1)                       # [B, 1, mem_dim]


# ── self-test ────────────────────────────────────────────────────────────────

def _selftest() -> None:
    """Invariants du write : FIFO, seeding, précédence des gates, pad-safety.

    Tout ce qui est testé ici a déjà coûté du temps de debug au moins une fois.
    CPU, quelques dizaines de ms.
    """
    torch.manual_seed(0)
    D, M, B, T, DM = 8, 4, 2, 6, 16

    def _cfg(**kw):
        return ThoughtBankConfig(d_model=DM, mem_dim=D, max_mem=M,
                                 mem_seed_slots=2, **kw)

    def _stream(**kw):
        torch.manual_seed(1234)                 # mêmes poids d'une variante à l'autre
        return ThoughtStream(_cfg(**kw))

    H = torch.randn(B, T, DM)

    # ── seeding ──────────────────────────────────────────────────────────────
    ts = _stream()
    for init, check in (("uniform01", lambda x: (x >= 0).all() and (x <= 1).all()),
                        ("pm1", lambda x: (x >= -1).all() and (x <= 1).all()),
                        ("zeros", lambda x: (x == 0).all())):
        ts.cfg.mem_seed_init = init
        bk = ts.seed_bank(B, torch.device("cpu"), torch.float32)
        assert bk.shape == (B, 2, D), (init, bk.shape)
        assert check(bk), init
    ts.cfg.mem_seed_init = "jamais-vu"
    try:
        ts.seed_bank(B, torch.device("cpu"), torch.float32)
    except ValueError:
        pass
    else:
        raise AssertionError("mem_seed_init inconnu : aurait dû lever")
    ts.cfg.mem_seed_init = "uniform01"
    # mem_seed_slots=0 ⇒ au moins un slot (une banque vide casserait le read)
    ts.cfg.mem_seed_slots = 0
    assert ts.seed_bank(B, torch.device("cpu"), torch.float32).size(1) == 1
    ts.cfg.mem_seed_slots = 2

    # ── FIFO : croissance puis éviction du DOYEN, ordre préservé ─────────────
    ts = _stream(mem_write_gate=False)
    bank = torch.randn(B, M - 1, D)
    grown = ts._write(H, bank)
    assert grown.shape == (B, M, D), grown.shape
    assert torch.equal(grown[:, :M - 1], bank), "croissance : l'existant a bougé"

    full = torch.randn(B, M, D)
    out = ts._write(H, full)
    assert out.shape == (B, M, D), out.shape
    assert torch.equal(out[:, :M - 1], full[:, 1:]), \
        "FIFO : le doyen doit partir et l'ordre rester"
    # au-delà de max_mem la banque ne grossit jamais, même en écrivant longtemps
    b = torch.randn(B, M, D)
    for _ in range(3 * M):
        b = ts._write(H, b)
        assert b.size(1) == M

    # ── le write est bien de dimension mem_dim et fini ───────────────────────
    m = ts._new_thought(H, None)
    assert m.shape == (B, 1, D) and torch.isfinite(m).all()

    # ── ligne entièrement pad : le pool ne doit pas produire de NaN ──────────
    pad = torch.ones(B, T, dtype=torch.bool)
    pad[1] = False                                    # ligne 1 : que du padding
    m_pad = ts._new_thought(H, None, pad)
    assert torch.isfinite(m_pad).all(), "ligne all-pad : softmax(-inf partout) = NaN"

    # ── gates : chacun sur son comportement propre ───────────────────────────
    # Référence : le thought BRUT, gates coupés (les gates s'appliquent après).
    raw = _stream(mem_write_gate=False)._new_thought(H, None)      # [B,1,D]
    dup = raw.expand(B, M, D).contiguous()            # banque = M copies exactes

    nov = _stream(mem_write_gate_novelty=True)._new_thought(H, dup)
    assert nov.norm() < 1e-5, f"novelty : un doublon exact doit être écrémé ({nov.norm():.3e})"

    dlt = _stream(mem_write_gate_delta=True)._new_thought(H, dup)
    assert dlt.norm() < 1e-5, f"delta : un doublon exact doit s'annuler ({dlt.norm():.3e})"

    # …et laissent passer une écriture NOUVELLE (banque orthogonale au write)
    orth = torch.zeros(B, M, D)
    orth[:, :, 0] = 1.0
    r2 = _stream(mem_write_gate=False)._new_thought(H, orth)
    for flag in ("mem_write_gate_novelty", "mem_write_gate_delta"):
        got = _stream(**{flag: True})._new_thought(H, orth)
        assert got.norm() > 0.5 * r2.norm(), f"{flag} : une écriture neuve est passée à la trappe"

    # merge (v2c) : le doublon REMPLACE son jumeau — le doyen DISTINCT survit
    ts_m = _stream(mem_write_gate=False, mem_write_gate_merge=True,
                   mem_write_merge_tau=0.5)
    old = torch.randn(B, M, D)
    old[:, 2] = raw.squeeze(1)                        # le jumeau est en position 2
    got = ts_m._write(H, old)
    assert got.size(1) == M
    assert torch.equal(got[:, 0], old[:, 0]), \
        "merge : le doyen distinct ne doit PAS payer pour un doublon"
    assert float(ts_m.last_merge_rate) == 1.0

    # précédence documentée : merge > delta > novelty > gate scalaire
    both = _stream(mem_write_gate_delta=True, mem_write_gate_novelty=True)
    only_delta = _stream(mem_write_gate_delta=True)
    half = torch.zeros(B, M, D)
    half[:, 0] = raw.squeeze(1) + torch.randn(B, D) * 0.5     # voisin partiel
    assert torch.allclose(both._new_thought(H, half),
                          only_delta._new_thought(H, half)), \
        "précédence : delta doit primer sur novelty"

    # ── télémétrie : les champs que le trainer logue sont peuplés ────────────
    ts = _stream()
    _ = ts._new_thought(H, torch.randn(B, M, D))
    for name in ("last_write_alpha", "last_write_penalty", "last_write_alpha_mean",
                 "last_write_redundancy"):
        v = getattr(ts, name)
        assert v is not None and torch.isfinite(v).all(), name
    # banque vide : la redondance est définie (0), pas None
    _ = ts._new_thought(H, None)
    assert float(ts.last_write_redundancy) == 0.0

    # ── le gradient atteint la tête de write (sinon elle n'apprend jamais) ───
    ts = _stream()
    H2 = torch.randn(B, T, DM, requires_grad=True)
    ts._write(H2, torch.randn(B, M, D)).sum().backward()
    assert ts.thought_head.weight.grad is not None
    assert ts.thought_head.weight.grad.abs().sum() > 0, "write head sans gradient"

    print("memory self-test: OK (FIFO + ordre, 3 seedings + init inconnu, "
          "gates novelty/delta/merge + précédence, ligne all-pad finie, "
          "télémétrie, gradient du write)")


if __name__ == "__main__":
    _selftest()
