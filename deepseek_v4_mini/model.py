"""
Two model variants:

TrunkLM  (single-stream, legacy)
  Text stream only. Optional bolt-on thought memory (cross-attention read).

ThoughtBankLM  (fast-weight thought bank, recommended)
  Text stream [B, T, d_model] processed by CSA/HCA blocks with mHC. A rolling
  thought bank [B, M, mem_dim] is READ as FAST WEIGHTS at every text block: each
  slot parametrises a low-rank MLP layer and the text stream is passed through the
  stack of them (slot → linear → activation → dropout → next slot). After the text
  blocks a gated write head appends one new thought vector to the bank (FIFO). The
  bank is single-stream: there is no separate thought transformer — the text model
  writes the vectors and reuses them directly as weights (continual-learning goal).
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ThoughtBankConfig
from .mhc import ManifoldHyperConnections, RMSNorm
from .attention import CompressedSparseAttention, HeavilyCompressedAttention
from .moe import DeepSeekMoE
from .memory import ThoughtStream


# ── Legacy bolt-on thought-memory components (TrunkLM only) ────────────

class _MemoryCrossAttention(nn.Module):
    """Read from a thought-vector memory bank via cross-attention."""

    def __init__(self, dim: int, mem_dim: int, n_heads: int) -> None:
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads  = n_heads
        self.head_dim = dim // n_heads
        self.q  = nn.Linear(dim, dim, bias=False)
        self.k  = nn.Linear(mem_dim, dim, bias=False)
        self.v  = nn.Linear(mem_dim, dim, bias=False)
        self.o  = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor, mem: Optional[torch.Tensor]) -> torch.Tensor:
        if mem is None or mem.size(1) == 0:
            return x
        B, T, H = x.shape
        M = mem.size(1)
        scale = self.head_dim ** -0.5
        q = self.q(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k(mem).view(B, M, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v(mem).view(B, M, self.n_heads, self.head_dim).transpose(1, 2)
        w = F.softmax((q @ k.transpose(-2, -1)) * scale, dim=-1)
        z = (w @ v).transpose(1, 2).contiguous().view(B, T, H)
        return x + self.o(z)


class _WriteGate(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.lin = nn.Linear(dim, 1, bias=True)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.lin(h))   # [B, 1]


class _ThoughtHead(nn.Module):
    def __init__(self, dim: int, mem_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(dim, mem_dim, bias=False)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.proj(h)                 # [B, mem_dim]


# ── Single transformer block (legacy) ─────────────────────────────────────────

class TrunkBlock(nn.Module):
    """One transformer block: two mHC-wrapped sub-layers (attention + MoE).
    Even layer_idx → CSA; odd → HCA."""

    def __init__(self, cfg: ThoughtBankConfig, layer_idx: int) -> None:
        super().__init__()
        if layer_idx % 2 == 0:
            attn: nn.Module = CompressedSparseAttention(
                d_model=cfg.d_model, n_heads=cfg.n_heads, d_head=cfg.d_head,
                csa_m=cfg.csa_m, top_k=cfg.top_k_csa, n_win=cfg.n_win,
                d_latent_q=cfg.d_latent_q, n_groups=cfg.n_groups,
                dropout=cfg.dropout,
            )
        else:
            attn = HeavilyCompressedAttention(
                d_model=cfg.d_model, n_heads=cfg.n_heads, d_head=cfg.d_head,
                hca_m=cfg.hca_m, n_win=cfg.n_win,
                d_latent_q=cfg.d_latent_q, n_groups=cfg.n_groups,
                dropout=cfg.dropout,
            )

        moe = DeepSeekMoE(
            d_model=cfg.d_model, n_experts=cfg.n_experts,
            n_shared=cfg.n_shared, top_k_experts=cfg.top_k_experts,
            d_ff=cfg.d_ff, dropout=cfg.dropout,
        )

        self.norm_attn = RMSNorm(cfg.d_model)
        self.norm_moe  = RMSNorm(cfg.d_model)
        _cf = bool(getattr(cfg, "sinkhorn_closed_form", False))
        self.mhc_attn = ManifoldHyperConnections(cfg.d_model, cfg.n_hc, cfg.sinkhorn_iters, _cf)
        self.mhc_moe  = ManifoldHyperConnections(cfg.d_model, cfg.n_hc, cfg.sinkhorn_iters, _cf)
        self._attn = attn
        self._moe  = moe

    def forward(self, X: torch.Tensor):
        """X: [B, T, n_hc, d_model] → (X_new, balance_loss)"""
        X = self.mhc_attn(X, lambda h: self._attn(self.norm_attn(h)))

        bal = torch.zeros((), device=X.device)

        def _moe_fn(h: torch.Tensor):
            nonlocal bal
            out, bl = self._moe(self.norm_moe(h))
            bal = bl
            return out

        X = self.mhc_moe(X, _moe_fn)
        return X, bal


# ── Full model (legacy) ───────────────────────────────────────────────────────

class TrunkLM(nn.Module):
    """Small DeepSeek-V4 reproduction with optional bolt-on thought memory."""

    def __init__(self, cfg: ThoughtBankConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.drop  = nn.Dropout(cfg.dropout)

        self.blocks = nn.ModuleList(
            [TrunkBlock(cfg, i) for i in range(cfg.n_layers)]
        )

        self.A_out_net = nn.Linear(cfg.d_model, cfg.n_hc, bias=False)
        self.norm_out  = RMSNorm(cfg.d_model)

        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight   # weight tying
        nn.init.normal_(self.embed.weight, mean=0.0, std=0.02)

        if cfg.use_thought_memory:
            self.mem_attn  = _MemoryCrossAttention(cfg.d_model, cfg.mem_dim, cfg.n_heads)
            self.gate      = _WriteGate(cfg.d_model)
            self.thought   = _ThoughtHead(cfg.d_model, cfg.mem_dim)
            self.norm_mem  = RMSNorm(cfg.d_model)

    def forward(
        self,
        input_ids: torch.LongTensor,
        init_mem: Optional[torch.Tensor] = None,
        compute_logits: bool = True,
    ) -> dict:
        B, T = input_ids.shape
        cfg  = self.cfg

        h = self.drop(self.embed(input_ids))              # [B, T, d]
        X = h.unsqueeze(2).expand(-1, -1, cfg.n_hc, -1).contiguous()

        total_balance = torch.zeros((), device=X.device)
        for block in self.blocks:
            X, bal = block(X)
            total_balance = total_balance + bal
        total_balance = total_balance / cfg.n_layers

        X_mean = X.mean(dim=2)
        A = torch.softmax(self.A_out_net(X_mean), dim=-1)
        H = (A.unsqueeze(-1) * X).sum(dim=2)
        H = self.norm_out(H)

        p_gates  = None
        mem_bank = init_mem
        if cfg.use_thought_memory:
            h_aug_list, p_list = [], []
            for t in range(T):
                h_t   = H[:, t:t+1, :]
                h_aug = self.mem_attn(h_t, mem_bank)
                h_aug_list.append(h_aug)
                p_t   = self.gate(h_aug.squeeze(1))
                m_t   = self.thought(h_aug.squeeze(1))
                p_list.append(p_t)
                m_write = (p_t * m_t).unsqueeze(1)
                mem_bank = m_write if mem_bank is None else torch.cat([mem_bank, m_write], dim=1)
                if mem_bank.size(1) > cfg.max_mem:
                    mem_bank = mem_bank[:, -cfg.max_mem:]
            H_final = torch.cat(h_aug_list, dim=1)
            p_gates = torch.cat(p_list, dim=1).squeeze(-1)
        else:
            H_final = H

        out = {"balance_loss": total_balance, "mem_bank": mem_bank, "p_gates": p_gates}
        if compute_logits:
            out["logits"] = self.lm_head(H_final)
        else:
            out["hidden"]         = H_final
            out["lm_head_weight"] = self.lm_head.weight
        return out

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @classmethod
    def from_config(cls, cfg: ThoughtBankConfig) -> "TrunkLM":
        return cls(cfg)


# ── Fast-weight text block ────────────────────────────────────────────────────

class DualModalBlock(nn.Module):
    """
    Text transformer block that reads the thought bank as FAST WEIGHTS.

    Forward:
      1. mHC(CSA or HCA)      – text self-attention
      2. Fast-weight read     – the token stream is passed through a stack of
                                low-rank MLP layers, one per bank slot (see
                                _cross_modal). Placed between attention and MoE so
                                the applied "weights" can influence FFN routing.
      3. mHC(MoE)             – text FFN

    Fast-weight read (memory-as-weights, not memory-as-data)
    ────────────────────────────────────────────────────────
    Each bank slot m_i ∈ R^mem_dim is expanded by a (frozen, learned) hypernet into
    a low-rank layer with weights A_i ∈ R^{r×d}, B_i ∈ R^{d×r}:

        y ← y + dropout( B_i · act( A_i · y ) )          (residual, per slot i)

    applied SEQUENTIALLY over the M slots → an M-layer fast-weight MLP whose weights
    the model wrote. The activation between slots is what makes the composition
    non-linear: without it, stacking/summing slots collapses to one low-rank linear
    map (the failure mode of the earlier outer-product read). The read's net effect
    is a delta added to the token: h + fw_o(y - y0), so a trivial bank ≈ identity.
    """

    def __init__(self, cfg: ThoughtBankConfig, layer_idx: int) -> None:
        super().__init__()
        d    = cfg.d_model
        n_hc = cfg.n_hc

        _fuse = bool(getattr(cfg, "decode_fuse", False))
        if layer_idx % 2 == 0:
            attn: nn.Module = CompressedSparseAttention(
                d_model=d, n_heads=cfg.n_heads, d_head=cfg.d_head,
                csa_m=cfg.csa_m, top_k=cfg.top_k_csa, n_win=cfg.n_win,
                d_latent_q=cfg.d_latent_q, n_groups=cfg.n_groups,
                dropout=cfg.dropout, fuse_rope=_fuse,
            )
        else:
            attn = HeavilyCompressedAttention(
                d_model=d, n_heads=cfg.n_heads, d_head=cfg.d_head,
                hca_m=cfg.hca_m, n_win=cfg.n_win,
                d_latent_q=cfg.d_latent_q, n_groups=cfg.n_groups,
                dropout=cfg.dropout, fuse_rope=_fuse,
            )

        moe = DeepSeekMoE(
            d_model=d, n_experts=cfg.n_experts, n_shared=cfg.n_shared,
            top_k_experts=cfg.top_k_experts, d_ff=cfg.d_ff, dropout=cfg.dropout,
        )

        self.norm_attn = RMSNorm(d)
        self.norm_moe  = RMSNorm(d)
        _cf = bool(getattr(cfg, "sinkhorn_closed_form", False))
        self.mhc_attn  = ManifoldHyperConnections(d, n_hc, cfg.sinkhorn_iters, _cf)
        self.mhc_moe   = ManifoldHyperConnections(d, n_hc, cfg.sinkhorn_iters, _cf)
        self._attn = attn
        self._moe  = moe

        # ── Fast-weight read: slot → low-rank MLP layer (A_i [r,d], B_i [d,r]) ──
        # mem_read_layers restricts which blocks read the bank (empty = all).
        # Reading at every block composes code-dependent transforms in depth —
        # code sensitivity escalates even with SN per matrix.
        _rl = list(getattr(cfg, "mem_read_layers", []) or [])
        self.read_bank = (not _rl) or (layer_idx in _rl)
        r = int(cfg.mem_read_rank)
        self.read_rank = r
        # SwiGLU read: fw_A emits TWO maps per slot (gate A_g, value A_v).
        self.fw_swiglu = bool(getattr(cfg, "mem_read_swiglu", False))
        _na = 2 if self.fw_swiglu else 1
        self.fw_A    = nn.Linear(cfg.mem_dim, _na * r * d, bias=False)  # slot → A_i
        self.fw_B    = nn.Linear(cfg.mem_dim, d * r, bias=False)  # slot → B_i
        if bool(getattr(cfg, "mem_read_spectral_norm", False)):
            from torch.nn.utils.parametrizations import spectral_norm
            self.fw_A = spectral_norm(self.fw_A)
            self.fw_B = spectral_norm(self.fw_B)
        self.fw_o    = nn.Linear(d, d, bias=False)               # output projection
        self.norm_fw = RMSNorm(d)
        self.fw_act  = nn.GELU()
        self.fw_drop = nn.Dropout(cfg.mem_read_dropout)

        # Dynamic mHC collapse for the read: same principle as A_out_net.
        self.A_cross_net = nn.Linear(d, n_hc, bias=False)

        # decode_fuse : mémo (banque → A, Bm), voir _fw_maps.
        self.decode_fuse = bool(getattr(cfg, "decode_fuse", False))
        self._fw_memo = None

    def _fw_maps(self, bank: torch.Tensor):
        """Projette la banque en poids fast-weight (A, Bm) — voir _cross_modal.

        A et Bm ne dépendent QUE de la banque, et la banque est constante entre
        deux writes (invariant du projet : sa seule modification est l'append
        d'un write au `<think>`, un torch.cat ⇒ NOUVEAU tenseur). Le décodage
        les reprojetait pourtant à chaque token — 2 Linear × 12 couches/token
        pour un résultat identique. Sous decode_fuse et no-grad, mémo par
        IDENTITÉ d'objet (`is`, pas data_ptr : l'allocateur réutilise les
        adresses) + compteur d'in-place `_version` (une mutation en place
        garderait l'identité mais changerait le contenu — jamais le cas dans le
        dépôt, la garde coûte une lecture d'attribut). Un write ⇒ nouvel objet
        ⇒ miss ⇒ recalcul ; le bras ablaté (seed_bank tire un tenseur NEUF par
        token) rate le `is` à chaque fois ⇒ comportement inchangé.
        """
        memo_ok = self.decode_fuse and not torch.is_grad_enabled()
        if memo_ok and self._fw_memo is not None:
            mb, mv, A, Bm = self._fw_memo
            if mb is bank and mv == bank._version:
                return A, Bm
        B, M, _ = bank.shape
        r = self.read_rank
        _na = 2 if self.fw_swiglu else 1
        d = self.fw_B.out_features // r
        A = self.fw_A(bank).view(B, M, _na, r, d)   # [B, M, 1|2, r, d]
        Bm = self.fw_B(bank).view(B, M, d, r)       # [B, M, d, r]
        # sous gradient le mémo est purgé : l'entraînement DOIT reprojeter à
        # chaque forward pour que le gradient traverse fw_A/fw_B vers la banque
        self._fw_memo = (bank, bank._version, A, Bm) if memo_ok else None
        return A, Bm

    def _cross_modal(self, h: torch.Tensor, bank: torch.Tensor) -> torch.Tensor:
        """
        h    : [B, T, d]         – current text representations
        bank : [B, M, mem_dim]   – thought bank (fast-weight codes)
        Returns [B, T, d] with the fast-weight read added as a residual.
        """
        B, M, _ = bank.shape
        d = h.size(-1)
        r = self.read_rank

        A, Bm = self._fw_maps(bank)                # [B,M,1|2,r,d], [B,M,d,r]

        ds = d ** -0.5
        rs = r ** -0.5
        y0 = self.norm_fw(h)
        y  = y0
        for i in range(M):
            if self.fw_swiglu:
                zg = torch.einsum("brd,btd->btr", A[:, i, 0], y) * ds
                zv = torch.einsum("brd,btd->btr", A[:, i, 1], y) * ds
                z  = (F.silu(zg) * zv).clamp(-8.0, 8.0)                       # gated, clamped
            else:
                z = self.fw_act(torch.einsum("brd,btd->btr", A[:, i, 0], y) * ds)  # [B,T,r]
            upd = torch.einsum("bdr,btr->btd", Bm[:, i], z) * rs              # [B,T,d]
            y   = y + self.fw_drop(upd)
        return h + self.fw_o(y - y0)

    def forward(self, X: torch.Tensor, bank: Optional[torch.Tensor], cache=None):
        """
        X     : [B, T, n_hc, d_model]
        bank  : [B, M, mem_dim] or None
        cache : AttnCache de CETTE couche (décodage incrémental) ou None
        Returns (X_new, balance_loss)
        """
        # 1. Text self-attention (mHC wrapped). Seule l'attention a besoin du
        #    passé : mHC, MoE et le read fast-weight sont positionwise, donc en
        #    décodage un token ne traverse le stack qu'une fois.
        X = self.mhc_attn(X, (lambda h: self._attn(self.norm_attn(h)))
                          if cache is None else
                          (lambda h: self._attn.forward_cached(self.norm_attn(h), cache)))

        # 2. Fast-weight read (thought bank → text)
        if self.read_bank and bank is not None and bank.size(1) > 0:
            h0 = (torch.softmax(self.A_cross_net(X.mean(dim=2)), dim=-1).unsqueeze(-1) * X).sum(dim=2)
            h1 = self._cross_modal(h0, bank)          # _cross_modal normalises internally
            delta = h1 - h0
            X = X + delta.unsqueeze(2)

        # 3. MoE (mHC wrapped)
        bal = torch.zeros((), device=X.device)

        def _moe_fn(h: torch.Tensor) -> torch.Tensor:
            nonlocal bal
            out, bl = self._moe(self.norm_moe(h))
            bal = bl
            return out

        X = self.mhc_moe(X, _moe_fn)
        return X, bal


# ── Fast-weight dual-stream model ─────────────────────────────────────────────

class ThoughtBankLM(nn.Module):
    """
    Text stream + fast-weight thought bank (single-stream bank).

    Forward flow
    ────────────
    1. Seed the bank with random-uniform[0,1] slots on a fresh conversation
       (init_mem is None); otherwise reuse the carried-in bank.
    2. Text blocks process the input; at each block the bank is READ as fast
       weights (DualModalBlock._cross_modal).
    3. After the text blocks, the write head appends one new gated thought vector
       to the bank (FIFO-evicting the oldest beyond max_mem).
    4. LM head produces logits from the final text hidden states.

    The updated bank is returned as `mem_bank`; pass it back as `init_mem` on the
    next turn for multi-turn / streaming continual learning.
    """

    def __init__(self, cfg: ThoughtBankConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.drop  = nn.Dropout(cfg.dropout)

        self.blocks = nn.ModuleList(
            [DualModalBlock(cfg, i) for i in range(cfg.n_layers)]
        )

        self.A_out_net = nn.Linear(cfg.d_model, cfg.n_hc, bias=False)
        self.norm_out  = RMSNorm(cfg.d_model)

        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight
        nn.init.normal_(self.embed.weight, mean=0.0, std=0.02)

        # Write head for the fast-weight bank.
        self.thought_stream = ThoughtStream(cfg)

    def forward(
        self,
        input_ids: torch.LongTensor,
        init_mem: Optional[torch.Tensor] = None,
        compute_logits: bool = True,
        pad_mask: Optional[torch.Tensor] = None,
        layer_banks: Optional[list] = None,
        write: bool = True,
        cache: Optional[list] = None,
    ) -> dict:
        B, T  = input_ids.shape
        cfg   = self.cfg
        n_hc  = cfg.n_hc

        # ── Step 1: embed text ────────────────────────────────────────────────
        h = self.drop(self.embed(input_ids))                              # [B,T,d]

        # ── Step 2: seed a fresh bank, or reuse the carried-in one ────────────
        if init_mem is None or init_mem.size(1) == 0:
            bank = self.thought_stream.seed_bank(B, h.device, h.dtype)
        else:
            bank = init_mem

        # ── Step 3: text blocks read the bank as fast weights ─────────────────
        X = h.unsqueeze(2).expand(-1, -1, n_hc, -1).contiguous()         # [B,T,n_hc,d]
        total_bal = torch.zeros((), device=X.device)
        # Cascade v3: layer_banks[i] = banque lue par la couche i (None = pas de
        # read). La banque VIVE (init_mem) reste seule sur le chemin d'écriture.
        for li, block in enumerate(self.blocks):
            X, bal = block(X, bank if layer_banks is None else layer_banks[li],
                           cache=None if cache is None else cache[li])
            total_bal = total_bal + bal
        total_bal = total_bal / cfg.n_layers

        # ── Step 4: dynamic collapse mHC → text hidden states ─────────────────
        X_mean = X.mean(dim=2)                                           # [B,T,d]
        A = torch.softmax(self.A_out_net(X_mean), dim=-1)               # [B,T,n_hc]
        H_text = (A.unsqueeze(-1) * X).sum(dim=2)                       # [B,T,d]
        H_text = self.norm_out(H_text)

        # ── Step 5: gated thought write + FIFO eviction ───────────────────────
        # write=False : le décodage jette de toute façon la banque écrite (il ne
        # garde que les logits), et le write est un pool attentionnel sur TOUTE
        # la séquence — donc le seul morceau du forward qui ne soit pas
        # cachable de manière incrémentale. Ne pas le calculer économise ce
        # pool à chaque token généré, et la banque ressort telle quelle.
        if write:
            mem_bank = self.thought_stream._write(H_text, bank, pad_mask)
        else:
            mem_bank = bank

        # ── Step 6: outputs ───────────────────────────────────────────────────
        # sans write, la télémétrie est explicitement None : rendre celle du
        # forward PRÉCÉDENT ferait passer une valeur périmée pour une mesure.
        ts = self.thought_stream
        out = {
            "balance_loss":     total_bal,
            "mem_bank":         mem_bank,
            "write_alpha":      ts.last_write_alpha if write else None,       # mean α (telemetry)
            "write_penalty":    ts.last_write_penalty if write else None,     # diff E[-log(1-α)]
            "write_alpha_mean": ts.last_write_alpha_mean if write else None,  # diff E[α]
            "write_redundancy": ts.last_write_redundancy if write else None,  # diff E[max cos]
        }
        if compute_logits:
            out["logits"] = self.lm_head(H_text)                           # [B,T,V]
        else:
            out["hidden"]         = H_text
            out["lm_head_weight"] = self.lm_head.weight
        return out

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def make_cache(self) -> list:
        """Un cache d'attention vide par couche, à passer à `forward(cache=…)`
        pour décoder de façon incrémentale (voir attention.AttnCache)."""
        from .attention import AttnCache
        return [AttnCache() for _ in self.blocks]


# Legacy aliases (pre-rename scripts)
DeepSeekV4Mini = TrunkLM
DualModalDeepSeekV4Mini = ThoughtBankLM


# ── self-test ────────────────────────────────────────────────────────────────

def _selftest() -> None:
    """Contrat du forward : formes, portage de banque, et surtout `mem_read_layers`.

    Le test qui compte est celui du read restreint : `mem_read_layers` est un
    levier d'architecture (lire à chaque bloc compose les transformées en
    profondeur), et rien dans une loss ne dit qu'il a été pris en compte. Ici on
    vérifie que le bloc non désigné est LITTÉRALEMENT insensible à la banque.
    """
    torch.manual_seed(0)
    V, B, T = 40, 2, 12

    def _cfg(**kw):
        base = dict(vocab_size=V, d_model=32, n_layers=4, n_heads=2, d_head=8,
                    csa_m=2, hca_m=4, top_k_csa=2, n_win=4, d_latent_q=8,
                    n_groups=1, n_experts=4, n_shared=1, top_k_experts=2, d_ff=32,
                    mem_dim=16, max_mem=4, mem_seed_slots=2, mem_read_rank=4,
                    sinkhorn_iters=5)
        base.update(kw)
        return ThoughtBankConfig(**base)

    ids = torch.randint(0, V, (B, T))

    # ── formes et sorties ───────────────────────────────────────────────────
    m = ThoughtBankLM(_cfg())
    o = m(ids)
    assert o["logits"].shape == (B, T, V)
    assert o["mem_bank"].shape == (B, 3, 16), \
        f"banque fraîche = mem_seed_slots + 1 write, vu {tuple(o['mem_bank'].shape)}"
    assert torch.isfinite(o["logits"]).all() and torch.isfinite(o["balance_loss"])
    for k in ("write_alpha", "write_penalty", "write_alpha_mean", "write_redundancy"):
        assert o[k] is not None, k
    # compute_logits=False : le head est rendu pour une CE fusionnée hors modèle
    o2 = m(ids, compute_logits=False)
    assert "logits" not in o2 and o2["hidden"].shape == (B, T, 32)
    assert o2["lm_head_weight"] is m.lm_head.weight

    # ── portage de banque : entrée → sortie = +1 write, plafonné à max_mem ──
    bank = torch.randn(B, 2, 16)
    assert m(ids, init_mem=bank)["mem_bank"].shape == (B, 3, 16)
    full = torch.randn(B, 4, 16)                      # déjà à max_mem
    assert m(ids, init_mem=full)["mem_bank"].shape == (B, 4, 16)
    # une banque vide est traitée comme absente (seedée), jamais comme une erreur
    assert m(ids, init_mem=torch.zeros(B, 0, 16))["mem_bank"].shape == (B, 3, 16)

    # ── la banque INFLUENCE la sortie (sinon tout le programme est vide) ────
    torch.manual_seed(7)
    b1, b2 = torch.randn(B, 4, 16), torch.randn(B, 4, 16)
    l1 = m(ids, init_mem=b1)["logits"]
    l2 = m(ids, init_mem=b2)["logits"]
    assert not torch.allclose(l1, l2, atol=1e-6), \
        "deux banques différentes donnent la même sortie : le read est mort"
    # …et le gradient remonte jusqu'à la banque (le read est entraînable)
    b3 = b1.clone().requires_grad_(True)
    m(ids, init_mem=b3)["logits"].sum().backward()
    assert b3.grad is not None and b3.grad.abs().sum() > 0

    # ── mem_read_layers : seuls les blocs désignés lisent ───────────────────
    mr = ThoughtBankLM(_cfg(mem_read_layers=[3]))
    assert [blk.read_bank for blk in mr.blocks] == [False, False, False, True]
    # le bloc non lecteur est insensible à la banque, exactement
    X = torch.randn(B, T, 2, 32)
    with torch.no_grad():
        a = mr.blocks[0](X, b1)[0]
        c = mr.blocks[0](X, b2)[0]
    assert torch.equal(a, c), "un bloc read_bank=False lit encore la banque"
    with torch.no_grad():
        d1 = mr.blocks[3](X, b1)[0]
        d2 = mr.blocks[3](X, b2)[0]
    assert not torch.equal(d1, d2), "le bloc désigné ne lit pas la banque"
    # liste vide = tous les blocs (comportement historique)
    assert all(blk.read_bank for blk in ThoughtBankLM(_cfg()).blocks)

    # ── layer_banks (cascade) : chaque couche lit la banque qu'on lui donne ─
    lb = [b1, None, b2, b1]
    assert torch.isfinite(m(ids, layer_banks=lb)["logits"]).all()
    # None sur une couche = pas de read, pas un plantage
    assert torch.isfinite(m(ids, layer_banks=[None] * 4)["logits"]).all()

    # ── déterminisme en eval (dropout coupé) ────────────────────────────────
    me = ThoughtBankLM(_cfg(dropout=0.1)).eval()
    with torch.no_grad():
        assert torch.equal(me(ids, init_mem=b1)["logits"],
                           me(ids, init_mem=b1)["logits"])

    # ── pad_mask : une ligne entièrement pad ne contamine pas le write ──────
    pm = torch.ones(B, T, dtype=torch.bool)
    pm[1] = False
    assert torch.isfinite(m(ids, pad_mask=pm)["mem_bank"]).all()

    # ── variantes de read : swiglu, spectral norm ───────────────────────────
    for kw in ({"mem_read_swiglu": True}, {"mem_read_spectral_norm": True}):
        mv = ThoughtBankLM(_cfg(**kw))
        ov = mv(ids, init_mem=b1)
        assert torch.isfinite(ov["logits"]).all(), kw
        ov["logits"].sum().backward()

    # ── poids liés embed/lm_head (économie de params assumée) ───────────────
    assert m.lm_head.weight is m.embed.weight

    # ── decode_fuse : hoisting fw_A/fw_B — mêmes bits, et le mémo n'est
    #    JAMAIS servi périmé (write = nouvel objet ; mutation = _version) ─────
    from .decode import generate
    torch.manual_seed(11)
    m_off = ThoughtBankLM(_cfg()).double().eval()
    m_on = ThoughtBankLM(_cfg(decode_fuse=True)).double().eval()
    m_on.load_state_dict(m_off.state_dict())
    bank64 = torch.randn(B, 4, 16, dtype=torch.float64)
    pr = torch.randint(0, V, (B, 5))
    for use_cache in (False, True):
        g1, l1_ = generate(m_off, pr, bank=bank64, max_new=10, use_cache=use_cache)
        g2, l2_ = generate(m_on, pr, bank=bank64, max_new=10, use_cache=use_cache)
        assert torch.equal(g1, g2) and torch.equal(l1_, l2_), \
            f"decode_fuse change le décodage (cache={use_cache})"
    # le mémo est bien SERVI (fw_A ne tourne qu'une fois pour une banque
    # donnée — y compris À TRAVERS les generate, tant qu'aucun write ne l'a
    # remplacée ; banque fraîche ici pour partir d'un mémo froid)
    calls = {"n": 0}
    orig = m_on.blocks[0].fw_A.forward
    m_on.blocks[0].fw_A.forward = lambda x: (calls.__setitem__("n", calls["n"] + 1),
                                             orig(x))[1]
    bank_c = bank64.clone()
    generate(m_on, pr, bank=bank_c, max_new=6, use_cache=True)
    assert calls["n"] == 1, f"fw_A appelé {calls['n']}× pour un generate (mémo mort)"
    generate(m_on, pr, bank=bank_c, max_new=4, use_cache=True)
    assert calls["n"] == 1, "même banque, second generate : le mémo doit tenir"
    # …un write (append via cat, comme au <think>) invalide par identité…
    calls["n"] = 0
    bank_w = torch.cat([bank64, torch.randn(B, 1, 16, dtype=torch.float64)], dim=1)
    with torch.no_grad():
        la = m_on(pr, init_mem=bank_w, write=False)["logits"]
        lb = m_off(pr, init_mem=bank_w, write=False)["logits"]
    assert calls["n"] == 1 and torch.equal(la, lb), "miss sur nouvelle banque cassé"
    # …et une mutation EN PLACE invalide par _version (même objet, autre
    # contenu ; le mémo de bank_w est encore chaud depuis le bloc précédent,
    # donc le SEUL appel attendu est le recalcul post-mutation)
    calls["n"] = 0
    with torch.no_grad():
        m_on(pr, init_mem=bank_w, write=False)          # hit (mémo chaud)
        bank_w[:, 0] = 0.0
        lc = m_on(pr, init_mem=bank_w, write=False)["logits"]
        ld = m_off(pr, init_mem=bank_w, write=False)["logits"]
    assert calls["n"] == 1 and torch.equal(lc, ld), \
        "banque mutée en place : le mémo a servi des poids périmés"
    m_on.blocks[0].fw_A.forward = orig
    # sous gradient, le mémo est purgé : l'entraînement reprojette à chaque pas
    bg = bank64.clone().requires_grad_(True)
    m_on.train()
    m_on(pr, init_mem=bg)["logits"].sum().backward()
    assert m_on.blocks[0]._fw_memo is None and bg.grad is not None

    print("model self-test: OK (formes + compute_logits, portage de banque "
          "plafonné, la banque influence la sortie et reçoit du gradient, "
          "mem_read_layers restreint LITTÉRALEMENT, layer_banks + None, "
          "déterminisme eval, pad_mask, read swiglu/SN, poids liés, "
          "decode_fuse : fw_A/fw_B hoistés BIT-identiques — mémo servi 1×/"
          "generate, invalidé par write (identité) ET mutation (_version), "
          "purgé sous gradient)")


if __name__ == "__main__":
    _selftest()
