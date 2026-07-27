"""
Hybrid attention – DeepSeek-V4 §2.3

Two complementary attention variants interleaved across layers:

CSA – Compressed Sparse Attention
  * Overlapping KV compression by factor m (two series Ca/Cb)
  * Sparse top-k selection of compressed blocks via a lightweight indexer
  * Sliding window branch for local fine-grained dependencies
  * Shared KV Multi-Query Attention + grouped output projection

HCA – Heavily Compressed Attention
  * Non-overlapping compression by factor m' (≫ m)
  * Dense causal attention over *all* compressed blocks (no top-k)
  * Same sliding window branch, MQA, grouped output projection

Both variants use:
  * RMSNorm on queries and KV entries before core attention
  * Partial RoPE (applied to all head dims in this mini model)
  * Attention sink (learnable denominator addend per head)
"""
from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mhc import RMSNorm


# ── RoPE utilities ────────────────────────────────────────────────────────────

def _rope_cache(
    seq_len: int, dim: int, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    half = dim // 2
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, half, device=device).float() / half))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)            # [T, half]
    return freqs.cos(), freqs.sin()             # [T, half]


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: [..., T, dim]; cos/sin: [T, dim//2]"""
    T, dim = x.shape[-2], x.shape[-1]
    half = dim // 2
    c = cos[:T]                                 # [T, half]
    s = sin[:T]
    # prepend necessary dims for broadcast
    for _ in range(x.dim() - 2):
        c, s = c.unsqueeze(0), s.unsqueeze(0)
    x_e = x[..., :half]
    x_o = x[..., half:]
    return torch.cat([x_e * c - x_o * s, x_e * s + x_o * c], dim=-1)


# ── Shared helpers ────────────────────────────────────────────────────────────

def _grouped_out_proj(
    out: torch.Tensor,          # [B, T, n_heads, d_head]
    n_groups: int,
    group_linears: nn.ModuleList,
    final_proj: nn.Linear,
) -> torch.Tensor:
    B, T, n_heads, d_head = out.shape
    hpg = n_heads // n_groups
    parts = [
        group_linears[g](out[:, :, g * hpg:(g + 1) * hpg, :].reshape(B, T, -1))
        for g in range(n_groups)
    ]
    return final_proj(torch.cat(parts, dim=-1))   # [B, T, d_model]


def _attn_sink_softmax(
    logits: torch.Tensor,       # [BT, n_heads, n_kv]
    sink_logits: torch.Tensor,  # [n_heads]
    drop: nn.Dropout,
) -> torch.Tensor:
    """Softmax with a learnable sink added to the denominator (eq. 27)."""
    a_max = logits.detach().max(dim=-1, keepdim=True).values
    e = (logits - a_max).exp()
    denom = e.sum(dim=-1) + sink_logits.exp().unsqueeze(0)     # [BT, n_heads]
    w = e / denom.unsqueeze(-1)
    return drop(w)


# ── Compressed Sparse Attention (CSA) ────────────────────────────────────────

class CompressedSparseAttention(nn.Module):
    """
    CSA: overlapping KV compression (factor m) → sparse top-k + sliding window.
    Layer index i (0-indexed) is even in the hybrid interleaving.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_head: int,
        csa_m: int,
        top_k: int,
        n_win: int,
        d_latent_q: int,
        n_groups: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        assert n_heads % n_groups == 0
        self.d_model, self.n_heads, self.d_head = d_model, n_heads, d_head
        self.m, self.top_k, self.n_win = csa_m, top_k, n_win
        self.n_groups = n_groups
        hpg = n_heads // n_groups

        # Overlapping KV compression – two series (Ca/Cb = values, Za/Zb = gates)
        # Positional biases are applied to Z (gates) only, NOT to C (values) – §2.3.1 eq. 9-12
        self.W_kv_a = nn.Linear(d_model, d_head, bias=False)
        self.W_kv_b = nn.Linear(d_model, d_head, bias=False)
        self.W_z_a  = nn.Linear(d_model, d_head, bias=False)
        self.W_z_b  = nn.Linear(d_model, d_head, bias=False)
        self.pos_a  = nn.Parameter(torch.zeros(csa_m, d_head))  # bias for Za only
        self.pos_b  = nn.Parameter(torch.zeros(csa_m, d_head))  # bias for Zb only

        # Low-rank queries: d → d_latent_q → n_heads * d_head
        self.W_dq = nn.Linear(d_model, d_latent_q, bias=False)
        self.W_uq = nn.Linear(d_latent_q, n_heads * d_head, bias=False)
        # Lightning indexer (§2.3.1 eqs. 13-16): multi-head indexer queries + head weights
        # W_IUQ: latent → n_idx_heads * d_head  (shared latent c_Q from W_dq)
        # W_w  : d_model → n_idx_heads  (per-head scalar weights for score aggregation)
        self.n_idx_heads = max(1, n_heads // 4)  # lightweight: n_h/4 heads
        self.W_iq = nn.Linear(d_latent_q, self.n_idx_heads * d_head, bias=False)
        self.W_w  = nn.Linear(d_model, self.n_idx_heads, bias=False)

        # Sliding window KV (uncompressed local tokens)
        self.W_wk = nn.Linear(d_model, d_head, bias=False)
        self.W_wv = nn.Linear(d_model, d_head, bias=False)

        # Grouped output projection
        d_g = d_model // n_groups
        self.out_group = nn.ModuleList(
            [nn.Linear(hpg * d_head, d_g, bias=False) for _ in range(n_groups)]
        )
        self.out_proj = nn.Linear(n_groups * d_g, d_model, bias=False)

        # Norms (applied just before core attention – avoids exploding logits)
        self.q_norm  = RMSNorm(d_head)
        self.kv_norm = RMSNorm(d_head)

        # Attention sink (one learnable scalar per head)
        self.sink_logits = nn.Parameter(torch.zeros(n_heads))
        self.drop = nn.Dropout(dropout)

    # ── KV compression ────────────────────────────────────────────────────────

    def _compress_kv(self, H_pad: torch.Tensor) -> torch.Tensor:
        """
        Overlapping compression: block i merges Ca[i] (current) with Cb[i-1] (previous).
        H_pad: [B, T_pad, d_model]  (T_pad divisible by m)
        Returns: CComp [B, n_blocks, d_head]
        """
        B, T_pad, _ = H_pad.shape
        m = self.m
        n_blocks = T_pad // m
        H_b = H_pad.view(B, n_blocks, m, -1)

        # Values (Ca, Cb): no positional bias (eq. 9)
        # Gates (Za, Zb): add learnable positional bias (eq. 10-11)
        Ca = self.W_kv_a(H_b)                 # [B, n_blocks, m, d_head]
        Cb = self.W_kv_b(H_b)
        Za = self.W_z_a(H_b) + self.pos_a     # bias on gates only
        Zb = self.W_z_b(H_b) + self.pos_b

        # Shift Cb by 1 block; mask block-0 predecessor with -∞
        Cb_prev = torch.cat([torch.zeros_like(Cb[:, :1]), Cb[:, :-1]], dim=1)
        Zb_shift = torch.cat([Zb[:, :1], Zb[:, :-1]], dim=1)
        # block 0 gets -inf so softmax assigns zero weight to the phantom predecessor
        inf_mask = torch.zeros(n_blocks, device=H_pad.device, dtype=torch.bool)
        inf_mask[0] = True
        Zb_prev = Zb_shift.masked_fill(inf_mask.view(1, n_blocks, 1, 1), float("-inf"))

        # Concatenate along m dimension; softmax over 2m entries per feature dim
        Z_cat = torch.cat([Za, Zb_prev], dim=2)    # [B, n_blocks, 2m, d_head]
        C_cat = torch.cat([Ca, Cb_prev], dim=2)
        S     = F.softmax(Z_cat, dim=2)            # [B, n_blocks, 2m, d_head]
        return (S * C_cat).sum(dim=2)              # [B, n_blocks, d_head]

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        B, T, d = H.shape
        m, n_win = self.m, self.n_win

        # Pad T to multiple of m
        pad = (-T) % m
        H_pad = F.pad(H, (0, 0, 0, pad)) if pad else H
        n_blocks = H_pad.shape[1] // m

        # 1. Compress KV
        CComp = self._compress_kv(H_pad)           # [B, n_blocks, d_head]

        # 2. Low-rank queries (shared latent c_Q for both indexer and core attention)
        cQ  = self.W_dq(H)                         # [B, T, d_latent_q]
        q   = self.W_uq(cQ).view(B, T, self.n_heads, self.d_head)

        # RoPE on queries
        cos, sin = _rope_cache(T, self.d_head, H.device)
        q = _apply_rope(q.permute(0, 2, 1, 3), cos, sin).permute(0, 2, 1, 3)
        q = self.q_norm(q)                         # [B, T, n_heads, d_head]

        # 3. Lightning indexer (§2.3.1 eqs. 13-16): multi-head with head weights + ReLU
        #    I_{t,s} = Σ_h  w_{t,h} · ReLU(q_I_{t,h} · K_IComp_s)
        n_ih = self.n_idx_heads
        qI   = self.W_iq(cQ).view(B, T, n_ih, self.d_head)    # [B, T, n_ih, d_head]
        w_h  = self.W_w(H)                                     # [B, T, n_ih] head weights
        # [B, T, n_ih, n_blocks]  via ReLU dot product
        idx_scores_h = F.relu(
            torch.einsum("bthd,bnd->bthn", qI, CComp) / math.sqrt(self.d_head)
        )
        idx_scores = torch.einsum("bth,bthn->btn", w_h, idx_scores_h)  # [B, T, n_blocks]

        # Causal block mask: token t can see block j only if j < t//m
        block_of_t = torch.arange(T, device=H.device) // m          # [T]
        block_j    = torch.arange(n_blocks, device=H.device)        # [nb]
        causal     = (block_of_t[:, None] <= block_j[None, :])      # [T, nb] True=masked
        idx_scores = idx_scores.masked_fill(causal.unsqueeze(0), float("-inf"))

        k = min(self.top_k, n_blocks)
        if k > 0:
            top_scores, top_idx = idx_scores.topk(k, dim=-1)        # [B, T, k]
            valid = (top_scores > -1e9)                              # [B, T, k]
            exp   = top_idx.unsqueeze(-1).expand(-1, -1, -1, self.d_head)
            CComp_exp = CComp.unsqueeze(1).expand(-1, T, -1, -1)
            KV_sel = CComp_exp.gather(2, exp)                        # [B, T, k, d_head]
            KV_sel = self.kv_norm(KV_sel)
        else:
            k = 0
            valid  = H.new_zeros(B, T, 0, dtype=torch.bool)
            KV_sel = H.new_zeros(B, T, 0, self.d_head)

        # 4. Sliding window KV (causal: last n_win tokens before position t)
        Wk = self.kv_norm(self.W_wk(H))            # [B, T, d_head]
        Wv = self.W_wv(H)
        # Pad left by n_win; for token t, gather indices [t, t+1, ..., t+n_win-1]
        Wk_p = F.pad(Wk, (0, 0, n_win, 0))
        Wv_p = F.pad(Wv, (0, 0, n_win, 0))
        win_idx = (
            torch.arange(T, device=H.device).unsqueeze(1)
            + torch.arange(n_win, device=H.device).unsqueeze(0)
        )                                           # [T, n_win]
        KV_wk = Wk_p[:, win_idx, :]               # [B, T, n_win, d_head]
        KV_wv = Wv_p[:, win_idx, :]

        # 5. Combined keys/values: compressed (key=value) + window (separate k,v)
        K_all = torch.cat([KV_sel, KV_wk], dim=2) # [B, T, k+n_win, d_head]
        V_all = torch.cat([KV_sel, KV_wv], dim=2)
        n_kv  = k + n_win

        # 6. MQA: all n_heads share the same K/V
        q_bt  = q.reshape(B * T, self.n_heads, self.d_head)
        K_bt  = K_all.reshape(B * T, n_kv, self.d_head)
        V_bt  = V_all.reshape(B * T, n_kv, self.d_head)

        logits = torch.einsum("bhd,bnd->bhn", q_bt, K_bt) / math.sqrt(self.d_head)

        # Mask invalid (causally blocked) compressed entries
        if k > 0:
            valid_bt = valid.reshape(B * T, k).unsqueeze(1).expand(-1, self.n_heads, -1)
            logits[:, :, :k] = logits[:, :, :k].masked_fill(~valid_bt, float("-inf"))

        attn_w = _attn_sink_softmax(logits, self.sink_logits, self.drop)
        out    = torch.einsum("bhn,bnd->bhd", attn_w, V_bt).view(B, T, self.n_heads, self.d_head)

        return _grouped_out_proj(out, self.n_groups, self.out_group, self.out_proj)

    # ── décodage incrémental ─────────────────────────────────────────────────

    def _fill_cache(self, H: torch.Tensor, cache) -> None:
        """Remplit le cache après un forward complet sur le préfixe."""
        B, T, _ = H.shape
        m = self.m
        nb = T // m                               # blocs FERMÉS par ce préfixe
        cache.comp = (self._compress_kv(H[:, :nb * m]) if nb
                      else H.new_zeros(B, 0, self.d_head))
        cache.hist = H[:, max(0, (nb - 1) * m):]  # de quoi fermer le bloc suivant
        cache.wk = _win_init(self.kv_norm(self.W_wk(H)), self.n_win)
        cache.wv = _win_init(self.W_wv(H), self.n_win)
        cache.pos = T

    def forward_cached(self, H_new: torch.Tensor, cache) -> torch.Tensor:
        """Un pas de décodage : H_new est [B, 1, d], le préfixe vit dans `cache`.

        Le premier appel (cache vide) délègue au forward COMPLET — même code,
        donc mêmes activations que l'entraînement — puis remplit le cache.
        """
        B, S, _ = H_new.shape
        if cache.pos == 0:
            out = self.forward(H_new)
            self._fill_cache(H_new, cache)
            return out
        assert S == 1, "après le préfixe, le cache avance token par token"

        m, n_win = self.m, self.n_win
        p = cache.pos                              # position ABSOLUE du token
        h = H_new

        # 1. requête à la position absolue p (RoPE décalé)
        cQ = self.W_dq(h)
        q = self.W_uq(cQ).view(B, 1, self.n_heads, self.d_head)
        cos, sin = _rope_at(p, 1, self.d_head, h.device)
        q = _apply_rope(q.permute(0, 2, 1, 3), cos, sin).permute(0, 2, 1, 3)
        q = self.q_norm(q)

        # 2. indexeur + top-k. Le cache ne détient QUE les blocs fermés, mais on
        #    présente au top-k le même nombre de candidats que le chemin
        #    complet (les blocs interdits par la causalité à -inf) : sur des
        #    scores ex æquo — le cas d'un indexeur encore plat en début
        #    d'entraînement, où le ReLU renvoie 0 partout — `topk` départage par
        #    l'INDICE, donc offrir moins de candidats change la sélection. Les
        #    entrées invalides sont ensuite masquées comme dans le chemin
        #    complet, et leur poids d'attention est exactement nul.
        comp = cache.comp                          # [B, nb_vis, d_head]
        nb = comp.size(1)
        nbf = -(-(p + 1) // m)                     # n_blocks vu par le forward
        n_ih = self.n_idx_heads
        qI = self.W_iq(cQ).view(B, 1, n_ih, self.d_head)
        w_h = self.W_w(h)
        k = min(self.top_k, nbf)
        if k > 0:
            if nb > 0:
                sc_h = F.relu(torch.einsum("bthd,bnd->bthn", qI, comp)
                              / math.sqrt(self.d_head))
                sc = torch.einsum("bth,bthn->btn", w_h, sc_h)      # [B,1,nb]
            else:
                sc = h.new_zeros(B, 1, 0)
            if nbf > nb:                           # blocs causalement interdits
                sc = F.pad(sc, (0, nbf - nb), value=float("-inf"))
            top_scores, top_idx = sc.topk(k, dim=-1)
            valid = top_scores > -1e9                              # [B,1,k]
            if nb > 0:
                exp = top_idx.clamp(max=nb - 1).unsqueeze(-1).expand(-1, -1, -1, self.d_head)
                KV_sel = self.kv_norm(
                    comp.unsqueeze(1).expand(-1, 1, -1, -1).gather(2, exp))
            else:
                KV_sel = h.new_zeros(B, 1, k, self.d_head)
        else:
            valid = h.new_zeros(B, 1, 0, dtype=torch.bool)
            KV_sel = h.new_zeros(B, 1, 0, self.d_head)

        # 3. fenêtre glissante : les n_win tokens qui PRÉCÈDENT p (le chemin
        #    complet n'inclut pas le token courant dans sa propre fenêtre)
        K_all = torch.cat([KV_sel, cache.wk.unsqueeze(1)], dim=2)
        V_all = torch.cat([KV_sel, cache.wv.unsqueeze(1)], dim=2)
        n_kv = k + n_win

        q_bt = q.reshape(B, self.n_heads, self.d_head)
        logits = torch.einsum("bhd,bnd->bhn", q_bt,
                              K_all.reshape(B, n_kv, self.d_head)) / math.sqrt(self.d_head)
        if k > 0:
            vb = valid.reshape(B, k).unsqueeze(1).expand(-1, self.n_heads, -1)
            logits[:, :, :k] = logits[:, :, :k].masked_fill(~vb, float("-inf"))
        attn_w = _attn_sink_softmax(logits, self.sink_logits, self.drop)
        out = torch.einsum("bhn,bnd->bhd", attn_w,
                           V_all.reshape(B, n_kv, self.d_head)
                           ).view(B, 1, self.n_heads, self.d_head)

        # 4. avancer le cache — APRÈS l'attention : le bloc que ce token vient
        #    de fermer le contient, et le masque causal l'interdit à lui-même.
        hist = torch.cat([cache.hist, h], dim=1)
        if (p + 1) % m == 0:
            newc = self._compress_kv(hist[:, -2 * m:])[:, -1:]
            cache.comp = torch.cat([cache.comp, newc], dim=1)
            hist = hist[:, -m:]
        cache.hist = hist[:, -2 * m:]
        cache.wk = _win_push(cache.wk, self.kv_norm(self.W_wk(h)), n_win)
        cache.wv = _win_push(cache.wv, self.W_wv(h), n_win)
        cache.pos = p + 1

        return _grouped_out_proj(out, self.n_groups, self.out_group, self.out_proj)


# ── Heavily Compressed Attention (HCA) ───────────────────────────────────────

class HeavilyCompressedAttention(nn.Module):
    """
    HCA: non-overlapping compression (factor m' ≫ m) → dense causal attention.
    No top-k selection – all preceding compressed blocks are attended.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_head: int,
        hca_m: int,
        n_win: int,
        d_latent_q: int,
        n_groups: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        assert n_heads % n_groups == 0
        self.d_model, self.n_heads, self.d_head = d_model, n_heads, d_head
        self.m_prime, self.n_win = hca_m, n_win
        self.n_groups = n_groups
        hpg = n_heads // n_groups

        # Single KV projection (no overlapping for HCA)
        self.W_kv = nn.Linear(d_model, d_head, bias=False)
        self.W_z  = nn.Linear(d_model, d_head, bias=False)
        self.pos  = nn.Parameter(torch.zeros(hca_m, d_head))

        # Low-rank queries
        self.W_dq = nn.Linear(d_model, d_latent_q, bias=False)
        self.W_uq = nn.Linear(d_latent_q, n_heads * d_head, bias=False)

        # Sliding window KV
        self.W_wk = nn.Linear(d_model, d_head, bias=False)
        self.W_wv = nn.Linear(d_model, d_head, bias=False)

        # Grouped output projection
        d_g = d_model // n_groups
        self.out_group = nn.ModuleList(
            [nn.Linear(hpg * d_head, d_g, bias=False) for _ in range(n_groups)]
        )
        self.out_proj = nn.Linear(n_groups * d_g, d_model, bias=False)

        self.q_norm  = RMSNorm(d_head)
        self.kv_norm = RMSNorm(d_head)

        self.sink_logits = nn.Parameter(torch.zeros(n_heads))
        self.drop = nn.Dropout(dropout)

    def _compress_kv(self, H_pad: torch.Tensor) -> torch.Tensor:
        """Non-overlapping compression."""
        B, T_pad, _ = H_pad.shape
        m = self.m_prime
        n_blocks = T_pad // m
        H_b = H_pad.view(B, n_blocks, m, -1)
        C = self.W_kv(H_b)                         # [B, nb, m, d_head]
        Z = self.W_z(H_b) + self.pos               # [B, nb, m, d_head]
        S = F.softmax(Z, dim=2)
        return (S * C).sum(dim=2)                  # [B, nb, d_head]

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        B, T, d = H.shape
        m, n_win = self.m_prime, self.n_win

        pad   = (-T) % m
        H_pad = F.pad(H, (0, 0, 0, pad)) if pad else H
        n_blocks = H_pad.shape[1] // m

        # 1. Compress KV
        CComp = self.kv_norm(self._compress_kv(H_pad))  # [B, nb, d_head]

        # 2. Queries
        cQ = self.W_dq(H)
        q  = self.W_uq(cQ).view(B, T, self.n_heads, self.d_head)
        cos, sin = _rope_cache(T, self.d_head, H.device)
        q = _apply_rope(q.permute(0, 2, 1, 3), cos, sin).permute(0, 2, 1, 3)
        q = self.q_norm(q)

        # 3. Dense causal attention over compressed blocks
        block_of_t = torch.arange(T, device=H.device) // m     # [T]
        block_j    = torch.arange(n_blocks, device=H.device)   # [nb]
        causal     = (block_of_t[:, None] <= block_j[None, :]) # [T, nb] True=masked

        q_bt = q.reshape(B * T, self.n_heads, self.d_head)
        CComp_bt = CComp.unsqueeze(1).expand(-1, T, -1, -1).reshape(B * T, n_blocks, self.d_head)

        logits_comp = torch.einsum("bhd,bnd->bhn", q_bt, CComp_bt) / math.sqrt(self.d_head)
        causal_bt = causal.unsqueeze(0).expand(B, -1, -1).reshape(B * T, n_blocks)
        logits_comp = logits_comp.masked_fill(causal_bt.unsqueeze(1), float("-inf"))

        # 4. Sliding window
        Wk = self.kv_norm(self.W_wk(H))
        Wv = self.W_wv(H)
        Wk_p = F.pad(Wk, (0, 0, n_win, 0))
        Wv_p = F.pad(Wv, (0, 0, n_win, 0))
        win_idx = (
            torch.arange(T, device=H.device).unsqueeze(1)
            + torch.arange(n_win, device=H.device).unsqueeze(0)
        )
        KV_wk = Wk_p[:, win_idx, :].reshape(B * T, n_win, self.d_head)
        KV_wv = Wv_p[:, win_idx, :].reshape(B * T, n_win, self.d_head)

        logits_win = torch.einsum("bhd,bnd->bhn", q_bt, KV_wk) / math.sqrt(self.d_head)

        # 5. Combined attention
        logits_all = torch.cat([logits_comp, logits_win], dim=-1)  # [BT, n_h, nb+n_win]
        V_all_bt   = torch.cat([CComp_bt, KV_wv], dim=1)          # [BT, nb+n_win, d_head]

        attn_w = _attn_sink_softmax(logits_all, self.sink_logits, self.drop)
        out    = torch.einsum("bhn,bnd->bhd", attn_w, V_all_bt).view(B, T, self.n_heads, self.d_head)

        return _grouped_out_proj(out, self.n_groups, self.out_group, self.out_proj)

    # ── décodage incrémental (même principe que CSA, sans top-k) ─────────────

    def _fill_cache(self, H: torch.Tensor, cache) -> None:
        B, T, _ = H.shape
        m = self.m_prime
        nb = T // m
        cache.comp = (self.kv_norm(self._compress_kv(H[:, :nb * m])) if nb
                      else H.new_zeros(B, 0, self.d_head))
        cache.hist = H[:, max(0, (nb - 1) * m):]
        cache.wk = _win_init(self.kv_norm(self.W_wk(H)), self.n_win)
        cache.wv = _win_init(self.W_wv(H), self.n_win)
        cache.pos = T

    def forward_cached(self, H_new: torch.Tensor, cache) -> torch.Tensor:
        B, S, _ = H_new.shape
        if cache.pos == 0:
            out = self.forward(H_new)
            self._fill_cache(H_new, cache)
            return out
        assert S == 1, "après le préfixe, le cache avance token par token"

        m, n_win = self.m_prime, self.n_win
        p = cache.pos
        h = H_new

        cQ = self.W_dq(h)
        q = self.W_uq(cQ).view(B, 1, self.n_heads, self.d_head)
        cos, sin = _rope_at(p, 1, self.d_head, h.device)
        q = _apply_rope(q.permute(0, 2, 1, 3), cos, sin).permute(0, 2, 1, 3)
        q = self.q_norm(q)
        q_bt = q.reshape(B, self.n_heads, self.d_head)

        # mêmes formes que le chemin complet : les blocs interdits par la
        # causalité sont présentés à -inf (poids exactement nul), plutôt
        # qu'omis — c'est ce qui rend la comparaison bit à bit possible.
        comp = cache.comp                          # blocs fermés (= visibles)
        nb = comp.size(1)
        nbf = -(-(p + 1) // m)
        K_all = torch.cat([comp, comp.new_zeros(B, nbf - nb, self.d_head),
                           cache.wk], dim=1)
        V_all = torch.cat([comp, comp.new_zeros(B, nbf - nb, self.d_head),
                           cache.wv], dim=1)
        logits = torch.einsum("bhd,bnd->bhn", q_bt, K_all) / math.sqrt(self.d_head)
        if nbf > nb:
            logits[:, :, nb:nbf] = float("-inf")
        attn_w = _attn_sink_softmax(logits, self.sink_logits, self.drop)
        out = torch.einsum("bhn,bnd->bhd", attn_w, V_all
                           ).view(B, 1, self.n_heads, self.d_head)

        hist = torch.cat([cache.hist, h], dim=1)
        if (p + 1) % m == 0:
            newc = self.kv_norm(self._compress_kv(hist[:, -m:]))
            cache.comp = torch.cat([cache.comp, newc], dim=1)
            hist = hist[:, -m:]
        cache.hist = hist[:, -2 * m:]
        cache.wk = _win_push(cache.wk, self.kv_norm(self.W_wk(h)), n_win)
        cache.wv = _win_push(cache.wv, self.W_wv(h), n_win)
        cache.pos = p + 1

        return _grouped_out_proj(out, self.n_groups, self.out_group, self.out_proj)


# ── Cache KV incrémental ─────────────────────────────────────────────────────
#
# Sans cache, générer un token coûte un forward complet sur TOUT le préfixe :
# à max_new=240 (tour de l'env code-exec) un seul tour paie ~125k tokens-forward
# là où le travail réel en demande 640. Comme le décodage est le goulot déclaré
# de rl_disagg, c'est deux ordres de grandeur jetés dans la phase RL.
#
# Ce qui rend le cache possible ici tient à un détail du masque causal : un token
# t ne voit un bloc compressé j que si j < t // m — donc uniquement des blocs
# COMPLETS, dont le contenu ne dépend que de tokens déjà vus. Le bloc en cours
# (celui qui contient t, et le rembourrage de fin de séquence) n'est JAMAIS lu.
# Un bloc, une fois fermé, est donc définitif : on le calcule une fois.
#
# Le reste suit : la fenêtre glissante garde ses n_win dernières entrées (avec
# les zéros à gauche du début de séquence, que le chemin complet produit aussi
# via son F.pad), et tout ce qui n'est pas l'attention — mHC, MoE, read
# fast-weight — est déjà positionwise.

class AttnCache:
    """État porté par UNE couche d'attention entre deux tokens décodés.

    `comp` ne contient que des blocs fermés — ceux que le masque causal autorise.
    `hist` garde juste assez de H brut pour fermer le bloc suivant (2m pour la
    compression chevauchante de CSA, qui fusionne le bloc courant et son
    prédécesseur ; m suffirait à HCA, on ne complique pas).
    """
    __slots__ = ("pos", "comp", "hist", "wk", "wv")

    def __init__(self) -> None:
        self.pos = 0            # tokens déjà consommés
        self.comp = None        # [B, nb, d_head] blocs compressés FERMÉS
        self.hist = None        # [B, ≤2m, d_model] queue de H
        self.wk = None          # [B, n_win, d_head] fenêtre (zéros à gauche)
        self.wv = None


def _rope_at(pos: int, n: int, dim: int, device: torch.device):
    """cos/sin de RoPE aux positions ABSOLUES [pos, pos+n).

    Élément par élément, `p * inv_freq` vaut la même chose qu'on le calcule dans
    un tenseur de longueur T ou de longueur 1 : le cache incrémental retrouve
    donc au bit près les angles du chemin complet.
    """
    half = dim // 2
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, half, device=device).float() / half))
    t = torch.arange(pos, pos + n, device=device).float()
    freqs = torch.outer(t, inv_freq)
    return freqs.cos(), freqs.sin()


def _win_push(buf, new, n_win):
    """Fait glisser la fenêtre d'un token. `buf` est [B, n_win, d]."""
    return torch.cat([buf[:, 1:], new], dim=1) if n_win > 0 else buf


def _win_init(vals, n_win):
    """Les n_win dernières entrées, complétées à GAUCHE par des zéros — la même
    convention que le F.pad(..., n_win, 0) du chemin complet."""
    B, T, d = vals.shape
    if n_win == 0:
        return vals.new_zeros(B, 0, d)
    if T >= n_win:
        return vals[:, T - n_win:]
    return torch.cat([vals.new_zeros(B, n_win - T, d), vals], dim=1)


# ── self-test ────────────────────────────────────────────────────────────────

def _mk(cls, dtype, **kw):
    """Un module de test dont l'indexeur n'est PAS dégénéré.

    À l'init par défaut les scores de l'indexeur passent par un ReLU qui rend 0
    partout : tous les blocs sont ex æquo, `topk` départage par l'indice, et un
    cache faux passerait le test sans qu'on voie rien. On rend donc les poids
    non triviaux — c'est le seul régime où la sélection top-k discrimine.
    """
    import torch.nn as nn
    m = cls(d_model=24, n_heads=4, d_head=8, n_win=5, d_latent_q=8,
            n_groups=2, **kw)
    for p in m.parameters():
        if p.dim() >= 2:
            nn.init.normal_(p, std=0.5)
    return m.to(dtype).eval()


def _selftest() -> None:
    """Le garde-fou du cache : décoder token par token doit rendre EXACTEMENT ce
    que rend un recompute complet — y compris (surtout) aux longueurs qui
    TRAVERSENT une frontière de bloc `csa_m` / `hca_m`.

    « Exactement » se mesure en float64, pas au bit près en float32 : le chemin
    complet contracte ses matmuls sur B*T lignes, le cache sur B — c'est une
    autre réduction BLAS, donc quelques ULP d'écart, sans rapport avec la
    correction de l'algorithme. En float64 l'écart tombe à l'epsilon machine
    (~1e-16) : toute VRAIE erreur (bloc fermé un token trop tôt, fenêtre
    décalée, RoPE à la mauvaise position absolue) est en O(1) et ne peut pas s'y
    cacher. C'est un test plus dur qu'une égalité bit à bit en float32, qui
    aurait dû tolérer ces mêmes ULP.
    """
    torch.manual_seed(0)
    M, MP, NW = 4, 6, 5

    for dtype, tol in ((torch.float64, 1e-13), (torch.float32, 1e-5)):
        for name, mod, m in (("CSA", _mk(CompressedSparseAttention, dtype,
                                         csa_m=M, top_k=3), M),
                             ("HCA", _mk(HeavilyCompressedAttention, dtype,
                                         hca_m=MP), MP)):
            H = torch.randn(2, 3 * m + 8, 24, dtype=dtype)
            worst = 0.0
            with torch.no_grad():
                # préfixes autour des frontières : avant, pile dessus, après —
                # plus un préfixe plus court que la fenêtre glissante
                for T0 in (1, 2, m - 1, m, m + 1, 2 * m, NW + 1):
                    cache = AttnCache()
                    out = mod.forward_cached(H[:, :T0], cache)
                    assert torch.equal(out, mod.forward(H[:, :T0])), \
                        f"{name}: le préfixe doit passer par le forward COMPLET"
                    assert cache.pos == T0
                    for i in range(T0, H.size(1)):
                        assert cache.comp.size(1) == cache.pos // m, \
                            (f"{name}: {cache.comp.size(1)} blocs fermés pour "
                             f"pos {cache.pos} (attendu {cache.pos // m})")
                        step = mod.forward_cached(H[:, i:i + 1], cache)
                        full = mod.forward(H[:, :i + 1])[:, -1:]
                        assert step.shape == full.shape
                        err = (step - full).abs().max().item()
                        rel = err / (full.abs().max().item() or 1.0)
                        worst = max(worst, rel)
                        assert rel <= tol, (
                            f"{name}/{dtype}: cache ≠ recompute à la position "
                            f"{i} (préfixe {T0}, m={m}, ferme un bloc="
                            f"{(i + 1) % m == 0}) — écart relatif {rel:.3e}")
            if dtype is torch.float64:
                assert worst < 1e-13, worst
                _worst64 = worst

    # ── la fenêtre glissante n'inclut PAS le token courant, et démarre sur des
    #    zéros — c'est la convention du F.pad du chemin complet, pas un choix ─
    mod = _mk(CompressedSparseAttention, torch.float64, csa_m=M, top_k=3)
    cache = AttnCache()
    H = torch.randn(1, 2, 24, dtype=torch.float64)
    with torch.no_grad():
        mod.forward_cached(H[:, :1], cache)
    assert cache.wk.shape == (1, NW, 8)
    assert float(cache.wk[:, :NW - 1].abs().max()) == 0.0, \
        "la fenêtre doit être remplie de zéros À GAUCHE au début de séquence"
    assert float(cache.wk[:, -1:].abs().max()) > 0.0, "le token 0 manque"

    print(f"attention self-test: OK (CSA+HCA, cache incrémental exact au "
          f"recompute complet — float64 ≤1e-13, float32 ≤1e-5 (ULP BLAS) — "
          f"7 longueurs de préfixe, franchissements csa_m={M}/hca_m={MP} "
          f"inclus, blocs fermés = pos//m, fenêtre zéro-paddée à gauche)")


if __name__ == "__main__":
    _selftest()
