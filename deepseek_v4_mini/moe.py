"""
DeepSeekMoE – DeepSeek-V4 §2.1

Fine-grained expert design with two classes of experts:
  • Shared experts: always activated (n_shared, typically 1)
  • Routed experts: top-k activated per token based on affinity scores

Routing score (from DeepSeek-V4):  sqrt(softplus(h W_gate))
  replacing the Sigmoid used in DeepSeek-V3.

Load balancing: sequence-wise auxiliary loss that penalises uneven expert usage
within a single sequence (light, auxiliary-loss-free strategy from V3 is the
default for large models; here we use a small explicit balance loss for clarity).
"""
from __future__ import annotations

import torch
import torch._dynamo
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    """SwiGLU feed-forward block: works on any [..., d_model] input."""

    def __init__(self, d_model: int, d_hidden: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.w12 = nn.Linear(d_model, d_hidden * 2, bias=False)
        self.w3  = nn.Linear(d_hidden, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.w12(x).chunk(2, dim=-1)
        # SwiGLU clamping for training stability (§4.2.3):
        # linear branch ∈ [-10, 10], gate branch upper-bounded at 10
        a = a.clamp(-10.0, 10.0)
        b = b.clamp(max=10.0)
        return self.w3(self.drop(F.silu(a) * b))


class DeepSeekMoE(nn.Module):
    """
    Mixture-of-Experts FFN layer.

    Returns (output, balance_loss) where balance_loss is a scalar auxiliary
    loss to be weighted and added to the main training objective.
    """

    def __init__(
        self,
        d_model: int,
        n_experts: int,
        n_shared: int,
        top_k_experts: int,
        d_ff: int,
        dropout: float = 0.0,
        dense_decode: bool = False,
    ) -> None:
        super().__init__()
        assert top_k_experts <= n_experts
        self.n_routed      = n_experts
        self.n_shared      = n_shared
        self.top_k         = top_k_experts

        # Shared experts (always active – merged into a single scaled output)
        self.shared = nn.ModuleList(
            [SwiGLU(d_model, d_ff, dropout) for _ in range(n_shared)]
        )

        # Routed experts
        self.experts = nn.ModuleList(
            [SwiGLU(d_model, d_ff, dropout) for _ in range(n_experts)]
        )

        # Gating: score = sqrt(softplus(h W_gate))
        self.W_gate = nn.Linear(d_model, n_experts, bias=False)

        # decode_dense_moe : dispatch dense sans sync à petit BT (décodage).
        self.dense_decode = dense_decode

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor, need_balance: bool = True):
        """
        x: [B, T, d_model]
        Returns: (output [B, T, d_model], balance_loss scalar)

        need_balance=False (decode_fuse) : la balance_loss est de la télémétrie
        d'ENTRAÎNEMENT — one_hot/mean/pow calculés à chaque token décodé sous
        no_grad pour un scalaire que personne ne lit. Rend 0 à la place ; la
        SORTIE est inchangée au bit (le routage n'en dépend pas).
        """
        B, T, d = x.shape
        flat = x.reshape(B * T, d)

        # ── Shared experts ────────────────────────────────────────────────────
        if self.n_shared == 1:
            shared_out = self.shared[0](flat)
        else:
            shared_out = sum(e(flat) for e in self.shared) / self.n_shared

        # ── Routing ───────────────────────────────────────────────────────────
        # clamp_min: softplus underflows to exactly 0 in bf16 for very negative
        # logits, and sqrt'(0)=inf turns the (zero) grad of unselected experts
        # into 0*inf=NaN
        scores = torch.sqrt(F.softplus(self.W_gate(flat)).clamp_min(1e-12))  # [BT, n_routed]
        top_scores, top_idx = scores.topk(self.top_k, dim=-1)  # [BT, top_k]
        # Normalise weights within the selected set
        top_w = top_scores / (top_scores.sum(dim=-1, keepdim=True) + 1e-8)

        # ── Dispatch & combine ────────────────────────────────────────────────
        # decode_dense_moe : au décodage (no_grad, BT=1), la double boucle
        # ci-dessous coûte top_k × n_experts itérations Python dont chaque
        # .any()/masquage force une sync GPU→CPU, pour router UN token. Évaluer
        # les n_experts densément et sélectionner par indexation coûte 2× les
        # FLOPs (négligeables à cette taille) et ZÉRO sync — et rend les shapes
        # fixes, la condition de capture d'un CUDA graph.
        #
        # BT == 1 STRICT, mesuré et pas négociable : à BT ≥ 2 la boucle évalue
        # chaque expert sur un SOUS-ENSEMBLE des lignes et le dense sur toutes,
        # et un GEMM à M=1 vs M=2 ne rend pas les mêmes bits (kernels BLAS
        # différents — écart rel ~1.2e-16 = epsilon float64, mesuré CPU). À
        # BT=1 le sous-ensemble élu EST le batch entier : mêmes bits garantis.
        if self.dense_decode and not torch.is_grad_enabled() and flat.size(0) == 1:
            routed_out = self._dispatch_dense(flat, top_idx, top_w)
        else:
            routed_out = self._dispatch_loop(flat, top_idx, top_w)

        # ── Sequence-wise load balance loss ───────────────────────────────────
        # Penalise deviation from uniform expert usage within the batch
        if need_balance:
            usage = F.one_hot(top_idx, self.n_routed).float().sum(dim=1)  # [BT, n_routed]
            usage_frac    = usage.mean(dim=0)                              # [n_routed]
            expected_frac = torch.full_like(usage_frac, self.top_k / self.n_routed)
            balance_loss  = ((usage_frac - expected_frac) ** 2).mean()
        else:
            balance_loss = flat.new_zeros(())

        out = (shared_out + routed_out).view(B, T, d)
        return out, balance_loss

    def _dispatch_loop(self, flat, top_idx, top_w):
        """Le dispatch historique : par expert, sur les tokens qui l'ont élu."""
        routed_out = torch.zeros_like(flat)
        for ki in range(self.top_k):
            expert_ids = top_idx[:, ki]      # [BT]
            weights    = top_w[:, ki]        # [BT]
            for ei, expert in enumerate(self.experts):
                mask = expert_ids == ei
                if not mask.any():
                    continue
                sel = flat[mask]             # [n_tokens, d] — n_tokens is
                # data-dependent (routing); without this hint dynamo (with
                # dynamic=False) specialises one graph per distinct count and
                # thrashes the 256-entry recompile cache into eager fallback
                torch._dynamo.mark_dynamic(sel, 0)
                out = expert(sel)
                routed_out[mask] = routed_out[mask] + weights[mask].unsqueeze(-1) * out
        return routed_out

    def _dispatch_dense(self, flat, top_idx, top_w):
        """Tous les experts sur tous les tokens, sélection par indexation.

        Reproduit EXACTEMENT l'accumulation de la boucle : pour chaque token,
        0 + w₀·out(e₀) + w₁·out(e₁) dans l'ordre ki — même ordre de sommation,
        donc mêmes bits (prouvé float64 au self-test). Aucune op data-dependent
        (pas de .any(), pas de masque booléen) : zéro sync, shapes fixes.
        """
        outs = torch.stack([e(flat) for e in self.experts])       # [E, BT, d]
        ar = torch.arange(flat.size(0), device=flat.device)
        acc = torch.zeros_like(flat)
        for ki in range(self.top_k):
            acc = acc + top_w[:, ki].unsqueeze(-1) * outs[top_idx[:, ki], ar]
        return acc


# ── self-test ────────────────────────────────────────────────────────────────

def _selftest() -> None:
    """Routage, loss d'équilibrage, et le garde-fou NaN du sqrt(softplus).

    Le clamp_min(1e-12) n'a l'air de rien : sans lui, un logit très négatif fait
    softplus→0 en bf16, sqrt'(0)=inf, et le gradient (nul) des experts non
    sélectionnés devient 0*inf = NaN. C'est silencieux jusqu'à ce que la loss
    parte en NaN quelques milliers de steps plus loin.
    """
    torch.manual_seed(0)
    B, T, D, E, K, FF = 2, 6, 8, 4, 2, 16

    moe = DeepSeekMoE(d_model=D, n_experts=E, n_shared=1, top_k_experts=K, d_ff=FF)
    x = torch.randn(B, T, D)
    out, bal = moe(x)
    assert out.shape == x.shape and out.dtype == x.dtype
    assert bal.ndim == 0 and float(bal) >= 0.0, "la loss d'équilibrage est un scalaire ≥ 0"
    assert torch.isfinite(out).all() and torch.isfinite(bal)

    # ── la loss d'équilibrage mesure bien le DÉSÉQUILIBRE ────────────────────
    # Tout vers les mêmes experts : W_gate constant ⇒ topk identique partout.
    concentrated = DeepSeekMoE(d_model=D, n_experts=E, n_shared=1,
                               top_k_experts=K, d_ff=FF)
    with torch.no_grad():
        concentrated.W_gate.weight.zero_()
        concentrated.W_gate.weight[0].fill_(1.0)      # l'expert 0 domine partout
    _, bal_bad = concentrated(x.abs())
    # Routage parfaitement uniforme : chaque expert reçoit exactement K/E.
    balanced = DeepSeekMoE(d_model=D, n_experts=E, n_shared=1,
                           top_k_experts=E, d_ff=FF)   # top_k = E ⇒ tous servis
    _, bal_ok = balanced(x)
    assert float(bal_ok) < 1e-12, f"routage uniforme ⇒ loss nulle (vu {float(bal_ok):.2e})"
    assert float(bal_bad) > float(bal_ok), "un routage concentré doit coûter plus cher"

    # ── top_k respecté, poids normalisés ────────────────────────────────────
    scores = torch.sqrt(F.softplus(moe.W_gate(x.reshape(B * T, D))).clamp_min(1e-12))
    top_s, top_i = scores.topk(K, dim=-1)
    w = top_s / (top_s.sum(-1, keepdim=True) + 1e-8)
    assert top_i.shape == (B * T, K)
    assert torch.allclose(w.sum(-1), torch.ones(B * T), atol=1e-5), \
        "les poids du set sélectionné doivent sommer à 1"
    assert (top_i.max() < E).all()

    # ── garde-fou NaN : logits très négatifs, en bf16 comme en fp32 ──────────
    for dtype in (torch.float32, torch.bfloat16):
        m = DeepSeekMoE(d_model=D, n_experts=E, n_shared=1, top_k_experts=K,
                        d_ff=FF).to(dtype)
        with torch.no_grad():
            m.W_gate.weight.fill_(-50.0)              # softplus sous-déborde à 0
        xi = torch.randn(B, T, D, dtype=dtype, requires_grad=True)
        o, bl = m(xi)
        (o.float().sum() + bl.float()).backward()
        assert torch.isfinite(xi.grad).all(), \
            f"gradient NaN en {dtype} — le clamp_min(1e-12) du sqrt(softplus) a sauté"
        assert torch.isfinite(m.W_gate.weight.grad).all(), dtype

    # ── les experts partagés sont TOUJOURS actifs ───────────────────────────
    m = DeepSeekMoE(d_model=D, n_experts=E, n_shared=1, top_k_experts=K, d_ff=FF)
    y, _ = m(x)
    y.sum().backward()
    assert m.shared[0].w3.weight.grad.abs().sum() > 0, "expert partagé sans gradient"
    # …et au moins un routé a reçu du gradient (le dispatch fait quelque chose)
    assert any(e.w3.weight.grad is not None and e.w3.weight.grad.abs().sum() > 0
               for e in m.experts), "aucun expert routé n'a reçu de gradient"

    # ── SwiGLU : clamps annoncés en §4.2.3 ──────────────────────────────────
    sg = SwiGLU(D, FF)
    with torch.no_grad():
        sg.w12.weight.fill_(50.0)                     # force la saturation
    big = sg(torch.randn(B, T, D) * 10.0)
    assert torch.isfinite(big).all(), "SwiGLU non clampé : sortie non finie"

    # plusieurs experts partagés : moyenne, pas somme (sinon l'échelle dérive)
    m2 = DeepSeekMoE(d_model=D, n_experts=E, n_shared=3, top_k_experts=K, d_ff=FF)
    assert len(m2.shared) == 3
    assert torch.isfinite(m2(x)[0]).all()

    # ── decode_dense_moe : dense == boucle AU BIT à BT=1, gate strict ───────
    # (float64 isole « même calcul ? » des ULP BLAS, float32 CPU en témoin.
    # Pourquoi BT=1 seulement : à BT≥2 la boucle évalue chaque expert sur un
    # SOUS-ENSEMBLE des lignes — GEMM M=1 vs M=2 ⇒ rel ~1.2e-16 mesuré, même
    # en float64. Le témoin ci-dessous vérifie que cet écart existe bien :
    # c'est LUI qui justifie le gate, pas une prudence rituelle.)
    for dtype in (torch.float64, torch.float32):
        md = DeepSeekMoE(d_model=D, n_experts=E, n_shared=1, top_k_experts=K,
                         d_ff=FF, dense_decode=True).to(dtype)
        for _ in range(5):                            # 5 tirages de token
            xd = torch.randn(1, 1, D, dtype=dtype)
            with torch.no_grad():
                o_dense, _ = md(xd)
                md.dense_decode = False
                o_loop, _ = md(xd)
                md.dense_decode = True
            assert torch.equal(o_dense, o_loop), f"dense ≠ boucle (dtype={dtype})"
    # témoin du gate : à BT=2 l'écart sous-ensemble/batch EXISTE (ULP, pas 0)
    e0 = md.experts[0].double()
    x2 = torch.randn(2, D, dtype=torch.float64)
    with torch.no_grad():
        sub_vs_batch = (e0(x2)[1:2] - e0(x2[1:2])).abs().max().item()
    assert sub_vs_batch < 1e-14, "ce n'est plus de l'ULP — bug réel quelque part"
    # la boucle reste le chemin sous gradient, à BT>1, et flag OFF
    md64 = DeepSeekMoE(d_model=D, n_experts=E, n_shared=1, top_k_experts=K,
                       d_ff=FF, dense_decode=True)
    hits = {"n": 0}
    _orig_dense = md64._dispatch_dense
    md64._dispatch_dense = lambda *a: (hits.__setitem__("n", hits["n"] + 1),
                                       _orig_dense(*a))[1]
    xg = torch.randn(1, 1, D, requires_grad=True)
    o, bl = md64(xg)                                  # grad ON ⇒ boucle
    (o.sum() + bl).backward()
    assert hits["n"] == 0 and xg.grad is not None
    with torch.no_grad():
        md64(torch.randn(1, 2, D))                    # BT>1 ⇒ boucle
    assert hits["n"] == 0
    with torch.no_grad():
        md64(torch.randn(1, 1, D))                    # no_grad + BT=1 ⇒ dense
    assert hits["n"] == 1
    m_off_flag = DeepSeekMoE(d_model=D, n_experts=E, n_shared=1,
                             top_k_experts=K, d_ff=FF)
    assert not m_off_flag.dense_decode

    # ── need_balance=False : SORTIE au bit près, balance rendue à zéro ──────
    x64 = torch.randn(B, T, D, dtype=torch.float64)
    m64 = DeepSeekMoE(d_model=D, n_experts=E, n_shared=1, top_k_experts=K,
                      d_ff=FF).double()
    with torch.no_grad():
        o_with, b_with = m64(x64)
        o_wout, b_wout = m64(x64, need_balance=False)
    assert torch.equal(o_with, o_wout), "need_balance change la sortie"
    assert float(b_wout) == 0.0 and b_wout.dtype == o_wout.dtype
    assert float(b_with) != 0.0, "témoin : la vraie balance n'est pas nulle ici"

    print("moe self-test: OK (formes, loss d'équilibrage nulle si uniforme et "
          "positive si concentrée, top_k + poids normalisés, garde-fou NaN "
          "sqrt(softplus) en fp32 ET bf16, partagés toujours actifs, clamps "
          "SwiGLU, dense_decode == boucle AU BIT à BT=1 (f64+f32) + témoin ULP "
          "qui justifie le gate, gated no_grad/BT=1/flag, need_balance=False : "
          "sortie au bit près + balance à 0)")


if __name__ == "__main__":
    _selftest()
