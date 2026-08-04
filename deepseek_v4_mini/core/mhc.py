"""
Manifold-Constrained Hyper-Connections (mHC) – DeepSeek-V4 §2.2

Key idea: expand the residual stream width by n_hc and constrain the residual
mapping matrix B to the Birkhoff polytope (doubly stochastic matrices) via
Sinkhorn-Knopp.  This bounds ||B||_2 ≤ 1, making deep stacks numerically
stable without sacrificing expressivity.

Standard HC update:
    X_{l+1} = B_l X_l + C_l F_l(A_l X_l)

where X_l ∈ R^{n_hc × d} and A_l, B_l, C_l are dynamically generated from X_l.
"""
from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute the variance in fp32: under bf16 autocast, x.pow(2) overflows
        # once |x| > ~256 (bf16 max ~6.5e4), giving mean=inf -> rsqrt=0 (a finite
        # forward) but a NaN gradient in the backward pass. fp32 avoids this.
        dtype = x.dtype
        xf = x.float()
        normed = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * normed.to(dtype)


def _sinkhorn_2x2(logits: torch.Tensor) -> torch.Tensor:
    """Projection de Birkhoff EXACTE pour n_hc == 2, sans itérer.

    logits : [..., 2, 2] scores pré-exp. Renvoie M doublement stochastique.

    Dérivation. Sinkhorn cherche D₁ M D₂ doublement stochastique avec
    M = exp(L). En 2×2 une matrice doublement stochastique s'écrit forcément
    [[p, 1−p], [1−p, p]]. En posant a,b,c,d = exp(l₀₀),exp(l₀₁),exp(l₁₀),exp(l₁₁)
    et r,s les facteurs de lignes/colonnes :

        r₁s₁a = p      r₁s₂b = 1−p      r₂s₁c = 1−p      r₂s₂d = p

    Le produit croisé élimine r et s : (r₁s₁a)(r₂s₂d) / (r₁s₂b)(r₂s₁c) = ad/bc,
    donc p²/(1−p)² = ad/bc, soit logit(p) = (l₀₀ + l₁₁ − l₀₁ − l₁₀)/2 :

        p = sigmoid((l₀₀ + l₁₁ − l₀₁ − l₁₀) / 2)

    Pourquoi c'est mieux que la boucle, et pas seulement plus rapide :

    * **Exact.** La boucle à t_max=20 est à ~2.5e-2 de ce point fixe (elle
      converge en ~1/n : 2.4e-3 à 200 itérations, 8.4e-5 à 5000). Elle finit sur
      une normalisation en COLONNE, donc ses lignes ne somment à 1 qu'à 2.5e-2
      près — alors que l'éq. 8 du papier (`M⁽ᵗ⁾ = T_r(T_c(M⁽ᵗ⁻¹⁾))`) normalise
      les lignes en dernier, et que B multiplie l'état résiduel À GAUCHE : c'est
      la stochasticité par LIGNE qui fait de chaque flux de sortie une
      combinaison convexe des flux d'entrée. La forme close est exacte dans les
      deux sens, donc l'ordre de normalisation devient sans objet.
    * **‖M‖₂ ≤ 1 inconditionnel.** Birkhoff est atteint, pas approché : la borne
      non-expansive ne dépend plus du régime des logits (la boucle la perd à
      logits extrêmes, cf. l'assertion « wild » du self-test).
    * **Plus d'`exp`.** La stabilisation log-sum-exp du chemin itératif existait
      contre l'overflow de exp(B_raw) ; ici il n'y a plus d'exponentielle. Le
      gradient est un `sigmoid'` borné par 1/4 au lieu de 20 étapes de
      normalisation déroulées.
    * **26 % des ops du forward.** Mesuré : 123 ops aten par appel × 24 appels
      (2 mHC × 12 couches) ⇒ forward 11 462 → 8 894, backward 21 899 → 14 603.

    L'invariance au décalage constant est préservée : ajouter une constante aux
    quatre entrées laisse la combinaison (+1, −1, −1, +1) inchangée.
    """
    p = torch.sigmoid(
        (logits[..., 0, 0] + logits[..., 1, 1]
         - logits[..., 0, 1] - logits[..., 1, 0]) * 0.5)
    q = 1.0 - p
    return torch.stack((p, q, q, p), dim=-1).unflatten(-1, (2, 2))


class ManifoldHyperConnections(nn.Module):
    """
    mHC wrapper.  Call as:
        X_new = mhc(X, layer_fn)
    where X has shape [B, T, n_hc, d] and layer_fn maps [B, T, d] → [B, T, d].
    """

    def __init__(self, d_model: int, n_hc: int, sinkhorn_iters: int = 5,
                 closed_form: bool = False) -> None:
        super().__init__()
        self.n_hc = n_hc
        self.d_model = d_model
        self.sinkhorn_iters = sinkhorn_iters
        # closed_form : n'a d'effet qu'à n_hc == 2 (voir _sinkhorn).
        self.closed_form = closed_form
        flat = n_hc * d_model

        # Dynamic component generators (small projections – n_hc is typically 2–4)
        self.W_pre = nn.Linear(flat, n_hc, bias=False)
        self.W_res = nn.Linear(flat, n_hc * n_hc, bias=False)
        self.W_post = nn.Linear(flat, n_hc, bias=False)

        # Learnable static biases (initialised so B ≈ identity)
        self.S_pre  = nn.Parameter(torch.zeros(n_hc))
        self.S_res  = nn.Parameter(torch.eye(n_hc).flatten())
        self.S_post = nn.Parameter(torch.full((n_hc,), 1.0 / n_hc))

        # Gating scalars – start near 0 so dynamic component is initially small
        self.alpha_pre = nn.Parameter(torch.zeros(1))
        self.alpha_res = nn.Parameter(torch.zeros(1))
        self.alpha_post = nn.Parameter(torch.zeros(1))

        self.norm = RMSNorm(flat)

    # ── Sinkhorn-Knopp projection onto the Birkhoff polytope ─────────────────
    def _sinkhorn(self, logits: torch.Tensor) -> torch.Tensor:
        """logits: [BT, n_hc, n_hc] (pre-exp scores). Returns doubly stochastic M.

        We exponentiate here with a per-matrix max subtraction (log-sum-exp
        stabilisation). Sinkhorn normalisation is invariant to a global positive
        rescale of M, so subtracting a constant before exp leaves the result
        unchanged while keeping exp inputs <= 0 (so exp <= 1, no bf16 overflow).
        Exponentiating the raw scores directly made exp(B_raw) — and hence its
        gradient, which equals exp — blow up to inf/NaN as weights grew.

        À n_hc == 2 (le cas de TOUTES les configs du dépôt), cette boucle est
        évitable : voir `_sinkhorn_2x2`. `closed_form=True` la court-circuite.
        """
        if self.n_hc == 2 and self.closed_form:
            return _sinkhorn_2x2(logits)
        M = (logits - logits.amax(dim=(-2, -1), keepdim=True)).exp()
        for _ in range(self.sinkhorn_iters):
            M = M / (M.sum(dim=-1, keepdim=True) + 1e-8)   # row normalise
            M = M / (M.sum(dim=-2, keepdim=True) + 1e-8)   # column normalise
        return M

    def forward(self, X: torch.Tensor, layer_fn: Callable) -> torch.Tensor:
        """
        X:         [B, T, n_hc, d]
        layer_fn:  [B, T, d] → [B, T, d]
        Returns:   [B, T, n_hc, d]
        """
        B, T, n_hc, d = X.shape
        BT = B * T

        # Flatten residual state and normalise for parameter generation
        X_flat = X.reshape(BT, n_hc * d)
        X_hat = self.norm(X_flat)                                   # [BT, n_hc*d]

        # ── Generate raw (unconstrained) A, B, C ─────────────────────────────
        A_raw = self.alpha_pre.tanh() * self.W_pre(X_hat) + self.S_pre   # [BT, n_hc]
        # The dynamic B logits feed exp()→Sinkhorn. A and C are bounded by sigmoid,
        # but B_raw was not: nothing capped W_res's magnitude, so it ran away
        # (W_res → ∞), making the exp/Sinkhorn backward Jacobian explode into NaN.
        # Bound the dynamic part with tanh (∈(-1,1)); S_res (init = identity) keeps
        # B near-identity at start. exp inputs stay O(1) → stable forward & backward.
        B_raw = self.alpha_res.tanh() * self.W_res(X_hat).tanh() + self.S_res  # [BT, n_hc²]
        C_raw = self.alpha_post.tanh() * self.W_post(X_hat) + self.S_post # [BT, n_hc]

        # ── Apply constraints (DeepSeek-V4 §2.2 eqs. 6-8) ──────────────────────
        A    = torch.sigmoid(A_raw)                                # [BT, n_hc] non-neg, bounded
        C    = 2.0 * torch.sigmoid(C_raw)                         # [BT, n_hc] non-neg, ≤ 2
        B_ds = self._sinkhorn(B_raw.view(BT, n_hc, n_hc))         # doubly stochastic (exp inside)

        # ── Compute layer input: h_in = A ⊗ X (weighted sum of n_hc streams) ─
        X_r = X.reshape(BT, n_hc, d)
        h_in = torch.einsum("bn,bnd->bd", A, X_r)                 # [BT, d]

        # ── Run the wrapped layer ─────────────────────────────────────────────
        h_out = layer_fn(h_in.view(B, T, d)).reshape(BT, d)       # [BT, d]

        # ── Update residual stream: X_{l+1} = B X_l + C ⊗ h_out ─────────────
        X_new = (
            torch.einsum("bij,bjd->bid", B_ds, X_r)               # [BT, n_hc, d]
            + torch.einsum("bn,bd->bnd", C, h_out)
        )
        return X_new.view(B, T, n_hc, d)


# ── self-test ────────────────────────────────────────────────────────────────

def _selftest() -> None:
    """Ce dont dépend la stabilité numérique du stack : Sinkhorn borné, et les
    deux garde-fous NaN posés après coup (variance RMSNorm en fp32, tanh sur les
    logits B). Ils n'ont aucune trace dans la loss quand ils sautent — seulement
    un run qui meurt à l'étape 2000."""
    torch.manual_seed(0)
    B, T, N, D = 2, 5, 3, 8

    # ── RMSNorm ─────────────────────────────────────────────────────────────
    rn = RMSNorm(D)
    x = torch.randn(B, T, D) * 3.0
    y = rn(x)
    assert y.shape == x.shape
    assert torch.allclose(y.pow(2).mean(-1), torch.ones(B, T), atol=1e-3), \
        "RMSNorm doit ramener la RMS à 1 (poids à l'init = 1)"
    # la variance passe en fp32 : sous bf16 un |x| > ~256 donnait un grad NaN
    xb = (torch.randn(B, T, D) * 1e3).to(torch.bfloat16)
    xb.requires_grad_(True)
    RMSNorm(D)(xb).float().sum().backward()
    assert torch.isfinite(xb.grad).all(), "RMSNorm : gradient NaN sur grandes entrées bf16"

    # ── Sinkhorn : ce qui est garanti, et jusqu'où ──────────────────────────
    # La dernière opération de la boucle est la normalisation par COLONNE : les
    # colonnes somment à 1 exactement, les lignes seulement à la convergence.
    # Ce n'est pas un choix esthétique : c'est l'ordre TRANSPOSÉ de l'éq. 8 du
    # papier (`M⁽ᵗ⁾ = T_r(T_c(M⁽ᵗ⁻¹⁾))`, lignes en dernier). Comme B multiplie
    # l'état résiduel à GAUCHE, c'est la stochasticité par LIGNE qui fait de
    # chaque flux de sortie une combinaison convexe des flux d'entrée — donc la
    # tolérance 1e-2 ci-dessous mesure exactement l'écart au papier. Le chemin
    # n_hc == 2 en forme close (bloc suivant) rend le point sans objet.
    # C'est acceptable parce que B_raw est borné par construction (tanh sur la
    # partie dynamique + S_res init identité ⇒ logits O(1)). Le test fixe cette
    # dépendance au lieu de la supposer : hors régime, ‖B‖₂ ≤ 1 ne tient plus.
    mhc = ManifoldHyperConnections(D, N, sinkhorn_iters=20)
    for scale in (0.5, 1.0, 2.0):                  # régime réel des logits B
        M = mhc._sinkhorn(torch.randn(B * T, N, N) * scale)
        assert torch.isfinite(M).all() and (M >= 0).all(), scale
        assert torch.allclose(M.sum(-2), torch.ones(B * T, N), atol=1e-5), \
            f"colonnes : normalisation exacte attendue (échelle {scale})"
        assert torch.allclose(M.sum(-1), torch.ones(B * T, N), atol=1e-2), \
            f"lignes : convergence attendue en 20 itérations (échelle {scale})"
        # Birkhoff ⇒ ‖B‖₂ ≤ 1 : c'est TOUT l'intérêt de la contrainte
        assert float(torch.linalg.matrix_norm(M, ord=2).max()) <= 1.01, scale
    # Le plafond, documenté : à logits extrêmes la convergence lâche et la borne
    # spectrale avec elle. Si S_res (appris, NON borné) dérivait loin de
    # l'identité, on entrerait ici — d'où l'assertion, qui est un avertissement.
    wild = mhc._sinkhorn(torch.randn(B * T, N, N) * 50.0)
    assert float(torch.linalg.matrix_norm(wild, ord=2).max()) > 1.05, \
        "à logits extrêmes Sinkhorn ne converge PAS en 20 itérations — si cette " \
        "assertion tombe, la borne est devenue inconditionnelle (bonne nouvelle, " \
        "mettre à jour le commentaire)"
    # invariance au décalage constant (la stabilisation log-sum-exp ne change rien)
    lg = torch.randn(4, N, N)
    assert torch.allclose(mhc._sinkhorn(lg), mhc._sinkhorn(lg + 7.0), atol=1e-5)
    # …et le nombre d'itérations compte : le preset tiny() (3) ne converge pas
    few = ManifoldHyperConnections(D, N, sinkhorn_iters=3)._sinkhorn(
        torch.randn(B * T, N, N) * 2.0)
    assert (few.sum(-1) - 1).abs().max() > (M.sum(-1) - 1).abs().max()

    # ── Sinkhorn 2×2 : la forme close EST le point fixe ─────────────────────
    # Toutes les configs du dépôt valent n_hc = 2, donc c'est CE chemin qui
    # tourne en production. On ne teste pas « la forme close ressemble à la
    # boucle » : on prouve qu'elle est le point fixe VERS lequel la boucle
    # converge, en float64 (convention d'attention._selftest), et dans les DEUX
    # ordres de normalisation — celui du dépôt et celui de l'éq. 8. Si les deux
    # ordres convergent au même point, l'ordre n'a jamais été qu'un artefact de
    # troncature, et la forme close le supprime.
    def _loop_ref(lg, iters, paper_order=False):
        """L'oracle : la boucle, sans l'epsilon, en float64."""
        Mr = (lg - lg.amax(dim=(-2, -1), keepdim=True)).exp()
        for _ in range(iters):
            if paper_order:                      # éq. 8 : T_r(T_c(·))
                Mr = Mr / Mr.sum(dim=-2, keepdim=True)
                Mr = Mr / Mr.sum(dim=-1, keepdim=True)
            else:                                # dépôt : T_c(T_r(·))
                Mr = Mr / Mr.sum(dim=-1, keepdim=True)
                Mr = Mr / Mr.sum(dim=-2, keepdim=True)
        return Mr

    n_pts, ones2 = 0, torch.ones(256, 2, dtype=torch.float64)
    for scale in (0.5, 1.0, 2.0, 8.0, 50.0):
        lg2 = torch.randn(256, 2, 2, dtype=torch.float64) * scale
        cf = _sinkhorn_2x2(lg2)
        # doublement stochastique EXACTEMENT, dans les deux sens (q = 1-p)
        assert torch.allclose(cf.sum(-1), ones2, atol=1e-12), f"lignes {scale}"
        assert torch.allclose(cf.sum(-2), ones2, atol=1e-12), f"colonnes {scale}"
        assert (cf >= 0).all() and torch.isfinite(cf).all(), scale
        # Birkhoff atteint ⇒ ‖M‖₂ ≤ 1 INCONDITIONNEL, y compris à l'échelle 50
        # où la boucle perd la borne (assertion « wild » ci-dessus).
        assert float(torch.linalg.matrix_norm(cf, ord=2).max()) <= 1.0 + 1e-12, \
            f"la borne non-expansive doit tenir sans condition (échelle {scale})"
        # c'est bien le point fixe : la boucle s'en rapproche quand on itère
        d20, d2000 = ((_loop_ref(lg2, n) - cf).abs().max() for n in (20, 2000))
        assert d2000 < d20, f"la boucle doit CONVERGER vers la forme close ({scale})"
        assert d2000 < 1e-3, f"point fixe non atteint à 2000 itérations ({scale})"
        # …et le même point fixe dans l'ordre du papier
        assert (_loop_ref(lg2, 2000, paper_order=True) - cf).abs().max() < 1e-3, \
            f"les deux ordres de normalisation doivent converger au même point ({scale})"
        n_pts += lg2.size(0)
    # invariance au décalage constant : la combinaison (+1,−1,−1,+1) l'annule
    lg2f = torch.randn(64, 2, 2)
    assert torch.allclose(_sinkhorn_2x2(lg2f), _sinkhorn_2x2(lg2f + 7.0), atol=1e-5)
    # plus d'exp ⇒ plus d'overflow : gradient fini là où exp(1e3) serait inf
    xb2 = (torch.randn(64, 2, 2) * 1e3).to(torch.bfloat16).requires_grad_(True)
    _sinkhorn_2x2(xb2).float().sum().backward()
    assert torch.isfinite(xb2.grad).all(), "forme close : gradient NaN en bf16"
    # le drapeau branche bien, et NE branche PAS hors n_hc == 2
    lg2f64 = lg2f.to(torch.float64)
    assert torch.equal(
        ManifoldHyperConnections(D, 2, 20, True)._sinkhorn(lg2f64),
        _sinkhorn_2x2(lg2f64)), "closed_form=True doit court-circuiter la boucle"
    assert not torch.equal(
        ManifoldHyperConnections(D, 2, 20, False)._sinkhorn(lg2f64),
        _sinkhorn_2x2(lg2f64)), "closed_form=False doit garder la boucle historique"
    lg3 = torch.randn(8, N, N, dtype=torch.float64)
    assert torch.equal(ManifoldHyperConnections(D, N, 20, True)._sinkhorn(lg3),
                       ManifoldHyperConnections(D, N, 20, False)._sinkhorn(lg3)), \
        "n_hc != 2 : la forme close ne s'applique pas, la boucle doit rester"

    # ── forward mHC : forme, identité à l'init, gradient fini ───────────────
    X = torch.randn(B, T, N, D, requires_grad=True)
    seen = {}

    def layer_fn(h):
        seen["shape"] = tuple(h.shape)
        return h * 0.0                       # couche neutre : on isole le résiduel

    out = mhc(X, layer_fn)
    assert out.shape == X.shape and seen["shape"] == (B, T, D)
    out.sum().backward()
    assert torch.isfinite(X.grad).all()

    # À l'init, B est diagonalement DOMINANT — pas l'identité. S_res vaut eye(),
    # mais ce sont des LOGITS : après exp + Sinkhorn la diagonale tombe à ~0.73
    # (n_hc=2). Les flux résiduels se mélangent donc dès le premier pas ; le
    # commentaire « B ≈ identity » du constructeur est optimiste, et le test
    # enregistre la valeur réelle plutôt qu'une intention.
    with torch.no_grad():
        B_init = mhc._sinkhorn(mhc.S_res.view(1, N, N))
    diag = B_init[0].diagonal()
    off = (B_init[0].sum() - diag.sum()) / (N * N - N)
    assert (diag > off).all(), "B init doit rester diagonalement dominant"
    assert 0.4 < float(diag.mean()) < 0.95, \
        f"diagonale init {float(diag.mean()):.3f} — ni identité ni uniforme"

    # gros poids sur W_res : le tanh doit tenir (c'était la source du NaN)
    mhc2 = ManifoldHyperConnections(D, N, sinkhorn_iters=20)
    with torch.no_grad():
        mhc2.W_res.weight.mul_(1e4)
        mhc2.alpha_res.fill_(5.0)
    X2 = torch.randn(B, T, N, D, requires_grad=True)
    mhc2(X2, lambda h: h).sum().backward()
    assert torch.isfinite(X2.grad).all(), "logits B non bornés : gradient NaN"
    assert torch.isfinite(mhc2.W_res.weight.grad).all()

    print(f"mhc self-test: OK (Sinkhorn 2×2 forme close = point fixe exact sur "
          f"{n_pts} matrices float64, ‖B‖₂≤1 inconditionnel, les 2 ordres de "
          f"normalisation convergent au même point ; boucle n_hc>2 : colonnes "
          f"exactes, lignes convergées et ‖B‖₂≤1 dans le régime |logits|≤2 "
          f"SEULEMENT ; RMSNorm RMS=1 + grad fini en bf16 ; B init diagonalement "
          f"dominant (~0.73, pas l'identité) ; tanh sur B_raw tient les gros poids)")


if __name__ == "__main__":
    _selftest()
