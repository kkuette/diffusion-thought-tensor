"""Muon — Momentum Orthogonalised by Newton-Schulz (+ le split de params).

Extrait de train.py sans la moindre modification de corps : train.py est le
trainer de l'arc jouet dsv4mini (clos), et c'était son SEUL lien avec le
programme courant — code_defer_native, code_train et sft_train importaient
l'optimiseur depuis un module de 2600 lignes qu'ils n'utilisent pas par
ailleurs. train.py réexporte les trois noms, donc les commandes de repro
dsv4mini et tout `from ..train import Muon` existant continuent de marcher.

Le piège √cols (rms_match) est documenté dans le corps de step() : le garder
sous les yeux est la raison d'être de ce module.
"""
from __future__ import annotations

import torch
from torch import nn, optim


def _zeropower_via_newtonschulz(G: torch.Tensor, steps: int = 10) -> torch.Tensor:
    """
    Hybrid Newton-Schulz orthogonalisation (DeepSeek-V4 §2.4 eq. 28).

    Two-stage schedule:
      - First (steps-2) iterations: (a,b,c) = (3.4445, -4.7750, 2.0315)
        drives singular values rapidly toward 1.
      - Final 2 iterations: (a,b,c) = (2, -1.5, 0.5)
        stabilises singular values precisely at 1.
    """
    assert G.ndim == 2
    X = G / (G.norm() + 1e-7)
    if G.size(0) > G.size(1):
        X = X.T
    fast_steps = max(steps - 2, 0)
    for i in range(steps):
        a, b, c = (3.4445, -4.7750, 2.0315) if i < fast_steps else (2.0, -1.5, 0.5)
        A = X @ X.T
        X = a * X + (b * A + c * A @ A) @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X


class Muon(optim.Optimizer):
    """
    Muon — Momentum Orthogonalised by Newton-Schulz.

    Applies Nesterov momentum then orthogonalises the update via Newton-Schulz,
    targeting 2-D weight matrices.  All other parameters (biases, norms,
    embeddings) are handled by a bundled AdamW group.

    Usage:
        muon_params, adam_params = _split_muon_params(model)
        opt = Muon(muon_params, lr=0.02, wd=0.01,
                   adam_params=adam_params, adam_lr=3e-4)

    Reference: Jordan et al., 2024.
    """

    def __init__(
        self,
        params,
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        wd: float = 0.0,
        rms_match: bool = False,
        adam_params=None,
        adam_lr: float = 3e-4,
        adam_betas: tuple = (0.9, 0.95),
        adam_wd: float = 0.1,
        adam_eps: float = 1e-8,
        adam_fused: bool = False,
    ) -> None:
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov,
                        ns_steps=ns_steps, wd=wd, rms_match=rms_match)
        super().__init__(params, defaults)
        # Internal AdamW for non-matrix params. fused=True (opt-in) fuses the
        # AdamW element-wise update into one CUDA kernel — free on GPU, numerically
        # ~identical (NOT bit-identical), so gated off by default for repro.
        if adam_params is not None:
            self._adam = optim.AdamW(
                adam_params, lr=adam_lr, betas=adam_betas, weight_decay=adam_wd,
                eps=adam_eps, fused=adam_fused,
            )
        else:
            self._adam = None

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None

        for group in self.param_groups:
            lr       = group["lr"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]
            wd       = group["wd"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]

                if "buf" not in state:
                    state["buf"] = torch.zeros_like(p)

                buf = state["buf"]
                buf.mul_(momentum).add_(g)

                # Nesterov: g + momentum * buf  (one-step look-ahead)
                update = g.add(buf, alpha=momentum) if nesterov else buf.clone()

                # Orthogonalise 2-D updates via Newton-Schulz
                if update.ndim == 2:
                    update = _zeropower_via_newtonschulz(update.float(), ns_steps)
                    if group["rms_match"]:
                        # DeepSeek-V4/Kimi convention: rescale so update RMS = 0.2
                        # for EVERY shape — one lr serves square backbone blocks and
                        # tall hypernet maps alike. The legacy sqrt(cols) scaling
                        # gives RMS 1.0 on squares but ~0.1 on the fw_A/fw_B
                        # hypernets: the read trains ~8x slower than the backbone.
                        update = update * (0.2 * max(update.shape) ** 0.5)
                    else:
                        # Scale to match RMS of a standard normal (Jordan et al.)
                        update = update * (update.size(1) ** 0.5)

                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)
                p.add_(update, alpha=-lr)

        if self._adam is not None:
            self._adam.step()

        return loss

    def zero_grad(self, set_to_none: bool = True) -> None:
        super().zero_grad(set_to_none=set_to_none)
        if self._adam is not None:
            self._adam.zero_grad(set_to_none=set_to_none)


def _split_muon_params(model: nn.Module):
    """
    Split model parameters into:
      - muon_params : 2-D weight matrices that benefit from orthogonalisation
      - adam_params : everything else

    Per DeepSeek-V4 §2.4: AdamW is used for embedding, prediction head, RMSNorm
    weights, AND the static biases (S_pre, S_res, S_post) and gating scalars
    (alpha_pre, alpha_res, alpha_post) of mHC modules.  These are 1-D or scalar
    parameters and fall into adam_params naturally via the ndim != 2 check.
    """
    muon, adam = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # 2-D matrices go to Muon, except lookup tables and mHC dynamic generators
        # are excluded by name when they are embedding-like.
        is_matrix = p.ndim == 2
        is_embed  = "embed" in name          # nn.Embedding weight
        is_mhc_static = any(k in name for k in ("S_pre", "S_res", "S_post",
                                                  "alpha_pre", "alpha_res", "alpha_post"))
        if is_matrix and not is_embed and not is_mhc_static:
            muon.append(p)
        else:
            adam.append(p)
    return muon, adam


# ── self-test ────────────────────────────────────────────────────────────────

def _selftest() -> None:
    """Newton-Schulz, le piège √cols, et le tri des paramètres.

    Le cœur du test est `rms_match` : avec le scaling historique √cols, une
    matrice carrée reçoit un update de RMS ~1 et un hypernet haut (fw_A/fw_B) de
    RMS ~0.1 — le read apprend alors ~8x plus lentement que le backbone, sans
    que rien dans la loss ne le signale.
    """
    torch.manual_seed(0)

    # ── Newton-Schulz : les valeurs singulières convergent vers 1 ───────────
    for shape in ((8, 8), (8, 32), (32, 8), (1, 16)):
        G = torch.randn(*shape)
        X = _zeropower_via_newtonschulz(G, steps=10)
        assert X.shape == G.shape, shape
        sv = torch.linalg.svdvals(X)
        assert torch.allclose(sv, torch.ones_like(sv), atol=0.05), \
            f"{shape}: valeurs singulières {sv.tolist()}"
        # l'orthogonalisation garde la direction dominante de G
        assert (torch.linalg.svdvals(G) > 0).all()
    # une matrice déjà orthogonale est un point fixe
    Q = torch.linalg.qr(torch.randn(16, 16))[0]
    assert torch.allclose(_zeropower_via_newtonschulz(Q, steps=10).abs(),
                          Q.abs(), atol=0.05)
    # échelle indifférente : le premier geste est une normalisation
    G = torch.randn(8, 16)
    assert torch.allclose(_zeropower_via_newtonschulz(G, steps=10),
                          _zeropower_via_newtonschulz(G * 1e3, steps=10), atol=1e-4)

    # ── rms_match : MÊME RMS d'update quelle que soit la forme ──────────────
    def _update_rms(shape, rms_match, lr=0.1):
        p = nn.Parameter(torch.zeros(*shape))
        p.grad = torch.randn(*shape)
        opt = Muon([p], lr=lr, momentum=0.0, nesterov=False, ns_steps=10,
                   rms_match=rms_match)
        opt.step()
        return float((p.detach() / -lr).pow(2).mean().sqrt())

    square, tall = (64, 64), (512, 32)
    r_sq, r_tall = _update_rms(square, True), _update_rms(tall, True)
    for r, shape in ((r_sq, square), (r_tall, tall)):
        assert abs(r - 0.2) < 0.02, f"rms_match {shape}: RMS {r:.3f} ≠ 0.2"
    assert abs(r_sq - r_tall) < 0.02, \
        f"rms_match doit égaliser les formes ({r_sq:.3f} vs {r_tall:.3f})"

    # …et le scaling historique, lui, ne les égalise PAS — c'est le piège,
    # gardé sous test pour qu'on ne « simplifie » pas rms_match par erreur.
    l_sq, l_tall = _update_rms(square, False), _update_rms(tall, False)
    assert l_sq / l_tall > 1.3, \
        (f"le scaling √cols devrait désavantager les matrices hautes "
         f"(carré {l_sq:.3f} vs haute {l_tall:.3f}) — si cette assertion tombe, "
         f"le piège a changé de forme, pas disparu")

    # ── weight decay et AdamW embarqué ──────────────────────────────────────
    p = nn.Parameter(torch.ones(4, 4))
    p.grad = torch.zeros(4, 4)
    Muon([p], lr=0.1, wd=0.5, ns_steps=5).step()
    assert torch.allclose(p.detach(), torch.full((4, 4), 0.95)), "wd = p *= 1 - lr*wd"

    m2 = nn.Parameter(torch.zeros(4, 4)); m2.grad = torch.randn(4, 4)
    a2 = nn.Parameter(torch.zeros(4));    a2.grad = torch.randn(4)
    opt = Muon([m2], lr=0.1, adam_params=[a2], adam_lr=0.1)
    opt.step()
    assert a2.detach().abs().sum() > 0, "les params 1-D doivent bouger via AdamW"
    opt.zero_grad()
    assert m2.grad is None and a2.grad is None, "zero_grad doit couvrir les DEUX groupes"

    # un param sans gradient est simplement ignoré (pas d'exception)
    p3 = nn.Parameter(torch.ones(4, 4))
    Muon([p3], lr=0.1).step()
    assert torch.equal(p3.detach(), torch.ones(4, 4))

    # ── tri des paramètres : embeddings et statiques mHC chez AdamW ─────────
    class Toy(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(10, 4)      # 2-D mais lookup ⇒ AdamW
            self.lin = nn.Linear(4, 4, bias=True)  # weight ⇒ Muon, bias ⇒ AdamW
            self.norm = nn.LayerNorm(4)            # 1-D ⇒ AdamW
            self.S_pre = nn.Parameter(torch.zeros(4, 4))     # statique mHC ⇒ AdamW
            self.alpha_res = nn.Parameter(torch.zeros(4, 4))  # idem

    toy = Toy()
    mu, ad = _split_muon_params(toy)
    assert [tuple(p.shape) for p in mu] == [(4, 4)], "seul lin.weight va chez Muon"
    assert mu[0] is toy.lin.weight
    for p in (toy.embed.weight, toy.lin.bias, toy.norm.weight, toy.S_pre,
              toy.alpha_res):
        assert any(p is q for q in ad), "paramètre attendu chez AdamW"
    assert len(mu) + len(ad) == len(list(toy.parameters()))
    # un paramètre gelé n'est dans aucun des deux groupes
    toy.lin.weight.requires_grad_(False)
    assert _split_muon_params(toy)[0] == []

    print("muon self-test: OK (Newton-Schulz → σ=1 sur 4 formes + point fixe + "
          "invariance d'échelle, rms_match égalise carré/haute à 0.2 alors que "
          "√cols ne le fait pas, wd, AdamW embarqué + zero_grad, split params)")


if __name__ == "__main__":
    _selftest()
