"""Muon — Momentum Orthogonalised by Newton-Schulz (+ le split de params).

Extrait de train.py sans la moindre modification de corps : train.py est le
trainer de l'arc jouet dsv4mini (clos), et c'était son SEUL lien avec le
programme courant — code_defer_native, code_train et sft_train importaient
l'optimiseur depuis un module de 2600 lignes qu'ils n'utilisent pas par
ailleurs. train.py réexporte les trois noms, donc les commandes de repro
dsv4mini et tout `from .train import Muon` existant continuent de marcher.

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
