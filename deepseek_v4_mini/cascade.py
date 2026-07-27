"""
Cascade mémoire v3 — spec utilisateur 2026-07-12 (verrouillée le soir même) :
« débordement en 2 temps × fractale max_mem ».

Structure. Le bloc 0 est la banque vive du modèle (inchangée, dans le graphe).
Chaque niveau k >= 1 a DEUX positions ; chaque position est un tenseur borné à
max_mem tranches, une tranche = une position PLEINE du niveau k-1 :

    niveau 1 : tranche = slot évincé [B, mem_dim]        pos pleine = [B, M, D]
    niveau 2 : tranche = matrice [B, M, D]               pos pleine = [B, M, M, D]
    niveau k : pos pleine = [B, M^k, ..., D]  (ordre k+2 avec le batch)

Flux (débordement en 2 temps) :
  temps 1 — la pos 0 se REMPLIT (une tranche à chaque arrivée) pendant que la
            pos 1 reste pleine : le read du niveau ne voit jamais d'init pur ;
  temps 2 — pos 0 pleine => la pos 1 (doyenne) descend ENTIÈRE comme une
            tranche de la pos 0 du niveau k+1, pos 0 glisse en pos 1, une
            neuve s'ouvre. Sous le niveau `depth`, descendre = disparaître
            (l'éviction du fond, seule destruction).

Read (merge-at-read v1 = moyenne, validée zero-shot jusqu'à avg64) : chaque
niveau rend TOUJOURS une matrice [B, <=M, D] — la moyenne de toutes les
matrices écrites qu'il contient (pos 1 + pos 0 partielle). Au niveau 1 la
pos 0 est partielle au grain SLOT : moyenne par index de slot sur le contenu
écrit seulement (jamais de padding zéro dans la moyenne — pas d'init dilué).

Tout est stocké DÉTACHÉ (hors graphe) : le TBPTT n'atteint que le bloc 0,
le read des niveaux profonds apprend, leur write non (pattern G2).
"""
from __future__ import annotations

from typing import Dict, List, Optional

import torch


def default_layer_map(n_layers: int, depth: int) -> List[int]:
    """Carte par défaut : les `depth` dernières couches lisent les niveaux 1..depth,
    les autres la banque vive. Partagée par le trainer et le dry-run `--check`,
    pour que le rapport montre la carte RÉELLEMENT utilisée."""
    return [0] * (n_layers - depth) + list(range(1, depth + 1))


class CascadeMemory:
    """État par conversation. Pas un nn.Module : aucune trainable var ici."""

    def __init__(self, depth: int, max_mem: int):
        assert depth >= 1
        self.depth = depth
        self.M = max_mem
        self.lv: Dict[int, dict] = {
            k: {"p0": [], "p1": None} for k in range(1, depth + 1)
        }
        self.n_pushed = 0     # slots reçus (stats/tests)
        self.n_destroyed = 0  # unités sorties par le fond (stats/tests)

    # ── écriture ────────────────────────────────────────────────────────────

    def push_slot(self, slot: torch.Tensor) -> None:
        """Reçoit le slot que la banque vive vient d'évincer. [B, D]."""
        self.n_pushed += 1
        self._push(1, slot.detach())

    def _push(self, k: int, unit: torch.Tensor) -> None:
        if k > self.depth:
            self.n_destroyed += 1
            return
        L = self.lv[k]
        L["p0"].append(unit)
        if len(L["p0"]) == self.M:
            if L["p1"] is not None:
                self._push(k + 1, L["p1"])
            L["p1"] = torch.stack(L["p0"], dim=1)   # [B, M, ...unit]
            L["p0"] = []

    # ── lecture ─────────────────────────────────────────────────────────────

    def read(self, k: int) -> Optional[torch.Tensor]:
        """Matrice mergée du niveau k : [B, M, D] (ou [B, m<M, D] en tout début
        de vie, ou None si le niveau n'a encore rien reçu)."""
        L, M = self.lv[k], self.M
        if k == 1:
            p1, p0 = L["p1"], L["p0"]
            if p1 is None and not p0:
                return None
            if p1 is None:
                return torch.stack(p0, dim=1)                     # [B, m, D]
            if not p0:
                return p1
            out = p1.clone()
            part = torch.stack(p0, dim=1)                         # [B, m, D]
            m = part.size(1)
            out[:, :m] = (out[:, :m] + part) / 2.0
            return out
        mats: List[torch.Tensor] = []
        if L["p1"] is not None:
            mats.append(self._flat(L["p1"]))
        for u in L["p0"]:
            mats.append(self._flat(u))
        if not mats:
            return None
        return torch.cat(mats, dim=1).mean(dim=1)                 # [B, M, D]

    @staticmethod
    def _flat(u: torch.Tensor) -> torch.Tensor:
        """[B, ..., M, D] → [B, n_mat, M, D] : toutes les matrices contenues."""
        B, D = u.size(0), u.size(-1)
        M = u.size(-2)
        return u.reshape(B, -1, M, D)

    # ── banques par couche pour le forward ──────────────────────────────────

    def layer_banks(
        self, live_bank: torch.Tensor, layer_map: List[int]
    ) -> List[Optional[torch.Tensor]]:
        """layer_map[i] = niveau lu par la couche i (0 = banque vive).
        Les niveaux > depth sont rabattus sur depth (flag de profondeur)."""
        cache: Dict[int, Optional[torch.Tensor]] = {0: live_bank}
        out: List[Optional[torch.Tensor]] = []
        for lvl in layer_map:
            lvl = min(lvl, self.depth)
            if lvl not in cache:
                r = self.read(lvl)
                cache[lvl] = None if r is None else r.to(live_bank.dtype)
            out.append(cache[lvl])
        return out

    # ── (dé)sérialisation : la banque complète comme artefact fichier ────────

    def state_dict(self) -> dict:
        return {"depth": self.depth, "M": self.M,
                "n_pushed": self.n_pushed, "n_destroyed": self.n_destroyed,
                "lv": {k: {"p0": [u.detach().cpu() for u in L["p0"]],
                           "p1": (None if L["p1"] is None
                                  else L["p1"].detach().cpu())}
                       for k, L in self.lv.items()}}

    @classmethod
    def from_state(cls, sd: dict, device=None) -> "CascadeMemory":
        c = cls(sd["depth"], sd["M"])
        c.n_pushed, c.n_destroyed = sd["n_pushed"], sd["n_destroyed"]
        mv = (lambda t: t.to(device)) if device is not None else (lambda t: t)
        c.lv = {k: {"p0": [mv(u) for u in L["p0"]],
                    "p1": None if L["p1"] is None else mv(L["p1"])}
                for k, L in sd["lv"].items()}
        return c

    # ── stats (probes / logs) ────────────────────────────────────────────────

    def stats(self) -> Dict[str, int]:
        s = {"pushed": self.n_pushed, "destroyed": self.n_destroyed}
        for k, L in self.lv.items():
            s[f"lv{k}_p0"] = len(L["p0"])
            s[f"lv{k}_p1"] = 0 if L["p1"] is None else 1
        return s


# ── self-test ────────────────────────────────────────────────────────────────

def _selftest() -> None:
    """Le débordement en 2 temps, et la promesse « jamais d'init dilué ».

    Le read rend une moyenne : si une position partielle était complétée par des
    zéros, le niveau rendrait des vecteurs artificiellement rétrécis — le genre
    de bug qui ne se voit pas dans une loss et fausse une conclusion.
    """
    torch.manual_seed(0)
    B, D, M = 2, 5, 4

    def slots(n):
        return [torch.randn(B, D) for _ in range(n)]

    # ── vide : rien à lire, rien à compter ──────────────────────────────────
    c = CascadeMemory(depth=2, max_mem=M)
    assert c.read(1) is None and c.read(2) is None
    assert c.stats() == {"pushed": 0, "destroyed": 0,
                         "lv1_p0": 0, "lv1_p1": 0, "lv2_p0": 0, "lv2_p1": 0}

    # ── temps 1 : la pos 0 se remplit, le read suit sans padding ────────────
    c, us = CascadeMemory(depth=2, max_mem=M), slots(M)
    for i, u in enumerate(us, 1):
        c.push_slot(u)
        r = c.read(1)
        if i < M:
            assert r.shape == (B, i, D), (i, r.shape)      # partiel = i tranches
            assert torch.equal(r, torch.stack(us[:i], dim=1))
        else:
            assert r.shape == (B, M, D)                    # pos 1 pleine
            assert torch.equal(r, torch.stack(us, dim=1))
    assert c.stats()["lv1_p1"] == 1 and c.stats()["lv1_p0"] == 0

    # ── pos 1 pleine + pos 0 partielle : moyenne par INDEX DE SLOT ──────────
    p1 = torch.stack(us, dim=1)
    part = slots(2)
    for u in part:
        c.push_slot(u)
        assert c.read(1).shape == (B, M, D), "le read garde toujours M tranches"
    r = c.read(1)
    got_part = torch.stack(part, dim=1)
    assert torch.allclose(r[:, :2], (p1[:, :2] + got_part) / 2.0), "moyenne partielle"
    assert torch.equal(r[:, 2:], p1[:, 2:]), \
        "les tranches non couvertes par la pos 0 doivent rester INTACTES (pas /2, pas 0)"

    # ── temps 2 : la doyenne descend d'un niveau, entière ────────────────────
    c = CascadeMemory(depth=2, max_mem=M)
    for i in range(1, 2 * M + 1):
        c.push_slot(torch.randn(B, D))
        lv2_seen = c.read(2) is not None
        assert lv2_seen == (i >= 2 * M), \
            f"le niveau 2 doit recevoir sa 1re tranche pile au push {2 * M} (vu à {i})"
    assert c.read(2).shape == (B, M, D), "un niveau rend TOUJOURS une matrice [B,M,D]"
    assert c.stats()["destroyed"] == 0, "rien ne sort tant qu'on est au-dessus du fond"

    # ── le fond détruit, et il le compte ────────────────────────────────────
    c = CascadeMemory(depth=1, max_mem=M)
    for _ in range(2 * M):
        c.push_slot(torch.randn(B, D))
    assert c.stats() == {"pushed": 2 * M, "destroyed": 1,
                         "lv1_p0": 0, "lv1_p1": 1}, c.stats()

    # ── banques par couche ──────────────────────────────────────────────────
    live = torch.randn(B, M, D)
    c = CascadeMemory(depth=2, max_mem=M)
    for _ in range(2 * M):
        c.push_slot(torch.randn(B, D))
    lb = c.layer_banks(live, [0, 0, 1, 2, 7])
    assert torch.equal(lb[0], live) and torch.equal(lb[1], live)
    assert torch.equal(lb[2], c.read(1)) and torch.equal(lb[3], c.read(2))
    assert torch.equal(lb[4], c.read(2)), "un niveau > depth se rabat sur depth"
    # niveau jamais alimenté ⇒ None (le forward doit gérer l'absence, pas des zéros)
    assert CascadeMemory(depth=2, max_mem=M).layer_banks(live, [1])[0] is None

    # ── (dé)sérialisation : l'artefact banque doit revenir à l'identique ────
    c = CascadeMemory(depth=3, max_mem=M)
    for _ in range(M * M + 3):
        c.push_slot(torch.randn(B, D))
    back = CascadeMemory.from_state(c.state_dict())
    assert back.stats() == c.stats()
    for k in range(1, 4):
        a, b = c.read(k), back.read(k)
        assert (a is None) == (b is None), k
        if a is not None:
            assert torch.equal(a, b), f"niveau {k} altéré par le round-trip"

    # ── carte de couches par défaut ─────────────────────────────────────────
    assert default_layer_map(6, 2) == [0, 0, 0, 0, 1, 2]
    assert default_layer_map(12, 4) == [0] * 8 + [1, 2, 3, 4]
    assert max(default_layer_map(6, 3)) == 3 and len(default_layer_map(6, 3)) == 6

    # ── tout est détaché : le TBPTT ne doit atteindre que la banque vive ────
    c = CascadeMemory(depth=1, max_mem=M)
    x = torch.randn(B, D, requires_grad=True)
    c.push_slot(x * 2)
    assert not c.read(1).requires_grad, "la cascade doit être hors du graphe"

    print("cascade self-test: OK (débordement 2 temps, read sans dilution d'init, "
          "descente entière, destruction au fond comptée, layer_banks + rabat, "
          "round-trip état, carte par défaut, détachement)")


if __name__ == "__main__":
    _selftest()
