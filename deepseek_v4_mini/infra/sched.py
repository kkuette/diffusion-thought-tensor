"""Schedules d'entraînement : LR (warmup + WSD) et β (fenêtre teacher).

Extrait de `code_defer_native.main` sans changer une seule valeur. Deux raisons
de le sortir : c'est de l'arithmétique pure, donc testable exhaustivement en
quelques millisecondes ; et le dry-run `--check` doit pouvoir imprimer la table
de schedule d'une config AVANT de louer le GPU — sans redériver le calcul dans
son coin, ce qui le ferait diverger du trainer au premier changement.

    lr_scale(step, ...)  → facteur multiplicatif appliqué à muon_lr et à adam_lr
    beta_at(step, ...)   → poids du code teacher, 1 pendant la fenêtre puis rampe
"""
from __future__ import annotations

import math

# Formes de décroissance sur la fenêtre WSD (p = fraction de fenêtre écoulée) :
#   linear : 1-p (historique)
#   step   : DeepSeek-V2/V3 — ×0.316 dès decay_start, ×0.1 sur le dernier quart
#            (même ratio de phases 3:1 que leurs bornes 60 %/90 %). Quitte en UN
#            step la zone plein-LR qui détruit le read.
#   1-sqrt : Hägele et al. 2024, vainqueur du cooldown WSD — chute rapide puis
#            longue traîne basse
#   cosine : le classique Chinchilla/LLaMA
#   stair  : escalier à `n` paliers GÉOMÉTRIQUES, du plein-LR à `end`. Motivé
#            par la mesure du run `fromsif_exec_tiled` : tout ce qui a bougé a
#            bougé DANS le decay (requote 0.16→0.29, codeexec 0→0.07, les deux à
#            bras ablaté 0.00) et rien avant. L'escalier donne plusieurs baisses
#            au lieu d'une rampe, et surtout chaque palier dure assez pour être
#            ÉVALUÉ : on mesure à quel LR le canal s'ouvre, au lieu de le
#            constater une fois à l'arrivée.
#            Paliers FIXES en step, pas déclenchés sur un plateau de CE mesurée :
#            un franchissement de seuil sur une loss dépend de réductions GPU non
#            bit-reproductibles (cf. l'amplification des ULP au top-k), et il
#            rendrait le schedule non rejouable — ce que la branche durcissement
#            vient justement d'aller chercher.
DECAY_SHAPES = ("linear", "step", "1-sqrt", "cosine", "stair")

# `stair` : nombre de paliers et facteur du DERNIER palier. Géométrique, donc
# chaque marche divise le LR par le même ratio r = end**(1/n) — 4 paliers vers
# 0.05 donnent 0.473 / 0.224 / 0.106 / 0.050. Le dernier palier n'est pas 0 :
# un plateau à LR nul ne consolide rien, il ne fait que brûler des steps.
STAIR_N, STAIR_END = 4, 0.05


def wsd_decay(step: int, *, steps: int, decay_start: int,
              shape: str = "linear", floor: float = 0.0,
              stair_n: int = STAIR_N, stair_end: float = STAIR_END) -> float:
    """Facteur de décroissance WSD à `step` (1.0 avant `decay_start`)."""
    if step <= decay_start:
        return 1.0
    p = (step - decay_start) / max(1, steps - decay_start)   # progression 0→1
    if shape == "step":
        return 0.316 if p <= 0.75 else 0.1
    if shape == "stair":
        n = max(1, int(stair_n))
        # index du palier 0..n-1. Compté depuis le PREMIER step décroissant
        # (decay_start+1, `decay_start` lui-même étant encore le plateau), pour
        # que les n paliers fassent exactement (steps-decay_start)/n steps
        # chacun — sinon le premier en perd un et le dernier en gagne un.
        k = min(n - 1, ((step - decay_start - 1) * n) // max(1, steps - decay_start))
        return float(stair_end) ** ((k + 1) / n)
    if shape == "1-sqrt":
        return floor + (1.0 - floor) * (1.0 - p ** 0.5)
    if shape == "cosine":
        return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * p))
    return floor + (1.0 - floor) * (1.0 - p)                 # linear (historique)


def lr_scale(step: int, *, steps: int, warmup: int, wsd: bool = True,
             decay_start: int | None = None, shape: str = "linear",
             floor: float = 0.0, stair_n: int = STAIR_N,
             stair_end: float = STAIR_END) -> float:
    """Facteur LR complet : warmup linéaire, puis décroissance WSD."""
    if decay_start is None:
        decay_start = int(steps * 0.66)
    scale = min(1.0, step / max(1, warmup))
    return scale * (wsd_decay(step, steps=steps, decay_start=decay_start,
                              shape=shape, floor=floor, stair_n=stair_n,
                              stair_end=stair_end) if wsd else 1.0)


def beta_at(step: int, *, enabled: bool, anneal_start: int, anneal_end: int) -> float:
    """Poids du code teacher : 1 jusqu'à `anneal_start`, rampe linéaire vers 0.

    β=0 signifie « teacher retiré » — le read ne consomme plus que le code écrit
    par le modèle. C'est la seule sortie qui compte pour un verdict.
    """
    if not enabled or step >= anneal_end:
        return 0.0
    if step <= anneal_start:
        return 1.0
    return 1.0 - (step - anneal_start) / max(1, anneal_end - anneal_start)


def describe(*, steps: int, warmup: int, wsd: bool, decay_start: int, shape: str,
             floor: float, muon_lr: float, lr: float,
             tf_on: bool, tf_a0: int, tf_a1: int,
             stair_n: int = STAIR_N, stair_end: float = STAIR_END) -> list[str]:
    """Table de schedule lisible, pour le dry-run `--check`."""
    marks = sorted({0, 1, warmup, decay_start, (decay_start + steps) // 2, steps}
                   | ({tf_a0, tf_a1} if tf_on else set()))
    if wsd and shape == "stair":         # une marque au 1er step de chaque palier
        w = steps - decay_start
        marks = sorted(set(marks) | {decay_start + 1 + (k * w) // max(1, stair_n)
                                     for k in range(max(1, stair_n))})
    out = [f"schedule: {steps} steps | warmup {warmup} | "
           f"wsd {'ON' if wsd else 'OFF'} shape={shape} start={decay_start} "
           f"floor={floor}"
           + (f" n={stair_n} end={stair_end}" if shape == "stair" else "")
           + (f" | teacher β=1 jusqu'à {tf_a0}, →0 à {tf_a1}"
              if tf_on else " | teacher OFF"),
           "  step      lr_muon      lr_adam    β"]
    for s in marks:
        if not 0 <= s <= steps:
            continue
        f = lr_scale(s, steps=steps, warmup=warmup, wsd=wsd,
                     decay_start=decay_start, shape=shape, floor=floor,
                     stair_n=stair_n, stair_end=stair_end)
        b = beta_at(s, enabled=tf_on, anneal_start=tf_a0, anneal_end=tf_a1)
        out.append(f"  {s:<8d} {muon_lr * f:<11.3e}  {lr * f:<11.3e}  {b:.2f}")
    return out


# ── self-test ────────────────────────────────────────────────────────────────

def _selftest() -> None:
    S, W, DS = 1000, 100, 600

    # warmup : monte linéairement de 0 à 1, plateau ensuite
    assert lr_scale(0, steps=S, warmup=W, wsd=False) == 0.0
    assert abs(lr_scale(50, steps=S, warmup=W, wsd=False) - 0.5) < 1e-12
    assert lr_scale(W, steps=S, warmup=W, wsd=False) == 1.0
    assert lr_scale(5 * W, steps=S, warmup=W, wsd=False) == 1.0
    assert lr_scale(0, steps=S, warmup=0, wsd=False) == 0.0    # warmup 0 : pas de div/0

    # WSD : plateau à 1 avant decay_start, décroissant après, jamais négatif
    for shape in DECAY_SHAPES:
        k = dict(steps=S, decay_start=DS, shape=shape, floor=0.0)
        assert wsd_decay(DS, **k) == 1.0, shape
        prev = 1.0
        for s in range(DS, S + 1, 10):
            v = wsd_decay(s, **k)
            assert 0.0 <= v <= 1.0, (shape, s, v)
            assert v <= prev + 1e-12, f"{shape} non monotone à {s}"
            prev = v
        # le plancher est respecté ('step' et 'stair' ont leurs paliers propres)
        if shape not in ("step", "stair"):
            assert abs(wsd_decay(S, **k)) < 1e-9, shape
            assert abs(wsd_decay(S, **{**k, "floor": 0.1}) - 0.1) < 1e-9, shape

    # continuité à la jonction : linéaire et cosine partent en douceur…
    for shape in ("linear", "cosine"):
        k = dict(steps=S, decay_start=DS, shape=shape, floor=0.0)
        assert abs(wsd_decay(DS + 1, **k) - 1.0) < 0.01, shape
    # …1-sqrt a une dérivée infinie en p=0 : il DOIT chuter vite au premier step
    # (c'est le « fast early drop » de Hägele et al.), sans être une falaise.
    v1 = wsd_decay(DS + 1, steps=S, decay_start=DS, shape="1-sqrt")
    assert 0.9 < v1 < 0.99, v1
    # 'step' saute délibérément (c'est sa raison d'être : quitter le plein LR)
    assert wsd_decay(DS + 1, steps=S, decay_start=DS, shape="step") == 0.316
    assert wsd_decay(S, steps=S, decay_start=DS, shape="step") == 0.1

    # 'stair' : n paliers EXACTS et CONSTANTS, ratio identique d'une marche à
    # l'autre, dernier palier == stair_end. On vérifie palier par palier — c'est
    # tout l'intérêt du schedule (chaque plateau doit être évaluable tel quel).
    for n, end in ((4, 0.05), (5, 0.1), (1, 0.2), (3, 1e-3)):
        ks = dict(steps=S, decay_start=DS, shape="stair",
                  stair_n=n, stair_end=end)
        assert wsd_decay(DS, **ks) == 1.0, (n, end)        # plateau intact avant
        # on ne redérive PAS les bornes : on lit la fonction step par step et on
        # regroupe en paliers constants — le test échoue donc si la découpe bouge
        runs = []                                          # [(valeur, longueur)]
        for s in range(DS + 1, S + 1):
            v = wsd_decay(s, **ks)
            if runs and abs(v - runs[-1][0]) < 1e-12:
                runs[-1][1] += 1
            else:
                runs.append([v, 1])
        assert len(runs) == n, (n, end, [r[0] for r in runs])
        lens = [r[1] for r in runs]
        assert max(lens) - min(lens) <= 1, (n, end, lens)   # paliers égaux à 1 près
        assert abs(runs[-1][0] - end) < 1e-12, (n, end, runs[-1][0])
        ratios = [runs[i + 1][0] / runs[i][0] for i in range(n - 1)]
        for r in ratios:                                   # géométrique
            assert abs(r - ratios[0]) < 1e-12, (n, end, ratios)
        if n > 1:                                          # et strictement < 1
            assert ratios[0] < 1.0, (n, end, ratios)
    # au-delà de `steps` (resume qui déborde) : on reste sur le dernier palier,
    # jamais de remontée ni de valeur négative
    assert abs(wsd_decay(S + 500, steps=S, decay_start=DS, shape="stair",
                         stair_n=4, stair_end=0.05) - 0.05) < 1e-12

    # forme inconnue = linéaire (le comportement historique, pas une exception)
    assert wsd_decay(800, steps=S, decay_start=DS, shape="???") == \
        wsd_decay(800, steps=S, decay_start=DS, shape="linear")

    # β : plateau à 1, rampe, puis 0 définitif ; OFF ⇒ 0 partout
    kb = dict(enabled=True, anneal_start=200, anneal_end=400)
    assert beta_at(0, **kb) == 1.0 and beta_at(200, **kb) == 1.0
    assert abs(beta_at(300, **kb) - 0.5) < 1e-12
    assert beta_at(400, **kb) == 0.0 and beta_at(10_000, **kb) == 0.0
    assert beta_at(300, enabled=False, anneal_start=200, anneal_end=400) == 0.0
    # fenêtre dégénérée (start == end) : bascule nette, pas de division par zéro
    assert beta_at(200, enabled=True, anneal_start=200, anneal_end=200) == 0.0

    # équivalence avec le calcul historique de code_defer_native.set_lr/_beta
    def _legacy_lr(step, steps, warmup, wsd, decay_start, shape, floor):
        scale = min(1.0, step / max(1, warmup))
        decay = 1.0
        if wsd and step > decay_start:
            p = (step - decay_start) / max(1, steps - decay_start)
            if shape == "step":
                decay = 0.316 if p <= 0.75 else 0.1
            elif shape == "1-sqrt":
                decay = floor + (1.0 - floor) * (1.0 - p ** 0.5)
            elif shape == "cosine":
                decay = floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * p))
            else:
                decay = floor + (1.0 - floor) * (1.0 - p)
        return scale * decay

    def _legacy_beta(s, tf_on, a0, a1):
        if not tf_on or s >= a1:
            return 0.0
        return 1.0 if s <= a0 else 1.0 - (s - a0) / max(1, a1 - a0)

    n = 0
    for shape in DECAY_SHAPES:
        if shape == "stair":       # postérieure au trainer historique, rien à
            continue               # comparer — sa spec est testée ci-dessus
        for floor in (0.0, 0.1):
            for s in range(0, S + 1, 7):
                assert lr_scale(s, steps=S, warmup=W, wsd=True, decay_start=DS,
                                shape=shape, floor=floor) == \
                    _legacy_lr(s, S, W, True, DS, shape, floor), (shape, floor, s)
                n += 1
    for s in range(0, 600, 3):
        assert beta_at(s, **kb) == _legacy_beta(s, True, 200, 400), s
        n += 1

    assert len(describe(steps=S, warmup=W, wsd=True, decay_start=DS, shape="1-sqrt",
                        floor=0.0, muon_lr=3e-3, lr=3e-4, tf_on=True,
                        tf_a0=200, tf_a1=400)) >= 6
    # `--check` doit montrer CHAQUE palier de l'escalier, sinon la table cache
    # précisément ce qu'on est venu y lire
    tbl = describe(steps=S, warmup=W, wsd=True, decay_start=DS, shape="stair",
                   floor=0.0, muon_lr=3e-3, lr=3e-4, tf_on=True,
                   tf_a0=200, tf_a1=400, stair_n=4, stair_end=0.05)
    lrs = {ln.split()[1] for ln in tbl[2:]}
    assert len(lrs) >= 4 + 1, tbl                   # 4 paliers + le plein LR

    print(f"sched self-test: OK (warmup, {len(DECAY_SHAPES)} formes WSD monotones "
          f"+ plancher, escalier n paliers géométriques, jonctions, β, "
          f"{n} points identiques au calcul historique)")


if __name__ == "__main__":
    _selftest()
