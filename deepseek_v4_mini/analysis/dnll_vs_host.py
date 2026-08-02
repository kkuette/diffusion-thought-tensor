"""dsv6 — C1 : le Δnll est-il gonflé par l'érosion de l'hôte ?

L'objection, formulée par la critique du 2026-08-02 (PLAN_EXPERIENCES §3) en
recollant deux entrées du journal :

  07-27  l'érosion hôte a doublé — ic_ppl codeparrot 19,5 → 115,7, fineweb
         141 → 663. Différé : « l'érosion est un problème pour plus tard ».
  07-26  « Δnll est une DIFFÉRENCE, et un meilleur hôte tire les DEUX bras. »

Mis bout à bout : plus l'hôte se dégrade, plus le Δnll a de place pour croître.
L'indicateur principal de santé mémoire serait mécaniquement favorisé par la
dégradation qu'on a choisi d'ignorer. C'est exactement le couplage qu'un
relecteur cherche, et il se teste sur des fichiers DÉJÀ écrits — aucun GPU.

Le test ne peut pas être « Δnll monte-t-il quand ic_ppl monte ? » : sur un run
qui progresse, tout est corrélé au temps. Trois discriminants, tous rendus ici :

  1. PAR KIND. Si l'érosion gonflait mécaniquement le Δ, elle le gonflerait
     PARTOUT. Un kind à Δ plat pendant que ic_ppl est multiplié par 6 réfute la
     version mécanique — c'est le rôle de `codeexec`.
  2. LE BRAS ABLATÉ. C'est le discriminant décisif, et le seul qui tranche pour
     un kind dont le Δ monte. Une érosion qui gonfle artificiellement l'écart
     fait RÉGRESSER le bras sans mémoire (sa nll monte). Si le bras ablaté
     s'AMÉLIORE pendant que l'écart se creuse, l'écart est de la mémoire apprise.
  3. LES DEUX DISTRIBUTIONS. `ic_ppl` est mesuré hors domaine (codeparrot,
     fineweb) ; les nll de `math` sont en domaine. Un SFT qui se spécialise fait
     exactement ça : perdre hors domaine, gagner en domaine. Reporter les nll
     ABSOLUES des deux bras (règle du §1 : « ne jamais interpréter un Δnll sans
     reporter la nll absolue des deux bras ») rend le diagnostic lisible.

Repro :
    PYTHONPATH=. python deepseek_v4_mini/analysis/dnll_vs_host.py
    PYTHONPATH=. python deepseek_v4_mini/analysis/dnll_vs_host.py --plot out.png
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

# Les deux runs qui portent ic_ppl ET math aux MÊMES steps. `valsif_stair` est
# celui dont FINDINGS 07-27 décrit l'érosion ; `recall_stair` est une
# réplication indépendante (autres kinds, même recette d'escalier).
RUNS = ("v350_sft_valsif_stair", "v350_sft_recall_stair")


def load(run: str, root: str) -> dict:
    """{step: {"ic": {source: ppl}, "math": {kind: {...}}}} — apparié par step."""
    rows: dict[int, dict] = {}
    with open(f"{root}/runs/{run}/metrics.jsonl") as f:
        for line in f:
            r = json.loads(line)
            s = r.get("step")
            if s is None:
                continue
            slot = rows.setdefault(s, {"ic": {}, "math": {}})
            if "ic_ppl" in r:
                slot["ic"][r.get("source")] = r["ic_ppl"]
            if "math" in r:
                slot["math"].update(r["math"])
    return {s: v for s, v in sorted(rows.items()) if v["ic"] and v["math"]}


def series(rows: dict, kind: str) -> list[tuple]:
    """(step, ic_cp, nll, nll_abl, dnll, n_ans, grade, grade_abl) par palier.

    `ans_nll` est la nll de la RÉPONSE (pas du tour entier) : c'est la quantité
    dont le Δ est revendiqué. Un kind peut être absent d'un palier, ou avoir
    n_ans = 0 — on le saute plutôt que de propager un None en 0.0.

    `n_ans` et les grades voyagent avec la série parce qu'ils se sont révélés
    être les colonnes qui TRANCHENT : un Δnll sur n=2 ne mesure rien, et un Δnll
    sans écart de grade ne dit pas ce qu'il a l'air de dire.
    """
    out = []
    for s, v in rows.items():
        m = v["math"].get(kind)
        if not m:
            continue
        a, b = m.get("ans_nll"), m.get("ans_nll_abl")
        if a is None or b is None or not m.get("n_ans"):
            continue
        out.append((s, v["ic"].get("codeparrot"), a, b, b - a,
                    m["n_ans"], m.get("grade"), m.get("grade_abl")))
    return out


def trend(xs: list[float]) -> float:
    """Pente d'une régression linéaire sur l'INDEX (steps équidistants ici).

    Pas de corrélation de Pearson entre Δnll et ic_ppl : sur un run qui
    progresse, les deux sont corrélés au temps et le coefficient ne dirait rien
    sur la causalité. Ce qu'on veut est le SIGNE de chaque tendance, pris
    séparément, et c'est ce que la pente donne.
    """
    n = len(xs)
    if n < 2:
        return float("nan")
    mx = (n - 1) / 2
    my = sum(xs) / n
    num = sum((i - mx) * (x - my) for i, x in enumerate(xs))
    den = sum((i - mx) ** 2 for i in range(n))
    return num / den


def report(run: str, root: str) -> dict:
    rows = load(run, root)
    if not rows:
        print(f"  (aucun step apparié ic_ppl × math dans {run})")
        return {}
    steps = list(rows)
    ic_cp = [rows[s]["ic"].get("codeparrot") for s in steps]
    ic_fw = [rows[s]["ic"].get("fineweb") for s in steps]
    kinds = sorted({k for v in rows.values() for k in v["math"]})

    print(f"\n=== {run} — {len(steps)} paliers appariés, "
          f"step {steps[0]}..{steps[-1]}")
    print(f"    hôte HORS domaine : ic_ppl codeparrot {ic_cp[0]:.1f} → "
          f"{ic_cp[-1]:.1f}  (×{ic_cp[-1]/ic_cp[0]:.1f}) | "
          f"fineweb {ic_fw[0]:.1f} → {ic_fw[-1]:.1f} "
          f"(×{ic_fw[-1]/ic_fw[0]:.1f})")

    print(f"\n    {'kind':<10} {'n':>3} {'Δnll dbt':>9} {'Δnll fin':>9} "
          f"{'nll dbt→fin':>16} {'ABLATÉ dbt→fin':>16} {'grade m/abl':>12} "
          f"  verdict")
    verdicts = {}
    for k in kinds:
        ser = series(rows, k)
        if len(ser) < 2:
            continue
        d0, d1 = ser[0][4], ser[-1][4]
        n0, n1 = ser[0][2], ser[-1][2]
        a0, a1 = ser[0][3], ser[-1][3]
        if abs(d0) < 1e-9 and abs(d1) < 1e-9:
            continue                       # kind sans écart mesurable (session)
        n_ans = ser[-1][5]
        g, ga = ser[-1][6], ser[-1][7]
        d_slope = trend([r[4] for r in ser])
        # Trois discriminants, dans l'ordre où ils tuent une interprétation.
        if n1 > n0 and a1 > a0:
            # LES DEUX bras se dégradent : le Δ qui se creuse n'est pas un gain
            # de mémoire, c'est un ÉCART DE VITESSE DE DÉCROISSANCE. C'est
            # exactement le motif que l'objection décrit.
            v = "⚠ LES DEUX bras régressent"
        elif a1 > a0:
            v = "⚠ ablaté RÉGRESSE"
        elif abs(d1) < 1e-3:
            v = "Δ plat (réfute le couplage)"
        else:
            v = "ablaté s'améliore"
        # Un Δ sans écart de grade ne dit pas ce qu'il a l'air de dire.
        if g is not None and ga is not None and g <= ga:
            v += " · aucun écart de grade"
        verdicts[k] = (v, d_slope, n_ans, g, ga)
        gs = "—" if g is None else f"{g:.3f}/{ga:.3f}"
        print(f"    {k:<10} {n_ans:>3} {d0:>+9.3f} {d1:>+9.3f} "
              f"{n0:>7.3f}→{n1:<8.3f} {a0:>7.3f}→{a1:<8.3f} {gs:>12}   {v}")
    return verdicts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/mnt/tb")
    ap.add_argument("--plot", default=None, help="chemin du png (optionnel)")
    args = ap.parse_args()

    all_v = {}
    for run in RUNS:
        all_v[run] = report(run, args.root)

    print("\n=== VERDICT")
    bad = [(r, k, v) for r, vs in all_v.items()
           for k, (v, *_) in vs.items() if "⚠" in v]
    solid = [(r, k) for r, vs in all_v.items()
             for k, (v, _, n, g, ga) in vs.items()
             if "⚠" not in v and g is not None and ga is not None
             and g > ga and n >= 10]
    if solid:
        print("    RÉFUTÉ là où c'est testable contre le COMPORTEMENT : "
              + ", ".join(f"{r}/{k}" for r, k in solid))
        print("      un Δ plat pendant que ic_ppl est multiplié par ~6, avec les"
              " deux bras qui s'améliorent")
        print("      en absolu et un écart de grade réel — l'érosion ne gonfle"
              " pas le Δ.")
    if bad:
        print("    MAIS le motif de l'objection EXISTE sur : "
              + ", ".join(f"{r}/{k}" for r, k, _ in bad))
        for r, k, v in bad:
            print(f"      {r}/{k} : {v}")
        print("      → sur ces kinds le Δnll n'est PAS un gain de mémoire. Ne"
              " jamais le revendiquer seul.")
    print("\n    Règle opératoire : un Δnll n'est interprétable qu'accompagné"
          " (a) des nll ABSOLUES des")
    print("    deux bras — si les deux montent, le Δ est un écart de vitesse de"
          " décroissance, pas un")
    print("    gain ; (b) d'une métrique de COMPORTEMENT à bras ablaté au"
          " plancher ; (c) de son n.")

    if args.plot:
        _plot(args.root, args.plot)


def _plot(root: str, out: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(RUNS), figsize=(6.2 * len(RUNS), 4.2))
    for ax, run in zip(axes if len(RUNS) > 1 else [axes], RUNS):
        rows = load(run, root)
        steps = list(rows)
        ic = [rows[s]["ic"].get("codeparrot") for s in steps]
        ax.plot(steps, ic, color="0.35", lw=2, label="ic_ppl codeparrot (hôte, HORS domaine)")
        ax.set_yscale("log")
        ax.set_xlabel("step")
        ax.set_ylabel("ic_ppl (log)")
        ax2 = ax.twinx()
        for k in sorted({k for v in rows.values() for k in v["math"]}):
            ser = series(rows, k)
            if len(ser) < 2 or max(abs(r[4]) for r in ser) < 1e-3:
                continue
            ax2.plot([r[0] for r in ser], [r[4] for r in ser], marker="o", ms=3,
                     lw=1.4, label=f"Δnll {k}")
        ax2.set_ylabel("Δnll (ablaté − mémoire)")
        ax2.axhline(0, color="0.8", lw=0.8, zorder=0)
        ax.set_title(run, fontsize=10)
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="upper left")
    fig.suptitle("C1 — l'hôte se dégrade HORS domaine ; le Δnll ne suit pas kind par kind",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"\nfigure → {out}")


if __name__ == "__main__":
    main()
