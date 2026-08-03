"""ph.11 (E) — OÙ W_K' DOCKE-T-IL LES CLÉS BANQUE DANS LE SPECTRE DU RoPE ?

LA QUESTION, ET POURQUOI ELLE SE POSE MAINTENANT (SPEC_MEMOIRE_V2 §2.5).
`kvproj` partage le q du backbone, ROPE COMPRIS : le score d'une ligne vaut

    s = (R(t)·W_Q x_t)ᵀ (W_K'·g_r)

et la rotation R(t) n'est PAS annulée côté banque. Le produit se faisant PAIRE
DE DIMS PAR PAIRE DE DIMS, une clé banque qui met son énergie sur les paires à
ω élevé voit son score osciller avec la POSITION DU LECTEUR dans la fenêtre
(cos(ω·t − φ)) ; une clé qui la met sur les paires quasi statiques (ω ≈ 0) est
lue à l'identique de la position 0 à la position T.

D'où la prédiction inscrite dans la spec : **les clés banque doivent se docker
dans la BANDE LENTE**, et le fait que kvproj GAGNE le carré factoriel
(Δcit +0,209 apparié, t = 6,3) suggère que W_K' l'apprend TOUT SEUL — HoPE
retrouvé par l'implémentation plutôt que par le design.

Ce script mesure exactement ça, SANS RIEN RÉENTRAÎNER : il ouvre les ckpts
kvproj déjà sur le disque, décompose W_K' par tête et par paire de dims de
SORTIE (celles-là mêmes que le RoPE de la requête fait tourner), et compare la
distribution d'énergie à l'uniforme.

CE QUE ÇA TRANCHE :
  * énergie concentrée dans la bande LENTE ⇒ la prédiction §2.5 tient, les
    plans de métadonnées de la phase 11 sont posés au bon endroit (ils
    partagent la bande que le modèle avait déjà choisie), et le fallback
    « dé-roter q pour les colonnes banque » reste au placard ;
  * énergie UNIFORME ⇒ le modèle ne s'est pas protégé de la contamination :
    soit elle ne coûte rien à max_mem=8 (à re-tester au 350M, fenêtre longue),
    soit W_K' compense autrement ;
  * énergie concentrée dans la bande RAPIDE ⇒ prédiction RÉFUTÉE, et il faut
    comprendre ce que le modèle y gagne avant de poser quoi que ce soit sur ces
    dims.

LA MESURE, EN CLAIR. W_K' est [d, d] (sortie × entrée). Sa ligne de sortie i
produit la coordonnée i de la clé ; la tête h consomme les lignes
[h·dh, (h+1)·dh) et sa paire p est le couple de lignes (2p, 2p+1), auquel le
RoPE applique rot(ω_p·t) côté requête. L'énergie de la paire p est donc
‖W_K'[2p]‖² + ‖W_K'[2p+1]‖² — la norme des LIGNES DE SORTIE, pas des colonnes :
c'est la seule qui dise « combien de clé arrive sur cette paire ».

Baseline : à l'init (tirage uniforme de nn.Linear) l'énergie est UNIFORME sur
les paires. Toute structure lue ici est donc APPRISE. On la quantifie par la
part d'énergie de la moitié lente contre 0,5, et par le rang de corrélation
entre l'énergie et log(ω).

Repro (CPU, quelques secondes) :
    python -m deepseek_v4_mini.analysis.kvproj_wk_spectrum \
        --ckpts '/mnt/tb/checkpoints/toy_read_lab_p10/read-kvproj_*'
Sortie : une table par ckpt + un verdict agrégé, et un JSON
`kvproj_wk_spectrum.json` déposé À CÔTÉ de chaque ckpt.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os

import torch


def head_pair_energy(w: torch.Tensor, n_heads: int) -> torch.Tensor:
    """W_K' [d, d] → énergie par (tête, paire de dims de sortie) [H, dh/2].

    Les lignes de SORTIE sont l'axe 0 : c'est la coordonnée de la clé, donc
    celle que le RoPE de la requête fait tourner.
    """
    d = w.shape[0]
    dh = d // n_heads
    e = w.float().pow(2).sum(1)                       # [d] : énergie par ligne
    e = e.view(n_heads, dh // 2, 2).sum(-1)           # [H, dh/2] : par PAIRE
    return e


def rope_omegas(d_head: int, theta: float) -> torch.Tensor:
    """ω_p = θ^(−2p/dh) — décroissantes : les paires LENTES sont les dernières."""
    return 1.0 / (theta ** (torch.arange(0, d_head, 2).float() / d_head))


def analyse(path: str, n_heads_default: int = 8,
            theta_default: float = 10000.0) -> dict | None:
    """Un ckpt → dict de mesures (None si le ckpt n'a pas de W_K')."""
    sd = torch.load(path, map_location="cpu", weights_only=False)
    cfg = sd.get("cfg") or {}
    model = sd.get("model") or sd
    n_heads = int(cfg.get("n_heads", n_heads_default))
    theta = float(cfg.get("rope_theta", theta_default))
    keys = sorted(k for k in model if k.endswith(".attn.bk.weight"))
    if not keys:
        return None
    per_layer = []
    tot = None
    for k in keys:
        e = head_pair_energy(model[k], n_heads)       # [H, P]
        per_layer.append(e)
        tot = e if tot is None else tot + e
    P = tot.shape[1]
    om = rope_omegas(2 * P, theta)
    # part d'énergie de la MOITIÉ LENTE (les P/2 paires de plus faible ω)
    slow = torch.argsort(om)[:P // 2]
    frac_slow = float(tot[:, slow].sum() / tot.sum())
    # corrélation de Spearman entre l'énergie (moyennée sur les têtes) et le
    # RANG DE LENTEUR : +1 = toute l'énergie sur les paires les plus lentes.
    ev = tot.mean(0)
    r_e = torch.argsort(torch.argsort(ev)).float()
    r_s = torch.argsort(torch.argsort(-om)).float()   # rang de LENTEUR
    rho = float(((r_e - r_e.mean()) * (r_s - r_s.mean())).sum()
                / (r_e.std(unbiased=False) * r_s.std(unbiased=False) * P))
    # concentration : part des 4 paires les plus chargées (le budget d'un plan
    # d'âge de la phase 11)
    top4 = float(torch.topk(ev, min(4, P)).values.sum() / ev.sum())
    return {"ckpt": path, "n_layers": len(keys), "n_heads": n_heads,
            "n_pairs": P, "theta": theta,
            "frac_energy_slow_half": frac_slow,
            "spearman_energy_vs_slowness": rho,
            "top4_pair_share": top4, "uniform_top4_share": min(4, P) / P,
            "energy_by_pair": [round(float(x), 6) for x in
                               (ev / ev.sum()).tolist()],
            "omega_by_pair": [round(float(x), 8) for x in om.tolist()]}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ckpts",
                    default="/mnt/tb/checkpoints/toy_read_lab_p10/read-kvproj_*",
                    help="glob de DOSSIERS de run (final.pt y est cherché) ou "
                         "de fichiers .pt")
    ap.add_argument("--file", default="final.pt")
    ap.add_argument("--no-write", action="store_true",
                    help="n'écrit pas le JSON à côté des ckpts")
    a = ap.parse_args(argv)
    paths = []
    for p in sorted(glob.glob(a.ckpts)):
        paths.append(p if p.endswith(".pt") else os.path.join(p, a.file))
    paths = [p for p in paths if os.path.exists(p)]
    if not paths:
        raise SystemExit(f"aucun ckpt sous {a.ckpts!r}")
    print(f"=== spectre de W_K' (clés banque dédiées de kvproj) — "
          f"{len(paths)} ckpt(s) ===")
    print("  Prédiction §2.5 : l'énergie se concentre dans la BANDE LENTE "
          "(les paires que le RoPE de la requête ne fait presque pas tourner "
          "sur la fenêtre). Baseline à l'init = UNIFORME (0.500 / ρ 0).")
    print(f"{'run':52s} {'lente':>7s} {'ρ':>7s} {'top4':>7s} {'unif':>6s}")
    rows = []
    for p in paths:
        r = analyse(p)
        if r is None:
            print(f"{os.path.basename(os.path.dirname(p)):52s}   (pas de W_K')")
            continue
        rows.append(r)
        print(f"{os.path.basename(os.path.dirname(p)):52s} "
              f"{r['frac_energy_slow_half']:7.3f} "
              f"{r['spearman_energy_vs_slowness']:+7.3f} "
              f"{r['top4_pair_share']:7.3f} {r['uniform_top4_share']:6.3f}")
        if not a.no_write:
            out = os.path.join(os.path.dirname(p), "kvproj_wk_spectrum.json")
            try:
                with open(out, "w") as f:
                    json.dump(r, f, indent=2)
            except OSError as e:
                # le partage de la ferme est monté en lecture seule côté poste
                # de dépouillement : la MESURE prime, l'écriture est un confort.
                if not getattr(main, "_warned", False):
                    print(f"  (JSON non écrit : {e.strerror} — mesure "
                          f"affichée quand même)")
                    main._warned = True
    if not rows:
        raise SystemExit("aucun ckpt ne portait de projection K dédiée")
    fs = [r["frac_energy_slow_half"] for r in rows]
    rh = [r["spearman_energy_vs_slowness"] for r in rows]
    mu = sum(fs) / len(fs)
    se = (math.sqrt(sum((x - mu) ** 2 for x in fs) / max(len(fs) - 1, 1))
          / math.sqrt(len(fs)))
    mrho = sum(rh) / len(rh)
    print(f"\n  MOYENNE sur {len(rows)} ckpts : part de la moitié lente "
          f"{mu:.4f} ± {se:.4f} (SE) | ρ énergie↔lenteur {mrho:+.3f}")
    # VERDICT, écrit AVANT d'avoir vu les chiffres (le seuil est celui du
    # bruit : ±2 SE autour de l'uniforme).
    if mu - 2 * se > 0.5:
        v = ("PRÉDICTION §2.5 SOUTENUE — W_K' docke ses clés dans la bande "
             "LENTE de lui-même. Les plans de métadonnées de la ph.11 sont "
             "posés là où le modèle allait déjà ; le fallback « dé-roter q » "
             "reste au placard. ⚠️ Réserve : mesuré à max_mem=8 et fenêtre "
             "courte — c'est la fenêtre LONGUE qui rend la contamination "
             "coûteuse, donc à re-mesurer au 350M.")
    elif mu + 2 * se < 0.5:
        v = ("PRÉDICTION §2.5 RÉFUTÉE — l'énergie va vers la bande RAPIDE. "
             "Comprendre ce que le modèle y gagne AVANT de poser des plans de "
             "métadonnées ; le garde-fou `rot_drift_max` de la ph.11 devient "
             "un choix contre le modèle, pas avec lui.")
    else:
        v = ("INDÉCIS — l'énergie est indiscernable de l'uniforme. Lecture la "
             "plus simple : à max_mem=8 et fenêtre courte la contamination par "
             "R(t) ne coûte RIEN, donc rien ne pousse W_K' à s'en protéger. La "
             "ph.11 pose ses plans dans la bande lente PAR CONSTRUCTION, ce "
             "qui reste le choix sûr ; l'arbitrage réel se joue au 350M.")
    print(f"  VERDICT : {v}")


if __name__ == "__main__":
    main()
