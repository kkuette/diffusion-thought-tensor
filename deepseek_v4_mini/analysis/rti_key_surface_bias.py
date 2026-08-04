"""dsv6 — kill-test 3 (SPEC_MEMOIRE_V2 §4) : biais SIF clé/surface, chiffres vs mots.

La critique du 07-31 prédit un bug avant le run : la clé procédurale ET la
sélection de surface de `build_group` passent toutes deux par les poids SIF,
et le teacher (a=1e-4) donne aux chiffres w̄ ≈ 0.02 contre une médiane 0.156.
La sélection rti utilise a=1e-2 précisément parce que 1e-4 écrasait la strate
`code` — ce probe mesure si 1e-2 suffit pour la strate `numeric`, celle de
l'exemple phare du claim (« quelle version de la lib ? »).

Trois mesures, par strate de l'env recall (persona/pref = lexicales, numeric =
chiffres purs, code = mixte nom+constante), sur les segments d'énonciation
EXACTS du chemin d'entraînement (`RecallEnvStream.segs`, val_mask compris) :

  surface   le span de LA VALEUR survit-il au top-k(13) SIF du tour entier ?
            inclusion pleine (tous ses tokens sélectionnés — condition
            nécessaire de la copie), fraction moyenne, et poids SIF de la
            valeur vs médiane du tour.
  clés      `key_separation` (cos entre clés + rank-1 sur banques de
            max_groups=8) en trois régimes de difficulté croissante :
            global (banque mélangée), intra-strate, et MÊME-SLOT — cinq
            locker codes ne diffèrent que par leurs chiffres : le rank-1
            même-slot est la discrimination portée par LA VALEUR seule.
  span      part de valeurs dont le span n'est même pas retrouvable dans le
            tour tokenisé (val_mask vide) — en amont de tout le reste.

Verdict attendu si la critique a raison : inclusion pleine et rank-1 même-slot
s'effondrent sur `numeric` (et sur la moitié constante de `code`) pendant que
persona/pref tiennent. Décision de la spec : corriger la pondération de
SÉLECTION (§3.1) avant tout run.

Usage (racine du repo) :
    PYTHONPATH=. python deepseek_v4_mini/analysis/rti_key_surface_bias.py \
        --ckpt /mnt/tb/checkpoints/v350_sft_recall_rti/step_1000.pt \
        [--lives 400] [--sif-a 0.01]
"""

import argparse
from collections import defaultdict

import torch

from deepseek_v4_mini.data.recall_env import RecallEnvStream, slot_id_map
from deepseek_v4_mini.rl.rti import RtiConfig, build_group, key_separation, sif_table


def collect(stream, lives: int) -> list:
    """[(slot, ids [T], atomes [T] bool)] des segs d'énonciation, chemin réel.

    Les atomes = `copy_mask` (TOUS les atomes citables — pour `code`, nom ET
    constante), repli sur val_mask si absent. C'est le dénominateur honnête :
    la copie exige chaque atome, pas seulement le champ `value`."""
    inv = {v: k for k, v in slot_id_map().items()}
    out = []
    for life in range(lives):
        sc = stream.script(life)
        for seg in stream.segs(sc):
            if "fact_slot" not in seg:
                continue
            slot = inv[int(seg["fact_slot"][0, 0])]
            cm = seg.get("copy_mask", seg["val_mask"])[0].bool()
            out.append((slot, seg["input_ids"][0], cm))
    return out


def _sel(w: torch.Tensor, k: int, cm: torch.Tensor = None) -> set:
    """Miroir exact de la sélection de `build_group` (SIF pur ou span garanti)."""
    if cm is None:
        return set(torch.topk(w, k).indices.tolist())
    keep = torch.nonzero(cm > 0).reshape(-1)[:k]
    if keep.numel() < k:
        wf = w.clone()
        wf[keep] = float("-inf")
        return set(keep.tolist()) | set(torch.topk(wf, k - keep.numel())
                                        .indices.tolist())
    return set(keep.tolist())


def surface_stats(facts: list, sif_w: torch.Tensor, cfg: RtiConfig) -> dict:
    """Par strate : les atomes survivent-ils au top-k ? SIF pur vs corrigé."""
    agg = defaultdict(lambda: {"n": 0, "no_span": 0, "full": 0, "full_fix": 0,
                               "frac": 0.0, "w_val": 0.0, "w_med": 0.0,
                               "max_atoms": 0})
    for slot, ids, vm in facts:
        st = agg[slot.split(":")[0]]
        if not vm.any():
            st["no_span"] += 1
            continue
        w = sif_w[ids].float()
        k = min(cfg.top_k, ids.numel())
        val = set(torch.nonzero(vm).reshape(-1).tolist())
        st["n"] += 1
        st["max_atoms"] = max(st["max_atoms"], len(val))
        sel = _sel(w, k)
        st["full"] += int(val <= sel)
        st["frac"] += len(val & sel) / len(val)
        st["full_fix"] += int(val <= _sel(w, k, vm))
        st["w_val"] += float(w[list(val)].mean())
        st["w_med"] += float(w.median())
    for st in agg.values():
        n = max(st["n"], 1)
        st["full"] /= n
        st["full_fix"] /= n
        st["frac"] /= n
        st["w_val"] /= n
        st["w_med"] /= n
    return dict(agg)


def key_stats(facts: list, embed_w, sif_w, cfg: RtiConfig, cap: int = 320) -> dict:
    """rank-1 de `key_separation` en trois régimes ; même-slot = le décisif."""
    by_slot, by_strat, all_segs = defaultdict(list), defaultdict(list), []
    for slot, ids, _vm in facts:
        by_slot[slot].append(ids)
        by_strat[slot.split(":")[0]].append(ids)
        all_segs.append(ids)
    out = {"global": key_separation(embed_w, sif_w, all_segs[:cap], cfg)}
    for name, segs in sorted(by_strat.items()):
        out[f"strate:{name}"] = key_separation(embed_w, sif_w, segs[:cap], cfg)
    # même-slot : agrégat par strate (moyenne des slots assez peuplés) + pire slot
    per = defaultdict(list)
    for slot, segs in sorted(by_slot.items()):
        if len(segs) >= 2 * cfg.max_groups:
            r = key_separation(embed_w, sif_w, segs[:64], cfg)
            per[slot.split(":")[0]].append((slot, r["rank1"], r["cos_med"]))
    for name, rows in sorted(per.items()):
        worst = min(rows, key=lambda t: t[1])
        out[f"slot:{name}"] = {
            "rank1": sum(r for _, r, _ in rows) / len(rows),
            "cos_med": sum(c for _, _, c in rows) / len(rows),
            "n_slots": len(rows), "worst": f"{worst[0]}={worst[1]:.3f}"}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--lives", type=int, default=400)
    ap.add_argument("--sif-a", type=float, default=None,
                    help="défaut : RtiConfig.sif_a (1e-2, le déployé)")
    ap.add_argument("--tokenizer", default="HuggingFaceTB/SmolLM2-135M")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    stream = RecallEnvStream(tok, seed=0)
    cfg = RtiConfig()
    if args.sif_a is not None:
        cfg.sif_a = args.sif_a

    sd = torch.load(args.ckpt, map_location="cpu", mmap=True)
    embed_w = sd["model"]["embed.weight"].float()
    sif_w = sif_table(stream, embed_w.size(0), cfg.sif_a)

    facts = collect(stream, args.lives)
    n_strat = defaultdict(int)
    for slot, _i, _v in facts:
        n_strat[slot.split(":")[0]] += 1
    print(f"ckpt step {sd.get('step')}, {len(facts)} énonciations "
          f"({dict(sorted(n_strat.items()))}), top_k={cfg.top_k}, "
          f"sif_a={cfg.sif_a}, banques rank-1 m={cfg.max_groups}")

    print("\n── SURFACE : les atomes survivent-ils au top-k ? ──")
    print(f"{'strate':<12}{'n':>5}{'sans-span':>10}{'SIF pur':>9}"
          f"{'corrigé':>9}{'frac':>7}{'w̄(val)':>9}{'w~(tour)':>9}{'max|a|':>7}")
    for name, st in sorted(surface_stats(facts, sif_w, cfg).items()):
        print(f"{name:<12}{st['n']:>5}{st['no_span']:>10}{st['full']:>9.3f}"
              f"{st['full_fix']:>9.3f}{st['frac']:>7.3f}{st['w_val']:>9.3f}"
              f"{st['w_med']:>9.3f}{st['max_atoms']:>7}")

    print("\n── CLÉS : rank-1 (chance = 1/8) ──")
    print(f"{'régime':<20}{'n':>5}{'rank1':>7}{'cos_med':>9}{'cos_p99':>9}")
    for name, r in key_stats(facts, embed_w, sif_w, cfg).items():
        extra = f"   pire {r['worst']}" if "worst" in r else ""
        n = r.get("n", r.get("n_slots", 0))
        p99 = f"{r['cos_p99']:>9.3f}" if "cos_p99" in r else f"{'—':>9}"
        print(f"{name:<20}{n:>5}{r['rank1']:>7.3f}{r['cos_med']:>9.3f}{p99}{extra}")


if __name__ == "__main__":
    main()
