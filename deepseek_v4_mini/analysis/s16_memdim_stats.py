"""S16 (SPEC §3) — le chiffrage qui dimensionne mem_dim et l'horizon effectif.

Trois nombres décident (§2.2) :
  1. ATOMES PAR TOUR (quantile haut → mem_dim) — vérité exacte sur recall_env,
     proxy SIF sur corpus réel (ultrachat_200k) ;
  2. TAUX DE WRITE — fraction des tours qui apportent ≥ 1 atome NOUVEAU
     (un tour redondant ne consomme pas de slot ⇒ horizon effectif =
     max_mem / taux-de-write) ;
  3. TAUX DE DÉDUP — fraction des atomes d'un tour déjà présents plus tôt
     dans la session (la redondance que la dédup Δn écarte à l'entrée).

Proxy « atome » sur corpus réel : token dont le poids SIF w = a/(a+p) dépasse
`--sif-thresh` (défaut 0,5, i.e. p ≤ a — la même famille de sélection que le
write du 350M, a = 1e-4), unigrammes estimés sur l'échantillon lui-même.
CPU only, aucun GPU.

Usage :
  python -m deepseek_v4_mini.analysis.s16_memdim_stats \
      [--lives 500] [--convs 2000] [--sif-a 1e-4] [--sif-thresh 0.5] [--out s16_stats.json]
"""
import argparse
import collections
import json
import os
import random
import statistics as st


def q(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))] if xs else 0


def summarize(name, per_turn, write_flags, dedup_pairs):
    """per_turn = [n_atomes_nouveaux] par tour ; write_flags = [bool] tour
    consomme un slot ; dedup_pairs = (atomes_totaux, atomes_déjà_vus)."""
    tot, seen = map(sum, zip(*dedup_pairs)) if dedup_pairs else (0, 0)
    wr = st.mean(write_flags) if write_flags else 0.0
    out = {
        "tours": len(per_turn),
        "atomes_par_tour": {
            "p50": q(per_turn, .50), "p90": q(per_turn, .90),
            "p99": q(per_turn, .99), "max": max(per_turn, default=0),
            "moyenne": round(st.mean(per_turn), 2) if per_turn else 0},
        "taux_write": round(wr, 3),
        "horizon_effectif_x": round(1.0 / wr, 2) if wr else None,
        "taux_dedup": round(seen / tot, 3) if tot else None,
    }
    print(f"\n== {name} ==")
    a = out["atomes_par_tour"]
    print(f"  atomes/tour : p50 {a['p50']}  p90 {a['p90']}  p99 {a['p99']}  "
          f"max {a['max']}  (moy {a['moyenne']})")
    print(f"  taux de write {out['taux_write']}  ⇒ horizon effectif = "
          f"max_mem × {out['horizon_effectif_x']}")
    print(f"  taux de dédup {out['taux_dedup']}")
    return out


def stats_recall_env(n_lives):
    from deepseek_v4_mini.recall_env import RecallEnvConfig, build_script
    from deepseek_v4_mini.toy_read_lab import build_tokenizer
    cfg = RecallEnvConfig()
    tok = build_tokenizer("HuggingFaceTB/SmolLM2-135M")
    ntoks = {}                       # atome → nb de tokens (cache)

    def nt(a):
        if a not in ntoks:
            ntoks[a] = len(tok(str(a), add_special_tokens=False)["input_ids"])
        return ntoks[a]

    per_turn, per_turn_tok, wf, dd = [], [], [], []
    for life in range(n_lives):
        sc = build_script(cfg, life)
        seen: set = set()
        atoms_by_seg = {f["seg"]: [str(a) for a in f["atoms"]]
                        for f in sc["facts"]}
        for t in sc["turns"]:
            ats = atoms_by_seg.get(t["user_seg"], [])
            new = [a for a in ats if a not in seen]
            dd.append((len(ats), len(ats) - len(new)))
            seen.update(ats)
            per_turn.append(len(new))
            per_turn_tok.append(sum(nt(a) for a in new))
            wf.append(len(new) > 0)
    out = summarize(f"recall_env ({n_lives} vies, vérité exacte)",
                    per_turn, wf, dd)
    out["tokens_atomes_par_tour"] = {
        "p50": q(per_turn_tok, .5), "p90": q(per_turn_tok, .9),
        "p99": q(per_turn_tok, .99), "max": max(per_turn_tok, default=0)}
    tt = out["tokens_atomes_par_tour"]
    print(f"  EN TOKENS (l'unité de mem_dim) : p50 {tt['p50']}  "
          f"p90 {tt['p90']}  p99 {tt['p99']}  max {tt['max']}")
    return out


def stats_ultrachat(n_convs, sif_a, thresh, seed=0):
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft",
                      streaming=False)
    idx = random.Random(seed).sample(range(len(ds)), min(n_convs * 2, len(ds)))
    convs = [ds[i]["messages"] for i in idx]
    # tokenizer du jouet/350M (SmolLM2) — même vocabulaire que la sélection
    from deepseek_v4_mini.toy_read_lab import build_tokenizer
    tok = build_tokenizer("HuggingFaceTB/SmolLM2-135M")

    # passe 1 : unigrammes sur la moitié de l'échantillon (rng dédié)
    counts = collections.Counter()
    total = 0
    enc_convs = []
    for msgs in convs[:n_convs]:
        enc = [(m["role"], tok(m["content"], add_special_tokens=False)
                ["input_ids"]) for m in msgs]
        enc_convs.append(enc)
        for _, ids in enc:
            counts.update(ids)
            total += len(ids)

    def sif_w(tid):
        p = counts[tid] / total
        return sif_a / (sif_a + p)

    per_turn, wf, dd = [], [], []
    by_role = {"user": [], "assistant": []}
    tok_lens = []
    for enc in enc_convs:
        seen: set = set()
        for role, ids in enc:
            tok_lens.append(len(ids))
            rare = [t for t in set(ids) if sif_w(t) >= thresh]
            new = [t for t in rare if t not in seen]
            dd.append((len(rare), len(rare) - len(new)))
            seen.update(rare)
            per_turn.append(len(new))
            by_role.setdefault(role, []).append(len(new))
            wf.append(len(new) > 0)
    out = summarize(f"ultrachat_200k ({len(enc_convs)} convs, proxy SIF "
                    f"a={sif_a:g} seuil {thresh})", per_turn, wf, dd)
    out["par_role"] = {r: {"p50": q(v, .5), "p90": q(v, .9), "p99": q(v, .99)}
                       for r, v in by_role.items() if v}
    for r, v in out["par_role"].items():
        print(f"  {r:9s} : p50 {v['p50']}  p90 {v['p90']}  p99 {v['p99']}")
    out["tokens_par_tour"] = {"p50": q(tok_lens, .5), "p90": q(tok_lens, .9),
                              "max": max(tok_lens, default=0)}
    print(f"  (tokens/tour : p50 {out['tokens_par_tour']['p50']}  "
          f"p90 {out['tokens_par_tour']['p90']})")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lives", type=int, default=500)
    ap.add_argument("--convs", type=int, default=2000)
    ap.add_argument("--sif-a", type=float, default=1e-4)
    ap.add_argument("--sif-thresh", type=float, default=0.5)
    ap.add_argument("--out", default="s16_stats.json")
    a = ap.parse_args()
    res = {"recall_env": stats_recall_env(a.lives)}
    try:
        res["ultrachat"] = stats_ultrachat(a.convs, a.sif_a, a.sif_thresh)
    except Exception as e:
        print(f"\nultrachat indisponible ({type(e).__name__}: {e}) — "
              f"partie corpus réel sautée")
    json.dump(res, open(a.out, "w"), indent=1)
    print(f"\nécrit {a.out}")


if __name__ == "__main__":
    main()
