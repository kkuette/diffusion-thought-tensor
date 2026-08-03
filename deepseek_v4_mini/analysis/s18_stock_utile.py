"""S18 (SPEC §3) — le STOCK utile à l'instant t (principe de suffisance §2.2).

Un atome (token SIF-rare ∪ numérique, même proxy que S16 v2) est VIVANT à t
s'il est apparu à un tour ≤ t ET resurgit à un tour > t : c'est l'ensemble
« encore intéressant » que la banque doit tenir — la borne basse mesurable de
la statistique suffisante (borne basse : la resurgence lexicale rate les
références sémantiques sans reprise de surface).

Sorties :
  - stock vivant (en atomes-tokens) par index de tour t : quantiles p50/p90 ;
  - distance de référence arrière (en tours) quand un atome resurgit ;
  - taille max du stock par conversation (le max_t : ce que la capacité doit
    couvrir) vs longueur de la conv.
CPU only. Limite connue : ultrachat = convs courtes (le stock long-session
demande un corpus de sessions longues — noté au registre).

Usage :
  python -m deepseek_v4_mini.analysis.s18_stock_utile [--convs 800] [--out s18.json]
"""
import argparse
import collections
import json
import random
import statistics as st


def q(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))] if xs else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--convs", type=int, default=800)
    ap.add_argument("--sif-a", type=float, default=1e-4)
    ap.add_argument("--sif-thresh", type=float, default=0.5)
    ap.add_argument("--out", default="s18_stock.json")
    a = ap.parse_args()

    from datasets import load_dataset
    from deepseek_v4_mini.toy_read_lab import build_tokenizer
    ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
    idx = random.Random(0).sample(range(len(ds)), a.convs)
    tok = build_tokenizer("HuggingFaceTB/SmolLM2-135M")

    counts = collections.Counter()
    total = 0
    enc_convs = []
    for i in idx:
        enc = [tok(m["content"], add_special_tokens=False)["input_ids"]
               for m in ds[i]["messages"]]
        enc_convs.append(enc)
        for ids in enc:
            counts.update(ids)
            total += len(ids)

    is_num = {}

    def atom(tid):
        if tid not in is_num:
            s = tok.convert_ids_to_tokens(int(tid)) or ""
            p = counts[tid] / total
            is_num[tid] = (a.sif_a / (a.sif_a + p) >= a.sif_thresh) \
                or any(c.isdigit() for c in s)
        return is_num[tid]

    stock_by_t = collections.defaultdict(list)   # t → [stock vivant]
    ref_dist = []                                # distances de résurgence
    stock_max, conv_len = [], []
    for enc in enc_convs:
        occ = collections.defaultdict(list)      # atome → [tours]
        for t, ids in enumerate(enc):
            for tid in set(ids):
                if atom(tid):
                    occ[tid].append(t)
        T = len(enc)
        conv_len.append(T)
        # stock(t) = atomes vus ≤ t qui resurgissent > t
        first = {k: v[0] for k, v in occ.items()}
        last = {k: v[-1] for k, v in occ.items()}
        peak = 0
        for t in range(T):
            live = sum(1 for k in occ if first[k] <= t < last[k])
            stock_by_t[t].append(live)
            peak = max(peak, live)
        stock_max.append(peak)
        for v in occ.values():
            ref_dist += [b - a_ for a_, b in zip(v, v[1:])]

    print(f"== S18 stock utile ({len(enc_convs)} convs ultrachat, "
          f"longueur p50 {q(conv_len, .5)} tours) ==")
    print("  stock vivant par tour t (atomes-tokens) :")
    for t in sorted(stock_by_t)[:12]:
        v = stock_by_t[t]
        print(f"    t={t:2d} (n={len(v):4d}) : p50 {q(v, .5):4d}  "
              f"p90 {q(v, .9):4d}")
    print(f"  PIC de stock par conv : p50 {q(stock_max, .5)}  "
          f"p90 {q(stock_max, .9)}  p99 {q(stock_max, .99)}  "
          f"max {max(stock_max, default=0)}")
    print(f"  distance de résurgence (tours) : p50 {q(ref_dist, .5)}  "
          f"p90 {q(ref_dist, .9)}  max {max(ref_dist, default=0)}")
    cap = 8 * 16
    over = st.mean([1.0 if s > cap else 0.0 for s in stock_max])
    print(f"  convs dont le pic dépasse une capacité 8×16={cap} lignes : "
          f"{over:.1%}")
    json.dump({"stock_by_t": {t: {"p50": q(v, .5), "p90": q(v, .9)}
                              for t, v in stock_by_t.items()},
               "pic": {"p50": q(stock_max, .5), "p90": q(stock_max, .9),
                       "p99": q(stock_max, .99)},
               "ref_dist": {"p50": q(ref_dist, .5), "p90": q(ref_dist, .9)},
               "part_pic_sup_128": over},
              open(a.out, "w"), indent=1)
    print(f"écrit {a.out}")


if __name__ == "__main__":
    main()
