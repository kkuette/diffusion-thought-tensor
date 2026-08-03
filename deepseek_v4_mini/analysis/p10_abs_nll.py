"""Audit du Δnll citation de la grille/carré ph.10 : DÉCOMPOSITION EN ABSOLUS.

Question (user, 08-03) : « le Δnll est-il un bait ? » — Δ = nll_abl − nll_live
peut grandir parce que le PLANCHER ablaté monte (le backbone s'appuie sur la
banque et régresse sans elle), sans que le bras live soit meilleur. Le verdict
kvproj > kv_append (+0,209) n'est honnête que si nll_live(kvproj) est
réellement PLUS BAS à bras appariés.

Méthode : recharge chaque final.pt du répertoire p10, rejoue `evaluate` à
l'identique de l'éval finale (seed 1234, mêmes streams), et intercepte les
appels à `seg_ce` (strictement alternés live/abl par tour gradé dans
`evaluate`) pour récupérer les deux termes. AUTO-CONTRÔLE : le dnll recalculé
doit coller au `results.json` stocké (sinon le harnais diverge et la mesure
est jetée).

Usage (ferme, GPU) :
  python -m deepseek_v4_mini.analysis.p10_abs_nll \
      deepseek_v4_mini/configs/toy_read_lab_d512_p10.yaml \
      --root $TB_ROOT/checkpoints/toy_read_lab_p10 \
      --glob 'read-kv*' --n-convs 200 --device cuda
Sortie : $root/abs_nll_audit.json + table imprimée.
"""
import argparse
import glob as globmod
import json
import os

import torch
import yaml

import deepseek_v4_mini.toy_read_lab as toy


def audit_one(run_dir, raw, device, n_convs):
    ck = torch.load(os.path.join(run_dir, "final.pt"), map_location="cpu")
    cfg = toy.ToyCfg(**ck["cfg"])
    tok = toy.build_tokenizer(raw["tokenizer"])
    env = toy.OracleEnv(tok, cfg.max_mem, write_mode=cfg.write_mode)
    P = (toy.PersonaRuleStream if cfg.cond
         else toy.chat_stream_class("persona"))

    def pk(split, **over):
        return {**toy.persona_kwargs(raw, split, smoke=False, cond=cfg.cond),
                **over}

    sif_w = None
    if cfg.code in toy.SIF_CODES:
        seed = int((raw.get("training") or {}).get("seed", 0))
        sif_w = toy.sif_weight_table(P, tok, pk("train"), cfg.vocab_size,
                                     cfg.sif_a, seed=seed)
    model = toy.ToyReadLM(cfg, env.n_slots, env.n_attrs, sif_w=sif_w)
    model.load_state_dict(ck["model"])
    model.to(device).eval()

    ev_stream = P(tok, seed=1234, **pk("eval"))
    a_open = torch.tensor(
        tok(toy.A_OPEN, add_special_tokens=False)["input_ids"],
        dtype=torch.long, device=device)[None]
    stop_id = tok.convert_tokens_to_ids("<|im_end|>")
    max_new = int((raw.get("training") or {}).get("max_new", 48))
    amp = bool((raw.get("training") or {}).get("amp", True)) \
        and device.startswith("cuda")

    # interception : dans `evaluate`, seg_ce est appelé STRICTEMENT en paires
    # (live puis ablaté, mêmes X/W) par tour gradé — on enregistre la séquence.
    rec = []
    orig = toy.seg_ce

    def spy(logits, ids, w):
        s, n = orig(logits, ids, w)
        rec.append((float(s), float(n)))
        return s, n

    toy.seg_ce = spy
    try:
        ev = toy.evaluate(model, env, ev_stream, 1234, n_convs, device, tok,
                          a_open, stop_id, max_new, int(cfg.max_seq_len),
                          amp, n_show=0)
    finally:
        toy.seg_ce = orig
    assert len(rec) % 2 == 0, len(rec)
    sl = sum(r[0] for r in rec[0::2])
    sa = sum(r[0] for r in rec[1::2])
    nl = sum(r[1] for r in rec[0::2])
    out = dict(nll_live=sl / nl, nll_abl=sa / nl,
               dnll_recomputed=(sa - sl) / nl, n_tok=nl,
               grade_live=ev["grade_live"], dnll_evaluate=ev["dnll"])
    stored = json.load(open(os.path.join(run_dir, "results.json")))
    out["dnll_stored"] = stored["citation"]["dnll"]
    out["combo"] = stored["combo"]
    # auto-contrôle : même seed, même stream ⇒ l'écart doit être ~0
    out["harness_ok"] = abs(out["dnll_recomputed"] - out["dnll_stored"]) < 5e-3
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--root", required=True)
    ap.add_argument("--glob", default="read-kv*")
    ap.add_argument("--n-convs", type=int, default=200)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    raw = yaml.safe_load(open(a.config))

    def expand(x):                      # le loader de main expanse ${TB_ROOT}
        if isinstance(x, str):
            return os.path.expandvars(x)
        if isinstance(x, dict):
            return {k: expand(v) for k, v in x.items()}
        if isinstance(x, list):
            return [expand(v) for v in x]
        return x
    raw = expand(raw)

    res = {}
    dirs = sorted(d for d in globmod.glob(os.path.join(a.root, a.glob))
                  if os.path.exists(os.path.join(d, "final.pt"))
                  and os.path.exists(os.path.join(d, "results.json")))
    for d in dirs:
        name = os.path.basename(d)
        try:
            r = audit_one(d, raw, a.device, a.n_convs)
        except Exception as e:                       # une cellule ne bloque pas
            print(f"{name}: ÉCHEC {type(e).__name__}: {e}", flush=True)
            continue
        res[name] = r
        print(f"{name:46s} live {r['nll_live']:.4f}  abl {r['nll_abl']:.4f}  "
              f"Δ {r['dnll_recomputed']:+.4f} (stocké {r['dnll_stored']:+.4f}"
              f"{' OK' if r['harness_ok'] else ' ⚠️ DIVERGE'})", flush=True)

    out_path = os.path.join(a.root, "abs_nll_audit.json")
    try:
        json.dump(res, open(out_path, "w"), indent=1)
    except PermissionError:
        out_path = os.path.abspath("abs_nll_audit.json")
        json.dump(res, open(out_path, "w"), indent=1)
    print(f"\nécrit {out_path}", flush=True)

    # la réponse à la question : deltas appariés sur les ABSOLUS
    import statistics as st
    for fam in ("kvproj", "dualheads"):
        dl, da = [], []
        for n, r in res.items():
            if fam not in n or "_bq" in n:
                continue
            twin = res.get(n.replace(fam, "kvappend"))
            if twin:
                dl.append(r["nll_live"] - twin["nll_live"])
                da.append(r["nll_abl"] - twin["nll_abl"])
        if len(dl) > 1:
            print(f"{fam} − kvappend apparié (n={len(dl)}) : "
                  f"Δnll_LIVE {st.mean(dl):+.4f} ± {st.stdev(dl):.4f} "
                  f"(négatif = {fam} meilleur) | "
                  f"Δnll_ABL {st.mean(da):+.4f} ± {st.stdev(da):.4f} "
                  f"(positif = plancher plus haut = part 'bait' du Δnll)",
                  flush=True)


if __name__ == "__main__":
    main()
