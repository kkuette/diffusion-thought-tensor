"""dsv6 — kill-tests 1 & 2 (SPEC_MEMOIRE_V2 §4) : la voie SURFACE bat-elle ses
déflations textuelles, à information égale et à forwards comptés ?

LA QUESTION
───────────
Le run `v350_sft_recall_copy` a fermé la chaîne retrieve-then-inject-then-copy
(grade 0.28 vs 0.00 ablaté, nll VALEUR 3.69 ON vs 6.19 OFF, p_copy 0.566 sur la
valeur contre 0.005 sur le template, r@1 0.818). Le claim v2 exige davantage :
que le canal batte la COMPACTION TEXTUELLE — « les mêmes tokens, mais écrits
dans le prompt » — et la CoT explicite à forwards appariés. Ce script mesure
les deux déflations sur le MÊME checkpoint, les MÊMES vies, les MÊMES sondes.

LES CINQ BRAS (une seule passe, banque partagée, sondes appariées)
─────────────────────────────────────────────────────────────────
  A_rti   banque ON : préfixe de G·(top_k+1) pseudo-tokens NATIFS injectés
          (`RtiRunner.parts`, sélection réelle du retriever, ordre = score) +
          tête de copie. C'est exactement le bras live de `evaluate_math`.
  B_text  KILL-TEST 1. Les MÊMES ids de tokens, dans le MÊME ordre, avec le
          MÊME séparateur (`<blank>`), mais en TEXTE dans la fenêtre : banque
          OFF, aucune injection, aucun type-vec, aucune tête de copie. Même
          longueur de séquence, mêmes positions RoPE que A — la seule
          différence est la NATURE du préfixe (tokens réels vs pseudo-tokens
          + pointeur de copie).
  B_nat   même contenu, mais DÉTOKENISÉ et présenté en note naturelle
          (`Notes:\\n…`). Sert de garde-fou au piège de layout : B_text impose
          au modèle une suite de tokens qu'il n'a jamais vue COMME TEXTE.
  C_off   OFF sec (l'ablaté existant) : borne basse.
  D_think KILL-TEST 2. OFF + `<think>` explicite : les énoncés VERBATIM de
          TOUS les faits résidents de la banque (jusqu'à max_groups), sans
          retrieval — la compaction honnête, qui paie sa mémoire en tokens de
          fenêtre. Le compte de forwards est rapporté des deux façons (§ plus
          bas), jamais arbitré en silence.

CE QUE LES BRAS B/B_nat REÇOIVENT GRATUITEMENT (et c'est délibéré)
─────────────────────────────────────────────────────────────────
Le contenu textuel de B/B_nat est le résultat du retrieval que le bras A a
payé (sa requête + son W_q). B est donc une BORNE SUPÉRIEURE de la déflation
« compaction avec indirection » : si A bat B malgré ce cadeau, le verdict est
d'autant plus solide ; si B bat A, le kill-test tombe sans appel. D, lui, ne
reçoit aucun retrieval — il porte TOUT (c'est le prix de la fenêtre).

COMPTABILITÉ DES FORWARDS (règle non négociable du §1 : jamais en « tokens
visibles »)
──────────────────────────────────────────────────────────────────────────
Par sonde, en teacher-forcing, le script compte pour chaque bras :
  * `calls`   — invocations du modèle ;
  * `tokens`  — positions traversées (préfixe injecté COMPRIS pour A).
Le bras A paie une invocation de plus que les autres : le forward de la
question qui produit h_query (le retrieval). Les writes de A ne coûtent AUCUN
forward — la sélection de surface est procédurale (`build_group` lit la table
d'embedding). Les deux comptes sont rapportés côte à côte ; le lecteur tranche.

CE QUE CE SCRIPT NE MESURE PAS
──────────────────────────────
Le protocole RESET (§5.4) — ici la fenêtre est déjà bornée au SEGMENT (le
chemin rti forwarde chaque tour isolément, `init_mem=None`), ce qui est le cas
le plus favorable à la banque et le plus dur pour la compaction : celle-ci doit
re-porter son résumé à CHAQUE tour. Et l'horizon de divergence en boucle
fermée (§5.5) : tout est teacher-forcé sauf le décodage greedy des réponses
gradées.

REPRO (racine du repo, GPU libre) :
    source ~/anaconda3/etc/profile.d/conda.sh && conda activate diffusion-thought
    PYTHONPATH=. TB_ROOT=/mnt/tb python -m deepseek_v4_mini.analysis.kt12_surface_vs_text \
        --ckpt /mnt/tb/checkpoints/v350_sft_recall_copy/final.pt \
        --config deepseek_v4_mini/configs/sft_recall_350m_copy.yaml \
        --lives 120 --out /mnt/tb/runs/kt12_surface_vs_text.json

Self-test CPU (mini-modèle, tokenizer stub, aucun réseau) :
    PYTHONPATH=. python -m deepseek_v4_mini.analysis.kt12_surface_vs_text --selftest
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections import defaultdict

import torch
import torch.nn.functional as F

from deepseek_v4_mini.code_defer_native import _greedy, _val_split
from deepseek_v4_mini.data.recall_env import (RecallEnvConfig, RecallEnvStream,
                                         grade_probe, slot_id_map)
from deepseek_v4_mini.rl.rti import RtiConfig, RtiRunner, sif_table

ARMS = ("A_rti", "B_text", "B_nat", "C_off", "D_think")
A_OPEN = "<|im_start|>assistant\n"

# ── agrégation ───────────────────────────────────────────────────────────────


def _new_acc() -> dict:
    """Sommes brutes d'un bras. `nv/dv` = numérateur/dénominateur VALEUR,
    `nt/dt` = TEMPLATE ; la somme des deux est la nll de réponse AU BIT (c'est
    une partition, cf. `_val_split`). `pv/pn_v` = porte de copie."""
    return {k: 0.0 for k in ("num", "den", "nv", "dv", "nt", "dt",
                             "pv", "pn_v", "pt", "pn_t", "g_ok", "g_n",
                             "calls", "tokens", "q_calls", "q_tok", "dec_tok")}


def _fin(a: dict) -> dict:
    """Sommes → moyennes lisibles."""
    d = lambda x, y: (x / y) if y > 0 else float("nan")
    return {"ans_nll": d(a["num"], a["den"]),
            "val_nll": d(a["nv"], a["dv"]),
            "tpl_nll": d(a["nt"], a["dt"]),
            "p_copy_val": d(a["pv"], a["pn_v"]),
            "p_copy_tpl": d(a["pt"], a["pn_t"]),
            "grade": d(a["g_ok"], a["g_n"]), "n_graded": int(a["g_n"]),
            "n_val_tok": a["dv"], "n_tpl_tok": a["dt"],
            # comptabilité : `ans_*` = le forward de la réponse (préfixe
            # injecté compris pour A), `q_*` = le forward de la QUESTION que
            # seul A paie (il produit h_query, donc le retrieval). Les totaux
            # sont la somme — mais les deux postes restent lisibles, c'est ce
            # que le kill-test 2 demande de ne pas arbitrer en silence.
            "ans_calls": int(a["calls"]), "ans_tokens": int(a["tokens"]),
            "q_calls": int(a["q_calls"]), "q_tokens": int(a["q_tok"]),
            "calls": int(a["calls"] + a["q_calls"]),
            "tokens": int(a["tokens"] + a["q_tok"]),
            "dec_tokens": int(a["dec_tok"])}


def wilson(k: float, n: float, z: float = 1.96) -> tuple:
    """IC95 de Wilson — le seul honnête aux petits n et aux p près de 0 (la
    normale simple a déjà fait conclure à tort sur cette série : à n=20 et
    p=0.28 elle donne un intervalle qui déborde sous zéro)."""
    if n <= 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def paired_delta(rows: list, a: str, b: str, key: str = "grade",
                 iters: int = 20000, seed: int = 7) -> dict:
    """Δ APPARIÉ a−b sur les mêmes sondes + IC95 bootstrap (rééchantillonnage
    des SONDES, pas des tirages : c'est l'unité d'observation)."""
    d = [float(r[key][a]) - float(r[key][b]) for r in rows
         if a in r[key] and b in r[key]]
    n = len(d)
    if n == 0:
        return {"delta": float("nan"), "lo": float("nan"), "hi": float("nan"),
                "n": 0}
    mean = sum(d) / n
    g = torch.Generator().manual_seed(seed)
    t = torch.tensor(d, dtype=torch.float64)
    idx = torch.randint(0, n, (iters, n), generator=g)
    bs = t[idx].mean(1).sort().values
    return {"delta": mean, "lo": float(bs[int(0.025 * iters)]),
            "hi": float(bs[int(0.975 * iters)]), "n": n}


# ── mesure d'un bras sur UN tour gradé ───────────────────────────────────────


@torch.no_grad()
def arm_nll(model, x, lmask, vmask, rti_lb, amp, inject=None, prefix_ids=None,
            copy_head=None) -> dict:
    """nll teacher-forcée d'un tour, ventilée VALEUR / TEMPLATE.

    `x` porte DÉJÀ son préfixe textuel éventuel (bras B/B_nat/D) ; `lmask` et
    `vmask` sont zéro-paddés d'autant, donc les positions supervisées sont
    EXACTEMENT les mêmes dans tous les bras — c'est l'invariant qui rend les
    nll comparables (vérifié au self-test : `den` identique bras à bras).
    """
    kw = {} if prefix_ids is None else {"prefix_ids": prefix_ids}
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
        o = model(x, init_mem=None, layer_banks=rti_lb, write=False,
                  inject=inject, **kw)
    lg = o["logits"].float()
    ce = F.cross_entropy(lg[:, :-1].reshape(-1, lg.size(-1)),
                         x[:, 1:].reshape(-1), reduction="none")
    m2 = lmask[:, 1:]
    vm2 = (vmask[:, 1:] > 0).to(m2.dtype)
    nv, dv, nt, dt = _val_split(ce, m2, vm2)
    out = {"num": float((ce * m2.reshape(-1)).sum()), "den": float(m2.sum()),
           "nv": float(nv), "dv": float(dv), "nt": float(nt), "dt": float(dt),
           "pv": 0.0, "pn_v": 0.0, "pt": 0.0, "pn_t": 0.0}
    gate = (getattr(copy_head, "last_gate", None)
            if (copy_head is not None and prefix_ids is not None) else None)
    if gate is not None:
        g = gate[:, :-1].float()
        sup = (m2 > 0).to(g.dtype)
        out["pv"] = float((g * sup * vm2).sum())
        out["pn_v"] = float((sup * vm2).sum())
        out["pt"] = float((g * sup * (1 - vm2)).sum())
        out["pn_t"] = float((sup * (1 - vm2)).sum())
    # tokens RÉELLEMENT traversés : le préfixe injecté en fait partie.
    out["_tok"] = int(x.size(1)) + (0 if inject is None else int(inject.size(1)))
    return out


def _prepend(x, lmask, vmask, pre_ids):
    """(x, lmask, vmask) avec `pre_ids` collé DEVANT, masques à zéro dessus."""
    if pre_ids is None or pre_ids.numel() == 0:
        return x, lmask, vmask
    p = pre_ids.reshape(1, -1).to(x.device)
    z = torch.zeros(1, p.size(1), device=x.device)
    return (torch.cat([p, x], 1), torch.cat([z, lmask], 1),
            torch.cat([z, vmask], 1))


# ── la passe ─────────────────────────────────────────────────────────────────


@torch.no_grad()
def run(model, stream, tok, rti, cfg_rti, device, *, lives: int,
        warmup: int = 3, life0: int = 0, max_new: int = 96, amp: bool = True,
        decode_arms=("A_rti", "B_text", "B_nat", "D_think"), use_cache: bool = True,
        copy_head=None, sep_id: int = 0, exec_timeout: float = 6.0,
        verbose: bool = True) -> dict:
    """Une vie après l'autre, cinq bras par sonde. L'ORDRE DES APPELS rti est
    celui de `evaluate_math` (parts → forward → write → query) : le déplacer
    d'un cran décalerait le préfixe d'un tour et le bras A mesurerait la
    mémoire d'un AUTRE tour.

    La banque n'est PAS remise à zéro entre les vies (comme à l'éval du
    trainer, `no_reset_files: 0`) : les vies antérieures fournissent les
    distracteurs résidents, et sans elles la FIFO serait trop courte pour
    fournir `eval_groups` groupes — auquel cas `parts` n'injecte RIEN (stricte
    isomorphie de layout) et le bras A serait éteint par construction. Les
    `possible` du script restent exacts : l'éviction ne dépend que des writes
    postérieurs au fait, tous internes à la vie.
    """
    model.eval()
    rti_lb = [None] * len(model.blocks)
    embed_w, type_vec = model.embed.weight, model.rti_type.vec
    a_open = torch.tensor(tok(A_OPEN, add_special_tokens=False)["input_ids"],
                          dtype=torch.long, device=device).unsqueeze(0)
    stop_id = tok.convert_tokens_to_ids("<|im_end|>")
    inv_slot = {v: k for k, v in slot_id_map().items()}

    acc = {a: _new_acc() for a in ARMS}
    by_st = {a: defaultdict(_new_acc) for a in ARMS}
    rows: list = []
    c_txt = None                       # bras OFF : sortie CONSTANTE (cf. eval)
    fifo: list = []                    # énoncés verbatim des groupes résidents
    n_probe = n_skip = n_noinj = 0
    t0 = time.time()

    for li in range(life0, life0 + warmup + lives):
        measured = li >= life0 + warmup
        sc = stream.script(li)
        segs = stream.segs(sc)
        probes = sc["probes"]
        for s in segs:
            x = s["input_ids"].to(device)
            lmask = s["loss_mask"].to(device)
            role = s["role"]
            # 1. le préfixe DÛ à ce seg (décidé au seg précédent = la question)
            rows_i, inj, pids = rti.parts(embed_w, type_vec, 1, roles=(role,),
                                          train=False, with_ids=True)[0]
            # `probe_id` est posé sur les DEUX segs de la sonde (la question ET
            # la réponse — cf. `RecallEnvStream.segs`) ; le tour GRADÉ est la
            # réponse, et elle seule porte `decode`. Confondre les deux faisait
            # compter la question comme une sonde « sans injection » (le
            # préfixe n'est dû qu'au seg SUIVANT la question).
            pid_probe = s.get("probe_id") if s.get("decode") else None
            if measured and pid_probe is not None:
                pr = probes[int(pid_probe)]
                n_probe += 1
                if not pr["possible"]:
                    n_skip += 1
                elif inj is None:
                    n_noinj += 1                # banque trop courte : hors mesure
                else:
                    st = pr["slot"].split(":")[0]
                    vmask = s.get("val_mask")
                    vmask = (torch.zeros_like(lmask) if vmask is None
                             else vmask.to(device))
                    # ── les préfixes textuels, tous dérivés du MÊME pids ──
                    sel = pids[0].clone()
                    sel[sel < 0] = int(sep_id)          # sentinelle → <blank>
                    grp = [g[g >= 0] for g in
                           pids[0].reshape(-1, cfg_rti.group_prefix)]
                    nat = "Notes:\n" + "\n".join(
                        tok.decode(g.tolist()) for g in grp) + "\n"
                    nat_ids = torch.tensor(
                        tok(nat, add_special_tokens=False)["input_ids"],
                        dtype=torch.long, device=device)
                    thk = ("<think>\n" + "\n".join(fifo) + "\n"
                           if fifo else "<think>\n\n")
                    thk_ids = torch.tensor(
                        tok(thk, add_special_tokens=False)["input_ids"],
                        dtype=torch.long, device=device)
                    pre = {"A_rti": None, "B_text": sel, "B_nat": nat_ids,
                           "C_off": None, "D_think": thk_ids}
                    r = {"life": li, "pid": int(pid_probe), "stratum": st,
                         "age": int(pr["age"]), "hit": rti.last_hit[0],
                         "top1": rti.last_top1[0], "grade": {}, "val_nll": {},
                         "ans_nll": {}}
                    for arm in ARMS:
                        xa, la, va = _prepend(x, lmask, vmask, pre[arm])
                        m = arm_nll(model, xa, la, va, rti_lb, amp,
                                    inject=inj if arm == "A_rti" else None,
                                    prefix_ids=pids if arm == "A_rti" else None,
                                    copy_head=copy_head)
                        tokn = m.pop("_tok")
                        for k, v in m.items():
                            acc[arm][k] += v
                            by_st[arm][st][k] += v
                        acc[arm]["calls"] += 1
                        acc[arm]["tokens"] += tokn
                        by_st[arm][st]["calls"] += 1
                        by_st[arm][st]["tokens"] += tokn
                        r["val_nll"][arm] = m["nv"] / max(m["dv"], 1e-6)
                        r["ans_nll"][arm] = m["num"] / max(m["den"], 1e-6)
                        # ── décodage greedy (le grade) ──
                        if arm == "C_off":
                            if c_txt is None:
                                out = _greedy(model, a_open, None, max_new,
                                              stop_id, amp, use_cache,
                                              layer_banks=rti_lb)
                                c_txt = tok.decode(out[0].tolist())
                                acc[arm]["dec_tok"] += int(out.size(1))
                            txt = c_txt
                        elif arm in decode_arms:
                            pf = (a_open if pre[arm] is None else
                                  torch.cat([pre[arm].reshape(1, -1), a_open], 1))
                            out = _greedy(model, pf, None, max_new, stop_id,
                                          amp, use_cache,
                                          inject=inj if arm == "A_rti" else None,
                                          prefix_ids=(pids if arm == "A_rti"
                                                      else None),
                                          layer_banks=rti_lb)
                            txt = tok.decode(out[0].tolist())
                            acc[arm]["dec_tok"] += int(out.size(1))
                        else:
                            txt = None
                        if txt is not None:
                            g = grade_probe(pr, txt, exec_timeout)["grade"]
                            acc[arm]["g_ok"] += g; acc[arm]["g_n"] += 1
                            by_st[arm][st]["g_ok"] += g
                            by_st[arm][st]["g_n"] += 1
                            r["grade"][arm] = g
                    rows.append(r)
            # 2. write PROCÉDURAL (zéro forward, zéro gradient)
            n_w = rti.write(embed_w, x, s.get("fact_slot"),
                            copy_mask=s.get("copy_mask"))
            if n_w:
                # L'ÉNONCÉ verbatim, échafaudage ChatML retiré : `<|im_start|>`
                # est un token spécial (donc sauté) mais le mot « user » qui
                # le suit est du texte ordinaire, et le laisser dans le bloc
                # `<think>` handicaperait le bras D pour rien.
                stm = tok.decode(x[0].tolist(), skip_special_tokens=True).strip()
                for lead in ("user\n", "user"):
                    if stm.startswith(lead):
                        stm = stm[len(lead):].strip()
                        break
                fifo.append(stm)
                fifo[:] = fifo[-cfg_rti.max_groups:]
            # 3. query : la requête due au seg SUIVANT (le seul forward que le
            #    bras A paie EN PLUS des autres — compté ici)
            if s.get("q_slot") is not None and int(s["q_slot"].max()):
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                    oq = model(x, init_mem=None, compute_logits=False,
                               layer_banks=rti_lb, write=False)
                rti.query(model.rti_retriever, s["q_slot"],
                          oq["hidden"].float())
                if measured:
                    acc["A_rti"]["q_calls"] += 1
                    acc["A_rti"]["q_tok"] += int(x.size(1))
        if verbose and measured and (li - life0 - warmup) % 10 == 9:
            print(f"  vie {li - life0 - warmup + 1}/{lives} — {len(rows)} sondes "
                  f"mesurées, {time.time() - t0:.0f} s", flush=True)

    return {"arms": {a: _fin(acc[a]) for a in ARMS},
            "by_stratum": {a: {k: _fin(v) for k, v in by_st[a].items()}
                           for a in ARMS},
            "rows": rows, "c_text": c_txt,
            "n_probes_seen": n_probe, "n_impossible": n_skip,
            "n_no_inject": n_noinj,
            "retrieval": rti.telemetry(),
            "seconds": time.time() - t0}


# ── construction du modèle (miroir EXACT de code_defer_native.main) ──────────


def build(ckpt: str, config: str, device: str):
    from transformers import AutoTokenizer
    from deepseek_v4_mini.infra.paths import load_yaml
    from deepseek_v4_mini.core.model import ThoughtBankConfig, ThoughtBankLM
    from deepseek_v4_mini.rl.rti import InjectType, Retriever
    from deepseek_v4_mini.rl.rti_copy import CopyHead, CopyHeadConfig

    raw = load_yaml(config)
    tok = AutoTokenizer.from_pretrained(raw["tokenizer"])
    add = [x for x in ("<think>", "<blank>") if x not in tok.get_vocab()]
    if add:
        tok.add_special_tokens({"additional_special_tokens": add})
    mcfg = dict(raw["model"]); mcfg["vocab_size"] = len(tok)
    cfg = ThoughtBankConfig(**mcfg)
    model = ThoughtBankLM(cfg).to(device)
    rt = raw.get("rti") or {}
    rcfg = RtiConfig(
        top_k=int(rt.get("top_k", 13)),
        max_groups=int(rt.get("max_groups", cfg.max_mem)),
        sif_a=float(rt.get("sif_a", 1e-2)),
        train_groups=int(rt.get("train_groups", 2)),
        eval_groups=int(rt.get("eval_groups", 2)),
        train_order=str(rt.get("train_order", "oracle_first")),
        write_every_turn=bool(rt.get("write_every_turn", False)),
        sep_token=str(rt.get("sep_token", "<blank>")))
    model.rti_retriever = Retriever(cfg.d_model).to(device)
    model.rti_type = InjectType(cfg.d_model).to(device)
    ccfg = CopyHeadConfig.from_raw(rt.get("copy_head"))
    if ccfg.enabled:
        model.rti_copy = CopyHead(cfg.d_model, ccfg).to(device)
    sd = torch.load(ckpt, map_location="cpu")
    miss, unexp = model.load_state_dict(sd["model"], strict=False)
    assert not miss and not unexp, (sorted(miss)[:8], sorted(unexp)[:8])
    model.eval()
    return model, tok, cfg, rcfg, raw, int(sd.get("step", -1))


# ── impression ───────────────────────────────────────────────────────────────


def report(res: dict, rows_key: str = "rows") -> None:
    a = res["arms"]
    print(f"\nsondes mesurées {len(res[rows_key])} "
          f"(vues {res['n_probes_seen']}, impossibles {res['n_impossible']}, "
          f"sans injection {res['n_no_inject']}) | "
          f"r@1 {res['retrieval']['recall1']:.3f} "
          f"r@2 {res['retrieval']['recall2']:.3f} | {res['seconds']:.0f} s")
    print(f"\n{'bras':<9}{'grade':>7}{'IC95':>16}{'nllVAL':>9}{'nllTPL':>9}"
          f"{'nllANS':>9}{'p_c val':>9}{'calls':>7}{'tok/rép':>9}{'tokens':>9}")
    for k in ARMS:
        v = a[k]
        lo, hi = wilson(v["grade"] * v["n_graded"], v["n_graded"])
        ic = f"[{lo:.3f},{hi:.3f}]" if v["n_graded"] else "—"
        pc = (f"{v['p_copy_val']:>9.3f}" if v["p_copy_val"] == v["p_copy_val"]
              and v["p_copy_val"] > 0 else f"{'—':>9}")
        n = max(len(res[rows_key]), 1)
        print(f"{k:<9}{v['grade']:>7.3f}{ic:>16}{v['val_nll']:>9.3f}"
              f"{v['tpl_nll']:>9.3f}{v['ans_nll']:>9.3f}{pc}"
              f"{v['calls']:>7}{v['ans_tokens'] / n:>9.1f}{v['tokens']:>9}")
    print(f"  (calls = invocations du modèle en teacher-forcing ; A_rti inclut "
          f"{a['A_rti']['q_calls']} forwards de QUESTION — "
          f"{a['A_rti']['q_tokens']} tokens — que lui seul paie ; les writes "
          f"de A coûtent 0 forward, la sélection est procédurale)")
    print("\nΔ APPARIÉS (bootstrap 20 k sur les sondes) — grade : POSITIF = A "
          "meilleur ; val_nll : NÉGATIF = A meilleur :")
    for b in ("B_text", "B_nat", "C_off", "D_think"):
        for key in ("grade", "val_nll"):
            d = paired_delta(res[rows_key], "A_rti", b, key)
            if d["n"]:
                print(f"  A_rti − {b:<8} {key:<8} {d['delta']:+.3f}  "
                      f"IC95 [{d['lo']:+.3f}, {d['hi']:+.3f}]  n={d['n']}")
    print("\npar STRATE (grade | nll VALEUR) :")
    sts = sorted({s for k in ARMS for s in res["by_stratum"][k]})
    print(f"{'strate':<12}" + "".join(f"{k:>16}" for k in ARMS))
    for st in sts:
        cells = []
        for k in ARMS:
            v = res["by_stratum"][k].get(st)
            cells.append("—".rjust(16) if v is None else
                         f"{v['grade']:.2f}|{v['val_nll']:.2f}".rjust(16))
        print(f"{st:<12}" + "".join(cells))


# ── self-test CPU ────────────────────────────────────────────────────────────


class _Tok:
    """Tokenizer stub : un caractère = un id. Suffit à tout ce que la passe
    demande d'un tokenizer (encoder, décoder, un id de séparateur, un id
    d'arrêt) et ne touche pas au réseau."""

    def __call__(self, s, add_special_tokens=False):
        return {"input_ids": [ord(c) % 200 for c in s]}

    def decode(self, ids, skip_special_tokens=False):
        return "".join(chr(int(i)) for i in ids)

    def convert_tokens_to_ids(self, t):
        return {"<|im_end|>": 3, "<blank>": 4}.get(t, 5)


def _selftest() -> None:
    from deepseek_v4_mini.core.model import ThoughtBankConfig, ThoughtBankLM
    from deepseek_v4_mini.rl.rti import InjectType, Retriever
    from deepseek_v4_mini.rl.rti_copy import CopyHead, CopyHeadConfig

    torch.manual_seed(20260803)
    V = 256
    mcfg = ThoughtBankConfig(vocab_size=V, d_model=32, n_layers=2, n_heads=2,
                             d_head=8, csa_m=2, hca_m=4, top_k_csa=2, n_win=4,
                             d_latent_q=8, n_groups=1, n_experts=4, n_shared=1,
                             top_k_experts=2, d_ff=32, mem_dim=16, max_mem=4,
                             mem_seed_slots=4, mem_read_rank=4, sinkhorn_iters=5,
                             max_seq_len=2048)
    model = ThoughtBankLM(mcfg)
    model.rti_retriever = Retriever(32)
    model.rti_type = InjectType(32)
    model.rti_copy = CopyHead(32, CopyHeadConfig(enabled=True))
    tok = _Tok()
    env = RecallEnvConfig(life_seed=1, n_facts=(2, 3), n_probes=(1, 2),
                          filler_per_fact=(1, 1), max_groups=4, inject_groups=2)
    stream = RecallEnvStream(tok, seed=0, cfg=env)
    rcfg = RtiConfig(top_k=3, max_groups=4, train_groups=2, eval_groups=2)
    sif_w = sif_table(stream, V, rcfg.sif_a)
    rti = RtiRunner(rcfg, sif_w, sep_id=4, n_lanes=1, seed=1)

    res = run(model, stream, tok, rti, rcfg, "cpu", lives=4, warmup=2,
              max_new=4, amp=False, use_cache=False, copy_head=model.rti_copy,
              sep_id=4, decode_arms=("A_rti", "B_text"), verbose=False)
    assert res["rows"], "aucune sonde mesurée — l'env ou l'injection est morte"

    # (1) L'INVARIANT DE COMPARABILITÉ : les positions supervisées sont les
    #     mêmes dans tous les bras (masse de la CE identique au bit). Sans lui,
    #     comparer des nll bras à bras n'aurait aucun sens.
    dens = {k: res["arms"][k]["n_val_tok"] + res["arms"][k]["n_tpl_tok"]
            for k in ARMS}
    assert len({round(v, 6) for v in dens.values()}) == 1, dens
    assert dens["A_rti"] > 0

    # (2) partition VALEUR/TEMPLATE : val + tpl == réponse, au bit.
    for k in ARMS:
        v = res["arms"][k]
        tot = (v["val_nll"] * v["n_val_tok"] + v["tpl_nll"] * v["n_tpl_tok"])
        assert abs(tot / (v["n_val_tok"] + v["n_tpl_tok"])
                   - v["ans_nll"]) < 1e-4, k

    # (3) le préfixe TEXTE de B porte EXACTEMENT autant de positions que le
    #     préfixe injecté de A (mêmes ids, séparateur substitué) : le forward
    #     de réponse des deux bras traverse le MÊME nombre de tokens. C'est
    #     tout le kill-test 1 — seule la NATURE du préfixe change.
    ar = res["arms"]
    assert ar["B_text"]["ans_tokens"] == ar["A_rti"]["ans_tokens"], \
        (ar["B_text"]["ans_tokens"], ar["A_rti"]["ans_tokens"])
    assert ar["A_rti"]["ans_tokens"] - ar["C_off"]["ans_tokens"] == \
        len(res["rows"]) * rcfg.eval_groups * rcfg.group_prefix
    n_g = ar["B_text"]["n_graded"]
    assert n_g == len(res["rows"]), (n_g, len(res["rows"]))

    # (4) la comptabilité : A paie un forward de PLUS (la requête, qui produit
    #     h_query) ; C n'a aucun préfixe ; D porte tous les faits résidents,
    #     donc le préfixe le plus long.
    assert ar["A_rti"]["q_calls"] > 0 and ar["C_off"]["q_calls"] == 0
    assert ar["A_rti"]["calls"] == ar["A_rti"]["ans_calls"] + ar["A_rti"]["q_calls"]
    assert ar["C_off"]["ans_tokens"] < ar["B_text"]["ans_tokens"] \
        < ar["D_think"]["ans_tokens"]

    # (5) la porte de copie n'est renseignée QUE sur le bras A (les autres n'ont
    #     pas de préfixe injecté : une valeur non nulle serait une fuite).
    assert ar["A_rti"]["p_copy_val"] > 0
    for k in ("B_text", "B_nat", "C_off", "D_think"):
        assert not (ar[k]["p_copy_val"] > 0), k

    # (6) REJOUABILITÉ : deux passes identiques rendent les mêmes nll.
    rti2 = RtiRunner(rcfg, sif_w, sep_id=4, n_lanes=1, seed=1)
    stream2 = RecallEnvStream(tok, seed=0, cfg=env)
    res2 = run(model, stream2, tok, rti2, rcfg, "cpu", lives=4, warmup=2,
               max_new=4, amp=False, use_cache=False,
               copy_head=model.rti_copy, sep_id=4,
               decode_arms=("A_rti", "B_text"), verbose=False)
    for k in ARMS:
        assert abs(res["arms"][k]["val_nll"] - res2["arms"][k]["val_nll"]) < 1e-6, k

    # (7) statistiques : Wilson borné, Δ apparié fini.
    lo, hi = wilson(3, 10)
    assert 0.0 <= lo < 0.3 < hi <= 1.0
    d = paired_delta(res["rows"], "A_rti", "B_text", "val_nll")
    assert d["n"] == len(res["rows"]) and math.isfinite(d["delta"])

    print(f"kt12 self-test OK — {len(res['rows'])} sondes, "
          f"{dens['A_rti']:.0f} tokens supervisés identiques sur les 5 bras, "
          f"partition val/tpl exacte, comptabilité et rejouabilité vérifiées")


# ── CLI ──────────────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt")
    ap.add_argument("--config",
                    default="deepseek_v4_mini/configs/sft_recall_350m_copy.yaml")
    ap.add_argument("--lives", type=int, default=120)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--life0", type=int, default=100000,
                    help="décalage des vies : disjoint des seeds d'entraînement")
    ap.add_argument("--life-seed", type=int, default=20260803)
    ap.add_argument("--max-new", type=int, default=96)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-decode", action="store_true",
                    help="nll seule (pas de grade) — passe de cadence")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--out", default="")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        _selftest()
        return
    assert args.ckpt, "--ckpt requis"

    model, tok, mcfg, rcfg, raw, step = build(args.ckpt, args.config,
                                              args.device)
    gen = ((raw.get("chat") or {}).get("eval_gen") or {}).get("gen") or {}
    per = next((s.get("gen") or {} for s in (gen.get("streams") or [])
                if s.get("stream") == "persona"), {})
    env = RecallEnvConfig(
        life_seed=args.life_seed, max_groups=rcfg.max_groups,
        inject_groups=rcfg.eval_groups,
        value_weight=float(per.get("value_weight", 4.0)),
        real_filler=str(per.get("real_filler", "")),
        real_cap=int(per.get("real_cap", 20000)),
        real_max_tok=int(per.get("real_max_tok", 96)),
        real_cache_dir=os.path.expandvars(str(per.get("real_cache_dir", ""))),
        p_real=float(per.get("p_real", 0.8)),
        surprisal_mode=str(per.get("surprisal_mode", "sif")),
        sif_a=float(per.get("sif_a", 1e-4)))
    stream = RecallEnvStream(tok, seed=args.life_seed, cfg=env)
    sif_w = sif_table(stream, mcfg.vocab_size, rcfg.sif_a)
    sep_id = tok.convert_tokens_to_ids(rcfg.sep_token)
    rti = RtiRunner(rcfg, sif_w, sep_id, 1, seed=4242)
    amp = bool((raw.get("training") or {}).get("amp", True)) \
        and args.device.startswith("cuda")

    print(f"ckpt {args.ckpt} (step {step}) | {model.num_params():,} params | "
          f"rti top_k {rcfg.top_k} groups {rcfg.eval_groups} "
          f"(préfixe {rcfg.eval_groups * rcfg.group_prefix} pseudo-tokens) "
          f"sép {rcfg.sep_token}={sep_id} | copy-head "
          f"{'ON' if hasattr(model, 'rti_copy') else 'OFF'} | "
          f"env life_seed {env.life_seed} lives {args.lives} "
          f"(+{args.warmup} de chauffe) depuis {args.life0}", flush=True)

    dec = () if args.no_decode else ("A_rti", "B_text", "B_nat", "D_think")
    res = run(model, stream, tok, rti, rcfg, args.device, lives=args.lives,
              warmup=args.warmup, life0=args.life0, max_new=args.max_new,
              amp=amp, decode_arms=dec, use_cache=not args.no_cache,
              copy_head=getattr(model, "rti_copy", None), sep_id=sep_id,
              exec_timeout=env.exec_timeout)
    report(res)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"ckpt": args.ckpt, "step": step,
                       "config": args.config, "env": env.to_dict(),
                       "rti": {"top_k": rcfg.top_k, "eval_groups":
                               rcfg.eval_groups, "max_groups": rcfg.max_groups},
                       "lives": args.lives, "life0": args.life0,
                       "max_new": args.max_new, "decode_arms": list(dec),
                       **res}, f, indent=1)
        print(f"\nécrit : {args.out}")


if __name__ == "__main__":
    main()
