#!/usr/bin/env python
"""Census de dispatch + chrono du décodage token-par-token.

Le décodage du 386M est LAUNCH-BOUND : ~11.5k ops aten par forward pour ~10 ms
de calcul GPU utile (FINDINGS 2026-07-25 : 218 ms/token sans cache KV, ~137 ms
avec). Ce bench mesure le NOMBRE de dispatches — la quantité à réduire — et
prépare le chrono GPU pour quand un GPU sera libre.

Trois modes, cumulables :

  * census (défaut)   : compte d'ops aten par forward, top-N par op et par
                        module (TorchDispatchMode + pile de hooks). Le compte ne
                        dépend PAS des largeurs (FINDINGS) : le mode --toy garde
                        les paramètres STRUCTURAUX du yaml (couches, itérations,
                        experts, top-k, fenêtres) et écrase les largeurs — le
                        chiffre obtenu sur CPU est celui du 386M.
  * --fingerprint     : sha256 des logits float64 de chaque pas (greedy, banque
                        explicite seedée par --bank-seed — le seed_bank interne
                        tire sur le RNG global à chaque token, il n'est pas
                        reproductible). Trois lancements — branche mère, branche
                        perf flags OFF, flags ON — doivent rendre le même hash :
                        c'est la preuve « bit-identique » exécutable.
  * --ab-check        : A/B float64 sur le vrai stack, flags OFF vs ON à poids
                        IDENTIQUES, top-k durs neutralisés PUIS actifs
                        (protocole de decode._selftest). torch.equal exigé.

Le chrono GPU (--time --device cuda) refuse de tourner si un process occupe
déjà le GPU (règle un-run-par-GPU) ; --force pour outrepasser SCIEMMENT.

Usage :
  python scripts/bench_decode.py deepseek_v4_mini/configs/v350_phase1_10b.yaml
  python scripts/bench_decode.py <cfg> --fingerprint --flags all
  python scripts/bench_decode.py <cfg> --ab-check --flags decode_fuse
  python scripts/bench_decode.py <cfg> --real --ckpt <pt> --time --device cuda
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import os
import subprocess
import sys
import time
from collections import Counter

sys.path.insert(0, os.getcwd())

import torch
from torch.utils._python_dispatch import TorchDispatchMode

from deepseek_v4_mini.config import ThoughtBankConfig
from deepseek_v4_mini.decode import generate
from deepseek_v4_mini.model import ThoughtBankLM
from deepseek_v4_mini.paths import load_yaml

# Paramètres STRUCTURAUX : ils fixent le nombre de dispatches (boucles Python,
# itérations, branches). Les autres (d_model, d_ff, vocab…) ne fixent que les
# largeurs — sans effet sur le COMPTE d'ops, écrasés en mode --toy.
_STRUCTURAL = (
    "n_layers", "n_heads", "n_hc", "sinkhorn_iters", "sinkhorn_closed_form",
    "csa_m", "hca_m", "top_k_csa", "n_win", "n_groups",
    "n_experts", "n_shared", "top_k_experts",
    "max_mem", "mem_seed_slots", "mem_read_swiglu", "mem_read_layers",
    "mem_read_spectral_norm", "use_dual_stream",
)
_TOY_WIDTHS = dict(vocab_size=61, d_model=48, d_head=8, d_latent_q=8,
                   d_ff=48, mem_dim=24, mem_read_rank=4, max_seq_len=2048,
                   dropout=0.0)

# Les flags de chemin de décodage, appliqués si présents dans la dataclass
# (le bench doit tourner tel quel sur la branche mère, qui ne les a pas).
_DECODE_FLAGS = ("decode_fuse", "decode_dense_moe", "decode_static_cache")


# ── census ───────────────────────────────────────────────────────────────────

class OpCensus(TorchDispatchMode):
    """Compte les ops aten, attribuées au module le plus INTERNE en cours de
    forward (pile posée par des hooks) et au forward modèle en cours (index
    posé par un pre-hook sur le modèle racine ; -1 = boucle de génération)."""

    def __init__(self):
        super().__init__()
        self.stack: list[str] = []
        self.fwd_idx = -1
        self.frames: list[tuple[Counter, Counter]] = []  # (par op, par module)
        self.outside = (Counter(), Counter())            # fwd_idx == -1

    def _frame(self):
        return self.frames[self.fwd_idx] if self.fwd_idx >= 0 else self.outside

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        by_op, by_mod = self._frame()
        by_op[func.overloadpacket.__name__] += 1
        by_mod[self.stack[-1] if self.stack else "<hors module>"] += 1
        return func(*args, **(kwargs or {}))


def _instrument(model: torch.nn.Module, census: OpCensus) -> list:
    """Pile de classes de modules + bornes de forward sur le modèle racine."""
    handles = []

    def _root_pre(_m, _inp):
        census.fwd_idx = len(census.frames)
        census.frames.append((Counter(), Counter()))

    def _root_post(_m, _inp, _out):
        census.fwd_idx = -1

    handles.append(model.register_forward_pre_hook(_root_pre))
    handles.append(model.register_forward_hook(_root_post))
    for mod in model.modules():
        if mod is model:
            continue
        name = type(mod).__name__

        def _pre(_m, _inp, _name=name):
            census.stack.append(_name)

        def _post(_m, _inp, _out, _name=name):
            census.stack.pop()

        handles.append(mod.register_forward_pre_hook(_pre))
        handles.append(mod.register_forward_hook(_post))
    return handles


# ── construction modèle ──────────────────────────────────────────────────────

def _mk_cfg(raw: dict, toy: bool, flags: list[str]) -> ThoughtBankConfig:
    mcfg = dict(raw["model"])
    if toy:
        mcfg = {k: v for k, v in mcfg.items() if k in _STRUCTURAL}
        mcfg.update(_TOY_WIDTHS)
    known = ThoughtBankConfig.__dataclass_fields__
    for f in flags:
        if f in known:
            mcfg[f] = True
        else:
            print(f"[bench] flag `{f}` inconnu de ThoughtBankConfig — ignoré "
                  f"(normal sur la branche mère)", flush=True)
    return ThoughtBankConfig(**mcfg)


def _mk_model(raw: dict, a, flags: list[str], dtype=None):
    """Modèle + prefix + banque, seedés — le triplet de tout bras de mesure."""
    if not a.toy:
        # Pattern eval_rl_ckpt : tokenizer réel (vocab), ckpt optionnel.
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(raw["tokenizer"])
        add = [x for x in ("<think>", "<blank>") if x not in tok.get_vocab()]
        if add:
            tok.add_special_tokens({"additional_special_tokens": add})
        cfg = _mk_cfg(raw, False, flags)
        cfg = dataclasses.replace(cfg, vocab_size=len(tok))
    else:
        cfg = _mk_cfg(raw, True, flags)

    torch.manual_seed(a.seed)
    model = ThoughtBankLM(cfg).eval()
    if a.ckpt:
        ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
        sd = ck.get("model", ck)
        model.load_state_dict({k.replace("_orig_mod.", ""): v
                               for k, v in sd.items()})
    for p in model.parameters():
        p.requires_grad_(False)
    if dtype is not None:
        model = model.to(dtype)
    model = model.to(a.device)

    g = torch.Generator().manual_seed(a.bank_seed)
    bank = torch.rand(a.batch, cfg.max_mem, cfg.mem_dim, generator=g)
    prefix = torch.randint(0, cfg.vocab_size, (a.batch, a.prefix), generator=g)
    bank = bank.to(a.device, dtype or torch.float32)
    return model, cfg, prefix.to(a.device), bank


# ── bras de mesure ───────────────────────────────────────────────────────────

def _census_arm(model, prefix, bank, tokens: int, use_cache: bool):
    census = OpCensus()
    handles = _instrument(model, census)
    try:
        with census:
            generate(model, prefix, bank=bank, max_new=tokens,
                     stop_id=None, use_cache=use_cache)
    finally:
        for h in handles:
            h.remove()
    return census


def _steady(census: OpCensus, use_cache: bool):
    """Ops/forward en régime établi. Avec cache le forward 0 avale le préfixe
    et un forward sur m ferme un bloc : on prend le DERNIER forward (aucun
    événement rare ne tombe dessus si tokens % csa_m != 1) et on donne le
    min/max des forwards de décodage pour montrer la dispersion."""
    frames = census.frames[1:] if use_cache and len(census.frames) > 1 \
        else census.frames
    counts = [sum(by_op.values()) for by_op, _ in frames]
    by_op, by_mod = frames[-1]
    return by_op, by_mod, counts


def _top(counter: Counter, n: int) -> list[tuple[str, int]]:
    return counter.most_common(n)


def _fmt_table(rows, headers, md: bool) -> str:
    if md:
        out = ["| " + " | ".join(headers) + " |",
               "|" + "|".join("---" for _ in headers) + "|"]
        out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
        return "\n".join(out)
    w = [max(len(str(x)) for x in [h] + [r[i] for r in rows])
         for i, h in enumerate(headers)]
    out = ["  ".join(h.ljust(w[i]) for i, h in enumerate(headers))]
    out += ["  ".join(str(c).ljust(w[i]) for i, c in enumerate(r)) for r in rows]
    return "\n".join(out)


def run_census(raw, a, md: bool):
    flags = _parse_flags(a.flags)
    model, cfg, prefix, bank = _mk_model(raw, a, flags)
    arms = {"both": (False, True), "off": (False,), "on": (True,)}[a.cache]
    print(f"\n== census d'ops aten ({'toy' if a.toy else 'réel'}, "
          f"{cfg.n_layers} couches, flags={flags or 'aucun'}) ==")
    for use_cache in arms:
        census = _census_arm(model, prefix, bank, a.tokens, use_cache)
        by_op, by_mod, counts = _steady(census, use_cache)
        total = sum(by_op.values())
        label = "avec cache KV" if use_cache else "sans cache"
        print(f"\n-- {label} : {total} ops/forward (régime établi ; "
              f"min {min(counts)} / max {max(counts)} sur {len(counts)} fwd) --")
        print(_fmt_table(_top(by_op, a.top), ["op aten", "n/fwd"], md))
        print()
        print(_fmt_table(_top(by_mod, a.top), ["module", "n/fwd"], md))
        ob, _ = census.outside
        print(f"(boucle generate hors forward : {sum(ob.values())} ops "
              f"pour {a.tokens} tokens)")

    # chrono CPU indicatif (le vrai chrono est --time sur GPU)
    for use_cache in arms:
        t0 = time.perf_counter()
        generate(model, prefix, bank=bank, max_new=a.tokens,
                 stop_id=None, use_cache=use_cache)
        dt = time.perf_counter() - t0
        print(f"chrono CPU {'cache' if use_cache else 'nocache'} : "
              f"{a.tokens / dt:.2f} tok/s ({1e3 * dt / a.tokens:.1f} ms/token)")


# ── fingerprint ──────────────────────────────────────────────────────────────

def _greedy_logits(model, prefix, bank, tokens: int, use_cache: bool):
    """La boucle de generate, en gardant les logits float64 de chaque pas."""
    cache = model.make_cache() if use_cache else None
    fed, out, logits = prefix, prefix, []
    with torch.no_grad():
        for _ in range(tokens):
            o = model(fed, init_mem=bank, write=False,
                      **({"cache": cache} if cache is not None else {}))
            lg = o["logits"][:, -1].double()
            logits.append(lg)
            nt = lg.argmax(-1, keepdim=True)
            out = torch.cat([out, nt], dim=1)
            fed = nt if cache is not None else out
    return out[:, prefix.size(1):], torch.stack(logits, dim=1)


def run_fingerprint(raw, a):
    flags = _parse_flags(a.flags)
    model, cfg, prefix, bank = _mk_model(raw, a, flags, dtype=torch.float64)
    prefix, bank = prefix.cpu(), bank.cpu().double()
    model = model.cpu()
    print(f"\n== fingerprint float64 CPU (flags={flags or 'aucun'}, "
          f"seed={a.seed}, bank_seed={a.bank_seed}) ==")
    for use_cache in ((False, True) if a.cache == "both"
                      else ((a.cache == "on"),)):
        toks, logits = _greedy_logits(model, prefix, bank, a.tokens, use_cache)
        h = hashlib.sha256()
        h.update(toks.cpu().numpy().tobytes())
        h.update(logits.cpu().contiguous().numpy().tobytes())
        print(f"cache={'on ' if use_cache else 'off'}  sha256={h.hexdigest()}")


# ── A/B flags OFF vs ON ──────────────────────────────────────────────────────

def run_ab(raw, a):
    flags = _parse_flags(a.flags)
    known = [f for f in flags if f in ThoughtBankConfig.__dataclass_fields__]
    if not known:
        print("[bench] --ab-check sans flag connu : rien à comparer "
              "(sur la branche mère c'est attendu)", flush=True)
        return
    print(f"\n== A/B float64 CPU : flags OFF vs ON ({known}) ==")
    base = dict(raw["model"])
    for neutral in (True, False):
        mcfg = {k: v for k, v in base.items() if k in _STRUCTURAL}
        mcfg.update(_TOY_WIDTHS)
        if neutral:  # protocole decode._selftest : aucun top-k dur ne tranche
            mcfg["top_k_experts"] = mcfg.get("n_experts", 8)
            mcfg["top_k_csa"] = 64
        torch.manual_seed(a.seed)
        m_off = ThoughtBankLM(ThoughtBankConfig(**mcfg)).double().eval()
        m_on = ThoughtBankLM(ThoughtBankConfig(
            **mcfg, **{f: True for f in known})).double().eval()
        m_on.load_state_dict(m_off.state_dict())
        g = torch.Generator().manual_seed(a.bank_seed)
        # B=1 exerce le chemin dense du MoE (gate BT==1) ; B=2 le chemin boucle
        for Bt in (1, 2):
            bank = torch.rand(Bt, mcfg["max_mem"], _TOY_WIDTHS["mem_dim"],
                              generator=g).double()
            for T0 in (1, 2, 5, 11, 17):
                pr = torch.randint(0, 61, (Bt, T0), generator=g)
                for use_cache in (False, True):
                    ta, la = _greedy_logits(m_off, pr, bank, 13, use_cache)
                    tb, lb = _greedy_logits(m_on, pr, bank, 13, use_cache)
                    assert torch.equal(ta, tb) and torch.equal(la, lb), (
                        f"A/B DIVERGE (neutral={neutral}, B={Bt}, T0={T0}, "
                        f"cache={use_cache})\n  OFF: {ta}\n  ON : {tb}")
        print(f"top-k {'neutralisés' if neutral else 'DURS'} : OFF == ON "
              f"(tokens ET logits, B∈{{1,2}} × 5 préfixes × cache on/off)")


# ── chrono GPU (gardé) ───────────────────────────────────────────────────────

def _gpu_guard(force: bool):
    """Un seul run par GPU — refuse si quoi que ce soit tourne déjà.
    Sous WSL2 nvidia-smi peut être aveugle : on double avec pgrep."""
    busy = []
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10).stdout.strip()
        if out:
            busy.append(f"nvidia-smi: {out}")
    except (OSError, subprocess.TimeoutExpired) as e:
        busy.append(f"nvidia-smi injoignable ({e}) — impossible de prouver "
                    f"que le GPU est libre")
    pg = subprocess.run(
        ["pgrep", "-af", "deepseek_v4_mini|rl_disagg|code_defer_native"],
        capture_output=True, text=True).stdout.strip()
    pg = "\n".join(l for l in pg.splitlines() if str(os.getpid()) not in l.split()[:1])
    if pg:
        busy.append(f"pgrep: {pg}")
    if busy and not force:
        sys.exit("[bench] GPU refusé (règle un-run-par-GPU) :\n  "
                 + "\n  ".join(busy) + "\n--force pour outrepasser SCIEMMENT.")
    if busy:
        print("[bench] --force : on passe outre :\n  " + "\n  ".join(busy))


def run_time(raw, a):
    if a.device == "cuda":
        _gpu_guard(a.force)
        if not torch.cuda.is_available():
            sys.exit("[bench] --time --device cuda : pas de CUDA visible.")
    flags = _parse_flags(a.flags)
    model, cfg, prefix, bank = _mk_model(raw, a, flags)
    arms = {"both": (False, True), "off": (False,), "on": (True,)}[a.cache]
    print(f"\n== chrono {a.device} ({'toy' if a.toy else 'réel'}, "
          f"B={a.batch}, {a.tokens} tokens, flags={flags or 'aucun'}) ==")
    for use_cache in arms:
        generate(model, prefix, bank=bank, max_new=4, stop_id=None,
                 use_cache=use_cache)                       # warmup
        if a.device == "cuda":
            torch.cuda.synchronize()
            e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
            e0.record()
            generate(model, prefix, bank=bank, max_new=a.tokens,
                     stop_id=None, use_cache=use_cache)
            e1.record()
            torch.cuda.synchronize()
            dt = e0.elapsed_time(e1) / 1e3
        else:
            t0 = time.perf_counter()
            generate(model, prefix, bank=bank, max_new=a.tokens,
                     stop_id=None, use_cache=use_cache)
            dt = time.perf_counter() - t0
        print(f"cache={'on ' if use_cache else 'off'}  "
              f"{a.tokens / dt:.2f} tok/s  ({1e3 * dt / a.tokens:.1f} ms/token)")


# ── CLI ──────────────────────────────────────────────────────────────────────

def _parse_flags(spec: str) -> list[str]:
    if spec == "none":
        return []
    if spec == "all":
        return list(_DECODE_FLAGS)
    return [f.strip() for f in spec.split(",") if f.strip()]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("config", help="yaml imbriqué (bloc model:), ${TB_ROOT} expansé")
    p.add_argument("--ckpt", default="", help="checkpoint à charger (mode --real)")
    p.add_argument("--tokens", type=int, default=24)
    p.add_argument("--prefix", type=int, default=33)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    p.add_argument("--real", dest="toy", action="store_false",
                   help="vraies largeurs + tokenizer (défaut : jouet structurel)")
    p.add_argument("--cache", default="both", choices=("both", "on", "off"))
    p.add_argument("--flags", default="none",
                   help="none | all | liste: decode_fuse,decode_dense_moe,…")
    p.add_argument("--top", type=int, default=10)
    p.add_argument("--time", action="store_true", help="chrono (GPU gardé)")
    p.add_argument("--force", action="store_true")
    p.add_argument("--fingerprint", action="store_true")
    p.add_argument("--ab-check", dest="ab_check", action="store_true")
    p.add_argument("--report", default="text", choices=("text", "md"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--bank-seed", dest="bank_seed", type=int, default=1234)
    a = p.parse_args(argv)

    raw = load_yaml(a.config)
    if "model" not in raw:
        sys.exit(f"[bench] {a.config}: pas de bloc `model:` — ce bench lit le "
                 f"format imbriqué (cf. config.from_yaml pour le format plat).")

    # le census est le mode par défaut ; demander un autre mode le remplace
    if a.fingerprint:
        run_fingerprint(raw, a)
    if a.ab_check:
        run_ab(raw, a)
    if a.time:
        run_time(raw, a)
    if not (a.fingerprint or a.ab_check or a.time):
        run_census(raw, a, md=(a.report == "md"))


if __name__ == "__main__":
    main()
