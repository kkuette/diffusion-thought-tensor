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

from deepseek_v4_mini.infra.config import ThoughtBankConfig
from deepseek_v4_mini.infra.decode import generate
from deepseek_v4_mini.core.model import ThoughtBankLM
from deepseek_v4_mini.infra.paths import load_yaml

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
    for p in model.parameters():
        p.requires_grad_(False)
    if dtype is not None:
        model = model.to(dtype)
    # sur GPU AVANT le load : le load_state_dict copie alors NFS→GPU tenseur
    # par tenseur. Avec mmap, le pic RAM CPU reste ~1 tenseur — un torch.load
    # plein de 3+ Go sur un rig déjà chargé, c'est la tempête de swap
    # documentée dans gpu_worker.sh.
    model = model.to(a.device)
    if a.ckpt:
        try:
            ck = torch.load(a.ckpt, map_location="cpu", weights_only=False,
                            mmap=True)
        except (TypeError, RuntimeError):
            ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
        sd = ck.get("model", ck)
        model.load_state_dict({k.replace("_orig_mod.", ""): v
                               for k, v in sd.items()})
        del ck, sd

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
    Sous WSL2 nvidia-smi peut être aveugle : on double avec pgrep.

    Sous un slot de la ferme (CUDA_VISIBLE_DEVICES posé par gpu_worker.sh sur
    UN index), le jugement se limite à CE GPU : les autres cartes du rig
    portent leurs propres runs, et pgrep les verrait toujours."""
    busy = []
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    slot = cvd.isdigit()                      # un index nu = slot ferme
    smi = ["nvidia-smi"] + (["-i", cvd] if slot else []) + \
          ["--query-compute-apps=pid,process_name", "--format=csv,noheader"]
    try:
        out = subprocess.run(smi, capture_output=True, text=True,
                             timeout=10).stdout.strip()
        if out:
            busy.append(f"nvidia-smi: {out}")
    except (OSError, subprocess.TimeoutExpired) as e:
        busy.append(f"nvidia-smi injoignable ({e}) — impossible de prouver "
                    f"que le GPU est libre")
    if not slot:
        pg = subprocess.run(
            ["pgrep", "-af", "deepseek_v4_mini|rl_disagg|code_defer_native"],
            capture_output=True, text=True).stdout.strip()
        pg = "\n".join(l for l in pg.splitlines()
                       if str(os.getpid()) not in l.split()[:1])
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

    if a.cuda_graphs:
        # bras CUDA graphs : GraphDecodeRunner (B=1, greedy). Sur CPU il
        # dégrade en eager et le dit — c'est le « préparé, pas lancé ».
        from deepseek_v4_mini.infra.decode_graphs import GraphDecodeRunner
        amp = None if a.amp == "none" else a.amp
        runner = GraphDecodeRunner(model, bank, amp=amp)
        if runner.eager_only:
            print("[bench] --cuda-graphs sur CPU : dégradation eager annoncée "
                  "(la capture attend un GPU libre)")
        # HORS chrono : préfixe, warmup mono-token, cycle de capture — le
        # chrono ne mesure que le régime établi. (Le 23,4 ms/token du
        # 2026-07-27 amortissait warmup + captures sur les 192 tokens.)
        o = runner.step(prefix)
        fed = o["logits"][:, -1].argmax(-1, keepdim=True)
        budget = runner.warmup + 3 * runner.lcm + 8
        while (not runner.eager_only and len(runner.graphs) < runner.lcm
               and budget > 0):
            o = runner.step(fed)
            fed = o["logits"][:, -1].argmax(-1, keepdim=True)
            budget -= 1
        label = "graphs+bf16" if amp else "graphs"

        def _clock(fn):
            if a.device == "cuda":
                torch.cuda.synchronize()
                e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
                e0.record()
                fn()
                e1.record()
                torch.cuda.synchronize()
                return e0.elapsed_time(e1) / 1e3
            t0 = time.perf_counter()
            fn()
            return time.perf_counter() - t0

        state = {"fed": fed}

        def _steps():
            f = state["fed"]
            for _ in range(a.tokens):
                o = runner.step(f)
                f = o["logits"][:, -1].argmax(-1, keepdim=True)
            state["fed"] = f

        dt = _clock(_steps)
        print(f"{label}/step   {a.tokens / dt:.2f} tok/s  "
              f"({1e3 * dt / a.tokens:.1f} ms/token)"
              + ("  [eager fallback]" if runner.eager_only else ""))
        # bras chaîné : argmax DANS le graph, zéro aller-retour host par token
        if not runner.eager_only and len(runner.graphs) == runner.lcm:
            dt = _clock(lambda: runner._chain(state["fed"], a.tokens))
            print(f"{label}/chain  {a.tokens / dt:.2f} tok/s  "
                  f"({1e3 * dt / a.tokens:.1f} ms/token)")
        runner.close()

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


# ── verify GPU (runner graphs vs eager largeur-pleine) ───────────────────────

def _full_eager_ref(model, bank, prefix, n_tokens, warmup):
    """Le programme du graph étendu, exécuté EAGER en largeur pleine sur GPU —
    la référence qui attrape la classe « couture du cycle » : un graph qui lit
    des adresses périmées diverge d'elle, alors que chain vs step (deux
    runners graphs) gèlent les mêmes adresses et restent d'accord entre eux."""
    from deepseek_v4_mini import attention
    from deepseek_v4_mini.infra.decode_graphs import GraphDecodeRunner
    assert n_tokens > warmup + 1, "verify : tokens ≤ warmup, rien à comparer"
    r = GraphDecodeRunner(model, bank, warmup=warmup)
    toks = []
    with torch.no_grad():
        o = r.step(prefix)
        fed = o["logits"][:, -1].argmax(-1, keepdim=True)
        toks.append(fed)
        while r._singles < r.warmup:            # même gate que decode() : la
            o = r.step(fed)                     # trajectoire s'aligne 1:1
            fed = o["logits"][:, -1].argmax(-1, keepdim=True)
            toks.append(fed)
        r.eager_only = True                     # PAS de capture : eager pur
        r._enter_full()
        r._in_buf.copy_(fed)
        r._wptr.zero_()
        rest = n_tokens - len(toks)
        for _ in range(rest):
            attention._ROPE_OVERRIDE = r._rope_bufs
            try:
                r._graph_head()
                o = r._eager(r._in_buf)
                r._graph_tail(o)
            finally:
                attention._ROPE_OVERRIDE = None
        out = torch.cat([torch.cat(toks, dim=1),
                         r._out_toks[:rest].t()], dim=1).clone()
    r.close()
    return out


def run_verify(raw, a):
    """A/B GPU des replays réels : graphs (chain ET step) vs eager largeur
    pleine — trois trajectoires greedy du même préfixe, torch.equal exigé.
    Ni le chrono ni le self-test CPU (émulation eager) ne voient cette classe
    de bugs. fp32 seulement (sous autocast, eager et graph recastent
    différemment — la référence n'est plus comparable)."""
    if a.device != "cuda":
        sys.exit("[bench] --verify : GPU requis (on teste les replays réels)")
    _gpu_guard(a.force)
    if a.amp != "none":
        sys.exit("[bench] --verify : fp32 seulement (référence eager comparable)")
    from deepseek_v4_mini.infra.decode import trim
    from deepseek_v4_mini.infra.decode_graphs import GraphDecodeRunner
    flags = _parse_flags(a.flags)
    model, cfg, prefix, bank = _mk_model(raw, a, flags)
    N, warm = a.tokens, 8
    print(f"\n== verify GPU (B={a.batch}, {N} tokens/bras) ==")
    ref = _full_eager_ref(model, bank, prefix, N, warm)

    fails = 0
    runs = {}
    for label, chain in (("chain", True), ("step", False)):
        r = GraphDecodeRunner(model, bank, chain=chain, warmup=warm)
        g, l = r.decode(prefix, max_new=N)
        armed = not r.eager_only and len(r.graphs) == r.lcm
        r.close()
        del r
        torch.cuda.empty_cache()
        if not armed:
            print(f"{label:>6} : graphs INACTIFS — test nul")
            fails += 1
            continue
        same = torch.equal(g, ref)              # trajectoires alignées 1:1
        runs[label] = g
        print(f"{label:>6} : "
              + ("OK — == eager largeur pleine au bit" if same
                 else "DIVERGE de la référence eager"))
        if not same:
            d = (g != ref).any(0).int().argmax().item()
            print(f"        première divergence au token {d}/{g.size(1)}\n"
                  f"        graphs : {g[:, max(0, d - 2):d + 3]}\n"
                  f"        eager  : {ref[:, max(0, d - 2):d + 3]}")
            fails += 1

    # contrat stop_id : chaîne par tranches vs pilotage step, lens compris
    if "chain" in runs:
        sid = int(runs["chain"][0, ref.size(1) * 2 // 3])
        r3 = GraphDecodeRunner(model, bank, chain=True, warmup=warm)
        g3, l3 = r3.decode(prefix, max_new=N, stop_id=sid, chunk=8)
        r3.close(); del r3; torch.cuda.empty_cache()
        r4 = GraphDecodeRunner(model, bank, chain=False, warmup=warm)
        g4, l4 = r4.decode(prefix, max_new=N, stop_id=sid)
        r4.close(); del r4; torch.cuda.empty_cache()
        ok = torch.equal(l3, l4) and all(
            torch.equal(x, y) for x, y in zip(trim(g3, l3), trim(g4, l4)))
        print(f"  stop : {'OK — tranches == step (lens + lignes trimées)' if ok else 'DIVERGE'}"
              f"  (stop_id={sid}, lens={l3.tolist()})")
        fails += 0 if ok else 1
    print("verify:", "TOUT OK" if fails == 0 else f"{fails} ÉCHEC(S)")


# ── rebind (verify au bit + chrono du coût par appel) ────────────────────────

def _rebound_eager_ref(model, bank, prefix, n_tokens):
    """La trajectoire attendue d'un décodage POST-REBIND, exécutée eager en
    largeur pleine : préfixe eager, singles eager tant que la garde de régime
    établi retient (`_enter_ready` — zéro warmup, il est déjà payé), puis le
    programme du graph émulé. C'est le calendrier exact du runner rebondi —
    un runner frais (warmup plein) suivrait un AUTRE calendrier, ULP-divergent."""
    from deepseek_v4_mini import attention
    from deepseek_v4_mini.infra.decode_graphs import GraphDecodeRunner
    r = GraphDecodeRunner(model, bank, warmup=0)
    r.eager_only = True                         # jamais de capture : eager pur
    toks = []
    with torch.no_grad():
        o = r.step(prefix)
        fed = o["logits"][:, -1].argmax(-1, keepdim=True)
        toks.append(fed)
        while not r._enter_ready():
            o = r.step(fed)
            fed = o["logits"][:, -1].argmax(-1, keepdim=True)
            toks.append(fed)
        r._enter_full()
        r._in_buf.copy_(fed)
        r._wptr.zero_()
        rest = n_tokens - len(toks)
        for _ in range(rest):
            attention._ROPE_OVERRIDE = r._rope_bufs
            try:
                r._graph_head()
                o = r._eager(r._in_buf)
                r._graph_tail(o)
            finally:
                attention._ROPE_OVERRIDE = None
        out = torch.cat([torch.cat(toks, dim=1),
                         r._out_toks[:rest].t()], dim=1).clone()
    r.close()
    return out, len(toks)


def run_rebind(raw, a):
    """Le rebind, prouvé puis chronométré sur les replays réels.

    Verify : decode 1 (arme + capture), rebind(banque neuve), decode 2 —
    torch.equal contre la trajectoire eager largeur-pleine au MÊME calendrier
    (`_rebound_eager_ref`). Deux préfixes : COURT (~6 tokens, le cas a_open
    des rollouts — la garde de régime établi doit retenir puis converger) et
    long (a.prefix). Plus le contrat stop_id post-rebind (chain vs step).

    Chrono : K appels successifs `rebind + decode` sur banques différentes vs
    le coût d'armement d'un runner FRAIS — le surcoût par appel est ce qui
    décide si le gain graphs existe en rollout."""
    if a.device != "cuda":
        sys.exit("[bench] --rebind : GPU requis (on teste les replays réels)")
    _gpu_guard(a.force)
    if a.amp != "none":
        sys.exit("[bench] --rebind : fp32 seulement (référence eager comparable)")
    from deepseek_v4_mini.infra.decode import trim
    from deepseek_v4_mini.infra.decode_graphs import GraphDecodeRunner
    flags = _parse_flags(a.flags)
    model, cfg, prefix, bank = _mk_model(raw, a, flags)
    N, warm = a.tokens, 8
    g2 = torch.Generator().manual_seed(a.bank_seed + 1)
    bank2 = torch.rand(a.batch, cfg.max_mem, cfg.mem_dim,
                       generator=g2).to(a.device)
    print(f"\n== rebind GPU (B={a.batch}, {N} tokens/bras) ==")

    fails = 0
    for label, pr2 in (("préfixe court", prefix[:, :6]),
                       ("préfixe long ", prefix)):
        r = GraphDecodeRunner(model, bank, warmup=warm)
        r.decode(prefix, max_new=N)             # decode 1 : arme + capture
        armed = not r.eager_only and len(r.graphs) == r.lcm
        if not armed:
            print(f"{label} : graphs INACTIFS après decode 1 — test nul")
            fails += 1
            r.close(); del r; torch.cuda.empty_cache()
            continue
        r.rebind(bank2)
        g, l = r.decode(pr2, max_new=N)         # decode 2 : replays
        r.close(); del r; torch.cuda.empty_cache()
        ref, n_eager = _rebound_eager_ref(model, bank2, pr2, N)
        same = torch.equal(g, ref)
        print(f"{label} : "
              + (f"OK — decode 2 == eager largeur pleine au bit "
                 f"({n_eager} tokens eager avant bascule)" if same
                 else "DIVERGE de la référence eager"))
        if not same:
            d = (g != ref).any(0).int().argmax().item()
            print(f"        première divergence au token {d}/{g.size(1)}\n"
                  f"        rebind : {g[:, max(0, d - 2):d + 3]}\n"
                  f"        eager  : {ref[:, max(0, d - 2):d + 3]}")
            fails += 1

    # contrat stop_id POST-rebind : chaîne par tranches vs pilotage step
    sid = None
    r3 = GraphDecodeRunner(model, bank, chain=True, warmup=warm)
    r3.decode(prefix, max_new=N)
    r3.rebind(bank2)
    g3, l3 = r3.decode(prefix[:, :6], max_new=N)
    sid = int(g3[0, N * 2 // 3])
    r3.rebind(bank2)
    g3, l3 = r3.decode(prefix[:, :6], max_new=N, stop_id=sid, chunk=8)
    r3.close(); del r3; torch.cuda.empty_cache()
    r4 = GraphDecodeRunner(model, bank, chain=False, warmup=warm)
    r4.decode(prefix, max_new=N)
    r4.rebind(bank2)
    g4, l4 = r4.decode(prefix[:, :6], max_new=N, stop_id=sid)
    r4.close(); del r4; torch.cuda.empty_cache()
    ok = torch.equal(l3, l4) and all(
        torch.equal(x, y) for x, y in zip(trim(g3, l3), trim(g4, l4)))
    print(f"  stop : {'OK — tranches == step post-rebind' if ok else 'DIVERGE'}"
          f"  (stop_id={sid}, lens={l3.tolist()})")
    fails += 0 if ok else 1
    print("rebind verify:", "TOUT OK" if fails == 0 else f"{fails} ÉCHEC(S)")

    # ── chrono : armement frais vs rebind+decode ─────────────────────────────
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    r = GraphDecodeRunner(model, bank, warmup=32)
    r.decode(prefix, max_new=64)                # warmup + 16 captures + solde
    torch.cuda.synchronize()
    t_fresh = time.perf_counter() - t0
    K, t_reb, t_dec = 8, [], []
    gk = torch.Generator().manual_seed(a.bank_seed + 7)
    for _ in range(K):
        bk = torch.rand(a.batch, cfg.max_mem, cfg.mem_dim,
                        generator=gk).to(a.device)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        r.rebind(bk)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        r.decode(prefix[:, :6], max_new=N)
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        t_reb.append(t1 - t0)
        t_dec.append(t2 - t1)
    r.close(); del r; torch.cuda.empty_cache()
    reb = sum(t_reb[1:]) / (K - 1)              # le 1er paie d'éventuels alloc
    dec = sum(t_dec[1:]) / (K - 1)
    per = reb + dec
    print(f"\nchrono : runner frais (warmup 32 + captures + 64 tokens) "
          f"{t_fresh * 1e3:8.1f} ms")
    print(f"         rebind seul (moy {K - 1} appels)              "
          f"{reb * 1e3:8.1f} ms")
    print(f"         decode {N} tokens post-rebind                "
          f"{dec * 1e3:8.1f} ms  = {dec / N / a.batch * 1e3:.2f} ms/token-ligne")
    print(f"         appel complet (rebind + decode)              "
          f"{per * 1e3:8.1f} ms  ({a.batch * N / per:,.0f} tok/s agrégés)")


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
    p.add_argument("--cuda-graphs", dest="cuda_graphs", action="store_true",
                   help="bras GraphDecodeRunner dans --time (eager sur CPU)")
    p.add_argument("--amp", default="none", choices=("none", "bf16"),
                   help="autocast bf16 dans le bras --cuda-graphs (classe ULP)")
    p.add_argument("--force", action="store_true")
    p.add_argument("--verify", action="store_true",
                   help="A/B GPU des replays réels : graphs (chain+step+stop) "
                        "vs eager largeur pleine, torch.equal")
    p.add_argument("--rebind", action="store_true",
                   help="rebind GPU : decode→rebind→decode == eager largeur "
                        "pleine au bit (préfixes court+long, stop) + chrono "
                        "du coût par appel vs runner frais")
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
    if a.verify:
        run_verify(raw, a)
    if a.rebind:
        run_rebind(raw, a)
    if a.time:
        run_time(raw, a)
    if not (a.fingerprint or a.ab_check or a.verify or a.rebind or a.time):
        run_census(raw, a, md=(a.report == "md"))


if __name__ == "__main__":
    main()
