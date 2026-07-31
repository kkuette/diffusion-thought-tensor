"""dsv6 disaggregated GRPO — rollouts on the 3070Ti farm, updates on the 3090.

Why: GRPO's bottleneck is rollout generation, and the farm's 97M/135M VRAM
frontier is a TRAINING frontier (optimizer + activations). Inference of the
386M fits a 3070Ti with room to spare — so the 6 farm cards generate while
the 3090 (or a pod) grades and updates. Weights ride the NFS share; one
iteration of staleness is tolerated (the stored logp_old makes the update
properly off-policy through the existing clipped ratio — nothing new to add).

Split of rl_defer_grpo_lives across the share (root = rl.disagg.root):

  WORKER (one per farm GPU, `WORKER` env = its id)
    owns a PARTITION of the lives (seeds offset by worker id — lives never
    migrate, so the no-reset invariant holds per worker without locks);
    samples episodes from its own mixer, forks the carried bank G ways,
    rolls out the write policy (rl_defer_grpo_lives.rollout verbatim),
    computes REWARDS in place: dense -ce - lam*n_writes by default, or the
    verifiable rubric for tool envs (decode the call turn from the final
    bank, rl_rewards.grade_calls x think_economy, n_think = n_writes: think
    turns ARE bank writes); dynamic-resamples degenerate groups; commits one
    rollout uniformly (never argmax — covert retention pressure); ships the
    group (actions, logp_old, per-turn bank_in/lb_in) to rollouts/incoming.

  LEARNER (single consumer)
    takes groups whose weights_step >= current - max_lag (older = stale/),
    normalizes rewards into advantages, replays grpo_backward (ratio vs the
    SHIPPED logp_old = the off-policy correction), steps, publishes weights
    (atomic tmp+rename, LATEST pointer). ref for the KL = the init
    checkpoint, never the moving weights.

  PROBES (worker-side, log-only, never shipped for training)
    every xdom_every groups: same eval episode rolled from the life's OWN
    bank vs a FOREIGN life's bank (the banque-xdom adversity probe, memory
    dsv6-grpo-recence-feature) + always/never anchors on the own bank.

Files: weights/step_%06d.pt + LATEST | rollouts/{incoming,stale}/ |
w{N}_lives.pt (auto-resumed) | STOP kills everyone politely;
STOP_LEARNER / STOP_WORKERS n'arrêtent qu'un côté (redémarrages ciblés).

  learner:  python -m deepseek_v4_mini.rl_disagg learner <cfg.yaml>
  worker:   python -m deepseek_v4_mini.rl_disagg worker  <cfg.yaml> [--worker N]

CPU self-test (tiny model, stub envs, in-process worker+learner):
  python -m deepseek_v4_mini.rl_disagg
"""
from __future__ import annotations

import copy
import json
import math
import os
import random as _random
import statistics as st
import sys
import time
import uuid

import torch
import torch.nn.functional as F

from .cascade import CascadeMemory
from .decode import generate
from .rl_defer_grpo import pos_write_corr
from .rl_defer_grpo_lives import (_lb, boundary_step, defer_ce, forced_reward,
                                  grpo_backward, rollout)
from .rl_lives import EnvMixer, EnvSpec, Life, LivesState, mem_fork
from .rl_rewards import make_exec_reward, make_recall_reward, make_tool_reward
from .rti_learner import learn_from_raw, step_groups, telemetry
from .rti_policy import attach_rti_modules, rti_from_raw
from .cfg_schema import check as check_cfg
from .paths import load_yaml


# ── shared-FS primitives ─────────────────────────────────────────────────────

def _atomic_save(obj, path: str) -> None:
    tmp = f"{path}.tmp.{uuid.uuid4().hex[:8]}"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _atomic_write(text: str, path: str) -> None:
    tmp = f"{path}.tmp.{uuid.uuid4().hex[:8]}"
    with open(tmp, "w") as fh:
        fh.write(text)
    os.replace(tmp, path)


class WeightHub:
    """Publish/fetch model weights through the share. LATEST is a pointer
    file (atomic replace): readers never see a half-written checkpoint."""

    def __init__(self, root: str, keep: int = 3):
        self.dir = os.path.join(root, "weights")
        self.keep = int(keep)
        os.makedirs(self.dir, exist_ok=True)

    def publish(self, model_sd: dict, step: int) -> None:
        name = f"step_{step:06d}.pt"
        _atomic_save({"model": {k: v.cpu() for k, v in model_sd.items()},
                      "step": step}, os.path.join(self.dir, name))
        _atomic_write(name, os.path.join(self.dir, "LATEST"))
        pts = sorted(p for p in os.listdir(self.dir)
                     if p.startswith("step_") and p.endswith(".pt"))
        for p in pts[:-self.keep]:
            try:
                os.remove(os.path.join(self.dir, p))
            except OSError:
                pass

    def latest_step(self):
        try:
            name = open(os.path.join(self.dir, "LATEST")).read().strip()
            return int(name[len("step_"):-len(".pt")])
        except (OSError, ValueError):
            return None

    def fetch(self, known_step):
        """(state_dict, step) if newer than known_step, else None."""
        s = self.latest_step()
        if s is None or (known_step is not None and s <= known_step):
            return None
        path = os.path.join(self.dir, f"step_{s:06d}.pt")
        try:
            ck = torch.load(path, map_location="cpu", weights_only=False)
        except (OSError, RuntimeError, EOFError):
            return None                        # pruned or racing — next poll
        return ck["model"], ck["step"]


class RolloutStore:
    """Group files through the share. Workers write atomically to incoming/;
    the single learner claims by os.replace into claimed/ (rename is the
    lock), loads, unlinks. Groups older than min_step land in stale/."""

    def __init__(self, root: str):
        self.inc = os.path.join(root, "rollouts", "incoming")
        self.clm = os.path.join(root, "rollouts", "claimed")
        self.stl = os.path.join(root, "rollouts", "stale")
        for d in (self.inc, self.clm, self.stl):
            os.makedirs(d, exist_ok=True)

    def put(self, group: dict, weights_step: int, worker: int) -> None:
        name = f"w{worker:02d}_s{weights_step:06d}_{uuid.uuid4().hex[:8]}.pt"
        _atomic_save(group, os.path.join(self.inc, name))

    def pending(self) -> int:
        return len([p for p in os.listdir(self.inc) if p.endswith(".pt")])

    def take(self, n: int, min_step: int):
        """Up to n groups, oldest first; stale ones moved aside, counted."""
        got, n_stale = [], 0
        names = sorted((p for p in os.listdir(self.inc) if p.endswith(".pt")),
                       key=lambda p: os.path.getmtime(os.path.join(self.inc, p)))
        for name in names:
            if len(got) >= n:
                break
            ws = int(name.split("_s")[1].split("_")[0])
            src = os.path.join(self.inc, name)
            if ws < min_step:
                os.replace(src, os.path.join(self.stl, name))
                n_stale += 1
                continue
            dst = os.path.join(self.clm, name)
            try:
                os.replace(src, dst)
            except OSError:
                continue                       # raced (should not happen: 1 learner)
            try:
                got.append(torch.load(dst, map_location="cpu",
                                      weights_only=False))
            except Exception as e:
                # groupe illisible (typique : NAS coupé entre le rename et le
                # sync serveur — le fichier existe, les octets non). Il est
                # perdu de toute façon ; le jeter vaut mieux que tuer le
                # learner. Vécu 2026-07-28 : EOFError post-coupure.
                print(f"rollout corrompu jeté: {name} ({type(e).__name__}: "
                      f"{e})", flush=True)
            finally:
                try:
                    os.remove(dst)
                except OSError:
                    pass
        return got, n_stale


# ── group (de)hydration ──────────────────────────────────────────────────────

def group_to_cpu(chunks, rollouts, env_name, weights_step, worker) -> dict:
    """Ship exactly what grpo_backward replays. x is NOT stored per rec —
    rollout() emits one rec per chunk in order, so position rebuilds it."""
    cpu = lambda t: t.detach().cpu()
    return {
        "env": env_name, "weights_step": int(weights_step),
        "worker": int(worker),
        "chunks": [cpu(x) for x in chunks],
        "rollouts": [{
            "reward": float(ro["reward"]), "ce": float(ro["ce"]),
            "n_writes": int(ro["n_writes"]),
            "recs": [{"a": r["a"], "logp_old": r["logp_old"], "p": r["p"],
                      "bank_in": cpu(r["bank_in"]),
                      "lb_in": None if r["lb_in"] is None else
                      [None if t is None else cpu(t) for t in r["lb_in"]]}
                     for r in ro["recs"]],
        } for ro in rollouts],
    }


def group_to_device(g: dict, device, dtype):
    """Rollout dicts in grpo_backward's shape, tensors on the learner."""
    mv = lambda t: t.to(device=device, dtype=dtype)
    chunks = [c.to(device) for c in g["chunks"]]
    out = []
    for ro in g["rollouts"]:
        recs = []
        for i, r in enumerate(ro["recs"]):
            recs.append({"x": chunks[i], "a": r["a"],
                         "logp_old": r["logp_old"], "p": r["p"],
                         "bank_in": mv(r["bank_in"]),
                         "lb_in": None if r["lb_in"] is None else
                         [None if t is None else mv(t) for t in r["lb_in"]]})
        out.append({"recs": recs, "reward": ro["reward"], "ce": ro["ce"],
                    "n_writes": ro["n_writes"]})
    return out


# ── env construction (worker side) ───────────────────────────────────────────

def build_envs(d: dict, r: dict, tok, seed: int, raw: dict = None):
    """EnvSpecs from the config's data.envs. kind: code (CodeChunkStream,
    dense -ce), tool (ToolSessionStream + verifiable rubric), exec
    (CodeExecStream + sandboxed unit tests), sota (SotaSessionStream, dense),
    recall (PersonaChatStream + grade_recall sur la dernière vérité plantée),
    recall_env (RecallEnvStream : vies SCRIPTÉES et appariées + reward
    vérifiable par sonde — le chemin rti, cf. Worker.one_group_rti).
    Chat-kind streams return conv DICTS — sample_episode below normalizes
    both shapes.

    `raw` : la config ENTIÈRE, pour la section `recall_env:` (le curriculum ne
    vit pas dans `data.envs[].gen` — c'est un bloc de premier rang, partagé par
    tous les workers, et c'est lui qui porte le `layout` du préfixe)."""
    from .code_data import CodeChunkStream
    envs = []
    for i, e in enumerate(d["envs"]):
        kind = e.get("kind", "code")
        w = float(e.get("weight", 1.0))
        if kind == "recall_env":
            from .recall_env import RecallEnvConfig, make_recall_env_reward
            from .streams import chat_stream_class
            rc = RecallEnvConfig.from_raw((raw or {}).get("recall_env") or {})
            stream = chat_stream_class("recall_env")(
                tok, seed=seed + 31 * i, cfg=rc, **(e.get("gen") or {}))
            spec = EnvSpec(e["name"], stream, weight=w,
                           reward_fn=make_recall_env_reward(
                               int(r.get("think_nmax", 8)),
                               float(r.get("think_floor", 0.4)),
                               float(rc.exec_timeout)))
            spec.kind = kind
            envs.append(spec)
            continue
        if kind == "code":
            sd_e = dict(seq_len=int(d["seq_len"]),
                        chunks_per_conv=int(d["chunks_per_conv"]), batch=1,
                        cache_dir=d.get("cache_dir", "data_cache"),
                        var_chunk=d.get("var_chunk"),
                        n_files=int(e.get("n_files", 800)),
                        dataset=e["dataset"], data_dir=e.get("data_dir", ""),
                        stream_cap=int(e.get("stream_cap", 60000)),
                        content_key=e.get("content_key", "content"),
                        config_name=e.get("config_name", ""),
                        min_chunks=int(e.get("min_chunks", 2)),
                        seed=seed + 31 * i)
            stream = CodeChunkStream(tok, split="train", **sd_e)
            spec = EnvSpec(e["name"], stream, weight=w)
            spec.kind = kind
            envs.append(spec)
        else:
            # tool / exec / sota : même contrat de construction (registre
            # streams.py), seul le reward diffère.
            from .streams import rl_stream_class
            stream = rl_stream_class(kind)(tok, seed=seed + 31 * i,
                                           **(e.get("gen") or {}))
            if kind == "tool":
                fn = make_tool_reward(int(r.get("think_nmax", 8)),
                                      float(r.get("think_floor", 0.4)))
            elif kind == "exec":
                fn = make_exec_reward(int(r.get("think_nmax", 8)),
                                      float(r.get("think_floor", 0.4)),
                                      float(e.get("exec_timeout", 6.0)))
            elif kind == "recall":
                fn = make_recall_reward(int(r.get("think_nmax", 8)),
                                        float(r.get("think_floor", 0.4)))
            else:
                fn = None                      # sota : reward par défaut
            spec = EnvSpec(e["name"], stream, weight=w, reward_fn=fn)
            spec.kind = kind
            envs.append(spec)
    return envs


def pick_env(mixer: EnvMixer):
    """Le TIRAGE PONDÉRÉ d'env, isolé de l'extraction d'épisode : le chemin rti
    déroule une VIE entière (script apparié) et ne passe pas par
    `sample_episode`, mais les deux doivent consommer le MÊME rng — sinon le
    mix observé cesse de suivre les poids déclarés."""
    return mixer.envs[mixer.rng.choices(mixer._names, weights=mixer._weights,
                                        k=1)[0]]


def sample_episode(mixer: EnvMixer, defer_len: int, device, rng=None, env=None):
    """Weighted env choice + episode extraction for BOTH stream shapes.
    Returns (env, chunks, tgt, info). Uses mixer.rng so mixer.state_dict()
    keeps full sampling determinism."""
    env = env if env is not None else pick_env(mixer)
    while True:
        got = env.stream.next_conv()
        segs, info = (got["segs"], dict(got.get("info", {}))) \
            if isinstance(got, dict) else (got, {})
        if len(segs) >= 3 and segs[-1]["input_ids"].size(1) >= 1:
            tgt = segs[-1]["input_ids"][:, :defer_len]
            if tgt.size(1) < 1:
                continue
            chunks = [s["input_ids"].to(device) for s in segs[:-1]]
            return env, chunks, tgt.to(device), info


# ── rubric decode (tool envs) ────────────────────────────────────────────────

def decode_lb(model, prefix, bank, lb, max_new, stop_id, amp):
    """Le tour d'appel est décodé depuis l'état exact où le rollout s'est arrêté
    (lectures seules). Mono-ligne ; pour les G rollouts d'un groupe, préférer
    `decode.generate` batché — c'est le goulot de ce worker."""
    gen, lens = generate(model, prefix, bank=bank, layer_banks=lb,
                         max_new=max_new, stop_id=stop_id, amp=amp)
    return gen[:, :int(lens[0])]


# ── worker ───────────────────────────────────────────────────────────────────

class Worker:
    def __init__(self, raw: dict, worker_id: int, *, tok=None, model=None,
                 envs=None, device=None):
        r, d = raw["rl"], raw["data"]
        self.r, self.d = r, d
        self.wid = int(worker_id)
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        seed = int(r.get("seed", 0)) + 1000 * self.wid
        torch.manual_seed(seed)
        self.rng = _random.Random(seed + 17)
        dg = r["disagg"]
        self.root = dg["root"]
        # TB_WEIGHTS_MIRROR (rig) : lire les poids depuis une copie locale
        # entretenue par scripts/rl_weights_mirror.sh — un pull NAS par
        # publication au lieu d'un par worker. Ne concerne que le fetch des
        # poids : rollouts, lives et STOP restent sur root (NFS).
        mroot = os.environ.get("TB_WEIGHTS_MIRROR")
        self.hub = WeightHub(mroot or self.root,
                             keep=int(dg.get("keep_weights", 3)))
        self.store = RolloutStore(self.root)
        self.max_pending = int(dg.get("max_pending", 24))
        self.poll_s = float(dg.get("poll_s", 2.0))

        self.tok = tok
        if self.tok is None:
            from transformers import AutoTokenizer
            self.tok = AutoTokenizer.from_pretrained(raw["tokenizer"])
            add = [x for x in ("<think>", "<blank>")
                   if x not in self.tok.get_vocab()]
            if add:
                self.tok.add_special_tokens({"additional_special_tokens": add})
        self.ids = (self.tok.convert_tokens_to_ids("<think>"),
                    self.tok.convert_tokens_to_ids("<blank>"))

        if model is None:
            from .config import ThoughtBankConfig
            from .model import ThoughtBankLM
            mcfg = dict(raw["model"])
            mcfg["vocab_size"] = len(self.tok)
            model = ThoughtBankLM(ThoughtBankConfig(**mcfg)).to(self.device)
        self.model = model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.mcfg = raw["model"]

        self.envs = envs if envs is not None else build_envs(d, r, self.tok,
                                                             seed, raw)
        self.mixer = EnvMixer(self.envs, seed=seed + 977)
        self.defer_len = int(d.get("defer_len", 16))

        self.G = int(r.get("group_size", 8))
        self.temp = float(r.get("temp", 1.0))
        lam_default = float(r.get("lambda_write", 0.03))
        self.lam = {e["name"]: float(e.get("lambda_write", lam_default))
                    for e in d["envs"]}
        self.min_std = float(r.get("min_reward_std", 1.0e-4))
        self.max_rs = int(r.get("max_resample", 4))
        # plancher d'exploration du write (amorçage à froid, voir rollout()) :
        # 0.0 = comportement historique
        self.explore_floor = float(r.get("explore_floor", 0.0))
        self.casc_depth = int(r.get("cascade_depth", 0))
        self.cmap = r.get("cascade_map") or [0] * int(self.mcfg["n_layers"])
        self.max_mem = int(self.mcfg["max_mem"])
        self.seed_slots = int(self.mcfg.get("mem_seed_slots", 0))
        self.max_new = int(r.get("max_new", 64))
        # per-env decode budget (code turns need more than call turns)
        self.max_new_env = {e["name"]: int(e.get("max_new", self.max_new))
                            for e in d["envs"]}
        self.amp = bool(r.get("amp", True))
        # decode_cache : cache KV incrémental pour le décodage des rollouts.
        # Un rollout est un ÉCHANTILLON — une bascule de routage MoE due aux
        # ULP y est aussi légitime qu'un autre tirage, alors qu'en éval elle
        # rendrait les chiffres incomparables. Voir decode.generate.
        self.decode_cache = bool(r.get("decode_cache", False))
        # decode_graphs : le tour de réponse par GraphDecodeRunner (CUDA
        # graphs + rebind — même classe ULP que decode_cache, opt-in). Exige
        # les 3 flags decode_* dans la config modèle ; s'arme au premier
        # bucket banque-pleine et se DÉSARME bruyamment au premier pépin,
        # generate reprend — jamais un rollout perdu. fp32 dans les graphs
        # (bf16-autocast = perte mesurée, FINDINGS 2026-07-27).
        self.decode_graphs = bool(r.get("decode_graphs", False))
        self._graph_runner = None
        self._graph_dead = False
        self._graph_p0 = None       # data_ptr des poids à l'armement (belt :
        #                             un load_state_dict assign les changerait)
        stop = "<|im_end|>"
        self.stop_id = (self.tok.convert_tokens_to_ids(stop)
                        if stop in self.tok.get_vocab() else -1)
        from .math_school_data import A_OPEN
        a_ids = self.tok(A_OPEN, add_special_tokens=False)["input_ids"]
        self.a_open = torch.tensor(a_ids, dtype=torch.long,
                                   device=self.device).unsqueeze(0)

        # ── bras rti : les trois actions de rti_policy ───────────────────────
        self._rti_setup(raw)

        n_lives = int(r.get("n_lives_per_worker", 2))
        p0 = next(self.model.parameters())
        self._p0 = p0
        self.lives = LivesState([self._fresh_life(i) for i in range(n_lives)],
                                self.mixer)
        self.lives_path = os.path.join(self.root, f"w{self.wid:02d}_lives.pt")
        if os.path.exists(self.lives_path):
            lk = torch.load(self.lives_path, map_location="cpu",
                            weights_only=False)
            self.lives.load_state_dict(lk, device=p0.device, dtype=p0.dtype)
            for lf, s in zip(self.lives.lives, lk["lives"]):
                lf.n_evict = s.get("n_evict", 0)
            # compteur de vies rti : sans lui, un worker redémarré rejouerait
            # exactement les mêmes scripts (les vies sont fonction de l'index).
            self._rti_n = int(lk.get("rti_n", 0))
            print(f"worker {self.wid}: lives resumed "
                  f"({[lf.n_episodes for lf in self.lives.lives]} episodes)",
                  flush=True)
        self.wstep = None
        self.li = 0
        self.n_groups = 0
        self.mfile = os.path.join(self.root, f"worker{self.wid:02d}_metrics.jsonl")

    # ── bras rti ─────────────────────────────────────────────────────────────
    def _rti_setup(self, raw: dict) -> None:
        """Arme le bras retrieve-then-inject, ou ne fait rien (chemin
        historique BIT À BIT inchangé sans la section `rti:`)."""
        from .rti_policy import attach_rti_modules, rti_from_raw
        self.rti_cfg, self.rti_pol, self.rti_on = rti_from_raw(raw or {})
        self._sif: dict = {}
        self._rti_n = 0
        self._gen = torch.Generator(device="cpu").manual_seed(
            int(self.r.get("seed", 0)) + 7919 * (self.wid + 1))
        if not self.rti_on:
            return
        # decode_graphs est capturé à forme FIXE ; le préfixe injecté change la
        # longueur du premier forward, et une banque en croissance change la
        # signature. Même refus que côté trainer, et à l'assert : un désarmement
        # silencieux ferait tourner un run entier sur un chemin non voulu.
        assert not self.decode_graphs, \
            "rti + rl.decode_graphs : INCOMPATIBLE (préfixe de longueur " \
            "variable vs runners à forme fixe) — mettre decode_graphs: false"
        kinds = {getattr(e, "kind", "code") for e in self.envs}
        assert kinds == {"recall_env"}, (
            f"sous rti, tout env doit être de kind recall_env (vu {sorted(kinds)}). "
            "La banque fast-weight est CONTOURNÉE par ce bras (layer_banks tout "
            "à None, write=False) : un env dense qui la lit optimiserait un "
            "canal inerte. La pression 'ne pas éroder le chat général' passe "
            "par la KL à la ref, pas par un env mélangé.")
        d_model = int(self.model.embed.weight.size(1))
        attach_rti_modules(self.model, d_model)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.rti_sep = int(self.tok.convert_tokens_to_ids(
            self.rti_cfg.sep_token))
        assert self.rti_sep >= 0, \
            f"rti.sep_token {self.rti_cfg.sep_token!r} absent du vocabulaire"
        print(f"worker {self.wid}: rti ON — top_k {self.rti_cfg.top_k}, "
              f"{self.rti_cfg.eval_groups} groupes injectés "
              f"(P={self.rti_cfg.eval_groups * self.rti_cfg.group_prefix} "
              f"pseudo-tokens), FIFO {self.rti_cfg.max_groups}, write "
              f"{self.rti_pol.write_mode} (temp {self.rti_pol.write_temp}, "
              f"floor {self.rti_pol.write_floor}), retrieve PL temp "
              f"{self.rti_pol.retr_temp}, decode temp "
              f"{self.rti_pol.decode_temp}", flush=True)

    def _sif_w(self, env):
        """Table SIF du stream (une par env, calculée une fois)."""
        from .rti import sif_table
        w = self._sif.get(env.name)
        if w is None:
            w = self._sif[env.name] = sif_table(
                env.stream, int(self.model.embed.weight.size(0)),
                self.rti_cfg.sif_a).to(self.device)
        return w

    def _rti_payload(self, conv, traces, env) -> dict:
        """De quoi REJOUER le groupe sans le stream, le tokenizer ni le corpus.

        La passe 2 refait des forwards : sans les ids des segs elle n'a rien à
        forwarder. Reconstruire le script côté learner (`conv_for_life`) le
        rendrait dépendant du corpus de filler réel et du tokenizer, et un
        écart d'un token y serait INVISIBLE (le digest porte sur le script, pas
        sur la tokenisation). On expédie donc les segs — ~1 k entiers, partagés
        par les G rollouts, négligeable devant les traces.

        `sif` est la table SIF RESTREINTE aux tokens que `build_group` indexe
        réellement (ids des segs + tokens décodés, seules sources d'un groupe
        écrit) : quelques centaines d'entrées au lieu du vocabulaire entier.
        """
        ids = set()
        for s in conv["segs"]:
            ids |= set(s["input_ids"].reshape(-1).tolist())
        for tr in traces:
            for t in tr["turns"]:
                if t["decode"] is not None:
                    ids |= set(t["decode"]["tokens"])
        ids.add(self.rti_sep)
        sid = torch.tensor(sorted(ids), dtype=torch.long)
        return {"segs": [s["input_ids"].detach().cpu() for s in conv["segs"]],
                "a_open": self.a_open.detach().cpu(),
                "sep_id": int(self.rti_sep),
                "sif": {"ids": sid,
                        "w": self._sif_w(env).detach().cpu()[sid].float()}}

    def _next_life(self) -> int:
        """Vie SUIVANTE de la partition de ce worker. Les vies ne migrent
        jamais d'un worker à l'autre (même invariant que les lives du chemin
        historique), et l'index est stable au resume."""
        self._rti_n += 1
        return (self.wid + 1) * 1_000_003 + self._rti_n

    def one_group_rti(self, env):
        """UN groupe GRPO sur une vie rti : G rollouts APPARIÉS sur le MÊME
        script (`conv_for_life`), donc l'avantage intra-groupe est
        contrefactuel et l'ancrage par index de tour est exact (GiGPO).

        On ne rappelle JAMAIS `next_conv()` par rollout : ce serait G scripts
        différents, et le baseline de groupe mesurerait la difficulté du script
        au lieu de la qualité de la politique.
        """
        from .rti_policy import RtiRollout
        roll = RtiRollout(self.model, self.tok, self.rti_cfg, self.rti_pol,
                          self._sif_w(env), self.rti_sep, a_open=self.a_open,
                          stop_id=self.stop_id,
                          max_new=self.max_new_env.get(env.name, self.max_new),
                          amp=self.amp, decode_cache=self.decode_cache)
        for _try in range(self.max_rs + 1):
            life = self._next_life()
            conv = env.stream.conv_for_life(life)
            traces = roll.run(conv, self.G, self.rng, reward_fn=env.reward_fn,
                              device=self.device, generator=self._gen)
            assert len({t["digest"] for t in traces}) == 1, \
                "les G rollouts d'un groupe ne partagent pas le même script"
            rs = [t["reward"] for t in traces]
            live = [x for x in rs if x is not None]
            if not live or not all(math.isfinite(x) for x in live):
                continue          # vie sans AUCUNE sonde possible : rien à noter
            if st.pstdev(live) < self.min_std:
                continue          # groupe plat (dynamic sampling, comme v1)
            self.store.put({"format": "rti", "env": env.name,
                            "weights_step": int(self.wstep),
                            "worker": self.wid, "life": int(life),
                            "digest": traces[0]["digest"],
                            "layout": traces[0]["layout"],
                            "rollouts": traces,
                            **self._rti_payload(conv, traces, env)},
                           self.wstep, self.wid)
            self.n_groups += 1
            dec = [t for tr in traces for t in tr["turns"] if t["decode"]]
            raw = [t["raw"] for t in dec if t.get("raw") is not None]
            ret = [t["retrieve"] for tr in traces for t in tr["turns"]
                   if t["retrieve"]]
            ps = [t["write"]["p"] for tr in traces for t in tr["turns"]
                  if t["write"]]
            return {"env": env.name, "reward": st.mean(live),
                    "ce": None,             # pas de CE dense sur ce bras
                    "writes": sum(t["n_writes"] for t in traces),
                    "turns": sum(len(t["turns"]) for t in traces),
                    "tries": _try, "life": life,
                    "grade": st.mean(raw) if raw else None,
                    "p_write": st.mean(ps) if ps else None,
                    # le retriever a-t-il ramené un positif ? Sans ça, un reward
                    # bas ne dit pas si la SÉLECTION ou la COPIE a échoué —
                    # les deux rendent 0.00 (leçon de l'éval rti).
                    "hit": st.mean([float(x["hit"]) for x in ret]) if ret else None,
                    "top1": st.mean([float(x["top1"]) for x in ret]) if ret else None,
                    "n_dec": len(dec), "pending": self.store.pending()}
        return None

    def _fresh_life(self, i):
        with torch.no_grad():
            b = self.model.thought_stream.seed_bank(1, self._p0.device,
                                                    self._p0.dtype)
        lf = Life(i, b, CascadeMemory(self.casc_depth, self.max_mem)
                  if self.casc_depth else None)
        lf.n_evict = 0
        return lf

    # ── weights ──────────────────────────────────────────────────────────────
    def refresh(self) -> bool:
        got = self.hub.fetch(self.wstep)
        if got is None:
            return False
        sd, s = got
        self.model.load_state_dict(sd)
        self.model.eval()
        self.wstep = s
        return True

    def wait_weights(self):
        while self.wstep is None:
            if self.refresh():
                print(f"worker {self.wid}: weights step {self.wstep}", flush=True)
                return
            time.sleep(self.poll_s)

    # ── reward ───────────────────────────────────────────────────────────────
    def _graphs_decode(self, bank, lb, max_new):
        """Le tour de réponse d'un bucket par GraphDecodeRunner, ou None si ce
        bucket doit passer par `generate` (banque en croissance, signature
        atypique, runner désarmé). UN runner par worker, baké B=G : les
        buckets partiels sont PADÉS à G (le batch est quasi gratuit — 9,0 vs
        6,5 ms/pas, FINDINGS 2026-07-28) et les lignes de bourrage jetées.
        Warmup + 16 captures payés au premier bucket ; ensuite chaque appel =
        rebind (copies en place) + replays."""
        if self._graph_dead or self.device.type != "cuda":
            return None
        if bank.size(1) != self.max_mem or bank.size(0) > self.G:
            return None                     # vie pas encore établie : eager
        from .decode_graphs import GraphDecodeRunner
        B = bank.size(0)
        if B < self.G:
            pad = self.G - B
            bank = torch.cat([bank, bank[:1].expand(pad, -1, -1)], dim=0)
            lb = None if lb is None else [
                None if x is None
                else torch.cat([x, x[:1].expand(pad, *x.shape[1:])], dim=0)
                for x in lb]
        if self._graph_runner is None:
            try:
                self._graph_runner = GraphDecodeRunner(
                    self.model, bank, layer_banks=lb)
            except ValueError as e:     # flags decode_* absents de la config
                print(f"worker {self.wid}: decode_graphs DÉSARMÉ ({e})",
                      flush=True)
                self._graph_dead = True
                return None
            self._graph_p0 = next(self.model.parameters()).data_ptr()
            print(f"worker {self.wid}: decode_graphs ARMÉ (B={self.G})",
                  flush=True)
        try:
            if next(self.model.parameters()).data_ptr() != self._graph_p0:
                raise RuntimeError(
                    "les poids ont changé d'adresse (load_state_dict "
                    "assign ?) — les graphs pointent sur l'ancien storage")
            self._graph_runner.rebind(bank, layer_banks=lb)
            gen, lens = self._graph_runner.decode(
                self.a_open.expand(self.G, -1), max_new=max_new,
                stop_id=self.stop_id)
        except ValueError:
            # signature inattendue (structure layer_banks d'un autre bucket) :
            # CE bucket repasse par generate, le runner reste armé
            return None
        except Exception as e:                          # noqa: BLE001
            print(f"worker {self.wid}: decode_graphs DÉSARMÉ ({e}) — "
                  f"retour à generate", flush=True)
            self._graph_dead = True
            if self._graph_runner is not None:
                self._graph_runner.close()
                self._graph_runner = None
            return None
        # le gate MoE dense revient à 1 entre deux appels : les autres
        # forwards du worker (rollout, decode_lb) gardent leur chemin
        for mod in self.model.modules():
            if hasattr(mod, "dense_max_bt"):
                mod.dense_max_bt = 1
        if self._graph_runner.eager_only:
            # capture ratée : les tokens rendus sont valides (eager largeur
            # pleine) mais le gain est perdu — désarmer pour ne pas payer
            # l'eager élargi à chaque bucket
            print(f"worker {self.wid}: decode_graphs DÉSARMÉ (fallback eager "
                  f"du runner, voir warning) — retour à generate", flush=True)
            self._graph_dead = True
            self._graph_runner.close()
            self._graph_runner = None
        return gen[:B], lens[:B]

    def _decode_calls(self, env, cands) -> list[str]:
        """Le tour d'appel de plusieurs rollouts, décodé EN UN SEUL batch.

        C'est le goulot du worker : chaque token coûtait un forward complet sur
        tout le préfixe, une ligne à la fois. Les G rollouts d'un groupe partent
        du même état et décodent le même préfixe — il n'y avait aucune raison de
        les faire un par un, sinon que la boucle de décodage câblait B=1.

        Les banques ne sont empilables que si elles ont la même forme, et ce
        n'est pas garanti : une banque grandit d'un slot par write, donc deux
        rollouts qui n'ont pas écrit autant diffèrent — jusqu'à saturation de
        `max_mem`, après quoi tous se valent (le cas courant d'une vie établie).
        On groupe donc par forme : batch quand c'est possible, ligne à ligne
        quand ça ne l'est pas, jamais faux dans un cas comme dans l'autre.
        """
        max_new = self.max_new_env.get(env.name, self.max_new)
        lbs = [_lb(c["casc"], c["bank"], self.cmap) for c in cands]

        def _sig(i):
            lb = lbs[i]
            return (tuple(cands[i]["bank"].shape),
                    None if lb is None else
                    tuple(None if x is None else tuple(x.shape) for x in lb))

        buckets = {}
        for i in range(len(cands)):
            buckets.setdefault(_sig(i), []).append(i)

        texts: list[str | None] = [None] * len(cands)
        for idx in buckets.values():
            bank = torch.cat([cands[i]["bank"] for i in idx], dim=0)
            ref = lbs[idx[0]]
            lb = None if ref is None else [
                None if ref[l] is None
                else torch.cat([lbs[i][l] for i in idx], dim=0)
                for l in range(len(ref))]
            got = (self._graphs_decode(bank, lb, max_new)
                   if self.decode_graphs else None)
            if got is not None:
                gen, lens = got
            else:
                gen, lens = generate(self.model,
                                     self.a_open.expand(len(idx), -1),
                                     bank=bank, layer_banks=lb,
                                     max_new=max_new, stop_id=self.stop_id,
                                     amp=self.amp, use_cache=self.decode_cache)
            for j, i in enumerate(idx):
                texts[i] = self.tok.decode(gen[j, :int(lens[j])].tolist())
        return texts

    def _rewards(self, env, cands, lam, info) -> list[float]:
        self.last_raw = []
        if env.reward_fn is None:
            return [-c["ce"] - lam * c["n_writes"] for c in cands]
        # rubric payload: the LAST episode's gold, whichever family (tool
        # envs read gold_calls, exec envs read tests)
        gold = (info.get("gold_calls") or [[]])[-1]
        tests = (info.get("tests") or [[]])[-1]
        truth = (info.get("truths") or [""])[-1]
        # les payloads sont matérialisés (et pas construits dans la
        # comprehension) pour qu'on puisse relire info["raw"] : le succès brut
        # que la fonction de reward calcule puis dilue dans l'économie de think.
        pl = [{"text": txt, "n_think": c["n_writes"],
               "gold_calls": gold, "tests": tests, "truth": truth}
              for c, txt in zip(cands, self._decode_calls(env, cands))]
        out = [env.reward(c["ce"], d) for c, d in zip(cands, pl)]
        self.last_raw = [d.get("raw") for d in pl]
        return out

    def _reward(self, env, ro, lam, info) -> float:
        return self._rewards(env, [ro], lam, info)[0]

    # ── one group ────────────────────────────────────────────────────────────
    def one_group(self):
        env0 = pick_env(self.mixer)
        if getattr(env0, "kind", None) == "recall_env":
            # chemin rti : une VIE scriptée, pas d'épisode ni de banque portée
            # (cf. rti_policy — la FIFO naît et meurt avec le script).
            return self.one_group_rti(env0)
        life = self.lives.lives[self.li % len(self.lives.lives)]
        self.li += 1
        max_epi = int(self.r.get("max_episodes_per_life", 0))
        if max_epi and life.n_episodes >= max_epi:
            self.lives.lives[life.id] = life = self._fresh_life(life.id)
        for _try in range(self.max_rs + 1):
            env, chunks, tgt, info = sample_episode(self.mixer, self.defer_len,
                                                    self.device, env=env0)
            env0 = None
            lam = self.lam[env.name]
            forks = mem_fork(life.bank, life.casc, self.G)
            cand = [rollout(self.model, chunks, tgt, self.temp, lam, self.ids,
                            self.rng, fb, fc, life.n_evict, self.seed_slots,
                            self.max_mem, self.cmap,
                            explore_floor=self.explore_floor)
                    for fb, fc in forks]
            for c, rw in zip(cand, self._rewards(env, cand, lam, info)):
                c["reward"] = rw
            rs = [c["reward"] for c in cand]
            if not all(math.isfinite(x) for x in rs):
                # ce non-fini sur un épisode pathologique : pstdev (py3.13)
                # crashe dessus ('float' object has no attribute 'numerator',
                # worker 3 le 07-27) et le learner recevrait des avantages
                # NaN — on jette le tirage comme dégénéré.
                print(f"worker {self.wid}: reward non-fini ({env.name}) — "
                      f"resample", flush=True)
                continue
            if st.pstdev(rs) >= self.min_std:
                keep = cand[self.rng.randrange(self.G)]
                life.bank, life.casc = keep["bank"], keep["casc"]
                life.n_evict = keep["n_evict"]
                life.advance(keep["bank"], env.name)
                self.store.put(group_to_cpu(chunks, cand, env.name,
                                            self.wstep, self.wid),
                               self.wstep, self.wid)
                self.n_groups += 1
                raw = [x for x in self.last_raw if x is not None]
                ps = [r["p"] for c in cand for r in c["recs"]]
                return {"env": env.name, "reward": st.mean(rs),
                        "ce": st.mean([c["ce"] for c in cand]),
                        "writes": sum(c["n_writes"] for c in cand),
                        "turns": sum(len(c["recs"]) for c in cand),
                        "tries": _try,
                        # grade : taux d'appels justes / de tests passés, avant
                        # économie de think. None pour les envs denses.
                        "grade": st.mean(raw) if raw else None,
                        "p_write": st.mean(ps) if ps else None,
                        "pending": self.store.pending()}
        return None                            # degenerate after resamples

    # ── probes (log-only) ────────────────────────────────────────────────────
    @torch.no_grad()
    def xdom_probe(self):
        """Same episode, own vs foreign bank + always/never anchors."""
        env, chunks, tgt, info = sample_episode(self.mixer, self.defer_len,
                                                self.device)
        lam = self.lam[env.name]
        own = self.lives.lives[0]
        other = self.lives.lives[1 % len(self.lives.lives)]
        out = {}
        for tag, src in (("own", own), ("xdom", other)):
            (fb, fc), = mem_fork(src.bank, src.casc, 1)
            ro = rollout(self.model, chunks, tgt, self.temp, lam, self.ids,
                         _random.Random(0), fb, fc, src.n_evict,
                         self.seed_slots, self.max_mem, self.cmap,
                         explore_floor=self.explore_floor)
            ro["reward"] = self._reward(env, ro, lam, info)
            out[f"r_{tag}"] = ro["reward"]
        args = (own.n_evict, self.seed_slots, self.max_mem, self.cmap)
        forks = mem_fork(own.bank, own.casc, 2)
        out["r_always"], _ = forced_reward(self.model, chunks, tgt, True, lam,
                                           self.ids, *forks[0], *args)
        out["r_never"], _ = forced_reward(self.model, chunks, tgt, False, lam,
                                          self.ids, *forks[1], *args)
        out["env"] = env.name
        return out

    def save_lives(self):
        sd = self.lives.state_dict()
        for ls_, lf in zip(sd["lives"], self.lives.lives):
            ls_["n_evict"] = getattr(lf, "n_evict", 0)
        sd["rti_n"] = self._rti_n
        _atomic_save(sd, self.lives_path)

    # ── loop ─────────────────────────────────────────────────────────────────
    def run(self):
        self.wait_weights()
        dg = self.r["disagg"]
        max_groups = int(dg.get("max_groups", 0))
        lives_every = int(dg.get("lives_save_every", 20))
        xdom_every = int(dg.get("xdom_every", 50))
        t0 = time.time()
        n_degen = 0
        # STOP = arrêt global ; STOP_WORKERS = seulement la ferme. Morsure du
        # 07-27 : STOP posé pour redémarrer le learner a aussi éteint les
        # workers (jobs "done" propres, production stoppée en silence).
        while not (os.path.exists(os.path.join(self.root, "STOP"))
                   or os.path.exists(os.path.join(self.root, "STOP_WORKERS"))):
            if max_groups and self.n_groups >= max_groups:
                break
            if self.store.pending() >= self.max_pending:
                time.sleep(self.poll_s)        # learner is behind — don't flood
                self.refresh()
                continue
            self.refresh()
            line = self.one_group()
            if line is None:
                # un groupe dégénéré est du calcul jeté : le compter, sinon la
                # perte de débit ne se voit que dans le stdout, que personne ne lit
                n_degen += 1
                print(f"worker {self.wid}: degenerate group (all resamples)",
                      flush=True)
                continue
            line.update(n=self.n_groups, wstep=self.wstep, degen=n_degen,
                        t=time.time(),
                        s_per_group=(time.time() - t0) / max(self.n_groups, 1))
            with open(self.mfile, "a") as fh:
                fh.write(json.dumps(line) + "\n")
            if self.n_groups % lives_every == 0:
                self.save_lives()
            # sonde xdom : elle compare deux BANQUES fast-weight portées, qui
            # n'existent pas sur le bras rti (la FIFO meurt avec le script).
            if xdom_every and not self.rti_on \
                    and self.n_groups % xdom_every == 0:
                probe = self.xdom_probe()
                probe["n"] = self.n_groups
                probe["probe"] = "xdom"
                with open(self.mfile, "a") as fh:
                    fh.write(json.dumps(probe) + "\n")
        self.save_lives()
        print(f"worker {self.wid}: done ({self.n_groups} groups)", flush=True)


# ── learner ──────────────────────────────────────────────────────────────────

class Learner:
    def __init__(self, raw: dict, *, tok_len=None, model=None, device=None):
        r = raw["rl"]
        self.r = r
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(int(r.get("seed", 0)))
        dg = r["disagg"]
        self.root = dg["root"]
        os.makedirs(self.root, exist_ok=True)
        self.hub = WeightHub(self.root, keep=int(dg.get("keep_weights", 3)))
        self.store = RolloutStore(self.root)
        self.publish_every = int(dg.get("publish_every", 1))
        self.max_lag = int(dg.get("max_lag", 2))
        self.poll_s = float(dg.get("poll_s", 2.0))

        if model is None:
            from transformers import AutoTokenizer
            from .config import ThoughtBankConfig
            from .model import ThoughtBankLM
            if tok_len is None:
                tok = AutoTokenizer.from_pretrained(raw["tokenizer"])
                add = [x for x in ("<think>", "<blank>")
                       if x not in tok.get_vocab()]
                if add:
                    tok.add_special_tokens(
                        {"additional_special_tokens": add})
                tok_len = len(tok)
                self._think_id = tok.convert_tokens_to_ids("<think>")
            mcfg = dict(raw["model"])
            mcfg["vocab_size"] = tok_len
            model = ThoughtBankLM(ThoughtBankConfig(**mcfg)).to(self.device)
            ck = torch.load(r["init_from"], map_location="cpu")
            if self._rti_attach(raw, model):
                # le ckpt SFT rti porte retriever + vecteur de type, jamais la
                # TÊTE DE WRITE (elle n'existe qu'au RL) : strict=False, mais
                # BRUYANT — un `missing` inattendu voudrait dire qu'on part sur
                # des poids partiellement aléatoires sans le dire.
                miss, unex = model.load_state_dict(ck["model"], strict=False)
                print(f"learner: rti ON | neuf (zéro-init) {sorted(miss)}"
                      + (f" | ckpt ignoré {sorted(unex)}" if unex else ""),
                      flush=True)
            else:
                model.load_state_dict(ck["model"])
        self.model = model
        self._rti_attach(raw, self.model)      # idempotent (modèle injecté)
        if not hasattr(self, "_think_id"):
            self._think_id = int(r.get("think_id", 0)) or 0
        self.ids = (self._think_id, -1)        # blank unused in the update
        # bras rti : les knobs des actions (partagés avec le worker) + l'algo
        self.rti_cfg, self.rti_pol, self.rti_on = rti_from_raw(raw or {})
        self.lcfg = learn_from_raw(raw)
        # PAS de référence sous rti : la recette n'a PAS de terme KL (le cliquet
        # SFT joue ce rôle entre phases). Un deepcopy du 350M coûterait 1,4 Go
        # de VRAM pour un tenseur jamais lu.
        self.ref = None
        if not self.rti_on:
            self.ref = copy.deepcopy(self.model).eval()
            for p in self.ref.parameters():
                p.requires_grad_(False)

        scope = r.get("train_scope", "think_row")
        if self.rti_on:
            assert scope in ("rti", "all"), (
                f"train_scope {scope!r} n'a pas de sens sous rti : la ligne "
                "`<think>` du lm_head n'est l'action de personne sur ce bras "
                "(le write est une tête dédiée). Mettre `rti` (retriever + "
                "write + type seuls) ou `all`.")
            names = attach_rti_modules(self.model,
                                       int(self.model.embed.weight.size(1)))
            if scope == "rti":
                for p in self.model.parameters():
                    p.requires_grad_(False)
                byname = dict(self.model.named_parameters())
                opt_params = [byname[k] for k in names]
                for p in opt_params:
                    p.requires_grad_(True)
            else:
                opt_params = list(self.model.parameters())
            print(f"learner: rti ON — scope {scope} "
                  f"({sum(p.numel() for p in opt_params)} params entraînés), "
                  f"CISPO [{self.lcfg.cispo_low}, {self.lcfg.cispo_high}], "
                  f"ω {self.lcfg.omega}, γ {self.lcfg.gamma}, contrefactuel "
                  f"{self.lcfg.cf_coef}, poids w/r/d "
                  f"{self.lcfg.w_write}/{self.lcfg.w_retr}/{self.lcfg.w_dec}",
                  flush=True)
        elif scope == "think_row":
            for p in self.model.parameters():
                p.requires_grad_(False)
            W = self.model.lm_head.weight
            W.requires_grad_(True)
            mask = torch.zeros_like(W)
            mask[self.ids[0]] = 1.0
            W.register_hook(lambda g: g * mask)
            opt_params = [W]
        else:
            opt_params = list(self.model.parameters())
        self.opt = torch.optim.AdamW(opt_params, lr=float(r.get("lr", 5.0e-6)),
                                     weight_decay=0.0)
        self.temp = float(r.get("temp", 1.0))
        self.clip_lo = float(r.get("clip_low", 0.2))
        self.clip_hi = float(r.get("clip_high", 0.28))
        self.kl_coef = float(r.get("kl_coef", 1.0e-3))
        self.gps = int(r.get("groups_per_step", 4))
        self.G = int(r.get("group_size", 8))
        self.step = 0
        self.mfile = os.path.join(self.root, "learner_metrics.jsonl")
        self.trace_path = os.path.join(self.root, "traces.jsonl")
        self.ck_path = os.path.join(self.root, "learner_last.pt")
        if os.path.exists(self.ck_path):
            ck = torch.load(self.ck_path, map_location="cpu",
                            weights_only=False)
            self.model.load_state_dict(ck["model"])
            self.opt.load_state_dict(ck["opt"])
            self.step = ck["step"]
            print(f"learner: resumed step {self.step}", flush=True)
        self.hub.publish(self.model.state_dict(), self.step)
        # Carte d'identité du run pour qui le regarde de l'extérieur : sans
        # elle, un lecteur des JSONL voit un step sans savoir vers quoi il va,
        # ni à partir de quel lag un groupe part à la poubelle.
        _atomic_write(json.dumps({
            "steps": int(r["steps"]), "group_size": self.G,
            "groups_per_step": self.gps, "max_lag": self.max_lag,
            "publish_every": self.publish_every, "lr": float(r.get("lr", 5.0e-6)),
            "kl_coef": self.kl_coef, "started": time.time(),
            # poids VISÉS du mix : de quoi juger si le mix observé (env_mix)
            # dérive parce qu'un env est plus lent que les autres
            "envs": {e["name"]: e.get("weight")
                     for e in raw.get("data", {}).get("envs", [])},
        }, indent=1), os.path.join(self.root, "meta.json"))

    @staticmethod
    def _rti_attach(raw: dict, model) -> bool:
        """Attache les modules rti au modèle du learner. IMPÉRATIF : le worker
        recharge le `state_dict` PUBLIÉ tel quel — deux modèles de formes
        différentes et le run s'arrête au premier fetch."""
        from .rti_policy import attach_rti_modules, rti_from_raw
        _, _, on = rti_from_raw(raw or {})
        if on:
            attach_rti_modules(model, int(model.embed.weight.size(1)))
        return on

    def archive(self, groups) -> None:
        """Trajectoires → traces.jsonl, AVANT consommation. Matière première du
        cliquet (SFT sur trajectoires positives) : tokens + actions + rewards
        suffisent — les banques (le gros du .pt) se recomputent par forward,
        on ne les stocke pas. Tout est gardé, y compris les groupes plats :
        contraste négatif + stats de touchabilité."""
        with open(self.trace_path, "a") as fh:
            for g in groups:
                if g.get("format") == "rti":
                    # la trace rti EST déjà du JSON pur (actions, log-probs,
                    # FIFO, rewards par sonde) : on l'archive telle quelle.
                    fh.write(json.dumps({
                        "learner_step": self.step, "env": g["env"],
                        "worker": g["worker"], "format": "rti",
                        "weights_step": g["weights_step"], "life": g["life"],
                        "digest": g["digest"], "layout": g["layout"],
                        "rollouts": g["rollouts"]}) + "\n")
                    continue
                fh.write(json.dumps({
                    "learner_step": self.step, "env": g["env"],
                    "worker": g["worker"], "weights_step": g["weights_step"],
                    "chunks": [c.reshape(-1).tolist() for c in g["chunks"]],
                    "rollouts": [{
                        "reward": ro["reward"], "ce": ro["ce"],
                        "n_writes": ro["n_writes"],
                        "actions": [r["a"] for r in ro["recs"]],
                        "p": [round(float(r["p"]), 4) for r in ro["recs"]],
                    } for ro in g["rollouts"]],
                }) + "\n")

    # ── l'update du bras rti ────────────────────────────────────────────────
    def step_rti(self, groups) -> dict:
        """Un pas CISPO sur des groupes rti (`rti_learner` : passe 2, avantage
        à deux niveaux, crédit contrefactuel). Le modèle reste en fp32 et la
        passe 2 ne prend PAS d'autocast : le décalage numérique sampler/learner
        est un facteur de premier ordre sur les ratios."""
        self.model.train()
        self.opt.zero_grad(set_to_none=True)
        agg = step_groups(self.model, groups, self.rti_cfg, self.rti_pol,
                          self.lcfg, device=self.device, amp=False)
        gn = float(torch.nn.utils.clip_grad_norm_(
            [p for p in self.model.parameters() if p.requires_grad],
            float(self.r.get("grad_clip", 1.0))))
        if agg["n_groups"]:
            self.opt.step()
        self.step += 1
        if self.step % self.publish_every == 0:
            self.hub.publish(self.model.state_dict(), self.step)
        line = telemetry(agg)
        line.update(step=self.step, grad_norm=gn,
                    lag=st.mean([self.step - 1 - g["weights_step"]
                                 for g in groups]),
                    env_mix={g["env"]: 1 for g in groups})
        return line

    @staticmethod
    def fmt_rti(line: dict, n_stale: int = 0) -> str:
        """La ligne du bras rti. `hit`/`top1` séparent un échec de SÉLECTION
        d'un échec de COPIE, les trois ratios disent lequel des trois canaux
        part off-policy, `cfΔ` dit si les slots injectés portent quelque
        chose. Une métrique absente s'affiche `—` plutôt que de tuer le run."""
        f = lambda k: ("—" if line.get(k) is None else f"{line[k]:+.3f}")
        return (f"step {line['step']:4d}  r {f('reward')}/{f('reward_max')}  "
                f"hit {f('hit')} top1 {f('top1')}  "
                f"p(w) {f('p_write')} H {f('h_write')}  "
                f"ratio w/r/d {f('r_write')}/{f('r_retr')}/{f('r_dec')} "
                f"clip {line['clip_frac']:.2f}  cfΔ {f('cf_delta')}  "
                f"|A| {f('adv')}  gn {line['grad_norm']:.2e}  "
                f"loss {line['loss']:+.3f}  groups {line['groups']}"
                f"(-{line['dropped']})  act {line['n_act']}  "
                f"lag {line['lag']:.1f}  stale {n_stale}  "
                f"{line.get('s_per_step', 0.0):.1f}s/step")

    def step_once(self, groups) -> dict:
        """One GRPO update from consumed groups (advantages in-group)."""
        fmt = {g.get("format") for g in groups}
        if fmt == {"rti"}:
            return self.step_rti(groups)
        assert "rti" not in fmt, (
            f"lot HÉTÉROGÈNE {fmt} : les deux bras n'ont ni les mêmes actions "
            "ni le même avantage, un lot mixte ferait deux updates masquées "
            "l'une par l'autre. Le worker refuse déjà de mélanger les kinds.")
        self.model.train()
        self.opt.zero_grad(set_to_none=True)
        p0 = next(self.model.parameters())
        m = {"reward": [], "ce": [], "writes": 0, "turns": 0, "loss": [],
             "kl": [], "env": {}, "p": [], "lag": []}
        rolls_flat = []
        for g in groups:
            m["lag"].append(self.step - g["weights_step"])
            rolls = group_to_device(g, self.device, p0.dtype)
            rs = [ro["reward"] for ro in rolls]
            mu, sd_r = st.mean(rs), st.pstdev(rs)
            advs = [(x - mu) / (sd_r + 1e-6) for x in rs]
            lo, kl = grpo_backward(self.model, self.ref, rolls, advs,
                                   self.temp, self.clip_lo, self.clip_hi,
                                   self.kl_coef, self.ids,
                                   scale=1.0 / (len(groups) * len(rolls)))
            rolls_flat += rolls
            m["loss"].append(lo)
            m["kl"].append(kl)
            m["reward"] += rs
            m["ce"] += [ro["ce"] for ro in rolls]
            m["writes"] += sum(ro["n_writes"] for ro in rolls)
            m["turns"] += sum(len(ro["recs"]) for ro in rolls)
            m["p"] += [rec["p"] for ro in rolls for rec in ro["recs"]]
            m["env"][g["env"]] = m["env"].get(g["env"], 0) + 1
        torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                       float(self.r.get("grad_clip", 1.0)))
        self.opt.step()
        self.step += 1
        if self.step % self.publish_every == 0:
            self.hub.publish(self.model.state_dict(), self.step)
        return {"step": self.step, "reward": st.mean(m["reward"]),
                "ce": st.mean(m["ce"]),
                "write_rate": m["writes"] / max(m["turns"], 1),
                # p_write = la POLITIQUE (la Bernoulli), write_rate = ce qu'elle
                # a tiré. Les deux divergent quand explore_floor porte le write.
                "p_write": st.mean(m["p"]) if m["p"] else None,
                "kl": st.mean(m["kl"]), "loss": sum(m["loss"]),
                "pos_corr": pos_write_corr(rolls_flat),
                "lag": st.mean(m["lag"]) if m["lag"] else None,
                "groups": len(groups), "env_mix": m["env"]}

    def run(self):
        r = self.r
        steps = int(r["steps"])
        save_every = int(r.get("save_every", 50))
        t0 = time.time()
        n_stale_tot = 0
        while self.step < steps:
            # STOP = global ; STOP_LEARNER = seulement moi (les workers
            # continuent de produire pendant un redémarrage du learner)
            if (os.path.exists(os.path.join(self.root, "STOP"))
                    or os.path.exists(os.path.join(self.root, "STOP_LEARNER"))):
                break
            groups, n_stale = self.store.take(self.gps,
                                              self.step - self.max_lag)
            n_stale_tot += n_stale
            if not groups:
                time.sleep(self.poll_s)
                continue
            self.archive(groups)
            line = self.step_once(groups)
            line["stale"] = n_stale_tot
            line["pending"] = self.store.pending()
            line["t"] = time.time()
            line["s_per_step"] = (time.time() - t0) / max(self.step, 1)
            pw = line["p_write"]
            if self.rti_on:
                print(self.fmt_rti(line, n_stale_tot), flush=True)
                with open(self.mfile, "a") as fh:
                    fh.write(json.dumps(line) + "\n")
                if self.step % save_every == 0 or self.step >= steps:
                    _atomic_save({"model": self.model.state_dict(),
                                  "opt": self.opt.state_dict(),
                                  "step": self.step}, self.ck_path)
                continue
            print(f"step {line['step']:4d}  r {line['reward']:+.3f}  "
                  f"ce {line['ce']:.3f}  "
                  f"p(w) {'—' if pw is None else f'{pw:.2f}'}  "
                  f"write% {line['write_rate']:.2f}  "
                  f"kl {line['kl']:.2e}  groups {line['groups']}  "
                  f"lag {line['lag']:.1f}  stale {n_stale_tot}  "
                  f"{line['env_mix']}  "
                  f"{line['s_per_step']:.1f}s/step", flush=True)
            with open(self.mfile, "a") as fh:
                fh.write(json.dumps(line) + "\n")
            if self.step % save_every == 0 or self.step >= steps:
                _atomic_save({"model": self.model.state_dict(),
                              "opt": self.opt.state_dict(),
                              "step": self.step}, self.ck_path)
        _atomic_save({"model": self.model.state_dict(),
                      "opt": self.opt.state_dict(), "step": self.step},
                     self.ck_path)
        print(f"learner: done at step {self.step}", flush=True)


# ── CLI ──────────────────────────────────────────────────────────────────────

def main(argv):
    role, cfg_path = argv[0], argv[1]
    raw = load_yaml(cfg_path)
    check_cfg(raw, "rl_disagg")
    if role == "learner":
        Learner(raw).run()
    elif role == "worker":
        if "--worker" in argv:
            wid = int(argv[argv.index("--worker") + 1])
        else:
            # farm convention: WORKER = "<hostname>-gpuN" (gpu_worker.sh) —
            # the trailing digits are the per-rig GPU index
            import re
            m = re.search(r"(\d+)$", os.environ.get("WORKER", "0"))
            wid = int(m.group(1)) if m else 0
        Worker(raw, wid).run()
    else:
        raise SystemExit(f"role {role!r} not in (learner, worker)")


# ── CPU self-test (tiny model, stub envs, in-process) ────────────────────────

def _self_test():
    import shutil
    import tempfile
    from .config import ThoughtBankConfig
    from .model import ThoughtBankLM

    root = tempfile.mkdtemp(prefix="rl_disagg_")
    torch.manual_seed(0)
    V, THINK, BLANK, IM_END = 96, 1, 2, 3
    cfg = ThoughtBankConfig(vocab_size=V, d_model=32, n_layers=2, n_heads=2,
                            d_head=8, max_seq_len=128, n_hc=2,
                            sinkhorn_iters=5, csa_m=4, hca_m=8, top_k_csa=2,
                            n_win=4, d_latent_q=16, n_groups=1, n_experts=2,
                            n_shared=1, top_k_experts=1, d_ff=64,
                            mem_dim=16, max_mem=4, mem_seed_slots=2,
                            use_dual_stream=True)
    model = ThoughtBankLM(cfg)

    class _Tok:                                # decode: call / code / garbage
        def __init__(self, rng):
            self._r = rng

        def decode(self, ids):
            return self._r.choice(
                ['{"name": "fn_0", "arguments": {"x": 0}}',
                 "```python\ndef add(a, b):\n    return a + b\n```",
                 "not a call"])

        def get_vocab(self):
            return {"<think>": THINK, "<blank>": BLANK, "<|im_end|>": IM_END}

        def convert_tokens_to_ids(self, t):
            return self.get_vocab().get(t, 0)

        def __call__(self, s, add_special_tokens=False):
            return {"input_ids": [5]}

    class StubStream:
        """Chat-shaped conv dicts with gold_calls (tool bridge)."""

        def __init__(self, seed):
            self.rng = _random.Random(seed)

        def next_conv(self):
            n = self.rng.randint(3, 5)
            segs = [{"input_ids": torch.randint(4, V, (1, 12))}
                    for _ in range(n)]
            return {"kind": "toolcall", "segs": segs,
                    "info": {"gold_calls":
                             [[{"name": "fn_0", "arguments": {"x": 0}}]]}}

    class ExecStub:
        """Chat-shaped conv dicts with tests (exec bridge, REAL sandbox)."""

        def __init__(self, seed):
            self.rng = _random.Random(seed)

        def next_conv(self):
            n = self.rng.randint(3, 5)
            segs = [{"input_ids": torch.randint(4, V, (1, 12))}
                    for _ in range(n)]
            return {"kind": "codeexec", "segs": segs,
                    "info": {"tests": [["assert add(1, 2) == 3"]]}}

    class CodeStub:
        def __init__(self, seed):
            self.rng = _random.Random(seed)

        def next_conv(self):
            return [{"input_ids": torch.randint(4, V, (1, 12))}
                    for _ in range(self.rng.randint(3, 5))]

    raw = {"model": {"n_layers": 2, "max_mem": 4, "mem_seed_slots": 2},
           "data": {"defer_len": 8,
                    "envs": [{"name": "code", "lambda_write": 0.03},
                             {"name": "tools", "lambda_write": 0.0},
                             {"name": "exec", "lambda_write": 0.0,
                              "max_new": 4}]},
           "rl": {"seed": 0, "steps": 3, "group_size": 4,
                  # temp 8: a random tiny model has p(think) ~ 1/V => every
                  # group degenerates (no writes anywhere); flattening the
                  # Bernoulli to ~0.5 exercises the dense-reward path too
                  "groups_per_step": 2, "lr": 1e-4, "temp": 8.0,
                  "min_reward_std": 1e-6, "max_resample": 8,
                  "n_lives_per_worker": 2, "max_new": 4, "amp": False,
                  "think_nmax": 8,
                  "disagg": {"root": root, "publish_every": 1, "max_lag": 2,
                             "poll_s": 0.01, "xdom_every": 0}}}

    # 1. hub round-trip + prune
    hub = WeightHub(root, keep=2)
    for s in (0, 1, 2):
        hub.publish(model.state_dict(), s)
    assert hub.latest_step() == 2
    assert hub.fetch(2) is None and hub.fetch(1)[1] == 2
    assert len([p for p in os.listdir(os.path.join(root, "weights"))
                if p.endswith(".pt")]) == 2   # pruned to keep

    # 2. learner init (publishes step 0 over the pruned hub)
    learner = Learner(raw, model=copy.deepcopy(model),
                      device=torch.device("cpu"))
    assert hub.latest_step() == 0
    meta = json.load(open(os.path.join(root, "meta.json")))
    assert meta["steps"] == 3 and meta["max_lag"] == 2
    assert set(meta["envs"]) == {"code", "tools", "exec"}

    # 3. worker: envs injected, produces groups against published weights
    tok = _Tok(_random.Random(3))
    envs = [EnvSpec("code", CodeStub(1), weight=1.0),
            EnvSpec("tools", StubStream(2), weight=1.0,
                    reward_fn=make_tool_reward(8)),
            EnvSpec("exec", ExecStub(4), weight=1.0,
                    reward_fn=make_exec_reward(8))]
    w = Worker(raw, 0, tok=tok, model=copy.deepcopy(model), envs=envs,
               device=torch.device("cpu"))
    w.ids = (THINK, BLANK)
    w.stop_id, w.max_new = IM_END, 4
    w.a_open = torch.tensor([[5]], dtype=torch.long)
    w.wait_weights()
    assert w.wstep == 0
    lines = []
    while len(lines) < 9 or \
            {ln["env"] for ln in lines} != {"code", "tools", "exec"}:
        got = w.one_group()
        if got:
            lines.append(got)
    assert w.store.pending() == len(lines)
    envs_seen = {ln["env"] for ln in lines}    # all three reward paths
    assert lines[0]["turns"] > 0
    # le succès BRUT remonte pour les envs à rubrique et reste absent des envs
    # denses : sans lui, un reward bas ne dit pas si l'appel est faux ou le
    # think trop long
    for ln in lines:
        assert 0.0 <= ln["p_write"] <= 1.0 and ln["pending"] >= 0
        if ln["env"] in ("tools", "exec"):
            assert 0.0 <= ln["grade"] <= 1.0, ln
        else:
            assert ln["grade"] is None, ln
    assert any(ln["grade"] for ln in lines if ln["env"] == "tools"), \
        "le stub renvoie parfois l'appel gold : au moins un grade non nul"

    # 4. staleness: a group tagged far behind gets quarantined
    g_old = torch.load(os.path.join(w.store.inc,
                                    sorted(os.listdir(w.store.inc))[0]),
                       map_location="cpu", weights_only=False)
    w.store.put(g_old, weights_step=-10, worker=9)

    # 5. learner consumes (quarantining the stale one), steps, republishes
    groups, n_stale = learner.store.take(len(lines) + 2, -5)
    assert n_stale == 1 and len(groups) == len(lines)
    line = learner.step_once(groups)
    assert line["step"] == 1 and line["groups"] == len(lines)
    # lag = retard des poids qui ont produit les groupes (0 ici : tout frais)
    assert line["lag"] == 0.0 and 0.0 <= line["p_write"] <= 1.0
    assert hub.latest_step() == 1
    assert w.refresh() and w.wstep == 1

    # 6. rewards sane: rubric rewards (tools AND exec) within [0, 1]
    for g in groups:
        if g["env"] in ("tools", "exec"):
            assert all(0.0 <= ro["reward"] <= 1.0 for ro in g["rollouts"])

    # 7. lives persisted + xdom probe returns the four figures
    w.save_lives()
    assert os.path.exists(w.lives_path)
    probe = w.xdom_probe()
    assert {"r_own", "r_xdom", "r_always", "r_never"} <= set(probe)

    # 8. decode_graphs : OFF par défaut (le chemin historique ne voit rien) ;
    #    ON sur CPU = dégradation propre — _graphs_decode rend None (pas de
    #    CUDA), generate reprend, un groupe sort quand même
    assert w.decode_graphs is False and w._graph_runner is None
    raw_g = copy.deepcopy(raw)
    raw_g["rl"]["decode_graphs"] = True
    wg = Worker(raw_g, 1, tok=tok, model=copy.deepcopy(model), envs=envs,
                device=torch.device("cpu"))
    wg.ids = (THINK, BLANK)
    wg.stop_id, wg.max_new = IM_END, 4
    wg.a_open = torch.tensor([[5]], dtype=torch.long)
    wg.wait_weights()
    assert wg.decode_graphs is True
    assert wg._graphs_decode(torch.rand(2, wg.max_mem, 4), None, 4) is None, \
        "sur CPU _graphs_decode doit rendre None (generate reprend)"
    got = None
    for _ in range(20):
        got = wg.one_group()
        if got:
            break
    assert got is not None and wg._graph_runner is None, \
        "decode_graphs ON sur CPU : les groupes doivent sortir via generate"

    # 9. BRAS RTI : un worker de bout en bout sur l'env de rappel scripté.
    #    Ce qui est prouvé ici et nulle part ailleurs : le kind `recall_env`
    #    est atteignable depuis la config, les G rollouts d'un groupe partagent
    #    UN script (appariement), le groupe expédié porte les trois actions
    #    avec leurs log-probs, et le learner v1 REFUSE bruyamment de le
    #    consommer (l'algo est le chantier de l'agent 3).
    from . import persona_chat_data as _P
    from .rti_policy import probe_pairs

    class _RtiTok(_P._StubTok):               # ids = ord(c) : vocab 512 suffit
        def get_vocab(self):
            return {"<think>": 1, "<blank>": 2, "<|im_end|>": 3}

        def convert_tokens_to_ids(self, t):
            return self.get_vocab().get(t, 0)

    Vr = 512
    cfg_r = ThoughtBankConfig(vocab_size=Vr, d_model=32, n_layers=2, n_heads=2,
                              d_head=8, max_seq_len=512, n_hc=2,
                              sinkhorn_iters=5, csa_m=4, hca_m=8, top_k_csa=2,
                              n_win=4, d_latent_q=16, n_groups=1, n_experts=2,
                              n_shared=1, top_k_experts=1, d_ff=64, mem_dim=16,
                              max_mem=4, mem_seed_slots=4, use_dual_stream=True)
    raw_r = copy.deepcopy(raw)
    raw_r["model"] = {"n_layers": 2, "max_mem": 4, "mem_seed_slots": 4}
    raw_r["data"]["envs"] = [{"name": "recall_env", "kind": "recall_env",
                              "weight": 1.0, "max_new": 4, "gen": {}}]
    # min_reward_std 0 : un modèle ALÉATOIRE ne tombe jamais sur la vérité,
    # tous les rewards valent 0 et le dynamic sampling rejetterait chaque
    # groupe. Ce bloc prouve le CÂBLAGE, pas la variance de la politique.
    raw_r["rl"].update(group_size=4, max_new=4, decode_graphs=False,
                       min_reward_std=0.0, train_scope="rti",
                       learner={"cispo_high": 2.0, "omega": 1.0,
                                "cf_coef": 0.5})
    raw_r["recall_env"] = {"life_seed": 0, "n_facts": [3, 3], "n_probes": [2, 2],
                           "filler_per_fact": [1, 1], "p_beyond": 0.0,
                           "inject_groups": 2, "max_groups": 4,
                           "age_bins": [[2, 4]], "age_weights": [1.0],
                           "strata": {"persona": 0.5, "numeric": 0.5},
                           "surprisal_mode": "nll"}
    root_r = tempfile.mkdtemp(prefix="rl_disagg_rti_")
    raw_r["rl"]["disagg"] = dict(raw["rl"]["disagg"], root=root_r)
    raw_r["rti"] = {"enabled": True, "top_k": 4, "max_groups": 4,
                    "train_groups": 2, "eval_groups": 2, "sep_token": "<blank>",
                    "write_mode": "head", "write_floor": 0.5,
                    "retr_temp": 1.0, "decode_temp": 1.0}
    tok_r = _RtiTok()
    # le learner d'abord : c'est lui qui PUBLIE le state_dict (modules rti
    # compris) que le worker recharge en strict.
    lr = Learner(raw_r, model=ThoughtBankLM(cfg_r), device=torch.device("cpu"))
    assert hasattr(lr.model, "rti_write"), "modules rti absents du learner"
    wr = Worker(raw_r, 2, tok=tok_r, model=ThoughtBankLM(cfg_r),
                envs=build_envs(raw_r["data"], raw_r["rl"], tok_r, 0, raw_r),
                device=torch.device("cpu"))
    assert wr.rti_on and wr.envs[0].kind == "recall_env"
    wr.stop_id, wr.max_new = -1, 4
    wr.a_open = torch.tensor([[65, 58]], dtype=torch.long)
    wr.wait_weights()
    got = None
    for _ in range(6):
        got = wr.one_group()
        if got:
            break
    assert got is not None and got["env"] == "recall_env", got
    assert got["n_dec"] > 0 and 0.0 <= got["p_write"] <= 1.0
    assert got["hit"] is not None and 0.0 <= got["hit"] <= 1.0
    names = [p for p in os.listdir(wr.store.inc) if p.startswith("w02_")]
    grp = torch.load(os.path.join(wr.store.inc, names[0]), map_location="cpu",
                     weights_only=False)
    assert grp["format"] == "rti" and len(grp["rollouts"]) == wr.G
    assert len({t["digest"] for t in grp["rollouts"]}) == 1, \
        "les G rollouts d'un groupe doivent partager UN script"
    anch = [[(t["seg"], t["turn"]) for t in tr["turns"]]
            for tr in grp["rollouts"]]
    assert all(a == anch[0] for a in anch), "ancres de tour désalignées"
    n_ret = n_tokdec = 0
    for tr in grp["rollouts"]:
        assert all(t["write"] is None or "logp" in t["write"] for t in tr["turns"])
        for q, dcd in probe_pairs(tr):
            if q is not None:
                n_ret += 1
                assert "logp" in q["retrieve"] and q["retrieve"]["order"]
            n_tokdec += len(dcd["decode"]["logp"])
            assert len(dcd["decode"]["logp"]) == len(dcd["decode"]["tokens"])
    assert n_ret and n_tokdec, (n_ret, n_tokdec)
    # de quoi REJOUER le groupe sans stream ni tokenizer (agent 3)
    assert len(grp["segs"]) == len(grp["rollouts"][0]["turns"])
    assert grp["sep_id"] == wr.rti_sep and grp["a_open"].numel() > 0
    assert grp["sif"]["ids"].numel() == grp["sif"]["w"].numel() > 0

    # 10. UN PAS D'UPDATE RTI (le (g) de l'agent 3) : les params rti bougent,
    #     le backbone GELÉ ne bouge pas, la loss est finie. Le groupe est
    #     archivé avant consommation, comme sur le chemin historique.
    gr, _ = lr.store.take(1, -5)
    lr.archive(gr)
    assert any(json.loads(l).get("format") == "rti"
               for l in open(lr.trace_path))
    before = {k: v.detach().clone() for k, v in lr.model.named_parameters()}
    # min_reward_std 0 côté worker : le groupe peut être PLAT (modèle
    # aléatoire ⇒ aucune sonde juste). On force alors de la variance pour que
    # le filtrage zéro-variance ne mange pas le seul groupe du test.
    rs = [t["reward"] for t in gr[0]["rollouts"]]
    if rs and all(x is not None for x in rs) and max(rs) - min(rs) < 1e-9:
        for j, t in enumerate(gr[0]["rollouts"]):
            for tt in t["turns"]:
                if tt["decode"] is not None and tt["reward"] is not None:
                    tt["reward"] = float(j % 2)
            ok = [x["reward"] for x in t["turns"]
                  if x["decode"] is not None and x["reward"] is not None]
            t["reward"] = sum(ok) / len(ok)
    line_r = lr.step_once(gr)
    assert math.isfinite(line_r["loss"]) and line_r["groups"] == 1, line_r
    assert math.isfinite(line_r["grad_norm"]) and line_r["n_act"] > 0
    assert 0.0 <= line_r["clip_frac"] <= 1.0 and line_r["p_write"] is not None
    moved = {k for k, v in lr.model.named_parameters()
             if not torch.equal(v.detach(), before[k])}
    assert moved and all(k.startswith("rti_") for k in moved), sorted(moved)
    assert {"rti_retriever.wq.weight", "rti_write.w.weight"} <= moved, \
        f"le retriever et la tête de write doivent bouger (vu {sorted(moved)})"
    # la ligne de télémétrie se FORMATE (un run entier meurt sur une clé
    # absente, et ça arrive au step 1, après le bring-up)
    txt = lr.fmt_rti({**line_r, "s_per_step": 1.0}, 0)
    assert "ratio w/r/d" in txt and "cfΔ" in txt, txt
    assert "—" in lr.fmt_rti({**line_r, "hit": None, "s_per_step": 1.0}, 0)

    shutil.rmtree(root)
    shutil.rmtree(root_r)
    print(f"rl_disagg self-test: OK (hub, store+staleness, worker groups "
          f"[{', '.join(sorted(envs_seen))}], learner step+republish, "
          f"refresh, lives, xdom probe, decode_graphs OFF défaut / ON-CPU "
          f"dégrade en generate, bras RTI de bout en bout : env recall_env "
          f"branché, {wr.G} rollouts appariés (1 script, ancres alignées), "
          f"groupe rti expédié avec les 3 log-probs ({n_ret} tirages "
          f"Plackett-Luce, {n_tokdec} tokens) + segs/a_open/sep/sif pour la "
          f"passe 2, un pas CISPO consommé : loss {line_r['loss']:+.3f}, "
          f"{line_r['n_act']} actions créditées, |∇| "
          f"{line_r['grad_norm']:.2e}, bougent {sorted(moved)} et RIEN "
          f"d'autre)")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(sys.argv[1:])
    else:
        _self_test()
