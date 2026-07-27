"""Le contexte d'exécution d'un run : DDP, tf32, compile, grad-checkpoint.

Ces quatre leviers étaient inlinés dans `code_defer_native.main`, mêlés à une
cinquantaine de locales. Ils n'ont pourtant rien à voir avec l'objectif
d'entraînement : ils décrivent COMMENT le forward tourne, pas CE QU'il apprend.
Sortis ici, ils deviennent testables sur CPU et réutilisables par les entrées RL.

Contrat de non-régression : le code est déplacé, pas réécrit. Tous les leviers
restent opt-in et OFF par défaut, donc une config existante produit exactement
le même run qu'avant l'extraction.

Le piège documenté dans ce fichier, celui qui a coûté des NaN en phase 1 :
    grad_checkpoint recalcule le forward au backward, et le recompute n'est PAS
    bit-identique (cuBLAS peut changer d'algo). Les topk DURS (routage MoE,
    sélection de blocs CSA) flippent alors sur un ULP, et le gradient est
    calculé sur un autre graphe que la loss. D'où `save_topk`.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import torch


# ── DDP ──────────────────────────────────────────────────────────────────────

@dataclass
class DDPInfo:
    """Ce que le reste du trainer a besoin de savoir du parallélisme."""
    world: int
    rank: int
    local_rank: int
    device: torch.device

    @property
    def enabled(self) -> bool:
        return self.world > 1

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def init_ddp(*, timeout_hours: float = 2.0, quiet_non_main: bool = True) -> DDPInfo:
    """Data parallelism SANS le wrapper DDP.

    La boucle de conv fait plusieurs forwards (in-context + defer + addr +
    reach) pour un seul backward, ce qui fait trébucher le reducer de DDP
    (« marked ready twice »). À la place : chaque rang fait ses propres convs
    (banque/cascade/carry = état rank-local, exactement la sémantique
    mono-GPU), et les gradients sont all-reduce à la main avant `opt.step()`.
    Muon/AdamW étant déterministes, grads identiques ⇒ poids identiques sur
    tous les rangs, sans dérive de synchro. Batch effectif = B * G * world_size.
    """
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    if world <= 1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return DDPInfo(world=1, rank=0, local_rank=0, device=device)

    import datetime
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)       # AVANT init : NCCL se lie au device
    #                                         courant — après, chaque rang fait
    #                                         pousser un contexte parasite de
    #                                         ~0.5 Go sur cuda:0
    torch.distributed.init_process_group(
        "nccl", timeout=datetime.timedelta(hours=timeout_hours))  # rank0 évalue,
    #                                         les autres attendent
    device = torch.device(f"cuda:{local_rank}")
    if rank != 0 and quiet_non_main:
        sys.stdout = open(os.devnull, "w")  # un seul flux de log ; stderr reste
    print(f"ddp: world {world} (this = rank {rank}, cuda:{local_rank})", flush=True)
    return DDPInfo(world=world, rank=rank, local_rank=local_rank, device=device)


# ── tf32 ─────────────────────────────────────────────────────────────────────

def enable_tf32(on: bool) -> bool:
    """Le plus gros gain de la phase 1 : MoE et sinkhorn tournent en fp32, et
    TF32 accélère exactement ces matmuls sur Ampere/Ada pour un coût de
    précision minime. Mesuré au sweep de bring-up avant d'être committé."""
    if not on:
        return False
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    print("tf32: enabled (fp32 matmul -> tf32)", flush=True)
    return True


# ── torch.compile ────────────────────────────────────────────────────────────

def compile_model(model, *, cache_dir: str | None = None):
    """Renvoie le module compilé. L'appelant GARDE le module eager (`base`)
    comme source de vérité pour state_dict / named_parameters / groupes
    d'optimiseur : `torch.compile` enveloppe dans un OptimizedModule et préfixe
    les clés en `_orig_mod.`, ce qui casserait le format de checkpoint et les
    groupes Muon nommés. Le compilé partage les mêmes Parameter, donc
    l'optimiseur construit sur `base` pilote bien le modèle appelé.
    """
    # dynamic=False : nos shapes sont statiques par construction ([B,512] et
    # [B,16] ; m ne change que le NOMBRE d'appels) — un graphe statique par
    # shape. Sans ça, la transition automatique statique→dynamique fait choisir
    # au recompute du grad_checkpoint un graphe différent du forward
    # (CheckpointError « different number of tensors », pytorch #166926).
    # cache_size_limit relevé : le mem_bank flippe requires_grad (write on/off)
    # ⇒ dynamo recompile à chaque flip ; la limite par défaut (8) est crevée
    # vers step 60 (observé pod 45185048 2026-07-17) et le fallback eager
    # désynchronise les graphes entre rangs DDP ⇒ deadlock NCCL (100% util,
    # ~95W). On monte la limite pour que les 8 rangs recompilent en lockstep
    # (depth_sync garantit m identique) sans jamais tomber en fallback.
    from torch._dynamo import config as _dynamo_config
    _dynamo_config.cache_size_limit = 256
    _dynamo_config.accumulated_cache_size_limit = 1024
    # cache_dir (opt-in) : cache inductor+triton PERSISTANT — les kernels
    # compilés survivent aux restarts (préemption pod, resume) au lieu de
    # repayer la compilation à froid à chaque boot. Sur pod : pointer sous
    # /workspace (volume persistant), tar-able vers HF avec le data cache pour
    # réchauffer un pod NEUF (clé de cache = arch GPU + version torch :
    # A100 + même image ⇒ hits). Les env sont lues à la PREMIÈRE compilation
    # (premier forward), donc les poser ici suffit ; setdefault ⇒ un export
    # shell garde la main.
    if cache_dir:
        os.makedirs(os.path.join(cache_dir, "triton"), exist_ok=True)
        os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", cache_dir)
        os.environ.setdefault("TRITON_CACHE_DIR", os.path.join(cache_dir, "triton"))
        print(f"compile: cache persistant → {cache_dir}", flush=True)
    out = torch.compile(model, dynamic=False)
    print("compile: torch.compile(model, dynamic=False) enabled "
          "(dynamo cache_size_limit=256)", flush=True)
    return out


# ── grad checkpointing ───────────────────────────────────────────────────────

def build_fwd(model, *, grad_checkpoint: bool = False, save_topk: bool = True,
              verbose: bool = True):
    """Fabrique l'appelable de forward du trainer.

    Sans grad_checkpoint c'est `model(...)` tout court. Avec, chaque forward est
    rematérialisé au backward : la boucle de conv garde le graphe de CHAQUE
    chunk vivant jusqu'à l'unique backward de fin de conv (TBPTT sur la conv
    entière), donc le pic d'activations est en O(K * B * tirage de profondeur) —
    et c'est exactement la variance de profondeur d'un step à l'autre qui fait
    OOM à B≥8 sur 80 Go. Le checkpoint ne garde que les entrées par forward
    ⇒ pic ≈ les activations d'un seul chunk, gradients EXACTS (même graphe,
    recalculé), ~+30% de compute.

    `save_topk` (défaut ON) : les grads du checkpoint ne sont « exacts » que si
    le recompute est bit-identique au forward — faux en pratique. cuBLAS peut
    choisir un autre algo au recompute (workspace différent pendant le
    backward), les scores bougent d'un ULP, et les TOPK DURS (routage MoE,
    sélection de blocs CSA) FLIPPENT ⇒ gradients calculés sur un autre graphe
    que la loss. Suspect n°1 des NaN (incident phase 1 step 2520, run 5e
    step 33 — tous deux GC ON). Fix : selective activation checkpointing — les
    sorties de topk sont SAUVÉES au forward et rejouées au recompute (coût
    mémoire = des indices).
    """
    if not grad_checkpoint:
        return lambda *a, **k: model(*a, **k)

    from torch.utils.checkpoint import checkpoint as _ckpt
    ckpt_ctx = None
    if save_topk:
        from torch.utils.checkpoint import (create_selective_checkpoint_contexts,
                                            CheckpointPolicy)
        save_ops = {torch.ops.aten.topk.default}

        def _sac_policy(ctx, op, *args, **kwargs):
            return (CheckpointPolicy.MUST_SAVE if op in save_ops
                    else CheckpointPolicy.PREFER_RECOMPUTE)

        ckpt_ctx = lambda: create_selective_checkpoint_contexts(_sac_policy)
        if verbose:
            print("grad_checkpoint: ON + save-topk (routage figé au recompute)",
                  flush=True)
    elif verbose:
        print("grad_checkpoint: ON (rematerialized forwards)", flush=True)

    def _fwd(*a, **k):
        if torch.is_grad_enabled():
            if ckpt_ctx is not None:
                return _ckpt(model, *a, use_reentrant=False,
                             context_fn=ckpt_ctx, **k)
            return _ckpt(model, *a, use_reentrant=False, **k)
        return model(*a, **k)

    return _fwd


# ── self-test ────────────────────────────────────────────────────────────────

def _selftest() -> None:
    """Ce qui doit tenir : le grad-checkpoint donne les MÊMES gradients que le
    forward direct (c'est toute sa promesse), il ne s'active pas sous no_grad,
    et le chemin save-topk est réellement pris quand on le demande.
    """
    import torch.nn as nn

    # ── DDP mono-process : le chemin par défaut de toutes les configs ────────
    for var in ("WORLD_SIZE", "RANK", "LOCAL_RANK"):
        os.environ.pop(var, None)
    info = init_ddp()
    assert info.world == 1 and info.rank == 0 and not info.enabled and info.is_main
    assert info.device.type in ("cpu", "cuda")

    # ── tf32 : no-op quand OFF (une config existante ne doit RIEN changer) ───
    before = torch.backends.cuda.matmul.allow_tf32
    assert enable_tf32(False) is False
    assert torch.backends.cuda.matmul.allow_tf32 == before, \
        "enable_tf32(False) a quand même touché aux backends"

    # ── build_fwd : gradients identiques avec et sans checkpoint ────────────
    torch.manual_seed(0)

    class Toy(nn.Module):
        """Renvoie un dict comme ThoughtBankLM : le checkpoint non-réentrant
        doit supporter une sortie structurée, pas seulement un tenseur."""

        def __init__(self):
            super().__init__()
            self.l1 = nn.Linear(8, 16)
            self.l2 = nn.Linear(16, 8)

        def forward(self, x):
            h = torch.relu(self.l1(x))
            return {"out": self.l2(h), "aux": h.sum()}

    x = torch.randn(4, 8)

    def _grads(fwd, m):
        m.zero_grad(set_to_none=True)
        o = fwd(x)
        (o["out"].sum() + o["aux"]).backward()
        return [p.grad.clone() for p in m.parameters()]

    plain = Toy()
    ref = _grads(build_fwd(plain, grad_checkpoint=False), plain)

    for topk in (False, True):
        m = Toy()
        m.load_state_dict(plain.state_dict())
        got = _grads(build_fwd(m, grad_checkpoint=True, save_topk=topk,
                               verbose=False), m)
        for a, b in zip(ref, got):
            assert torch.allclose(a, b, atol=0, rtol=0), \
                f"grad_checkpoint (save_topk={topk}) change les gradients"

    # ── sous no_grad, le checkpoint ne doit pas s'interposer ────────────────
    m = Toy()
    fwd = build_fwd(m, grad_checkpoint=True, save_topk=True, verbose=False)
    with torch.no_grad():
        o = fwd(x)
    assert not o["out"].requires_grad, "no_grad : le checkpoint s'est interposé"
    assert torch.allclose(o["out"], m(x)["out"])

    # ── kwargs traversent le checkpoint (le trainer appelle _fwd(x, init_mem=…)) ─
    class Kw(nn.Module):
        def __init__(self):
            super().__init__()
            self.l = nn.Linear(8, 8)

        def forward(self, x, *, scale=1.0):
            return {"out": self.l(x) * scale}

    k = Kw()
    fk = build_fwd(k, grad_checkpoint=True, save_topk=False, verbose=False)
    ok = fk(x, scale=2.0)
    ok["out"].sum().backward()
    assert torch.allclose(ok["out"].detach(), k(x, scale=2.0)["out"].detach())
    assert k.l.weight.grad is not None, "aucun gradient à travers le checkpoint"

    print("runtime self-test: OK (ddp mono-process, tf32 no-op quand OFF, "
          "grad_checkpoint = gradients BIT-IDENTIQUES avec et sans save-topk, "
          "inerte sous no_grad, kwargs préservés)")


if __name__ == "__main__":
    _selftest()
