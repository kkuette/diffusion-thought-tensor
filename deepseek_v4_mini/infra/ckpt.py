"""Sauvegarde et reprise d'un run — la seule chose qui rende une préemption
de pod inoffensive.

Un checkpoint qui ne porte que les poids ne reprend pas un run : il en démarre
un autre depuis les mêmes poids. Pour reprendre il faut aussi l'optimiseur, les
EMA de télémétrie, et surtout les RNG — celui de torch, ceux de cuda, et ceux
des streams de données, sinon la reprise rejoue les mêmes convs.

Deux invariants tiennent ce fichier :

  * **écriture atomique** — `.tmp` puis `os.replace`. Une préemption au milieu
    d'un `torch.save` de 3 Go laisserait sinon un fichier tronqué à la place du
    dernier checkpoint valide, et le run serait perdu, pas interrompu.
  * **la banque est un artefact séparé** — `bank_step_N.pt` se recharge au
    resume, mais s'échange aussi entre runs (`bank_init`) et s'inspecte hors
    trainer. C'est de l'état de session, pas des poids.
"""
from __future__ import annotations

import glob
import os
import re

import torch

from ..core.cascade import CascadeMemory


# ── écriture ─────────────────────────────────────────────────────────────────

def atomic_save(obj, path: str) -> None:
    """`.tmp` + `os.replace` : jamais de fichier déchiré sous préemption."""
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def save_train_state(path: str, *, step: int, model, cfg, opt, delta=None,
                     ema_ic=None, ema_d=None, train_stream=None,
                     eval_stream=None, chat_stream=None,
                     extra: dict | None = None) -> None:
    """L'état d'entraînement COMPLET : resume sûr sur GPU loué / spot.

    `model` est le module EAGER (`base`), jamais le compilé : `torch.compile`
    préfixe les clés en `_orig_mod.` et casserait le format.
    """
    atomic_save({"step": step, "model": model.state_dict(), "cfg": cfg.__dict__,
                 "opt": opt.state_dict(),
                 "adam": opt._adam.state_dict() if opt._adam else None,
                 "delta": delta.state_dict() if delta is not None else None,
                 "ema_ic": ema_ic, "ema_d": ema_d,
                 "rng_torch": torch.get_rng_state(),
                 "rng_cuda": (torch.cuda.get_rng_state_all()
                              if torch.cuda.is_available() else None),
                 "rng_train_stream": (train_stream.rng.getstate()
                                      if train_stream is not None else None),
                 "rng_eval_stream": (eval_stream.rng.getstate()
                                     if eval_stream is not None else None),
                 "rng_chat_stream": (chat_stream.rng.getstate()
                                     if chat_stream is not None else None),
                 # État NON reconstructible depuis le rng : un stream qui tire
                 # des sessions d'avance (chat_batch : pool de pavage, files de
                 # lane) doit dire ce qu'il a déjà en main, sinon le resume
                 # repart au milieu d'une autre session. Protocole optionnel.
                 "chat_stream_state": _stream_state(chat_stream),
                 # états annexes du trainer (ex. tables du teacher value_table)
                 **(extra or {})}, path)


def _stream_state(stream):
    """`stream.state_dict()` s'il l'expose, sinon None (contrat optionnel)."""
    fn = getattr(stream, "state_dict", None) if stream is not None else None
    return fn() if callable(fn) else None


def save_bank(path: str, *, step: int, bank, casc=None, n_evict: int = 0,
              w_total: int = 0) -> bool:
    """La banque COMPLÈTE (vive + cascade). Renvoie False si rien à sauver."""
    if bank is None:
        return False
    atomic_save({"step": step, "bank": bank.detach().cpu(),
                 "casc": casc.state_dict() if casc is not None else None,
                 "n_evict": n_evict, "w_total": w_total}, path)
    return True


def load_bank(path: str, tag: str, device):
    """(bank, cascade, n_evict, w_total) — le pendant de `save_bank`."""
    bk = torch.load(path, map_location="cpu", weights_only=False)
    casc = (CascadeMemory.from_state(bk["casc"], device=device)
            if bk.get("casc") is not None else None)
    print(f"{tag}: banque chargée depuis {path} (step d'origine "
          f"{bk.get('step')}, cascade {'oui' if casc else 'non'})", flush=True)
    return (bk["bank"].to(device), casc,
            int(bk.get("n_evict", 0)), int(bk.get("w_total", 0)))


# ── reprise ──────────────────────────────────────────────────────────────────

def find_resume(save_dir: str):
    """(step, chemin, terminé) — le plus grand `step_N.pt` du répertoire.

    `terminé` est vrai si `final.pt` existe : le run est allé au bout, une
    reprise n'aurait rien à faire (et écraserait un résultat en le refaisant).
    Renvoie (None, None, …) si aucun checkpoint intermédiaire n'est présent.
    """
    done = os.path.exists(os.path.join(save_dir, "final.pt"))
    cks = {}
    for p in glob.glob(os.path.join(save_dir, "step_*.pt")):
        mt = re.match(r"step_(\d+)\.pt$", os.path.basename(p))
        if mt:
            cks[int(mt.group(1))] = p
    if not cks:
        return None, None, done
    step = max(cks)
    return step, cks[step], done


def restore_train_state(ck, *, model, opt, delta=None, train_stream=None,
                        eval_stream=None, chat_stream=None, ddp_rank: int = 0,
                        train_seed: int = 0, base_seed: int = 0,
                        start_step: int = 0):
    """Recharge poids, optimiseur, EMA et RNG. Renvoie (ema_ic, ema_d).

    Les RNG de streams sont le point délicat en DDP : le checkpoint ne porte
    QUE celui du rang 0. Le restaurer sur tous les rangs les ferait échantillonner
    les MÊMES convs — le parallélisme de données s'évaporerait en silence. Le
    rang 0 reprend son flux exact ; les autres se re-sèment sur (rang, step).
    """
    model.load_state_dict(ck["model"])
    if ck.get("opt") is not None:
        opt.load_state_dict(ck["opt"])
    if ck.get("adam") is not None and opt._adam is not None:
        opt._adam.load_state_dict(ck["adam"])
    if ck.get("delta") is not None and delta is not None:
        delta.load_state_dict(ck["delta"])

    if ck.get("rng_torch") is not None:
        torch.set_rng_state(ck["rng_torch"])
    if ck.get("rng_cuda") is not None and torch.cuda.is_available():
        # migration de world size (ex. 8→6 GPUs) : le ck porte un état RNG par
        # device du node d'origine — ne restaurer que les nôtres
        torch.cuda.set_rng_state_all(ck["rng_cuda"][:torch.cuda.device_count()])

    if train_stream is not None:
        if ck.get("rng_train_stream") is not None and ddp_rank == 0:
            train_stream.rng.setstate(ck["rng_train_stream"])
        elif ddp_rank != 0:
            train_stream.rng.seed(train_seed + start_step)
        # depth_sync : rng_m doit rester en LOCKSTEP entre rangs après resume —
        # re-seed déterministe sur (seed de base, start_step), le même partout.
        if train_stream.rng_m is not train_stream.rng:
            train_stream.rng_m.seed(base_seed + start_step)
    if eval_stream is not None and ck.get("rng_eval_stream") is not None:
        eval_stream.rng.setstate(ck["rng_eval_stream"])
    if chat_stream is not None and ck.get("rng_chat_stream") is not None:
        if ddp_rank == 0:
            chat_stream.rng.setstate(ck["rng_chat_stream"])
        else:
            chat_stream.rng.seed(train_seed + 1 + start_step)
    # Sessions déjà tirées (pool de pavage, files de lane). Rang 0 seulement :
    # les autres rangs ont re-seedé, donc reprendre l'état du rang 0 les
    # désynchroniserait de leur propre suite de tirages.
    st = ck.get("chat_stream_state")
    if st is not None and ddp_rank == 0:
        fn = getattr(chat_stream, "load_state_dict", None)
        if callable(fn):
            fn(st)

    return ck.get("ema_ic"), ck.get("ema_d")


# ── self-test ────────────────────────────────────────────────────────────────

def _selftest() -> None:
    """Le test qui compte : après restore, le RNG rejoue la SUITE, pas le début.

    Un resume qui restaure les poids mais pas les RNG a l'air de marcher — la
    loss repart au bon niveau — et refait pourtant exactement les mêmes tirages
    qu'avant la coupure. C'est invisible dans les métriques et ça fausse un run.
    """
    import random
    import tempfile
    import torch.nn as nn

    class _FakeOpt:
        """L'interface que ckpt.py utilise d'un Muon : state_dict + _adam."""

        def __init__(self, v=0.0):
            self.v = v
            self._adam = None

        def state_dict(self):
            return {"v": self.v}

        def load_state_dict(self, sd):
            self.v = sd["v"]

    class _FakeStream:
        def __init__(self, seed=0, split_m=False):
            self.rng = random.Random(seed)
            self.rng_m = random.Random(seed + 1) if split_m else self.rng

    class _Cfg:
        def __init__(self):
            self.__dict__.update(d_model=8, max_mem=4)

    work = tempfile.mkdtemp(prefix="ckpt_selftest_")

    # ── écriture atomique : pas de .tmp qui traîne, pas de fichier déchiré ──
    p = os.path.join(work, "x.pt")
    atomic_save({"a": 1}, p)
    assert os.path.exists(p) and not os.path.exists(p + ".tmp")
    assert torch.load(p, weights_only=False)["a"] == 1

    # ── find_resume : le max, le bruit ignoré, final.pt signalé ─────────────
    sd_dir = os.path.join(work, "run")
    os.makedirs(sd_dir)
    assert find_resume(sd_dir) == (None, None, False), "répertoire vide"
    for name in ("step_50.pt", "step_200.pt", "step_100.pt",
                 "step_abc.pt", "bank_step_200.pt", "step_200.pt.tmp"):
        open(os.path.join(sd_dir, name), "w").close()
    step, path, done = find_resume(sd_dir)
    assert step == 200 and os.path.basename(path) == "step_200.pt" and not done, \
        f"find_resume a choisi {step} / {path}"
    open(os.path.join(sd_dir, "final.pt"), "w").close()
    assert find_resume(sd_dir)[2] is True, "final.pt non signalé ⇒ run refait"

    # ── round-trip complet, et surtout la reprise des RNG ───────────────────
    torch.manual_seed(0)
    model = nn.Linear(8, 8)
    opt = _FakeOpt(v=3.5)
    tr, ev, ch = _FakeStream(1, split_m=True), _FakeStream(2), _FakeStream(3)

    # on brûle quelques tirages pour que l'état sauvé ne soit pas l'état initial
    for _ in range(5):
        tr.rng.random(); ev.rng.random(); ch.rng.random()
    torch.randn(3)

    ckp = os.path.join(work, "step_10.pt")
    save_train_state(ckp, step=10, model=model, cfg=_Cfg(), opt=opt,
                     ema_ic=1.25, ema_d=0.5,
                     train_stream=tr, eval_stream=ev, chat_stream=ch)

    # la suite attendue : ce que les RNG produisent APRÈS le point de sauvegarde
    want_tr = [tr.rng.random() for _ in range(3)]
    want_ev = [ev.rng.random() for _ in range(3)]
    want_ch = [ch.rng.random() for _ in range(3)]
    want_torch = torch.randn(3)

    with torch.no_grad():
        model.weight.mul_(7.0)                       # on abîme tout…
    opt.v = -1.0
    tr2, ev2, ch2 = _FakeStream(1, split_m=True), _FakeStream(2), _FakeStream(3)
    model2 = nn.Linear(8, 8)
    opt2 = _FakeOpt()
    torch.manual_seed(999)

    ck = torch.load(ckp, map_location="cpu", weights_only=False)
    ema_ic, ema_d = restore_train_state(ck, model=model2, opt=opt2,
                                        train_stream=tr2, eval_stream=ev2,
                                        chat_stream=ch2, ddp_rank=0,
                                        train_seed=1, base_seed=1, start_step=10)
    assert (ema_ic, ema_d) == (1.25, 0.5)
    assert opt2.v == 3.5, "état d'optimiseur non restauré"
    assert torch.equal(model2.weight, ck["model"]["weight"])
    assert ck["step"] == 10 and ck["cfg"]["d_model"] == 8

    got_tr = [tr2.rng.random() for _ in range(3)]
    got_ev = [ev2.rng.random() for _ in range(3)]
    got_ch = [ch2.rng.random() for _ in range(3)]
    assert got_tr == want_tr, "le stream train rejoue les mêmes convs après resume"
    assert got_ev == want_ev, "le stream d'éval a dérivé"
    assert got_ch == want_ch, "le stream chat a dérivé"
    assert torch.equal(torch.randn(3), want_torch), "RNG torch non restauré"

    # ── DDP : un rang ≠ 0 NE reprend PAS le flux du rang 0 ──────────────────
    tr3 = _FakeStream(1, split_m=True)
    restore_train_state(ck, model=nn.Linear(8, 8), opt=_FakeOpt(),
                        train_stream=tr3, ddp_rank=2,
                        train_seed=1 + 9973 * 2, base_seed=1, start_step=10)
    assert [tr3.rng.random() for _ in range(3)] != want_tr, \
        "tous les rangs échantillonneraient les mêmes convs"
    # …mais rng_m, lui, DOIT être en lockstep (depth_sync)
    tr4 = _FakeStream(1, split_m=True)
    restore_train_state(ck, model=nn.Linear(8, 8), opt=_FakeOpt(),
                        train_stream=tr4, ddp_rank=0,
                        train_seed=1, base_seed=1, start_step=10)
    assert [tr3.rng_m.random() for _ in range(3)] == \
           [tr4.rng_m.random() for _ in range(3)], \
        "rng_m désynchronisé entre rangs ⇒ profondeurs de conv différentes"

    # ── banque : round-trip, et le no-op quand il n'y a rien ────────────────
    bp = os.path.join(work, "bank_step_10.pt")
    assert save_bank(bp, step=10, bank=None) is False, "sauver une banque None"
    assert not os.path.exists(bp)
    bank = torch.randn(2, 4, 16)
    assert save_bank(bp, step=10, bank=bank, casc=None, n_evict=7, w_total=42)
    got, casc, nev, wt = load_bank(bp, "selftest", torch.device("cpu"))
    assert torch.equal(got, bank) and casc is None and (nev, wt) == (7, 42)

    # ── état de stream chat (protocole OPTIONNEL state_dict/load_state_dict) ──
    # Un stream qui tire des sessions d'avance (chat_batch : pool de pavage) a un
    # état que le rng ne reconstruit PAS. Sans lui, le resume repart au milieu
    # d'une autre session — invisible dans les métriques.
    class _StatefulStream(_FakeStream):
        def __init__(self, seed=0):
            super().__init__(seed)
            self.pending = ["a", "b"]

        def state_dict(self):
            return {"pending": list(self.pending)}

        def load_state_dict(self, sd):
            self.pending = list(sd["pending"])

    sp = os.path.join(work, "st_step_20.pt")
    chs = _StatefulStream(3)
    chs.pending = ["x", "y", "z"]
    save_train_state(sp, step=20, model=model, cfg=_Cfg(), opt=_FakeOpt(1.0),
                     train_stream=_FakeStream(0), eval_stream=_FakeStream(1),
                     chat_stream=chs)
    chs.pending = ["MUTÉ"]                      # l'instantané ne doit pas suivre
    ck5 = torch.load(sp, weights_only=False)
    ch5 = _StatefulStream(3)
    restore_train_state(ck5, model=nn.Linear(8, 8), opt=_FakeOpt(),
                        chat_stream=ch5, ddp_rank=0, start_step=20)
    assert ch5.pending == ["x", "y", "z"], ch5.pending
    # un rang DDP != 0 a re-seedé son rng : lui imposer l'état du rang 0 le
    # désynchroniserait de sa propre suite de tirages
    ch6 = _StatefulStream(3)
    restore_train_state(ck5, model=nn.Linear(8, 8), opt=_FakeOpt(),
                        chat_stream=ch6, ddp_rank=1, start_step=20)
    assert ch6.pending == ["a", "b"], ch6.pending
    # et un stream SANS le protocole passe sans lever (contrat optionnel)
    plain = os.path.join(work, "plain_step_20.pt")
    save_train_state(plain, step=20, model=model, cfg=_Cfg(), opt=_FakeOpt(),
                     chat_stream=_FakeStream(7))
    ckp2 = torch.load(plain, weights_only=False)
    assert ckp2["chat_stream_state"] is None
    restore_train_state(ckp2, model=nn.Linear(8, 8), opt=_FakeOpt(),
                        chat_stream=_FakeStream(7), ddp_rank=0, start_step=20)

    import shutil
    shutil.rmtree(work, ignore_errors=True)
    print("ckpt self-test: OK (save atomique sans .tmp résiduel, find_resume "
          "max/bruit/final.pt, round-trip poids+opt+EMA, RNG torch ET streams "
          "reprennent la SUITE, rangs DDP dé-corrélés mais rng_m en lockstep, "
          "banque round-trip, état de stream chat round-trip + no-op sans "
          "le protocole)")


if __name__ == "__main__":
    _selftest()
