#!/usr/bin/env python
"""Le resume redonne-t-il EXACTEMENT le run qu'il interrompt ?

C'est la seule propriété qui rende une préemption de pod inoffensive, et rien
ne la teste : on l'a toujours supposée. Le protocole est direct —

    A : run complet de N steps, checkpoint à N/2
    B : repart du checkpoint N/2 de A, va jusqu'à N
    verdict : les poids finaux de A et B sont-ils bit à bit identiques ?

Si oui, l'état restauré (poids, optimiseur, RNG torch/cuda, RNG des streams)
suffit à reconstruire la trajectoire. Sinon, le rapport dit QUELS tenseurs
divergent et de combien — un écart en 1e-7 pointe une non-déterminisme cuDNN,
un écart franc pointe un morceau d'état qu'on ne sauve pas.

    python scripts/resume_check.py                       # smoke 97M sur GPU
    python scripts/resume_check.py --cpu                 # jouet 2 couches sur CPU
    python scripts/resume_check.py --steps 40 --config <cfg.yaml>

Le mode `--cpu` rétrécit le modèle (2 couches, d_model 128, amp off) et tourne
sans GPU : c'est le test le PLUS strict des deux, puisque le CPU n'a pas la
non-déterminisme kernel qui brouille une comparaison bit à bit sur GPU. Il ne
dépend d'aucun réseau (le cache de chunks `data_cache_smoke/` suffit), donc il
tourne aussi quand la carte est prise par un vrai run.

Hors CI dans les deux cas : quelques minutes, ce n'est pas un test d'unité.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DEFAULT_CFG = "deepseek_v4_mini/configs/v350_fastpath_smoke.yaml"


def _shrink(cfg: dict) -> None:
    """Un jouet qui exerce le MÊME chemin de code : mêmes blocs, mêmes writes,
    même resume — seulement assez petit pour tourner sur CPU. `seq_len` ne bouge
    pas : il entre dans la clé du cache de chunks, et le rétrécir forcerait une
    re-tokenisation (donc du réseau)."""
    cfg["model"].update(d_model=128, n_layers=2, n_heads=2, d_head=64,
                        d_ff=128, n_experts=2, top_k_experts=1, n_groups=1,
                        mem_dim=64, max_mem=4, mem_seed_slots=2,
                        mem_read_rank=8, sinkhorn_iters=5)
    cfg["data"].update(batch_size=1, chunks_per_conv=2)
    cfg["training"].update(amp=False, tf32=False, compile=False,
                           grad_checkpoint=False, prefetch=False,
                           chunk_budget=0)


def _write_cfg(base: dict, out: Path, *, steps: int, save_dir: Path,
               metrics: Path, cpu: bool) -> Path:
    cfg = yaml.safe_load(yaml.safe_dump(base))          # copie profonde
    if cpu:
        _shrink(cfg)
    t = cfg["training"]
    t["steps"] = steps
    t["save_every"] = steps // 2                        # ⇒ un step_{N/2}.pt
    t["log_every"] = max(1, steps // 10)
    t["save_dir"] = str(save_dir)
    t["metrics_file"] = str(metrics)
    t["skip_final_eval"] = True                         # l'éval ne dit rien ici
    out.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return out


def _run(cfg_path: Path, *, resume: bool, log: Path, cpu: bool) -> None:
    cmd = [sys.executable, "-m", "deepseek_v4_mini.code_defer_native", str(cfg_path)]
    if resume:
        cmd.append("--resume")
    env = dict(os.environ)
    if cpu:
        env["CUDA_VISIBLE_DEVICES"] = ""     # ne pas empiéter sur un run en cours
        # mono-thread : les réductions BLAS multi-thread somment dans un ordre
        # qui dépend de l'ordonnancement, ce qui suffit à faire diverger deux
        # exécutions identiques d'un ULP — et un ULP par step se propage. Sans
        # ça, le test ne peut RIEN affirmer de plus fort qu'« à peu près pareil ».
        env["OMP_NUM_THREADS"] = "1"
        env["MKL_NUM_THREADS"] = "1"
    print(f"  $ {' '.join(cmd[2:])}", flush=True)
    with open(log, "w") as f:
        r = subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, env=env)
    if r.returncode != 0:
        print(Path(log).read_text()[-3000:], file=sys.stderr)
        raise SystemExit(f"le run a échoué (rc={r.returncode}) — log : {log}")


def _diff(a: Path, b: Path):
    """(n_tenseurs, liste triée de (rel, abs, nom)) entre deux `final.pt`."""
    ck_a = torch.load(a, map_location="cpu", weights_only=False)
    ck_b = torch.load(b, map_location="cpu", weights_only=False)
    assert ck_a["step"] == ck_b["step"], (ck_a["step"], ck_b["step"])

    sd_a, sd_b = ck_a["model"], ck_b["model"]
    assert set(sd_a) == set(sd_b), "les checkpoints n'ont pas les mêmes tenseurs"

    diffs = []
    for k in sd_a:
        ta, tb = sd_a[k].float(), sd_b[k].float()
        if not torch.equal(ta, tb):
            d = (ta - tb).abs().max().item()
            scale = ta.abs().max().item() or 1.0
            diffs.append((d / scale, d, k))
    diffs.sort(reverse=True)
    return len(sd_a), diffs


def _show(diffs, n, label):
    if not diffs:
        print(f"  {label}: identique ({n} tenseurs bit à bit égaux)")
        return
    print(f"  {label}: {len(diffs)}/{n} tenseurs diffèrent "
          f"(écart relatif max {diffs[0][0]:.2e}, absolu {diffs[0][1]:.2e})")
    for rel, absd, k in diffs[:8]:
        print(f"     {rel:>10.2e} rel  {absd:>10.2e} abs   {k}")
    if len(diffs) > 8:
        print(f"     … et {len(diffs) - 8} autres")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=DEFAULT_CFG)
    ap.add_argument("--steps", type=int, default=20,
                    help="total (la coupure tombe à steps/2)")
    ap.add_argument("--keep", action="store_true", help="garder le répertoire de travail")
    ap.add_argument("--cpu", action="store_true",
                    help="modèle jouet sur CPU (plus strict, et laisse le GPU libre)")
    a = ap.parse_args()

    if not a.cpu and not torch.cuda.is_available():
        print("resume_check: pas de GPU — relance avec --cpu (jouet 2 couches)")
        return 0

    base = yaml.safe_load((ROOT / a.config).read_text())
    work = Path(tempfile.mkdtemp(prefix="resume_check_"))
    half = a.steps // 2
    where = "CPU (jouet 2 couches)" if a.cpu else "GPU"
    print(f"resume_check: {a.config} | {a.steps} steps, coupure à {half} | {where}\n"
          f"  travail : {work}")

    try:
        dir_a, dir_b = work / "A", work / "B"
        cfg_a = _write_cfg(base, work / "a.yaml", steps=a.steps, save_dir=dir_a,
                           metrics=work / "a.jsonl", cpu=a.cpu)
        print("\n[A] run complet")
        _run(cfg_a, resume=False, log=work / "a.log", cpu=a.cpu)

        mid = dir_a / f"step_{half}.pt"
        if not mid.exists():
            raise SystemExit(f"pas de checkpoint intermédiaire {mid} — "
                             f"save_every n'a pas produit step_{half}.pt")
        dir_b.mkdir(parents=True, exist_ok=True)
        shutil.copy2(mid, dir_b / mid.name)             # B repart d'où A en était
        bank = dir_a / f"bank_step_{half}.pt"           # …banque comprise, quand
        if bank.exists():                               # le run en écrit une
            shutil.copy2(bank, dir_b / bank.name)
        else:
            print(f"  (pas de {bank.name} : ce run ne persiste pas la banque — "
                  f"le carry inter-step repart donc de zéro en B)")

        cfg_b = _write_cfg(base, work / "b.yaml", steps=a.steps, save_dir=dir_b,
                           metrics=work / "b.jsonl", cpu=a.cpu)
        print(f"\n[B] reprise depuis step_{half}.pt")
        _run(cfg_b, resume=True, log=work / "b.log", cpu=a.cpu)

        n, d_ab = _diff(dir_a / "final.pt", dir_b / "final.pt")
        print(f"\nverdict @step {a.steps}")
        _show(d_ab, n, "A vs B (reprise)")
        if not d_ab:
            print("\n✅ IDENTIQUE — le resume reconstruit exactement la "
                  "trajectoire : poids, optimiseur et RNG sauvés suffisent.")
            return 0

        # Un écart n'est interprétable que contre un PLANCHER DE BRUIT : sans
        # contrôle, on ne sait pas distinguer « le resume perd un état » de
        # « deux exécutions du même code ne donnent déjà pas la même chose »
        # (réductions multi-thread côté CPU, choix d'algo cuBLAS côté GPU).
        dir_c = work / "C"
        cfg_c = _write_cfg(base, work / "c.yaml", steps=a.steps, save_dir=dir_c,
                           metrics=work / "c.jsonl", cpu=a.cpu)
        print("\n[C] contrôle : le MÊME run que A, refait d'un trait")
        _run(cfg_c, resume=False, log=work / "c.log", cpu=a.cpu)
        _, d_ac = _diff(dir_a / "final.pt", dir_c / "final.pt")
        _show(d_ac, n, "A vs C (contrôle, sans reprise)")

        floor = d_ac[0][0] if d_ac else 0.0
        if d_ac and d_ab[0][0] <= max(floor * 10, 1e-6):
            print("\n⚠️  BRUIT — la reprise ne diverge pas plus que deux "
                  "exécutions identiques du même run. Le déterminisme bit à bit "
                  "n'est pas atteignable ici, mais aucun état ne manque : le "
                  "resume est sain.")
            return 0
        print("\n❌ ÉTAT MANQUANT — la reprise diverge NETTEMENT plus que le "
              "contrôle : un morceau d'état n'est pas sauvé (ordre des données, "
              "carry de banque, RNG d'un stream…).")
        return 1
    finally:
        if a.keep:
            print(f"\n(répertoire conservé : {work})")
        else:
            shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
