#!/usr/bin/env python3
"""État d'un run RL désagrégé, lu depuis le share — stdlib uniquement.

Le dashboard de la ferme lit le STDOUT des jobs ; un run RL n'y met presque
rien (le learner n'est même pas un job de la ferme, et un worker n'imprime que
sur groupe dégénéré). Tout le signal est dans les JSONL que rl_disagg écrit :

  <root>/learner_metrics.jsonl     une ligne par step d'optimisation
  <root>/worker{NN}_metrics.jsonl  une ligne par groupe commité (+ sondes xdom)
  <root>/meta.json                 écrit à l'init du learner (steps, tailles)
  <root>/weights/LATEST            pointeur vers les poids publiés
  <root>/rollouts/{incoming,stale} la file de groupes
  <root>/STOP                      présence = arrêt demandé

Ce module est le SEUL endroit qui connaît ce format : farm_dashboard l'importe,
et `rl_status.py --tb /mnt/tb` en donne le même résumé en terminal. Stdlib pur
et aucun import du paquet : le dashboard tourne sur la VM data, sans torch.

Usage :
  rl_status.py                     # self-test hermétique (aucun accès au NAS)
  rl_status.py --tb /mnt/tb        # résumé de tous les runs
  rl_status.py --tb /mnt/tb 350m   # filtre sur le nom du run
"""
import json, os, sys, time

WINDOW = 20          # steps (learner) / groupes (worker) de la fenêtre courante
SPARK_N = 120        # points envoyés au dashboard pour les micro-courbes
STALE_FLOOR_S = 180  # plancher de vivacité : voir alive_after()


# ── lecture ──────────────────────────────────────────────────────────────────

def tail_jsonl(path, nbytes=96000):
    """Les dernières lignes JSON d'un fichier append-only.

    On lit la queue en octets (les fichiers grossissent tout le run) : la
    première ligne du bloc est donc coupée au milieu et se jette. Les lignes
    illisibles sont ignorées plutôt que fatales — le fichier est écrit pendant
    qu'on le lit, la dernière peut être incomplète.
    """
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - nbytes))
            raw = f.read().decode(errors="replace")
    except OSError:
        return []
    lines = raw.splitlines()
    if size > nbytes and lines:
        lines = lines[1:]
    out = []
    for ln in lines:
        ln = ln.strip()
        if not ln:
            continue
        try:
            rec = json.loads(ln)
        except ValueError:
            continue
        if isinstance(rec, dict):
            out.append(rec)
    return out


def _load_json(path):
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def _age(path, now):
    try:
        return int(now - os.path.getmtime(path))
    except OSError:
        return None


def _count(d, suffix=".pt"):
    try:
        return len([p for p in os.listdir(d) if p.endswith(suffix)])
    except OSError:
        return 0


# ── agrégats ─────────────────────────────────────────────────────────────────

def _mean(xs):
    xs = [x for x in xs if isinstance(x, (int, float))]
    return sum(xs) / len(xs) if xs else None


def _win(lines, key):
    return _mean([ln.get(key) for ln in lines])


def _trend(lines, key, window=WINDOW):
    """Moyenne de la fenêtre courante moins celle de la précédente.

    Le reward RL est bruité : le dernier point ne dit pas dans quel sens ça va,
    et c'est le sens qui décide de couper un run ou de le laisser tourner.
    """
    cur, prev = lines[-window:], lines[-2 * window:-window]
    a, b = _win(cur, key), _win(prev, key)
    return None if a is None or b is None else a - b


def _period(lines, cum_key, n_key):
    """Durée réelle d'un pas, reconstruite pas à pas.

    `s_per_step` / `s_per_group` sont des moyennes CUMULÉES depuis le lancement
    du processus : elles portent l'attente initiale (4661 s au step 1 du run
    350M) et lissent tout le reste. Or cumul × compteur = temps écoulé depuis
    t0, donc la différence entre deux lignes redonne la durée exacte du pas.
    La clé `t` (horodatage), quand elle est là, court-circuite le calcul ; les
    lignes écrites avant son ajout n'en ont pas.
    """
    out = []
    for a, b in zip(lines, lines[1:]):
        na, nb = a.get(n_key), b.get(n_key)
        if not isinstance(na, (int, float)) or not isinstance(nb, (int, float)):
            continue
        dn = nb - na
        if "t" in a and "t" in b:
            dt = b["t"] - a["t"]
        else:
            ca, cb = a.get(cum_key), b.get(cum_key)
            if not isinstance(ca, (int, float)) or not isinstance(cb, (int, float)):
                continue
            dt = cb * max(nb, 1) - ca * max(na, 1)
        if dt > 0 and dn > 0:
            out.append(dt / dn)
    return out


def alive_after(period_s):
    """Seuil de silence au-delà duquel on déclare un processus muet.

    Adaptatif, et c'est tout l'enjeu : le dashboard peint en rouge un job dont
    le log n'a pas bougé depuis 300 s, alors qu'un step du learner 350M dure
    ~765 s. Un seuil fixe transforme un run sain en alerte permanente — et une
    alerte permanente ne se lit plus.
    """
    if not isinstance(period_s, (int, float)) or period_s <= 0:
        return STALE_FLOOR_S * 4
    return max(STALE_FLOOR_S, 3.0 * period_s)


def _spark(lines, key, n=SPARK_N):
    return [ln[key] for ln in lines[-n:]
            if isinstance(ln.get(key), (int, float))]


# ── un run ───────────────────────────────────────────────────────────────────

def _learner(root, now):
    path = os.path.join(root, "learner_metrics.jsonl")
    if not os.path.exists(path):
        return None
    lines = tail_jsonl(path)
    win = lines[-WINDOW:]
    per = _period(lines, "s_per_step", "step")
    sps = _mean(per[-WINDOW:])
    last = lines[-1] if lines else {}
    age = _age(path, now)
    out = {
        "step": last.get("step"),
        "last": last,
        "win": {k: _win(win, k) for k in
                ("reward", "ce", "write_rate", "kl", "pos_corr", "p_write",
                 "lag", "groups")},
        "trend": {k: _trend(lines, k) for k in ("reward", "ce", "write_rate", "kl")},
        "s_per_step": sps,
        "stale": last.get("stale"),
        "pending": last.get("pending"),
        "env_mix": last.get("env_mix") or {},
        "age_s": age,
        "spark": {"reward": _spark(lines, "reward"),
                  "kl": _spark(lines, "kl"),
                  "write_rate": _spark(lines, "write_rate"),
                  "sps": per[-SPARK_N:]},
    }
    out["alive"] = age is not None and age < alive_after(sps)
    return out


def _workers(root, now, learner_step):
    out = []
    try:
        names = sorted(p for p in os.listdir(root)
                       if p.startswith("worker") and p.endswith("_metrics.jsonl"))
    except OSError:
        return out
    for name in names:
        path = os.path.join(root, name)
        try:
            wid = int(name[len("worker"):-len("_metrics.jsonl")])
        except ValueError:
            continue
        lines = tail_jsonl(path)
        groups = [ln for ln in lines if ln.get("probe") is None]
        probes = [ln for ln in lines if ln.get("probe") == "xdom"]
        last = groups[-1] if groups else {}
        per = _period(groups, "s_per_group", "n")
        spg = _mean(per[-WINDOW:])
        age = _age(path, now)
        wstep = last.get("wstep")
        lag = (learner_step - wstep
               if isinstance(wstep, int) and isinstance(learner_step, int) else None)
        out.append({
            "wid": wid,
            "n": last.get("n"),
            "wstep": wstep,
            "lag": lag,
            "env": last.get("env"),
            "reward": last.get("reward"),
            "s_per_group": spg,
            "per_hour": 3600.0 / spg if spg else None,
            "degen": last.get("degen"),
            "tries": last.get("tries"),
            "age_s": age,
            "alive": age is not None and age < alive_after(spg),
            "groups": groups,      # consommé par _envs, retiré avant sérialisation
            "probe": probes[-1] if probes else None,
        })
    return out


def _envs(workers):
    """Reward par environnement — le seul agrégat qui veuille dire quelque chose.

    Le `reward` du learner moyenne des échelles incompatibles : les envs denses
    (code, sota) rendent -ce, autour de -8 ; les envs à rubrique (tools, exec)
    rendent [0, 1]. Leur moyenne bouge quand le MIX bouge, pas quand la
    politique s'améliore. Découpé par env, chaque colonne redevient lisible.
    """
    by = {}
    for w in workers:
        for ln in w["groups"][-WINDOW:]:
            env = ln.get("env")
            if env:
                by.setdefault(env, []).append(ln)
    out = {}
    for env, lns in sorted(by.items()):
        turns = sum(ln.get("turns", 0) for ln in lns)
        out[env] = {
            "n": len(lns),
            "reward": _mean([ln.get("reward") for ln in lns]),
            "grade": _mean([ln.get("grade") for ln in lns]),
            "p_write": _mean([ln.get("p_write") for ln in lns]),
            "write_rate": (sum(ln.get("writes", 0) for ln in lns) / turns
                           if turns else None),
        }
    return out


def _weights(root, now):
    d = os.path.join(root, "weights")
    step = None
    try:
        with open(os.path.join(d, "LATEST")) as fh:
            name = fh.read().strip()
        step = int(name[len("step_"):-len(".pt")])
    except (OSError, ValueError):
        pass
    return {"step": step, "age_s": _age(os.path.join(d, "LATEST"), now)}


def run_status(root, now=None):
    """L'état complet d'un run RL, prêt à sérialiser."""
    now = time.time() if now is None else now
    meta = _load_json(os.path.join(root, "meta.json")) or {}
    learner = _learner(root, now)
    lstep = learner.get("step") if learner else None
    workers = _workers(root, now, lstep)
    envs = _envs(workers)
    probes = [w["probe"] for w in workers if w["probe"]]
    for w in workers:
        del w["groups"], w["probe"]
    steps = meta.get("steps")
    sps = learner.get("s_per_step") if learner else None
    stopped = os.path.exists(os.path.join(root, "STOP"))
    inc = _count(os.path.join(root, "rollouts", "incoming"))
    # Un learner silencieux n'est pas forcément planté : sans groupe à
    # consommer il attend, et le vrai coupable est du côté des workers. Les
    # deux cas n'appellent pas le même geste, le monitor doit les séparer.
    state = ("stop" if stopped else
             "ok" if learner and learner["alive"] else
             "sec" if inc == 0 else "muet")
    return {
        "run": os.path.basename(root.rstrip("/")),
        "root": root,
        "meta": meta,
        "steps": steps,
        "stopped": stopped,
        "state": state,
        "learner": learner,
        "workers": workers,
        "envs": envs,
        "xdom": probes[-1] if probes else None,
        "queue": {
            "incoming": inc,
            "stale": _count(os.path.join(root, "rollouts", "stale")),
            "traces_mb": round(os.path.getsize(os.path.join(root, "traces.jsonl"))
                               / 1e6, 1)
            if os.path.exists(os.path.join(root, "traces.jsonl")) else None,
        },
        "weights": _weights(root, now),
        "eta_s": (int((steps - lstep) * sps)
                  if steps and lstep and sps and lstep < steps else None),
    }


def discover(tb):
    """Les runs RL présents sur le share, plus récent en premier."""
    base = os.path.join(tb, "rl")
    try:
        names = os.listdir(base)
    except OSError:
        return []
    roots = []
    for n in sorted(names):
        r = os.path.join(base, n)
        if os.path.isdir(r) and (
                os.path.exists(os.path.join(r, "learner_metrics.jsonl"))
                or os.path.exists(os.path.join(r, "weights", "LATEST"))):
            roots.append(r)
    roots.sort(key=lambda r: os.path.getmtime(r), reverse=True)
    return roots


def all_runs(tb, now=None):
    return [run_status(r, now) for r in discover(tb)]


# ── résumé terminal ──────────────────────────────────────────────────────────

def _f(x, fmt="{:.3f}", dash="—"):
    return fmt.format(x) if isinstance(x, (int, float)) else dash


def _dur(s):
    if not isinstance(s, (int, float)):
        return "—"
    s = int(s)
    if s < 90:
        return f"{s}s"
    if s < 5400:
        return f"{s // 60}min"
    return f"{s // 3600}h{(s % 3600) // 60:02d}"


def format_run(st):
    L, out = st["learner"], []
    head = f"■ {st['run']}  [{st['state']}]"
    if L and L["step"] is not None:
        head += f"  step {L['step']}" + (f"/{st['steps']}" if st["steps"] else "")
    if st["eta_s"]:
        head += f"  eta {_dur(st['eta_s'])}"
    out.append(head)
    if L:
        w, t = L["win"], L["trend"]
        arr = lambda k: ("↑" if t.get(k) and t[k] > 0 else
                         "↓" if t.get(k) and t[k] < 0 else " ")
        out.append(f"  learner  r {_f(w['reward'], '{:+.3f}')}{arr('reward')}"
                   f"  ce {_f(w['ce'])}"
                   f"  p(w) {_f(w['p_write'], '{:.2f}')}"
                   f"  write% {_f(w['write_rate'], '{:.2f}')}{arr('write_rate')}"
                   f"  kl {_f(w['kl'], '{:.1e}')}"
                   f"  lag {_f(w['lag'], '{:.1f}')}"
                   f"  {_dur(L['s_per_step'])}/step"
                   f"  vu il y a {_dur(L['age_s'])}")
    out.append(f"  file     incoming {st['queue']['incoming']}"
               f"  stale {st['queue']['stale']}"
               f"  poids step {st['weights']['step']}"
               f" (il y a {_dur(st['weights']['age_s'])})")
    for env, e in st["envs"].items():
        out.append(f"  env {env:<6} n {e['n']:<3} r {_f(e['reward'], '{:+.3f}')}"
                   f"  grade {_f(e['grade'], '{:.2f}')}"
                   f"  write% {_f(e['write_rate'], '{:.2f}')}")
    for wk in st["workers"]:
        out.append(f"  w{wk['wid']:02d}      {'ok ' if wk['alive'] else 'MUET'}"
                   f"  groupes {wk['n']}  lag {wk['lag']}"
                   f"  {_dur(wk['s_per_group'])}/groupe"
                   f"  env {wk['env']}  r {_f(wk['reward'], '{:+.3f}')}"
                   f"  vu il y a {_dur(wk['age_s'])}")
    x = st["xdom"]
    if x:
        out.append(f"  xdom     own {_f(x.get('r_own'), '{:+.3f}')}"
                   f"  xdom {_f(x.get('r_xdom'), '{:+.3f}')}"
                   f"  always {_f(x.get('r_always'), '{:+.3f}')}"
                   f"  never {_f(x.get('r_never'), '{:+.3f}')}  @{x.get('n')}")
    return "\n".join(out)


def main(argv):
    tb = os.environ.get("TB_MNT", "/mnt/tb")
    if "--tb" in argv:
        tb = argv[argv.index("--tb") + 1]
        argv = [a for i, a in enumerate(argv)
                if i not in (argv.index("--tb"), argv.index("--tb") + 1)]
    pats = [a for a in argv if not a.startswith("-")]
    runs = [st for st in all_runs(tb)
            if not pats or any(p in st["run"] for p in pats)]
    if not runs:
        print(f"aucun run RL sous {tb}/rl")
        return 1
    print("\n\n".join(format_run(st) for st in runs))
    return 0


# ── self-test (hermétique) ───────────────────────────────────────────────────

def _self_test():
    import tempfile, shutil
    tmp = tempfile.mkdtemp(prefix="rl_status_")
    try:
        root = os.path.join(tmp, "rl", "fake")
        os.makedirs(os.path.join(root, "rollouts", "incoming"))
        os.makedirs(os.path.join(root, "rollouts", "stale"))
        os.makedirs(os.path.join(root, "weights"))
        w = lambda p, s: open(p, "w").write(s)

        # learner : 3 steps ANCIEN format (pas de `t`, pas de p_write/lag) —
        # les lignes déjà sur le share doivent rester lisibles.
        lm = os.path.join(root, "learner_metrics.jsonl")
        with open(lm, "w") as fh:
            for i, (r, sps) in enumerate([(-9.0, 100.0), (-8.0, 75.0),
                                          (-7.0, 60.0)], start=1):
                fh.write(json.dumps({"step": i, "reward": r, "ce": 8.0,
                                     "write_rate": 0.4, "kl": 1e-6,
                                     "pos_corr": 0.1, "groups": 2,
                                     "env_mix": {"code": 2}, "stale": 1,
                                     "s_per_step": sps}) + "\n")
        # cumul × step = temps écoulé : 100, 150, 180 → pas de 50 s puis 30 s
        per = _period(tail_jsonl(lm), "s_per_step", "step")
        assert per == [50.0, 30.0], per

        # workers : w00 à jour, w01 en retard d'une version de poids
        w(os.path.join(root, "worker00_metrics.jsonl"),
          "".join(json.dumps({"env": "exec", "reward": 0.5, "ce": 8.0,
                              "writes": 10, "turns": 40, "tries": 0, "n": n,
                              "wstep": 3, "s_per_group": 30.0, "grade": 0.8,
                              "p_write": 0.25, "degen": 1,
                              "t": 1000.0 + 30 * n}) + "\n"
                  for n in (1, 2, 3))
          + json.dumps({"probe": "xdom", "n": 3, "env": "exec", "r_own": 0.6,
                        "r_xdom": 0.2, "r_always": 0.4, "r_never": 0.3}) + "\n")
        w(os.path.join(root, "worker01_metrics.jsonl"),
          json.dumps({"env": "code", "reward": -8.0, "ce": 8.0, "writes": 4,
                      "turns": 40, "n": 1, "wstep": 1,
                      "s_per_group": 60.0}) + "\n")
        w(os.path.join(root, "weights", "LATEST"), "step_000003.pt")
        open(os.path.join(root, "rollouts", "incoming", "a.pt"), "w").close()
        w(os.path.join(root, "meta.json"), json.dumps({"steps": 10}))

        st = run_status(root)
        assert st["run"] == "fake" and st["state"] == "ok", st["state"]
        assert st["learner"]["step"] == 3 and st["steps"] == 10
        assert st["weights"]["step"] == 3 and st["queue"]["incoming"] == 1

        # lag = step learner − version des poids du groupe
        lag = {x["wid"]: x["lag"] for x in st["workers"]}
        assert lag == {0: 0, 1: 2}, lag

        # tendance : reward monte (fenêtre trop courte ici -> None, pas un crash)
        assert st["learner"]["trend"]["reward"] is None
        long = [{"step": i, "reward": float(i)} for i in range(1, 60)]
        assert _trend(long, "reward") == 20.0, _trend(long, "reward")

        # découpage par env : le grade rubrique survit, l'env dense n'en a pas
        assert abs(st["envs"]["exec"]["grade"] - 0.8) < 1e-9
        assert st["envs"]["code"]["grade"] is None
        assert st["envs"]["code"]["write_rate"] == 0.1
        assert st["xdom"]["r_own"] == 0.6

        # eta = pas restants × moyenne des durées RÉELLES (50, 30) = 40 s.
        # Via `s_per_step` cumulé (60 s au dernier pas) on aurait annoncé 420 s.
        assert st["eta_s"] == 7 * 40, st["eta_s"]

        # vivacité ADAPTATIVE : 700 s de silence, c'est mort pour un pas de 30 s
        # et parfaitement sain pour un pas de 765 s (le run 350M).
        assert alive_after(30.0) == STALE_FLOOR_S
        assert alive_after(765.0) > 2000
        assert alive_after(None) > 0
        # learner muet AVEC des groupes en attente = il ne mange pas ce qu'on
        # lui sert : c'est lui le problème.
        open(os.path.join(root, "rollouts", "incoming", "b.pt"), "w").close()
        old = run_status(root, now=time.time() + 700)
        assert not old["learner"]["alive"] and old["state"] == "muet"
        # file vide : il attend, le problème est chez les workers
        for f in os.listdir(os.path.join(root, "rollouts", "incoming")):
            os.remove(os.path.join(root, "rollouts", "incoming", f))
        assert run_status(root, now=time.time() + 700)["state"] == "sec"

        # STOP l'emporte sur tout le reste
        open(os.path.join(root, "STOP"), "w").close()
        assert run_status(root, now=time.time() + 700)["state"] == "stop"
        os.remove(os.path.join(root, "STOP"))

        # robustesse : ligne tronquée en tête, ligne illisible, fichier vide
        assert tail_jsonl(os.path.join(tmp, "nexistepas.jsonl")) == []
        w(os.path.join(tmp, "vide.jsonl"), "")
        assert tail_jsonl(os.path.join(tmp, "vide.jsonl")) == []
        w(os.path.join(tmp, "sale.jsonl"),
          '{"a": 1}\n{cassé\n\n{"a": 2}\n"pas un dict"\n')
        assert [r["a"] for r in tail_jsonl(os.path.join(tmp, "sale.jsonl"))] == [1, 2]
        big = os.path.join(tmp, "gros.jsonl")
        with open(big, "w") as fh:
            for i in range(400):
                fh.write(json.dumps({"i": i, "pad": "x" * 300}) + "\n")
        t = tail_jsonl(big, nbytes=4000)
        assert 5 < len(t) < 20 and t[-1]["i"] == 399, len(t)

        # découverte + un run sans aucune métrique ne fait pas tomber le lecteur
        os.makedirs(os.path.join(tmp, "rl", "muet", "weights"))
        w(os.path.join(tmp, "rl", "muet", "weights", "LATEST"), "cassé")
        runs = all_runs(tmp)
        assert {r["run"] for r in runs} == {"fake", "muet"}
        muet = [r for r in runs if r["run"] == "muet"][0]
        assert muet["learner"] is None and muet["weights"]["step"] is None
        assert muet["state"] == "sec"          # rien écrit, rien en attente

        # le formateur avale les trous sans exploser
        assert "fake" in format_run(st) and format_run(muet)
        print("rl_status: OK (lecture, lag, envs, vivacité adaptative, trous)")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        _self_test()
    else:
        sys.exit(main(sys.argv[1:]))
