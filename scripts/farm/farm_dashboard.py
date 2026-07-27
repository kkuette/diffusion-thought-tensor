#!/usr/bin/env python3
"""Dashboard central de la ferme — stdlib uniquement, à lancer sur une machine
qui monte le share NFS (VM data par défaut). Agrège :
  - $TB_MNT/status/*.json   (écrits par node_agent.sh sur chaque rig)
  - $TB_MNT/queue/          (file, running, done, failed)
  - $TB_MNT/runs/*.workerlog (dernier step + dernière éval par job actif)
  - $TB_MNT/rl/*/           (runs RL désagrégés, via rl_status.py)
Le front (preact/htm, vendoré) vit dans web/ à côté de ce fichier et est servi
tel quel : changer le CSS ou le JS ne demande qu'un `git pull`, sans redémarrer
le service.
Usage : farm_dashboard.py [port] [tb_mnt]   (défauts : 8787, /mnt/tb)
"""
import json, os, re, sys, time
from http.server import HTTPServer, BaseHTTPRequestHandler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rl_status  # noqa: E402  (même dossier ; stdlib pur, pas de torch)

_ARGS = [a for a in sys.argv[1:] if not a.startswith("-")]
TB = _ARGS[1] if len(_ARGS) > 1 else os.environ.get("TB_MNT", "/mnt/tb")
PORT = int(_ARGS[0]) if _ARGS else 8787
STALE_S = 45   # agent muet depuis > STALE_S => nœud offline
JOB_STALE_S = 300  # défaut pour un job de pretraining (voir job_info)
CACHE_S = 3    # le serveur est mono-thread et snapshot() retaille tous les logs

WEB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")
VENDOR = "htm-preact-standalone-3.1.1.mjs"
# Épinglé : voir web/vendor/README.md. Le self-test revérifie cette empreinte,
# donc une substitution ou une édition accidentelle casse la CI au lieu de
# partir en production.
VENDOR_SHA256 = "72284e8e9079c87817145df1110f74e8a2aa040b2fc384922e18dfcb46fc1fd7"

# Table BLANCHE explicite : aucun chemin fourni par le client n'est joint à WEB,
# donc pas de traversée possible (le serveur écoute sur 0.0.0.0 du LAN).
STATIC = {
    "dashboard.css": (os.path.join(WEB, "dashboard.css"), "text/css; charset=utf-8"),
    "dashboard.mjs": (os.path.join(WEB, "dashboard.mjs"),
                      "text/javascript; charset=utf-8"),
    VENDOR: (os.path.join(WEB, "vendor", VENDOR), "text/javascript; charset=utf-8"),
}
INDEX = os.path.join(WEB, "index.html")

# ic/ppl optionnels : les runs SFT persona loggent « chat X  dist Y » sans ic
# quand aucun step ic n'est passé — seuls step + s/step sont garantis.
# (Ce correctif n'a vécu que dans la copie /usr/local/bin de la VM pendant six
# jours, sans commit : il est ici pour ne plus disparaître au prochain déploi.)
STEP_RE = re.compile(r"^step\s+(\d+)\s+.*?([\d.]+)s/step")
IC_RE = re.compile(r"\bic ([\d.]+) \(ppl ([\d.]+)\)")
CHAT_RE = re.compile(r"\bchat ([\d.]+)")
# Le tag de source n'est émis que par les runs MULTI-sources (code_defer_native
# ne met [src] que si src_name est vrai) : rendu obligatoire, il faisait
# disparaître tout GAP d'un run mono-source.
EVAL_RE = re.compile(r"^\[eval @(\d+)\](?:\s+\[(\w+)\])?.*GAP ([+\-][\d.]+)")
# Le learner RL imprime une autre grammaire — utile s'il passe un jour par la
# file ; aujourd'hui il tourne à la main et c'est la section RL qui le montre.
RL_STEP_RE = re.compile(r"^step\s+(\d+)\s+r ([+\-][\d.]+)\s+ce ([\d.]+).*?([\d.]+)s/step")
RL_W_RE = re.compile(r"^worker (\d+): ")
RL_DONE_RE = re.compile(r"^worker (\d+): done \((\d+) groups\)")


def tail(path, nbytes=48000):
    # 48 Ko : un bloc d'éval multi-domaine (v2e_divmix = 13 sources × 2 lignes
    # ~250 chars) fait ~6,5 Ko à lui seul — l'ancien tail de 6 Ko tronquait
    # silencieusement les premières sources du bloc.
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - nbytes))
            return f.read().decode(errors="replace").splitlines()
    except OSError:
        return []


def job_info(job_path, rl_runs=()):
    name = os.path.basename(job_path).replace(".job", "")
    log = os.path.join(TB, "runs", name + ".workerlog")
    info = {"job": name, "step": None, "evals": [], "kind": "train",
            "stale_after": JOB_STALE_S}
    last_evals = {}
    for line in tail(log):
        m = STEP_RE.match(line)
        if m:
            mi, mc = IC_RE.search(line), CHAT_RE.search(line)
            ic = f"{float(mi.group(1)):.3f}" if mi else (
                f"chat {float(mc.group(1)):.3f}" if mc else "")
            info["step"] = {"n": int(m.group(1)), "ic": ic,
                            "ppl": float(mi.group(2)) if mi else None,
                            "sps": float(m.group(2))}
        # après STEP_RE, qui est assez lâche pour attraper aussi la ligne du
        # learner RL : la branche RL doit gagner, donc elle passe en dernier.
        m = RL_STEP_RE.match(line)
        if m:
            info["kind"] = "rl_learner"
            info["step"] = {"n": int(m.group(1)), "r": float(m.group(2)),
                            "ce": float(m.group(3)), "sps": float(m.group(4))}
        m = EVAL_RE.match(line)
        if m:
            src = m.group(2) or "—"
            last_evals[src] = {"at": int(m.group(1)), "src": src,
                               "gap": float(m.group(3))}
        m = RL_W_RE.match(line)
        if m:
            info["kind"] = "rl_worker"
            info["rl_wid"] = int(m.group(1))
        m = RL_DONE_RE.match(line)
        if m:
            info["kind"] = "rl_done"
            info["rl_groups"] = int(m.group(2))
    info["evals"] = list(last_evals.values())
    try:
        info["log_age_s"] = int(time.time() - os.path.getmtime(log))
    except OSError:
        info["log_age_s"] = None
    # Un worker RL n'écrit sur stdout que sur groupe dégénéré : son .workerlog
    # est muet par construction et le seuil de 300 s le peignait en rouge en
    # permanence. Sa vraie horloge est son JSONL, dans la section RL — et le
    # seuil vient de SON débit (un groupe = ~60 s ici, un step learner ~13 min).
    if info["kind"] == "rl_worker":
        w = _rl_worker(rl_runs, info.get("rl_wid"))
        info["stale_after"] = rl_status.alive_after(w["s_per_group"] if w else None)
        if w:
            info["rl"] = {"n": w["n"], "lag": w["lag"], "env": w["env"],
                          "reward": w["reward"], "age_s": w["age_s"]}
    elif info["kind"] in ("rl_done", "rl_learner"):
        info["stale_after"] = None      # terminé : le silence est la normale
    return info


def _rl_worker(rl_runs, wid):
    """L'entrée worker la plus fraîche portant cet id, tous runs confondus.

    L'id vient des chiffres finaux de $WORKER (hostname-gpuN) : deux rigs
    entreraient en collision, mais la ferme n'en a qu'un et le run le plus
    récent gagne.
    """
    cands = [w for r in rl_runs for w in r["workers"]
             if w["wid"] == wid and w["age_s"] is not None]
    return min(cands, key=lambda w: w["age_s"]) if cands else None


def snapshot():
    q = os.path.join(TB, "queue")
    ls = lambda d: sorted(
        f for f in (os.listdir(os.path.join(q, d)) if os.path.isdir(os.path.join(q, d)) else [])
        if f.endswith(".job"))
    nodes = []
    sdir = os.path.join(TB, "status")
    if os.path.isdir(sdir):
        for f in sorted(os.listdir(sdir)):
            if not f.endswith(".json"):
                continue
            try:
                n = json.load(open(os.path.join(sdir, f)))
                n["offline"] = (time.time() - n.get("ts", 0)) > STALE_S
                nodes.append(n)
            except (json.JSONDecodeError, OSError):
                pass
    rl = rl_status.all_runs(TB)
    return {
        "ts": int(time.time()),
        "nodes": nodes,
        "rl": rl,
        "queued": [f for f in sorted(os.listdir(q)) if f.endswith(".job")] if os.path.isdir(q) else [],
        "running": [job_info(j, rl) for j in ls("running")],
        "done": ls("done"),
        "failed": ls("failed"),
    }


_cache = {"ts": 0.0, "body": b""}


def cached_body():
    """snapshot() retaille tous les logs à chaque requête, sur un serveur
    mono-thread : deux onglets ouverts suffisent à le sérialiser. 3 s de cache
    ne coûtent rien (le front rafraîchit toutes les 10 s) et bornent le coût."""
    now = time.time()
    if now - _cache["ts"] > CACHE_S:
        _cache["body"] = json.dumps(snapshot()).encode()
        _cache["ts"] = now
    return _cache["body"]


def serve(path):
    """Route une requête GET → (code, corps, type, cache).

    Séparée du handler HTTP pour être testable sans ouvrir de socket.
    Un fichier statique manquant rend 500 avec le chemin attendu : « oublié au
    déploiement » doit se lire, pas donner une page blanche.
    """
    path = path.split("?", 1)[0]
    if path.startswith("/data.json"):
        return 200, cached_body(), "application/json", "no-cache"
    if path in ("/", "/index.html"):
        target, ctype, cache = INDEX, "text/html; charset=utf-8", "no-cache"
    elif path.startswith("/static/"):
        hit = STATIC.get(path[len("/static/"):])
        if hit is None:
            return 404, b"404", "text/plain; charset=utf-8", "no-cache"
        target, ctype = hit
        # Le vendor porte sa version dans son nom : immuable, donc cachable
        # pour toujours. Le CSS et le JS de l'app, eux, changent au git pull.
        cache = "max-age=31536000, immutable" if path.endswith(VENDOR) else "no-cache"
    else:
        return 404, b"404", "text/plain; charset=utf-8", "no-cache"
    try:
        with open(target, "rb") as fh:
            return 200, fh.read(), ctype, cache
    except OSError as e:
        msg = f"fichier du front introuvable : {target}\n{e}\n"
        return 500, msg.encode(), "text/plain; charset=utf-8", "no-cache"


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):  # silence
        pass

    def do_GET(self):
        code, body, ctype, cache = serve(self.path)
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", cache)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def _self_test():
    """Hermétique : un faux share en tmpdir, aucune requête réseau."""
    import shutil, tempfile
    tmp = tempfile.mkdtemp(prefix="farm_dash_")
    globals()["TB"] = tmp
    try:
        os.makedirs(os.path.join(tmp, "runs"))
        os.makedirs(os.path.join(tmp, "queue", "running"))
        os.makedirs(os.path.join(tmp, "status"))
        rl = os.path.join(tmp, "rl", "run1")
        os.makedirs(rl)
        w = lambda p, s: open(p, "w").write(s)
        job = lambda n: w(os.path.join(tmp, "queue", "running", n + ".job"), "#")

        # 1. pretraining multi-sources ET mono-source : le tag [src] est
        # facultatif côté trainer, il doit l'être côté regex.
        job("h-gpu0__10_pre")
        w(os.path.join(tmp, "runs", "h-gpu0__10_pre.workerlog"),
          "step  100 ic 3.210 (ppl 24.8) truc 1.50s/step\n"
          "[eval @100] [py] blah GAP +0.31 fin\n"
          "[eval @100] [md] blah GAP -0.02 fin\n")
        job("h-gpu1__10_mono")
        w(os.path.join(tmp, "runs", "h-gpu1__10_mono.workerlog"),
          "step  200 ic 3.000 (ppl 20.1) truc 2.00s/step\n"
          "[eval @200] ic_ppl 20.1 GAP +0.12 fin\n")

        # 2. worker RL vivant : stdout quasi muet, métriques dans le JSONL
        job("h-gpu2__20_rl_a")
        w(os.path.join(tmp, "runs", "h-gpu2__20_rl_a.workerlog"),
          "worker 2: weights step 4\nsurprisal SIF: table unigram\n")
        w(os.path.join(rl, "worker02_metrics.jsonl"),
          "".join(json.dumps({"env": "exec", "reward": 0.5, "ce": 8.0,
                              "writes": 4, "turns": 40, "n": n, "wstep": 4,
                              "s_per_group": 60.0, "t": 1000.0 + 60 * n}) + "\n"
                  for n in (1, 2, 3)))
        w(os.path.join(rl, "learner_metrics.jsonl"),
          json.dumps({"step": 5, "reward": -1.0, "ce": 8.0, "write_rate": 0.4,
                      "kl": 1e-6, "s_per_step": 100.0}) + "\n")

        # 3. worker RL terminé proprement : son silence n'est pas une alerte
        job("h-gpu3__20_rl_b")
        w(os.path.join(tmp, "runs", "h-gpu3__20_rl_b.workerlog"),
          "worker 3: weights step 4\nworker 3: done (12 groups)\n")

        snap = snapshot()
        by = {j["job"]: j for j in snap["running"]}
        assert len(by) == 4 and len(snap["rl"]) == 1

        # 4. run SFT persona : pas d'ic, une perte « chat » à la place
        job("h-gpu4__10_persona")
        w(os.path.join(tmp, "runs", "h-gpu4__10_persona.workerlog"),
          "step  300  chat 2.100  dist 0.030  3.00s/step\n")

        snap = snapshot()
        by = {j["job"]: j for j in snap["running"]}
        p = by["h-gpu4__10_persona"]
        assert p["step"] == {"n": 300, "ic": "chat 2.100", "ppl": None,
                             "sps": 3.0}, p["step"]

        p = by["h-gpu0__10_pre"]
        assert p["kind"] == "train" and p["step"]["n"] == 100
        assert p["step"]["ic"] == "3.210" and p["step"]["ppl"] == 24.8
        assert {e["src"] for e in p["evals"]} == {"py", "md"}
        assert p["stale_after"] == JOB_STALE_S
        # mono-source : GAP visible, source affichée « — » au lieu de rien
        m = by["h-gpu1__10_mono"]
        assert [e["src"] for e in m["evals"]] == ["—"], m["evals"]
        assert m["evals"][0]["gap"] == 0.12

        # le worker RL n'affiche plus « init… » et hérite d'un seuil de silence
        # tiré de SON débit (3 × 60 s), pas des 300 s du trainer
        a = by["h-gpu2__20_rl_a"]
        assert a["kind"] == "rl_worker" and a["rl_wid"] == 2
        assert a["stale_after"] == max(rl_status.STALE_FLOOR_S, 180.0)
        assert a["rl"]["n"] == 3 and a["rl"]["lag"] == 1, a["rl"]

        b = by["h-gpu3__20_rl_b"]
        assert b["kind"] == "rl_done" and b["rl_groups"] == 12
        assert b["stale_after"] is None

        # ligne stdout du learner RL (s'il passait un jour par la file), avec
        # et sans p(w) renseigné — le tiret est le cas d'un groupe sans tour
        for pw in ("0.47", "—"):
            m = RL_STEP_RE.match(
                f"step    7  r -6.048  ce 8.976  p(w) {pw}  write% 0.47  "
                f"kl 6.68e-07  groups 3  lag 0.5  stale 0  {{'sota': 2}}  "
                f"764.8s/step")
            assert m and m.group(1) == "7" and m.group(4) == "764.8", pw

        # le cache sert le même corps dans la fenêtre, et c'est du JSON valide
        body = cached_body()
        assert json.loads(body)["rl"][0]["run"] == "run1"
        assert cached_body() is body

        # ── le front ────────────────────────────────────────────────────────
        # Contrat front/back : le JS lit ces clés de premier niveau. Un
        # renommage côté serveur doit casser ICI, pas vider une carte en
        # silence sur la VM.
        assert {"ts", "nodes", "rl", "running", "queued", "done",
                "failed"} <= set(snapshot())

        # Les routes servent de vrais fichiers : « oublié au déploiement » est
        # le mode de panne qui donnait une page blanche.
        for path, ctype in (("/", "text/html; charset=utf-8"),
                            ("/index.html", "text/html; charset=utf-8"),
                            ("/static/dashboard.css", "text/css; charset=utf-8"),
                            ("/static/dashboard.mjs", "text/javascript; charset=utf-8"),
                            (f"/static/{VENDOR}", "text/javascript; charset=utf-8")):
            code, b, ct, cache = serve(path)
            assert code == 200 and b and ct == ctype, (path, code, ct)
        # le vendor est immuable (son nom porte sa version), l'app ne l'est pas
        assert "immutable" in serve(f"/static/{VENDOR}")[3]
        assert serve("/static/dashboard.mjs")[3] == "no-cache"

        # table blanche : rien d'autre ne sort, traversée comprise
        for bad in ("/static/../farm_dashboard.py", "/static/vendor/x.mjs",
                    "/etc/passwd", "/static/", "/nimportequoi"):
            assert serve(bad)[0] == 404, bad
        # …et le corps servi est bien celui du disque, pas une page de secours
        assert b"preact" in serve("/static/dashboard.mjs")[1]
        assert b"<div id=\"app\">" in serve("/")[1]

        # empreinte du paquet vendoré : une substitution casse la CI
        import hashlib
        got = hashlib.sha256(serve(f"/static/{VENDOR}")[1]).hexdigest()
        assert got == VENDOR_SHA256, f"{VENDOR} altéré : {got}"

        print("farm_dashboard: OK (jobs RL, GAP mono-source, seuils, cache, "
              "routes du front, empreinte du vendor)")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _self_test()
    else:
        print(f"dashboard sur 0.0.0.0:{PORT}, TB_MNT={TB}")
        HTTPServer(("0.0.0.0", PORT), H).serve_forever()
