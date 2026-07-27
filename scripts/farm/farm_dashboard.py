#!/usr/bin/env python3
"""Dashboard central de la ferme — stdlib uniquement, à lancer sur une machine
qui monte le share NFS (VM data par défaut). Agrège :
  - $TB_MNT/status/*.json   (écrits par node_agent.sh sur chaque rig)
  - $TB_MNT/queue/          (file, running, done, failed)
  - $TB_MNT/runs/*.workerlog (dernier step + dernière éval par job actif)
  - $TB_MNT/rl/*/           (runs RL désagrégés, via rl_status.py)
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


PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>ferme tb</title>
<style>
 body{background:#101418;color:#cdd6e0;font:14px/1.5 monospace;margin:20px;max-width:1100px}
 h1{font-size:18px;color:#7ec8a8} h2{font-size:15px;color:#8fb4d8;margin:18px 0 6px}
 table{border-collapse:collapse;width:100%} td,th{padding:3px 10px;text-align:left;border-bottom:1px solid #232a31}
 th{color:#67707a;font-weight:normal} .ok{color:#7ec8a8} .warn{color:#e0b060} .bad{color:#e07070}
 .dim{color:#5a636d} .bar{display:inline-block;height:9px;background:#2e6e54;vertical-align:middle;border-radius:2px}
 .barbox{display:inline-block;width:90px;height:9px;background:#20262c;border-radius:2px;margin-right:6px}
 .chip{display:inline-block;background:#1a2027;border-radius:3px;padding:0 6px;margin:1px 4px 1px 0;white-space:nowrap}
 .stat{display:inline-block;margin:0 16px 5px 0;white-space:nowrap} .k{color:#67707a}
 .spk{vertical-align:middle;margin-left:5px}
 h3{font-size:14px;color:#8fb4d8;margin:14px 0 4px;font-weight:normal}
</style></head><body>
<h1>ferme thought-bank</h1><div id="c">chargement…</div>
<script>
const pct=(a,b)=>b?Math.round(100*a/b):0;
const bar=(a,b)=>`<span class="barbox"><span class="bar" style="width:${pct(a,b)*0.9}px"></span></span>${pct(a,b)}%`;
const nf=(x,d=3)=>(x==null||isNaN(x))?'<span class=dim>—</span>':(Math.abs(x)<1e-3&&x!=0?x.toExponential(1):x.toFixed(d));
const dur=s=>s==null?'—':s<90?`${Math.round(s)} s`:s<5400?`${Math.round(s/60)} min`:`${Math.floor(s/3600)} h${String(Math.floor(s%3600/60)).padStart(2,'0')}`;
// flèche de tendance = fenêtre courante vs précédente. Le reward RL est bruité :
// le dernier point ne dit pas le SENS, et c'est le sens qui décide.
const tr=t=>t==null?'':t>0?' <span class=ok>↑</span>':t<0?' <span class=bad>↓</span>':'';
function spark(a,col,w=64,h=14){
 if(!a||a.length<2) return '';
 const mn=Math.min(...a),mx=Math.max(...a),r=(mx-mn)||1;
 const p=a.map((v,i)=>`${(1+i*(w-2)/(a.length-1)).toFixed(1)},${(h-1-(v-mn)/r*(h-2)).toFixed(1)}`).join(' ');
 return `<svg class=spk width=${w} height=${h} viewBox="0 0 ${w} ${h}"><polyline points="${p}" fill=none stroke="${col}" stroke-width=1/></svg>`;}
const stat=(k,v,t,sp)=>`<span class=stat><span class=k>${k}</span> ${v}${tr(t)}${sp||''}</span>`;
async function r(){
 let d;try{d=await(await fetch('data.json')).json()}catch(e){document.getElementById('c').innerHTML='<span class=bad>dashboard injoignable</span>';return}
 let h='';
 for(const n of d.nodes){
  h+=`<h2>${n.host} ${n.offline?'<span class=bad>OFFLINE</span>':'<span class=ok>en ligne</span>'}
      <span class=dim>load ${n.load} · RAM ${n.mem_mb[0]}/${n.mem_mb[1]} Mo · swap ${n.swap_mb[0]}/${n.swap_mb[1]} Mo</span></h2>`;
  h+='<table><tr><th>gpu</th><th>util</th><th>vram</th><th>W</th><th>°C</th></tr>';
  for(const g of n.gpus||[]) h+=`<tr><td>${g.i}</td><td>${bar(g.util,100)}</td><td>${bar(g.vram,g.vram_tot)} <span class=dim>${g.vram} Mo</span></td><td>${g.w}</td><td class="${g.temp>80?'bad':g.temp>70?'warn':''}">${g.temp}</td></tr>`;
  h+='</table>';}
 for(const R of d.rl||[]){
  const L=R.learner, cls=R.state=='ok'?'ok':R.state=='stop'?'dim':R.state=='sec'?'warn':'bad';
  // « à sec » = le learner attend, aucun rollout ne rentre : regarder les
  // workers. « muet » = il a des groupes en attente et ne les prend pas.
  const lbl={ok:'en cours',stop:'STOP',sec:'à sec — aucun rollout'}[R.state]||'MUET';
  let hd=`<h2>RL ${R.run} <span class=${cls}>[${lbl}]</span>`;
  if(L&&L.step!=null) hd+=` <span class=dim>step ${L.step}${R.steps?'/'+R.steps+' '+bar(L.step,R.steps):''}${R.eta_s?' · eta '+dur(R.eta_s):''}</span>`;
  h+=hd+'</h2>';
  h+='<div>'+stat('poids',R.weights.step==null?'<span class=dim>—</span>':'step '+R.weights.step)
   +stat('file',`${R.queue.incoming}`+(R.queue.stale?` <span class=warn>+${R.queue.stale} stale</span>`:''))
   +(R.queue.traces_mb!=null?stat('traces',R.queue.traces_mb+' Mo'):'')
   +stat('learner vu il y a',`<span class="${L&&L.alive?'ok':'bad'}">${L?dur(L.age_s):'—'}</span>`)+'</div>';
  // Un run arrêté reste sur le share pour toujours : réduit à son en-tête, il
  // ne pousse pas le run vivant hors de l'écran. Le détail reste en CLI.
  if(R.state=='stop'){h+=`<div class=dim>run terminé — détail : rl_status.py ${R.run}</div>`;continue;}
  if(L){const w=L.win,t=L.trend,S=L.spark;
   h+='<div>'+stat('r',nf(w.reward),t.reward,spark(S.reward,'#7ec8a8'))
    +stat('ce',nf(w.ce),t.ce)
    +stat('p(w)',nf(w.p_write,2))
    +stat('write%',nf(w.write_rate,2),t.write_rate,spark(S.write_rate,'#8fb4d8'))
    +stat('kl',nf(w.kl),t.kl,spark(S.kl,'#e0b060'))
    +stat('poscorr',nf(w.pos_corr,2))+stat('lag',nf(w.lag,1))
    +stat('stale',L.stale==null?'<span class=dim>—</span>':L.stale)
    +stat('s/step',dur(L.s_per_step),null,spark(S.sps,'#67707a'))+'</div>';}
  const envs=Object.entries(R.envs||{});
  if(envs.length){
   h+='<h3>par environnement</h3><table><tr><th>env</th><th>groupes</th><th>reward</th><th>grade</th><th>write%</th><th>p(w)</th></tr>';
   for(const [e,v] of envs) h+=`<tr><td>${e}</td><td>${v.n}</td><td>${nf(v.reward)}</td><td>${nf(v.grade,2)}</td><td>${nf(v.write_rate,2)}</td><td>${nf(v.p_write,2)}</td></tr>`;
   // Colonnes NON comparables entre elles : les envs denses rendent -ce (~-8),
   // les envs à rubrique rendent [0,1]. C'est pour ça que le reward agrégé du
   // learner ne veut rien dire — il bouge avec le mix, pas avec la politique.
   h+='</table><div class=dim>reward dense (-ce) pour code/sota, rubrique [0,1] pour tools/exec — ne pas comparer les lignes entre elles. grade = succès brut avant économie de think.</div>';}
  if(R.workers.length){
   h+='<h3>workers</h3><table><tr><th>worker</th><th>groupes</th><th>lag</th><th>s/groupe</th><th>dégén.</th><th>dernier env</th><th>reward</th><th>vu il y a</th></tr>';
   for(const k of R.workers) h+=`<tr><td>w${String(k.wid).padStart(2,'0')}</td><td>${k.n??'—'}</td>`
    +`<td class="${(k.lag!=null&&R.meta.max_lag!=null&&k.lag>R.meta.max_lag)?'bad':''}">${k.lag??'—'}</td>`
    +`<td>${dur(k.s_per_group)}</td><td>${k.degen??'—'}</td><td>${k.env||'—'}</td><td>${nf(k.reward)}</td>`
    +`<td class="${k.alive?'':'warn'}">${dur(k.age_s)}</td></tr>`;
   h+='</table>';}
  const x=R.xdom;
  if(x) h+=`<div><span class=k>sonde xdom @${x.n}</span> `
   +`<span class="chip ${x.r_own>x.r_xdom?'ok':'bad'}">own ${nf(x.r_own)} vs xdom ${nf(x.r_xdom)}</span>`
   +`<span class=chip>always ${nf(x.r_always)}</span><span class=chip>never ${nf(x.r_never)}</span></div>`;
 }
 h+=`<h2>jobs actifs (${d.running.length})</h2><table><tr><th>job</th><th>step / état</th><th>ic · détail</th><th>s/step</th><th>dernier GAP par domaine</th><th>log il y a</th></tr>`;
 for(const j of d.running){
  // GAP par domaine en chips (mix divmix = 13 sources) : vert/rouge par signe,
  // @step affiché seulement si la source est en retard sur l'éval la plus récente.
  const s=j.step, at=(j.evals&&j.evals.length)?Math.max(...j.evals.map(x=>x.at)):0;
  const e=(j.evals||[]).map(x=>`<span class="chip ${x.gap>0?'ok':'bad'}">${x.src} <b>${x.gap>0?'+':''}${x.gap.toFixed(2)}</b>${x.at<at?` <span class=dim>@${x.at}</span>`:''}</span>`).join('');
  // seuil de silence fourni par le serveur : 300 s pour un trainer bavard,
  // 3×s/groupe pour un worker RL (muet par construction), aucun pour un job fini.
  const lim=j.stale_after;
  const age=j.log_age_s==null?'?':((lim&&j.log_age_s>lim)?`<span class=bad>${dur(j.log_age_s)}</span>`:dur(j.log_age_s));
  let c2,c3,c4='';
  if(j.kind=='rl_worker'){c2=`<span class=ok>RL w${String(j.rl_wid).padStart(2,'0')}</span>`;
   c3=j.rl?`${j.rl.n} groupes · lag ${j.rl.lag??'—'}`:'<span class=dim>pas de métriques</span>';}
  else if(j.kind=='rl_done'){c2='<span class=dim>terminé</span>';c3=`${j.rl_groups} groupes`;}
  else if(j.kind=='rl_learner'&&s){c2=s.n;c3=`r ${nf(s.r)} · ce ${nf(s.ce)}`;c4=s.sps;}
  else {c2=s?s.n:'<span class=dim>init…</span>';c3=s?s.ic:'';c4=s?s.sps:'';}
  h+=`<tr><td>${j.job}</td><td>${c2}</td><td>${c3}</td><td>${c4}</td><td>${e?`<span class=dim>@${at}</span> ${e}`:'<span class=dim>—</span>'}</td><td>${age}</td></tr>`;}
 h+='</table>';
 h+=`<h2>file (${d.queued.length})</h2><div class=dim>${d.queued.join('<br>')||'vide'}</div>`;
 h+=`<h2>terminés (${d.done.length}) — <span class=bad>échecs (${d.failed.length})</span></h2>`;
 h+=`<div class=dim>${d.done.join('<br>')||'—'}</div>`;
 if(d.failed.length) h+=`<div class=bad>${d.failed.join('<br>')}</div>`;
 h+=`<p class=dim>maj ${new Date(d.ts*1000).toLocaleTimeString()}</p>`;
 document.getElementById('c').innerHTML=h;}
r();setInterval(r,10000);
</script></body></html>"""


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):  # silence
        pass

    def do_GET(self):
        if self.path.startswith("/data.json"):
            body, ctype = cached_body(), "application/json"
        else:
            body, ctype = PAGE.encode(), "text/html; charset=utf-8"
        self.send_response(200)
        self.send_header("Content-Type", ctype)
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
        print("farm_dashboard: OK (jobs RL, GAP mono-source, seuils, cache)")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _self_test()
    else:
        print(f"dashboard sur 0.0.0.0:{PORT}, TB_MNT={TB}")
        HTTPServer(("0.0.0.0", PORT), H).serve_forever()
