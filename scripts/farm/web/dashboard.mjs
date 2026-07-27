/* Dashboard de la ferme — front preact/htm, servi par farm_dashboard.py.
 *
 * Pourquoi un moteur à diff plutôt que la chaîne innerHTML d'avant : la page se
 * rafraîchit toutes les 10 s, et un innerHTML= complet efface tout état
 * d'interface — un <details> qu'on vient d'ouvrir se referme, le scroll d'une
 * table repart en haut, la sélection de texte saute. Preact ne touche que les
 * nœuds qui changent, donc les replis et les tables défilantes tiennent.
 *
 * Aucune ressource externe : le paquet est vendoré à côté (la VM data qui sert
 * cette page n'a pas d'accès réseau).
 */
import { html, render, useState, useEffect } from './htm-preact-standalone-3.1.1.mjs';

const POLL_MS = 10000;

// ── formatage ───────────────────────────────────────────────────────────────

const pct = (a, b) => (b ? Math.round(100 * a / b) : 0);

const nf = (x, d = 3) =>
  (x == null || Number.isNaN(x)) ? html`<span class="dim">—</span>`
    : (x !== 0 && Math.abs(x) < 1e-3) ? x.toExponential(1)
      : x.toFixed(d);

const dur = s =>
  s == null ? '—'
    : s < 90 ? `${Math.round(s)} s`
      : s < 5400 ? `${Math.round(s / 60)} min`
        : `${Math.floor(s / 3600)} h${String(Math.floor(s % 3600 / 60)).padStart(2, '0')}`;

/* Flèche de tendance = fenêtre courante vs précédente. Le reward RL est bruité :
   le dernier point ne dit pas le SENS, et c'est le sens qui décide de couper un
   run ou de le laisser tourner.
   `good` porte la polarité, et elle n'est pas la même partout : un reward qui
   monte est une bonne nouvelle, une entropie croisée qui monte est une mauvaise.
   Peindre les deux en vert, c'est mentir. `none` = direction sans jugement. */
const Trend = ({ t, good = 'up' }) => {
  if (t == null || t === 0) return null;
  const up = t > 0;
  const cls = good === 'none' ? 'dim' : (up === (good === 'up')) ? 'ok' : 'bad';
  return html`<span class=${cls}> ${up ? '↑' : '↓'}</span>`;
};

const Stat = ({ k, children }) =>
  html`<span><span class="k">${k}</span> <span class="num">${children}</span></span>`;

// ── briques ─────────────────────────────────────────────────────────────────

const Bar = ({ v, max, warn = 0.75, bad = 0.9, title }) => {
  const r = max ? v / max : 0;
  const cls = r >= bad ? 'bad' : r >= warn ? 'warn' : '';
  return html`<span class=${'barbox ' + cls} title=${title || ''}>
    <i style=${`width:${Math.min(100, Math.round(r * 100))}%`}></i></span>`;
};

const Ratio = ({ v, max, unit = '', ...rest }) => html`
  <${Bar} v=${v} max=${max} ...${rest} />
  <span class="num dim">${pct(v, max)} %${unit ? ` · ${v} ${unit}` : ''}</span>`;

/* Sparkline : aire + point final + ligne de base à 0 quand la série change de
   signe (sans elle, une courbe de reward négatif ressemble à une courbe de
   reward positif). Le <title> donne min/max/dernier au survol, sans JS. */
const Spark = ({ pts, col = 'var(--accent)', w = 92, h = 26 }) => {
  if (!pts || pts.length < 2) return null;
  const mn = Math.min(...pts), mx = Math.max(...pts), r = (mx - mn) || 1;
  const x = i => 1 + i * (w - 2) / (pts.length - 1);
  const y = v => h - 2 - (v - mn) / r * (h - 4);
  const line = pts.map((v, i) => `${x(i).toFixed(1)},${y(v).toFixed(1)}`).join(' ');
  const last = pts[pts.length - 1];
  const zero = mn < 0 && mx > 0 ? y(0) : null;
  const f = v => (v !== 0 && Math.abs(v) < 1e-3) ? v.toExponential(1) : v.toFixed(3);
  return html`
    <svg class="spk" width=${w} height=${h} viewBox=${`0 0 ${w} ${h}`}
         preserveAspectRatio="none" role="img">
      <title>${`min ${f(mn)} · max ${f(mx)} · dernier ${f(last)} · ${pts.length} pts`}</title>
      ${zero != null && html`<line x1="0" x2=${w} y1=${zero} y2=${zero}
        stroke="var(--dim)" stroke-width="1" stroke-dasharray="2 2" opacity="0.5"/>`}
      <polygon points=${`${x(0).toFixed(1)},${h} ${line} ${x(pts.length - 1).toFixed(1)},${h}`}
        fill=${col} opacity="0.12"/>
      <polyline points=${line} fill="none" stroke=${col} stroke-width="1.2"
        stroke-linejoin="round"/>
      <circle cx=${x(pts.length - 1).toFixed(1)} cy=${y(last).toFixed(1)} r="1.9" fill=${col}/>
    </svg>`;
};

const Tile = ({ k, v, trend, good, pts, col, cls = '' }) => html`
  <div class=${'tile ' + cls}>
    <span class="k">${k}</span>
    <b>${v}<${Trend} t=${trend} good=${good} /></b>
    <${Spark} pts=${pts} col=${col} />
  </div>`;

// ── nœuds ───────────────────────────────────────────────────────────────────

const NodeCard = ({ n }) => html`
  <section class=${'card ' + (n.offline ? 's-bad' : 's-ok')}>
    <h2><span class=${'dot ' + (n.offline ? 'bad' : 'ok')}></span>${n.host}
      <span class=${'badge ' + (n.offline ? 'bad' : 'ok')}>
        ${n.offline ? 'hors ligne' : 'en ligne'}</span>
      <span class="sp k">load <span class="num">${n.load}</span></span>
    </h2>
    <div class="scroll"><table>
      <thead><tr><th class="n">gpu</th><th>util</th><th>vram</th>
        <th class="n">W</th><th class="n">°C</th></tr></thead>
      <tbody>${(n.gpus || []).map(g => html`
        <tr key=${g.i}>
          <td class="n">${g.i}</td>
          <td><${Ratio} v=${g.util} max=${100} warn=${0.999} bad=${1.1} /></td>
          <td><${Ratio} v=${g.vram} max=${g.vram_tot} unit="Mo" warn=${0.85} bad=${0.95}
                title=${`${g.vram} / ${g.vram_tot} Mo`} /></td>
          <td class="n">${g.w}</td>
          <td class=${'n ' + (g.temp > 80 ? 'bad' : g.temp > 70 ? 'warn' : '')}>${g.temp}</td>
        </tr>`)}
      </tbody>
    </table></div>
    <h3>mémoire</h3>
    <div class="stats">
      <span><span class="k">RAM </span>
        <${Bar} v=${n.mem_mb[0]} max=${n.mem_mb[1]} warn=${0.85} bad=${0.95}
          title=${`${n.mem_mb[0]} / ${n.mem_mb[1]} Mo`} />
        <span class="num">${n.mem_mb[0]} / ${n.mem_mb[1]} Mo</span></span>
      ${/* Le swap n'est pas une statistique de confort : une tempête de swap a
            déjà bloqué la ferme ~50 min (cf. gpu_worker.sh). Dès qu'il monte,
            il doit se voir — d'où des seuils bien plus bas que pour la RAM. */''}
      <span><span class="k">swap </span>
        <${Bar} v=${n.swap_mb[0]} max=${n.swap_mb[1]} warn=${0.08} bad=${0.35}
          title=${`${n.swap_mb[0]} / ${n.swap_mb[1]} Mo`} />
        <span class="num">${n.swap_mb[0]} / ${n.swap_mb[1]} Mo</span></span>
    </div>
  </section>`;

// ── run RL ──────────────────────────────────────────────────────────────────

const STATE_LABEL = { ok: 'en cours', stop: 'STOP', sec: 'à sec — aucun rollout' };
const STATE_CLS = { ok: 'ok', stop: 'dim', sec: 'warn' };

const EnvTable = ({ envs }) => {
  const rows = Object.entries(envs || {});
  if (!rows.length) return null;
  return html`
    <h3>par environnement</h3>
    <div class="scroll"><table>
      <thead><tr><th>env</th><th class="n">groupes</th><th class="n">reward</th>
        <th class="n">grade</th><th class="n">write%</th><th class="n">p(w)</th></tr></thead>
      <tbody>${rows.map(([e, v]) => html`
        <tr key=${e}>
          <td class="num">${e}</td>
          <td class="n">${v.n}</td>
          <td class="n">${nf(v.reward)}</td>
          <td class="n" style="color:var(--accent)">${nf(v.grade, 2)}</td>
          <td class="n">${nf(v.write_rate, 2)}</td>
          <td class="n">${nf(v.p_write, 2)}</td>
        </tr>`)}
      </tbody>
    </table></div>
    ${/* Colonnes NON comparables entre elles : les envs denses rendent -ce
          (~-8), les envs à rubrique rendent [0,1]. C'est pour ça que le reward
          agrégé du learner ne veut rien dire — il bouge avec le mix, pas avec
          la politique. Hors de la <table> : en <caption> la note héritait de la
          largeur de la table, devenue étroite, et se cassait en 4 lignes. */''}
    <p class="note">reward dense (−ce) pour code/sota, rubrique [0,1] pour
      tools/exec : ne pas comparer les lignes entre elles. <b>grade</b> = succès
      brut avant économie de think (toolcall, pass-rate).</p>`;
};

/* Les workers RL étaient listés DEUX fois : ici et dans « jobs actifs », avec
   des colonnes et des couleurs d'âge contradictoires pour le même processus.
   Une seule table désormais, qui absorbe le nom du job (on en a besoin pour
   re-filer un .job dans la queue). */
const WorkerTable = ({ ws, maxLag, jobOf }) => {
  if (!ws.length) return null;
  return html`
    <h3>workers</h3>
    <div class="scroll"><table>
      <thead><tr><th>worker</th><th>job</th><th class="n">groupes</th><th class="n">lag</th>
        <th class="n">s/groupe</th><th class="n">dégén.</th><th>dernier env</th>
        <th class="n">reward</th><th class="n">vu il y a</th></tr></thead>
      <tbody>${ws.map(k => html`
        <tr key=${k.wid}>
          <td class="num">${'w' + String(k.wid).padStart(2, '0')}</td>
          <td class="num dim">${jobOf.get(k.wid) || '—'}</td>
          <td class="n">${k.n ?? '—'}</td>
          <td class=${'n ' + (k.lag != null && maxLag != null && k.lag > maxLag ? 'bad' : '')}>
            ${k.lag ?? '—'}</td>
          <td class="n">${dur(k.s_per_group)}</td>
          <td class=${'n ' + (k.degen ? 'warn' : '')}>${k.degen ?? '—'}</td>
          <td class="num">${k.env || '—'}</td>
          <td class="n">${nf(k.reward)}</td>
          <td class=${'n ' + (k.alive ? '' : 'warn')}>${dur(k.age_s)}</td>
        </tr>`)}
      </tbody>
    </table></div>`;
};

const RunBody = ({ R, jobOf }) => {
  const L = R.learner, w = L?.win || {}, t = L?.trend || {}, S = L?.spark || {};
  const x = R.xdom;
  return html`
    ${L && html`
      <div class="tiles">
        <${Tile} k="reward" v=${nf(w.reward)} trend=${t.reward} good="up"
          pts=${S.reward} col="var(--ok)" />
        <${Tile} k="ce" v=${nf(w.ce)} trend=${t.ce} good="down" pts=${S.ce}
          col="var(--accent)" />
        ${/* write% : ni bien ni mal en soi — la politique d'écriture est
              justement ce que le RL doit trouver. Direction sans jugement. */''}
        <${Tile} k="write%" v=${nf(w.write_rate, 2)} trend=${t.write_rate} good="none"
          pts=${S.write_rate} col="var(--accent)" />
        <${Tile} k="lag" v=${nf(w.lag, 1)} />
        <${Tile} k="s/step" v=${dur(L.s_per_step)} pts=${S.sps} col="var(--dim)" />
      </div>
      <div class="stats">
        <${Stat} k="p(w)">${nf(w.p_write, 2)}<//>
        <${Stat} k="kl">${nf(w.kl)}<//>
        <${Stat} k="poscorr">${nf(w.pos_corr, 2)}<//>
        <${Stat} k="stale">${L.stale ?? '—'}<//>
        <${Stat} k="poids">${R.weights.step == null ? '—' : 'step ' + R.weights.step}<//>
        <${Stat} k="file">${R.queue.incoming}${R.queue.stale
          ? html` <span class="warn">+${R.queue.stale} stale</span>` : ''}<//>
        ${R.queue.traces_mb != null && html`<${Stat} k="traces">${R.queue.traces_mb} Mo<//>`}
        <${Stat} k="learner vu il y a">
          <span class=${L.alive ? 'ok' : 'bad'}>${dur(L.age_s)}</span><//>
      </div>`}
    ${/* Les tables se calent sur leur contenu, donc elles sont étroites : côte
          à côte, elles occupent enfin la largeur de la carte au lieu de laisser
          les trois quarts vides. Elles se replient l'une sous l'autre d'elles-
          mêmes quand la place manque. */''}
    <div class="cols">
      <div><${EnvTable} envs=${R.envs} /></div>
      <div><${WorkerTable} ws=${R.workers} maxLag=${R.meta?.max_lag} jobOf=${jobOf} /></div>
    </div>
    ${x && html`
      <h3>sonde xdom <span class="dim num">@${x.n}</span></h3>
      <div>
        <span class=${'chip ' + (x.r_own > x.r_xdom ? 'ok' : 'bad')}>
          own ${nf(x.r_own)} vs xdom ${nf(x.r_xdom)}</span>
        <span class="chip">always ${nf(x.r_always)}</span>
        <span class="chip">never ${nf(x.r_never)}</span>
      </div>`}`;
};

const RunRL = ({ R, jobOf }) => {
  const L = R.learner, stopped = R.state === 'stop';
  const cls = STATE_CLS[R.state] || 'bad';
  const label = STATE_LABEL[R.state] || 'MUET';
  const head = html`
    <h2><span class=${'dot ' + (cls === 'dim' ? '' : cls)}></span>RL ${R.run}
      <span class=${'badge ' + cls}>${label}</span>
      ${/* htm élague les blancs qui contiennent un saut de ligne : toute espace
            significative est écrite dans le texte, pas dans l'indentation. */''}
      ${L?.step != null && html`
        <span class="sp"><span class="k">step </span><b class="big">${L.step}</b><span
          class="num dim">${R.steps ? ' / ' + R.steps : ''}</span>${R.eta_s
            ? html`<span class="k"> · eta </span><span class="num">${dur(R.eta_s)}</span>` : ''}</span>`}
    </h2>
    ${L?.step != null && R.steps && html`
      <div class="prog" title=${`${pct(L.step, R.steps)} %`}>
        <i style=${`width:${pct(L.step, R.steps)}%`}></i></div>`}`;

  // Un run arrêté reste sur le share pour toujours. Replié, il ne pousse pas le
  // run vivant hors de l'écran — mais son détail reste à un clic (avant, il
  // fallait repasser par le CLI).
  return html`
    <section class=${'card s-' + (stopped ? 'none' : cls === 'ok' ? 'ok' : cls === 'warn' ? 'warn' : 'bad')
      + (stopped ? '' : ' wide')}>
      ${head}
      ${stopped
        ? html`<details><summary>détail du run terminé</summary>
            <div><${RunBody} R=${R} jobOf=${jobOf} /></div></details>`
        : html`<${RunBody} R=${R} jobOf=${jobOf} />`}
    </section>`;
};

// ── jobs de la ferme ────────────────────────────────────────────────────────

const JobsCard = ({ jobs, merged }) => html`
  <section class="card">
    <h2>jobs actifs <span class="badge">${jobs.length + merged}</span>
      ${merged > 0 && html`<span class="sp dim">${merged} worker${merged > 1 ? 's' : ''}${' '}
        RL ${merged > 1 ? 'affichés' : 'affiché'} dans la carte du run</span>`}
    </h2>
    ${jobs.length === 0
      ? html`<div class="dim">aucun autre job</div>`
      : html`<div class="scroll"><table>
        <thead><tr><th>job</th><th>step / état</th><th>ic · détail</th>
          <th class="n">s/step</th><th>dernier GAP par domaine</th>
          <th class="n">log il y a</th></tr></thead>
        <tbody>${jobs.map(j => html`<${JobRow} j=${j} key=${j.job} />`)}</tbody>
      </table></div>`}
  </section>`;

const JobRow = ({ j }) => {
  const s = j.step;
  const at = (j.evals && j.evals.length) ? Math.max(...j.evals.map(x => x.at)) : 0;
  // Seuil de silence fourni par le serveur : 300 s pour un trainer bavard,
  // 3 × s/groupe pour un worker RL (muet par construction), aucun pour un job
  // fini. Un seuil fixe peignait en rouge un run parfaitement sain.
  const lim = j.stale_after;
  const late = lim && j.log_age_s > lim;
  let c2, c3, c4 = '';
  if (j.kind === 'rl_worker') {
    c2 = html`<span class="ok num">RL w${String(j.rl_wid).padStart(2, '0')}</span>`;
    c3 = j.rl ? `${j.rl.n} groupes · lag ${j.rl.lag ?? '—'}`
      : html`<span class="warn">aucune métrique écrite</span>`;
  } else if (j.kind === 'rl_done') {
    c2 = html`<span class="dim">terminé</span>`;
    c3 = `${j.rl_groups} groupes`;
  } else if (j.kind === 'rl_learner' && s) {
    c2 = s.n; c3 = html`r ${nf(s.r)} · ce ${nf(s.ce)}`; c4 = s.sps;
  } else {
    c2 = s ? s.n : html`<span class="dim">init…</span>`;
    c3 = s ? s.ic : ''; c4 = s ? s.sps : '';
  }
  return html`
    <tr>
      <td class="num">${j.job}</td>
      <td class="n num">${c2}</td>
      <td class="num">${c3}</td>
      <td class="n">${c4}</td>
      <td>${j.evals && j.evals.length
        ? html`<span class="dim num">@${at}</span> ${j.evals.map(x => html`
            <span class=${'chip ' + (x.gap > 0 ? 'ok' : 'bad')} key=${x.src}>
              ${x.src} <b>${x.gap > 0 ? '+' : ''}${x.gap.toFixed(2)}</b>
              ${x.at < at ? html` <span class="dim">@${x.at}</span>` : ''}</span>`)}`
        : html`<span class="dim">—</span>`}</td>
      <td class=${'n ' + (late ? 'bad' : '')}>${j.log_age_s == null ? '?' : dur(j.log_age_s)}</td>
    </tr>`;
};

const JobList = ({ title, items, cls = '', open }) => html`
  <details open=${open || undefined}>
    <summary class=${cls}>${title} <span class="badge">${items.length}</span></summary>
    <ul class="jobs">${items.length
      ? items.map(f => html`<li key=${f}>${f}</li>`)
      : html`<li class="dim">—</li>`}</ul>
  </details>`;

/* Les trois listes tenaient chacune une cellule de la grille pour n'afficher
   qu'une ligne de résumé, et laissaient un trou vertical énorme. Une seule
   carte, trois replis. */
const ListsCard = ({ d }) => html`
  <section class=${'card ' + (d.failed.length ? 's-bad' : '')}>
    <h2>file et historique</h2>
    <${JobList} title="en attente" items=${d.queued} open=${d.queued.length > 0} />
    <${JobList} title="terminés" items=${d.done} />
    <${JobList} title="échecs" items=${d.failed} cls=${d.failed.length ? 'bad' : ''}
      open=${d.failed.length > 0} />
  </section>`;

// ── bandeau ─────────────────────────────────────────────────────────────────

const StatusBar = ({ d, downFor }) => html`
  <header class="bar">
    <h1>ferme thought-bank</h1>
    ${(d?.nodes || []).map(n => html`
      <span class="pill" key=${n.host}>
        <span class=${'dot ' + (n.offline ? 'bad' : 'ok')}></span>${n.host}</span>`)}
    ${(d?.rl || []).filter(R => R.state !== 'stop').map(R => html`
      <span class="pill" key=${R.run}>
        <span class=${'dot ' + (STATE_CLS[R.state] || 'bad')}></span>
        RL ${R.run.replace(/^disagg_/, '')}
        ${R.learner?.step != null && html`<span class="num">
          ${' ' + R.learner.step}${R.steps ? '/' + R.steps : ''}</span>`}</span>`)}
    ${d && html`<span class="pill k">file <span class="num">${d.queued.length}</span></span>`}
    ${d && d.failed.length > 0 && html`
      <span class="pill bad">✗ <span class="num">${d.failed.length}</span></span>`}
    <span class="sp dim num">
      ${d ? new Date(d.ts * 1000).toLocaleTimeString() : '—'}</span>
    ${downFor != null && html`
      <div class="offline">dashboard injoignable depuis ${dur(downFor)} —
        affichage figé sur le dernier instantané</div>`}
  </header>`;

// ── application ─────────────────────────────────────────────────────────────

/* Garde le dernier instantané en cas d'échec : la version précédente remplaçait
   toute la page par « dashboard injoignable », ce qui effaçait l'information au
   moment précis où on en avait besoin. Et se met en pause onglet caché — le
   serveur est mono-thread. */
function useSnapshot(ms = POLL_MS) {
  const [st, setSt] = useState({ d: null, down: null, now: Date.now() });
  useEffect(() => {
    let live = true, timer = null;
    const schedule = () => { timer = setTimeout(tick, ms); };
    const tick = async () => {
      if (document.hidden) return schedule();
      try {
        const r = await fetch('/data.json', { cache: 'no-store' });
        if (!r.ok) throw new Error('HTTP ' + r.status);
        const d = await r.json();
        if (live) setSt({ d, down: null, now: Date.now() });
      } catch {
        if (live) setSt(s => ({ ...s, down: s.down || Date.now(), now: Date.now() }));
      }
      if (live) schedule();
    };
    const wake = () => { if (!document.hidden) { clearTimeout(timer); tick(); } };
    tick();
    document.addEventListener('visibilitychange', wake);
    return () => {
      live = false; clearTimeout(timer);
      document.removeEventListener('visibilitychange', wake);
    };
  }, [ms]);
  return st;
}

const App = () => {
  const { d, down, now } = useSnapshot();

  // Un onglet en arrière-plan devient informatif : le titre porte l'état.
  useEffect(() => {
    const R = (d?.rl || []).find(r => r.state !== 'stop');
    document.title = R && R.learner?.step != null
      ? `RL ${R.learner.step}${R.steps ? '/' + R.steps : ''} · ferme`
      : 'ferme thought-bank';
  }, [d]);

  if (!d) {
    return html`<${StatusBar} d=${null} downFor=${down ? (now - down) / 1000 : null} />
      <p class="boot">${down ? 'aucune donnée : le dashboard ne répond pas.' : 'chargement…'}</p>`;
  }

  // Appariement worker RL ↔ job de la ferme, pour la table fusionnée. Un job
  // rl_worker SANS ligne worker correspondante n'est PAS masqué : c'est
  // justement celui qu'il faut voir (il tourne sans rien produire).
  const jobOf = new Map(), shown = new Set();
  for (const R of d.rl || []) {
    const wids = new Set(R.workers.map(w => w.wid));
    for (const j of d.running) {
      if (j.kind === 'rl_worker' && wids.has(j.rl_wid)) {
        jobOf.set(j.rl_wid, j.job);
        shown.add(j.job);
      }
    }
  }
  const jobs = d.running.filter(j => !shown.has(j.job));

  return html`
    <${StatusBar} d=${d} downFor=${down ? (now - down) / 1000 : null} />
    <main>
      ${(d.rl || []).map(R => html`<${RunRL} R=${R} jobOf=${jobOf} key=${R.run} />`)}
      ${d.nodes.map(n => html`<${NodeCard} n=${n} key=${n.host} />`)}
      <${JobsCard} jobs=${jobs} merged=${shown.size} />
      <${ListsCard} d=${d} />
    </main>`;
};

render(html`<${App} />`, document.getElementById('app'));
