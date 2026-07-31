"""rti_learner — la PASSE 2 du bras retrieve-then-inject : avantage, loss, crédit.

CE QUE CE MODULE FAIT, ET CE QU'IL NE FAIT PAS
──────────────────────────────────────────────
`rti_policy.py` échantillonne les trois actions (write Bernoulli, retrieve
Plackett-Luce, tokens de décodage) et écrit leurs log-probs de COMPORTEMENT
dans la trace. Ici on RECALCULE ces log-probs sous les poids courants du
learner (la « passe 2 »), on en tire un ratio d'importance par action, on
construit l'avantage, et on rend une loss différentiable. L'optimizer, la
publication des poids et la boucle restent dans `rl_disagg.Learner`.

LA RECETTE (décisions ACTÉES, pas du GRPO vanilla)
──────────────────────────────────────────────────
1. AVANTAGE Dr.GRPO — baseline = moyenne de groupe, JAMAIS de division par
   l'écart-type du groupe. Diviser par σ surpondère mécaniquement les groupes
   faciles/difficiles (biais démontré) ; une constante fixe (`adv_scale`) est
   permise, une statistique de groupe non. Pas de KL à une référence : le
   cliquet SFT joue ce rôle ENTRE phases, pas dedans (et sur ce bras la ref
   n'existe même pas — le learner ne la construit pas).

2. LOSS CISPO (MiniMax) — on clippe le POIDS D'IMPORTANCE, pas la mise à jour :

       L = − stopgrad(clip(r, r_lo, r_hi)) · A · log π(a)

   Un PPO-clip annule le gradient dès que le ratio sort de la bande ; comme la
   politique de write démarre FROIDE et sous plancher d'exploration
   (`write_floor`), ses actions rares sont précisément celles dont le ratio
   sort en premier — un PPO-clip les tuerait avant qu'elles n'apprennent quoi
   que ce soit. CISPO garde toujours un gradient de direction A·∇log π, borné
   en amplitude. Voir le self-test (f) : la comparaison au gradient PPO-clip
   (exactement nul) est faite chiffres en main.

3. AVANTAGE À DEUX NIVEAUX (GiGPO) ancré sur l'INDEX DE TOUR —

       A_total(action au tour t) = A_épisode + ω · A_tour(t)

   A_épisode = reward de vie − moyenne du groupe. A_tour(t) = retour depuis t
   (somme escomptée des rewards des sondes POSTÉRIEURES OU ÉGALES à t) moins la
   moyenne du groupe sur cette même ancre. L'ancre est exacte parce que les G
   rollouts partagent UN script (`recall_env.script_digest`) : le tour t de
   deux rollouts est LE MÊME tour. Sans ce second niveau, l'avantage épisode
   devient non informatif au-delà d'une dizaine de tours — tout se noie dans la
   moyenne d'une vie.

   Convention `>= t` : la sonde du tour t crédite les actions du tour t. C'est
   indispensable pour les tokens décodés (sinon l'action qui PRODUIT la réponse
   n'est créditée par rien) et légèrement généreux pour le write du seg
   assistant du même tour, qui arrive après le décodage. Assumé : γ=1 et le
   reward est partagé de toute façon.

4. CRÉDIT WRITE CONTREFACTUEL (leave-one-slot) — pour chaque write ayant fini
   dans le préfixe d'une sonde notée :

       Δ = logp(réponse produite | slot masqué) − logp(réponse | sélection)

   Δ < 0 ⇒ le slot PORTAIT la réponse. Le crédit ajouté à l'avantage du write
   est donc −Δ (signe inversé, sinon on punirait exactement le write utile),
   pondéré par (2·reward − 1) en mode `signed` : un slot qui a solidement porté
   une MAUVAISE réponse ne mérite pas d'être renforcé. C'est le remplacement
   assumé du crédit « hindsight factuel » (« le fait était dans la banque au
   moment de la sonde réussie »), qui récompense l'écriture de tout, tout le
   temps. Coût : 1 forward teacher-forcé par (write, sonde) — `cf_coef: 0.0`
   l'éteint.

   Le contrôle n'est PAS la log-prob du rollout mais la même fonction appelée
   sans masque (`mask_gid=None`) : la sélection y est rejouée en argmax comme
   dans le cas masqué, donc Δ isole l'effet DU SLOT et pas la différence entre
   un tirage Plackett-Luce et un argmax.

5. FILTRAGE ZÉRO-VARIANCE — un groupe dont toutes les vies rendent le même
   reward ne produit aucun gradient d'épisode (tous les A_épisode sont nuls) :
   il est jeté, et compté. Sauf si `keep_turn_only` et qu'il reste de la
   variance PAR TOUR — cas réel : deux rollouts réussissent chacun une sonde
   différente, la moyenne de vie est identique mais le crédit par tour, lui,
   discrimine.

6. FP32 — la passe 2 tourne sans autocast, en fp32 de bout en bout (le modèle
   du learner est en fp32 : `ThoughtBankLM(...)` n'est jamais casté). Le
   décalage numérique sampler/learner est un facteur de premier ordre sur les
   ratios ; on ne l'aggrave pas avec du bf16 côté update.

7. RATIOS PAR FAMILLE — r = exp(logp_passe2 − logp_trace), une famille à la
   fois (write / retrieve / tokens). Les poids `w_write`, `w_retr`, `w_dec`
   pondèrent la somme. La staleness (jusqu'à `max_lag` publications de retard)
   n'a pas de correction dédiée : c'est exactement ce que le clipping CISPO
   régularise.

CE QUE LA TRACE DOIT PORTER (contrat, agent 2 + l'ajout de la passe 2)
─────────────────────────────────────────────────────────────────────
Le groupe expédié porte, en plus des G traces : `segs` (les ids du script —
sans eux la passe 2 ne peut pas refaire un seul hidden), `a_open`, `sep_id`, et
`sif` (la table SIF RESTREINTE aux tokens réellement indexés : ids des segs +
tokens décodés). Ces quatre champs rendent le learner INDÉPENDANT du
tokenizer, du corpus de filler et du stream — il ne reconstruit rien, il rejoue.

Self-test CPU :
    python -m deepseek_v4_mini.rti_learner
"""
from __future__ import annotations

import math
import statistics as st
from dataclasses import dataclass

import torch

from .rti import RtiConfig, build_group, build_prefix
from .rti_policy import (RtiPolicyConfig, forward_probe_with_masked_slot,
                         null_banks, pl_logp, probe_pairs, replay_logp,
                         write_logp)


# ── config ───────────────────────────────────────────────────────────────────

@dataclass
class RtiLearnConfig:
    """Les knobs de l'ALGO. Section `rl.learner:` du YAML."""
    algo: str = "cispo"
    cispo_low: float = 0.0      # borne BASSE du poids d'importance. 0.0 = pas
                                # de borne basse (r >= 0 par construction) : les
                                # actions rares gardent tout leur gradient.
    cispo_high: float = 2.0     # borne HAUTE. Large exprès : le plancher
                                # d'exploration du write met déjà les ratios
                                # autour de p/p_b, souvent ~2.
    omega: float = 1.0          # poids de l'avantage de TOUR (GiGPO).
    gamma: float = 1.0          # escompte du retour par tour. 1.0 : les vies
                                # font quelques sondes, escompter n'apporterait
                                # qu'un biais de récence non désiré.
    adv_scale: float = 1.0      # normalisation batch-level DOUCE : constante
                                # fixe, jamais une statistique de groupe.
    cf_coef: float = 0.5        # crédit contrefactuel leave-one-slot (0 = off).
    cf_signed: bool = True      # pondérer le crédit par (2·reward − 1).
    cf_clip: float = 5.0        # |Δ| borné : un Δ de −40 nats (réponse vide,
                                # décodage dégénéré) noierait l'avantage.
    w_write: float = 1.0
    w_retr: float = 1.0
    w_dec: float = 1.0
    tok_norm: float = 0.0       # diviseur FIXE de la famille tokens (Dr.GRPO :
                                # jamais la longueur de la réponse, qui
                                # introduit un biais de longueur). 0 ⇒ le
                                # learner met `rl.max_new`.
    drop_zero_var: bool = True
    keep_turn_only: bool = True
    write_cost: float = 0.0     # pénalité de coût par write, RETRANCHÉE à
                                # l'avantage d'épisode (n_writes normalisé par
                                # les tours). OFF par défaut : la FIFO price
                                # déjà les writes par éviction.

    def __post_init__(self):
        assert self.algo == "cispo", self.algo
        assert 0.0 <= self.cispo_low <= 1.0 <= self.cispo_high
        assert self.gamma > 0 and self.cf_clip > 0


def learn_from_raw(raw: dict) -> RtiLearnConfig:
    """Section `rl.learner:`, avec le défaut de `tok_norm` pris sur `rl.max_new`
    (le diviseur fixe de Dr.GRPO doit valoir la longueur MAXIMALE possible, pas
    la longueur réalisée)."""
    r = dict((raw or {}).get("rl") or {})
    cfg = RtiLearnConfig(**dict(r.get("learner") or {}))
    if cfg.tok_norm <= 0:
        cfg.tok_norm = float(r.get("max_new", 64))
    return cfg


# ── avantages ────────────────────────────────────────────────────────────────

def _anchor(t: dict):
    """L'ancre GiGPO d'un seg : l'index de TOUR du script (le seg user et le
    seg assistant d'un même tour partagent donc leur avantage de tour). Repli
    sur l'index de seg pour un seg hors script — il n'y en a pas dans
    `recall_env`, mais un repli muet vaut mieux qu'un KeyError en production."""
    tn = t.get("turn")
    return int(tn) if tn is not None else f"s{int(t['seg'])}"


def anchor_pos(trace: dict) -> dict:
    """ancre → index du PREMIER seg qui la porte (l'ordre temporel réel)."""
    pos: dict = {}
    for t in trace["turns"]:
        pos.setdefault(_anchor(t), int(t["seg"]))
    return pos


def turn_returns(trace: dict, gamma: float) -> dict:
    """ancre → retour depuis cette ancre = Σ γ^k · reward des sondes NOTÉES
    dont le seg est >= celui de l'ancre. Les sondes impossibles (reward None)
    n'entrent nulle part : elles ne sont ni un succès ni un échec."""
    probes = [(int(t["seg"]), float(t["reward"])) for t in trace["turns"]
              if t.get("decode") is not None and t.get("reward") is not None]
    out = {}
    for a, s0 in anchor_pos(trace).items():
        post = [r for s, r in probes if s >= s0]
        out[a] = sum((gamma ** k) * r for k, r in enumerate(post))
    return out


def advantages(traces: list, cfg: RtiLearnConfig) -> dict:
    """Les deux niveaux d'avantage d'UN groupe.

    Rend {"ep": [G], "turn": [G dicts], "usable", "zero_var_ep", "var_turn",
    "reward"}. `usable=False` = groupe à jeter (aucune sonde notée, ou
    zéro-variance sur les deux niveaux).
    """
    rs = [tr.get("reward") for tr in traces]
    if any(r is None for r in rs) or not rs:
        # toutes les traces partagent le même ensemble de sondes possibles :
        # un None quelque part veut dire « aucune sonde notée dans cette vie ».
        return {"usable": False, "reason": "no_scored", "ep": [], "turn": [],
                "zero_var_ep": True, "var_turn": False, "reward": None}
    rs = [float(r) for r in rs]
    mu = st.mean(rs)
    ep = [r - mu for r in rs]
    if cfg.write_cost:
        # coût de write, RETRANCHÉ à l'avantage (pas au reward : le reward
        # reste le grade de sonde, verifiable et comparable entre runs).
        nt = max(len(traces[0]["turns"]), 1)
        cw = [cfg.write_cost * int(tr["n_writes"]) / nt for tr in traces]
        ep = [a - (c - st.mean(cw)) for a, c in zip(ep, cw)]
    rets = [turn_returns(tr, cfg.gamma) for tr in traces]
    keys = set().union(*[set(r) for r in rets]) if rets else set()
    turn = [dict() for _ in traces]
    var_turn = False
    for a in keys:
        vals = [r.get(a, 0.0) for r in rets]
        m = st.mean(vals)
        if max(vals) - min(vals) > 1e-12:
            var_turn = True
        for g, v in enumerate(vals):
            turn[g][a] = v - m
    zero_ep = max(ep) - min(ep) <= 1e-12
    usable = True
    if zero_ep and cfg.drop_zero_var:
        usable = bool(cfg.keep_turn_only and var_turn and cfg.omega > 0)
    return {"usable": usable, "reason": "zero_var" if not usable else None,
            "ep": ep, "turn": turn, "zero_var_ep": zero_ep,
            "var_turn": var_turn, "reward": rs}


# ── la passe 2 : recalcul des log-probs ──────────────────────────────────────

def _hidden(model, x, amp: bool = False):
    """h [1,T,d] d'un seg NU — la copie exacte de `RtiRollout._hidden`, GRAD
    ACTIF. Toute divergence ici (autocast, layer_banks) se paie en ratios qui
    ne valent plus 1 en on-policy, c'est-à-dire en gradient faux mais fini."""
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp and x.is_cuda):
        o = model(x, init_mem=None, layer_banks=null_banks(model), write=False,
                  compute_logits=False)
    return o["hidden"].float()


def _hidden_batch(model, ids: list, sep_id: int, amp: bool = False):
    """h [G,d] des réponses décodées — copie exacte de `RtiRollout._hidden_batch`
    (right-padding au `sep_id`, dernière position UTILE de chaque ligne). La
    composition du batch est la même qu'au rollout (les G rollouts), donc la
    valeur l'est aussi."""
    L = max(int(t.size(1)) for t in ids)
    dev = ids[0].device
    x = torch.full((len(ids), L), int(sep_id), dtype=torch.long, device=dev)
    for g, t in enumerate(ids):
        x[g, :t.size(1)] = t[0]
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp and x.is_cuda):
        o = model(x, init_mem=None, layer_banks=null_banks(model), write=False,
                  compute_logits=False)
    h = o["hidden"].float()
    return torch.stack([h[g, max(int(t.size(1)) - 1, 0)]
                        for g, t in enumerate(ids)])


def sif_dense(group: dict, vocab: int, device) -> torch.Tensor:
    """Table SIF reconstruite depuis la table CREUSE expédiée avec le groupe.

    Les tokens absents retombent à 1.0 (poids uniforme) : ils n'existent pas
    dans les segs du script ni dans les réponses décodées, donc `build_group`
    ne les indexe jamais. Si un jour il le faisait, le pooling resterait défini
    — simplement moins discriminant — au lieu de rendre un NaN silencieux.
    """
    w = torch.ones(int(vocab), dtype=torch.float32, device=device)
    sif = group.get("sif")
    if sif is not None:
        ids = sif["ids"].to(device).long()
        w[ids] = sif["w"].to(device=device, dtype=torch.float32)
    return w


class Pass2:
    """Rejoue UN groupe rti sous les poids courants et rend, par rollout, les
    log-probs DIFFÉRENTIABLES des trois familles d'actions."""

    def __init__(self, model, rcfg: RtiConfig, pcfg: RtiPolicyConfig,
                 *, amp: bool = False):
        self.model, self.rcfg, self.pcfg = model, rcfg, pcfg
        self.amp = bool(amp)

    # ── forwards mutualisés ─────────────────────────────────────────────────
    def _needed(self, traces, n):
        """(segs à forwarder nus, segs décodés à forwarder en batch)."""
        nu, dec = set(), set()
        for i in range(n):
            ts = [tr["turns"][i] for tr in traces]
            has_w = any(t["write"] is not None for t in ts)
            has_r = any(t["retrieve"] is not None for t in ts)
            is_d = ts[0]["decode"] is not None
            if has_r:
                assert not is_d, "un seg de QUESTION ne décode pas"
            if has_r or (has_w and not is_d):
                nu.add(i)
            if has_w and is_d:
                dec.add(i)
        return sorted(nu), sorted(dec)

    def group(self, group: dict, device=None) -> list:
        model, rcfg, pcfg = self.model, self.rcfg, self.pcfg
        dev = device or model.embed.weight.device
        traces = group["rollouts"]
        G = len(traces)
        segs = [t.to(dev) for t in group["segs"]]
        a_open = group["a_open"].to(dev)
        sep = int(group["sep_id"])
        sif_w = sif_dense(group, int(model.embed.weight.size(0)), dev)
        n = len(traces[0]["turns"])
        assert n == len(segs), (n, len(segs))

        nu, dec = self._needed(traces, n)
        h_seg = {i: _hidden(model, segs[i], self.amp)[0, -1] for i in nu}
        h_dec = {}
        for i in dec:
            ids = [torch.tensor([tr["turns"][i]["decode"]["tokens"] or [sep]],
                                dtype=torch.long, device=dev) for tr in traces]
            h_dec[i] = _hidden_batch(model, ids, sep, self.amp)

        out = []
        for g, tr in enumerate(traces):
            src: dict = {}                     # gid -> ids du seg SOURCE
            acts = {"write": [], "retr": [], "dec": []}
            last_ret = None
            for i, t in enumerate(tr["turns"]):
                anc = _anchor(t)
                # 1. DÉCODAGE (le préfixe a été décidé au seg précédent)
                d = t.get("decode")
                if d is not None:
                    if d["tokens"]:
                        inj = None
                        if d["n_prefix"]:
                            assert last_ret is not None and \
                                len(last_ret["order"]) == rcfg.eval_groups
                            tk = torch.tensor(
                                [last_ret["cand_toks"][j]
                                 for j in last_ret["order"]],
                                dtype=torch.long, device=dev)
                            inj = build_prefix(model.embed.weight,
                                               model.rti_type.vec, sep,
                                               [[tk[j] for j in range(tk.size(0))]],
                                               rcfg)
                        lp = replay_logp(
                            model, a_open,
                            torch.tensor([d["tokens"]], dtype=torch.long,
                                         device=dev),
                            inject=inj, temp=pcfg.decode_temp, amp=self.amp)[0]
                        acts["dec"].append({
                            "anchor": anc, "lp_new": lp,
                            "lp_old": torch.tensor(d["logp"], device=dev),
                            "probe_id": d["probe_id"], "n": len(d["tokens"])})
                    last_ret = None
                # 2. WRITE
                w = t.get("write")
                if w is not None:
                    h = h_dec[i][g] if d is not None else h_seg[i]
                    lg = model.rti_write(h)
                    acts["write"].append({
                        "anchor": anc, "gid": w["gid"], "a": int(w["a"]),
                        "lp_new": write_logp(lg, int(w["a"]), pcfg.write_temp),
                        "lp_old": float(w["logp"]),
                        "p": float(torch.sigmoid(lg.detach().float()
                                                 / pcfg.write_temp))})
                    if w["a"] and w["gid"] is not None:
                        src[w["gid"]] = (
                            torch.tensor(t["decode"]["tokens"] or [sep],
                                         dtype=torch.long, device=dev)
                            if d is not None else segs[i].reshape(-1))
                # 3. RETRIEVE (la query due au seg SUIVANT)
                r = t.get("retrieve")
                if r is not None:
                    keys = torch.stack([
                        build_group(model.embed.weight, sif_w, src[gid], rcfg)[0]
                        for gid in r["cand_gids"]])
                    q = h_seg[i].reshape(1, -1).to(
                        model.rti_retriever.wq.weight.dtype)
                    sc = model.rti_retriever(q, keys.unsqueeze(0).to(q.dtype))[0]
                    acts["retr"].append({
                        "anchor": anc, "lp_new": pl_logp(sc, r["order"],
                                                         pcfg.retr_temp),
                        "lp_old": float(r["logp"]),
                        "hit": bool(r["hit"]), "top1": bool(r["top1"])})
                    last_ret = r
            out.append(acts)
        return out


# ── crédit contrefactuel ─────────────────────────────────────────────────────

@torch.no_grad()
def leave_one_slot(model, trace: dict, rcfg: RtiConfig, sep_id: int, a_open,
                   cfg: RtiLearnConfig, *, temp: float = 1.0, amp: bool = False,
                   device=None) -> dict:
    """gid → crédit (à AJOUTER à l'avantage du write correspondant).

    Δ = logp(réponse | slot masqué) − logp(réponse | sélection rejouée). Δ < 0
    ⇒ le slot portait la réponse ⇒ crédit POSITIF (on renforce le write).
    En mode `signed`, le crédit change de signe quand la réponse était fausse :
    un slot qui a solidement porté une erreur n'est pas un bon write.
    """
    out: dict = {}
    if cfg.cf_coef == 0.0:
        return out
    raw: dict = {}
    for q, d in probe_pairs(trace):
        if q is None or d.get("reward") is None or not d["decode"]["tokens"]:
            continue
        pair = {"retrieve": q["retrieve"], "decode": d["decode"]}
        base = forward_probe_with_masked_slot(model, pair, None, rcfg, sep_id,
                                              a_open, temp=temp, amp=amp,
                                              device=device)
        for gid in q["retrieve"]["gids"]:
            got = forward_probe_with_masked_slot(model, pair, gid, rcfg, sep_id,
                                                 a_open, temp=temp, amp=amp,
                                                 device=device)
            if not got["dropped"]:
                continue                        # slot déjà hors des candidats
            delta = max(-cfg.cf_clip,
                        min(cfg.cf_clip, got["logp"] - base["logp"]))
            raw.setdefault(gid, []).append((delta, float(d["reward"])))
    for gid, vals in raw.items():
        cr = [(-dl) * ((2.0 * r - 1.0) if cfg.cf_signed else 1.0)
              for dl, r in vals]
        out[gid] = (cfg.cf_coef * st.mean(cr), st.mean([d for d, _ in vals]))
    return out


# ── la loss CISPO ────────────────────────────────────────────────────────────

def cispo_term(lp_new, lp_old, adv: float, lo: float, hi: float):
    """(terme de loss, ratios détachés, masque clippé). Vectoriel : `lp_new` et
    `lp_old` peuvent être des scalaires (write, retrieve) ou des vecteurs de
    tokens (décodage)."""
    if not torch.is_tensor(lp_old):
        lp_old = lp_new.new_tensor(float(lp_old))
    r = torch.exp(lp_new.detach() - lp_old)
    w = r.clamp(lo, hi)
    return -(w * float(adv) * lp_new).sum(), r, (r < lo) | (r > hi)


def group_loss(acts: list, adv: dict, cfg: RtiLearnConfig, credits: list,
               device) -> tuple:
    """Loss (scalaire différentiable) d'UN groupe + télémétrie brute."""
    loss = torch.zeros((), device=device)
    m = {"r_write": [], "r_retr": [], "r_dec": [], "clip": [0, 0],
         "p": [], "ent": [], "hit": [], "top1": [], "cf": [], "n_act": 0}
    fam = (("write", cfg.w_write, 1.0), ("retr", cfg.w_retr, 1.0),
           ("dec", cfg.w_dec, max(cfg.tok_norm, 1.0)))
    for g, a in enumerate(acts):
        A_ep = adv["ep"][g]
        for name, wf, norm in fam:
            if wf == 0.0:
                continue
            for act in a[name]:
                A = (A_ep + cfg.omega * adv["turn"][g].get(act["anchor"], 0.0))
                if name == "write" and act["gid"] is not None:
                    A = A + credits[g].get(act["gid"], (0.0, 0.0))[0]
                t, r, cl = cispo_term(act["lp_new"], act["lp_old"],
                                      A * cfg.adv_scale, cfg.cispo_low,
                                      cfg.cispo_high)
                loss = loss + wf * t / norm
                m[f"r_{name}"] += [float(v) for v in r.reshape(-1)]
                m["clip"][0] += int(cl.sum())
                m["clip"][1] += int(cl.numel())
                m["n_act"] += 1
                if name == "write":
                    p = act["p"]
                    m["p"].append(p)
                    m["ent"].append(-(p * math.log(max(p, 1e-9))
                                      + (1 - p) * math.log(max(1 - p, 1e-9))))
                elif name == "retr":
                    m["hit"].append(float(act["hit"]))
                    m["top1"].append(float(act["top1"]))
        m["cf"] += [d for _, d in credits[g].values()]
    return loss, m


# ── un pas complet sur un lot de groupes ─────────────────────────────────────

def step_groups(model, groups: list, rcfg: RtiConfig, pcfg: RtiPolicyConfig,
                cfg: RtiLearnConfig, *, device=None, amp: bool = False) -> dict:
    """Backward de tous les groupes du step (l'optimizer reste à l'appelant).

    Le backward est fait GROUPE PAR GROUPE : le graphe d'un groupe (une vie
    entière de forwards) est libéré avant de monter le suivant, sinon le pic
    mémoire vaut `groups_per_step` vies.
    """
    dev = device or next(model.parameters()).device
    p2 = Pass2(model, rcfg, pcfg, amp=amp)
    agg = {"r_write": [], "r_retr": [], "r_dec": [], "clip": [0, 0], "p": [],
           "ent": [], "hit": [], "top1": [], "cf": [], "reward": [],
           "reward_max": [], "loss": 0.0, "n_groups": 0, "n_dropped": 0,
           "n_act": 0, "adv": []}
    kept = []
    for grp in groups:
        traces = grp["rollouts"]
        adv = advantages(traces, cfg)
        if adv["reward"] is not None:
            agg["reward"] += adv["reward"]
            agg["reward_max"].append(max(adv["reward"]))
        if not adv["usable"]:
            agg["n_dropped"] += 1
            continue
        kept.append((grp, adv))
    scale = 1.0 / max(sum(len(g["rollouts"]) for g, _ in kept), 1)
    for grp, adv in kept:
        traces = grp["rollouts"]
        acts = p2.group(grp, device=dev)
        a_open = grp["a_open"].to(dev)
        credits = [leave_one_slot(model, tr, rcfg, int(grp["sep_id"]), a_open,
                                  cfg, temp=pcfg.decode_temp, amp=amp,
                                  device=dev)
                   for tr in traces]
        loss, m = group_loss(acts, adv, cfg, credits, dev)
        (loss * scale).backward()
        agg["loss"] += float(loss.detach()) * scale
        agg["n_groups"] += 1
        agg["adv"] += [abs(x) for x in adv["ep"]]
        for k in ("r_write", "r_retr", "r_dec", "p", "ent", "hit", "top1", "cf"):
            agg[k] += m[k]
        agg["clip"][0] += m["clip"][0]
        agg["clip"][1] += m["clip"][1]
        agg["n_act"] += m["n_act"]
    return agg


def telemetry(agg: dict) -> dict:
    """La ligne unique du step (style repo : tout ce qui décide d'un
    diagnostic, rien d'autre). `hit`/`top1` séparent un reward bas dû à la
    SÉLECTION d'un reward bas dû à la COPIE — les deux rendent 0.00."""
    mean = lambda v: (st.mean(v) if v else None)
    return {
        "reward": mean(agg["reward"]), "reward_max": mean(agg["reward_max"]),
        "groups": agg["n_groups"], "dropped": agg["n_dropped"],
        "drop_frac": agg["n_dropped"] / max(agg["n_groups"]
                                            + agg["n_dropped"], 1),
        "loss": agg["loss"], "n_act": agg["n_act"],
        "r_write": mean(agg["r_write"]), "r_retr": mean(agg["r_retr"]),
        "r_dec": mean(agg["r_dec"]),
        "clip_frac": agg["clip"][0] / max(agg["clip"][1], 1),
        "p_write": mean(agg["p"]), "h_write": mean(agg["ent"]),
        "hit": mean(agg["hit"]), "top1": mean(agg["top1"]),
        "cf_delta": mean(agg["cf"]), "adv": mean(agg["adv"]),
    }


# ── self-test ────────────────────────────────────────────────────────────────

def _fake_group(rewards, *, n_turns=4, probe_at=None, G=None):
    """Un groupe SYNTHÉTIQUE réduit à ce que les avantages regardent : des
    tours ancrés et des sondes notées. Pas de modèle, pas de tenseur."""
    G = G or len(rewards)
    probe_at = probe_at if probe_at is not None else [n_turns - 1]
    traces = []
    for g in range(G):
        turns = []
        for i in range(n_turns):
            t = {"seg": i, "turn": i, "kind": "fact", "role": "user",
                 "write": None, "retrieve": None, "decode": None,
                 "reward": None}
            if i in probe_at:
                k = probe_at.index(i)
                t["decode"] = {"tokens": [1], "logp": [0.0], "n_prefix": 0,
                               "n_tokens": 1, "probe_id": k, "text": ""}
                rv = rewards[g][k] if isinstance(rewards[g], (list, tuple)) \
                    else rewards[g]
                t["reward"] = rv
            turns.append(t)
        ok = [t["reward"] for t in turns if t["reward"] is not None]
        traces.append({"life": 0, "digest": "d", "layout": {}, "turns": turns,
                       "n_writes": 0, "n_probes": len(probe_at),
                       "n_scored": len(ok),
                       "reward": (sum(ok) / len(ok)) if ok else None,
                       "slot_to_writes": {}, "probe_to_slot": {},
                       "probe_possible": {}})
    return traces


def _selftest() -> None:
    """Ce qui casse en silence ici : un ratio qui ne vaut pas 1 en on-policy
    (la passe 2 ne rejoue pas le même calcul que le rollout — gradient faux
    mais fini), un avantage de tour qui crédite après coup, un groupe plat qui
    passe quand même, un Δ contrefactuel de signe inversé (on renforcerait
    exactement les writes inutiles), une sonde impossible comptée comme un 0.
    """
    import random as _random

    import torch.nn as nn

    torch.manual_seed(20260731)
    cfgL = RtiLearnConfig()

    # ── (b) avantage GiGPO : seul le rollout 1 réussit la sonde du tour 2 ────
    tr = _fake_group([0.0, 1.0, 0.0, 0.0], n_turns=5, probe_at=[2])
    ad = advantages(tr, cfgL)
    assert ad["usable"] and abs(ad["ep"][1] - 0.75) < 1e-9
    for t in range(3):                          # ancres <= 2 : la sonde compte
        assert ad["turn"][1][t] > 0.7, (t, ad["turn"][1][t])
        assert ad["turn"][0][t] < 0.0
    for t in (3, 4):                            # après la DERNIÈRE sonde : nul
        assert all(abs(ad["turn"][g][t]) < 1e-12 for g in range(4)), t
    # deux sondes : l'ancre du milieu ne voit que la sonde postérieure
    tr2 = _fake_group([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]],
                      n_turns=5, probe_at=[1, 3])
    ad2 = advantages(tr2, cfgL)
    assert abs(ad2["ep"][0] - ad2["ep"][1]) < 1e-12, "même reward de vie"
    assert ad2["turn"][0][1] > 0 and ad2["turn"][0][2] < 0, \
        "la sonde 0 est PASSÉE au tour 2 : elle ne doit plus créditer"
    assert ad2["turn"][1][2] > 0
    print(f"  [b] GiGPO : A_ep {[round(x, 3) for x in ad['ep']]}, A_tour du "
          f"gagnant {[round(ad['turn'][1][t], 3) for t in range(5)]} — positif "
          f"jusqu'à l'ancre de la sonde, EXACTEMENT nul après la dernière")

    # ── (c) zéro-variance ───────────────────────────────────────────────────
    flat = advantages(_fake_group([0.5] * 4), cfgL)
    assert not flat["usable"] and flat["zero_var_ep"] and not flat["var_turn"]
    assert all(abs(x) < 1e-12 for x in flat["ep"]), "gradient d'épisode non nul"
    # variance de TOUR seulement : même reward de vie, sondes différentes
    keep = advantages(tr2, cfgL)
    assert keep["zero_var_ep"] is False or keep["usable"]
    flat_t = _fake_group([[1.0, 0.0], [0.0, 1.0]], n_turns=5, probe_at=[1, 3])
    kt = advantages(flat_t, cfgL)
    assert kt["zero_var_ep"] and kt["var_turn"] and kt["usable"], kt
    assert not advantages(flat_t, RtiLearnConfig(keep_turn_only=False))["usable"]
    # …et le gradient qui en sort est bien PUREMENT de tour
    # l'ancre 1 ne discrimine PAS (les deux sondes lui sont postérieures, γ=1
    # ⇒ même retour) ; c'est l'ancre 2, entre les deux sondes, qui porte tout
    # le signal — la démonstration que l'ancrage par tour est le bon niveau.
    a0 = kt["ep"]
    assert all(abs(x) < 1e-12 for x in a0)
    assert abs(kt["turn"][0][1]) < 1e-12
    assert kt["turn"][0][2] < -0.4 and kt["turn"][1][2] > 0.4
    print(f"  [c] zéro-variance : groupe plat jeté (A_ep tous nuls), groupe à "
          f"variance de TOUR seulement gardé sous keep_turn_only "
          f"(A_tour ±{abs(kt['turn'][0][2]):.2f} à l'ancre 2), jeté sans le "
          f"knob")

    # ── (e) sondes impossibles : None == la sonde n'existe pas ──────────────
    with_imp = _fake_group([[1.0, None], [0.0, None], [1.0, None], [0.0, None]],
                           n_turns=5, probe_at=[1, 3])
    without = _fake_group([1.0, 0.0, 1.0, 0.0], n_turns=5, probe_at=[1])
    a_i, a_w = advantages(with_imp, cfgL), advantages(without, cfgL)
    assert a_i["ep"] == a_w["ep"], (a_i["ep"], a_w["ep"])
    for g in range(4):
        for t in range(5):
            assert abs(a_i["turn"][g][t] - a_w["turn"][g][t]) < 1e-12, (g, t)
    # et un None PARTOUT (aucune sonde notée) rend le groupe inutilisable
    assert not advantages(_fake_group([None] * 4), cfgL)["usable"]
    print("  [e] sondes impossibles : reward None exclu (jamais 0) — avantages "
          "d'épisode ET de tour IDENTIQUES à un script sans cette sonde")

    # ── (f) CISPO vs PPO-clip : le gradient survit hors bande ───────────────
    # l'action RARE que la politique s'est mise à aimer : r = e² au-dessus de
    # la borne, avantage positif. C'est le cas que PPO-clip supprime.
    lp = torch.tensor(1.0, requires_grad=True)
    lp_old, A = -1.0, 1.0
    lo, hi = 0.0, 2.0
    t, r, cl = cispo_term(lp, lp_old, A, lo, hi)
    t.backward()
    g_cispo = float(lp.grad)
    assert bool(cl) and abs(float(r) - math.exp(2)) < 1e-5
    assert abs(g_cispo + hi * A) < 1e-6, g_cispo   # −clip(r)·A, JAMAIS zéro
    # le PPO-clip du learner v1, sur la même action : gradient EXACTEMENT nul
    lp2 = torch.tensor(1.0, requires_grad=True)
    r2 = (lp2 - lp_old).exp()
    ppo = -torch.min(r2 * A, r2.clamp(1 - 0.2, 1 + 0.28) * A)
    ppo.backward()
    g_ppo = float(lp2.grad)
    # r > 1+ε et A > 0 : le min prend la branche clippée, constante ⇒ ∇ = 0
    assert abs(g_ppo) < 1e-9, g_ppo
    # r dans la bande ⇒ CISPO == REINFORCE-with-baseline exactement
    lp3 = torch.tensor(-1.0, requires_grad=True)
    t3, r3, cl3 = cispo_term(lp3, -1.0, 0.7, 0.0, 2.0)
    t3.backward()
    assert abs(float(r3) - 1.0) < 1e-9 and not bool(cl3)
    assert abs(float(lp3.grad) + 0.7) < 1e-6
    print(f"  [f] CISPO : r={float(r):.3f} hors [{lo}, {hi}] ⇒ ∇ = {g_cispo:+.3f} "
          f"(= −r_lo·A), PPO-clip sur la MÊME action ⇒ ∇ = {g_ppo:+.1e} ; "
          f"r=1 ⇒ ∇ = −A (REINFORCE-with-baseline)")

    # ── (d) leave-one-slot sur un COPIEUR construit ─────────────────────────
    # Le modèle jouet réel n'a aucune raison de copier le préfixe ; on teste
    # donc l'ARITHMÉTIQUE du crédit sur un modèle qui copie par construction :
    # ses logits recopient, position par position, les tokens du GROUPE 0 du
    # préfixe. Masquer le groupe élu doit effondrer la log-prob ; masquer un
    # groupe de la 2ᵉ position (remplacé par un autre distracteur) ne doit rien
    # changer. C'est exactement ce que le crédit doit distinguer.
    Vc, K = 16, 2
    rc = RtiConfig(top_k=K, max_groups=4, train_groups=2, eval_groups=2)

    class _Copier(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(Vc, Vc)
            with torch.no_grad():
                self.embed.weight.copy_(torch.eye(Vc))
            self.blocks = [None, None]
            self.rti_type = type("T", (nn.Module,), {})()
            self.rti_type.vec = torch.zeros(Vc)

        def forward(self, x, init_mem=None, layer_banks=None, write=False,
                    inject=None, compute_logits=True):
            B, T = x.shape
            lg = torch.zeros(B, T, Vc)
            if inject is not None:
                # les K premières lignes du préfixe = le groupe 0 (one-hot)
                ids = inject[:, :K].argmax(-1)               # [B,K]
                for b in range(B):
                    for t in range(T):
                        lg[b, t, int(ids[b, min(t, K - 1)])] = 8.0
            return {"logits": lg, "hidden": torch.zeros(B, T, Vc)}

    cop = _Copier()
    a_open_c = torch.tensor([[0]], dtype=torch.long)
    #     cand 0 = le VRAI (tokens 5,6), 1 et 2 = distracteurs
    toks = [[5, 6], [9, 10], [11, 12]]
    ret = {"tag": 1, "order": [0, 1], "gids": [100, 101],
           "logp": 0.0, "temp": 1.0, "greedy": True,
           "scores": [3.0, 2.0, 1.0], "cand_gids": [100, 101, 102],
           "cand_toks": toks, "pos": [0], "hit": True, "top1": True}
    dec = {"probe_id": 0, "tokens": [5, 6], "logp": [0.0, 0.0],
           "n_prefix": rc.eval_groups * rc.group_prefix, "n_tokens": 2,
           "text": ""}
    pair = {"retrieve": ret, "decode": dec}
    base = forward_probe_with_masked_slot(cop, pair, None, rc, 0, a_open_c)
    d_true = forward_probe_with_masked_slot(cop, pair, 100, rc, 0,
                                            a_open_c)["logp"] - base["logp"]
    d_dist = forward_probe_with_masked_slot(cop, pair, 101, rc, 0,
                                            a_open_c)["logp"] - base["logp"]
    assert d_true < -5.0, d_true
    assert abs(d_dist) < 1e-6, d_dist
    # …et le crédit qui en sort : POSITIF pour le slot nécessaire (Δ<0), ~0
    # pour le distracteur ; NÉGATIF si la réponse copiée était fausse.
    trace_c = {"turns": [{"seg": 0, "turn": 0, "retrieve": ret, "write": None,
                          "decode": None, "reward": None},
                         {"seg": 1, "turn": 0, "retrieve": None, "write": None,
                          "decode": dec, "reward": 1.0}]}
    cr = leave_one_slot(cop, trace_c, rc, 0, a_open_c, cfgL)
    assert cr[100][0] > 0 and abs(cr[101][0]) < 1e-6, cr
    trace_bad = {"turns": [dict(trace_c["turns"][0]),
                           dict(trace_c["turns"][1], reward=0.0)]}
    cr_bad = leave_one_slot(cop, trace_bad, rc, 0, a_open_c, cfgL)
    assert cr_bad[100][0] < 0, cr_bad
    assert leave_one_slot(cop, trace_c, rc, 0, a_open_c,
                          RtiLearnConfig(cf_coef=0.0)) == {}
    print(f"  [d] leave-one-slot : Δ(slot nécessaire) = {d_true:+.2f} nats, "
          f"Δ(distracteur, préfixe de longueur CONSTANTE) = {d_dist:+.1e} ; "
          f"crédit {cr[100][0]:+.2f} si la réponse est juste, "
          f"{cr_bad[100][0]:+.2f} si elle est fausse")

    # ── (a) ON-POLICY : les ratios valent 1, et CISPO == REINFORCE ──────────
    from . import persona_chat_data as P
    from .config import ThoughtBankConfig
    from .model import ThoughtBankLM
    from .recall_env import (RecallEnvConfig, RecallEnvStream,
                             make_recall_env_reward)
    from .rti import sif_table
    from .rti_policy import RtiRollout, attach_rti_modules

    tok = P._StubTok()
    V = 512
    mcfg = ThoughtBankConfig(vocab_size=V, d_model=32, n_layers=2, n_heads=2,
                             d_head=8, max_seq_len=512, n_hc=2, sinkhorn_iters=5,
                             csa_m=3, hca_m=5, top_k_csa=64, n_win=4,
                             d_latent_q=16, n_groups=1, n_experts=2, n_shared=1,
                             top_k_experts=2, d_ff=64, mem_dim=16, max_mem=4,
                             mem_seed_slots=2, use_dual_stream=True)
    model = ThoughtBankLM(mcfg).double().eval()
    attach_rti_modules(model, 32)
    model.double()
    with torch.no_grad():
        model.rti_retriever.wq.weight.normal_(0, 0.1)
        model.rti_type.vec.normal_(0, 0.02)
        model.rti_write.w.weight.normal_(0, 0.5)
    rcfg2 = RecallEnvConfig(life_seed=7, n_facts=(3, 3), n_probes=(2, 2),
                            strata={"persona": 0.4, "numeric": 0.3,
                                    "preference": 0.3},
                            filler_per_fact=(1, 1), p_beyond=0.0,
                            inject_groups=2, max_groups=4,
                            age_bins=((2, 4),), age_weights=(1.0,),
                            surprisal_mode="nll")
    stream = RecallEnvStream(tok, seed=1, cfg=rcfg2)
    conv = stream.conv_for_life(3)
    rc2 = RtiConfig(top_k=4, max_groups=4, train_groups=2, eval_groups=2)
    sif_w = sif_table(stream, V, rc2.sif_a)
    a_open = torch.tensor([[7, 8]], dtype=torch.long)
    # write_floor 0 : la log-prob de COMPORTEMENT est celle de la politique,
    # donc l'on-policy est exact (avec un plancher, r = p/p_b ≠ 1 par design).
    pol = RtiPolicyConfig(write_mode="head", write_floor=0.0, retr_temp=1.0,
                          decode_temp=1.0)
    roll = RtiRollout(model, tok, rc2, pol, sif_w, sep_id=9, a_open=a_open,
                      stop_id=-1, max_new=6, amp=False)
    traces = roll.run(conv, 4, _random.Random(1),
                      reward_fn=make_recall_env_reward(),
                      generator=torch.Generator().manual_seed(1))
    ids_used = set()
    for s in conv["segs"]:
        ids_used |= set(s["input_ids"].reshape(-1).tolist())
    for tr in traces:
        for t in tr["turns"]:
            if t["decode"]:
                ids_used |= set(t["decode"]["tokens"])
    sid = torch.tensor(sorted(ids_used), dtype=torch.long)
    group = {"format": "rti", "env": "recall_env", "weights_step": 0,
             "worker": 0, "life": 3, "digest": traces[0]["digest"],
             "layout": traces[0]["layout"], "rollouts": traces,
             "segs": [s["input_ids"] for s in conv["segs"]],
             "a_open": a_open, "sep_id": 9,
             "sif": {"ids": sid, "w": sif_w[sid]}}
    acts = Pass2(model, rc2, pol).group(group)
    err = {"write": 0.0, "retr": 0.0, "dec": 0.0}
    n = {"write": 0, "retr": 0, "dec": 0}
    for g, a in enumerate(acts):
        for k in err:
            for act in a[k]:
                lo_ = act["lp_old"] if torch.is_tensor(act["lp_old"]) \
                    else torch.tensor(act["lp_old"])
                d = (act["lp_new"].detach().cpu().double()
                     - lo_.cpu().double()).abs().max()
                err[k] = max(err[k], float(d))
                n[k] += 1
    assert all(v > 0 for v in n.values()), n
    for k, v in err.items():
        assert v < 1e-6, (k, v)
    # …donc les ratios valent 1, et le gradient CISPO == REINFORCE-with-baseline
    adv = advantages(traces, cfgL)
    adv = {**adv, "usable": True,
           "ep": [1.0, -1.0, 0.5, -0.5], "turn": [dict() for _ in traces]}
    cr0 = [dict() for _ in traces]
    loss, m = group_loss(acts, adv, RtiLearnConfig(cf_coef=0.0, tok_norm=1.0),
                         cr0, torch.device("cpu"))
    model.zero_grad(set_to_none=True)
    loss.backward(retain_graph=True)
    g_cispo = {k: p.grad.clone() for k, p in model.named_parameters()
               if p.grad is not None}
    # REINFORCE-with-baseline : −Σ A · log π, sans aucun poids d'importance
    ref = torch.zeros((), dtype=torch.double)
    for g, a in enumerate(acts):
        for k in ("write", "retr", "dec"):
            for act in a[k]:
                ref = ref - adv["ep"][g] * act["lp_new"].sum()
    model.zero_grad(set_to_none=True)
    ref.backward()
    g_ref = {k: p.grad.clone() for k, p in model.named_parameters()
             if p.grad is not None}
    assert set(g_cispo) == set(g_ref) and g_cispo
    # écart RELATIF : les ratios ne valent 1 qu'à la précision du rejeu
    # (~1e-6, plancher mesuré par rti_policy (c)), donc le poids CISPO vaut 1
    # à la même précision — pas au bit. Comparer en absolu ne mesurerait que
    # cette tolérance-là, multipliée par la norme du gradient.
    scale_g = max(float(g_ref[k].abs().max()) for k in g_ref)
    gmax = max(float((g_cispo[k] - g_ref[k]).abs().max())
               for k in g_cispo) / max(scale_g, 1e-12)
    assert gmax < 1e-5, gmax
    assert all(abs(x - 1.0) < 1e-5 for x in m["r_write"] + m["r_retr"]
               + m["r_dec"])
    print(f"  [a] on-policy : max|logp_passe2 − logp_trace| = "
          f"write {err['write']:.1e} ({n['write']} actions) / retrieve "
          f"{err['retr']:.1e} ({n['retr']}) / tokens {err['dec']:.1e} "
          f"({n['dec']}) ⇒ ratios == 1, et ∇CISPO == "
          f"∇REINFORCE-with-baseline à {gmax:.1e} (relatif) sur {len(g_cispo)} tenseurs")

    print("rti_learner self-test: OK (on-policy ratios 1 + CISPO == REINFORCE, "
          "avantage GiGPO ancré au tour, filtrage zéro-variance avec repli "
          "tour, leave-one-slot signé, sondes impossibles neutres, gradient "
          "CISPO préservé là où PPO-clip l'annule)")


if __name__ == "__main__":
    _selftest()
