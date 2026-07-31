"""rti_policy — les TROIS ACTIONS RL du bras retrieve-then-inject.

CE QUE CE MODULE AJOUTE À `rti.py`
──────────────────────────────────
`rti.py` est le mécanisme SFT : le write est procédural (les segs PORTEURS
écrivent, privilège d'oracle) et le read est un top-k DUR. Aucun des deux n'est
une action : ni l'un ni l'autre n'a de log-probabilité, donc rien à créditer.
Ici on rend les deux ÉCHANTILLONNABLES, avec des log-probs qui tiennent :

  1. WRITE — Bernoulli par tour sur une tête minuscule (`WriteHead`, d→1,
     zéro-init ⇒ p = 0.5 au step 0). Le mode `facts` conserve le déclenchement
     procédural du SFT comme BASELINE, bit à bit (self-test (e)).
  2. RETRIEVE — Plackett-Luce : tirage SÉQUENTIEL SANS REMISE sur le softmax
     des scores du retriever. La log-prob de l'action est la SOMME des
     log-softmax conditionnels, renormalisés après chaque retrait. C'est la
     seule log-prob bien définie d'un top-k échantillonné : un top-k « dur »
     n'est pas une distribution sur des ensembles qu'on sache normaliser, un
     tirage ordonné si (voir `pl_logp`, prouvé par énumération exhaustive).
     À l'ÉVAL, argmax séquentiel = `rti.pick_eval_groups` (tie-break de récence
     compris) — le chemin d'inférence du SFT, à l'identique.
  3. DÉCODAGE — token-level, log-probs rendues par `decode.generate`
     (`return_logp=True`).

LE MASQUAGE (trick Search-R1, « la différence entre convergence et collapse »)
─────────────────────────────────────────────────────────────────────────────
Les P pseudo-tokens injectés ne sont l'action de PERSONNE : ils ne sont pas
échantillonnés, ils sont construits. Aucune de leurs positions ne doit entrer
dans les log-probs collectées. Ici c'est vrai par CONSTRUCTION — `generate` ne
rend que les tokens produits après le prompt, et `model.forward` retranche déjà
le préfixe de `H_text` (Step 1bis) — mais « vrai par construction » se casse
silencieusement au premier refactor : `replay_logp` REJOUE la même séquence et
le self-test exige l'égalité au 1e-6 (float64). Un décalage de préfixe s'y voit.

Ce rejeu n'est légitime que parce que le forward du bras rti est PRÉFIXE-CAUSAL
— et il ne l'est que sous `layer_banks = null_banks(model)` : voir la note de
cette fonction, c'est le piège le plus coûteux du chemin (0.25 sur les logits,
et un forward non déterministe, quand on l'oublie).

LA VIE, ET POURQUOI ELLE NE PORTE PAS DE BANQUE
───────────────────────────────────────────────
Sous `recall_env`, une VIE = UN script (`conv_for_life(life)`), c'est-à-dire
une longue session multi-tours qui contient déjà ses propres « convs » (les
tours de remplissage réels). La FIFO de groupes naît et meurt avec le script :
elle n'est PAS resetée entre les tours (l'invariant no-reset), elle l'est entre
deux vies. Ce n'est pas un raccourci — le générateur calcule `possible` en
supposant une banque VIDE au début du script (`age_writes` compté depuis le
premier write de la vie) ; charrier des groupes d'une vie à l'autre évincerait
plus vite que la politique de référence et déclarerait possibles des sondes qui
ne le sont plus. La conséquence pratique est qu'aucun `LivesState` (banque
tensorielle) n'intervient sur ce chemin.

LE LOCKSTEP DES G ROLLOUTS (et pourquoi il est EXACT, pas une approximation)
───────────────────────────────────────────────────────────────────────────
Sous rti, le forward d'un seg ne dépend PAS de la banque : `layer_banks` est
tout à None et `write=False` — le seul canal mémoire est le préfixe injecté.
Deux rollouts du même script voient donc EXACTEMENT le même état caché sur tout
seg non préfixé, quelles qu'aient été leurs actions passées. On mutualise ce
forward (B=1) pour les G rollouts, et on ne paie séparément que ce qui diffère
vraiment : le décodage des sondes (préfixes différents), batché par longueur de
préfixe. Ce n'est pas une approximation, c'est une factorisation.

Self-test CPU (vrai ThoughtBankLM minuscule, vrai `generate`) :
    python -m deepseek_v4_mini.rti_policy
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .decode import generate
from .rti import (LaneBank, RtiConfig, build_group, build_prefix,
                  pick_eval_groups, sif_table)


# ── config ───────────────────────────────────────────────────────────────────

@dataclass
class RtiPolicyConfig:
    """Les knobs des ACTIONS. `RtiConfig` garde ceux du MÉCANISME."""
    write_mode: str = "head"    # "head" = Bernoulli apprise ; "facts" = le
                                # déclenchement procédural du SFT (baseline).
    write_temp: float = 1.0     # température de la Bernoulli (sur le logit).
    write_floor: float = 0.0    # plancher d'EXPLORATION sur la probabilité de
                                # COMPORTEMENT (p_b = max(p, floor)). La log-prob
                                # enregistrée est celle du comportement : le
                                # ratio de la passe 2 corrige off-policy. Même
                                # sémantique que `rl.explore_floor` du bras
                                # historique (pré-vol 07-27 : sans plancher, la
                                # banque des rollouts reste vide).
    write_on: str = "all"       # "all" = tout seg est candidat ; "user" = seuls
                                # les tours utilisateur (économise le forward de
                                # la réponse décodée).
    retr_temp: float = 1.0      # température du Plackett-Luce.
    decode_temp: float = 1.0    # température du décodage (0 = greedy, log-probs
                                # dégénérées : réservé à l'éval).
    decode_top_p: float = 1.0
    greedy: bool = False        # True = argmax séquentiel + décodage greedy
                                # (chemin d'ÉVAL, identique au SFT).

    def __post_init__(self):
        assert self.write_mode in ("head", "facts"), self.write_mode
        assert self.write_on in ("all", "user"), self.write_on
        assert self.write_temp > 0 and self.retr_temp > 0
        assert 0.0 <= self.write_floor < 1.0


def rti_from_raw(raw: dict):
    """(RtiConfig, RtiPolicyConfig, enabled) depuis la section `rti:` du YAML.

    Une seule lecture de la section pour le worker ET le learner : les deux
    doivent construire le MÊME modèle (mêmes modules attachés), sinon le
    `state_dict` publié ne se recharge pas.
    """
    rt = dict(raw.get("rti") or {})
    on = bool(rt.pop("enabled", False))
    pol = {k: rt.pop(k) for k in
           ("write_mode", "write_temp", "write_floor", "write_on", "retr_temp",
            "decode_temp", "decode_top_p", "greedy") if k in rt}
    cfg = RtiConfig(**rt) if on else None
    return cfg, RtiPolicyConfig(**pol), on


# ── 1. WRITE : la Bernoulli par tour ─────────────────────────────────────────

class WriteHead(nn.Module):
    """p(write | h) = σ(w·h + b), zéro-init ⇒ p = 0.5 EXACTEMENT au step 0.

    Zéro-init et non petit-aléatoire : le gradient de la Bernoulli vaut
    (a − p)·h ≠ 0, la tête démarre donc indifférente et non morte, et surtout
    la politique de départ est IDENTIQUE pour tous les workers et reproductible
    — un tirage d'init changerait le taux de write initial, donc la densité de
    la banque, donc la difficulté de toutes les sondes du premier millier de
    groupes.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.w = nn.Linear(d_model, 1)
        nn.init.zeros_(self.w.weight)
        nn.init.zeros_(self.w.bias)

    def forward(self, h):                      # h [..., d] -> logit [...]
        return self.w(h.to(self.w.weight.dtype)).squeeze(-1)


def write_logp(logit, a: int, temp: float = 1.0):
    """log p de l'action Bernoulli, DIFFÉRENTIABLE (tenseur en entrée/sortie).

    La température agit sur le LOGIT (pas sur la probabilité) : c'est la même
    convention que `rl_defer_grpo_lives.boundary_step`, où temp>1 aplatit la
    Bernoulli vers 0.5 sans jamais la saturer.
    """
    z = logit / float(temp)
    return F.logsigmoid(z) if a else F.logsigmoid(-z)


def write_sample(logit, rng, temp: float = 1.0, floor: float = 0.0,
                 forced: int | None = None):
    """(a, logp_comportement, p, p_b). `forced` court-circuite le tirage (mode
    procédural `facts`) mais garde la log-prob de la politique — le crédit
    reste défini même quand l'action ne vient pas d'elle."""
    p = float(torch.sigmoid(logit.detach().float() / float(temp)))
    p_b = max(p, float(floor))
    p_b = min(max(p_b, 1e-6), 1.0 - 1e-6)
    a = int(forced) if forced is not None else int(rng.random() < p_b)
    lp = math.log(p_b) if a else math.log1p(-p_b)
    return a, lp, p, p_b


# ── 2. RETRIEVE : Plackett-Luce ──────────────────────────────────────────────

def pl_logp(scores: torch.Tensor, order, temp: float = 1.0,
            valid: int | None = None):
    """log P(order) sous Plackett-Luce, DIFFÉRENTIABLE.

        P(i1..ik) = Π_t softmax(s / τ)[i_t] restreint aux candidats RESTANTS.

    C'est exactement la log-prob du tirage séquentiel sans remise de
    `pl_sample` : le learner la recalcule sur les scores de la passe 2 et le
    ratio GRPO est bien défini. Un top-k dur n'aurait pas d'équivalent — la
    probabilité d'un ENSEMBLE sous tirage sans remise n'a pas de forme close.
    """
    s = scores.float().reshape(-1) / float(temp)
    n = s.numel()
    if valid is not None and valid < n:
        s = s.masked_fill(torch.arange(n, device=s.device) >= int(valid),
                          float("-inf"))
    out = s.new_zeros(())
    mask = torch.zeros(n, dtype=torch.bool, device=s.device)
    for i in order:
        av = s.masked_fill(mask, float("-inf"))
        out = out + av[int(i)] - torch.logsumexp(av, -1)
        # HORS PLACE, et ce n'est pas un détail : `masked_fill` garde le masque
        # pour son backward. Le muter en place ferait échouer le backward du
        # learner (« modified by an inplace operation », version du tenseur) —
        # or c'est précisément ici que passe le gradient du retriever.
        mask = mask.clone()
        mask[int(i)] = True
    return out


def pl_sample(scores: torch.Tensor, k: int, *, temp: float = 1.0,
              generator=None, greedy: bool = False, valid: int | None = None):
    """(order, logp) — tirage séquentiel SANS REMISE de k groupes sur n.

    L'ORDRE EST L'ACTION : le préfixe est injecté dans l'ordre du tirage, et le
    circuit de copie est câblé sur le LAYOUT du préfixe (leçon r5 v1, grade
    0.067 malgré un retriever parfait). Deux tirages du même ENSEMBLE dans deux
    ordres différents sont deux actions différentes, et c'est correct.

    `greedy` : argmax séquentiel avec tie-break de RÉCENCE (indice le plus
    grand d'abord), donc rigoureusement `rti.pick_eval_groups` — la sélection
    d'inférence du SFT. Prouvé par le self-test (b).
    """
    s = scores.detach().float().reshape(-1)
    n = s.numel() if valid is None else min(int(valid), s.numel())
    k = min(int(k), n)
    logp = s.new_zeros(())
    if k <= 0:
        return [], logp
    z = s[:n] / float(temp)
    mask = torch.zeros(n, dtype=torch.bool, device=s.device)
    order: list = []
    for _ in range(k):
        av = z.masked_fill(mask, float("-inf"))
        if greedy:
            # récence prioritaire à score égal : on balaie du PLUS RÉCENT au
            # plus ancien et `max` garde le premier rencontré.
            i = max((j for j in range(n - 1, -1, -1) if not bool(mask[j])),
                    key=lambda j: float(s[j]))
        else:
            i = int(torch.multinomial(torch.softmax(av, -1), 1,
                                      generator=generator))
        logp = logp + av[i] - torch.logsumexp(av, -1)
        order.append(int(i))
        mask[i] = True
    return order, logp


# ── 3. DÉCODAGE : rejeu des log-probs (et la preuve du masquage) ─────────────

def null_banks(model) -> list:
    """`layer_banks` du bras rti : AUCUNE couche ne lit la banque fast-weight.

    Ce n'est pas cosmétique. `layer_banks=None` fait lire à chaque couche la
    banque `init_mem`, et `init_mem=None` en fait SEMER une neuve — or
    `seed_bank` TIRE AU HASARD (mem_seed_slots). Le forward cesse alors d'être
    déterministe et, pire, cesse d'être PRÉFIXE-CAUSAL : les logits d'une
    position dépendent de la longueur de la séquence (mesuré : écart 0.25 sur
    les logits entre un forward de longueur 12 et un de longueur 20). Le rejeu
    teacher-forcé des log-probs de décodage serait alors faux, donc le ratio du
    learner aussi. Avec cette liste, l'écart tombe à 5.6e-17 (float64) et le
    forward est en prime invariant au batch. C'est ce que fait déjà
    `evaluate_math` (rti_lb) et `_chat_loss_rti` côté trainer.
    """
    return [None] * len(model.blocks)


def replay_logp(model, prompt, tokens, *, inject=None, temp: float = 1.0,
                amp: bool = False):
    """log-probs TEACHER-FORCÉES des `tokens` produits après `prompt`.

    [B,Tp] + [B,Tg] -> [B,Tg]. Le préfixe injecté (P pseudo-tokens) entre par
    `inject` et `model.forward` le retranche de `H_text` : les positions
    rendues sont celles du TOUR RÉEL, donc l'indice du premier token généré est
    `Tp - 1` et PAS `P + Tp - 1`. C'est ce décalage que le self-test verrouille
    en comparant au log-prob collecté pendant le rollout.
    """
    x = torch.cat([prompt, tokens], dim=1)
    with torch.autocast("cuda", dtype=torch.bfloat16,
                        enabled=amp and x.is_cuda):
        o = model(x, init_mem=None, layer_banks=null_banks(model), write=False,
                  **({"inject": inject} if inject is not None else {}))
    lg = o["logits"].float()[:, prompt.size(1) - 1:-1] / float(temp)
    return torch.log_softmax(lg, -1).gather(-1, tokens.unsqueeze(-1)).squeeze(-1)


# ── l'état d'UN rollout ──────────────────────────────────────────────────────

class _Roll:
    """La FIFO d'un rollout + sa trace. `gid` = identité STABLE d'un groupe
    (monotone par rollout) : c'est la clé du crédit leave-one-slot, un indice
    de banque ne survivrait pas à la première éviction."""

    def __init__(self):
        self.bank = LaneBank()
        self.gids: list = []
        self.next_gid = 0
        self.pending = None
        self.turns: list = []
        self.n_writes = 0

    def push(self, key, tok, tag, max_groups: int) -> int:
        gid = self.next_gid
        self.next_gid += 1
        self.bank.push(key, tok, tag, max_groups)
        self.gids.append(gid)
        del self.gids[:len(self.gids) - len(self.bank)]
        self.n_writes += 1
        return gid


# ── LE PILOTE : une vie RTI, G rollouts en lockstep ──────────────────────────

class RtiRollout:
    """Déroule UN script pour G rollouts appariés et rend G traces.

    Ordre par seg, IDENTIQUE à l'invariant du trainer SFT :
        1. le préfixe DÛ à ce seg (décidé au seg PRÉCÉDENT, qui posait la
           question) — un seg ne se cite JAMAIS lui-même ;
        2. le forward / le décodage ;
        3. le WRITE de ce seg ;
        4. la QUERY de ce seg (due au seg suivant).
    """

    def __init__(self, model, tok, cfg: RtiConfig, pcfg: RtiPolicyConfig,
                 sif_w, sep_id: int, *, a_open, stop_id: int, max_new: int = 96,
                 amp: bool = False, decode_cache: bool = False):
        self.model, self.tok = model, tok
        self.cfg, self.pcfg = cfg, pcfg
        self.sif_w, self.sep_id = sif_w, int(sep_id)
        self.a_open, self.stop_id = a_open, int(stop_id)
        self.max_new, self.amp = int(max_new), bool(amp)
        self.decode_cache = bool(decode_cache)
        self.lb = null_banks(model)
        assert cfg.train_groups == cfg.eval_groups, \
            "isomorphie train/éval : train_groups doit valoir eval_groups"

    # ── forwards mutualisés ─────────────────────────────────────────────────
    @torch.no_grad()
    def _hidden(self, x):
        """h [1,T,d] d'un seg NU (aucun préfixe) — partagé par les G rollouts
        (cf. l'en-tête : sous rti le forward ne dépend pas de la banque)."""
        with torch.autocast("cuda", dtype=torch.bfloat16,
                            enabled=self.amp and x.is_cuda):
            o = self.model(x, init_mem=None, layer_banks=self.lb, write=False,
                           compute_logits=False)
        return o["hidden"].float()

    @torch.no_grad()
    def _prefix(self, rolls, sel):
        """[(rows, inject|None)] — partition des rollouts par LONGUEUR de
        préfixe. Une tranche de séquence ne peut pas varier dans un batch
        (crash payé au job 58) ; et l'isomorphie STRICTE veut qu'une banque
        trop courte donne AUCUNE injection plutôt qu'un préfixe d'une autre
        forme."""
        want = self.cfg.eval_groups
        buckets: dict = {}
        for g, order in enumerate(sel):
            ok = order is not None and len(order) == want
            buckets.setdefault(bool(ok), []).append(g)
        out = []
        for ok, rows in sorted(buckets.items()):
            if not ok:
                out.append((rows, None))
                continue
            toks = [[rolls[g].bank.toks[i] for i in sel[g]] for g in rows]
            out.append((rows, build_prefix(self.model.embed.weight,
                                           self.model.rti_type.vec,
                                           self.sep_id, toks, self.cfg)))
        return out

    # ── la boucle ────────────────────────────────────────────────────────────
    @torch.no_grad()
    def run(self, conv: dict, G: int, rng, *, reward_fn=None, device=None,
            generator=None) -> list:
        cfg, pcfg = self.cfg, self.pcfg
        script = conv["info"]["script"]
        assert int(script["layout"]["inject_groups"]) == cfg.eval_groups, \
            (f"layout du script {script['layout']['inject_groups']} ≠ "
             f"rti.eval_groups {cfg.eval_groups} — le circuit de copie est "
             f"câblé sur le LAYOUT du préfixe (leçon r5 v1)")
        segs = conv["segs"]
        dev = device or self.model.embed.weight.device
        seg2turn = {}
        for t in script["turns"]:
            seg2turn[t["user_seg"]] = (t["turn"], t["kind"])
            if t.get("assistant_seg") is not None:
                seg2turn[t["assistant_seg"]] = (t["turn"], t["kind"])
        probes = script["probes"]
        rolls = [_Roll() for _ in range(G)]

        for i, s in enumerate(segs):
            x = s["input_ids"].to(dev)
            role = s["role"]
            turn, kind = seg2turn.get(i, (None, "?"))
            rec = [{"seg": i, "turn": turn, "kind": kind, "role": role,
                    "write": None, "retrieve": None, "decode": None,
                    "reward": None} for _ in range(G)]

            # 1. le préfixe DÛ à ce seg (consommé, comme `RtiRunner.parts`)
            sel = [None] * G
            if role == "assistant":
                for g, r in enumerate(rolls):
                    if r.pending is not None:
                        sel[g] = r.pending["order"]
            for r in rolls:
                r.pending = None

            # 2. décodage (sondes) ou forward nu (tout le reste)
            hid = None
            wid = [x] * G                      # ids servant au write de ce seg
            if s.get("decode"):
                pid = int(s["probe_id"])
                texts, toks, lps, npre = self._decode(rolls, sel, G, dev)
                for g in range(G):
                    rec[g]["decode"] = {
                        "probe_id": pid, "tokens": toks[g], "logp": lps[g],
                        "n_prefix": npre[g], "n_tokens": len(toks[g]),
                        "text": texts[g]}
                    if reward_fn is not None:
                        info = {"text": texts[g], "probe": probes[pid],
                                "n_think": rolls[g].n_writes}
                        rw = reward_fn(None, info)
                        rec[g]["reward"] = None if rw is None else float(rw)
                        rec[g]["raw"] = info.get("raw")
                        rec[g]["possible"] = info.get("possible")
                # le write d'une réponse porte sur le texte RÉELLEMENT produit
                # (on-policy), pas sur le gold du script.
                wid = [torch.tensor([toks[g] or [self.sep_id]], dtype=torch.long,
                                    device=dev) for g in range(G)]
                if pcfg.write_on == "all":
                    hid = self._hidden_batch(wid)
            else:
                hid = self._hidden(x)

            # 3. WRITE
            cand = pcfg.write_on == "all" or role == "user"
            fs = s.get("fact_slot")
            tag = int(fs.max()) if fs is not None else 0
            tag = tag if tag > 0 else None
            for g, r in enumerate(rolls):
                if not cand:
                    continue
                if pcfg.write_mode == "facts":
                    # baseline : le déclenchement PROCÉDURAL du SFT
                    # (`rti.write_every_turn: false`), bit à bit.
                    a = int(tag is not None or cfg.write_every_turn)
                    lp, p, p_b = 0.0, float(a), float(a)
                    if hid is not None:
                        lg = self.model.rti_write(hid[g if hid.size(0) > 1 else 0, -1])
                        a2, lp, p, p_b = write_sample(lg, rng, pcfg.write_temp,
                                                      pcfg.write_floor, forced=a)
                else:
                    h = hid[g if hid.size(0) > 1 else 0, -1]
                    lg = self.model.rti_write(h)
                    a, lp, p, p_b = write_sample(lg, rng, pcfg.write_temp,
                                                 pcfg.write_floor)
                gid = None
                if a:
                    row = wid[g].reshape(-1)
                    if row.numel():
                        key, tk = build_group(self.model.embed.weight,
                                              self.sif_w, row, cfg)
                        gid = r.push(key, tk, tag, cfg.max_groups)
                rec[g]["write"] = {"a": int(a), "logp": float(lp),
                                   "p": float(p), "p_b": float(p_b),
                                   "tag": tag, "gid": gid}

            # 4. QUERY (due au seg SUIVANT)
            qs = s.get("q_slot")
            if qs is not None and int(qs.max()) > 0 and hid is not None:
                self._query(rolls, rec, int(qs.max()), hid[0, -1], G, generator)

            for g in range(G):
                rec[g]["fifo"] = list(rolls[g].gids)
                rec[g]["fifo_tags"] = list(rolls[g].bank.tags)
                rolls[g].turns.append(rec[g])

        return [self._trace(conv, script, r) for r in rolls]

    # ── sous-étapes ─────────────────────────────────────────────────────────
    @torch.no_grad()
    def _hidden_batch(self, ids: list):
        """h des réponses décodées (longueurs différentes ⇒ right-padding, et
        on ne relit que la DERNIÈRE position utile de chaque ligne)."""
        L = max(int(t.size(1)) for t in ids)
        dev = ids[0].device
        x = torch.full((len(ids), L), self.sep_id, dtype=torch.long, device=dev)
        for g, t in enumerate(ids):
            x[g, :t.size(1)] = t[0]
        with torch.autocast("cuda", dtype=torch.bfloat16,
                            enabled=self.amp and x.is_cuda):
            o = self.model(x, init_mem=None, layer_banks=self.lb, write=False,
                           compute_logits=False)
        h = o["hidden"].float()
        out = h.new_zeros(len(ids), 1, h.size(-1))
        for g, t in enumerate(ids):
            out[g, 0] = h[g, max(int(t.size(1)) - 1, 0)]
        return out

    @torch.no_grad()
    def _query(self, rolls, rec, tag, h, G, generator):
        """Scores du retriever + TIRAGE Plackett-Luce, un par rollout.

        `h` est le caché du DERNIER token du seg USER — jamais un token de la
        réponse (invariant du bras : en teacher-forcing celui-ci contient déjà
        la valeur à retrouver).
        """
        cfg, pcfg = self.cfg, self.pcfg
        Gm = cfg.max_groups
        rows, keys, msk = [], [], []
        for g, r in enumerate(rolls):
            if len(r.bank) == 0:
                continue
            kb = torch.stack(r.bank.keys)
            pad = kb.new_zeros(Gm, kb.size(1))
            pad[:kb.size(0)] = kb
            m = torch.zeros(Gm, dtype=torch.bool, device=kb.device)
            m[:kb.size(0)] = True
            rows.append(g); keys.append(pad); msk.append(m)
        if not rows:
            return
        H = h.reshape(1, -1).to(self.model.rti_retriever.wq.weight.dtype) \
             .expand(len(rows), -1)
        sc = self.model.rti_retriever(H, torch.stack(keys),
                                      torch.stack(msk)).float()
        for j, g in enumerate(rows):
            r = rolls[g]
            n_res = len(r.bank)
            order, lp = pl_sample(sc[j, :n_res], cfg.eval_groups,
                                  temp=pcfg.retr_temp, generator=generator,
                                  greedy=pcfg.greedy)
            pos = r.bank.positives(tag)
            r.pending = {"order": order, "tag": tag}
            rec[g]["retrieve"] = {
                "tag": tag, "order": order,
                "gids": [r.gids[i] for i in order],
                "logp": float(lp), "temp": pcfg.retr_temp,
                "greedy": bool(pcfg.greedy),
                "scores": [float(v) for v in sc[j, :n_res]],
                "cand_gids": list(r.gids),
                # de quoi REJOUER la sélection sans le modèle (leave-one-slot) :
                "cand_toks": [t.tolist() for t in r.bank.toks],
                "pos": pos, "hit": bool(set(order) & set(pos)),
                "top1": bool(order and order[0] in pos)}

    @torch.no_grad()
    def _decode(self, rolls, sel, G, dev):
        """Décodage des G réponses, batché PAR LONGUEUR DE PRÉFIXE."""
        pcfg = self.pcfg
        texts = [""] * G
        toks: list = [[] for _ in range(G)]
        lps: list = [[] for _ in range(G)]
        npre = [0] * G
        prompt = self.a_open.to(dev)
        for rows, inj in self._prefix(rolls, sel):
            n = len(rows)
            gen, lens, lp = generate(
                self.model, prompt.expand(n, -1), bank=None, layer_banks=self.lb,
                max_new=self.max_new, stop_id=self.stop_id, amp=self.amp,
                temp=pcfg.decode_temp, top_p=pcfg.decode_top_p,
                use_cache=self.decode_cache, inject=inj, return_logp=True)
            for j, g in enumerate(rows):
                L = int(lens[j])
                toks[g] = gen[j, :L].tolist()
                lps[g] = [float(v) for v in lp[j, :L]]
                npre[g] = 0 if inj is None else int(inj.size(1))
                texts[g] = self.tok.decode(toks[g])
        return texts, toks, lps, npre

    def _trace(self, conv, script, r: _Roll) -> dict:
        info = conv["info"]
        rw = [t["reward"] for t in r.turns if t["decode"] is not None]
        ok = [v for v in rw if v is not None]
        # fait -> writes qui l'ont porté (le crédit contrefactuel de l'agent 3)
        f2w: dict = {}
        for t in r.turns:
            w = t["write"]
            if w and w["a"] and w["tag"] is not None:
                f2w.setdefault(str(w["tag"]), []).append(w["gid"])
        return {"life": info["life"], "digest": info["digest"],
                "layout": info["layout"], "turns": r.turns,
                "n_writes": r.n_writes,
                "reward": (sum(ok) / len(ok)) if ok else None,
                "n_probes": len(script["probes"]), "n_scored": len(ok),
                "slot_to_writes": f2w,
                "probe_to_slot": {str(p["pid"]): p["slot_id"]
                                  for p in script["probes"]},
                "probe_possible": {str(p["pid"]): bool(p["possible"])
                                   for p in script["probes"]}}


# ── crédit contrefactuel : leave-one-slot ────────────────────────────────────

def forward_probe_with_masked_slot(model, turn: dict, mask_gid, cfg: RtiConfig,
                                   sep_id: int, prompt, *, temp: float = 1.0,
                                   amp: bool = False, device=None):
    """Rejoue UNE sonde en RETIRANT un groupe de la sélection.

    `turn` = un tour de trace portant `retrieve` (le tour de QUESTION) et le
    `decode` de la réponse — en pratique on passe le dict fusionné
    {**turn_question["retrieve"], "decode": turn_reponse["decode"]}, ou plus
    simplement les deux via `probe_pair`.

    Le slot n'est pas SUPPRIMÉ du préfixe : son score est mis à −inf et la
    sélection est REJOUÉE en argmax séquentiel, donc le préfixe garde
    EXACTEMENT sa longueur. Retirer une ligne changerait le layout, et le
    circuit de copie est câblé dessus (r5 v1) : la contrefactuelle mesurerait
    alors le changement de layout, pas la perte du fait.

    Rend {"order", "gids", "dropped", "logp", "logps"} — `logp` = somme des
    log-probs TEACHER-FORCÉES des tokens que le rollout a réellement produits.
    Δ = logp(sans le slot) − logp(rollout) est le crédit du slot.
    """
    ret, dec = turn["retrieve"], turn["decode"]
    dev = device or model.embed.weight.device
    sc = torch.tensor(ret["scores"], dtype=torch.float32, device=dev)
    gids = ret["cand_gids"]
    drop = [i for i, g in enumerate(gids) if g == mask_gid]
    if drop:
        sc = sc.clone()
        sc[drop[0]] = float("-inf")
    order, _ = pl_sample(sc, cfg.eval_groups, temp=1.0, greedy=True)
    toks = torch.tensor([ret["cand_toks"][i] for i in order],
                        dtype=torch.long, device=dev)
    inj = None
    if len(order) == cfg.eval_groups:
        inj = build_prefix(model.embed.weight, model.rti_type.vec, sep_id,
                           [[toks[j] for j in range(toks.size(0))]], cfg)
    gen = torch.tensor([dec["tokens"]], dtype=torch.long, device=dev)
    if gen.numel() == 0:
        return {"order": order, "gids": [gids[i] for i in order],
                "dropped": bool(drop), "logp": 0.0, "logps": []}
    lp = replay_logp(model, prompt.to(dev), gen, inject=inj, temp=temp, amp=amp)
    return {"order": order, "gids": [gids[i] for i in order],
            "dropped": bool(drop), "logp": float(lp.sum()),
            "logps": [float(v) for v in lp[0]]}


def probe_pairs(trace: dict) -> list:
    """[(tour de QUESTION, tour de RÉPONSE)] d'une trace — le tour de question
    porte `retrieve`, la réponse porte `decode` et le reward. Le préfixe d'une
    réponse est décidé au seg PRÉCÉDENT : l'appariement est donc « le dernier
    tour portant `retrieve` avant ce `decode` »."""
    out, last = [], None
    for t in trace["turns"]:
        if t.get("retrieve") is not None:
            last = t
        if t.get("decode") is not None:
            out.append((last, t))
            last = None
    return out


# ── self-test ────────────────────────────────────────────────────────────────

def _selftest() -> None:
    """Ce qui peut casser en silence ici : une log-prob de top-k qui ne somme
    pas à 1 (le ratio GRPO devient du bruit), un argmax séquentiel qui n'est
    pas le top-k du SFT (train/éval désynchronisés), une position de préfixe
    qui fuit dans les log-probs (collapse, trick Search-R1), une ancre de tour
    qui glisse entre deux rollouts appariés (l'avantage cesse d'être
    contrefactuel), et un mode `facts` qui ne reproduit plus le write du SFT.
    """
    import itertools
    import random as _random

    torch.manual_seed(20260731)

    # ── (a) Plackett-Luce : la log-prob EST une distribution ────────────────
    n, k = 4, 2
    sc = torch.randn(n) * 1.7
    tot = 0.0
    for perm in itertools.permutations(range(n), k):
        lp = float(pl_logp(sc, list(perm)))
        # produit des conditionnels, calculé À LA MAIN
        man, left = 0.0, list(range(n))
        for i in perm:
            z = torch.tensor([float(sc[j]) for j in left])
            man += float(torch.log_softmax(z, -1)[left.index(i)])
            left.remove(i)
        assert abs(lp - man) < 1e-5, (perm, lp, man)
        tot += math.exp(lp)
    assert abs(tot - 1.0) < 1e-5, tot
    # k = n : la somme sur toutes les PERMUTATIONS vaut aussi 1
    tot_n = sum(math.exp(float(pl_logp(sc, list(p))))
                for p in itertools.permutations(range(n)))
    assert abs(tot_n - 1.0) < 1e-5, tot_n
    # la température aplatit : entropie croissante
    def _ent(t):
        return -sum(math.exp(float(pl_logp(sc, list(p), t)))
                    * float(pl_logp(sc, list(p), t))
                    for p in itertools.permutations(range(n), k))
    assert _ent(4.0) > _ent(1.0) > _ent(0.25), (_ent(0.25), _ent(1.0), _ent(4.0))
    # pl_sample et pl_logp s'accordent, et l'échantillonnage suit la loi
    gen = torch.Generator().manual_seed(0)
    hits: dict = {}
    N = 20000
    for _ in range(N):
        o, lp = pl_sample(sc, k, temp=1.0, generator=gen)
        assert abs(float(lp) - float(pl_logp(sc, o))) < 1e-5
        hits[tuple(o)] = hits.get(tuple(o), 0) + 1
    for perm, c in hits.items():
        assert abs(c / N - math.exp(float(pl_logp(sc, list(perm))))) < 0.02, perm
    print(f"  [a] Plackett-Luce : Σ p(tirages ordonnés) = {tot:.6f} (k=2) / "
          f"{tot_n:.6f} (k=n), log-prob == produit des conditionnels, "
          f"empirique à ±0.02 sur {N} tirages")

    # ── (b) température → 0 : le tirage EST le top-k du SFT ─────────────────
    cfg = RtiConfig(top_k=4, max_groups=8, train_groups=2, eval_groups=2)
    for trial in range(200):
        m = int(torch.randint(2, 9, (1,)))
        s = torch.randn(m)
        if trial % 3 == 0:                       # ex æquo : le cas qui piège
            s = torch.zeros(m) if trial % 6 == 0 else s.round()
        ref = pick_eval_groups(s, m, cfg)
        assert pl_sample(s, cfg.eval_groups, greedy=True)[0] == ref, (s, ref)
        if len(set(s.tolist())) == m:            # scores distincts : τ→0 == argmax
            o, _ = pl_sample(s, cfg.eval_groups, temp=1e-3,
                             generator=torch.Generator().manual_seed(trial))
            assert o == ref, (s, o, ref)
    print("  [b] τ→0 : argmax séquentiel == pick_eval_groups (récence "
          "prioritaire à égalité) sur 200 cas, dont ex æquo")

    # ── modèle jouet RÉEL (vrai forward, vrai inject, vrai generate) ────────
    from .config import ThoughtBankConfig
    from .model import ThoughtBankLM
    from .recall_env import RecallEnvConfig, RecallEnvStream, make_recall_env_reward
    from . import persona_chat_data as P

    tok = P._StubTok()                     # ids = ord(c) : tout tient sous 512
    V = 512
    # float64 + AUCUN top-k dur (top_k_experts == n_experts, top_k_csa large) :
    # même précaution que `decode._selftest`. Sans elle, le rejeu du (c) ne
    # mesurerait pas le masquage mais la sensibilité chaotique du routage MoE
    # aux ULP (mémoire dsv6-topk-dur-amplifie-les-ulp) — mesuré ici : écart de
    # log-prob 1e-1, du même ordre que ce qu'un décalage de préfixe produirait.
    mcfg = ThoughtBankConfig(vocab_size=V, d_model=32, n_layers=2, n_heads=2,
                             d_head=8, max_seq_len=512, n_hc=2, sinkhorn_iters=5,
                             csa_m=3, hca_m=5, top_k_csa=64, n_win=4,
                             d_latent_q=16, n_groups=1, n_experts=2, n_shared=1,
                             top_k_experts=2, d_ff=64, mem_dim=16, max_mem=4,
                             mem_seed_slots=2, use_dual_stream=True)
    model = ThoughtBankLM(mcfg).double().eval()
    attach_rti_modules(model, 32)
    model.double()
    with torch.no_grad():                        # zéro-init : rien ne bouge
        model.rti_retriever.wq.weight.normal_(0, 0.1)
        model.rti_type.vec.normal_(0, 0.02)
        model.rti_write.w.weight.normal_(0, 0.5)
    for p in model.parameters():
        p.requires_grad_(False)

    # strates sans `code` : le juge de la strate code EXÉCUTE du python en bac
    # à sable, ce qui n'apporte rien ici et coûte un fork par sonde.
    rcfg = RecallEnvConfig(life_seed=7, n_facts=(3, 3), n_probes=(2, 2),
                           strata={"persona": 0.4, "numeric": 0.3,
                                   "preference": 0.3},
                           filler_per_fact=(1, 1), p_beyond=0.0,
                           inject_groups=2, max_groups=4,
                           age_bins=((2, 4),), age_weights=(1.0,),
                           surprisal_mode="nll")
    stream = RecallEnvStream(tok, seed=1, cfg=rcfg)
    conv = stream.conv_for_life(3)
    cfg = RtiConfig(top_k=4, max_groups=4, train_groups=2, eval_groups=2)
    sif_w = sif_table(stream, V, cfg.sif_a)
    a_open = torch.tensor([[7, 8]], dtype=torch.long)

    def _mk(pcfg, G=4, seed=0):
        r = RtiRollout(model, tok, cfg, pcfg, sif_w, sep_id=9, a_open=a_open,
                       stop_id=-1, max_new=6, amp=False)
        return r, r.run(conv, G, _random.Random(seed),
                        reward_fn=make_recall_env_reward(),
                        generator=torch.Generator().manual_seed(seed))

    # ── (c) MASQUAGE : aucune position de préfixe dans les log-probs ────────
    pol = RtiPolicyConfig(write_mode="head", write_floor=0.9, retr_temp=1.0,
                          decode_temp=1.0)
    runner, traces = _mk(pol, G=4, seed=1)
    P_len = cfg.eval_groups * cfg.group_prefix
    n_inj = n_dec = 0
    for tr in traces:
        for q, d in probe_pairs(tr):
            dd = d["decode"]
            n_dec += 1
            assert len(dd["logp"]) == len(dd["tokens"]) == dd["n_tokens"], dd
            assert dd["n_prefix"] in (0, P_len), dd["n_prefix"]
            if dd["n_prefix"]:
                n_inj += 1
                assert q is not None and len(q["retrieve"]["order"]) == 2
            # REJEU : les mêmes log-probs, préfixe compris, au 1e-4 près.
            # Un décalage d'indice (préfixe compté dans les positions) ferait
            # exploser cet écart — c'est LA preuve du masquage.
            inj = None
            if dd["n_prefix"]:
                tk = torch.tensor([q["retrieve"]["cand_toks"][i]
                                   for i in q["retrieve"]["order"]],
                                  dtype=torch.long)
                inj = build_prefix(model.embed.weight, model.rti_type.vec, 9,
                                   [[tk[j] for j in range(tk.size(0))]], cfg)
            if dd["tokens"]:
                rp = replay_logp(model, a_open,
                                 torch.tensor([dd["tokens"]], dtype=torch.long),
                                 inject=inj, temp=pol.decode_temp)
                err = max(abs(float(a) - b) for a, b in zip(rp[0], dd["logp"]))
                assert err < 1e-6, (err, dd["logp"], rp.tolist())
    assert n_inj > 0, "aucune sonde n'a reçu de préfixe : le test ne prouve rien"
    print(f"  [c] masquage : {n_dec} sondes, {n_inj} avec préfixe P={P_len} ; "
          f"len(logp) == len(tokens) partout et le rejeu teacher-forcé "
          f"recolle au 1e-6 en float64 (décalage de préfixe exclu)")

    # ── (d) TRACE : ancres de tour alignées entre rollouts appariés ─────────
    anchors = [[(t["seg"], t["turn"], t["kind"], t["role"]) for t in tr["turns"]]
               for tr in traces]
    assert all(a == anchors[0] for a in anchors), "ancres de tour désalignées"
    assert len({tr["digest"] for tr in traces}) == 1, "rollouts non appariés"
    assert all(tr["life"] == conv["info"]["life"] for tr in traces)
    # les actions, elles, DOIVENT différer (sinon pas de variance intra-groupe)
    acts = [tuple(t["write"]["a"] for t in tr["turns"] if t["write"])
            for tr in traces]
    assert len(set(acts)) > 1, "tous les rollouts ont écrit pareil"
    # cohérence FIFO : |fifo| <= max_groups, monotone par write, gids uniques
    for tr in traces:
        seen = set()
        for t in tr["turns"]:
            assert len(t["fifo"]) <= cfg.max_groups
            assert len(t["fifo"]) == len(t["fifo_tags"]) == len(set(t["fifo"]))
            if t["write"] and t["write"]["gid"] is not None:
                assert t["write"]["gid"] not in seen
                seen.add(t["write"]["gid"])
                assert t["write"]["gid"] in t["fifo"]
        assert len(seen) == tr["n_writes"]
        assert tr["n_scored"] <= tr["n_probes"]
        for t in tr["turns"]:
            if t["decode"] and t["reward"] is None:
                assert t.get("possible") is False, "sonde POSSIBLE non notée"
    # leave-one-slot : retirer le slot élu change la sélection, et la log-prob
    n_ls = 0
    for tr in traces:
        for q, d in probe_pairs(tr):
            if q is None or not d["decode"]["tokens"]:
                continue
            pair = {"retrieve": q["retrieve"], "decode": d["decode"]}
            base = forward_probe_with_masked_slot(model, pair, None, cfg, 9,
                                                  a_open, temp=pol.decode_temp)
            if len(q["retrieve"]["cand_gids"]) <= cfg.eval_groups:
                continue
            gone = forward_probe_with_masked_slot(
                model, pair, q["retrieve"]["gids"][0], cfg, 9, a_open,
                temp=pol.decode_temp)
            assert gone["dropped"] and q["retrieve"]["gids"][0] not in gone["gids"]
            assert len(gone["gids"]) == len(base["gids"]) == cfg.eval_groups, \
                "le préfixe contrefactuel doit garder EXACTEMENT sa longueur"
            n_ls += 1
    print(f"  [d] trace : ancres (seg, turn, kind, role) identiques sur "
          f"{len(traces)} rollouts appariés (digest {traces[0]['digest']}), "
          f"actions distinctes, FIFO cohérente, {n_ls} contrefactuelles "
          f"leave-one-slot à longueur de préfixe constante")

    # ── (e) WRITE : log-prob correcte, mode `facts` == le write du SFT ──────
    lg = torch.tensor(0.7)
    for a in (0, 1):
        assert abs(float(write_logp(lg, a)) -
                   math.log(float(torch.sigmoid(lg)) if a
                            else 1 - float(torch.sigmoid(lg)))) < 1e-6
    assert abs(float(write_logp(lg, 1, temp=4.0)) -
               math.log(float(torch.sigmoid(lg / 4)))) < 1e-6
    rr = _random.Random(0)
    n1 = sum(write_sample(lg, rr)[0] for _ in range(4000))
    assert abs(n1 / 4000 - float(torch.sigmoid(lg))) < 0.02, n1
    # plancher d'exploration : la log-prob enregistrée est celle du COMPORTEMENT
    a, lp, p, p_b = write_sample(torch.tensor(-9.0), rr, floor=0.5)
    assert abs(p_b - 0.5) < 1e-9 and p < 1e-3
    assert abs(lp - (math.log(0.5) if a else math.log(0.5))) < 1e-6

    # mode `facts` : MÊME banque que `RtiRunner` (le chemin SFT), bit à bit
    from .rti import RtiRunner
    pol_f = RtiPolicyConfig(write_mode="facts", write_on="all")
    _, tr_f = _mk(pol_f, G=2, seed=5)
    ref = RtiRunner(cfg, sif_w, sep_id=9, n_lanes=1, seed=0)
    for s in conv["segs"]:
        ref.write(model.embed.weight, s["input_ids"], s.get("fact_slot"))
    for tr in tr_f:
        got = [t["write"]["gid"] for t in tr["turns"]
               if t["write"] and t["write"]["a"]]
        assert len(got) == len([s for s in conv["segs"]
                                if s.get("fact_slot") is not None])
        assert tr["turns"][-1]["fifo_tags"] == ref.banks[0].tags, \
            (tr["turns"][-1]["fifo_tags"], ref.banks[0].tags)
    print(f"  [e] write : log-prob Bernoulli exacte (temp comprise), plancher "
          f"d'exploration sur le COMPORTEMENT, mode `facts` == "
          f"RtiRunner.write (tags {ref.banks[0].tags})")

    print("rti_policy self-test: OK (Plackett-Luce normalisée + τ→0 == top-k "
          "SFT, masquage du préfixe prouvé par rejeu, trace appariée à ancres "
          "de tour, leave-one-slot à layout constant, write Bernoulli et "
          "baseline procédurale)")


def attach_rti_modules(model, d_model: int) -> list:
    """Attache retriever / vecteur de type / tête de write au modèle.

    UN SEUL point : le worker et le learner DOIVENT construire le même modèle,
    sinon le `state_dict` publié ne se recharge pas (et un `strict=False` le
    dirait trop tard). Rend les noms des paramètres neufs, pour que le
    chargement du ckpt SFT puisse les annoncer.
    """
    from .rti import InjectType, Retriever
    dev = next(model.parameters()).device
    if not hasattr(model, "rti_retriever"):
        model.rti_retriever = Retriever(d_model).to(dev)
    if not hasattr(model, "rti_type"):
        model.rti_type = InjectType(d_model).to(dev)
    if not hasattr(model, "rti_write"):
        model.rti_write = WriteHead(d_model).to(dev)
    return ["rti_retriever.wq.weight", "rti_retriever.log_temp",
            "rti_type.vec", "rti_write.w.weight", "rti_write.w.bias"]


if __name__ == "__main__":
    _selftest()
