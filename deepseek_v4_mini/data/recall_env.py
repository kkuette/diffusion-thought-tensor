"""recall_env — VIES DE RAPPEL SCRIPTÉES ET APPARIÉES (l'env du cliquet GRPO).

CE QUE CE MODULE EST, ET CE QU'IL N'EST PAS
───────────────────────────────────────────
C'est le GÉNÉRATEUR + le JUGE, rien d'autre. Aucun modèle, aucun rollout,
aucune banque : le rollout (agent 2) et l'algo (agent 3) se branchent dessus.
Tout ici tourne sur CPU, sans checkpoint, et se prouve par `_selftest`.

LE SCRIPT DE VIE, ET POURQUOI IL EST REJOUABLE
──────────────────────────────────────────────
`build_script(cfg, life)` rend une STRUCTURE PURE (JSON-sérialisable, aucun
tenseur, aucun tokenizer) décrivant une vie multi-tours :

    tour FACT     — un fait est énoncé (contenu + strate + identifiant)
    tour FILLER   — remplissage/distracteur (texte RÉEL quand un pool est
                    fourni, gabarits persona en repli hermétique)
    tour PROBE    — une sonde interroge un fait antérieur, avec sa vérité et
                    sa fonction de vérification

Le script ne dépend QUE de `(cfg.life_seed, life)` : deux appels rendent le
même objet bit à bit (`script_digest`). C'est l'APPARIEMENT — les G rollouts
d'un groupe GRPO partagent le même script (mêmes faits, mêmes sondes, même
ordre), donc l'avantage intra-groupe est CONTREFACTUEL (« ce rollout-ci a-t-il
mieux fait sur CETTE sonde »), et l'ancrage par index de tour (GiGPO) est bien
défini : le tour `t` de deux rollouts d'un groupe est LE MÊME tour.

L'ÂGE, LA FIFO, ET LES SONDES IMPOSSIBLES
─────────────────────────────────────────
Deux âges, à ne pas confondre :
  * `age`        = distance en SEGMENTS entre énonciation et question. C'est la
                   difficulté de RÉCUPÉRATION : combien de distracteurs se sont
                   intercalés, donc ce que le retriever doit trancher.
  * `age_writes` = distance en WRITES sous la POLITIQUE DE RÉFÉRENCE. C'est
                   elle, et elle seule, qui décide de la résidence en FIFO
                   (`age_writes <= max_groups`, max_groups = max_mem = 8).

Le générateur simule cette banque théorique et MARQUE `possible: false` les
sondes dont la cible est évincée. Elles sont NEUTRES au reward (exclues de
l'agrégat), jamais punitives — punir un rappel physiquement impossible
apprendrait au modèle à ne pas essayer.

LE CHOIX QUI COMPTE : QUELLE POLITIQUE DE RÉFÉRENCE. Le défaut est `facts`
(seuls les segs porteurs écrivent) et pas `all` (chaque seg écrit), pour une
raison de sémantique de reward : « impossible » doit vouloir dire « AUCUNE
politique d'écriture ne pouvait garder ce fait », c'est-à-dire la borne
OPTIMISTE. Sous `all`, un fait vieux de 9 segments serait déclaré neutre alors
qu'une politique qui écrit à propos l'aurait gardé — on neutraliserait
exactement le signal que le cliquet doit apprendre (écrire sélectivement). Un
échec sur une sonde `possible` EST imputable à la politique, et c'est ce qu'on
veut. `write_policy: all` reste disponible comme borne pessimistique de
diagnostic. Le défaut coïncide en outre avec le bras rti déployé
(`rti.write_every_turn: false`).

Conséquence de curriculum : sous `facts`, une sonde n'est impossible que si
PLUS de `max_groups` FAITS ont été plantés après elle — les vies à sondes
impossibles sont donc les vies à beaucoup de faits (`n_facts` > max_groups),
et `p_beyond` vise alors le fait le plus ancien encore libre.

C'est le seul « oracle » du module, et il est du côté du GÉNÉRATEUR : le
rollout ne le voit pas, il ne fait que ne pas être noté dessus.

REWARD
──────
Binaire par sonde, agrégat = moyenne sur les sondes POSSIBLES. Deux modes de
vérification, aucun juge appris :
  * `exact`  — match normalisé (casse, ponctuation, espaces) de la vérité dans
               la réponse, avec frontière de mot (« Biscuits » ne paie pas
               « Biscuit ») ;
  * `exec`   — le code décodé est EXÉCUTÉ contre des tests unitaires
               (`exec_sandbox`), binaire = tous les tests passent.
Pas de reward composite : un grade composite à 0 ne localise rien (leçon
valsif). L'économie de think reste un modulateur EXTERNE, optionnel
(`make_recall_env_reward`), exactement comme pour tool/exec.

STRATES (registre, pas un enum figé)
────────────────────────────────────
`register_stratum(...)` ; les quatre livrées :
  persona    — entité/nom rare (pools de `persona_chat_data`, réutilisés tels
               quels : ce sont eux qui rendent le grade non-babblable)
  numeric    — valeur numérique (codes, budgets, numéros de vol)
  code       — nom de fonction + constante à RÉUTILISER dans un code exécuté :
               la vérité se vérifie par EXÉCUTION, pas par match de surface
  preference — consigne en langage naturel à restituer

LE LAYOUT EST UN KNOB (leçon la plus chère de la série)
──────────────────────────────────────────────────────
Le circuit de copie se câble sur le LAYOUT du préfixe injecté : r5 v1
entraînait sur 1 groupe, évaluait sur top-2, et le grade s'est effondré à
0.067 MALGRÉ un retriever parfait. `cfg.inject_groups` voyage donc DANS le
script (`script["layout"]`) : le rollout doit configurer `RtiConfig` avec cette
valeur, et train/éval restent isomorphes par construction. Le SFT courant est à
2 groupes fixes ; le knob existe pour que le curriculum puisse le varier sans
qu'on aille rouvrir le rollout.

LES DISTRACTEURS COMPTENT
─────────────────────────
Une vie sans distracteur résident donne un retriever qui ne discrimine rien :
la densité de remplissage est un knob de première classe
(`filler_per_fact`), et les tours de remplissage sont du TEXTE RÉEL dès qu'un
pool l'est (`real_filler` → `PersonaChatStream._load_real_pairs`, le même
chemin disque-caché que le SFT).

Self-test CPU hermétique (stub tokenizer, vrai bac à sable) :
    python -m deepseek_v4_mini.data.recall_env
"""
from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import dataclass, field, asdict
from typing import Callable

import torch

from . import persona_chat_data as P
from ..rl.exec_sandbox import pass_frac
from ..rl.rl_rewards import extract_code

# Base des identifiants de slot. Les ids persona (`fact_id_maps`) sont 1-based
# et petits ; en décaler les nôtres de 1000 garantit qu'un fait persona et une
# sonde recall_env ne se confondent JAMAIS quand les deux streams partagent une
# lane (le lien fact_slot ↔ q_slot du retriever est une simple égalité d'entier).
# Contrepartie : la value_table du teacher appris dimensionne ses embeddings sur
# `n_slot_ids` de persona — le teacher DOIT rester off avec cet env, ce qui est
# de toute façon obligatoire sous rti (sa banque fast-weight est contournée).
SLOT_ID_BASE = 1000


# ── normalisation & match exact ──────────────────────────────────────────────

_PUNCT = re.compile(r"[^\w\s]+", re.UNICODE)
_WS = re.compile(r"\s+")


def norm_text(s: str) -> str:
    """Forme canonique pour le match exact : minuscules, ponctuation retirée,
    espaces normalisés. `« Biscuit! »` et `biscuit` deviennent le même mot."""
    return _WS.sub(" ", _PUNCT.sub(" ", str(s).lower())).strip()


def exact_match(text: str, truth: str) -> bool:
    """La vérité apparaît-elle comme SUITE DE MOTS COMPLÈTE dans la réponse ?

    Le padding par espaces fait la frontière de mot après normalisation :
    « Biscuits are tasty » ne paie PAS « Biscuit », et une vérité multi-mots
    (une consigne) se teste exactement de la même façon.
    """
    t, v = norm_text(text), norm_text(truth)
    if not v:
        return False                      # vérité vide : jamais un succès
    return f" {v} " in f" {t} "


# ── strates ──────────────────────────────────────────────────────────────────

@dataclass
class Stratum:
    """Une strate de faits. `sample(rng, taken)` rend un FAIT :

        {"slot": "<strate>:<slot>", "value": str, "statement": str,
         "question": str, "answer": str, "check": {...}}

    `taken` = valeurs déjà plantées dans la vie (deux faits ne partagent jamais
    une valeur : sinon la sonde serait ambiguë et le grade ne localiserait
    rien). `check` porte tout ce que le juge doit savoir, et RIEN d'autre.
    """
    name: str
    slots: tuple
    sample: Callable


STRATA: dict = {}


def register_stratum(st: Stratum) -> None:
    """Ajoute une strate au registre. Change la table des slot ids (recalculée
    à la volée) : à faire à l'import, jamais au milieu d'un run."""
    assert st.name not in STRATA, f"strate déjà enregistrée : {st.name}"
    STRATA[st.name] = st


def slot_id_map() -> dict:
    """{nom de slot → entier 1-based décalé}. Trié : stable entre instances,
    seeds et processus, comme `persona.fact_id_maps`."""
    names = sorted(n for st in STRATA.values() for n in st.slots)
    return {n: SLOT_ID_BASE + i + 1 for i, n in enumerate(names)}


# ── strate 1 : persona (pools de persona_chat_data, réutilisés tels quels) ───

_PERSONA_SLOTS = tuple(f"persona:{k}" for k in sorted(P.SLOTS))


def _sample_persona(rng: random.Random, taken: set) -> dict:
    k = rng.choice(sorted(P.SLOTS))
    st, qs, ans, _upd, pool = P.SLOTS[k]
    free = [v for v in pool if v not in taken]
    v = rng.choice(free or pool)
    p = rng.choice(P.PET_TYPES if k == "pet"
                   else P.SIBLINGS if k == "sibling" else [""])
    return {"slot": f"persona:{k}", "value": v,
            "statement": rng.choice(st).format(v=v, p=p),
            "question": rng.choice(qs).format(p=p),
            "answer": rng.choice(ans).format(v=v, p=p),
            "check": {"mode": "exact", "truth": v}}


register_stratum(Stratum("persona", _PERSONA_SLOTS, _sample_persona))


# ── strate 2 : valeurs numériques ────────────────────────────────────────────
# Espace combinatoire (~10^5) : effectivement inépuisable, jamais babblable, et
# le grader word-boundary ne peut pas false-positiver sur un nombre.

_NUM = {
    "locker": ("My locker code is {v}.",
               "What is my locker code?",
               "Your locker code is {v}.", (10000, 99999)),
    "budget": ("The budget I set for the trip is {v} euros.",
               "How much did I budget for the trip?",
               "You budgeted {v} euros for the trip.", (1100, 98000)),
    "flight": ("My flight number is {v}.",
               "Which flight number did I say I was on?",
               "You said your flight number is {v}.", (1000, 9999)),
    "apartment": ("I live in apartment {v}.",
                  "What apartment number did I give you?",
                  "You live in apartment {v}.", (100, 9999)),
    "meter": ("The meter reading I noted this morning was {v}.",
              "What meter reading did I note this morning?",
              "The meter reading you noted was {v}.", (20000, 99999)),
}
_NUMERIC_SLOTS = tuple(f"numeric:{k}" for k in sorted(_NUM))


def _sample_numeric(rng: random.Random, taken: set) -> dict:
    k = rng.choice(sorted(_NUM))
    stmt, q, ans, (lo, hi) = _NUM[k]
    for _ in range(64):
        v = str(rng.randint(lo, hi))
        if v not in taken:
            break
    return {"slot": f"numeric:{k}", "value": v,
            "statement": stmt.format(v=v), "question": q,
            "answer": ans.format(v=v),
            "check": {"mode": "exact", "truth": v}}


register_stratum(Stratum("numeric", _NUMERIC_SLOTS, _sample_numeric))


# ── strate 3 : code (vérité vérifiée par EXÉCUTION) ──────────────────────────
# Le fait est une CONVENTION de projet (nom de fonction + constante) énoncée
# tôt ; la sonde demande d'écrire le helper. Le code ne peut passer les tests
# que si les DEUX moitiés du fait ont survécu — le nom (strate lexicale) ET la
# constante (strate numérique) — et c'est l'interpréteur qui tranche.

_CODE_STEMS = ["frobnicate", "widgetize", "quantulate", "burnish", "kludge",
               "marshalize", "torque", "flense", "gimbal", "winnow",
               "calcify", "tessellate", "vulcanize", "yaw", "zigzag"]
_CODE_OPS = {
    "scale":  ("multiplies its integer argument by the constant",
               "def {f}(n):\n    return n * {k}",
               ["assert {f}(2) == {r2}", "assert {f}(0) == 0",
                "assert {f}(-3) == {rm3}"]),
    "offset": ("adds the constant to its integer argument",
               "def {f}(n):\n    return n + {k}",
               ["assert {f}(2) == {r2}", "assert {f}(0) == {k}",
                "assert {f}(-3) == {rm3}"]),
}
_CODE_SLOTS = tuple(f"code:{k}" for k in sorted(_CODE_OPS))


def _sample_code(rng: random.Random, taken: set) -> dict:
    op = rng.choice(sorted(_CODE_OPS))
    desc, body, tests = _CODE_OPS[op]
    for _ in range(64):
        f = f"{rng.choice(_CODE_STEMS)}_{rng.randint(100, 999)}"
        if f not in taken:
            break
    k = rng.randint(3, 97)
    r2, rm3 = (2 * k, -3 * k) if op == "scale" else (2 + k, -3 + k)
    sol = body.format(f=f, k=k)
    return {
        "slot": f"code:{op}", "value": f,
        # atomes CITABLES du fait : les tests exec exigent le nom ET la
        # constante — le copy_mask doit couvrir les deux (kill-test 3 :
        # la constante perdait la sélection SIF, et n'était pas dans val_mask)
        "atoms": [f, str(k)],
        "statement": (f"For this project we agreed on a convention: the helper "
                      f"is called `{f}` and the constant is {k}."),
        "question": (f"Now write the helper we agreed on: a function that "
                     f"{desc}. Reply with a single ```python``` block."),
        "answer": f"```python\n{sol}\n```",
        "check": {"mode": "exec",
                  "tests": [t.format(f=f, k=k, r2=r2, rm3=rm3) for t in tests],
                  "gold": sol},
    }


register_stratum(Stratum("code", _CODE_SLOTS, _sample_code))


# ── strate 4 : préférence / consigne en langage naturel ──────────────────────

_PREF_PHRASES = ["Onwards and upwards", "Yours in haste", "Keep the kettle on",
                 "Mind the gap", "Steady as she goes", "Until the next tide",
                 "Warmly from the workshop", "With rhubarb and regards"]
_PREF_UNITS = ["nautical miles", "furlongs", "kilopascals", "stone",
               "degrees Kelvin", "cubits"]
_PREF = {
    "signoff": ("Please always sign off your messages with the phrase "
                "\"{v}\".",
                "Which sign-off did I ask you to use?",
                "You asked me to sign off with \"{v}\".", _PREF_PHRASES),
    "units": ("From now on, please give every measurement in {v}.",
              "Which unit did I ask you to use for measurements?",
              "You asked for measurements in {v}.", _PREF_UNITS),
}
_PREF_SLOTS = tuple(f"pref:{k}" for k in sorted(_PREF))


def _sample_pref(rng: random.Random, taken: set) -> dict:
    k = rng.choice(sorted(_PREF))
    stmt, q, ans, pool = _PREF[k]
    free = [v for v in pool if v not in taken]
    v = rng.choice(free or pool)
    return {"slot": f"pref:{k}", "value": v,
            "statement": stmt.format(v=v), "question": q,
            "answer": ans.format(v=v),
            "check": {"mode": "exact", "truth": v}}


register_stratum(Stratum("preference", _PREF_SLOTS, _sample_pref))


# ── config ───────────────────────────────────────────────────────────────────

@dataclass
class RecallEnvConfig:
    """Le CURRICULUM. Tout ce qui suit est déclaré dans `cfg_schema` sous la
    section `recall_env` — une faute de frappe arrête le run."""

    life_seed: int = 0
    # mix de strates : {nom de strate: poids}. Un nom absent du registre lève.
    strata: dict = field(default_factory=lambda: {
        "persona": 0.4, "numeric": 0.2, "code": 0.2, "preference": 0.2})
    n_facts: tuple = (2, 4)          # faits plantés par vie
    n_probes: tuple = (1, 3)         # sondes par vie (<= n_facts)
    filler_per_fact: tuple = (1, 3)  # paires de remplissage après chaque fait
    p_ack: float = 0.5               # accusé de réception après une énonciation
    # distribution des ÂGES visés (en segments = en writes). Le padding de
    # remplissage amène la sonde à l'âge tiré ; un âge inatteignable (la vie a
    # déjà dépassé) est CLAMPÉ, et c'est l'âge ATTEINT qui est consigné.
    # (mesuré au self-test avec les défauts : âge atteint moyen 10.8, médiane 9,
    #  p10 5 / p90 18 — le clamp ne peut que VIEILLIR une sonde, jamais la
    #  rajeunir, et l'âge consigné est toujours l'âge ATTEINT.)
    age_bins: tuple = ((2, 6), (7, 12))
    age_weights: tuple = (0.5, 0.5)
    p_beyond: float = 0.15           # part de sondes visant DÉLIBÉRÉMENT un fait
                                     # évincé (grade neutre, jamais punitif)
    max_groups: int = 8              # FIFO de groupes = max_mem du 350M
    # POLITIQUE DE RÉFÉRENCE qui décide de la RÉSIDENCE — le choix de design le
    # moins trivial du module, cf. l'en-tête :
    #   "facts" (défaut) = seuls les segs porteurs écrivent. C'est ce que fait
    #     le bras rti déployé (`write_every_turn: false`) ET la borne
    #     OPTIMISTE : une sonde n'est neutre que si AUCUNE politique d'écriture
    #     ne pouvait garder le fait. Neutraliser plus large éteindrait
    #     exactement le signal que le cliquet doit apprendre (écrire à propos).
    #   "all" = chaque seg écrit (borne PESSIMISTE, diagnostic).
    write_policy: str = "facts"
    inject_groups: int = 2           # LAYOUT du préfixe rti (train == éval)
    value_weight: float = 4.0        # surpondération du span-valeur (SFT/CE)
    exec_timeout: float = 6.0        # bac à sable de la strate code
    # remplissage RÉEL (mêmes clés que persona : même chemin, même cache)
    real_filler: str = ""
    real_split: str = "train"
    real_cap: int = 20000
    real_max_tok: int = 96
    real_cache_dir: str = ""
    p_real: float = 0.8
    # teacher SIF (poids par token) — repris tel quel de persona
    surprisal_mode: str = "sif"
    sif_a: float = 1e-4

    def __post_init__(self):
        self.n_facts = tuple(int(v) for v in self.n_facts)
        self.n_probes = tuple(int(v) for v in self.n_probes)
        self.filler_per_fact = tuple(int(v) for v in self.filler_per_fact)
        self.age_bins = tuple(tuple(int(x) for x in b) for b in self.age_bins)
        self.age_weights = tuple(float(w) for w in self.age_weights)
        assert len(self.age_bins) == len(self.age_weights), \
            "age_bins et age_weights doivent avoir la même longueur"
        assert sum(self.age_weights) > 0, "age_weights tous nuls"
        assert self.n_facts[0] >= 1 and self.n_facts[1] >= self.n_facts[0]
        assert self.n_probes[0] >= 1 and self.n_probes[1] >= self.n_probes[0]
        assert self.max_groups >= 1 and self.inject_groups >= 1
        assert self.write_policy in ("facts", "all"), self.write_policy
        assert 0.0 <= self.p_beyond <= 1.0 and 0.0 <= self.p_ack <= 1.0
        bad = [k for k in self.strata if k not in STRATA]
        assert not bad, f"strate(s) inconnue(s) : {bad} (connues : {sorted(STRATA)})"
        assert sum(float(w) for w in self.strata.values()) > 0, \
            "tous les poids de strate sont nuls"

    @classmethod
    def from_raw(cls, raw: dict) -> "RecallEnvConfig":
        """Construit depuis la section `recall_env:` d'une config YAML."""
        return cls(**{k: v for k, v in (raw or {}).items()})

    def to_dict(self) -> dict:
        return asdict(self)


# ── le SCRIPT ────────────────────────────────────────────────────────────────

def _life_rng(cfg: RecallEnvConfig, life: int) -> random.Random:
    """RNG d'une vie. Graine ENTIÈRE (jamais un tuple : `hash()` sur des str
    est randomisé par processus, le script ne serait pas rejouable d'un worker
    à l'autre)."""
    return random.Random((int(cfg.life_seed) * 1_000_003 + int(life)) & 0x7FFFFFFF)


def _pick(rng: random.Random, names: list, weights: list):
    return rng.choices(names, weights=weights, k=1)[0]


def build_script(cfg: RecallEnvConfig, life: int, filler_pool: list = None) -> dict:
    """LE livrable central : une vie complète, en données pures.

    `filler_pool` = liste de couples (user, assistant) de texte RÉEL. Absent,
    on retombe sur les gabarits persona — le même repli hermétique que le SFT,
    et le seul chemin qu'emprunte le self-test.

    Deux compteurs, et il faut les distinguer :
      * `age`        = distance en SEGS entre l'énonciation et la question —
                       c'est la difficulté de RÉCUPÉRATION (combien de
                       distracteurs se sont intercalés) ;
      * `age_writes` = distance en WRITES sous la politique de référence — c'est
                       elle, et elle seule, qui décide de la RÉSIDENCE en FIFO.
    """
    rng = _life_rng(cfg, life)
    ids = slot_id_map()
    st_names = [k for k, w in sorted(cfg.strata.items()) if float(w) > 0]
    st_w = [float(cfg.strata[k]) for k in st_names]
    all_write = cfg.write_policy == "all"

    turns: list = []
    n_segs = 0
    n_writes = 0                 # writes émis sous la politique de référence

    def emit(kind: str, user: str, assistant, **extra) -> dict:
        """Un tour = 1 seg user (+ 1 seg assistant). Les writes sont comptés
        ICI, seg par seg : sous `all` chaque seg écrit, sous `facts` seul le
        seg d'énonciation écrit (c'est `rti.write_every_turn`)."""
        nonlocal n_segs, n_writes
        t = {"kind": kind, "turn": len(turns), "user_seg": n_segs,
             "user": user, "assistant": assistant}
        n_segs += 1
        n_writes += int(all_write or kind == "fact")
        if assistant is not None:
            t["assistant_seg"] = n_segs
            n_segs += 1
            n_writes += int(all_write)
        t.update(extra)
        turns.append(t)
        return t

    def emit_filler() -> None:
        if filler_pool:
            u, a = filler_pool[rng.randrange(len(filler_pool))] \
                if rng.random() < cfg.p_real else _template_filler(rng)
        else:
            u, a = _template_filler(rng)
        emit("filler", u, a)

    # ── faits ────────────────────────────────────────────────────────────────
    facts: list = []
    taken: set = set()
    used_slots: set = set()
    n_f = rng.randint(*cfg.n_facts)
    for fid in range(n_f):
        for _ in range(16):                       # un slot par vie au plus :
            name = _pick(rng, st_names, st_w)     # deux faits du même slot
            f = STRATA[name].sample(rng, taken)   # rendraient la sonde ambiguë
            if f["slot"] not in used_slots:
                break
        else:
            continue
        used_slots.add(f["slot"])
        taken.add(f["value"])
        w0 = n_writes                             # ordinal du write de CE fait
        t = emit("fact", f["statement"], None, fid=fid)
        facts.append({"fid": fid, "stratum": name, "slot": f["slot"],
                      "slot_id": ids[f["slot"]], "value": f["value"],
                      "atoms": f.get("atoms", [f["value"]]),
                      "seg": t["user_seg"], "write_idx": w0,
                      "question": f["question"],
                      "answer": f["answer"], "check": f["check"]})
        if rng.random() < cfg.p_ack:
            emit("filler", *_ack_pair(rng, f["value"]))
        for _ in range(rng.randint(*cfg.filler_per_fact)):
            emit_filler()

    # ── sondes ───────────────────────────────────────────────────────────────
    n_p = min(len(facts), rng.randint(*cfg.n_probes))
    # Les âges visés sont tirés D'ABORD, puis triés CROISSANT : une vie est une
    # flèche (on ne peut que pousser en avant), donc la sonde la plus jeune doit
    # partir la première, sinon tout serait clampé par le padding des autres.
    wants, beyond = [], []
    for _ in range(n_p):
        if rng.random() < cfg.p_beyond:
            wants.append(rng.randint(cfg.max_groups + 2, cfg.max_groups + 6))
            beyond.append(True)
        else:
            lo, hi = _pick(rng, list(cfg.age_bins), list(cfg.age_weights))
            wants.append(rng.randint(lo, hi))
            beyond.append(False)
    order = sorted(range(n_p), key=lambda i: wants[i])

    probes: list = []
    free = list(facts)
    for i in order:
        if not free:
            break
        want = wants[i]
        if beyond[i]:
            # viser DÉLIBÉRÉMENT un fait évincé : le plus ANCIEN encore libre,
            # c'est-à-dire celui qui a le plus de writes derrière lui.
            f = free[0]
        else:
            # Le fait est choisi APRÈS l'âge : celui dont l'âge minimal
            # atteignable (S − seg, on ne peut pas rajeunir) tient dans la
            # cible ; le plus ANCIEN de ceux-là (on garde les faits récents pour
            # les sondes jeunes qui suivent).
            fit = [f for f in free if n_segs - f["seg"] <= want]
            f = fit[0] if fit else min(free, key=lambda x: n_segs - x["seg"])
        free.remove(f)
        target = f["seg"] + want
        while n_segs + 2 <= target:               # remplissage = 2 segs
            emit_filler()
        pid = len(probes)
        # RÉSIDENCE : évaluée AVANT le write de la question (le préfixe injecté
        # devant la réponse est construit sur la banque telle qu'elle est au
        # moment où la question arrive).
        aw = n_writes - f["write_idx"]
        possible = aw <= cfg.max_groups
        t = emit("probe", f["question"], f["answer"], pid=pid, fid=f["fid"])
        age = t["user_seg"] - f["seg"]
        probes.append({"pid": pid, "fid": f["fid"], "turn": t["turn"],
                       "q_seg": t["user_seg"], "a_seg": t["assistant_seg"],
                       "slot": f["slot"], "slot_id": f["slot_id"],
                       "value": f["value"], "age": age, "age_writes": aw,
                       "possible": possible,
                       "check": f["check"], "gold": f["answer"]})
        t["age"] = age
        t["possible"] = possible

    script = {
        "v": 1, "life": int(life), "life_seed": int(cfg.life_seed),
        "layout": {"inject_groups": int(cfg.inject_groups),
                   "max_groups": int(cfg.max_groups),
                   "write_policy": cfg.write_policy},
        "facts": facts, "turns": turns, "probes": probes,
        "n_segs": n_segs, "n_writes": n_writes,
        # le crédit contrefactuel « leave-one-slot » de l'agent 3 : quelle sonde
        # dépend de quel fait, et de quel seg vient ce fait.
        "fact_to_probes": {str(f["fid"]): [p["pid"] for p in probes
                                           if p["fid"] == f["fid"]]
                           for f in facts},
    }
    script["digest"] = script_digest(script)
    return script


def _template_filler(rng: random.Random) -> tuple:
    u, a = rng.choice(P.FILLERS)
    return (u.format(w=rng.choice(P.WEATHER), a=rng.choice(P.ACTIVITIES),
                     h=rng.choice(P.HOBBIES), s=rng.choice(P.SHOWS)), a)


def _ack_pair(rng: random.Random, v: str) -> tuple:
    """L'accusé de réception est un tour ASSISTANT seul dans persona ; ici on le
    rend comme un tour filler à contenu d'accusé, pour garder l'alternance
    user/assistant du template ChatML (un seg user « nu » casserait le
    register)."""
    return ("Thanks.", rng.choice(P.ACK_TMPL).format(v=v))


def script_digest(script: dict) -> str:
    """Empreinte canonique du script, hors champ `digest` lui-même. C'est LA
    preuve d'appariement : deux rollouts d'un groupe doivent afficher la même."""
    body = {k: v for k, v in script.items() if k != "digest"}
    return hashlib.sha256(
        json.dumps(body, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()[:16]


# ── grading ──────────────────────────────────────────────────────────────────

def grade_probe(probe: dict, text: str, timeout: float = 6.0) -> dict:
    """{"grade": 0/1, "raw": float, "possible": bool}. BINAIRE, sans juge.

    `raw` garde la fraction continue (tests passés) : le grade seul ne
    distingue pas « 3 tests sur 4 » de « rien ne tourne », et ces deux
    diagnostics n'appellent pas la même correction (même convention que
    `rl_rewards.make_exec_reward`).
    """
    chk = probe["check"]
    if chk["mode"] == "exact":
        raw = 1.0 if exact_match(text, chk["truth"]) else 0.0
    elif chk["mode"] == "exec":
        raw = pass_frac(extract_code(text), chk["tests"], timeout=timeout)
    else:
        raise ValueError(f"mode de vérification inconnu : {chk['mode']}")
    return {"grade": 1.0 if raw >= 1.0 else 0.0, "raw": float(raw),
            "possible": bool(probe["possible"])}


def grade_life(script: dict, texts, timeout: float = 6.0) -> dict:
    """Note toutes les sondes d'une vie.

    `texts` = les réponses décodées, DANS L'ORDRE DES SONDES (une par sonde).
    Rend {"per_probe": [...], "grade": agrégat, "n_possible": int}.

    L'agrégat ne moyenne QUE les sondes possibles : une sonde dont la cible est
    évincée est neutre, pas un échec. Aucune sonde possible ⇒ 1.0 (rien à
    noter — même convention que `persona.grade_conv` sur le smalltalk).
    """
    probes = script["probes"]
    assert len(texts) == len(probes), \
        f"{len(texts)} réponses pour {len(probes)} sondes"
    per = [grade_probe(p, t, timeout) for p, t in zip(probes, texts)]
    ok = [g["grade"] for g in per if g["possible"]]
    return {"per_probe": per, "grade": sum(ok) / len(ok) if ok else 1.0,
            "n_possible": len(ok), "n_probes": len(per)}


def make_recall_env_reward(n_max: int = 16, floor: float = 1.0,
                           timeout: float = 6.0):
    """`EnvSpec.reward_fn` pour cet env : grade de LA sonde décodée, puis
    l'économie de think (neutre par défaut, `floor=1.0` ⇒ eff ≡ 1 — décision
    07-29 : le FIFO price déjà les writes).

    info : {"text": str, "probe": <entrée de script["probes"]>, "n_think": int}
    Écrit en retour info["raw"] (fraction brute) et info["possible"].
    Une sonde IMPOSSIBLE rend None : à l'appelant de l'exclure du groupe GRPO
    plutôt que de la compter comme un échec.
    """
    from ..rl.rl_rewards import think_economy

    def fn(ce, info):
        g = grade_probe(info["probe"], info.get("text") or "", timeout)
        info["raw"], info["possible"] = g["raw"], g["possible"]
        if not g["possible"]:
            return None
        return think_economy(g["grade"], int(info.get("n_think", 0)),
                             n_max, floor)
    return fn


# ── le STREAM : script → segments consommables par le pipeline chat ──────────

class RecallEnvStream(P.PersonaChatStream):
    """Contrat de stream chat (`next_conv` / `rng` / `grade_conv`), donc
    utilisable par `chat_batch`, `chat_mix` et `rl_disagg` sans rien changer.

    Hérite de `PersonaChatStream` UNIQUEMENT pour sa plomberie de tokens
    (`_ids`, `_seg`, `_user`, `_assistant`, `_val_span`, table SIF, chargement
    du filler réel) — exactement comme `CodeExecStream` et `SotaSessionStream`.
    Le contenu, lui, vient du script.

    CLÉS POSÉES SUR LES SEGS (ce que l'agent 2 doit savoir) :
      * seg d'ÉNONCIATION : `fact_slot` [1,T] long constant (= slot_id) et
        `val_mask` sur le span de la valeur ;
      * seg de QUESTION   : `q_slot` [1,T] long constant, MÊME entier ;
      * seg de RÉPONSE    : `probe_id` (int), `decode` (True) — c'est LÀ que le
        rollout décode. Ces deux clés sont des scalaires : `_collate` les
        ignore (il ne propage que `_ZERO_PAD_KEYS` + `roles`) et `_split_seg`
        les recopie. Le rollout travaille sur la LISTE de segs, pas sur le
        collate, donc il les voit.
    """

    def __init__(self, tok, *, seed: int = 0, cfg: RecallEnvConfig = None,
                 **kw) -> None:
        cfg = cfg or RecallEnvConfig(**{k: v for k, v in kw.items()})
        self.cfg = cfg
        super().__init__(tok, real_filler=cfg.real_filler or None,
                         real_split=cfg.real_split, real_cap=cfg.real_cap,
                         real_max_tok=cfg.real_max_tok,
                         real_cache_dir=cfg.real_cache_dir or None,
                         p_real=cfg.p_real, value_weight=cfg.value_weight,
                         surprisal_mode=cfg.surprisal_mode, sif_a=cfg.sif_a,
                         seed=seed)
        self.slot_ids_env = slot_id_map()

    # ── scripts ─────────────────────────────────────────────────────────────

    def script(self, life: int) -> dict:
        """LE point d'appariement : même `life` ⇒ même script, toujours."""
        return build_script(self.cfg, life, self.real_pairs)

    def next_conv(self) -> dict:
        """Tire une vie NEUVE. L'index de vie sort du rng du stream, donc le
        round-trip rng du checkpoint rejoue la même suite de vies — et
        `info["life"]` permet de REJOUER n'importe laquelle à l'identique
        (`conv_for_life`), ce dont le groupe GRPO a besoin."""
        return self.conv_for_life(self.rng.randrange(1 << 31))

    def conv_for_life(self, life: int) -> dict:
        sc = self.script(life)
        return {"kind": "recall_env", "segs": self.segs(sc),
                "info": self.info(sc)}

    # ── script → segs ────────────────────────────────────────────────────────

    def segs(self, script: dict) -> list:
        by_fid = {f["fid"]: f for f in script["facts"]}
        out: list = []
        for t in script["turns"]:
            if t["kind"] == "fact":
                f = by_fid[t["fid"]]
                out.append(self._user_fact(t["user"], f["value"], f["slot_id"],
                                           f.get("atoms")))
            elif t["kind"] == "probe":
                p = script["probes"][t["pid"]]
                out.append(self._user_probe(t["user"], p["slot_id"], p["pid"]))
                a = self._assistant_valued(t["assistant"], p["value"])
                a["probe_id"] = p["pid"]
                a["decode"] = True
                out.append(a)
            else:
                out.append(self._user(t["user"]))
                if t["assistant"] is not None:
                    out.append(self._assistant(t["assistant"]))
        assert len(out) == script["n_segs"], (len(out), script["n_segs"])
        return out

    def _user_fact(self, text: str, value: str, slot_id: int,
                   atoms: list = None) -> dict:
        seg = self._user(text)
        ids = seg["input_ids"][0].tolist()
        T = len(ids)
        vm = torch.zeros(T)
        span = self._val_span(ids, value)
        if span is not None:
            vm[span[0]:span[1]] = 1.0
        seg["val_mask"] = vm.unsqueeze(0)
        seg["fact_slot"] = torch.full((1, T), int(slot_id), dtype=torch.long)
        # copy_mask = TOUS les atomes citables (kill-test 3 : pour `code`, la
        # constante est exigée par les tests exec mais absente de val_mask —
        # qui, lui, reste la cible du teacher value_sif et ne bouge pas)
        cm = vm.clone()
        for a in (atoms or [value]):
            sp = self._val_span(ids, str(a))
            if sp is not None:
                cm[sp[0]:sp[1]] = 1.0
        seg["copy_mask"] = cm.unsqueeze(0)
        return seg

    def _user_probe(self, text: str, slot_id: int, pid: int) -> dict:
        seg = self._user(text)
        T = seg["input_ids"].size(1)
        seg["q_slot"] = torch.full((1, T), int(slot_id), dtype=torch.long)
        seg["probe_id"] = int(pid)
        return seg

    # ── info & grade ─────────────────────────────────────────────────────────

    def info(self, script: dict) -> dict:
        """`truths` / `ages` / `q_slots` gardent les noms de persona (les
        graders et le suivi existants les lisent), le reste est propre à l'env.
        """
        pr = script["probes"]
        return {"truths": [p["value"] for p in pr],
                "ages": [p["age"] for p in pr],
                "q_slots": [p["slot"] for p in pr],
                "possible": [p["possible"] for p in pr],
                "probes": pr, "facts": script["facts"],
                "fact_to_probes": script["fact_to_probes"],
                "layout": script["layout"], "life": script["life"],
                "digest": script["digest"], "script": script,
                "exec_timeout": self.cfg.exec_timeout}

    def grade_conv(self, conv: dict, texts) -> float:
        """Contrat d'éval du trainer : les tours notés sont les `len(truths)`
        DERNIERS tours assistant décodés."""
        sc = conv["info"]["script"]
        n = len(sc["probes"])
        if n == 0:
            return 1.0
        tail = list(texts)[-n:]
        if len(tail) < n:                     # décodage tronqué : les manquants
            tail = [""] * (n - len(tail)) + tail
        return grade_life(sc, tail, self.cfg.exec_timeout)["grade"]


# ── self-test ────────────────────────────────────────────────────────────────

def _selftest() -> None:
    """Ce que cet env peut casser en silence : un script non rejouable (les
    groupes GRPO ne seraient plus appariés et l'avantage cesserait d'être
    contrefactuel), une sonde impossible notée comme un échec, un grade de
    surface qui paie un code faux, un seg dont `q_slot` ne pointe pas le
    `fact_slot` de son fait (la CE du retriever apprendrait le mauvais
    groupe), et un knob de curriculum qui ne bouge rien."""
    import statistics

    # ── 1. normalisation & match exact : positifs ET négatifs ───────────────
    assert exact_match("Your dog is named Biscuit.", "Biscuit")
    assert exact_match("your DOG is named   biscuit !!", "Biscuit")
    assert exact_match("Sign off with \"Mind the gap\", ok?", "Mind the gap")
    assert not exact_match("Biscuits are tasty.", "Biscuit")     # frontière
    assert not exact_match("I have no idea.", "Biscuit")
    assert not exact_match("code 4819200", "481920")             # sous-chaîne
    assert not exact_match("anything", "")                       # vérité vide
    print("  [1] match exact : casse/ponctuation/espaces OK, frontière tenue")

    cfg = RecallEnvConfig(life_seed=7)

    # ── 2. déterminisme seed→script, BIT À BIT ──────────────────────────────
    a, b = build_script(cfg, 3), build_script(cfg, 3)
    ja = json.dumps(a, sort_keys=True, ensure_ascii=False)
    assert ja == json.dumps(b, sort_keys=True, ensure_ascii=False)
    assert a["digest"] == b["digest"]
    assert build_script(cfg, 4)["digest"] != a["digest"], \
        "deux vies différentes ne doivent pas coïncider"
    assert build_script(RecallEnvConfig(life_seed=8), 3)["digest"] != a["digest"]
    # appariement : un « groupe » de G rollouts partage UN script
    G = 8
    grp = [build_script(cfg, 11) for _ in range(G)]
    assert len({s["digest"] for s in grp}) == 1
    assert len({json.dumps(s, sort_keys=True) for s in grp}) == 1
    # ancrage par index de tour : le tour t est LE MÊME tour partout
    for s in grp[1:]:
        assert [(t["kind"], t["turn"], t["user_seg"]) for t in s["turns"]] == \
               [(t["kind"], t["turn"], t["user_seg"]) for t in grp[0]["turns"]]
    print(f"  [2] déterminisme : script bit à bit, {G} rollouts même digest "
          f"{a['digest']}, ancrage de tour stable")

    # ── 3. FIFO théorique & sondes impossibles ──────────────────────────────
    # (a) sous la politique par défaut (`facts`), une vie de 2-4 faits tient
    #     ENTIÈRE dans une FIFO de 8 : RIEN n'est impossible, et c'est la
    #     réponse correcte — un échec y est imputable à la politique.
    n_imp = n_tot = 0
    for life in range(400):
        sc = build_script(cfg, life)
        for p in sc["probes"]:
            n_tot += 1
            n_imp += not p["possible"]
            assert p["possible"] == (p["age_writes"] <= cfg.max_groups), p
            f = next(f for f in sc["facts"] if f["fid"] == p["fid"])
            assert p["age"] == p["q_seg"] - f["seg"]          # âge = segs
    assert n_imp == 0, (n_imp, n_tot)
    # (b) vies SATURÉES (14 faits > 8 slots) : les faits précoces SONT évincés.
    sat = RecallEnvConfig(life_seed=7, n_facts=(14, 14), n_probes=(4, 4),
                          p_beyond=0.5)
    s_imp = s_tot = 0
    for life in range(200):
        sc = build_script(sat, life)
        for p in sc["probes"]:
            s_tot += 1
            s_imp += not p["possible"]
    assert 0.15 < s_imp / s_tot < 0.9, (s_imp, s_tot)
    # (c) borne PESSIMISTE `all` : chaque seg écrit, donc âge_segs == âge_writes
    pes = RecallEnvConfig(life_seed=7, write_policy="all")
    p_imp = p_tot = 0
    for life in range(200):
        sc = build_script(pes, life)
        for p in sc["probes"]:
            assert p["age"] == p["age_writes"], p
            p_tot += 1
            p_imp += not p["possible"]
    assert p_imp > 0, "sous `all`, une FIFO de 8 segs évince forcément"
    # (d) neutralité : une vie dont TOUTES les sondes sont impossibles, mal
    #     répondue, ne doit pas être punie
    dead = build_script(RecallEnvConfig(life_seed=7, n_facts=(14, 14),
                                        n_probes=(2, 2), p_beyond=1.0), 5)
    assert not any(p["possible"] for p in dead["probes"]), \
        [(p["age"], p["age_writes"]) for p in dead["probes"]]
    g = grade_life(dead, ["nonsense"] * len(dead["probes"]))
    assert g["n_possible"] == 0 and g["grade"] == 1.0
    print(f"  [3] FIFO : défaut `facts` {n_imp}/{n_tot} impossibles (borne "
          f"optimiste), saturé 14 faits {100 * s_imp / s_tot:.0f} %, borne "
          f"pessimiste `all` {100 * p_imp / p_tot:.0f} %, agrégat neutre "
          f"quand tout est évincé")

    # ── 4. reward : exact-match ET exécution, positifs ET négatifs ──────────
    sc = build_script(RecallEnvConfig(life_seed=3, strata={"code": 1.0},
                                      n_facts=(2, 2), n_probes=(2, 2),
                                      p_beyond=0.0, age_bins=((2, 4),),
                                      age_weights=(1.0,)), 1)
    assert all(p["check"]["mode"] == "exec" for p in sc["probes"])
    gold = [p["gold"] for p in sc["probes"]]
    gg = grade_life(sc, gold)
    assert all(x["grade"] == 1.0 for x in gg["per_probe"]), gg
    # NÉGATIF : le bon nom, la MAUVAISE constante — un grade de surface
    # (« le nom est là ») paierait ; l'interpréteur, non.
    wrong = [g.replace("* ", "* 1 + ").replace("+ ", "+ 1 + ") for g in gold]
    gw = grade_life(sc, wrong)
    assert all(x["grade"] == 0.0 for x in gw["per_probe"]), gw
    assert grade_life(sc, ["I don't remember."] * 2)["grade"] == 0.0
    # exact : positif / négatif sur une vie non-code
    sc2 = build_script(RecallEnvConfig(life_seed=3, strata={"persona": 0.5,
                                                            "numeric": 0.25,
                                                            "preference": 0.25},
                                       p_beyond=0.0, n_probes=(2, 2),
                                       n_facts=(2, 3), age_bins=((2, 4),),
                                       age_weights=(1.0,)), 2)
    assert grade_life(sc2, [p["gold"] for p in sc2["probes"]])["grade"] == 1.0
    assert grade_life(sc2, ["Noted!"] * len(sc2["probes"]))["grade"] == 0.0
    # partiel : une bonne, une mauvaise
    mixed = [sc2["probes"][0]["gold"]] + ["Noted!"] * (len(sc2["probes"]) - 1)
    assert 0.0 < grade_life(sc2, mixed)["grade"] < 1.0
    # hook EnvSpec : impossible ⇒ None (exclu du groupe), possible ⇒ float
    fn = make_recall_env_reward()
    p0 = sc2["probes"][0]
    assert fn(None, {"text": p0["gold"], "probe": p0, "n_think": 3}) == 1.0
    imp = dict(dead["probes"][0])
    assert fn(None, {"text": "x", "probe": imp, "n_think": 0}) is None
    print("  [4] reward : exec gold 1.0 / constante fausse 0.0, exact "
          "positif+négatif, agrégat partiel, hook impossible → None")

    # ── 5. segments : format, clés rti, collate ─────────────────────────────
    tok = P._StubTok()
    st = RecallEnvStream(tok, seed=1, cfg=RecallEnvConfig(
        life_seed=7, surprisal_mode="nll"))          # nll+ref None = surp off
    conv = st.conv_for_life(42)
    segs, info = conv["segs"], conv["info"]
    fs = {}
    for s in segs:
        assert s["input_ids"].dim() == 2 and s["input_ids"].size(0) == 1
        assert s["loss_mask"].shape == s["input_ids"].shape
        assert s["role"] in ("user", "assistant")
        if "fact_slot" in s:
            v = s["fact_slot"][0].unique().tolist()
            assert v == [int(s["fact_slot"][0, 0])] and v[0] >= SLOT_ID_BASE
            fs[v[0]] = True
        if "q_slot" in s:
            v = int(s["q_slot"][0, 0])
            assert (s["q_slot"][0] == v).all() and v in fs, \
                "q_slot doit pointer un fact_slot DÉJÀ énoncé"
            assert s["role"] == "user" and "fact_slot" not in s
    n_dec = [s for s in segs if s.get("decode")]
    assert len(n_dec) == len(info["probes"]) and \
        [s["probe_id"] for s in n_dec] == list(range(len(info["probes"])))
    # les segs sont bit à bit identiques pour la même vie (appariement côté tok)
    for s1, s2 in zip(segs, st.conv_for_life(42)["segs"]):
        assert torch.equal(s1["input_ids"], s2["input_ids"])
        assert torch.equal(s1["loss_mask"], s2["loss_mask"])
    # collate : les clés par-token survivent et se zéro-paddent
    from .chat_batch import ChatBatchStream
    cb = ChatBatchStream(tok, batch=2, slots=8, _inner=st, max_seg_tok=64)
    seen_q = seen_f = 0
    for _ in range(8):                      # plusieurs fenêtres : les sondes
        col = cb.next_conv_batch()          # arrivent tard dans une vie
        assert len(col) == 8
        for seg in col:
            assert seg["input_ids"].size(0) == 2 and seg["pad_mask"].any()
            for k in ("q_slot", "fact_slot", "val_mask"):
                if k in seg:
                    assert seg[k].shape == seg["input_ids"].shape
                    pad = ~seg["pad_mask"]
                    assert (seg[k][pad] == 0).all(), f"{k} fuit dans le padding"
            seen_q += "q_slot" in seg
            seen_f += "fact_slot" in seg
    assert seen_q and seen_f, (seen_q, seen_f)
    # grade_conv (contrat trainer) : les golds passent, le babble non
    golds = [p["gold"] for p in info["probes"]]
    assert st.grade_conv(conv, ["blah"] * 3 + golds) == 1.0
    assert st.grade_conv(conv, ["blah"] * len(golds)) == 0.0
    print(f"  [5] segs : {len(segs)} segs, fact_slot/q_slot appariés "
          f"({len(fs)} slots), {len(n_dec)} sondes marquées `decode`, "
          f"collate B=2 propre")

    # ── 6. knobs de curriculum : bouger le knob bouge la statistique ────────
    def stat(c, n=250):
        ages, fill, imp, strat = [], [], 0, {}
        tot = 0
        for life in range(n):
            s = build_script(c, life)
            ages += [p["age"] for p in s["probes"]]
            imp += sum(not p["possible"] for p in s["probes"])
            tot += len(s["probes"])
            fill.append(sum(t["kind"] == "filler" for t in s["turns"])
                        / max(1, len(s["turns"])))
            for f in s["facts"]:
                strat[f["stratum"]] = strat.get(f["stratum"], 0) + 1
        return (statistics.mean(ages), statistics.mean(fill), imp / max(1, tot),
                strat)

    base = RecallEnvConfig(life_seed=1, p_beyond=0.0)
    a_lo, f_lo, _, _ = stat(base)
    a_hi, _, _, _ = stat(RecallEnvConfig(life_seed=1, p_beyond=0.0,
                                         age_bins=((14, 20),),
                                         age_weights=(1.0,)))
    assert a_hi > a_lo + 5, (a_lo, a_hi)
    _, f_hi, _, _ = stat(RecallEnvConfig(life_seed=1, p_beyond=0.0,
                                         filler_per_fact=(6, 9)))
    assert f_hi > f_lo, (f_lo, f_hi)
    # densité de sondes impossibles : sous `facts`, elle se pilote par la
    # SATURATION (nombre de faits) croisée avec p_beyond, pas par l'âge.
    _, _, i_lo, _ = stat(RecallEnvConfig(life_seed=1, n_facts=(14, 14),
                                         n_probes=(3, 3), p_beyond=0.0))
    _, _, i_hi, _ = stat(RecallEnvConfig(life_seed=1, n_facts=(14, 14),
                                         n_probes=(3, 3), p_beyond=0.9))
    assert i_lo < 0.05 < 0.4 < i_hi, (i_lo, i_hi)
    _, _, _, mix = stat(RecallEnvConfig(life_seed=1, p_beyond=0.0,
                                        strata={"code": 1.0}))
    assert set(mix) == {"code"}, mix
    _, _, _, mix2 = stat(base)
    assert len(mix2) == 4, mix2
    # le layout est un knob qui VOYAGE avec le script (le rollout le lit)
    assert build_script(RecallEnvConfig(inject_groups=3), 0)["layout"][
        "inject_groups"] == 3
    print(f"  [6] curriculum : âge moyen {a_lo:.1f} → {a_hi:.1f}, "
          f"filler {f_lo:.2f} → {f_hi:.2f}, impossibles {i_lo:.2f} → "
          f"{i_hi:.2f}, mix de strates pilotable, layout dans le script")

    # ── 7. la config d'exemple construit VRAIMENT le curriculum ────────────
    # (le schéma dit que la clé existe ; ceci dit que la dataclass la lit.)
    import pathlib
    import yaml as _yaml
    p = pathlib.Path(__file__).parents[1] / "configs" / "rl_recall_env_smoke.yaml"
    if p.exists():
        raw = _yaml.safe_load(p.read_text())
        c = RecallEnvConfig.from_raw(raw["recall_env"])
        sc = build_script(c, 0)
        assert sc["layout"]["inject_groups"] == c.inject_groups
        assert sc["probes"] and sc["facts"]
        print(f"  [7] rl_recall_env_smoke.yaml : {len(raw['recall_env'])} clés "
              f"lues par RecallEnvConfig, vie 0 = {sc['n_segs']} segs / "
              f"{len(sc['facts'])} faits / {len(sc['probes'])} sondes")

    print("recall_env self-test: OK (déterminisme+appariement, FIFO/impossibles, "
          "reward exact+exec, segs rti/collate, curriculum)")


if __name__ == "__main__":
    _selftest()
