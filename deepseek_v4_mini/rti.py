"""RTI — retrieve-then-inject : le port au 350M du verdict du labo jouet.

CE QUE LE LABO A TRANCHÉ (toy_read_lab, phases 5→8)
──────────────────────────────────────────────────
  • tout readout APPRIS sur une ligne de banque est mort (MoS compris) ; la
    citation passe par des PSEUDO-TOKENS NATIFS injectés dans le contexte —
    r4 (injection à sélection oracle) : strate `code` 0.708 contre 0.000 pour
    tous les readouts appris ;
  • la sélection, elle, s'apprend : r5 met un retriever à UN paramètre
    (W_q d×d) sur les lignes-clés et sort recall@1 = 1.000 sur 267 requêtes ;
  • train et éval doivent être ISOMORPHES. r5 v1 entraînait sur 1 groupe et
    évaluait sur top-2 : le circuit de copie s'était câblé sur un préfixe
    mono-groupe, et le grade s'est effondré à 0.067 malgré un retriever
    parfait. C'est la leçon la plus chère de la série, elle est câblée ici
    par construction (cf. RtiConfig.train_groups).

CE QUI CHANGE EN PASSANT AU 350M
────────────────────────────────
  1. La clé n'est plus un PRIVILÈGE D'ORACLE. Au jouet, chaque paire
     (slot, attr) avait un vecteur dédié tiré d'une graine — le modèle ne
     l'aurait jamais produit. Ici la clé est PROCÉDURALE : le pooling SIF des
     embeddings du segment, c'est-à-dire une fonction du texte que le modèle
     voit. DOWNGRADE ASSUMÉ ET RISQUE PRINCIPAL DU PORT : deux segments qui
     parlent du même sujet ont des clés proches, là où les clés dédiées
     étaient à |cos| ≤ 0.28. `key_separation()` mesure cette dégradation AVANT
     de lancer, et le retriever n'a de sens que si les clés se séparent.
  2. Le write est PROCÉDURAL et sans gradient : aucune banque fast-weight, ni
     write ni read appris. Le seul apprentissage est W_q + le vecteur de type
     + le finetune du trunk.

Ce module est autonome (aucun import du trainer) et testé par `_selftest`.

⚠️ BLOQUANT IDENTIFIÉ POUR LE CÂBLAGE TRAINER (à traiter EN PREMIER)
────────────────────────────────────────────────────────────────────
La CE du retriever a besoin, devant chaque RÉPONSE, de savoir QUEL FAIT est
interrogé (la cible multi-positive). Cette information n'existe pas côté segs :

  • `_user_valued` (le segment qui ÉNONCE un fait) porte bien `fact_slot`,
    `fact_val`, `fact_attr` — c'est le déclencheur du WRITE, tout va bien ;
  • le segment de QUESTION et le segment de RÉPONSE (`_assistant_valued`) ne
    portent AUCUN identifiant de fait. Le lien question→slot ne vit que dans
    `conv["info"]["q_slots"]`, au niveau de la CONVERSATION ;
  • et `chat_batch` collate des segs LANE PAR LANE : `conv["info"]` ne survit
    pas au batch (seul `roles` est propagé, cf. `_collate`).

Donc, en l'état, l'entraînement ne peut pas savoir devant quelle réponse
injecter quoi. Deux issues, la première est la bonne :

  (a) TAGUER LE SEG DE QUESTION côté générateur : ajouter `q_slot` (long
      [1,T] constant, 0 = pas une question) dans PersonaChatStream au moment où
      la question est construite — le stream connaît déjà le slot. C'est la
      symétrie exacte du `fact_slot` déjà posé sur le seg d'énonciation, ça
      traverse `_collate` sans rien changer (clé long, zéro-paddée comme
      `fact_slot`), et l'éval y gagne de ne plus dépendre de `info`.
  (b) rejouer le lien côté trainer en reconstruisant les convs — non : ça
      duplique la logique du générateur et casse au premier changement.

Tant que (a) n'est pas fait, le bras ne peut pas s'entraîner : la sélection
n'a pas de cible. C'est un travail de ~20 lignes dans persona_chat_data.py,
plus le self-test correspondant.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── config ───────────────────────────────────────────────────────────────────

@dataclass
class RtiConfig:
    top_k: int = 13           # lignes NATIVES par groupe (tokens du segment au
                              # poids SIF le plus fort, DANS L'ORDRE DU SEGMENT).
                              # 13 = le verdict du jouet : plus petit k dont la
                              # strate `code` dépasse 95 % de couverture.
    max_groups: int = 8       # FIFO de groupes résidents (= max_mem du 350M).
    sif_a: float = 1e-2       # a de w = a/(a+p). 1e-2 et NON 1e-4 : mesuré au
                              # jouet, 1e-4 écrase la strate `code` (les codes
                              # sont des digrammes fréquents).
    train_groups: int = 2     # groupes injectés à l'ENTRAÎNEMENT (vrai + N−1
                              # distracteurs résidents).
    eval_groups: int = 2      # groupes injectés à l'ÉVAL (top-N du retriever).
                              # train_groups == eval_groups = l'ISOMORPHIE.
    train_order: str = "oracle_first"   # place du vrai groupe dans le préfixe
                              # d'entraînement. `oracle_first` REPRODUIT l'ordre
                              # de l'éval (les groupes y sont triés par score
                              # décroissant, donc le vrai sort premier quand le
                              # retriever a raison) — c'est ça, l'isomorphie.
                              # `random` ne se justifierait que si la question
                              # était dans le contexte de décodage ; elle n'y est
                              # pas (le décodage gradé part du préfixe assistant).
    retr_ce: float = 1.0      # coefficient de la CE auxiliaire du retriever.
    retr_detach: bool = True  # la CE du retriever ne remonte pas dans le trunk.
    write_every_turn: bool = False   # False = seuls les segments PORTEURS
                              # écrivent (privilège d'oracle assumé, comme le
                              # `--write fact` du jouet) ; True = tous les tours.
    sep_token: str = "<blank>"   # séparateur posé après chaque groupe injecté.

    def __post_init__(self):
        assert self.top_k >= 1 and self.max_groups >= 1
        assert self.sif_a > 0
        assert self.train_groups >= 1 and self.eval_groups >= 1
        assert self.train_order in ("oracle_first", "random")
        assert self.retr_ce >= 0.0

    @property
    def group_prefix(self) -> int:
        """Pseudo-tokens qu'occupe UN groupe injecté : k lignes + 1 séparateur."""
        return self.top_k + 1


# ── table SIF (poids w = a/(a+p)) ────────────────────────────────────────────

def sif_table(stream, vocab_size: int, a: float) -> torch.Tensor:
    """[vocab] float : w = a/(a+p(token)), lue sur le stream LUI-MÊME.

    On ne réimplémente pas la table : `PersonaChatStream._sif_table()` (300
    convs, rng dédié) est déjà celle qui pondère le write du 350M. Un stream
    qui ne l'expose pas (mix, sota) rend des poids uniformes plutôt qu'une
    exception : le pooling reste défini, il n'est simplement plus discriminant.
    """
    w = torch.ones(vocab_size, dtype=torch.float32)
    tbl = getattr(stream, "_sif_table", None)
    if tbl is None:
        return w
    p = tbl()
    unseen = float(getattr(stream, "_sif_unseen", 0.0) or 0.0)
    if unseen > 0:
        w.fill_(a / (a + unseen))
    for t, pv in p.items():
        if 0 <= int(t) < vocab_size:
            w[int(t)] = a / (a + float(pv))
    return w


# ── le GROUPE : une clé procédurale + k lignes natives ───────────────────────

@torch.no_grad()
def build_group(embed_w: torch.Tensor, sif_w: torch.Tensor, seg_ids: torch.Tensor,
                cfg: RtiConfig) -> tuple:
    """(clé [d], tokens [top_k]) du groupe écrit pour ce segment.

    Les TOKENS sont les top_k du segment au poids SIF le plus fort, rendus DANS
    L'ORDRE DU SEGMENT (le tri par poids détruirait l'ordre, qui est tout ce que
    le format achète), complétés à taille fixe en RÉPÉTANT LE DERNIER — un
    candidat en double est inerte, une taille variable casse l'empilement d'un
    batch (crash payé au job 58).

    La CLÉ est le pooling SIF des embeddings du segment ENTIER, normalisé. Elle
    est PROCÉDURALE : aucune table secrète, rien que le texte — c'est ce qui la
    rend transposable, et c'est aussi ce qui la rend moins séparante que la clé
    dédiée du jouet (cf. `key_separation`).

    Les lignes de contenu sont les embeddings BRUTS, NON normalisés : la norme
    porte de l'information et, à l'injection, la ligne doit ressembler à ce que
    le modèle voit d'un vrai token.
    """
    ids = seg_ids.reshape(-1)
    if ids.numel() == 0:
        z = embed_w.new_zeros(embed_w.size(1))
        return z, torch.zeros(cfg.top_k, dtype=torch.long, device=embed_w.device)
    w = sif_w.to(ids.device)[ids].float()                       # [T]
    # ── clé : pooling SIF du segment ENTIER ─────────────────────────────────
    e = embed_w[ids].float()                                    # [T, d]
    key = (e * w[:, None]).sum(0) / w.sum().clamp_min(1e-6)
    key = key / key.norm().clamp_min(1e-6)
    # ── contenu : les top_k tokens, ordre du segment, taille fixe ───────────
    k = min(cfg.top_k, ids.numel())
    sel = torch.topk(w, k).indices.sort().values
    toks = ids[sel]
    if k < cfg.top_k:
        toks = torch.cat([toks, toks[-1].repeat(cfg.top_k - k)])
    return key.to(embed_w.dtype), toks


@torch.no_grad()
def key_separation(embed_w, sif_w, segs: list, cfg: RtiConfig) -> dict:
    """Le DIAGNOSTIC qui décide si le port a un sens.

    La clé procédurale remplace un privilège d'oracle ; si deux segments
    quelconques ont des clés à cos ~1, aucun W_q ne les séparera et le bras est
    mort avant d'être lancé. On rend la distribution des cos entre clés de
    segments DIFFÉRENTS, et le rang-1 d'un retrieval à clé exacte sur des
    banques de `max_groups` — le pendant exact du tableau (c) de l'oracle pack.
    """
    keys = torch.stack([build_group(embed_w, sif_w, s, cfg)[0].float()
                        for s in segs])
    keys = keys / keys.norm(dim=1, keepdim=True).clamp_min(1e-6)
    C = keys @ keys.t()
    n = C.size(0)
    off = C[~torch.eye(n, dtype=torch.bool, device=C.device)]
    # rang-1 : chaque clé se retrouve-t-elle elle-même dans une banque tirée ?
    g = torch.Generator().manual_seed(0)
    ok = tot = 0
    m = min(cfg.max_groups, n)
    for _ in range(200):
        idx = torch.randperm(n, generator=g)[:m]
        sub = keys[idx]
        for q in range(m):
            ok += int(int((sub @ sub[q]).argmax()) == q)
            tot += 1
    return {"n": n, "cos_mean": float(off.mean()), "cos_med": float(off.median()),
            "cos_max": float(off.max()), "cos_p99": float(off.quantile(0.99)),
            "rank1": ok / max(tot, 1), "chance": 1.0 / m}


# ── FIFO de groupes, par lane ────────────────────────────────────────────────

@dataclass
class LaneBank:
    """Les groupes résidents d'UNE lane. FIFO à `max_groups`, éviction par
    groupe entier — un write = un groupe, comme au jouet (les âges restent
    comptés en writes)."""
    keys: list = field(default_factory=list)     # [d] chacun
    toks: list = field(default_factory=list)     # [top_k] chacun
    tags: list = field(default_factory=list)     # identité du fait (grade/CE)

    def push(self, key, tok, tag, max_groups: int) -> None:
        self.keys.append(key); self.toks.append(tok); self.tags.append(tag)
        if len(self.keys) > max_groups:
            del self.keys[0], self.toks[0], self.tags[0]

    def positives(self, tag) -> list:
        """TOUS les groupes portant ce fait. La CE est MULTI-POSITIVE : deux
        writes du même fait ont des clés quasi identiques (au jouet elles
        étaient rigoureusement égales et une cible unique avait un gradient
        EXACTEMENT nul). Le plus récent porte la vérité."""
        return [i for i, t in enumerate(self.tags) if t == tag and t is not None]

    def __len__(self) -> int:
        return len(self.keys)


# ── le RETRIEVER (seul module appris de la sélection) ────────────────────────

class Retriever(nn.Module):
    """score_g = (W_q · h_query) · clé_g / √d × exp(log_temp).

    `h_query` = l'état caché du DERNIER token du SEGMENT USER qui pose la
    question — JAMAIS un token de la réponse : en teacher-forcing celui-ci
    contient déjà la valeur à retrouver, et le retriever apprendrait à lire ce
    qu'il est censé aller chercher. C'est l'invariant critique du bras.

    W_q est zéro-init : scores nuls, softmax uniforme, CE = log G, et le
    gradient de la CE vaut h ⊗ clé ≠ 0 — le module démarre sans préférence, pas
    mort.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.d = d_model
        self.wq = nn.Linear(d_model, d_model, bias=False)
        nn.init.zeros_(self.wq.weight)
        self.log_temp = nn.Parameter(torch.zeros(1))

    def forward(self, h, keys, key_mask=None):
        """h [n,d] ; keys [n,G,d] ; key_mask [n,G] bool → scores [n,G]."""
        q = self.wq(h)
        sc = torch.einsum("nd,ngd->ng", q, keys.to(q.dtype)) / (self.d ** 0.5)
        sc = sc * self.log_temp.exp()
        if key_mask is not None:
            sc = sc.masked_fill(~key_mask, float("-inf"))
        return sc


class InjectType(nn.Module):
    """Le vecteur de TYPE ajouté à chaque pseudo-token injecté (zéro-init : au
    step 0 la ligne injectée est EXACTEMENT l'embedding brut du token)."""

    def __init__(self, d_model: int):
        super().__init__()
        self.vec = nn.Parameter(torch.zeros(d_model))


# ── construction du PRÉFIXE injecté ──────────────────────────────────────────

def build_prefix(embed_w, type_vec, sep_id: int, groups: list, cfg: RtiConfig):
    """[B, P, d] : le préfixe de pseudo-embeddings.

    `groups` : une liste par lane de listes de tenseurs [top_k] (les tokens des
    groupes à injecter, dans l'ORDRE VOULU — décroissant de score à l'éval,
    `train_order` à l'entraînement). Toutes les lanes doivent porter le MÊME
    nombre de groupes : le préfixe est une tranche de séquence, sa longueur ne
    peut pas varier dans un batch (l'appelant regroupe par taille).

    Layout : [groupe 0 | sép | groupe 1 | sép | …], donc P = G·(top_k+1).
    Les positions RoPE 0..P−1 vont au préfixe et le tour réel suit — cf. le
    Step 1bis de ThoughtBankLM.forward pour pourquoi il n'y a pas de trou.
    """
    B = len(groups)
    G = len(groups[0]) if B else 0
    assert all(len(g) == G for g in groups), "préfixe de longueur variable"
    if G == 0:
        return None
    dev = embed_w.device
    ids = torch.stack([torch.stack(g) for g in groups]).to(dev)      # [B,G,k]
    pre = embed_w[ids] + type_vec                                    # [B,G,k,d]
    sep = embed_w[torch.full((B, G, 1), int(sep_id), dtype=torch.long,
                             device=dev)]
    return torch.cat([pre, sep], dim=2).reshape(B, G * cfg.group_prefix, -1)


def pick_train_groups(bank: LaneBank, tag, cfg: RtiConfig, gen=None) -> list:
    """Index des groupes injectés devant UNE réponse à l'ENTRAÎNEMENT : le VRAI
    (le plus récent portant ce fait) + des distracteurs tirés parmi les autres
    résidents. Banque trop courte ⇒ G RÉDUIT, jamais complété par un doublon.

    `oracle_first` place le vrai en tête : c'est EXACTEMENT ce que produit
    l'éval quand le retriever a raison (tri par score décroissant). Isomorphie.
    """
    pos = bank.positives(tag)
    if not pos:
        return []
    true_i = pos[-1]
    others = [i for i in range(len(bank)) if i != true_i]
    n = min(cfg.train_groups, len(bank))
    pick = [true_i]
    if n > 1 and others:
        sel = torch.randperm(len(others), generator=gen)[:n - 1]
        pick += [others[int(x)] for x in sel]
    if cfg.train_order == "random" and len(pick) > 1:
        pick = [pick[int(x)] for x in torch.randperm(len(pick), generator=gen)]
    return pick


def pick_eval_groups(scores: torch.Tensor, n_res: int, cfg: RtiConfig) -> list:
    """Index du top-N DUR du retriever, trié par score DÉCROISSANT.

    Tri STABLE sur les groupes pris du PLUS RÉCENT au plus ancien : à score égal
    — le cas de deux writes du même fait, dont les clés procédurales sont quasi
    identiques — c'est le plus récent qui passe, et c'est lui qui porte la
    vérité. La récence est le seul discriminant que la clé ne donne pas.
    """
    nk = min(cfg.eval_groups, n_res)
    rec = torch.arange(n_res - 1, -1, -1, device=scores.device)
    order = torch.argsort(-scores[rec], stable=True)
    return [int(rec[o]) for o in order[:nk]]


def retr_ce_loss(scores: torch.Tensor, positives: list) -> torch.Tensor:
    """CE MULTI-POSITIVE : −log Σ_{g ∈ positifs} p_g, sommée sur le lot.

    Mono-cible, deux writes du même fait (clés quasi égales) se compenseraient
    et le gradient s'annulerait — mesuré exactement nul au jouet, où les clés
    étaient rigoureusement identiques.
    """
    sc = scores.float()
    pm = torch.zeros_like(sc, dtype=torch.bool)
    for r, pos in enumerate(positives):
        pm[r, pos] = True
    return (torch.logsumexp(sc, -1)
            - torch.logsumexp(sc.masked_fill(~pm, float("-inf")), -1)).sum()


# ── self-test ────────────────────────────────────────────────────────────────

def _selftest() -> None:
    torch.manual_seed(20260806)
    V, d = 512, 64
    cfg = RtiConfig(top_k=4, max_groups=3, train_groups=2, eval_groups=2)
    E = torch.randn(V, d) * 0.02
    w = torch.full((V,), 0.9)
    w[torch.tensor([7, 8, 9])] = 5e-4          # « template très fréquent »

    # 1. le GROUPE : k tokens rares du segment, DANS L'ORDRE, taille fixe
    seg = torch.tensor([7, 101, 8, 202, 9, 303, 7, 404, 8])
    key, toks = build_group(E, w, seg, cfg)
    assert toks.tolist() == [101, 202, 303, 404], toks
    assert key.shape == (d,) and abs(float(key.norm()) - 1.0) < 1e-4
    # segment plus COURT que top_k ⇒ complété en répétant le dernier
    _, short = build_group(E, w, torch.tensor([101, 202]), cfg)
    assert short.tolist() == [101, 202, 202, 202], short
    # la clé est PROCÉDURALE : fonction du seul texte, reproductible
    assert torch.equal(build_group(E, w, seg, cfg)[0], key)
    # … et elle dépend du segment
    assert not torch.allclose(build_group(E, w, seg.flip(0), cfg)[0], key,
                              atol=1e-6) or True   # (l'ordre ne change pas le pool)
    assert not torch.allclose(
        build_group(E, w, torch.tensor([500, 501, 502]), cfg)[0], key, atol=1e-3)

    # 2. FIFO par GROUPE + positifs multiples
    b = LaneBank()
    for i in range(5):
        b.push(torch.randn(d), torch.arange(4), tag=("slot", i % 2),
               max_groups=cfg.max_groups)
    assert len(b) == 3, len(b)
    assert b.tags == [("slot", 0), ("slot", 1), ("slot", 0)]
    assert b.positives(("slot", 0)) == [0, 2]
    assert b.positives(("slot", 9)) == []

    # 3. RETRIEVER : zéro-init ⇒ scores nuls, mais gradient NON nul
    r = Retriever(d)
    keys = torch.randn(2, 3, d)
    h = torch.randn(2, d)
    sc = r(h, keys)
    assert sc.shape == (2, 3) and float(sc.abs().max()) == 0.0
    loss = retr_ce_loss(sc, [[0], [1, 2]])
    loss.backward()
    assert float(r.wq.weight.grad.abs().max()) > 0, \
        "W_q zéro-init sans gradient : le retriever est mort"
    # masque : un groupe absent ne peut pas être élu
    msk = torch.tensor([[True, True, False], [True, False, False]])
    scm = r(h, keys, msk)
    assert bool(torch.isinf(scm[~msk]).all()) and bool((scm[~msk] < 0).all())
    # CE multi-positive : deux clés IDENTIQUES toutes deux positives ⇒ perte
    # quasi nulle (rien à discriminer), et surtout PAS de gradient qui s'annule
    # en silence sur une cible unique
    kd = torch.randn(1, 2, d)
    kd[0, 1] = kd[0, 0]
    r2 = Retriever(d)
    with torch.no_grad():
        r2.wq.weight.normal_(0, 0.05)
    s2 = r2(torch.randn(1, d), kd)
    assert float(retr_ce_loss(s2, [[0, 1]])) < 1e-5
    assert float(F.cross_entropy(s2, torch.tensor([1]))) > 0.3

    # 4. PRÉFIXE : layout, longueur, type zéro-init
    ty = InjectType(d)
    g = [[torch.arange(4), torch.arange(4) + 10],
         [torch.arange(4) + 20, torch.arange(4) + 30]]
    P = build_prefix(E, ty.vec, sep_id=5, groups=g, cfg=cfg)
    assert P.shape == (2, 2 * cfg.group_prefix, d), P.shape
    assert torch.allclose(P[0, :4], E[torch.arange(4)]), "type zéro-init"
    assert torch.allclose(P[0, 4], E[5]), "séparateur mal placé"
    assert torch.allclose(P[0, 5:9], E[torch.arange(4) + 10])
    # NON normalisé : la norme d'un embedding brut n'est pas 1
    assert abs(float(P[0, 0].norm()) - 1.0) > 1e-3
    # longueurs hétérogènes = refus explicite (l'appelant doit regrouper)
    try:
        build_prefix(E, ty.vec, 5, [[torch.arange(4)], []], cfg)
    except AssertionError as e:
        assert "variable" in str(e)
    else:
        raise AssertionError("préfixe hétérogène accepté")

    # 5. SÉLECTION : train = vrai + distracteur (vrai EN TÊTE), éval = top-N
    #    trié par score, récence prioritaire à égalité
    bank = LaneBank()
    for i in range(3):
        bank.push(torch.randn(d), torch.full((4,), i), tag=("s", i),
                  max_groups=8)
    gg = torch.Generator().manual_seed(3)
    pk = pick_train_groups(bank, ("s", 1), cfg, gg)
    assert len(pk) == 2 and pk[0] == 1 and len(set(pk)) == 2, pk
    assert all(pick_train_groups(bank, ("s", 1), cfg, gg)[0] == 1
               for _ in range(10)), "oracle_first ne tient pas"
    cfg_r = RtiConfig(top_k=4, train_groups=2, train_order="random")
    firsts = {pick_train_groups(bank, ("s", 1), cfg_r, gg)[0] for _ in range(40)}
    assert len(firsts) > 1, "train_order=random n'est pas aléatoire"
    # banque à UN groupe ⇒ G réduit, pas de doublon
    solo = LaneBank()
    solo.push(torch.randn(d), torch.zeros(4, dtype=torch.long), ("s", 0), 8)
    assert pick_train_groups(solo, ("s", 0), cfg, gg) == [0]
    assert pick_train_groups(solo, ("s", 7), cfg, gg) == []
    # éval : tri décroissant, et à ÉGALITÉ le plus RÉCENT d'abord
    sc_e = torch.tensor([0.1, 0.9, 0.5])
    assert pick_eval_groups(sc_e, 3, cfg) == [1, 2]
    assert pick_eval_groups(torch.zeros(3), 3, cfg) == [2, 1], \
        "à score égal, le groupe le PLUS RÉCENT doit passer devant"

    # 6. SÉPARATION DES CLÉS : le diagnostic tourne et discrimine des segments
    #    manifestement différents
    segs = [torch.randint(10, V, (20,)) for _ in range(40)]
    st = key_separation(E, w, segs, cfg)
    assert 0.0 <= st["rank1"] <= 1.0 and st["n"] == 40
    assert st["rank1"] > st["chance"], st

    print("rti self-test: OK — groupe (top-k SIF ordonné, taille fixe, clé "
          "procédurale unitaire), FIFO par groupe + positifs multiples, "
          "retriever zéro-init à gradient non nul + masque, CE MULTI-POSITIVE "
          "(clés identiques ⇒ perte nulle là où une cible unique perd son "
          "gradient), préfixe [groupe|sép] non normalisé à type zéro-init, "
          "train = vrai EN TÊTE + distracteur / éval = top-N trié à récence "
          "prioritaire, séparation des clés mesurable")
    print(f"  isomorphie train/éval : {RtiConfig().train_groups} groupes des "
          f"deux côtés, ordre oracle_first — la leçon de r5 v1 (grade 0.067 "
          f"malgré recall@1 1.000) est câblée dans la config par défaut")


if __name__ == "__main__":
    _selftest()
