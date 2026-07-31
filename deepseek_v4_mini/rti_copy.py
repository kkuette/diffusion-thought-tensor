"""COPY-HEAD — la classe de fonctions qui manque au read du 350M.

LE MOTIF (diagnostic offline @800 du run sft_recall_350m_rti, 2026-07-31)
────────────────────────────────────────────────────────────────────────
La chaîne RTI est CORRECTE et le diagnostic l'a établi de bout en bout : les
tokens de la vérité sont dans le groupe injecté, DANS L'ORDRE, EN TÊTE du
préfixe. Et pourtant, même TEACHER-FORCÉ, l'argmax lâche à la continuation de
sous-mot :

    « Phil…»   → « Pl…»          « mosaic »  → « mask »
    « Cann »+« oli » → « Cann »+« stra »

Le contrôle qui ferme la question : LES MÊMES ERREURS depuis un contexte PLAT
natif, sans banque du tout. Ce backbone n'a pas de tête de copie littérale, et
ça ne s'apprend pas en 800 steps de SFT — la nll des tokens VALEUR est restée
PLATE de 200 à 800 pendant que celle du template passait 3.0 → 1.7. Le gradient
du canal LM apprend le format, pas la citation.

Ce n'est donc pas un problème d'optimisation ni de données : c'est un problème
de CLASSE DE FONCTIONS. Le préfixe injecté rend la cible triviale — les tokens
à citer sont là, on connaît leurs ids — mais le modèle n'a aucun chemin qui
puisse dire « émets le token de la position j du préfixe ». Ce module ajoute ce
chemin.

LE MÉCANISME : pointer-generator minimal (See et al. 2017), aux LOGITS
──────────────────────────────────────────────────────────────────────
Aux positions de réponse, on mélange DEUX distributions :

    P(v)  =  (1 − p_copy) · softmax(logits_lm)[v]  +  p_copy · P_copy(v)

    α_t   = softmax_P( (W_c h_t) · h_préfixe / √d )      attention de copie
    P_copy(v) = Σ_{j : prefix_ids[j] = v} α_t[j]         scatter sur le vocab
    p_copy    = σ(w_g · h_t + b_g)                       la porte

Trois choix qui ne sont pas décoratifs :

  1. LE MÉLANGE EST DANS LE LOG, pas dans les logits. On somme des
     PROBABILITÉS puis on prend le log (vraisemblance marginale d'un modèle à
     variable latente « ce token vient-il du préfixe ? »). Mélanger des logits
     n'aurait aucun sens probabiliste et le gradient de la porte serait faux.
  2. b_g EST TRÈS NÉGATIF À L'INIT (−4 ⇒ p_copy ≈ 0.018). Au step 0 le chemin
     est QUASI NEUTRE : le run ne subit pas un choc de distribution, la porte
     s'ouvre si et seulement si la copie paie. C'est le même réflexe que le
     zéro-init de W_q et du vecteur de type.
  3. LES h DU PRÉFIXE SONT CEUX DE LA DERNIÈRE COUCHE, déjà calculés par le
     forward (le préfixe traverse tout le stack, cf. model Step 1bis) — on les
     récupère AVANT la tranche `H_text[:, npre:]`. Zéro recalcul.

Coût : W_c (d×d) + porte (d+1) = 590 593 params à d=768, soit +0.17 % du 350M.
Coût STRICTEMENT NUL quand `prefix_ids is None` (donc pour tout le reste du
dépôt, et pour le bras ablaté) : le chemin est bit-identique à l'existant, et
c'est un invariant self-testé (`model._selftest`, cas (a) ici).

CE QUE LE MODULE REND, ET POURQUOI C'EST DES LOG-PROBS
─────────────────────────────────────────────────────
`mix_logprobs` rend log P(·), une distribution NORMALISÉE, à la place des
logits. C'est exact pour les deux consommateurs et ça ne demande à aucun d'eux
de changer de code :

  * la CE : `cross_entropy(x)` = `−log_softmax(x)[y]`, et
    `log_softmax(log P) = log P` puisque log P est déjà normalisé ⇒ la CE lit
    la vraie −log P(y) sans le savoir ;
  * le décodage : `argmax(log P) = argmax(P)`, et `softmax(log P / τ)` est la
    loi tempérée du MÉLANGE — exactement ce qu'on veut échantillonner.

Le forward pose `out["logits_are_logprobs"] = True` pour que rien ne se croie
en face de logits bruts (un `.exp()` naïf sur des logits serait une erreur
silencieuse ; ici il rend des probabilités justes).

    python -m deepseek_v4_mini.rti_copy      # self-test (CPU, ~1 min)
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── config ───────────────────────────────────────────────────────────────────

@dataclass
class CopyHeadConfig:
    enabled: bool = False
    gate_bias_init: float = -4.0   # σ(−4) = 0.018 : le chemin de copie est
                                   # QUASI NEUTRE au step 0. Ce n'est pas de la
                                   # prudence rituelle — un run est en cours et
                                   # un bras qui repart d'un ckpt ne doit pas
                                   # voir sa distribution basculer au step 1.
    sentinel_id: int = -1          # valeur de `prefix_ids` marquant une position
                                   # NON copiable (séparateurs). Négative : elle
                                   # ne peut pas collisionner avec un vrai id.
    wc_init: str = "zero"          # "zero" : scores nuls ⇒ α uniforme sur le
                                   # préfixe, mais gradient NON nul (∂s/∂W_c =
                                   # h_t ⊗ h_pre) — le module démarre sans
                                   # préférence, pas mort. "identity" : α part
                                   # d'une similarité h_t·h_pre brute, un prior
                                   # de MATCH (et non de SUCCESSEUR), qui n'est
                                   # pas ce qu'une tête d'induction fait.
    temp_init: float = 0.0         # log-température de α (0 ⇒ ×1).

    def __post_init__(self):
        assert self.wc_init in ("zero", "identity"), self.wc_init
        assert self.sentinel_id < 0, \
            "sentinel_id doit être négatif (sinon il masque un vrai token)"

    @classmethod
    def from_raw(cls, raw) -> "CopyHeadConfig":
        """Depuis le YAML (`rti.copy_head:`). Une clé inconnue lève — la
        dataclass est le gardien réel, `cfg_schema` ne fait que l'avancer au
        `--check`."""
        return cls(**dict(raw or {}))


# ── le module ────────────────────────────────────────────────────────────────

class CopyHead(nn.Module):
    """W_c (d×d) + une porte (d+1). Rien d'autre.

    Appelée par `ThoughtBankLM.forward` quand — et seulement quand —
    `prefix_ids` est fourni ; sinon elle n'est même pas touchée.
    """

    def __init__(self, d_model: int, cfg: CopyHeadConfig | None = None):
        super().__init__()
        self.cfg = cfg or CopyHeadConfig()
        self.d = int(d_model)
        self.wc = nn.Linear(d_model, d_model, bias=False)
        if self.cfg.wc_init == "zero":
            nn.init.zeros_(self.wc.weight)
        else:
            with torch.no_grad():
                self.wc.weight.copy_(torch.eye(d_model))
        # La porte est faite de DEUX PARAMÈTRES 1-D et non d'un nn.Linear(d, 1).
        # Un Linear(d, 1) porte un poids de forme [1, d] : `_split_muon_params`
        # ne regarde que `p.ndim == 2` et l'enverrait à Muon, qui
        # orthogonaliserait un VECTEUR DÉGUISÉ EN MATRICE — le piège √cols, payé
        # une fois déjà sur les hypernets du read. w_g [d] et b_g scalaire vont
        # à AdamW comme tout le reste des 1-D, sans qu'on ait rien à déclarer.
        self.w_g = nn.Parameter(torch.zeros(d_model))
        self.b_g = nn.Parameter(torch.tensor(float(self.cfg.gate_bias_init)))
        self.log_temp = nn.Parameter(torch.full((1,), float(self.cfg.temp_init)))
        # télémétrie de la dernière passe (hors gradient) — le trainer la lit
        # pour savoir si la porte S'OUVRE, seule preuve que la copie sert.
        #
        # Ce sont des TENSEURS 0-d détachés et non des floats : matérialiser
        # ici coûtait une SYNCHRO DE DEVICE par forward, et le bras rti fait un
        # forward par part et par seg (~40/step) dans un régime déjà
        # launch-bound. `float(x)` et `f"{x:.3f}"` marchent tels quels sur un
        # tenseur 0-d, donc aucun lecteur ne change.
        self.last_p_copy = None
        self.last_alpha_max = None
        # `last_gate` [B,T] : p_copy POSITION PAR POSITION. C'est elle qui
        # permet de séparer « la porte s'ouvre sur la VALEUR » de « la porte
        # s'ouvre partout » — la moyenne globale ne distingue pas les deux, et
        # c'est précisément la distinction que le run doit trancher.
        self.last_gate = None

    # ── α : l'attention de copie sur les positions du préfixe ───────────────

    def attn(self, h, h_pre, prefix_ids):
        """log α [B,T,P]. Les positions sentinelles sont à −inf : leur masse ne
        dilue même pas le softmax (les masquer APRÈS aurait laissé α non
        normalisé)."""
        q = self.wc(h.float())                                   # [B,T,d]
        s = torch.einsum("btd,bpd->btp", q, h_pre.float()) / (self.d ** 0.5)
        s = s * self.log_temp.exp().float()
        ok = prefix_ids >= 0                                      # [B,P]
        s = s.masked_fill(~ok[:, None, :], float("-inf"))
        return torch.log_softmax(s, dim=-1)

    def forward(self, h, h_pre, prefix_ids, lm_logits,
                alpha_override=None, gate_override=None):
        """log P(·) mélangées [B,T,V].

        `alpha_override` [B,T,P] / `gate_override` [B,T] ou scalaire : outils de
        DIAGNOSTIC (self-test oracle, ablations). Jamais utilisés en
        entraînement — ils court-circuitent précisément ce qu'on veut apprendre.
        """
        loga = (self.attn(h, h_pre, prefix_ids) if alpha_override is None
                else alpha_override.float().clamp_min(1e-30).log())
        if gate_override is None:
            g = h.float() @ self.w_g.float() + self.b_g.float()   # [B,T]
            log_pc, log_1mpc = F.logsigmoid(g), F.logsigmoid(-g)
        else:
            pc = torch.as_tensor(gate_override, dtype=torch.float32,
                                 device=h.device).expand(h.shape[:2])
            log_pc = pc.clamp_min(1e-30).log()
            log_1mpc = (1.0 - pc).clamp_min(1e-30).log()
        self.last_gate = log_pc.exp().detach()                    # [B,T]
        self.last_p_copy = self.last_gate.mean()
        self.last_alpha_max = loga.exp().max(-1).values.mean().detach()
        return mix_logprobs(lm_logits, loga, log_pc, log_1mpc, prefix_ids)


# ── le mélange, hors module (testable seul, réutilisable au décodage) ────────

def mix_logprobs(lm_logits, log_alpha, log_pc, log_1mpc, prefix_ids):
    """log[(1−p)·P_lm + p·P_copy] — exact, et SANS tenseur [B,T,V] de copie.

    LE PIÈGE QU'ON ÉVITE : matérialiser P_copy dense sur le vocabulaire coûte un
    [B,T,V] float32 de plus (≈ 1 Go à B=8, T=640, V=49k) alors que la copie ne
    touche QUE P ≤ 28 colonnes. On corrige donc `base = log(1−p) + log P_lm`
    uniquement sur ces colonnes, par gather → logaddexp → scatter.

    LE PIÈGE DANS LE PIÈGE : un id peut apparaître DEUX FOIS dans le préfixe
    (`build_group` complète les groupes courts en répétant le dernier token, et
    deux groupes peuvent partager un mot). Un `scatter_` avec index dupliqué a
    un gagnant NON SPÉCIFIÉ. On agrège donc α par CLASSE D'ÉGALITÉ D'ID avant le
    scatter — `agg[j] = Σ_{k : ids_k = ids_j} α_k`, pour TOUTES les positions j,
    pas seulement la première — de sorte que les écritures concurrentes portent
    la MÊME valeur et que le conflit devienne inoffensif. Les α s'additionnent,
    ce qui est la sémantique voulue : deux pointeurs vers le même mot, c'est
    deux fois plus de masse sur ce mot.

    LE TROISIÈME PIÈGE, payé au self-test de `model` : rabattre les sentinelles
    sur l'index 0 (un `clamp_min(0)` naïf) les fait ÉCRIRE sur le token 0. Si le
    token 0 est par ailleurs un vrai token du préfixe, la sentinelle réécrit
    `base` par-dessus la valeur corrigée et la masse de copie DISPARAÎT — un
    trou de normalisation mesuré à 4.5e-3, silencieux, et d'autant plus vicieux
    que le token 0 est souvent réservé (pad/unk). Les sentinelles visent donc
    une COLONNE POUBELLE ajoutée en bout de vocabulaire, qu'on retire ensuite.
    """
    lm_lp = torch.log_softmax(lm_logits.float(), dim=-1)          # [B,T,V]
    V = lm_lp.size(-1)
    base = F.pad(lm_lp + log_1mpc[..., None], (0, 1))             # [B,T,V+1]
    B, T, P = log_alpha.shape
    ok = prefix_ids >= 0                                          # [B,P]
    eq = (prefix_ids[:, :, None] == prefix_ids[:, None, :])
    eq = (eq & ok[:, None, :] & ok[:, :, None]).to(log_alpha.dtype)   # [B,P,P]
    agg = torch.einsum("btp,bpq->btq", log_alpha.exp(), eq)       # [B,T,P]
    tgt = torch.where(ok, prefix_ids, torch.full_like(prefix_ids, V))
    tgt = tgt[:, None, :].expand(B, T, P)                         # [B,T,P]
    hit = torch.logaddexp(base.gather(2, tgt),
                          log_pc[..., None] + agg.clamp_min(1e-30).log())
    return base.scatter(2, tgt, hit)[..., :V]


def copy_mix_logprobs(model, out, prefix_ids, h_prefix=None):
    """Le mélange APRÈS coup, pour un appelant qui a déjà son `out` de forward.

    Sert quand le forward n'a pas pu faire le mélange lui-même (décodage avec
    cache, où le préfixe n'entre qu'au premier pas : `h_prefix` est alors celui
    mémorisé du pas 0). Le chemin nominal reste le forward, qui pose déjà
    `out["logits_are_logprobs"]`.
    """
    head = getattr(model, "rti_copy", None)
    if head is None or prefix_ids is None:
        return out["logits"]
    hp = out["h_prefix"] if h_prefix is None else h_prefix
    return head(out["hidden"], hp, prefix_ids, out["logits"])


# ── self-test ────────────────────────────────────────────────────────────────

class _ToyLM(nn.Module):
    """Un LM jouet au CONTRAT de ThoughtBankLM pour la partie qui nous occupe :
    il embed, il empile un transformer minuscule, il rend des logits — et il
    accepte un préfixe injecté qu'il retranche de sa sortie (Step 1bis).

    Il sert à prouver la SÉPARATION DES CLASSES DE FONCTIONS : entraîné sur des
    littéraux rares qu'il n'a aucune raison de savoir produire, il doit rester
    plat SANS tête de copie et s'effondrer AVEC.
    """

    def __init__(self, V=64, d=32, copy: CopyHead | None = None):
        super().__init__()
        self.emb = nn.Embedding(V, d)
        self.blk = nn.TransformerEncoderLayer(d, 2, 2 * d, batch_first=True,
                                              dropout=0.0)
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, V, bias=False)
        self.rti_copy = copy

    def forward(self, ids, inject=None, prefix_ids=None):
        h = self.emb(ids)
        npre = 0
        if inject is not None:
            npre = inject.size(1)
            h = torch.cat([inject, h], dim=1)
        T = h.size(1)
        m = torch.triu(torch.ones(T, T, dtype=torch.bool), 1)
        h = self.norm(self.blk(h, src_mask=m, is_causal=False))
        h_pre = h[:, :npre] if npre else None
        h = h[:, npre:]
        lg = self.head(h)
        out = {"logits": lg, "hidden": h, "h_prefix": h_pre}
        if self.rti_copy is not None and prefix_ids is not None \
                and h_pre is not None:
            out["logits"] = self.rti_copy(h, h_pre, prefix_ids, lg)
            out["logits_are_logprobs"] = True
        return out


def _selftest() -> None:
    torch.manual_seed(20260731)
    B, T, P, V, d = 3, 7, 10, 64, 32

    h = torch.randn(B, T, d)
    h_pre = torch.randn(B, P, d)
    lg = torch.randn(B, T, V)
    ids = torch.randint(0, V, (B, P))
    ids[:, 4] = -1                       # séparateurs = sentinelles
    ids[:, 9] = -1
    head = CopyHead(d, CopyHeadConfig(enabled=True))

    # ── (a) chemin OFF : prefix_ids=None ⇒ BIT-identique au forward nu ──────
    m_off = _ToyLM(V, d, copy=None).eval()
    m_on = _ToyLM(V, d, copy=CopyHead(d, CopyHeadConfig(enabled=True))).eval()
    m_on.load_state_dict(m_off.state_dict(), strict=False)
    x = torch.randint(0, V, (B, T))
    inj = torch.randn(B, P, d)
    pid = ids.clone()
    a = m_off(x, inject=inj)["logits"]
    b = m_on(x, inject=inj, prefix_ids=None)["logits"]
    assert torch.equal(a, b), "prefix_ids=None n'est pas le forward nu (bit)"
    assert torch.equal(m_off(x)["logits"], m_on(x)["logits"]), \
        "sans injection, la tête de copie doit être invisible AU BIT"

    # ── (b) porte à b_g = −4 : quasi neutre, mais gradient bien vivant ──────
    #     « Neutre » se mesure sur la LOSS, pas sur le pire token du vocabulaire.
    #     À W_c zéro-init α est UNIFORME, donc la porte dépose 0.018/P ≈ 2e-3 sur
    #     chaque token du préfixe : sur un token que le LM tenait à 1e-5, ça
    #     multiplie la probabilité par 200 (+5 nats). C'est le comportement
    #     VOULU — la copie n'est neutre que là où le LM était déjà crédible — et
    #     c'est pourquoi le critère porte sur la CE réelle et sur la masse
    #     déplacée (KL), les deux quantités que le run ressent.
    on = m_on(x, inject=inj, prefix_ids=pid)["logits"]
    ref = torch.log_softmax(a.float(), -1)
    ce_off = F.nll_loss(ref.reshape(-1, V), x.reshape(-1))
    ce_on = F.nll_loss(on.reshape(-1, V), x.reshape(-1))
    gap = float((ce_on - ce_off).abs().detach())
    kl = float((ref.exp() * (ref - on)).sum(-1).mean().detach())
    assert gap < 2e-2, f"b_g=-4 devrait être quasi neutre, ΔCE {gap:.4f}"
    assert abs(kl) < 2e-2, f"b_g=-4 déplace trop de masse, KL {kl:.4f}"
    # et le DÉCODAGE ne bouge pas : l'argmax est le même token partout
    assert torch.equal(on.argmax(-1), ref.argmax(-1)), \
        "b_g=-4 fait déjà basculer l'argmax : le greedy du run changerait"
    hc = m_on.rti_copy
    m_on.zero_grad()
    F.nll_loss(on.reshape(-1, V), x.reshape(-1)).backward()
    gw = float(hc.wc.weight.grad.abs().max())
    gg = float(hc.w_g.grad.abs().max())
    gb = float(hc.b_g.grad.abs().max())
    assert gw > 0 and gg > 0 and gb > 0, \
        f"chemin de copie sans gradient : W_c {gw} porte {gg}/{gb}"
    # la porte est en paramètres 1-D : AdamW la prendra, jamais Muon
    assert hc.w_g.ndim == 1 and hc.b_g.ndim == 0, "porte happée par Muon"

    # ── (c) ORACLE : porte à 1 et α sur LA bonne position ⇒ log P(y) ≈ 0 ────
    #     C'est la preuve MÉCANIQUE que la copie exacte est atteignable : si ce
    #     test passe, tout échec ultérieur est un échec d'APPRENTISSAGE, pas de
    #     classe de fonctions.
    j = 2                                        # la position qui porte la cible
    y = ids[:, j][:, None].expand(B, T)          # cible = le token du préfixe j
    al = torch.zeros(B, T, P)
    al[:, :, j] = 1.0
    lp = head(h, h_pre, ids, lg, alpha_override=al, gate_override=1.0)
    got = lp.gather(-1, y[..., None]).squeeze(-1)
    assert float(got.abs().max()) < 1e-4, \
        f"oracle : log P(valeur) devrait être ~0, vaut {float(got.min()):.4f}"
    assert float(lp.exp().sum(-1).sub(1).abs().max()) < 1e-4, "log P non normalisé"

    # ── (e) sentinelles jamais copiées, et duplicats qui S'ADDITIONNENT ─────
    #     α uniforme sur les positions valides : la sentinelle doit recevoir 0,
    #     et un id présent deux fois doit recevoir la SOMME des deux α.
    ids2 = torch.tensor([[5, 5, 11, -1]])
    hp2, h2 = torch.randn(1, 4, d), torch.randn(1, 2, d)
    lg2 = torch.zeros(1, 2, V)
    a2 = torch.tensor([[[0.25, 0.25, 0.5, 0.0]] * 2])
    lp2 = head(h2, hp2, ids2, lg2, alpha_override=a2, gate_override=1.0)
    p2 = lp2.exp()[0, 0]
    assert abs(float(p2[5]) - 0.5) < 1e-5, f"duplicats non sommés : {float(p2[5])}"
    assert abs(float(p2[11]) - 0.5) < 1e-5, float(p2[11])
    # une sentinelle ne reçoit JAMAIS de masse : α est masqué AVANT le softmax
    la = head.attn(h2, hp2, ids2).detach()
    assert float(la[..., 3].exp().max()) == 0.0, "sentinelle copiable"
    assert abs(float(la.exp().sum(-1).max()) - 1.0) < 1e-5, "α non normalisé"
    # et la masse totale du mélange reste 1 même avec des duplicats
    assert float(lp2.exp().sum(-1).sub(1).abs().max()) < 1e-4
    # RÉGRESSION : le token 0 dans le préfixe EN MÊME TEMPS qu'une sentinelle.
    # Une sentinelle rabattue sur l'index 0 écraserait ici la masse de copie du
    # vrai token 0 — trou de normalisation silencieux, mesuré 4.5e-3 au
    # self-test de `model` avant correction (colonne poubelle).
    ids3 = torch.tensor([[0, 11, -1, -1]])
    a3 = torch.tensor([[[0.5, 0.5, 0.0, 0.0]] * 2])
    lp3 = head(h2, hp2, ids3, lg2, alpha_override=a3, gate_override=1.0)
    p3 = lp3.exp()[0, 0]
    assert abs(float(p3[0]) - 0.5) < 1e-5, \
        f"la sentinelle a écrasé le token 0 : {float(p3[0])}"
    assert float(lp3.exp().sum(-1).sub(1).abs().max()) < 1e-4, "masse perdue"

    # ── (d) MICRO-TRAIN : la classe de fonctions manquante, mesurée ─────────
    #     Tâche : « préfixe injecté = un littéral rare ; le contexte demande de
    #     le citer ». Les littéraux sont tirés HORS de la distribution que le LM
    #     voit ailleurs — exactement le régime des noms/codes du 350M, où la nll
    #     des tokens VALEUR est restée plate de 200 à 800 steps.
    def _micro(with_copy: bool, steps: int = 400, seed: int = 7):
        g = torch.Generator().manual_seed(seed)
        torch.manual_seed(seed)
        Vv, dd, kk = 48, 32, 3          # kk tokens de valeur à citer
        hd = CopyHead(dd, CopyHeadConfig(enabled=True)) if with_copy else None
        m = _ToyLM(Vv, dd, copy=hd)
        ps = [p for p in m.parameters()]
        opt = torch.optim.Adam(ps, lr=3e-3)
        CUE, SEP = 0, 1                  # template : tokens FRÉQUENTS
        VAL_LO = 8                       # littéraux : jamais ailleurs
        hist = []
        for st in range(steps):
            n = 8
            val = torch.randint(VAL_LO, Vv, (n, kk), generator=g)
            # le préfixe DIT le littéral (+ une sentinelle de séparateur)
            pids = torch.cat([val, torch.full((n, 1), -1)], 1)          # [n,k+1]
            seq = torch.cat([torch.full((n, 1), CUE),
                             torch.full((n, 1), SEP), val], 1)          # [n,k+2]
            injv = m.emb(pids.clamp_min(0)).detach()
            o = m(seq, inject=injv, prefix_ids=pids if with_copy else None)
            lp_ = (o["logits"].float() if o.get("logits_are_logprobs")
                   else torch.log_softmax(o["logits"].float(), -1))
            # CE des seuls tokens VALEUR (positions 2..k+1, prédites depuis 1..k)
            tgt = seq[:, 2:]
            pred = lp_[:, 1:1 + kk]
            ce = -pred.gather(-1, tgt[..., None]).squeeze(-1).mean()
            opt.zero_grad(); ce.backward(); opt.step()
            hist.append(float(ce))
        p_end = (float(hd.last_p_copy) if hd is not None else 0.0)
        return sum(hist[:20]) / 20, sum(hist[-20:]) / 20, p_end

    c0, c1, pc = _micro(True)
    n0, n1, _ = _micro(False)
    print(f"  micro-train (littéraux rares, {48 - 8} valeurs possibles, "
          f"chance = {torch.tensor(40.0).log():.3f} nats)")
    print(f"    copy-head ON  : CE valeur {c0:.3f} -> {c1:.3f}   "
          f"p_copy final {pc:.3f}")
    print(f"    copy-head OFF : CE valeur {n0:.3f} -> {n1:.3f}")
    # Mesuré 2026-07-31 (CPU, 400 steps) : ON 3.532 → 0.224 (p_copy 0.995),
    # OFF 3.929 → 2.212, chance 3.689. Le bras OFF descend un peu — le jouet
    # peut attendre les embeddings injectés, qui SONT ceux des valeurs — mais il
    # plafonne à 6 fois la CE du bras ON. C'est ça, la classe de fonctions
    # manquante : elle ne se compense pas par plus d'optimisation.
    assert c1 < 0.50, f"copy-head ON n'apprend pas la citation (CE {c1:.3f})"
    assert n1 > 1.5, (f"le contrôle OFF apprend la citation (CE {n1:.3f}) : "
                      "la tâche n'isole pas la classe de fonctions")
    assert n1 - c1 > 1.5, f"séparation trop faible : OFF {n1:.3f} / ON {c1:.3f}"
    assert pc > 0.5, f"la porte ne s'est pas ouverte (p_copy {pc:.3f})"

    print("rti_copy self-test: OK — (a) prefix_ids=None bit-identique au "
          "forward nu, (b) b_g=-4 quasi neutre (ΔCE "
          f"{gap:.1e}, KL {kl:.1e}) à gradient non nul sur W_c ET la porte, "
          "(c) oracle porte=1/α=δ ⇒ log P(valeur) ~ 0 nat, (d) micro-train : "
          f"CE valeur {n1:.2f} (OFF) -> {c1:.2f} (ON), (e) sentinelles jamais "
          "copiées et duplicats sommés")


if __name__ == "__main__":
    _selftest()
