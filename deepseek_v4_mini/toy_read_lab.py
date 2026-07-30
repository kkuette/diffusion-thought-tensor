"""Labo jouet : 4 classes de READ d'une banque mémoire, WRITE ORACLE.

Question posée (et UNE seule) : « le mur copie-argmax du 350M est-il la CLASSE
DE FONCTIONS du read ? »

Verdict expérimental du 350M (dsv6, sft_recall) : sur le rappel persona à
vocabulaire ouvert (~7 878 valeurs), le canal est OUVERT en Δnll (jusqu'à
+0.33) mais la copie argmax est FERMÉE (grade 0.00, même sur le split TRAIN).
Deux suspects : (a) le WRITE ne sait pas fabriquer un code citable, (b) le READ
fast-weight ne sait pas CITER une ligne — la banque y entre comme FONCTION
(chaque ligne engendre une transformation low-rank appliquée aux hidden
states), jamais comme CONTENU.

Ce labo ÉLIMINE le suspect (a) : la banque est posée par l'ENVIRONNEMENT,
parfaite, identique pour les 4 variantes, jamais apprise. Ce qui reste est
purement une question de classe de fonctions du read.

Les 4 variantes
---------------
  r0  fast-weight (CONTRÔLE)  — port fidèle de model.py:282-394 (fw_A/fw_B,
      rank 8, SwiGLU na=2, boucle séquentielle sur les M slots, résiduel fw_o).
      Attendu : reproduit la signature Δnll-ouvert / argmax-fermé.
  r1  cross-attention CONTENU — la banque entre comme K/V d'une attention
      croisée par couche, résiduel gaté (gate init 0).
  r2  r0 + TÊTE DE COPIE (pointer readout) — le read fast-weight inchangé, plus
      un biais de logits `(s_t @ P) @ embed^T` où s_t est un mélange de lignes
      sélectionné par attention depuis h_t, ouvert par une porte par token.
  r3  banque en ESPACE D'EMBEDDINGS — mem_dim = d_model, lignes = pools
      d'embeddings bruts RMS-normés ; cross-attention SANS projection de V, et
      pointer readout SANS P (la ligne est DÉJÀ dans l'espace d'embedding :
      biais = s_t @ embed^T). Variante « la géométrie est déjà bonne ».

Backbone : transformer decoder-only VANILLA (RMSNorm pré-norm, MHA causale via
scaled_dot_product_attention, RoPE, MLP SwiGLU, embeddings tiés). AUCUN élément
dsv4 (pas de mHC, pas de Sinkhorn, pas de MoE, pas de CSA) : si le mur tombe
ici, ce n'est pas l'archi exotique du 350M qui le tenait.

Write oracle
------------
Après chaque seg porteur de fait, l'environnement appende à un FIFO de M=8
lignes :  ligne = RMSnorm(K[slot] + A[attr] + V[val])  avec K/A embeddings
aléatoires FIGÉS (seed fixe, ×0.2) et V[val] = RMSnorm(moyenne uniforme des
embeddings COURANTS du modèle sur les tokens de " "+val) projetée par une
matrice aléatoire figée d_model→mem_dim (identité si mem_dim == d_model).
V dépend d'embeddings qui bougent : le code est recalculé à la volée (detach) à
chaque write. Aucun gradient ne traverse jamais la banque.

Éval (le verdict)
-----------------
Sur des valeurs HELD-OUT (pool_split=eval, sha1 20 %) ET sur le split train
(contrôle mécanisme décisif : un grade nul sur du DÉJÀ VU dit que le read ne
sait pas citer, pas qu'il ne généralise pas) :
  grade_recall LIVE, grade ABLATÉ (banque vide), Δnll = nll_ablaté − nll_live.

Phase 2 : l'axe FORMAT DE CODE (`--code`)
-----------------------------------------
Verdict de la phase 1 : grade held-out 0.000 pour les 4 variantes, mais Δnll
r0 +0.87 / r1 +1.22 / r2 +0.77 / **r3 +3.80**. Diagnostic : le POOL MOYEN
détruit l'ordre des tokens — la ligne est un SAC DE TOKENS, l'information de
séquence n'existe plus DANS LA BANQUE, donc aucun read ne peut la ressusciter.
La phase 2 attaque l'axe orthogonal (le format du code), sur le read gagnant
r3 uniquement :

  mean   (DÉFAUT, contrôle bit-à-bit de la phase 1) V = moyenne uniforme.
  chunk  la ligne est découpée en n_pos blocs de d_model/n_pos dims ; le token
         k occupe le bloc k, projeté par P_k (frame orthonormé FIGÉ). Ordre =
         position physique dans la ligne. Décodable à 1/√blk près (JL).
  phase  binding positionnel SUPERPOSÉ style HRR/RoPE :
         ligne = RMSnorm(K + A + Σ_k rot(θ·k)·ê(t_k)). L'ordre survit dans la
         phase ; le readout déroule en appliquant rot(−θ·j). Capacité ~ n_pos/d.
  rows   BORNE HAUTE de décodabilité : UNE ligne par token,
         ligne_k = RMSnorm(K + A + 0.2·pos[k] + ê(t_k)). Le FIFO 8 est
         inchangé ⇒ un fait long mange la banque (économie dégradée : c'est
         le prix de la borne).

Le readout pointer devient CANDIDAT-BASED quand code ≠ mean : chaque ligne
engendre n_pos candidats (bloc dé-projeté / ligne dé-tournée / la ligne
elle-même), l'attention plate du pointer choisit parmi les M×n_pos candidats.
Le modèle APPREND l'alignement position↔décodage — aucun compteur dur. La
cross-attention de CONTENU de r3 reste inchangée (elle voit les lignes brutes).

Usage
-----
  python -m deepseek_v4_mini.toy_read_lab CONFIG.yaml --variant r0
  python -m deepseek_v4_mini.toy_read_lab CONFIG.yaml --variant r1 --smoke --device cpu
  python -m deepseek_v4_mini.toy_read_lab CONFIG.yaml --variant r3 --code phase
  python -m deepseek_v4_mini.toy_read_lab --selftest
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import random
import time
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from .math_school_data import A_OPEN
from .paths import load_yaml
from .persona_chat_data import fact_id_maps, grade_recall
from .streams import chat_stream_class

VARIANTS = ("r0", "r1", "r2", "r3")
CODES = ("mean", "chunk", "phase", "rows")
# slots dont la valeur est un CODE arbitraire (5-8 tokens, zéro prior LM) —
# c'est la strate que le pool moyen ne peut structurellement pas rendre.
CODE_SLOTS = ("code", "ref", "plate")
GROUPS = ("short", "word", "code")


# ── config ───────────────────────────────────────────────────────────────────

@dataclass
class ToyCfg:
    vocab_size: int = 49152
    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 8
    d_ff_mult: int = 4
    mem_dim: int = 512
    max_mem: int = 8
    max_seq_len: int = 640
    dropout: float = 0.0
    rope_theta: float = 10000.0
    # read
    variant: str = "r0"
    read_layers: list = field(default_factory=list)   # vide = toutes
    fw_rank: int = 8
    x_dim: int = 0            # dim interne de la cross-attn (0 = d_model)
    ptr_bias_init: float = -8.0
    oracle_ka_scale: float = 0.2
    oracle_seed: int = 20260730
    # ── axe FORMAT DE CODE (phase 2) ────────────────────────────────────────
    code: str = "mean"
    n_pos: int = 8            # tokens de valeur retenus (couverture 100 % de
                              # la distribution ÉCHANTILLONNÉE, cf. rapport)
    rope_base: float = 0.0    # <=0 : binding DFT (défaut, cf. phase_tables) ;
                              # >0 : forme RoPE littérale (ablation)

    def __post_init__(self):
        if self.variant == "r3":
            # R3 : la banque VIT dans l'espace d'embedding — pas de projection.
            self.mem_dim = self.d_model
        if not self.x_dim:
            self.x_dim = self.d_model
        assert self.d_model % self.n_heads == 0
        assert self.mem_dim % self.n_heads == 0
        assert self.x_dim % self.n_heads == 0
        assert self.variant in VARIANTS
        assert self.code in CODES, f"code inconnu {self.code!r} (∈ {CODES})"
        if self.code != "mean":
            # les nouveaux formats supposent banque == espace d'embedding et
            # pointer nu : c'est la définition de r3, on ne les porte pas
            # ailleurs (r0/r1/r2 restent le contrôle de la phase 1).
            assert self.variant == "r3", (
                f"--code {self.code} n'est supporté QUE par --variant r3 "
                f"(banque en espace d'embedding + pointer nu) ; reçu "
                f"--variant {self.variant}. Phase 1 = --code mean.")
            assert self.mem_dim == self.d_model
            assert self.n_pos >= 1
            if self.code == "chunk":
                assert self.d_model % self.n_pos == 0, (
                    f"chunk : d_model {self.d_model} doit être divisible par "
                    f"n_pos {self.n_pos}")
        assert self.d_model % 2 == 0

    @property
    def n_cand(self) -> int:
        """Candidats engendrés PAR LIGNE par le readout position-conscient."""
        return 1 if self.code in ("mean", "rows") else self.n_pos

    @property
    def uses_fw(self) -> bool:
        return self.variant in ("r0", "r2")

    @property
    def uses_xattn(self) -> bool:
        return self.variant in ("r1", "r3")

    @property
    def uses_ptr(self) -> bool:
        return self.variant in ("r2", "r3")


# ── briques vanilla ──────────────────────────────────────────────────────────

class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        n = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * n).type_as(x) * self.w


def safe_bank_mask(bank_mask):
    """(masque utilisable pour un softmax, lanes SANS aucune ligne).

    Dans un groupe, une lane peut n'avoir encore RIEN écrit alors qu'une autre
    a déjà 3 lignes : son masque est tout-faux et un softmax sur des -inf
    partout rend NaN (payé au premier smoke). On ouvre artificiellement la
    ligne 0 (qui est du zéro-padding) pour que le softmax soit défini, et on
    annule la sortie de ces lanes après coup.
    """
    if bank_mask is None:
        return None, None
    empty = ~bank_mask.any(-1)                     # [B]
    if not bool(empty.any()):
        return bank_mask, None
    m = bank_mask.clone()
    m[empty, 0] = True
    return m, empty


def rms_unit(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """RMS-normalisation SANS paramètre (codes de banque)."""
    return x * x.float().pow(2).mean(-1, keepdim=True).add(eps).rsqrt().type_as(x)


def _rope_tables(T: int, d_head: int, theta: float, device, dtype):
    inv = 1.0 / (theta ** (torch.arange(0, d_head, 2, device=device).float() / d_head))
    t = torch.arange(T, device=device).float()
    f = torch.outer(t, inv)                              # [T, dh/2]
    return f.cos().to(dtype), f.sin().to(dtype)


def rot_pairs(x: torch.Tensor, cos, sin) -> torch.Tensor:
    """Rotation 2D par paires de dims (x0,x1), (x2,x3)… — forme générique.

    x : [..., d] ; cos/sin : [..., d/2] BROADCASTABLES sur x[..., 0::2].
    rot(−θ) s'obtient en passant −sin (le format `phase` déroule comme ça).
    """
    x1, x2 = x[..., 0::2], x[..., 1::2]
    o1 = x1 * cos - x2 * sin
    o2 = x1 * sin + x2 * cos
    return torch.stack((o1, o2), dim=-1).flatten(-2)


def apply_rope(x: torch.Tensor, cos, sin) -> torch.Tensor:
    """x : [B, H, T, dh] — rotation par paires (x0,x1)."""
    return rot_pairs(x, cos[None, None], sin[None, None])


def phase_tables(n_pos: int, d: int, base: float, device=None, dtype=None):
    """(cos, sin) [n_pos, d/2] du binding positionnel `phase`.

    La position k applique rot(θ_i·k) à la paire de dims i ; le readout
    applique rot(−θ_i·j) pour lire la position j. Le terme k=j revient à
    l'identité, les termes k≠j restent tournés d'un offset Δ=k−j : leur
    contribution parasite au score du bon token vaut ⟨cos(θ_i·Δ)⟩_i.

    base <= 0 → binding DFT (DÉFAUT) : θ_i = 2π·(i mod n_pos)/n_pos, pour
      lequel ⟨cos(θ_i·Δ)⟩_i = 0 EXACTEMENT pour tout Δ ≢ 0 [n_pos]
      (orthogonalité des caractères) — c'est le HRR discret.
    base > 0 → forme RoPE littérale θ_i = base^(−2i/d), gardée comme ABLATION.
      ⚠️ MESURÉ : elle ne décorrèle PAS n_pos=8 positions (θ_max = 1 rad ⇒
      ⟨cos(θ_i·1)⟩ = 0.95 à base 100, 0.54 à base 1) ; round-trip oracle
      plafonné à 3-7/8 selon la base contre 8/8 en DFT.
    """
    if base is None or base <= 0:
        r = (torch.arange(0, d // 2, device=device) % n_pos).float()
        inv = 2.0 * math.pi * r / n_pos                        # [d/2]
    else:
        inv = 1.0 / (base ** (torch.arange(0, d, 2, device=device).float() / d))
    k = torch.arange(n_pos, device=device).float()
    f = torch.outer(k, inv)                                   # [n_pos, d/2]
    c, s = f.cos(), f.sin()
    if dtype is not None:
        c, s = c.to(dtype), s.to(dtype)
    return c, s


class CausalSelfAttn(nn.Module):
    def __init__(self, cfg: ToyCfg):
        super().__init__()
        d, h = cfg.d_model, cfg.n_heads
        self.h, self.dh = h, d // h
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.o = nn.Linear(d, d, bias=False)
        self.theta = cfg.rope_theta

    def forward(self, x):
        B, T, d = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, T, self.h, self.dh).transpose(1, 2)
        k = k.view(B, T, self.h, self.dh).transpose(1, 2)
        v = v.view(B, T, self.h, self.dh).transpose(1, 2)
        cos, sin = _rope_tables(T, self.dh, self.theta, x.device, q.dtype)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        # right-pad uniquement : une query non-pad n'attend que des clés non-pad
        # (causalité) ⇒ le masque causal suffit, pas de key-padding mask.
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.o(y.transpose(1, 2).reshape(B, T, d))


class SwiGLU(nn.Module):
    def __init__(self, d: int, mult: int = 4):
        super().__init__()
        hidden = int(mult * d * 2 / 3 / 8) * 8 or 8
        self.wg = nn.Linear(d, hidden, bias=False)
        self.wu = nn.Linear(d, hidden, bias=False)
        self.wd = nn.Linear(hidden, d, bias=False)

    def forward(self, x):
        return self.wd(F.silu(self.wg(x)) * self.wu(x))


# ── R0/R2 : read fast-weight (port fidèle de model.py:282-394) ───────────────

class FastWeightRead(nn.Module):
    """Chaque ligne de banque ENGENDRE une couche low-rank appliquée à h.

    Port fidèle du 350M : fw_A émet DEUX cartes par slot (gate + valeur, SwiGLU
    na=2), fw_B la remontée, application SÉQUENTIELLE sur les M slots (les
    transformations se composent en profondeur), résiduel via fw_o sur l'écart
    au point de départ normalisé.
    """

    def __init__(self, cfg: ToyCfg):
        super().__init__()
        d, r = cfg.d_model, cfg.fw_rank
        self.r, self.d = r, d
        self.fw_A = nn.Linear(cfg.mem_dim, 2 * r * d, bias=False)
        self.fw_B = nn.Linear(cfg.mem_dim, d * r, bias=False)
        self.fw_o = nn.Linear(d, d, bias=False)
        self.norm_fw = RMSNorm(d)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, h, bank, bank_mask):
        B, M, _ = bank.shape
        d, r = self.d, self.r
        A = self.fw_A(bank).view(B, M, 2, r, d)
        Bm = self.fw_B(bank).view(B, M, d, r)
        if bank_mask is not None:                 # lignes de padding = inertes
            m = bank_mask.to(A.dtype)
            A = A * m[:, :, None, None, None]
            Bm = Bm * m[:, :, None, None]
        ds, rs = d ** -0.5, r ** -0.5
        y0 = self.norm_fw(h)
        y = y0
        for i in range(M):
            zg = torch.einsum("brd,btd->btr", A[:, i, 0], y) * ds
            zv = torch.einsum("brd,btd->btr", A[:, i, 1], y) * ds
            z = (F.silu(zg) * zv).clamp(-8.0, 8.0)
            upd = torch.einsum("bdr,btr->btd", Bm[:, i], z) * rs
            y = y + self.drop(upd)
        return h + self.fw_o(y - y0)


# ── R1/R3 : cross-attention contenu ──────────────────────────────────────────

class CrossAttnRead(nn.Module):
    """La banque entre comme K/V : le read peut LIRE le contenu d'une ligne.

    R3 (`project_v=False`) : V = la ligne BRUTE — la banque vit déjà dans
    l'espace d'embedding, la projeter la ferait sortir de la géométrie où la
    citation est triviale.
    """

    def __init__(self, cfg: ToyCfg, project_v: bool = True):
        super().__init__()
        d, h, x = cfg.d_model, cfg.n_heads, cfg.x_dim
        self.h, self.dq = h, x // h
        self.project_v = project_v
        self.norm = RMSNorm(d)
        self.q = nn.Linear(d, x, bias=False)
        self.k = nn.Linear(cfg.mem_dim, x, bias=False)
        if project_v:
            self.v = nn.Linear(cfg.mem_dim, x, bias=False)
            self.dv = x // h
        else:
            self.v = None
            self.dv = cfg.mem_dim // h
        self.o = nn.Linear(self.dv * h, d, bias=False)
        # porte scalaire init 0 : au step 0 le read est un no-op exact, le
        # backbone démarre comme un transformer nu (pas de choc d'init).
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, h, bank, bank_mask):
        B, M, _ = bank.shape
        T = h.size(1)
        hn = self.norm(h)
        q = self.q(hn).view(B, T, self.h, self.dq).transpose(1, 2)
        k = self.k(bank).view(B, M, self.h, self.dq).transpose(1, 2)
        vsrc = self.v(bank) if self.project_v else bank
        v = vsrc.view(B, M, self.h, self.dv).transpose(1, 2)
        att = torch.einsum("bhtd,bhmd->bhtm", q, k) / math.sqrt(self.dq)
        sm, empty = safe_bank_mask(bank_mask)
        if sm is not None:
            att = att.masked_fill(~sm[:, None, None, :], float("-inf"))
        w = att.softmax(-1)
        y = torch.einsum("bhtm,bhmd->bhtd", w, v)
        y = y.transpose(1, 2).reshape(B, T, self.h * self.dv)
        out = self.o(y)
        if empty is not None:
            out = out * (~empty)[:, None, None].to(out.dtype)
        return h + self.gate * out


# ── R2/R3 : tête de copie (pointer readout) ──────────────────────────────────

class PointerReadout(nn.Module):
    """Biais de logits par CITATION d'une ligne.

    h_t sélectionne une ligne (attention 1 tête sur la banque), la ligne
    sélectionnée est renvoyée dans l'espace des logits :
      R2 : biais = (s_t @ P) @ embed^T  (P apprise, zéro-init ⇒ biais exact 0
           au step 0)
      R3 : biais = s_t @ embed^T        (pas de P : la ligne EST déjà un
           embedding poolé — c'est tout le point de la variante)
    Porte par token σ(w·h_t + b), b très négatif : la porte s'ouvre si et
    seulement si citer paie.

    PHASE 2 : quand cfg.code ≠ mean, on lui passe non plus la banque brute mais
    l'ENSEMBLE DE CANDIDATS [B, M·n_pos, d] construit par le modèle (bloc
    dé-projeté / ligne dé-tournée / ligne). Le module est le MÊME (mem_dim ==
    d_model en r3) : seule la source des « lignes » change, donc `--code mean`
    est bit-à-bit la phase 1.
    """

    def __init__(self, cfg: ToyCfg, project: bool = True):
        super().__init__()
        d = cfg.d_model
        self.dk = d
        self.last_gate = None       # σ(porte) du dernier forward (télémétrie)
        self.norm = RMSNorm(d)
        self.wq = nn.Linear(d, self.dk, bias=False)
        self.wk = nn.Linear(cfg.mem_dim, self.dk, bias=False)
        self.P = nn.Linear(cfg.mem_dim, d, bias=False) if project else None
        if self.P is not None:
            nn.init.zeros_(self.P.weight)
        self.scale = nn.Parameter(torch.ones(1))
        self.gate = nn.Linear(d, 1, bias=True)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, cfg.ptr_bias_init)

    def forward(self, h, bank, bank_mask, embed_w):
        B, M, _ = bank.shape
        hn = self.norm(h)
        q = self.wq(hn)                                    # [B,T,dk]
        k = self.wk(bank)                                  # [B,M,dk]
        att = torch.einsum("btd,bmd->btm", q, k) / math.sqrt(self.dk)
        sm, empty = safe_bank_mask(bank_mask)
        if sm is not None:
            att = att.masked_fill(~sm[:, None, :], float("-inf"))
        s = att.softmax(-1)
        sel = torch.einsum("btm,bmd->btd", s, bank)        # [B,T,mem_dim]
        if empty is not None:
            sel = sel * (~empty)[:, None, None].to(sel.dtype)
        if self.P is not None:
            sel = self.P(sel)
        bias = self.scale * (sel @ embed_w.t())            # [B,T,V]
        g = torch.sigmoid(self.gate(hn))                   # [B,T,1]
        self.last_gate = g.detach()
        return g * bias


# ── le modèle jouet ──────────────────────────────────────────────────────────

class ToyBlock(nn.Module):
    def __init__(self, cfg: ToyCfg, layer_idx: int):
        super().__init__()
        self.n1 = RMSNorm(cfg.d_model)
        self.attn = CausalSelfAttn(cfg)
        self.n2 = RMSNorm(cfg.d_model)
        self.mlp = SwiGLU(cfg.d_model, cfg.d_ff_mult)
        rl = list(cfg.read_layers or [])
        self.read_bank = (not rl) or (layer_idx in rl)
        self.read = None
        if self.read_bank:
            if cfg.uses_fw:
                self.read = FastWeightRead(cfg)
            elif cfg.uses_xattn:
                self.read = CrossAttnRead(cfg, project_v=(cfg.variant != "r3"))

    def forward(self, x, bank, bank_mask):
        x = x + self.attn(self.n1(x))
        if self.read is not None and bank is not None and bank.size(1) > 0:
            x = self.read(x, bank, bank_mask)
        x = x + self.mlp(self.n2(x))
        return x


class ToyReadLM(nn.Module):
    def __init__(self, cfg: ToyCfg, n_slots: int, n_attrs: int):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        nn.init.normal_(self.embed.weight, std=0.02)
        self.blocks = nn.ModuleList(ToyBlock(cfg, i) for i in range(cfg.n_layers))
        self.norm_f = RMSNorm(cfg.d_model)
        self.ptr = (PointerReadout(cfg, project=(cfg.variant != "r3"))
                    if cfg.uses_ptr else None)
        # ── tables ORACLE (buffers, jamais apprises) ────────────────────────
        g = torch.Generator().manual_seed(cfg.oracle_seed)
        sd = cfg.mem_dim ** -0.5
        self.register_buffer("K_slot", torch.randn(n_slots, cfg.mem_dim,
                                                   generator=g) * sd * cfg.oracle_ka_scale)
        self.register_buffer("A_attr", torch.randn(n_attrs, cfg.mem_dim,
                                                   generator=g) * sd * cfg.oracle_ka_scale)
        if cfg.mem_dim != cfg.d_model:
            proj = torch.randn(cfg.d_model, cfg.mem_dim, generator=g) * (cfg.d_model ** -0.5)
        else:
            proj = torch.eye(cfg.d_model)
        self.register_buffer("val_proj", proj)
        # ── tables ORACLE de la PHASE 2 (tirées APRÈS val_proj : l'état du
        # générateur vu par K/A/val_proj est inchangé ⇒ --code mean est
        # bit-à-bit la phase 1) ─────────────────────────────────────────────
        if cfg.code == "chunk":
            blk = cfg.d_model // cfg.n_pos
            raw = torch.randn(cfg.n_pos, cfg.d_model, blk, generator=g)
            # frame ORTHONORMÉ par position : e @ P_k = coordonnées de e dans
            # un sous-espace aléatoire de dim blk, et P_k^T dé-projette
            # EXACTEMENT (projection orthogonale). Le bruit résiduel de la
            # dé-projection est purement JL : SNR ≈ √blk.
            P = torch.stack([torch.linalg.qr(raw[k])[0] for k in range(cfg.n_pos)])
            self.register_buffer("chunk_P", P)                 # [n_pos, d, blk]
        elif cfg.code == "phase":
            c, s = phase_tables(cfg.n_pos, cfg.d_model, cfg.rope_base)
            self.register_buffer("ph_cos", c)                  # [n_pos, d/2]
            self.register_buffer("ph_sin", s)
        elif cfg.code == "rows":
            pos = torch.randn(cfg.n_pos, cfg.d_model, generator=g) * (cfg.d_model ** -0.5)
            self.register_buffer("pos_emb", pos)               # [n_pos, d]

    # ── write ORACLE ────────────────────────────────────────────────────────
    @torch.no_grad()
    def oracle_code(self, slot_id: int, attr_id: int, val_tok: torch.Tensor
                    ) -> torch.Tensor:
        """Ligne de banque d'un fait : RMSnorm(K[slot] + A[attr] + V[val]).

        V[val] = RMSnorm(moyenne UNIFORME des embeddings courants des tokens de
        " "+val) @ val_proj. Recalculé à chaque write : les embeddings bougent
        au train, l'oracle doit rester cohérent avec eux (sinon on rejoue le
        conflit de géométries du run table v1). Detach total : aucun gradient
        ne traverse la banque, c'est l'invariant du design.
        """
        e = self.embed.weight[val_tok.to(self.embed.weight.device)].float().mean(0)
        v = rms_unit(e) @ self.val_proj.float()
        code = self.K_slot[slot_id].float() + self.A_attr[attr_id].float() + v
        return rms_unit(code).detach()

    @torch.no_grad()
    def oracle_lines(self, slot_id: int, attr_id: int, val_tok: torch.Tensor
                     ) -> torch.Tensor:
        """LES LIGNES d'un fait : [n_lignes, mem_dim]. Dispatch sur cfg.code.

        `mean` en rend UNE et est bit-à-bit identique à la phase 1 ; `chunk` et
        `phase` en rendent une aussi (formats à ordre INTERNE) ; `rows` en rend
        une PAR TOKEN (borne haute de décodabilité, économie dégradée : le FIFO
        de max_mem lignes est inchangé, un fait long mange la banque).

        Comme en phase 1 : recalculé à la volée sur les embeddings COURANTS,
        detach total, aucun gradient ne traverse la banque.
        """
        c = self.cfg
        if c.code == "mean":
            return self.oracle_code(slot_id, attr_id, val_tok).unsqueeze(0)
        dev = self.embed.weight.device
        toks = val_tok.to(dev)[:c.n_pos]                       # troncature n_pos
        e = self.embed.weight[toks].float()                    # [n, d]
        e = rms_unit(e)                                        # ê(t_k), RMS 1
        ka = self.K_slot[slot_id].float() + self.A_attr[attr_id].float()
        n = e.size(0)
        if c.code == "chunk":
            blk = c.d_model // c.n_pos
            line = torch.zeros(c.d_model, dtype=torch.float32, device=dev)
            # token k → bloc k, projeté par le frame figé P_k ; blocs vides = 0
            coeff = torch.einsum("nd,ndb->nb", e, self.chunk_P[:n].float())
            line[:n * blk] = coeff.reshape(-1)
            out = rms_unit(ka + line).unsqueeze(0)
        elif c.code == "phase":
            cs = self.ph_cos[:n].float()
            sn = self.ph_sin[:n].float()
            bound = rot_pairs(e, cs, sn).sum(0)                # Σ_k rot(θk)·ê
            out = rms_unit(ka + bound).unsqueeze(0)
        else:                                                   # rows
            pos = self.pos_emb[:n].float() * c.oracle_ka_scale
            out = rms_unit(ka[None] + pos + e)                 # [n, d]
        return out.detach()

    # ── candidats du readout position-conscient ─────────────────────────────
    def candidates(self, bank, bank_mask):
        """Banque [B,M,d] → candidats [B, M·n_cand, d] (+ masque étendu).

        La position n'est PAS lue par un compteur dur : on expose tous les
        candidats (ligne i, position j) au pointer, c'est son attention plate
        qui apprend l'alignement position↔décodage.
        """
        c = self.cfg
        if c.code in ("mean", "rows"):
            return bank, bank_mask                 # la ligne EST le candidat
        B, M, d = bank.shape
        n = c.n_pos
        if c.code == "chunk":
            blk = d // n
            blocks = bank.view(B, M, n, blk)
            # dé-projection EXACTE du bloc j par P_j^T (frame orthonormé)
            Pt = self.chunk_P.transpose(1, 2).to(bank.dtype)   # [n, blk, d]
            cand = torch.einsum("bmjc,jcd->bmjd", blocks, Pt)
        else:                                                   # phase
            cs = self.ph_cos.to(bank.dtype)[None, None]        # [1,1,n,d/2]
            sn = self.ph_sin.to(bank.dtype)[None, None]
            cand = rot_pairs(bank[:, :, None, :], cs, -sn)     # rot(−θj)
        cand = cand.reshape(B, M * n, d)
        cm = None if bank_mask is None else \
            bank_mask[:, :, None].expand(B, M, n).reshape(B, M * n)
        return cand, cm

    # ── forward ─────────────────────────────────────────────────────────────
    def forward(self, ids, bank=None, bank_mask=None):
        x = self.embed(ids)
        for blk in self.blocks:
            x = blk(x, bank, bank_mask)
        x = self.norm_f(x)
        logits = x @ self.embed.weight.t()             # embeddings tiés
        if self.ptr is not None and bank is not None and bank.size(1) > 0:
            # code == mean  → candidats = les lignes (chemin phase 1, inchangé)
            cand, cmask = self.candidates(bank, bank_mask)
            logits = logits + self.ptr(x, cand, cmask, self.embed.weight)
        return logits

    # ── décodage greedy (sans cache : préfixes courts) ──────────────────────
    @torch.no_grad()
    def greedy(self, prefix, bank, bank_mask, max_new: int, stop_id: int):
        ids = prefix
        out = []
        for _ in range(max_new):
            lg = self.forward(ids[:, -self.cfg.max_seq_len:], bank, bank_mask)
            nxt = int(lg[0, -1].argmax())
            if nxt == stop_id:
                break
            out.append(nxt)
            ids = torch.cat([ids, torch.tensor([[nxt]], device=ids.device)], 1)
        return out


# ── comptage de paramètres par bloc ──────────────────────────────────────────

def param_report(model: ToyReadLM) -> dict:
    buckets = {"embed": 0, "attn": 0, "mlp": 0, "norm": 0, "read": 0,
               "pointer": 0}
    for n, p in model.named_parameters():
        if n.startswith("embed"):
            b = "embed"
        elif n.startswith("ptr."):
            b = "pointer"
        elif ".read." in n:
            b = "read"
        elif ".attn." in n:
            b = "attn"
        elif ".mlp." in n:
            b = "mlp"
        else:
            b = "norm"
        buckets[b] += p.numel()
    buckets["total"] = sum(p.numel() for p in model.parameters())
    return buckets


# ── environnement : replay de conversations + write oracle ───────────────────

class OracleEnv:
    """Rejoue une conv seg par seg et pose la banque à la place du modèle."""

    def __init__(self, tok, max_mem: int):
        self.tok = tok
        self.max_mem = max_mem
        slot_ids, val_ids, attr_ids = fact_id_maps()
        self.id2val = {i: v for v, i in val_ids.items()}
        self.n_slots = len(slot_ids) + 1
        self.n_attrs = len(attr_ids) + 1
        self.n_vals = len(val_ids) + 1
        self._tokcache: dict[int, torch.Tensor] = {}

    def val_tokens(self, val_id: int) -> torch.Tensor:
        t = self._tokcache.get(val_id)
        if t is None:
            s = " " + self.id2val[val_id]
            ids = self.tok(s, add_special_tokens=False)["input_ids"]
            t = torch.tensor(ids, dtype=torch.long)
            self._tokcache[val_id] = t
        return t

    @staticmethod
    def fact_of(seg: dict):
        """(slot_id, attr_id, val_id) si le seg PORTE un fait, sinon None."""
        if "fact_slot" not in seg:
            return None
        sl = int(seg["fact_slot"][0, 0])
        if sl == 0:
            return None
        return sl, int(seg["fact_attr"][0, 0]), int(seg["fact_val"][0, 0])

    def write(self, model: ToyReadLM, bank: list, seg: dict) -> list:
        """FIFO de max_mem lignes ; les segs sans fait n'écrivent rien.

        `--code rows` appende PLUSIEURS lignes d'un coup (une par token de la
        valeur) : le FIFO reste à max_mem, donc un fait long évince les
        précédents. C'est le prix assumé de la borne haute.
        """
        f = self.fact_of(seg)
        if f is None:
            return bank
        rows = model.oracle_lines(f[0], f[1], self.val_tokens(f[2]))
        bank = bank + list(rows)
        return bank[-self.max_mem:]

    def value_group(self, slot: str | None, truth: str) -> str:
        """Strate de la valeur gradée : `code` (slots code/ref/plate, valeur
        arbitraire sans prior LM), `short` (≤2 tokens), `word` (3+ tokens
        linguistiques). Prédiction phase 2 : mean n'ouvre que `short`,
        phase/chunk/rows doivent ouvrir `code`."""
        if slot in CODE_SLOTS:
            return "code"
        n = len(self.tok(" " + truth, add_special_tokens=False)["input_ids"])
        return "short" if n <= 2 else "word"


def pad_bank(banks: list, device, dtype=torch.float32):
    """Liste de listes de lignes → ([B,M,mem_dim], [B,M] bool) ou (None, None)."""
    M = max((len(b) for b in banks), default=0)
    if M == 0:
        return None, None
    dim = banks[0][0].numel() if banks[0] else next(
        b[0].numel() for b in banks if b)
    out = torch.zeros(len(banks), M, dim, device=device, dtype=dtype)
    msk = torch.zeros(len(banks), M, dtype=torch.bool, device=device)
    for i, b in enumerate(banks):
        if b:
            out[i, :len(b)] = torch.stack(b).to(device=device, dtype=dtype)
            msk[i, :len(b)] = True
    return out, msk


def pad_segs(segs: list, device, max_len: int):
    """Right-pad des segs d'un groupe → ids [n,T], w [n,T] (poids de CE)."""
    ids = [s["input_ids"][0][:max_len] for s in segs]
    lms = [s["loss_mask"][0][:max_len] for s in segs]
    T = max(x.numel() for x in ids)
    X = torch.zeros(len(ids), T, dtype=torch.long)
    W = torch.zeros(len(ids), T, dtype=torch.float)
    for i, (a, m) in enumerate(zip(ids, lms)):
        X[i, :a.numel()] = a
        W[i, :m.numel()] = m
    return X.to(device), W.to(device)


def seg_ce(logits, ids, w):
    """CE décalée standard, pondérée par le loss_mask. Retourne (somme, poids)."""
    lg = logits[:, :-1].float()
    tgt = ids[:, 1:]
    ww = w[:, 1:]
    ce = F.cross_entropy(lg.reshape(-1, lg.size(-1)), tgt.reshape(-1),
                         reduction="none").view_as(tgt)
    return (ce * ww).sum(), ww.sum()


# ── entraînement ─────────────────────────────────────────────────────────────

def train_step(model, env, convs, device, max_len, amp, scale_by):
    """Un pas = un groupe de convs. Backward PAR GROUPE DE SEGS, banque
    détachée entre segs — strictement le même gradient que le backward
    monolithique (le write est oracle, aucun gradient ne traverse la banque)
    pour une fraction de la VRAM."""
    banks = [[] for _ in convs]
    total_w = sum(float(s["loss_mask"][0][1:max_len].sum())
                  for c in convs for s in c["segs"]) or 1.0
    nseg = max(len(c["segs"]) for c in convs)
    loss_sum, tok_sum = 0.0, 0.0
    for j in range(nseg):
        lanes = [i for i, c in enumerate(convs) if j < len(c["segs"])]
        if not lanes:
            continue
        segs = [convs[i]["segs"][j] for i in lanes]
        X, W = pad_segs(segs, device, max_len)
        bank, bmask = pad_bank([banks[i] for i in lanes], device)
        with torch.autocast(device.split(":")[0], dtype=torch.bfloat16,
                            enabled=amp):
            logits = model(X, bank, bmask)
        s, n = seg_ce(logits, X, W)
        if float(n) > 0:
            (s / total_w * scale_by).backward()
            loss_sum += float(s.detach())
            tok_sum += float(n)
        for i, seg in zip(lanes, segs):
            banks[i] = env.write(model, banks[i], seg)
    return loss_sum / max(tok_sum, 1.0)


# ── évaluation ───────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, env, stream, seed, n_convs, device, tok, a_open, stop_id,
             max_new, max_len, amp, n_show=3):
    """Replay teacher-forcé (la banque oracle avance) + décodage greedy des
    tours gradés, bras LIVE (banque) vs ABLATÉ (banque vide)."""
    model.eval()
    stream.rng = random.Random(seed)
    live_ans, abl_ans, truths_all, groups = [], [], [], []
    dnll_num, dnll_den = 0.0, 0.0
    gate_num, gate_den = 0.0, 0.0
    shown = []
    abl_txt = None                       # sans banque, le décodage est unique
    done = 0
    guard = 0
    while done < n_convs and guard < n_convs * 20:
        guard += 1
        conv = stream.next_conv()
        truths = (conv.get("info") or {}).get("truths") or []
        if not truths:
            continue                     # smalltalk : rien à grader
        done += 1
        q_slots = (conv.get("info") or {}).get("q_slots") or []
        a_idx = [i for i, s in enumerate(conv["segs"]) if s["role"] == "assistant"]
        graded = set(a_idx[-len(truths):])
        bank: list = []
        qi = 0
        prev = ""
        for i, seg in enumerate(conv["segs"]):
            X = seg["input_ids"][:, :max_len].to(device)
            W = seg["loss_mask"][:, :max_len].to(device)
            b, bm = pad_bank([bank], device)
            if i in graded:
                with torch.autocast(device.split(":")[0], dtype=torch.bfloat16,
                                    enabled=amp):
                    lg_live = model(X, b, bm)
                    lg_abl = model(X, None, None)
                sl, nl = seg_ce(lg_live, X, W)
                sa, _ = seg_ce(lg_abl, X, W)
                if float(nl) > 0:
                    dnll_num += float(sa - sl)
                    dnll_den += float(nl)
                # ouverture de la porte du pointer sur les tokens SUPERVISÉS
                # (diagnostic du suspect « le modèle s'en sort au prior LM »)
                g_ptr = getattr(model.ptr, "last_gate", None) if model.ptr else None
                if g_ptr is not None and g_ptr.shape[:2] == W.shape:
                    gate_num += float((g_ptr[..., 0].float() * W).sum())
                    gate_den += float(W.sum())
                if abl_txt is None:
                    abl_txt = tok.decode(model.greedy(a_open, None, None,
                                                      max_new, stop_id))
                live = tok.decode(model.greedy(a_open, b, bm, max_new, stop_id))
                tr = truths[qi] if qi < len(truths) else "?"
                live_ans.append(live)
                abl_ans.append(abl_txt)
                truths_all.append(tr)
                groups.append(env.value_group(
                    q_slots[qi] if qi < len(q_slots) else None, tr))
                if len(shown) < n_show:
                    shown.append((prev.strip(), tr, live.strip(),
                                  abl_txt.strip()))
                qi += 1
            bank = env.write(model, bank, seg)
            prev = tok.decode(seg["input_ids"][0].tolist())
    model.train()
    out = {
        "grade_live": grade_recall(live_ans, truths_all) if truths_all else 0.0,
        "grade_abl": grade_recall(abl_ans, truths_all) if truths_all else 0.0,
        "dnll": dnll_num / max(dnll_den, 1.0),
        "n": len(truths_all),
        "show": shown,
        "ptr_gate": (gate_num / gate_den) if gate_den > 0 else float("nan"),
    }
    # grade PAR STRATE de valeur (short / word / code)
    for gname in GROUPS:
        idx = [i for i, x in enumerate(groups) if x == gname]
        out[f"n_{gname}"] = len(idx)
        out[f"grade_{gname}"] = (
            grade_recall([live_ans[i] for i in idx],
                         [truths_all[i] for i in idx]) if idx else float("nan"))
        out[f"grade_{gname}_abl"] = (
            grade_recall([abl_ans[i] for i in idx],
                         [truths_all[i] for i in idx]) if idx else float("nan"))
    return out


# ── plomberie ────────────────────────────────────────────────────────────────

def build_tokenizer(name):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    add = [x for x in ("<think>", "<blank>") if x not in tok.get_vocab()]
    if add:
        tok.add_special_tokens({"additional_special_tokens": add})
    return tok


def persona_kwargs(raw, split, smoke):
    gen = dict((raw.get("persona") or {}).get("gen") or {})
    # Le toy n'utilise PAS surp_w : garder surprisal_mode ne ferait que payer
    # une passe de 300 convs pour construire la table unigram, par instance.
    gen.pop("surprisal_mode", None)
    gen.pop("sif_a", None)
    gen["pool_split"] = split
    if smoke:
        gen.pop("real_filler", None)      # smoke hermétique : filler template
        gen.pop("real_cache_dir", None)
    return gen


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("config", nargs="?")
    ap.add_argument("--variant", choices=VARIANTS, default="r0")
    ap.add_argument("--code", choices=CODES, default="mean",
                    help="format du code de banque (phase 2, r3 seulement) ; "
                         "mean = phase 1 inchangée")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)

    if a.selftest:
        _selftest()
        return

    assert a.config, "config YAML requise (ou --selftest)"
    if a.code != "mean" and a.variant != "r3":
        raise SystemExit(
            f"--code {a.code} n'est supporté QUE par --variant r3 (banque en "
            f"espace d'embedding + pointer nu). Les variantes r0/r1/r2 sont le "
            f"contrôle de la phase 1 : lance-les sans --code (mean).")
    raw = load_yaml(a.config)              # le toy parse son YAML lui-même :
    t = dict(raw.get("training") or {})    # PAS de cfg_schema (trainer principal)
    mc = dict(raw.get("model") or {})
    cb = dict(raw.get("code") or {})       # knobs de l'axe format de code
    if "n_pos" in cb:
        mc["n_pos"] = int(cb["n_pos"])
    if "rope_base" in cb:
        mc["rope_base"] = float(cb["rope_base"])
    device = a.device or t.get("device") or ("cuda" if torch.cuda.is_available()
                                             else "cpu")
    steps = int(a.steps or t.get("steps", 3000))
    b_convs = int(t.get("batch_convs", 8))
    eval_every = int(t.get("eval_every", 200))
    eval_convs = int(t.get("eval_convs", 24))
    max_new = int(t.get("max_new", 48))
    if a.smoke:
        mc.update(d_model=64, n_layers=2, n_heads=4, mem_dim=64, x_dim=0)
        steps, b_convs, eval_every, eval_convs, max_new = 2, 2, 1, 1, 8

    torch.manual_seed(int(t.get("seed", 0)))
    tok = build_tokenizer(raw["tokenizer"])
    env = OracleEnv(tok, int(mc.get("max_mem", 8)))

    mc["variant"] = a.variant
    mc["code"] = a.code
    mc["vocab_size"] = len(tok)
    cfg = ToyCfg(**mc)
    model = ToyReadLM(cfg, env.n_slots, env.n_attrs).to(device)

    # phase 1 → <variant>/ (inchangé) ; phase 2 → <variant>_<code>/
    run_name = a.variant if a.code == "mean" else f"{a.variant}_{a.code}"
    save_dir = os.path.join(t.get("save_dir", "./checkpoints/toy_read_lab"),
                            run_name)
    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, "metrics.csv")
    new_csv = not os.path.exists(csv_path)

    pr = param_report(model)
    print(f"=== toy_read_lab variante {a.variant} | code {a.code} | "
          f"device {device} ===", flush=True)
    print(f"  cfg d_model {cfg.d_model} L {cfg.n_layers} H {cfg.n_heads} "
          f"mem_dim {cfg.mem_dim} M {cfg.max_mem} x_dim {cfg.x_dim} "
          f"vocab {cfg.vocab_size}", flush=True)
    if a.code != "mean":
        print(f"  code {a.code} : n_pos {cfg.n_pos} "
              + (f"blk {cfg.d_model // cfg.n_pos} " if a.code == "chunk" else "")
              + (f"rope_base {cfg.rope_base} " if a.code == "phase" else "")
              + f"| candidats pointer {cfg.max_mem * cfg.n_cand} "
              f"({cfg.max_mem}×{cfg.n_cand})", flush=True)
    print("  params : " + "  ".join(f"{k} {v/1e6:.2f}M" for k, v in pr.items()),
          flush=True)
    print(f"  read+pointer = {(pr['read']+pr['pointer'])/1e6:.2f}M "
          f"({100*(pr['read']+pr['pointer'])/pr['total']:.1f} % du total) — "
          f"appariement de budget entre variantes via model.x_dim (référence "
          f"= le hypernetwork fast-weight de r0 ; r3 reste structurellement "
          f"plus léger, sa V n'est pas projetée).", flush=True)
    print(f"  save_dir {save_dir}", flush=True)

    amp = bool(t.get("amp", True)) and device.startswith("cuda")
    opt = torch.optim.AdamW(model.parameters(), lr=float(t.get("lr", 3e-4)),
                            weight_decay=float(t.get("weight_decay", 0.01)),
                            betas=(0.9, 0.95))
    warmup = int(t.get("warmup_steps", 100))

    def lr_at(s):
        if s < warmup:
            return (s + 1) / max(warmup, 1)
        p = (s - warmup) / max(steps - warmup, 1)
        return 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))

    base_lr = float(t.get("lr", 3e-4))
    max_len = int(cfg.max_seq_len)
    clip = float(t.get("grad_clip", 1.0))

    P = chat_stream_class("persona")
    tr_stream = P(tok, seed=int(t.get("seed", 0)),
                  **persona_kwargs(raw, "train", a.smoke))
    ev_stream = P(tok, seed=1234, **persona_kwargs(raw, "eval", a.smoke))
    tc_stream = P(tok, seed=4321, **persona_kwargs(raw, "train", a.smoke))

    a_open = torch.tensor(tok(A_OPEN, add_special_tokens=False)["input_ids"],
                          dtype=torch.long, device=device).unsqueeze(0)
    stop_id = tok.convert_tokens_to_ids("<|im_end|>")

    best = -1.0
    t0 = time.time()
    for step in range(steps):
        for g in opt.param_groups:
            g["lr"] = base_lr * lr_at(step)
        convs = [tr_stream.next_conv() for _ in range(b_convs)]
        opt.zero_grad(set_to_none=True)
        loss = train_step(model, env, convs, device, max_len, amp, 1.0)
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        opt.step()
        if step % int(t.get("log_every", 20)) == 0:
            print(f"step {step:5d} | loss {loss:.4f} | gnorm {float(gn):.2f} "
                  f"| lr {base_lr*lr_at(step):.2e} | {time.time()-t0:.0f}s",
                  flush=True)
        last = step == steps - 1
        if (step + 1) % eval_every == 0 or last:
            ev = evaluate(model, env, ev_stream, 1234, eval_convs, device, tok,
                          a_open, stop_id, max_new, max_len, amp)
            tc = evaluate(model, env, tc_stream, 4321, eval_convs, device, tok,
                          a_open, stop_id, max_new, max_len, amp, n_show=0)
            print(f"  [eval {step+1}] HELD-OUT grade live {ev['grade_live']:.3f} "
                  f"abl {ev['grade_abl']:.3f} Δnll {ev['dnll']:+.4f} "
                  f"(n={ev['n']}) | TRAIN grade live {tc['grade_live']:.3f} "
                  f"abl {tc['grade_abl']:.3f} Δnll {tc['dnll']:+.4f} "
                  f"(n={tc['n']})", flush=True)
            print("    strates held-out : " + "  ".join(
                f"{g} {ev['grade_' + g]:.3f} (n={ev['n_' + g]})"
                for g in GROUPS) + f"  | porte pointer σ {ev['ptr_gate']:.4f}",
                flush=True)
            for q, tr, lv, ab in ev["show"]:
                print(f"    Q {q[:90]!r}\n      vérité {tr!r}\n"
                      f"      LIVE   {lv[:120]!r}\n      ABLATÉ {ab[:120]!r}",
                      flush=True)
            with open(csv_path, "a", newline="") as f:
                w = csv.writer(f)
                if new_csv:
                    w.writerow(["step", "loss", "grade_eval_live",
                                "grade_eval_abl", "dnll_eval",
                                "grade_train_live", "grade_train_abl",
                                "dnll_train", "n_eval", "n_train"]
                               + [c for g in GROUPS for c in
                                  (f"grade_eval_{g}", f"grade_eval_{g}_abl",
                                   f"n_eval_{g}")]
                               + ["ptr_gate", "sec"])
                    new_csv = False

                def _f(x):
                    return "" if x != x else f"{x:.4f}"   # NaN (strate vide)
                w.writerow([step + 1, f"{loss:.5f}", f"{ev['grade_live']:.4f}",
                            f"{ev['grade_abl']:.4f}", f"{ev['dnll']:.5f}",
                            f"{tc['grade_live']:.4f}", f"{tc['grade_abl']:.4f}",
                            f"{tc['dnll']:.5f}", ev["n"], tc["n"]]
                           + [v for g in GROUPS for v in
                              (_f(ev[f"grade_{g}"]), _f(ev[f"grade_{g}_abl"]),
                               ev[f"n_{g}"])]
                           + [_f(ev["ptr_gate"]), f"{time.time()-t0:.0f}"])
            if ev["grade_live"] > best:
                best = ev["grade_live"]
                torch.save({"step": step + 1, "model": model.state_dict(),
                            "cfg": cfg.__dict__, "grade": best},
                           os.path.join(save_dir, "best.pt"))
    torch.save({"step": steps, "model": model.state_dict(),
                "cfg": cfg.__dict__}, os.path.join(save_dir, "final.pt"))
    print(f"done — best grade held-out {best:.3f} | ckpt {save_dir}", flush=True)


# ── round-trip ORACLE d'un format de code ────────────────────────────────────

@torch.no_grad()
def code_roundtrip(model: ToyReadLM, slot_id: int, attr_id: int,
                   val_tok: torch.Tensor) -> tuple:
    """(top-1 exacts, positions testées) du décodage ORACLE d'un fait.

    Pose les lignes du fait, construit les candidats du readout, et vérifie
    pour chaque position j que `argmax(cand_j @ embed^T) == t_j`. C'est la
    BORNE SUPÉRIEURE de ce que le pointer peut apprendre : si l'oracle
    lui-même ne rend pas l'ordre, aucun read ne le rendra (le verdict de la
    phase 1 en un chiffre).
    """
    cfg = model.cfg
    lines = model.oracle_lines(slot_id, attr_id, val_tok)
    cand, _ = model.candidates(lines.unsqueeze(0), None)      # [1, N, d]
    E = model.embed.weight.float()
    n = len(val_tok) if cfg.code == "mean" else min(len(val_tok), cfg.n_pos)
    ok = 0
    for j in range(n):
        if cfg.code == "mean":
            c = cand[0, 0]                    # une seule ligne, zéro position
        elif cfg.code == "rows":
            c = cand[0, j]                    # une ligne par token
        else:
            c = cand[0, j]                    # une ligne ⇒ candidats 0..n_pos-1
        ok += int(int(torch.argmax(c.float() @ E.t())) == int(val_tok[j]))
    return ok, n


# ── self-test (CPU, dimensions minuscules) ───────────────────────────────────

def _selftest() -> None:
    from .persona_chat_data import PersonaChatStream, _StubTok

    tok = _StubTok()
    tok.decode = lambda ids, **kw: "".join(chr(i) for i in ids)

    env = OracleEnv(tok, 8)
    cfg = ToyCfg(vocab_size=512, d_model=32, n_layers=2, n_heads=4,
                 mem_dim=32, variant="r0", max_seq_len=256)
    model = ToyReadLM(cfg, env.n_slots, env.n_attrs)

    # 1. write oracle DÉTERMINISTE à embeddings figés
    stream = PersonaChatStream(tok, seed=7)
    conv = None
    for _ in range(50):
        c = stream.next_conv()
        if (c["info"]["truths"] and
                any(OracleEnv.fact_of(s) for s in c["segs"])):
            conv = c
            break
    assert conv is not None

    def replay(m):
        b = []
        for s in conv["segs"]:
            b = env.write(m, b, s)
        return b

    b1, b2 = replay(model), replay(model)
    assert len(b1) == len(b2) >= 1
    for x, y in zip(b1, b2):
        assert torch.equal(x, y), "write oracle non déterministe"
    # et il DÉPEND des embeddings (ils bougent au train)
    with torch.no_grad():
        model.embed.weight.add_(0.5)
    b3 = replay(model)
    assert not torch.allclose(b1[0], b3[0]), "code oracle insensible aux embeddings"
    with torch.no_grad():
        model.embed.weight.sub_(0.5)
    # RMS unitaire
    for x in b1:
        assert abs(float(x.pow(2).mean().sqrt()) - 1.0) < 1e-3

    # 2. FIFO correct à > 8 faits
    seg_tpl = [s for s in conv["segs"] if OracleEnv.fact_of(s)][0]
    bank, codes = [], []
    for k in range(12):
        s = dict(seg_tpl)
        vid = 1 + k
        s["fact_val"] = torch.full_like(seg_tpl["fact_val"], vid)
        bank = env.write(model, bank, s)
        codes.append(model.oracle_code(int(s["fact_slot"][0, 0]),
                                       int(s["fact_attr"][0, 0]),
                                       env.val_tokens(vid)))
    assert len(bank) == 8, len(bank)
    for x, y in zip(bank, codes[-8:]):
        assert torch.equal(x, y), "FIFO : mauvaises lignes conservées"
    # un seg SANS fait n'écrit rien
    nofact = [s for s in conv["segs"] if OracleEnv.fact_of(s) is None][0]
    assert len(env.write(model, bank, nofact)) == 8

    # 3. porte R2 fermée à l'init : biais de logits ≈ 0
    for var in ("r2", "r3"):
        c2 = ToyCfg(vocab_size=512, d_model=32, n_layers=2, n_heads=4,
                    mem_dim=32, variant=var, max_seq_len=256)
        m2 = ToyReadLM(c2, env.n_slots, env.n_attrs)
        bk = torch.stack(bank).unsqueeze(0)
        bm = torch.ones(1, bk.size(1), dtype=torch.bool)
        h = torch.randn(1, 5, 32)
        with torch.no_grad():
            bias = m2.ptr(h, bk, bm, m2.embed.weight)
        assert float(bias.abs().max()) < 1e-2, (var, float(bias.abs().max()))
        # et le forward tourne, logits finis
        ids = torch.randint(0, 512, (2, 9))
        b2k = bk.expand(2, -1, -1).contiguous()
        b2m = bm.expand(2, -1).contiguous()
        lg = m2(ids, b2k, b2m)
        assert lg.shape == (2, 9, 512) and torch.isfinite(lg).all()

    # 4. le masque de CE ne couvre QUE des tokens assistant
    ntok_user = ntok_asst = 0
    for s in conv["segs"]:
        w = s["loss_mask"][0]
        if s["role"] == "user":
            ntok_user += int((w > 0).sum())
        else:
            ntok_asst += int((w > 0).sum())
            n_open = len(A_OPEN)
            assert float(w[:n_open].sum()) == 0.0, "A_OPEN supervisé"
    assert ntok_user == 0, "des tokens user sont supervisés"
    assert ntok_asst > 0
    # et la CE pondérée ignore bien ce qui est hors masque
    X, W = pad_segs([s for s in conv["segs"] if s["role"] == "user"], "cpu", 256)
    lg = model(X, None, None)
    s_, n_ = seg_ce(lg, X, W)
    assert float(n_) == 0.0 and float(s_) == 0.0

    # 5. banque padée : les lignes de padding sont INERTES (R0)
    m0 = ToyReadLM(ToyCfg(vocab_size=512, d_model=32, n_layers=2, n_heads=4,
                          mem_dim=32, variant="r0", max_seq_len=256),
                   env.n_slots, env.n_attrs).eval()
    ids = torch.randint(0, 512, (1, 7))
    b_small = torch.stack(bank[:3]).unsqueeze(0)
    m_small = torch.ones(1, 3, dtype=torch.bool)
    b_pad = torch.cat([b_small, torch.randn(1, 2, 32)], dim=1)
    m_pad = torch.tensor([[True, True, True, False, False]])
    with torch.no_grad():
        assert torch.allclose(m0(ids, b_small, m_small), m0(ids, b_pad, m_pad),
                              atol=1e-4), "padding de banque non inerte (R0)"

    # 5bis. lane SANS aucune ligne dans un groupe où d'autres en ont : le
    # softmax de r1/r2/r3 doit rester défini (bug NaN payé au premier smoke).
    for var in VARIANTS:
        cv = ToyCfg(vocab_size=512, d_model=32, n_layers=2, n_heads=4,
                    mem_dim=32, variant=var, max_seq_len=256)
        mv = ToyReadLM(cv, env.n_slots, env.n_attrs)
        bmix = torch.cat([torch.stack(bank[:2]).unsqueeze(0),
                          torch.zeros(1, 2, 32)], dim=0)
        mmix = torch.tensor([[True, True], [False, False]])
        with torch.no_grad():
            lg = mv(torch.randint(0, 512, (2, 6)), bmix, mmix)
        assert torch.isfinite(lg).all(), f"NaN sur lane à banque vide ({var})"
        # la lane vide doit valoir EXACTEMENT le bras ablaté
        with torch.no_grad():
            ids = torch.randint(0, 512, (1, 6))
            a1 = mv(ids, torch.zeros(1, 2, 32),
                    torch.tensor([[False, False]]))
            a2 = mv(ids, None, None)
        assert torch.allclose(a1, a2, atol=1e-5), f"lane vide != ablaté ({var})"

    # 6. comptage de params + forward de chaque variante
    for var in VARIANTS:
        cv = ToyCfg(vocab_size=512, d_model=32, n_layers=2, n_heads=4,
                    mem_dim=32, variant=var, max_seq_len=256)
        mv = ToyReadLM(cv, env.n_slots, env.n_attrs)
        pr = param_report(mv)
        assert pr["total"] > 0
        lg = mv(torch.randint(0, 512, (2, 6)),
                bk.expand(2, -1, -1).contiguous(),
                bm.expand(2, -1).contiguous())
        assert torch.isfinite(lg).all()
        lg2 = mv(torch.randint(0, 512, (2, 6)), None, None)   # bras ablaté
        assert torch.isfinite(lg2).all()

    # ════════════════ PHASE 2 : l'axe FORMAT DE CODE ════════════════════════
    # 7. round-trip ORACLE par format (embeddings figés, vocab jouet 512).
    #    d_model=256 / n_pos=8 = les dims RÉELLES du run (blocs chunk 32 dims).
    def _mk(code, d=256, n_pos=8, vocab=512, base=0.0):
        c = ToyCfg(vocab_size=vocab, d_model=d, n_layers=1, n_heads=4,
                   mem_dim=d, variant="r3", max_seq_len=64, code=code,
                   n_pos=n_pos, rope_base=base)
        return ToyReadLM(c, env.n_slots, env.n_attrs).eval()

    tok8 = torch.tensor([11, 77, 200, 41, 305, 9, 128, 460])   # 8 positions
    rt = {}
    for code in CODES:
        m = _mk(code)
        rt[code] = code_roundtrip(m, 3, 2, tok8)
    for code in ("chunk", "phase", "rows"):
        ok, n = rt[code]
        assert n == 8, (code, n)
        assert ok == n, (
            f"round-trip {code} : seulement {ok}/{n} positions top-1 — le "
            f"format ne porte pas l'ordre, l'expérience n'a pas de sens")
    # `mean` : la démonstration du BUG de la phase 1 — deux valeurs ANAGRAMMES
    # en tokens donnent la MÊME ligne, donc l'ordre est physiquement absent.
    m_mean = _mk("mean")
    perm = tok8[torch.tensor([3, 0, 5, 7, 1, 6, 2, 4])]
    l1 = m_mean.oracle_lines(3, 2, tok8)
    l2 = m_mean.oracle_lines(3, 2, perm)
    assert torch.allclose(l1, l2, atol=1e-6), \
        "mean : les anagrammes devraient collapser (sinon le diagnostic tombe)"
    ok_mean, n_mean = rt["mean"]
    assert ok_mean < n_mean, "mean ne devrait PAS décoder l'ordre"

    # 8. lane vide ≡ ablaté, et forward fini, pour CHAQUE format
    for code in CODES:
        m = _mk(code, d=32, n_pos=4, vocab=512)
        rows = m.oracle_lines(3, 2, tok8[:3])
        bmix = torch.cat([torch.cat([rows[:1], torch.zeros(1, 32)])[None],
                          torch.zeros(1, 2, 32)], dim=0)         # [2,2,32]
        mmix = torch.tensor([[True, False], [False, False]])
        ids = torch.randint(0, 512, (2, 6))
        with torch.no_grad():
            lg = m(ids, bmix, mmix)
        assert torch.isfinite(lg).all(), f"NaN sur lane vide (code {code})"
        with torch.no_grad():
            i1 = torch.randint(0, 512, (1, 6))
            a1 = m(i1, torch.zeros(1, 2, 32), torch.tensor([[False, False]]))
            a2 = m(i1, None, None)
        assert torch.allclose(a1, a2, atol=1e-5), \
            f"lane vide != ablaté (code {code})"
        # 8bis. porte du pointer FERMÉE à l'init : biais ≈ 0
        b1 = rows[:1][None].expand(1, -1, -1)
        cand, cm = m.candidates(b1, torch.ones(1, b1.size(1), dtype=torch.bool))
        with torch.no_grad():
            bias = m.ptr(torch.randn(1, 5, 32), cand, cm, m.embed.weight)
        assert float(bias.abs().max()) < 1e-2, (code, float(bias.abs().max()))
        assert m.ptr.last_gate is not None and \
            float(m.ptr.last_gate.max()) < 1e-3, f"porte ouverte à l'init ({code})"

    # 9. FIFO avec `rows` : un fait de 3 tokens = 3 lignes, cap à max_mem
    m_rows = _mk("rows", d=32, n_pos=4, vocab=512)
    assert m_rows.oracle_lines(3, 2, tok8[:3]).shape[0] == 3
    # via l'env (FIFO réel, max_mem=8) : 3 faits de 3 tokens ⇒ 9 lignes → 8
    env3 = OracleEnv(tok, 8)
    env3.val_tokens = lambda vid: tok8[:3]
    b = []
    for k in range(3):
        s = dict(seg_tpl)
        s["fact_val"] = torch.full_like(seg_tpl["fact_val"], 1 + k)
        b = env3.write(m_rows, b, s)
    assert len(b) == 8, f"FIFO rows : {len(b)} lignes (attendu 8)"
    last = list(m_rows.oracle_lines(int(seg_tpl["fact_slot"][0, 0]),
                                    int(seg_tpl["fact_attr"][0, 0]), tok8[:3]))
    for x, y in zip(b[-3:], last):
        assert torch.equal(x, y), "FIFO rows : mauvaises lignes conservées"

    # 10. r0/r1/r2 REFUSENT les nouveaux formats (message clair)
    for var in ("r0", "r1", "r2"):
        try:
            ToyCfg(vocab_size=512, d_model=32, n_heads=4, mem_dim=32,
                   variant=var, code="phase", n_pos=4)
        except AssertionError as e:
            assert "r3" in str(e)
        else:
            raise AssertionError(f"{var} aurait dû refuser --code phase")

    print("toy_read_lab self-test: OK (write oracle déterministe & "
          "embedding-dépendant, FIFO 8, porte pointer fermée à l'init, "
          "masque CE assistant-seul, padding de banque inerte, "
          "4 variantes forward live+ablaté)")
    print("  phase 2 — round-trip ORACLE (top-1 exacts / positions, vocab 512, "
          "d_model 256, n_pos 8) : " +
          "  ".join(f"{c} {rt[c][0]}/{rt[c][1]}" for c in CODES) +
          "   [mean DOIT échouer : anagrammes ⇒ ligne identique]")
    print("  phase 2 — lane vide ≡ ablaté & porte fermée pour les 4 formats, "
          "FIFO rows (3 faits × 3 tokens ⇒ 8 lignes), r0/r1/r2 refusent "
          "chunk/phase/rows")


if __name__ == "__main__":
    main()
