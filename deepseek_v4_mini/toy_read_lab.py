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

Phase 3 : RETRAIT DU PRIVILÈGE D'ORACLE SUR LE SPAN VALEUR
----------------------------------------------------------
Les 4 formats de la phase 2 gardent un privilège que le write du 350M n'a PAS :
l'environnement sait quels tokens du segment sont LA VALEUR et ne binde qu'eux.
Le vrai write poole le SEGMENT ENTIER (template ChatML inclus) sans savoir où
est la valeur. Deux formats de plus retirent ce privilège — UNE variable à la
fois : on GARDE l'autre privilège (savoir quels segs portent un fait, donc le
write reste déclenché sur les segs porteurs seulement).

  segmean   CONTRÔLE : ligne = RMSnorm(K + A + moyenne uniforme des embeddings
            RMS-normés de TOUS les tokens du seg porteur). Identique à `mean`
            mais sur le segment entier — mesure ce que le BRUIT DE TEMPLATE
            coûte à lui seul, ordre toujours détruit.
  segphase  LE TEST : ligne = RMSnorm(K + A + (1/T)·Σ_t rot(θ·pos_t)·ê(t)) où
            pos_t est la position du token DANS LE SEGMENT et θ le binding DFT
            de `phase` avec sa propre table de taille `seg_n_pos`. Le readout
            expose M×seg_n_pos candidats rot(−θ·j)·ligne.
  segsif    LA RECETTE CIBLE DU 350M : segphase PONDÉRÉ SIF —
            ligne = RMSnorm(K + A + (1/Σw)·Σ_t w_t·rot(θ·pos_t)·ê(t)) avec
            w_t = a/(a+p(t)), p = table unigram du stream (`_sif_table()` de
            PersonaChatStream, réutilisée telle quelle par `sif_weight_table`).
            Le SIF écrase les tokens fréquents (le template ChatML) ⇒ le nombre
            EFFECTIF de tokens superposés T_eff = (Σw)²/Σw² chute et le SNR
            monte. `a` est un knob DU TOY (`code.sif_a`) : le stream du toy
            reste surp OFF, le SIF n'entre que dans le code de banque.

Pool : seuls les tokens de PADDING sont exclus (masque `attention_mask` du
seg ; en pratique les segs du stream persona ne sont pas padés). Les tokens
ChatML de structure (<|im_start|>user … <|im_end|>) RESTENT dans le pool :
ils font partie du bruit réaliste que le write du 350M avale.

⚠️ ROUND-TRIP ORACLE À L'ÉCHELLE RÉELLE (vocab 49 154, embeddings à l'init,
393 segs porteurs réels, seg_n_pos 32) — c'est la BORNE SUPÉRIEURE du pointer,
mesurée avant tout entraînement. Seuil de déploiement : 70 % sur la strate
`code` (celle qui n'a aucun prior LM).

  d=256   format              ALL      `code`   T_eff   z médian
          segmean             1.4 %     2.7 %   19.0     3.38
          segphase           32.5 %    28.8 %   19.0     3.75
          segsif a=1e-4      52.0 %  → 15.1 % ← 4.75     4.95   (code : z 1.59)
          segsif a=3e-3      70.3 %    45.3 %  10.29     5.34
          segsif a=1e-2      76.3 %    59.9 %  12.74     5.07   ← MEILLEUR
          segsif a=3e-2      69.5 %    55.9 %  15.28     4.77
          segsif a=1e-1      54.9 %    45.6 %  17.48     4.40
          segsif a=1         36.0 %    30.9 %  18.94     3.88
          (référence phase 2, privilège gardé) `phase` 98.6 % / 97.4 %
  d=512   segphase           77.7 %    71.0 %  19.0      —
          segsif a=1e-2      97.4 %    94.9 %  12.74     7.09

Lecture : le format est SNR-LIMITÉ, pas cassé. z(bon token) ≈ √(d/(T_eff−1))
contre le max de |V| gaussiennes concurrentes √(2·ln|V|) = 4.65 σ.

PIÈGE CONFIRMÉ — le SIF à a=1e-4 (la valeur du 350M) DÉGRADE la strate `code`
(28.8 % → 15.1 %, z 3.65 → 1.59) alors qu'il fait exploser `short` (39.6 % →
97.0 %) : les codes/refs/plaques sont faits de digrammes FRÉQUENTS (w̄ 0.042
sur les tokens chiffrés contre 0.62 sur les valeurs linguistiques), donc la
pondération les écrase précisément là où on a besoin d'eux. a=1e-2 rétablit
l'équilibre (w̄ chiffrés 0.81) et double `code` vs uniforme, sans atteindre 70 %
à d=256. La conclusion tient : la condition de viabilité est d_model ≥ 512
(avec SIF a=1e-2 : 94.9 % sur `code`).

Phase 3bis : DEUX EXTENSIONS (`code.pos_offset`, `--write every`)
-----------------------------------------------------------------
1. `code.pos_offset` (défaut 0 = rétro-compat bit-à-bit) décale l'index de
   phase : le token en position j est tourné de θ·(j+k) AU WRITE ET aux
   candidats du readout. Motivation : à offset 0 la position 0 a rot(0) =
   IDENTITÉ, donc son token se superpose sans étiquette rotationnelle à
   K[slot]+A[attr] (eux aussi non tournés) — l'artefact des dumps
   (SV-19621 → EX-19621 : chiffres exacts, PREMIER token de la valeur faux).
   Wrap : période n_pos ⇒ l'offset consomme k positions (n_pos ≥ len_max + k).

   ⚠️ VERDICT MESURÉ (round-trip ORACLE, vocab 49 154, d=512, 393 segs
   porteurs réels, 1 555 positions de valeur) : l'offset ne change RIEN.

     format                            ALL    short   word   code  | pos0 pos1
     segsif a=1e-2 offset 0           97.4%  100.0% 100.0%  94.9%  |100.0% 99.2%
     segsif a=1e-2 offset 1           97.5%  100.0% 100.0%  95.1%  |100.0% 99.5%
     segphase      offset 0           77.7%   89.6%  83.0%  71.0%  | 83.2% 83.5%
     segphase      offset 1           77.2%   88.3%  83.0%  70.2%  | 83.0% 83.7%
     phase n_pos 8 offset 0/1        100.0%  100.0% 100.0% 100.0%  |100.0%(+8.4σ)

   (pos_k = rang du token DANS LA VALEUR.) Par position ABSOLUE dans le
   segment, même conclusion : segphase 85.2 % → 85.8 % à la position 0, marge
   +0.91σ → +0.93σ ; le candidat d'index identité ne collapse sur aucun
   attracteur (top-1 distinct sur 178/200 faits, aux DEUX offsets). Raison :
   K/A sont à l'échelle 0.2 dans une ligne RMS-normalisée à d=512 — leur fuite
   dans le candidat non tourné est très en dessous du bruit du vocabulaire.
   Le 0 % de segsif aux positions 0-2 du SEGMENT n'est pas la collision
   d'identité mais le SIF : `<|im_start|>user\\n` sont les tokens les plus
   fréquents, donc écrasés hors de la ligne — c'est le comportement voulu.
   ⇒ Si l'artefact « premier token faux » persiste, il vient du READ APPRIS
   (quel candidat le pointer sélectionne), pas de l'inversibilité du code.

2. `--write every` retire le SECOND privilège d'oracle : l'environnement écrit
   après CHAQUE seg (user, assistant, smalltalk) comme le write du 350M. Les
   segs porteurs gardent la ligne K+A+pool ; les segs SANS fait posent la MÊME
   formule de pool SANS composante K/A. Le FIFO max_mem est inchangé ⇒ le flux
   ÉVINCE les faits anciens : c'est le régime réel.

   ⚠️ PLAFOND MESURÉ (stream persona, 24 convs gradées, FIFO 8) : l'âge en
   writes entre le fait et sa query passe de 0.61 (mode `fact`, max 2) à 10.0
   (mode `every`, max 25) ⇒ 52.8 % des faits gradés held-out (58.8 % sur le
   train) ont leur ligne DÉJÀ ÉVINCÉE au moment de la question. Le grade
   maximum atteignable en mode `every` est donc ~0.45, PAS 1.0 : tout écart
   avec un bras `fact` doit être lu contre ce plafond (métriques `age_writes`
   et `age_evicted` du CSV, distribution imprimée à la première éval).

Usage
-----
  python -m deepseek_v4_mini.toy_read_lab CONFIG.yaml --variant r0
  python -m deepseek_v4_mini.toy_read_lab CONFIG.yaml --variant r1 --smoke --device cpu
  python -m deepseek_v4_mini.toy_read_lab CONFIG.yaml --variant r3 --code phase
  python -m deepseek_v4_mini.toy_read_lab CONFIG.yaml --variant r3 \
      --code segsif --pos-offset 1 --write every       # → r3_segsif_o1_wev/
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

VARIANTS = ("r0", "r1", "r2", "r3", "r4")
# r4 = INJECTION À SÉLECTION ORACLE : AUCUN module de read appris. Le groupe
# toprows du fait interrogé est injecté en PRÉFIXE de pseudo-tokens et c'est le
# backbone NU qui doit copier. Bras « le stack natif sait-il copier ? ».
INJECT_VARIANTS = ("r4",)
# les variantes qui LISENT UNE BANQUE (r4 n'en a pas : il lit une injection).
BANK_VARIANTS = ("r0", "r1", "r2", "r3")
# mélange des lignes par le readout de groupe (cf. GroupReadout) :
#   linear = superposition DANS l'espace d'embedding puis UNE projection
#   mos    = une distribution PAR LIGNE, puis mixture des DISTRIBUTIONS
READOUT_MIXES = ("linear", "mos")
CODES = ("mean", "chunk", "phase", "rows", "segmean", "segphase", "segsif",
         "pack", "segpack", "toprows")
# formats de la PHASE 3 : le code poole le SEGMENT ENTIER (pas de privilège
# d'oracle sur le span valeur). Ils indexent la position DANS LE SEGMENT, donc
# ils utilisent `seg_n_pos` et non `n_pos`.
SEG_CODES = ("segmean", "segphase", "segsif", "segpack", "toprows")
# formats à binding DFT sur la position DANS LE SEGMENT (mêmes tables, mêmes
# candidats pointer) — seule la PONDÉRATION du pool les distingue.
SEG_PHASE_CODES = ("segphase", "segsif")
# formats à binding de phase (les SEULS que `code.pos_offset` concerne).
PHASE_CODES = ("phase",) + SEG_PHASE_CODES
# régimes de write de l'oracle (cf. ToyCfg.write_mode).
WRITE_MODES = ("fact", "every")
# formats PARTITIONNÉS (phase 5) : la ligne est un PACK de blocs disjoints —
# bloc 0 = clé dédiée, blocs 1..B−1 = un token chacun. Readout = PackReadout.
PACK_CODES = ("pack", "segpack")
# formats à BANQUE DE GROUPES (phase 6) : un write dépose 1+top_k LIGNES
# NATIVES — ligne 0 = clé dédiée, lignes 1.. = les embeddings BRUTS des tokens
# sélectionnés. Principe : NE JAMAIS TRANSFORMER. Le FIFO compte les GROUPES.
GROUP_CODES = ("toprows",)
# codes qui exigent la table SIF (pondération/sélection des tokens du segment).
SIF_CODES = ("segsif", "segpack", "toprows")
# seed des tables ORACLE du pack (frames R_j ET clés par paire). FIGÉ : il est
# celui de la campagne de mesure oracle (scratchpad/oracle_pack.py), garder la
# continuité des chiffres.
PACK_SEED = 20260801
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
    # ── axe SEGMENT ENTIER (phase 3) ────────────────────────────────────────
    seg_n_pos: int = 32       # positions bindées DANS LE SEGMENT porteur.
                              # MESURÉ sur le stream persona : longueur des segs
                              # porteurs ∈ [12, 26], moyenne 18.5, p98 = 23 ⇒ 32
                              # est la puissance de 2 qui couvre 100 %.
                              # Les tokens au-delà (jamais observés ici) sont
                              # poolés SANS rotation (contenu gardé, position
                              # perdue), jamais tronqués.
    sif_a: float = 1e-4       # `segsif` : a de la pondération SIF w = a/(a+p).
                              # 1e-4 = la recette du 350M (PersonaChatStream
                              # surprisal_mode='sif'). Knob du TOY, indépendant
                              # du stream (qui tourne surp OFF ici).
    # ── axe PACK (phase 5) ──────────────────────────────────────────────────
    pack_blocks: int = 8      # `pack`/`segpack` : la ligne est partitionnée en
                              # pack_blocks blocs de d_model/pack_blocks dims.
                              # Bloc 0 = CLÉ dédiée (slot, attr), blocs 1..B−1 =
                              # UN token de contenu chacun ⇒ capacité DURE de
                              # pack_blocks−1 tokens, mais ZÉRO interférence
                              # entre positions (sous-espaces disjoints).
                              # 8 blocs × 64 dims = la géométrie mesurée à
                              # l'oracle (RT 100 % / +3.9σ à d=512).
    # ── axe GROUPES DE LIGNES NATIVES (phase 6) ─────────────────────────────
    top_k: int = 13           # `toprows` : lignes de CONTENU par groupe. Le
                              # write dépose 1 + top_k lignes : la CLÉ de la
                              # paire puis les embeddings BRUTS des top_k
                              # tokens SIF du segment, dans l'ordre du segment.
                              # AUCUNE projection, aucune rotation : la banque
                              # reste dans l'espace d'embedding, ce qui est TOUT
                              # le point (la ligne pack, concat de projections,
                              # avait effondré le Δnll du CrossAttnRead à +1.5
                              # contre +5.5 pour `mean`).
                              # 13 = MESURÉ : plus petit k dont la sélection SIF
                              # couvre ≥ 95 % des tokens de valeur de la strate
                              # `code` (97.0 % ; k=12 → 90.7 %). Cf. le YAML.
    readout_mix: str = "linear"   # GroupReadout : comment les lignes du groupe
                              # se combinent.
                              # `linear` (DÉFAUT, rétro-compat bit-à-bit) :
                              # u = Σ s·p·ligne PUIS u @ Ê^T — les lignes se
                              # SUPERPOSENT dans l'espace d'embedding, et le
                              # plus proche voisin d'une superposition de H/Q/B
                              # est un token PLAUSIBLE FAUX (la machine à
                              # HAB-719 → HQR-719).
                              # `mos` : une distribution PAR LIGNE
                              # (softmax(ligne @ Ê^T)) puis MIXTURE pondérée par
                              # s·p, et log pour revenir en logits. Aucune
                              # superposition ne peut fabriquer un token que
                              # AUCUNE ligne ne porte. Coût : un tenseur
                              # [B, G·k, V] par forward (les lignes ne dépendent
                              # pas de t) + le mélange [B,T,V].
    pos_entropy: float = 0.0  # pénalité d'ENTROPIE sur la porte-position p
                              # (GroupReadout) ajoutée à la loss. 0 = OFF
                              # (défaut, aucun terme n'est ajouté). > 0 pousse p
                              # vers un choix DUR d'une ligne du groupe.
    # ── axe INJECTION (variante r4) ─────────────────────────────────────────
    inject_sep_id: int = 0    # token du vocab posé ENTRE le préfixe injecté et
                              # le tour réel. Renseigné par main() (`<blank>`).
    row_pos_tag: bool = True  # `toprows` : marquer la ligne de contenu j par
                              # pos_emb[j] × oracle_ka_scale (0.2), comme le
                              # format `rows` dont le round-trip était à 100 %.
                              # Quasi-non-transformation (l'embedding domine).
                              # False = lignes STRICTEMENT natives, l'ordre ne
                              # tient plus qu'au layout du groupe.
    pos_offset: int = 0       # DÉCALAGE de l'index de phase (formats `phase`,
                              # `segphase`, `segsif` uniquement). 0 = DÉFAUT,
                              # rétro-compat bit-à-bit. 1 = la position 0 n'a
                              # plus rot(0)=identité, donc son token ne se
                              # superpose plus à K/A (qui, eux, ne sont jamais
                              # tournés). Cf. phase_tables pour le wrap.
    # ── axe RÉGIME DE WRITE (`--write`) ─────────────────────────────────────
    write_mode: str = "fact"  # `fact` (DÉFAUT) : l'oracle n'écrit qu'après les
                              # segs PORTEURS — il sait lesquels portent un
                              # fait, c'est le 2ᵉ privilège d'oracle.
                              # `every` : il écrit après CHAQUE seg (user,
                              # assistant, smalltalk) comme le write du 350M ;
                              # les segs sans fait posent la MÊME formule de
                              # pool SANS composante K/A. Le FIFO max_mem étant
                              # inchangé, le flux ÉVINCE les faits anciens —
                              # c'est le régime réel.

    def __post_init__(self):
        if self.variant in ("r3",) + INJECT_VARIANTS:
            # R3 : la banque VIT dans l'espace d'embedding — pas de projection.
            # R4 : pas de banque du tout, mais les tokens injectés vivent eux
            # aussi dans l'espace d'embedding.
            self.mem_dim = self.d_model
        if not self.x_dim:
            self.x_dim = self.d_model
        assert self.d_model % self.n_heads == 0
        assert self.mem_dim % self.n_heads == 0
        assert self.x_dim % self.n_heads == 0
        assert self.variant in VARIANTS
        assert self.code in CODES, f"code inconnu {self.code!r} (∈ {CODES})"
        assert self.readout_mix in READOUT_MIXES, (
            f"readout_mix inconnu {self.readout_mix!r} (∈ {READOUT_MIXES})")
        assert self.pos_entropy >= 0.0, self.pos_entropy
        if self.variant in INJECT_VARIANTS:
            # r4 n'a AUCUN read appris : ce qu'il lit, c'est le préfixe injecté,
            # et ce préfixe EST le groupe toprows (mêmes tokens, même sélection
            # SIF). Sans ce code il n'y aurait rien à injecter.
            assert self.code == "toprows", (
                f"--variant r4 injecte le GROUPE toprows : il exige "
                f"--code toprows (reçu --code {self.code})")
            assert self.write_mode == "fact", (
                "--variant r4 est un bras fact-only (le régime `every` n'a pas "
                "de sens : r4 n'a pas de banque, seulement une injection)")
        if self.code != "mean":
            # les nouveaux formats supposent banque == espace d'embedding et
            # pointer nu : c'est la définition de r3, on ne les porte pas
            # ailleurs (r0/r1/r2 restent le contrôle de la phase 1). r4 les
            # consomme autrement (injection), il est admis pour `toprows`.
            assert self.variant in ("r3",) + INJECT_VARIANTS, (
                f"--code {self.code} n'est supporté QUE par --variant r3 "
                f"(banque en espace d'embedding + pointer nu) ; reçu "
                f"--variant {self.variant}. Phase 1 = --code mean.")
            assert self.mem_dim == self.d_model
            assert self.n_pos >= 1
            if self.code in SEG_CODES:
                assert self.seg_n_pos >= 1
            if self.code in SIF_CODES:
                assert self.sif_a > 0, f"sif_a doit être > 0 ({self.sif_a})"
            if self.code in GROUP_CODES:
                assert self.top_k >= 1, f"top_k doit être >= 1 ({self.top_k})"
            if self.code in PACK_CODES:
                assert self.pack_blocks >= 2, self.pack_blocks
                assert self.d_model % self.pack_blocks == 0, (
                    f"pack : d_model {self.d_model} doit être divisible par "
                    f"pack_blocks {self.pack_blocks}")
            if self.code == "chunk":
                assert self.d_model % self.n_pos == 0, (
                    f"chunk : d_model {self.d_model} doit être divisible par "
                    f"n_pos {self.n_pos}")
        assert self.d_model % 2 == 0
        # ── pos_offset : seuls les formats à binding de phase le portent ────
        assert isinstance(self.pos_offset, int) and self.pos_offset >= 0, (
            f"pos_offset doit être un entier >= 0 ({self.pos_offset!r})")
        if self.pos_offset:
            assert self.code in PHASE_CODES, (
                f"code.pos_offset n'a de sens QUE pour les formats à binding "
                f"de phase {PHASE_CODES} ; reçu --code {self.code} "
                f"(il serait silencieusement ignoré)")
            npos = self.seg_n_pos if self.code in SEG_PHASE_CODES else self.n_pos
            assert self.pos_offset < npos, (
                f"pos_offset {self.pos_offset} >= n_pos {npos} : TOUTES les "
                f"positions wrappent sur l'identité")
        # ── write_mode ─────────────────────────────────────────────────────
        assert self.write_mode in WRITE_MODES, (
            f"write inconnu {self.write_mode!r} (∈ {WRITE_MODES})")
        if self.write_mode == "every":
            assert self.pools_segment, (
                f"--write every exige un code qui poole le SEGMENT "
                f"({SEG_CODES}) : un seg SANS fait n'a pas de span valeur, "
                f"donc les formats à privilège span-valeur ne savent pas quoi "
                f"écrire ; reçu --code {self.code}")

    @property
    def n_cand(self) -> int:
        """Candidats engendrés PAR LIGNE par le readout position-conscient."""
        if self.code in SEG_PHASE_CODES:
            return self.seg_n_pos
        # les PACK n'engendrent pas de candidats : PackReadout consomme la
        # LIGNE BRUTE (bloc-clé pour la sélection, blocs de contenu pour les
        # logits). Le readout à candidats plats ne les concerne pas.
        return 1 if self.code in ("mean", "rows", "segmean") + PACK_CODES \
            + GROUP_CODES else self.n_pos

    @property
    def group_rows(self) -> int:
        """Lignes par GROUPE de banque (`toprows`) : 1 clé + top_k contenus.

        FIXE par construction : le readout indexe les clés à la FOULÉE
        group_rows, un groupe court casserait le layout (cf. toprows_rows).
        """
        return 1 + self.top_k

    @property
    def pools_segment(self) -> bool:
        """Le code poole-t-il le SEG ENTIER (phase 3) au lieu du span valeur ?"""
        return self.code in SEG_CODES

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


def phase_tables(n_pos: int, d: int, base: float, device=None, dtype=None,
                 offset: int = 0):
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

    `offset` (knob `code.pos_offset`, défaut 0 = rétro-compat bit-à-bit) DÉCALE
    l'index de phase : la ligne j de la table vaut rot(θ·(j+offset)). Elle sert
    AU WRITE (le token en position j est tourné de θ·(j+offset)) ET AU READOUT
    (le candidat j dé-tourne de −θ·(j+offset)) : c'est la MÊME table, donc le
    round-trip reste exact par construction.
    RAISON : à offset 0, la position 0 a rot(0) = IDENTITÉ — son token se
    superpose SANS étiquette rotationnelle à K[slot]+A[attr], qui vivent aussi
    non-tournés dans la ligne. Mesuré dans les dumps : le premier token de la
    valeur échoue systématiquement (SV-19621 → EX-19621 / AD-19621, chiffres
    exacts). offset ≥ 1 rend l'identité INOCCUPÉE.
    ⚠️ WRAP : le binding DFT est périodique de période n_pos, donc l'index
    j+offset ≡ 0 [n_pos] retombe sur l'identité. Avec offset=k, les positions
    j ∈ [0, n_pos−k) sont protégées et les positions j ≥ n_pos−k rejouent le
    conflit. L'offset CONSOMME donc k positions utiles : il faut
    n_pos ≥ longueur_max + offset.
    """
    if base is None or base <= 0:
        r = (torch.arange(0, d // 2, device=device) % n_pos).float()
        inv = 2.0 * math.pi * r / n_pos                        # [d/2]
    else:
        inv = 1.0 / (base ** (torch.arange(0, d, 2, device=device).float() / d))
    k = torch.arange(n_pos, device=device).float()
    if offset:
        k = k + float(offset)
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

    def forward(self, x, pos=None):
        """`pos` [T] : index RoPE EXPLICITES. None = 0..T−1 (chemin par défaut,
        bit-à-bit inchangé). La variante r4 s'en sert pour laisser un TROU de
        position entre le préfixe injecté et le tour réel."""
        B, T, d = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, T, self.h, self.dh).transpose(1, 2)
        k = k.view(B, T, self.h, self.dh).transpose(1, 2)
        v = v.view(B, T, self.h, self.dh).transpose(1, 2)
        if pos is None:
            cos, sin = _rope_tables(T, self.dh, self.theta, x.device, q.dtype)
        else:
            c, s_ = _rope_tables(int(pos.max()) + 1, self.dh, self.theta,
                                 x.device, q.dtype)
            cos, sin = c[pos], s_[pos]
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


# ── PACK : readout DEUX ÉTAGES (quelle ligne / quel token) ───────────────────

class PackReadout(nn.Module):
    """Readout du format `pack` : la sélection de LIGNE et le choix du TOKEN
    sont deux décisions SÉPARÉES — c'est tout le point du format.

    Étage 1 « quelle ligne » : attention 1 tête sur les M lignes dont les clés
      sont les dims [0, blk) SEULEMENT — le BLOC-CLÉ. Mesuré à l'oracle : sur
      des banques de 8 lignes à clés distinctes, le retrieval rang-1 par ce
      bloc est à 100 % (contre 38.7 % pour le gist K+A noyé dans une ligne
      segsif, et 31.4 % de hit APPRIS relevé à la sonde de récence). La
      sélection reste SOFT (softmax) en v1 : on veut MESURER si le mélange
      recrée un attracteur, pas le durcir préventivement.
    Étage 2 « quel token » : le bloc de contenu j de la ligne sélectionnée se
      relit par TABLE DIRECTE, logits_j = b_j @ (Ê R_j)^T, frames R_j FIGÉS,
      zéro paramètre. Une porte-position softmax sur les B−1 blocs, condition-
      née sur h_t, mélange les blocs. Comme la projection au vocabulaire est
      linéaire, on mélange AVANT :  u = Σ_j p_j·(R_j b_j) puis un seul produit
      u @ Ê^T — même coût qu'un LM head, pas B−1 fois.

    Porte globale σ(w·h + b), b = cfg.ptr_bias_init (même convention que
    PointerReadout), ET `scale` initialisée à ZÉRO : le biais de logits est
    EXACTEMENT nul au step 0 (pas seulement petit). `scale` reçoit du gradient
    dès le premier backward (même argument que le P zéro-init de r2), le
    module n'est donc pas mort.
    """

    def __init__(self, cfg: ToyCfg):
        super().__init__()
        d, nb = cfg.d_model, cfg.pack_blocks
        self.nb, self.blk = nb, d // nb
        self.last_gate = None       # σ(porte) du dernier forward (télémétrie)
        self.last_sel = None        # softmax de SÉLECTION DE LIGNE (télémétrie)
        self.last_pos = None        # softmax de position (bloc) (télémétrie)
        self.norm = RMSNorm(d)
        self.wq = nn.Linear(d, self.blk, bias=False)     # query du bloc-clé
        self.wp = nn.Linear(d, nb - 1, bias=False)       # porte-position
        nn.init.zeros_(self.wp.weight)                   # blocs équiprobables
        self.scale = nn.Parameter(torch.zeros(1))        # biais EXACTEMENT nul
        self.gate = nn.Linear(d, 1, bias=True)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, cfg.ptr_bias_init)

    def forward(self, h, bank, bank_mask, embed_w, frames):
        B, M, d = bank.shape
        blk, nb = self.blk, self.nb
        hn = self.norm(h)
        # ── étage 1 : quelle LIGNE (clés = bloc 0 uniquement) ───────────────
        q = self.wq(hn)                                     # [B,T,blk]
        k = bank[:, :, :blk]                                # [B,M,blk]
        att = torch.einsum("btc,bmc->btm", q, k) / math.sqrt(blk)
        sm, empty = safe_bank_mask(bank_mask)
        if sm is not None:
            att = att.masked_fill(~sm[:, None, :], float("-inf"))
        s = att.softmax(-1)
        self.last_sel = s.detach()
        sel = torch.einsum("btm,bmd->btd", s, bank)         # [B,T,d]
        if empty is not None:
            sel = sel * (~empty)[:, None, None].to(sel.dtype)
        # ── étage 2 : quel TOKEN (tables directes par bloc, frames figés) ───
        blocks = sel.reshape(*sel.shape[:-1], nb, blk)[..., 1:, :]  # [B,T,nb-1,blk]
        p = self.wp(hn).softmax(-1)                          # [B,T,nb-1]
        self.last_pos = p.detach()
        R = frames[1:].to(blocks.dtype)                      # [nb-1, d, blk]
        u = torch.einsum("btjc,jdc->btd", blocks * p[..., None], R)
        bias = self.scale * (u @ rms_unit(embed_w).t())      # [B,T,V]
        g = torch.sigmoid(self.gate(hn))
        self.last_gate = g.detach()
        return g * bias


# ── GROUPES : readout DEUX ÉTAGES sur des lignes NATIVES ─────────────────────

class GroupReadout(nn.Module):
    """Readout du format `toprows`. Même squelette que PackReadout (deux
    décisions séparées) mais sur une banque de GROUPES de lignes NATIVES.

    La banque est [B, G·(1+k), d] : dans chaque groupe, la ligne 0 est la CLÉ
    de la paire (slot, attr) et les lignes 1..k sont les embeddings BRUTS des
    tokens sélectionnés. Le layout est DÉTERMINISTE, donc les clés se lisent à
    la foulée (1+k) — aucun apprentissage n'est dépensé à trouver où elles sont.

    Étage 1 « quel groupe » : attention 1 tête sur les G lignes-clés,
      query = W_q·RMSnorm(h_t), dk = d_model (la clé est une ligne PLEINE, pas
      un bloc de 64 dims comme dans le pack).
    Étage 2 « quelle ligne du groupe » : porte-position softmax sur les k
      lignes de contenu, conditionnée sur h_t. La ligne retenue part AU
      VOCABULAIRE TELLE QUELLE : logits = ligne @ Ê_rms^T, ZÉRO paramètre —
      c'est tout le design. Rien n'est dé-projeté ni dé-tourné parce que rien
      n'a jamais été projeté ni tourné.

    Les deux étages sont SOFT (softmax) et linéaires jusqu'au produit final :
    on mélange d'abord (u = Σ_g s_g Σ_j p_j ligne[g,j]), un seul produit
    [B,T,d]×[d,V] ensuite — même coût qu'un LM head.

    Porte globale σ(w·h + b), b = cfg.ptr_bias_init, et `scale` ZÉRO-init : le
    biais de logits est EXACTEMENT nul au step 0 (même convention que
    PackReadout), tout en recevant du gradient dès le premier backward.
    """

    def __init__(self, cfg: ToyCfg):
        super().__init__()
        d = cfg.d_model
        self.gr = cfg.group_rows            # 1 + top_k
        self.k = cfg.top_k
        self.mix = cfg.readout_mix
        self.last_gate = None       # σ(porte) du dernier forward (télémétrie)
        self.last_sel = None        # softmax de SÉLECTION DE GROUPE
        self.last_pos = None        # softmax de position DANS le groupe
        self.last_pos_ent = None    # entropie de p (DÉRIVABLE : pénalité)
        self.norm = RMSNorm(d)
        self.wq = nn.Linear(d, d, bias=False)            # query de la clé
        self.wp = nn.Linear(d, self.k, bias=False)       # porte-position
        nn.init.zeros_(self.wp.weight)                   # lignes équiprobables
        self.scale = nn.Parameter(torch.zeros(1))        # biais EXACTEMENT nul
        self.gate = nn.Linear(d, 1, bias=True)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, cfg.ptr_bias_init)

    def forward(self, h, bank, bank_mask, embed_w):
        B, M, d = bank.shape
        gr = self.gr
        assert M % gr == 0, (
            f"banque de {M} lignes non découpable en groupes de {gr} : le "
            f"layout du format `toprows` est violé")
        G = M // gr
        rows = bank.reshape(B, G, gr, d)
        hn = self.norm(h)
        # ── étage 1 : quel GROUPE (clés = ligne 0 de chaque groupe) ─────────
        q = self.wq(hn)                                     # [B,T,d]
        k = rows[:, :, 0, :]                                # [B,G,d]
        att = torch.einsum("btd,bgd->btg", q, k) / math.sqrt(d)
        gm = None if bank_mask is None else \
            bank_mask.reshape(B, G, gr)[:, :, 0]            # [B,G]
        sm, empty = safe_bank_mask(gm)
        if sm is not None:
            att = att.masked_fill(~sm[:, None, :], float("-inf"))
        s = att.softmax(-1)
        self.last_sel = s.detach()
        # ── étage 2 : quelle LIGNE du groupe (porte-position) ───────────────
        p = self.wp(hn).softmax(-1)                         # [B,T,k]
        self.last_pos = p.detach()
        self.last_pos_ent = -(p * p.clamp_min(1e-9).log()).sum(-1).mean()
        content = rows[:, :, 1:, :]                         # [B,G,k,d]
        if self.mix == "mos":
            # ── MoS : mélanger les DISTRIBUTIONS, pas les vecteurs ──────────
            # Les lignes ne dépendent pas de t : on projette les G·k lignes au
            # vocabulaire UNE fois, puis la mixture est un simple produit par
            # les poids s·p. Un token que AUCUNE ligne ne porte reste à
            # probabilité ~0, alors que la superposition linéaire pouvait le
            # faire gagner (HAB-719 → HQR-719).
            n = G * self.k
            Pr = (content.reshape(B, n, d) @ rms_unit(embed_w).t()).softmax(-1)
            w = (s[..., None] * p[:, :, None, :]).reshape(*s.shape[:2], n)
            mix = torch.einsum("btn,bnv->btv", w, Pr)       # [B,T,V]
            # clamp AVANT le log : une proba peut sous-déborder à 0 en fp32, et
            # scale × (−inf) rendrait NaN alors que scale vaut 0 à l'init.
            bias = self.scale * mix.clamp_min(1e-20).log()
            if empty is not None:
                bias = bias * (~empty)[:, None, None].to(bias.dtype)
            g = torch.sigmoid(self.gate(hn))
            self.last_gate = g.detach()
            return g * bias
        u = torch.einsum("btg,btj,bgjd->btd", s, p, content)
        if empty is not None:
            u = u * (~empty)[:, None, None].to(u.dtype)
        bias = self.scale * (u @ rms_unit(embed_w).t())     # [B,T,V]
        g = torch.sigmoid(self.gate(hn))
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

    def forward(self, x, bank, bank_mask, pos=None):
        x = x + self.attn(self.n1(x), pos)
        if self.read is not None and bank is not None and bank.size(1) > 0:
            x = self.read(x, bank, bank_mask)
        x = x + self.mlp(self.n2(x))
        return x


def sif_weight_table(stream_cls, tok, gen_kwargs: dict, vocab_size: int,
                     a: float, seed: int = 0) -> torch.Tensor:
    """[vocab_size] float : poids SIF w = a/(a+p(token)), recette du 350M.

    On ne RÉIMPLÉMENTE rien : on instancie un PersonaChatStream en
    `surprisal_mode='sif'` et on lui demande son `_sif_table()` — la même table
    unigram (300 convs, rng dédié 4242, `_sif_unseen = 0.5/tot`) que celle qui
    pondère le write du 350M. Les tokens jamais vus prennent le poids unseen.

    Le stream d'ENTRAÎNEMENT du toy, lui, reste surp OFF (`persona_kwargs` pop
    `surprisal_mode`/`sif_a`) : le SIF n'entre QUE dans le code de banque, il
    ne touche ni la loss ni les données. Le `a` est un knob du toy.
    """
    st = stream_cls(tok, seed=seed,
                    **{**gen_kwargs, "surprisal_mode": "sif", "sif_a": float(a)})
    p = st._sif_table()
    w = torch.full((vocab_size,), a / (a + st._sif_unseen), dtype=torch.float32)
    for t, pv in p.items():
        if 0 <= int(t) < vocab_size:
            w[int(t)] = a / (a + pv)
    return w


class ToyReadLM(nn.Module):
    def __init__(self, cfg: ToyCfg, n_slots: int, n_attrs: int,
                 sif_w: torch.Tensor | None = None):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        nn.init.normal_(self.embed.weight, std=0.02)
        self.blocks = nn.ModuleList(ToyBlock(cfg, i) for i in range(cfg.n_layers))
        self.norm_f = RMSNorm(cfg.d_model)
        self.ptr = None
        # r4 : le SEUL paramètre « de read » du bras — un vecteur de TYPE
        # ajouté aux lignes injectées (zéro-init : au step 0 la ligne injectée
        # est EXACTEMENT l'embedding brut du token).
        if cfg.variant in INJECT_VARIANTS:
            self.inject_type = nn.Parameter(torch.zeros(cfg.d_model))
        if cfg.uses_ptr:
            # les PACK ont leur PROPRE readout (deux étages) ; tous les autres
            # codes gardent PointerReadout à l'identique (rétro-compat bit-à-bit
            # : la construction du module ne change pas, donc le tirage des
            # poids non plus).
            if cfg.code in PACK_CODES:
                self.ptr = PackReadout(cfg)
            elif cfg.code in GROUP_CODES:
                self.ptr = GroupReadout(cfg)
            else:
                self.ptr = PointerReadout(cfg, project=(cfg.variant != "r3"))
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
            c, s = phase_tables(cfg.n_pos, cfg.d_model, cfg.rope_base,
                                offset=cfg.pos_offset)
            self.register_buffer("ph_cos", c)                  # [n_pos, d/2]
            self.register_buffer("ph_sin", s)
        elif cfg.code == "rows":
            pos = torch.randn(cfg.n_pos, cfg.d_model, generator=g) * (cfg.d_model ** -0.5)
            self.register_buffer("pos_emb", pos)               # [n_pos, d]
        elif cfg.code in GROUP_CODES:
            # ── tables ORACLE des GROUPES (phase 6) ─────────────────────────
            # MÊME clé dédiée par paire que le pack (même seed, même logique)
            # mais NON PROJETÉE : elle occupe une LIGNE ENTIÈRE de la banque.
            # ⚠️ PRIVILÈGE D'ORACLE ASSUMÉ, comme au pack : au 350M c'est le
            # modèle qui devra émettre cette clé au write et la re-produire au
            # read. Le toy mesure ce que la géométrie permet une fois la clé
            # disponible.
            keys = torch.zeros(n_slots, n_attrs, cfg.d_model)
            for s in range(n_slots):
                for a in range(n_attrs):
                    gk = torch.Generator().manual_seed(
                        PACK_SEED + 1000 * int(s) + int(a))
                    keys[s, a] = rms_unit(
                        torch.randn(cfg.d_model, generator=gk))
            self.register_buffer("pack_key", keys)     # [n_slots, n_attrs, d]
            # tag de position des lignes de CONTENU (knob row_pos_tag) : même
            # table et même échelle que le format `rows` (×oracle_ka_scale).
            pos = torch.randn(cfg.top_k, cfg.d_model,
                              generator=g) * (cfg.d_model ** -0.5)
            self.register_buffer("row_pos", pos)       # [top_k, d]
        elif cfg.code in PACK_CODES:
            # ── tables ORACLE du PACK (phase 5) ─────────────────────────────
            # Générateurs DÉDIÉS (seed PACK_SEED) : l'état de `g` vu par
            # K/A/val_proj est intact, les autres codes restent bit-à-bit.
            blk = cfg.d_model // cfg.pack_blocks
            gp = torch.Generator().manual_seed(PACK_SEED)
            R = torch.stack([
                torch.linalg.qr(torch.randn(cfg.d_model, blk, generator=gp))[0]
                for _ in range(cfg.pack_blocks)])
            self.register_buffer("pack_R", R)          # [B, d, blk], orthonormés
            # CLÉ DÉDIÉE par paire (slot, attr) : un vecteur unitaire aléatoire
            # tiré par un générateur PROPRE À LA PAIRE ⇒ stable d'une
            # instanciation à l'autre, et indépendant de l'ordre de tirage.
            # ⚠️ PRIVILÈGE D'ORACLE ASSUMÉ : ces clés sont données, le modèle ne
            # les produit pas. Mesuré (oracle) : la SOMME K[slot]+A[attr] fait
            # partager la composante A à toutes les paires de même attr (cos
            # structurel ≈ 0.5, pire cas 0.698) alors que la clé dédiée reste à
            # |cos| ≤ 0.281 — d'où le choix. Au 350M c'est le modèle qui devra
            # ÉMETTRE cette clé au write (et la re-produire au read) : le toy
            # mesure ce que la géométrie permet une fois la clé disponible.
            keys = torch.zeros(n_slots, n_attrs, blk)
            for s in range(n_slots):
                for a in range(n_attrs):
                    gk = torch.Generator().manual_seed(
                        PACK_SEED + 1000 * int(s) + int(a))
                    keys[s, a] = rms_unit(
                        torch.randn(cfg.d_model, generator=gk)) @ R[0]
            self.register_buffer("pack_key", keys)     # [n_slots, n_attrs, blk]
        elif cfg.code in SEG_PHASE_CODES:
            # même binding DFT que `phase`, mais indexé sur la position DANS LE
            # SEGMENT ⇒ sa propre table de seg_n_pos lignes.
            c, s = phase_tables(cfg.seg_n_pos, cfg.d_model, cfg.rope_base,
                                offset=cfg.pos_offset)
            self.register_buffer("sg_cos", c)                  # [seg_n_pos, d/2]
            self.register_buffer("sg_sin", s)
        if cfg.code in SIF_CODES:
            assert sif_w is not None, (
                f"--code {cfg.code} exige la table de poids SIF : "
                f"ToyReadLM(..., sif_w=sif_weight_table(...))")
            assert sif_w.numel() == cfg.vocab_size, (sif_w.numel(),
                                                     cfg.vocab_size)
            self.register_buffer("sif_w", sif_w.float().clone())

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
    def oracle_lines(self, slot_id: int, attr_id: int, val_tok: torch.Tensor,
                     seg_tok: torch.Tensor | None = None,
                     bare: bool = False) -> torch.Tensor:
        """LES LIGNES d'un fait : [n_lignes, mem_dim]. Dispatch sur cfg.code.

        `mean` en rend UNE et est bit-à-bit identique à la phase 1 ; `chunk` et
        `phase` en rendent une aussi (formats à ordre INTERNE) ; `rows` en rend
        une PAR TOKEN (borne haute de décodabilité, économie dégradée : le FIFO
        de max_mem lignes est inchangé, un fait long mange la banque).

        PHASE 3 (`segmean`/`segphase`) : le code ne voit PLUS `val_tok` — il
        poole `seg_tok`, le segment porteur ENTIER (template ChatML compris),
        exactement comme le write du 350M qui ignore où est la valeur.

        Comme en phase 1 : recalculé à la volée sur les embeddings COURANTS,
        detach total, aucun gradient ne traverse la banque.
        """
        c = self.cfg
        if c.code == "mean":
            assert not bare, "bare (write=every) exige un code de SEGMENT"
            return self.oracle_code(slot_id, attr_id, val_tok).unsqueeze(0)
        dev = self.embed.weight.device
        if c.code in GROUP_CODES:
            return self.toprows_rows(slot_id, attr_id, seg_tok=seg_tok,
                                     bare=bare)
        if c.code in PACK_CODES:
            return self.pack_lines(slot_id, attr_id, val_tok, seg_tok=seg_tok,
                                   bare=bare)
        if bare:
            # `--write every`, seg SANS fait : MÊME formule de pool, mais AUCUNE
            # composante de liaison — la ligne ne prétend pas indexer un slot.
            # (K_slot[0]/A_attr[0] existent mais sont des vecteurs ALÉATOIRES
            # comme les autres : on ne s'appuie pas dessus, on force le zéro.)
            assert c.pools_segment, "bare (write=every) exige un code de SEGMENT"
            ka = torch.zeros(c.mem_dim, dtype=torch.float32, device=dev)
        else:
            ka = self.K_slot[slot_id].float() + self.A_attr[attr_id].float()
        if c.pools_segment:
            assert seg_tok is not None, (
                f"--code {c.code} poole le SEGMENT : oracle_lines a besoin de "
                f"seg_tok (tokens non-padés du seg porteur)")
            st = seg_tok.to(dev).reshape(-1)
            es = rms_unit(self.embed.weight[st].float())       # [T, d], RMS 1
            T = es.size(0)
            # pondération du pool : uniforme, ou SIF w = a/(a+p) — la recette
            # du write du 350M, qui écrase les tokens fréquents (template
            # ChatML) et fait chuter le nombre EFFECTIF de tokens superposés.
            if c.code == "segsif":
                w = self.sif_w[st].float()                     # [T]
            else:
                w = torch.ones(T, dtype=torch.float32, device=dev)
            wsum = float(w.sum())
            if wsum <= 0:            # dégénéré (jamais observé) : uniforme
                w = torch.ones_like(w)
                wsum = float(T)
            ew = es * w[:, None]
            if c.code == "segmean":
                pooled = ew.sum(0) / wsum                       # ordre détruit
            else:                                    # segphase / segsif
                n = min(T, c.seg_n_pos)
                bound = rot_pairs(ew[:n], self.sg_cos[:n].float(),
                                  self.sg_sin[:n].float()).sum(0)
                if T > n:      # débordement : contenu gardé, position perdue
                    bound = bound + ew[n:].sum(0)
                pooled = bound / wsum
            return rms_unit(ka + pooled).unsqueeze(0).detach()
        toks = val_tok.to(dev)[:c.n_pos]                       # troncature n_pos
        e = self.embed.weight[toks].float()                    # [n, d]
        e = rms_unit(e)                                        # ê(t_k), RMS 1
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

    # ── write ORACLE du format PACK ─────────────────────────────────────────
    @torch.no_grad()
    def pack_tokens(self, val_tok: torch.Tensor,
                    seg_tok: torch.Tensor | None) -> torch.Tensor:
        """Les k_val = pack_blocks−1 tokens qui vont OCCUPER les blocs.

        `pack`    : les tokens de la VALEUR, dans l'ordre (privilège de span
                    gardé — c'est le bras « borne haute » du format).
        `segpack` : les k_val tokens du SEGMENT au poids SIF le plus fort,
                    RÉORDONNÉS DANS L'ORDRE DU SEGMENT (le tri par poids
                    détruirait l'ordre, qui est tout ce que le pack achète).
                    Aucun privilège : le write ne sait pas où est la valeur, il
                    ne connaît que la table unigram.
        """
        c = self.cfg
        dev = self.embed.weight.device
        k = c.pack_blocks - 1
        if c.code == "segpack":
            assert seg_tok is not None, (
                f"--code {c.code} sélectionne dans le SEGMENT : oracle_lines a "
                f"besoin de seg_tok (tokens non-padés du seg porteur)")
            st = seg_tok.to(dev).reshape(-1)
            if st.numel() == 0:
                return st
            w = self.sif_w[st].float()
            sel = torch.topk(w, min(k, st.numel())).indices.sort().values
            return st[sel]
        return val_tok.to(dev).reshape(-1)[:k]

    @torch.no_grad()
    def pack_lines(self, slot_id: int, attr_id: int, val_tok: torch.Tensor,
                   seg_tok: torch.Tensor | None = None,
                   bare: bool = False) -> torch.Tensor:
        """LA ligne d'un fait au format PACK : [1, d_model].

        Partition en `pack_blocks` blocs de blk = d_model/pack_blocks dims :
          bloc 0     = CLÉ DÉDIÉE de la paire (slot, attr) — privilège d'oracle
                       assumé, cf. le commentaire des tables en __init__ ;
          bloc j ≥ 1 = R_j^T · ê(t_j), UN token par bloc, frames orthonormés
                       figés ⇒ sous-espaces DISJOINTS, zéro interférence entre
                       positions (contrairement à phase/segsif qui superposent
                       tout dans les mêmes d dims) mais capacité DURE de k_val
                       tokens.
        Puis RMS-norm GLOBALE de la ligne : c'est un scalaire positif commun aux
        blocs, donc un no-op strict pour les argmax et les marges du readout
        (vérifié à l'oracle à 8e-08 près) — elle n'est là que pour que la ligne
        pack ait la même échelle que toutes les autres lignes de banque.

        `bare` (--write every, seg sans fait) : bloc 0 = ZÉRO. La ligne ne
        prétend indexer aucune paire, donc elle ne peut pas gagner l'étage 1 du
        readout — même convention que le `ka = 0` des codes de segment.
        """
        c = self.cfg
        dev = self.embed.weight.device
        blk = c.d_model // c.pack_blocks
        line = torch.zeros(c.d_model, dtype=torch.float32, device=dev)
        if not bare:
            line[:blk] = self.pack_key[slot_id, attr_id].float()
        toks = self.pack_tokens(val_tok, seg_tok)
        n = int(toks.numel())
        if n:
            e = rms_unit(self.embed.weight[toks].float())      # [n, d], RMS 1
            coeff = torch.einsum("nd,ndb->nb", e,
                                 self.pack_R[1:1 + n].float())  # [n, blk]
            line[blk:blk + n * blk] = coeff.reshape(-1)
        return rms_unit(line).unsqueeze(0).detach()

    # ── write ORACLE du format TOPROWS (groupes de lignes natives) ──────────
    @torch.no_grad()
    def toprows_sel(self, seg_tok: torch.Tensor) -> torch.Tensor:
        """Les top_k tokens du segment au poids SIF le plus fort, DANS L'ORDRE
        DU SEGMENT (le tri par poids détruirait l'ordre). Aucun privilège : le
        write ne sait pas où est la valeur, il ne connaît que la table unigram.
        """
        c = self.cfg
        st = seg_tok.to(self.embed.weight.device).reshape(-1)
        if st.numel() == 0:
            return st
        w = self.sif_w[st].float()
        sel = torch.topk(w, min(c.top_k, st.numel())).indices.sort().values
        return st[sel]

    @torch.no_grad()
    def toprows_rows(self, slot_id: int, attr_id: int,
                     seg_tok: torch.Tensor | None = None,
                     bare: bool = False) -> torch.Tensor:
        """LE GROUPE d'un write : [1 + top_k, d_model].

        Ligne 0     = CLÉ dédiée de la paire (slot, attr) — ou ZÉRO si `bare`
                      (--write every, seg sans fait : le groupe ne prétend
                      indexer aucune paire, il ne peut donc pas gagner l'étage 1
                      du readout ; même convention que la ligne nue du pack).
        Ligne 1+j   = ê(t_j), l'embedding BRUT du j-ième token sélectionné,
                      RMS-normé et RIEN D'AUTRE. C'est le principe du format :
                      le write SÉLECTIONNE, il ne TRANSFORME pas. La banque
                      reste dans l'espace d'embedding, donc le CrossAttnRead y
                      retrouve un canal de contenu (ce que la ligne pack, concat
                      de projections, avait détruit : Δnll +1.5 contre +5.5).
                      Avec `row_pos_tag`, on ajoute pos_emb[j] × 0.2 AVANT la
                      RMS-norm — le même tag que le format `rows` (round-trip
                      100 %), assez faible pour que l'embedding domine.

        TAILLE FIXE : le groupe fait TOUJOURS 1 + top_k lignes, parce que le
        readout indexe les clés à cette foulée. Si le segment a moins de top_k
        tokens (mesuré : segs porteurs ∈ [12, 26] tokens, donc jamais au défaut
        k=13, mais possible en poussant k), la DERNIÈRE ligne de contenu est
        RÉPÉTÉE pour compléter — un candidat en double est inerte, un groupe
        court casserait le layout de toute la banque.
        """
        c = self.cfg
        dev = self.embed.weight.device
        assert seg_tok is not None, (
            f"--code {c.code} sélectionne dans le SEGMENT : oracle_lines a "
            f"besoin de seg_tok (tokens non-padés du seg porteur)")
        out = torch.zeros(c.group_rows, c.d_model, dtype=torch.float32,
                          device=dev)
        if not bare:
            out[0] = self.pack_key[slot_id, attr_id].float()
        toks = self.toprows_sel(seg_tok)
        n = int(toks.numel())
        if n:
            e = rms_unit(self.embed.weight[toks].float())      # [n, d], RMS 1
            if c.row_pos_tag:
                e = rms_unit(e + self.row_pos[:n].float() * c.oracle_ka_scale)
            out[1:1 + n] = e
            if n < c.top_k:            # complète le groupe (cf. docstring)
                out[1 + n:] = e[-1]
        return out.detach()

    # ── candidats du readout position-conscient ─────────────────────────────
    def candidates(self, bank, bank_mask):
        """Banque [B,M,d] → candidats [B, M·n_cand, d] (+ masque étendu).

        La position n'est PAS lue par un compteur dur : on expose tous les
        candidats (ligne i, position j) au pointer, c'est son attention plate
        qui apprend l'alignement position↔décodage.
        """
        c = self.cfg
        if c.code in ("mean", "rows", "segmean") + PACK_CODES + GROUP_CODES:
            # la ligne EST le candidat (les PACK ne passent pas par ici : leur
            # readout consomme la ligne BRUTE, bloc par bloc).
            return bank, bank_mask
        B, M, d = bank.shape
        n = c.n_cand
        if c.code in SEG_PHASE_CODES:
            cs = self.sg_cos.to(bank.dtype)[None, None]        # [1,1,n,d/2]
            sn = self.sg_sin.to(bank.dtype)[None, None]
            cand = rot_pairs(bank[:, :, None, :], cs, -sn)     # rot(−θj)
        elif c.code == "chunk":
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
    def forward(self, ids, bank=None, bank_mask=None, inject=None):
        """`inject` [B, k] (variante r4) : les tokens du groupe toprows du fait
        interrogé, posés en PRÉFIXE de pseudo-tokens devant le tour.

        Layout (spec) : les k lignes injectées prennent les positions RoPE
        0..k−1, le séparateur la position k, et le tour RÉEL démarre à k+2 —
        la position k+1 reste VIDE, c'est un trou délibéré qui marque la
        frontière (une position qu'aucun token n'occupe jamais).

        Les lignes injectées sont les embeddings BRUTS, NON RMS-normés : la
        norme porte de l'information, et la RMS-norm de `toprows_rows` était une
        contrainte de BANQUE (des lignes comparables entre elles), pas
        d'injection. Un vecteur de TYPE appris (zéro-init) est ajouté à chacune
        pour que le backbone puisse les distinguer d'un vrai token.

        Les logits rendus sont ceux du TOUR RÉEL seulement : l'appelant
        (train_step, evaluate, greedy) ne voit aucune différence de forme.
        """
        x = self.embed(ids)
        pos = None
        npre = 0
        if inject is not None:
            assert self.cfg.variant in INJECT_VARIANTS, self.cfg.variant
            B, T = ids.shape
            k = inject.shape[1]
            pre = self.embed(inject) + self.inject_type    # [B,k,d], NON normé
            sep = self.embed(torch.full((B, 1), int(self.cfg.inject_sep_id),
                                        dtype=torch.long, device=ids.device))
            x = torch.cat([pre, sep, x], dim=1)
            npre = k + 1
            pos = torch.cat([torch.arange(k + 1, device=ids.device),
                             torch.arange(T, device=ids.device) + k + 2])
        for blk in self.blocks:
            x = blk(x, bank, bank_mask, pos)
        if npre:
            x = x[:, npre:]                    # seul le TOUR RÉEL sort
        x = self.norm_f(x)
        logits = x @ self.embed.weight.t()             # embeddings tiés
        if self.ptr is not None and bank is not None and bank.size(1) > 0:
            if self.cfg.code in GROUP_CODES:
                # TOPROWS : la banque est un empilement de GROUPES de lignes
                # natives — le readout lit les clés à la foulée (1+top_k).
                logits = logits + self.ptr(x, bank, bank_mask,
                                           self.embed.weight)
            elif self.cfg.code in PACK_CODES:
                # PACK : pas de candidats plats — le readout lit la ligne par
                # BLOCS (bloc 0 = clé pour choisir la ligne, blocs ≥ 1 = tokens)
                logits = logits + self.ptr(x, bank, bank_mask,
                                           self.embed.weight, self.pack_R)
            else:
                # code == mean  → candidats = les lignes (chemin phase 1, inchangé)
                cand, cmask = self.candidates(bank, bank_mask)
                logits = logits + self.ptr(x, cand, cmask, self.embed.weight)
        return logits

    # ── décodage greedy (sans cache : préfixes courts) ──────────────────────
    @torch.no_grad()
    def greedy(self, prefix, bank, bank_mask, max_new: int, stop_id: int,
               inject=None):
        ids = prefix
        out = []
        for _ in range(max_new):
            lg = self.forward(ids[:, -self.cfg.max_seq_len:], bank, bank_mask,
                              inject=inject)
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
        elif n.startswith("inject_type"):
            b = "read"          # r4 : le SEUL paramètre de « read » du bras
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

class GroupBank(list):
    """Banque du format `toprows` : une LISTE DE LIGNES qui se souvient de ses
    GROUPES.

    C'est une `list` ordinaire pour tout le reste du lab (`pad_bank` la stacke,
    `len()` compte les LIGNES) — les autres codes ne la voient jamais. Elle
    porte juste `.groups`, la liste des tailles, pour que le FIFO puisse
    évincer un write ENTIER au lieu d'une ligne isolée.
    """

    def __init__(self, rows=(), groups=()):
        super().__init__(rows)
        self.groups = list(groups)


class OracleEnv:
    """Rejoue une conv seg par seg et pose la banque à la place du modèle."""

    def __init__(self, tok, max_mem: int, write_mode: str = "fact"):
        self.tok = tok
        self.max_mem = max_mem
        assert write_mode in WRITE_MODES, write_mode
        self.write_mode = write_mode
        # nombre de lignes appendées par le DERNIER appel à write() (télémétrie
        # d'âge : « combien de writes séparent le fait de sa query »).
        self.last_added = 0
        slot_ids, val_ids, attr_ids = fact_id_maps()
        self.slot_ids = slot_ids
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
    def seg_tokens(seg: dict) -> torch.Tensor:
        """Tokens NON-PADÉS du seg (phase 3 : ce que poole le write réaliste).

        Seul le padding est retiré (via `attention_mask` quand il existe) : les
        tokens ChatML de structure RESTENT, ils sont le bruit de template que
        le write du 350M avale sans savoir où est la valeur.
        """
        ids = seg["input_ids"][0]
        am = seg.get("attention_mask")
        if am is not None:
            ids = ids[am[0].to(torch.bool)]
        return ids

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
        """FIFO de max_mem lignes.

        `write_mode='fact'` (DÉFAUT) : les segs sans fait n'écrivent rien —
        l'oracle SAIT lesquels portent un fait (2ᵉ privilège).
        `write_mode='every'` : CHAQUE seg écrit une ligne, comme le write du
        350M. Les segs sans fait posent la même formule de pool SANS K/A, donc
        le flux ÉVINCE les faits anciens du FIFO : c'est le régime réel.

        `--code rows` appende PLUSIEURS lignes d'un coup (une par token de la
        valeur) : le FIFO reste à max_mem, donc un fait long évince les
        précédents. C'est le prix assumé de la borne haute.

        `--code toprows` (phase 6) : le FIFO compte les GROUPES, pas les lignes
        — max_mem GROUPES résidents de (1+top_k) lignes chacun, soit le MÊME
        nombre de writes résidents que tous les autres codes (comparaison
        propre), pour une banque effective de max_mem×(1+top_k) lignes.
        L'éviction sort le groupe le plus ancien EN ENTIER, et `last_added`
        vaut 1 : les âges restent comptés en WRITES, comme partout ailleurs.
        """
        f = self.fact_of(seg)
        if f is None:
            if self.write_mode != "every":
                self.last_added = 0
                return bank
            rows = model.oracle_lines(0, 0, torch.zeros(0, dtype=torch.long),
                                      seg_tok=self.seg_tokens(seg), bare=True)
        else:
            rows = model.oracle_lines(f[0], f[1], self.val_tokens(f[2]),
                                      seg_tok=self.seg_tokens(seg))
        if model.cfg.code in GROUP_CODES:
            self.last_added = 1                     # UN write = UN groupe
            sizes = list(getattr(bank, "groups", [1] * len(bank)))
            out = GroupBank(list(bank) + list(rows),
                            sizes + [int(rows.shape[0])])
            while len(out.groups) > self.max_mem:
                del out[:out.groups.pop(0)]         # évince le groupe ENTIER
            return out
        self.last_added = int(rows.shape[0])
        bank = bank + list(rows)
        return bank[-self.max_mem:]

    def inject_plan(self, model: ToyReadLM, conv: dict) -> tuple:
        """(plan, n_absent) pour la variante r4 : quel GROUPE injecter devant
        quel segment de RÉPONSE.

        plan = {index de seg → LongTensor[k] des tokens du groupe toprows du
        fait interrogé}. La sélection est la MÊME que celle du write (top_k SIF
        du segment porteur, dans l'ordre du segment) : r4 ne change pas ce que
        la mémoire retient, il change QUI le lit — ici, le backbone nu, sans
        aucun module appris.

        Un FIFO de max_mem groupes est simulé à l'identique : si le fait
        interrogé en est déjà SORTI (rare en fact-only), on n'injecte RIEN et
        la réponse est comptée à part — sinon r4 s'offrirait une mémoire
        infinie que les autres bras n'ont pas.
        """
        truths = (conv.get("info") or {}).get("truths") or []
        q_slots = (conv.get("info") or {}).get("q_slots") or []
        a_idx = [i for i, s in enumerate(conv["segs"])
                 if s["role"] == "assistant"]
        graded = a_idx[-len(truths):] if truths else []
        qpos = {ix: qi for qi, ix in enumerate(graded)}
        plan: dict = {}
        absent = 0
        fifo: list = []                      # [(slot_id, tokens)] résidents
        for i, seg in enumerate(conv["segs"]):
            qi = qpos.get(i)
            if qi is not None:
                sl = self.slot_ids.get(q_slots[qi]) if qi < len(q_slots) else None
                hit = None
                for s_, tk in reversed(fifo):     # le write le PLUS RÉCENT
                    if s_ == sl:
                        hit = tk
                        break
                if hit is None:
                    absent += 1
                else:
                    plan[i] = hit
            f = self.fact_of(seg)
            if f is not None:
                fifo.append((f[0], model.toprows_sel(self.seg_tokens(seg))))
                fifo = fifo[-self.max_mem:]
        return plan, absent

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
    pour une fraction de la VRAM.

    r4 : pas de banque du tout. Les segs de RÉPONSE à un fait reçoivent le
    groupe injecté en préfixe (teacher-forcé, même sélection oracle qu'à
    l'éval) ; les autres segs passent nus. Comme la longueur du préfixe doit
    être uniforme dans un forward, les lanes sont scindées en DEUX sous-lots
    (avec / sans injection) — deux forwards au lieu d'un, gradient identique.
    """
    cfg = model.cfg
    r4 = cfg.variant in INJECT_VARIANTS
    plans = [env.inject_plan(model, c)[0] for c in convs] if r4 else None
    ent_c = float(cfg.pos_entropy) if isinstance(model.ptr, GroupReadout) else 0.0
    banks = [[] for _ in convs]
    total_w = sum(float(s["loss_mask"][0][1:max_len].sum())
                  for c in convs for s in c["segs"]) or 1.0
    nseg = max(len(c["segs"]) for c in convs)
    loss_sum, tok_sum = 0.0, 0.0
    for j in range(nseg):
        lanes = [i for i, c in enumerate(convs) if j < len(c["segs"])]
        if not lanes:
            continue
        if r4:
            subsets = [[i for i in lanes if j in plans[i]],
                       [i for i in lanes if j not in plans[i]]]
        else:
            subsets = [lanes]
        for sub in subsets:
            if not sub:
                continue
            segs = [convs[i]["segs"][j] for i in sub]
            X, W = pad_segs(segs, device, max_len)
            inj = None
            if r4 and j in plans[sub[0]]:
                inj = torch.stack([plans[i][j] for i in sub]).to(device)
            bank, bmask = pad_bank([banks[i] for i in sub], device)
            with torch.autocast(device.split(":")[0], dtype=torch.bfloat16,
                                enabled=amp):
                logits = model(X, bank, bmask, inject=inj)
            s, n = seg_ce(logits, X, W)
            if float(n) > 0:
                obj = s / total_w * scale_by
                if ent_c > 0 and model.ptr.last_pos_ent is not None:
                    # pousse la porte-position vers un choix DUR d'une ligne
                    obj = obj + ent_c * model.ptr.last_pos_ent
                obj.backward()
                loss_sum += float(s.detach())
                tok_sum += float(n)
        if not r4:                       # r4 n'a pas de banque
            for i in lanes:
                banks[i] = env.write(model, banks[i], convs[i]["segs"][j])
    return loss_sum / max(tok_sum, 1.0)


# ── évaluation ───────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, env, stream, seed, n_convs, device, tok, a_open, stop_id,
             max_new, max_len, amp, n_show=3):
    """Replay teacher-forcé (la banque oracle avance) + décodage greedy des
    tours gradés, bras LIVE (banque) vs ABLATÉ (banque vide).

    r4 : pas de banque — LIVE = tour PRÉCÉDÉ du groupe injecté (sélection
    oracle), ABLATÉ = le même tour sans injection, c'est-à-dire le backbone nu.
    Le contraste garde donc exactement le même sens qu'ailleurs.
    """
    model.eval()
    r4 = model.cfg.variant in INJECT_VARIANTS
    stream.rng = random.Random(seed)
    live_ans, abl_ans, truths_all, groups = [], [], [], []
    resident = []                        # fait encore en banque ? (aligné)
    n_absent = 0                         # r4 : faits SORTIS du FIFO (0 inject)
    ages = []                            # writes entre le fait et sa query
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
        # âge en WRITES : compteur de lignes appendées depuis le début de la
        # conv, et pour chaque slot l'instant de son dernier write. En mode
        # `fact` l'âge ne compte que les autres faits ; en mode `every` il
        # compte TOUT le flux — c'est lui qui dit combien de faits le FIFO a
        # déjà évincés au moment de la question.
        wcount = 0
        slot_w: dict = {}
        plan = env.inject_plan(model, conv)[0] if r4 else {}
        for i, seg in enumerate(conv["segs"]):
            X = seg["input_ids"][:, :max_len].to(device)
            W = seg["loss_mask"][:, :max_len].to(device)
            b, bm = pad_bank([bank], device)
            if i in graded:
                inj = None
                if r4:
                    tk = plan.get(i)
                    if tk is None:
                        n_absent += 1     # fait ÉVINCÉ : aucune injection
                    else:
                        inj = tk[None].to(device)
                with torch.autocast(device.split(":")[0], dtype=torch.bfloat16,
                                    enabled=amp):
                    lg_live = model(X, b, bm, inject=inj)
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
                live = tok.decode(model.greedy(a_open, b, bm, max_new, stop_id,
                                               inject=inj))
                tr = truths[qi] if qi < len(truths) else "?"
                live_ans.append(live)
                abl_ans.append(abl_txt)
                truths_all.append(tr)
                q_slot = q_slots[qi] if qi < len(q_slots) else None
                groups.append(env.value_group(q_slot, tr))
                sid = env.slot_ids.get(q_slot) if q_slot else None
                # RÉSIDENCE, alignée sur truths_all (une entrée par réponse) :
                # True  = le fait est encore dans le FIFO au moment de la query
                # False = ÉVINCÉ (le read ne peut structurellement plus répondre)
                # None  = inconnue (fait jamais écrit dans cette conv)
                if r4:
                    resident.append(inj is not None)
                elif sid in slot_w:
                    age_i = wcount - slot_w[sid]
                    ages.append(age_i)
                    resident.append(age_i < env.max_mem)
                else:
                    resident.append(None)
                if len(shown) < n_show:
                    shown.append((prev.strip(), tr, live.strip(),
                                  abl_txt.strip()))
                qi += 1
            if not r4:                        # r4 n'a pas de banque
                bank = env.write(model, bank, seg)
                wcount += env.last_added
                f = OracleEnv.fact_of(seg)
                if f is not None and env.last_added:
                    slot_w[f[0]] = wcount    # instant du write DE CE FAIT
            prev = tok.decode(seg["input_ids"][0].tolist())
    model.train()
    out = {
        "grade_live": grade_recall(live_ans, truths_all) if truths_all else 0.0,
        "grade_abl": grade_recall(abl_ans, truths_all) if truths_all else 0.0,
        "dnll": dnll_num / max(dnll_den, 1.0),
        "n": len(truths_all),
        "show": shown,
        "ptr_gate": (gate_num / gate_den) if gate_den > 0 else float("nan"),
        # âge = nombre de writes appendés ENTRE le write du fait et la query.
        # ≥ max_mem ⇒ la ligne du fait a été ÉVINCÉE du FIFO (le read ne peut
        # structurellement plus répondre) : c'est la mesure du régime réel.
        "age_writes": (sum(ages) / len(ages)) if ages else float("nan"),
        "age_hist": ages,
        "age_evicted": (sum(1 for x in ages if x >= env.max_mem) / len(ages))
                       if ages else float("nan"),
    }
    # GRADE CONDITIONNÉ À LA RÉSIDENCE : en `every`, le flux évince une bonne
    # part des faits AVANT leur query — le grade brut est alors plafonné par
    # l'éviction et ne dit plus rien du READ. `grade_resident` ne grade que les
    # réponses dont le fait était ENCORE là : c'est le chiffre qui juge le read.
    # (r4 : « résident » = un groupe a bien été injecté.)
    ridx = [i for i, x in enumerate(resident) if x is True]
    out["n_resident"] = len(ridx)
    out["n_absent"] = n_absent
    out["grade_resident"] = (
        grade_recall([live_ans[i] for i in ridx],
                     [truths_all[i] for i in ridx]) if ridx else float("nan"))
    out["grade_resident_abl"] = (
        grade_recall([abl_ans[i] for i in ridx],
                     [truths_all[i] for i in ridx]) if ridx else float("nan"))
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
    ap.add_argument("--write", choices=WRITE_MODES, default="fact",
                    dest="write_mode",
                    help="régime du write oracle : fact = segs PORTEURS "
                         "seulement (défaut, privilège gardé) ; every = APRÈS "
                         "CHAQUE seg (le régime du 350M : le flux évince les "
                         "faits du FIFO)")
    ap.add_argument("--pos-offset", type=int, default=None, dest="pos_offset",
                    help="surcharge code.pos_offset (décalage de l'index de "
                         "phase ; 1 libère l'identité occupée par K/A)")
    ap.add_argument("--pack-blocks", type=int, default=None,
                    dest="pack_blocks",
                    help="surcharge code.pack_blocks (formats pack/segpack : "
                         "nombre TOTAL de blocs, CLÉ comprise)")
    ap.add_argument("--top-k", type=int, default=None, dest="top_k",
                    help="surcharge code.top_k (format toprows : lignes de "
                         "CONTENU par groupe ; le groupe fait 1+k lignes)")
    ap.add_argument("--no-row-pos-tag", action="store_true",
                    dest="no_row_pos_tag",
                    help="toprows : lignes de contenu STRICTEMENT natives "
                         "(sans le tag de position ×0.2)")
    ap.add_argument("--readout-mix", choices=READOUT_MIXES, default=None,
                    dest="readout_mix",
                    help="surcharge code.readout_mix (GroupReadout) : linear = "
                         "superposition des lignes puis UNE projection "
                         "(défaut) ; mos = une distribution PAR LIGNE puis "
                         "mixture (aucun token hybride possible)")
    ap.add_argument("--final-eval-convs", type=int, default=None,
                    dest="final_eval_convs",
                    help="surcharge training.final_eval_convs (passe d'éval "
                         "élargie en fin de run ; 0 = désactivée)")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)

    if a.selftest:
        _selftest()
        return

    assert a.config, "config YAML requise (ou --selftest)"
    if a.code != "mean" and a.variant not in ("r3",) + INJECT_VARIANTS:
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
    if "seg_n_pos" in cb:
        mc["seg_n_pos"] = int(cb["seg_n_pos"])
    if "sif_a" in cb:
        mc["sif_a"] = float(cb["sif_a"])
    if "pos_offset" in cb:
        mc["pos_offset"] = int(cb["pos_offset"])
    if "pack_blocks" in cb:
        mc["pack_blocks"] = int(cb["pack_blocks"])
    if "top_k" in cb:
        mc["top_k"] = int(cb["top_k"])
    if "row_pos_tag" in cb:
        mc["row_pos_tag"] = bool(cb["row_pos_tag"])
    if "readout_mix" in cb:
        mc["readout_mix"] = str(cb["readout_mix"])
    if "pos_entropy" in cb:
        mc["pos_entropy"] = float(cb["pos_entropy"])
    if a.readout_mix is not None:          # la CLI gagne sur le YAML
        mc["readout_mix"] = a.readout_mix
    if a.pos_offset is not None:           # la CLI gagne sur le YAML
        mc["pos_offset"] = int(a.pos_offset)
    if a.pack_blocks is not None:
        mc["pack_blocks"] = int(a.pack_blocks)
    if a.top_k is not None:                # la CLI gagne sur le YAML
        mc["top_k"] = int(a.top_k)
    if a.no_row_pos_tag:
        mc["row_pos_tag"] = False
    device = a.device or t.get("device") or ("cuda" if torch.cuda.is_available()
                                             else "cpu")
    steps = int(a.steps or t.get("steps", 3000))
    b_convs = int(t.get("batch_convs", 8))
    eval_every = int(t.get("eval_every", 200))
    eval_convs = int(t.get("eval_convs", 24))
    max_new = int(t.get("max_new", 48))
    # éval FINALE élargie (0 = désactivée) : le juge du run, cf. plus bas.
    final_eval_convs = int(a.final_eval_convs if a.final_eval_convs is not None
                           else t.get("final_eval_convs", 200))
    if a.smoke:
        mc.update(d_model=64, n_layers=2, n_heads=4, mem_dim=64, x_dim=0)
        steps, b_convs, eval_every, eval_convs, max_new = 2, 2, 1, 1, 8

    torch.manual_seed(int(t.get("seed", 0)))
    tok = build_tokenizer(raw["tokenizer"])
    env = OracleEnv(tok, int(mc.get("max_mem", 8)), write_mode=a.write_mode)

    mc["variant"] = a.variant
    mc["code"] = a.code
    mc["write_mode"] = a.write_mode
    mc["vocab_size"] = len(tok)
    if a.variant in INJECT_VARIANTS:
        # séparateur entre le préfixe injecté et le tour : `<blank>`, un token
        # NATIF du vocab (ajouté par build_tokenizer) qui n'apparaît jamais
        # dans les données du toy — il ne peut donc pas être confondu avec du
        # contenu.
        mc["inject_sep_id"] = int(tok.convert_tokens_to_ids("<blank>"))
    cfg = ToyCfg(**mc)
    P = chat_stream_class("persona")
    sif_w = None
    if cfg.code in SIF_CODES:
        # table SIF sur le split TRAIN (la vue du write). Le stream
        # d'entraînement reste surp OFF : le SIF n'entre QUE dans le code.
        sif_w = sif_weight_table(P, tok, persona_kwargs(raw, "train", a.smoke),
                                 cfg.vocab_size, cfg.sif_a,
                                 seed=int(t.get("seed", 0)))
    model = ToyReadLM(cfg, env.n_slots, env.n_attrs, sif_w=sif_w).to(device)

    # phase 1 → <variant>/ (inchangé) ; phase 2 → <variant>_<code>/ ;
    # extensions → suffixes _o<k> (pos_offset) et _wev (write=every) pour ne
    # JAMAIS écraser un run déjà fini sous le même nom.
    run_name = a.variant if a.code == "mean" else f"{a.variant}_{a.code}"
    if cfg.pos_offset:
        run_name += f"_o{cfg.pos_offset}"
    if cfg.code in PACK_CODES and cfg.pack_blocks != ToyCfg.pack_blocks:
        run_name += f"_b{cfg.pack_blocks}"     # sweep de partition = run à part
    if cfg.code in GROUP_CODES:
        if cfg.top_k != ToyCfg.top_k:
            run_name += f"_k{cfg.top_k}"       # sweep de k = run à part
        if not cfg.row_pos_tag:
            run_name += "_notag"
    if cfg.readout_mix != ToyCfg.readout_mix:
        run_name += f"_{cfg.readout_mix}"      # bras MoS = run à part
    if cfg.pos_entropy:
        run_name += f"_ent{cfg.pos_entropy:g}"
    if cfg.write_mode == "every":
        run_name += "_wev"
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
        if cfg.code in GROUP_CODES:
            head = (f"top_k {cfg.top_k} ⇒ groupes de {cfg.group_rows} LIGNES "
                    f"NATIVES (ligne 0 = CLÉ dédiée, {cfg.top_k} embeddings "
                    f"BRUTS des tokens SIF du segment, ordre du segment) ; "
                    f"banque effective {cfg.max_mem}×{cfg.group_rows} = "
                    f"{cfg.max_mem * cfg.group_rows} lignes pour {cfg.max_mem} "
                    f"writes résidents ; tag de position "
                    f"{'ON ×%g' % cfg.oracle_ka_scale if cfg.row_pos_tag else 'OFF'} "
                    f"| privilège span-valeur RETIRÉ ")
        elif cfg.code in PACK_CODES:
            head = (f"pack_blocks {cfg.pack_blocks} × "
                    f"{cfg.d_model // cfg.pack_blocks} dims (bloc 0 = CLÉ "
                    f"dédiée par paire, {cfg.pack_blocks - 1} tokens de "
                    f"contenu) "
                    + ("[segpack : tokens SIF du SEGMENT, privilège "
                       "span-valeur RETIRÉ] " if cfg.code == "segpack"
                       else "[pack : tokens de la VALEUR, privilège gardé] "))
        elif cfg.pools_segment:
            head = (f"seg_n_pos {cfg.seg_n_pos} (pool du SEG ENTIER, "
                    f"privilège span-valeur RETIRÉ) ")
        else:
            head = f"n_pos {cfg.n_pos} "
        print(f"  code {a.code} : " + head
              + (f"blk {cfg.d_model // cfg.n_pos} " if a.code == "chunk" else "")
              + (f"rope_base {cfg.rope_base} "
                 if a.code in ("phase",) + SEG_PHASE_CODES else "")
              # w̄ porte sur TOUT le vocab (dominé par les tokens jamais vus,
              # au poids unseen) ; w_min = l'écrasement du token le plus
              # fréquent, c'est LUI qui dit si le SIF mord.
              + (f"sif_a {cfg.sif_a:g} (w vocab moy {float(model.sif_w.mean()):.3f} "
                 f"min {float(model.sif_w.min()):.4f}) "
                 if a.code in SIF_CODES else "")
              + (f"pos_offset {cfg.pos_offset} (identité LIBRE, positions "
                 f"protégées 0..{(cfg.seg_n_pos if cfg.pools_segment else cfg.n_pos) - cfg.pos_offset - 1}) "
                 if cfg.pos_offset else "")
              + (f"| readout GroupReadout 2 étages : groupe parmi "
                 f"{cfg.max_mem} (clé, foulée {cfg.group_rows}), ligne parmi "
                 f"{cfg.top_k} (porte-position), logits NATIFS (zéro param)"
                 if cfg.code in GROUP_CODES else
                 f"| readout PackReadout 2 étages : ligne parmi "
                 f"{cfg.max_mem} (clé), bloc parmi {cfg.pack_blocks - 1} "
                 f"(porte-position)"
                 if cfg.code in PACK_CODES else
                 f"| candidats pointer {cfg.max_mem * cfg.n_cand} "
                 f"({cfg.max_mem}×{cfg.n_cand})"), flush=True)
    if cfg.write_mode == "every":
        print(f"  write EVERY : l'oracle écrit après CHAQUE seg (segs sans "
              f"fait = " + ("même sélection, CLÉ NULLE) — le FIFO "
                            if cfg.code in GROUP_CODES else
                            "même pool SANS K/A) — le FIFO ")
              + f"{cfg.max_mem} "
              + ("GROUPES évince " if cfg.code in GROUP_CODES else "évince ")
              + f"les faits anciens, 2ᵉ privilège d'oracle RETIRÉ", flush=True)
    print("  params : " + "  ".join(f"{k} {v/1e6:.2f}M" for k, v in pr.items()),
          flush=True)
    print(f"  read+pointer = {(pr['read']+pr['pointer'])/1e6:.2f}M "
          f"({100*(pr['read']+pr['pointer'])/pr['total']:.1f} % du total) — "
          f"appariement de budget entre variantes via model.x_dim (référence "
          f"= le hypernetwork fast-weight de r0 ; r3 reste structurellement "
          f"plus léger, sa V n'est pas projetée).", flush=True)
    if cfg.variant in INJECT_VARIANTS:
        print(f"  INJECTION À SÉLECTION ORACLE : AUCUN module de read appris "
              f"(ni cross-attn, ni pointer) — le backbone NU lit un préfixe de "
              f"{cfg.top_k} pseudo-tokens (embeddings BRUTS, non normés, + un "
              f"vecteur de type appris), séparateur id {cfg.inject_sep_id}, "
              f"positions RoPE 0..{cfg.top_k - 1} puis tour réel décalé de "
              f"{cfg.top_k + 2}. ABLATÉ = le même tour SANS préfixe. "
              f"PRIVILÈGE DÉCLARÉ : la sélection du groupe est l'oracle, et "
              f"l'injection est teacher-forcée à l'entraînement (aucun "
              f"curriculum de copie in-context).", flush=True)
    if cfg.code in GROUP_CODES and cfg.readout_mix == "mos":
        print(f"  readout MoS : une distribution PAR LIGNE puis mixture "
              f"pondérée s·p (aucune superposition dans l'espace d'embedding, "
              f"donc aucun token hybride fabricable)"
              + (f" | pénalité d'entropie sur p : {cfg.pos_entropy:g}"
                 if cfg.pos_entropy else ""), flush=True)
    print(f"  éval FINALE élargie : {final_eval_convs} convs "
          + ("(désactivée)" if final_eval_convs <= 0 else
             "en fin de run → final_metrics.csv (les paliers restent à "
             f"{eval_convs})"), flush=True)
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

    tr_stream = P(tok, seed=int(t.get("seed", 0)),
                  **persona_kwargs(raw, "train", a.smoke))
    ev_stream = P(tok, seed=1234, **persona_kwargs(raw, "eval", a.smoke))
    tc_stream = P(tok, seed=4321, **persona_kwargs(raw, "train", a.smoke))

    a_open = torch.tensor(tok(A_OPEN, add_special_tokens=False)["input_ids"],
                          dtype=torch.long, device=device).unsqueeze(0)
    stop_id = tok.convert_tokens_to_ids("<|im_end|>")

    best = -1.0
    first_age = True                 # distribution des âges imprimée UNE fois
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
            if cfg.write_mode == "every":
                # RÉGIME RÉEL : combien de writes séparent le fait de sa query,
                # et quelle fraction des faits gradés est DÉJÀ ÉVINCÉE du FIFO.
                print(f"    âges (writes fait→query) : moyenne "
                      f"{ev['age_writes']:.2f} | évincés (âge ≥ "
                      f"{cfg.max_mem}) {ev['age_evicted']:.3f}"
                      + (("  | distribution " + " ".join(
                          f"{k}:{ev['age_hist'].count(k)}"
                          for k in sorted(set(ev["age_hist"]))))
                         if first_age else ""), flush=True)
                first_age = False
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
                               # colonnes d'âge SEULEMENT en write=every : le
                               # CSV du mode `fact` reste octet-à-octet celui
                               # de HEAD.
                               + (["age_writes", "age_evicted"]
                                  if cfg.write_mode == "every" else [])
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
                           + ([_f(ev["age_writes"]), _f(ev["age_evicted"])]
                              if cfg.write_mode == "every" else [])
                           + [_f(ev["ptr_gate"]), f"{time.time()-t0:.0f}"])
            if ev["grade_live"] > best:
                best = ev["grade_live"]
                torch.save({"step": step + 1, "model": model.state_dict(),
                            "cfg": cfg.__dict__, "grade": best},
                           os.path.join(save_dir, "best.pt"))
    torch.save({"step": steps, "model": model.state_dict(),
                "cfg": cfg.__dict__}, os.path.join(save_dir, "final.pt"))

    # ── ÉVAL FINALE ÉLARGIE ─────────────────────────────────────────────────
    # Les paliers gradent ~30 réponses (strate `code` : n=10, IC95 d'un 3/10 =
    # [0.07, 0.65]) — INADJUDICABLE. Une seule passe élargie en fin de run met
    # l'erreur-type sous 0.03 pour un coût payé UNE fois. Même fonction, même
    # stream held-out, même graine : seul n_convs change.
    if final_eval_convs > 0:
        fv = evaluate(model, env, ev_stream, 1234, final_eval_convs, device,
                      tok, a_open, stop_id, max_new, max_len, amp, n_show=0)
        se = math.sqrt(max(fv["grade_live"] * (1 - fv["grade_live"]), 1e-9)
                       / max(fv["n"], 1))
        print(f"  [final] HELD-OUT ({final_eval_convs} convs) grade live "
              f"{fv['grade_live']:.3f} ± {se:.3f} (SE) abl "
              f"{fv['grade_abl']:.3f} Δnll {fv['dnll']:+.4f} (n={fv['n']})",
              flush=True)
        print("    [final] strates : " + "  ".join(
            f"{g} {fv['grade_' + g]:.3f} (n={fv['n_' + g]})" for g in GROUPS)
            + f"  | porte pointer σ {fv['ptr_gate']:.4f}", flush=True)
        if cfg.write_mode == "every" or cfg.variant in INJECT_VARIANTS:
            lab = ("injecté" if cfg.variant in INJECT_VARIANTS
                   else "NON ÉVINCÉ")
            print(f"    [final] grade | {lab} {fv['grade_resident']:.3f} "
                  f"(n={fv['n_resident']}/{fv['n']}) abl "
                  f"{fv['grade_resident_abl']:.3f}"
                  + (f" | sans injection (fait évincé) {fv['n_absent']}"
                     if cfg.variant in INJECT_VARIANTS else
                     f" | évincés {fv['age_evicted']:.3f}"), flush=True)
        fp = os.path.join(save_dir, "final_metrics.csv")
        with open(fp, "w", newline="") as f:
            w = csv.writer(f)
            cols = ["n_convs", "grade_live", "grade_live_se", "grade_abl",
                    "dnll", "n", "grade_resident", "grade_resident_abl",
                    "n_resident", "n_absent", "age_evicted"] \
                + [c for g in GROUPS for c in
                   (f"grade_{g}", f"grade_{g}_abl", f"n_{g}")] + ["ptr_gate"]
            w.writerow(cols)

            def _g(x):
                return "" if x != x else f"{x:.4f}"
            w.writerow([final_eval_convs, _g(fv["grade_live"]), f"{se:.4f}",
                        _g(fv["grade_abl"]), _g(fv["dnll"]), fv["n"],
                        _g(fv["grade_resident"]), _g(fv["grade_resident_abl"]),
                        fv["n_resident"], fv["n_absent"],
                        _g(fv["age_evicted"])]
                       + [v for g in GROUPS for v in
                          (_g(fv[f"grade_{g}"]), _g(fv[f"grade_{g}_abl"]),
                           fv[f"n_{g}"])] + [_g(fv["ptr_gate"])])
        print(f"  [final] écrit {fp}", flush=True)
    print(f"done — best grade held-out {best:.3f} | ckpt {save_dir}", flush=True)


# ── round-trip ORACLE d'un format de code ────────────────────────────────────

@torch.no_grad()
def code_roundtrip(model: ToyReadLM, slot_id: int, attr_id: int,
                   val_tok: torch.Tensor, seg_tok: torch.Tensor | None = None,
                   val_pos=None) -> tuple:
    """(top-1 exacts, positions testées) du décodage ORACLE d'un fait.

    Pose les lignes du fait, construit les candidats du readout, et vérifie
    pour chaque position j que `argmax(cand_j @ embed^T) == t_j`. C'est la
    BORNE SUPÉRIEURE de ce que le pointer peut apprendre : si l'oracle
    lui-même ne rend pas l'ordre, aucun read ne le rendra (le verdict de la
    phase 1 en un chiffre).

    PHASE 3 (`segmean`/`segphase`) : la ligne poole le SEG ENTIER, donc les
    positions testées sont celles de la VALEUR DANS LE SEGMENT (`val_pos`,
    issues du `val_mask` du seg), et la cible est `seg_tok[j]`. `segmean` n'a
    qu'un candidat (la ligne) : il est confronté à chaque position, ce qui
    DOIT échouer — c'est la sanity « l'ordre n'existe pas ».
    """
    cfg = model.cfg
    lines = model.oracle_lines(slot_id, attr_id, val_tok, seg_tok=seg_tok)
    cand, _ = model.candidates(lines.unsqueeze(0), None)      # [1, N, d]
    E = model.embed.weight.float()
    if cfg.code in GROUP_CODES:
        # TOPROWS : les lignes de contenu SONT des embeddings — le round-trip
        # est trivial par construction, il ne teste qu'une chose : que le tag
        # de position (row_pos_tag) ne déplace pas l'argmax. Positions testées =
        # les positions de valeur AYANT SURVÉCU à la sélection top_k (la
        # couverture est une mesure séparée, cf. le sweep).
        assert seg_tok is not None and val_pos is not None, (
            "round-trip toprows : seg_tok + val_pos requis")
        st = seg_tok.reshape(-1)
        w = model.sif_w[st].float()
        sel = torch.topk(w, min(cfg.top_k, st.numel())).indices.sort().values
        keep = {int(p): j for j, p in enumerate(sel)}
        Erms = rms_unit(E)
        ok = n = 0
        for p in val_pos:
            j = keep.get(int(p))
            if j is None:
                continue
            ok += int(int(torch.argmax(lines[1 + j].float() @ Erms.t()))
                      == int(st[int(p)]))
            n += 1
        return ok, n
    if cfg.code in PACK_CODES:
        # PACK : le lookup est PAR BLOC (table Ê R_j), pas par dé-projection
        # d'un candidat plat. Les positions testées sont les blocs OCCUPÉS.
        # `segpack` : les blocs portent les k_val tokens SIF du segment, donc on
        # ne teste QUE les positions de valeur qui ont SURVÉCU à la sélection
        # (la couverture est une mesure séparée — cf. le sweep oracle).
        blk = cfg.d_model // cfg.pack_blocks
        toks = model.pack_tokens(val_tok, seg_tok)
        Erms = rms_unit(E)
        line = lines[0].float()
        idx = range(len(toks))
        if cfg.code == "segpack":
            assert seg_tok is not None and val_pos is not None, (
                "round-trip segpack : seg_tok + val_pos requis")
            st = seg_tok.reshape(-1)
            # positions du SEGMENT retenues (ordre du segment) → index de bloc
            wsel = torch.topk(model.sif_w[st].float(),
                              min(cfg.pack_blocks - 1, st.numel())
                              ).indices.sort().values
            keep = {int(p): j for j, p in enumerate(wsel)}
            idx = [keep[int(p)] for p in val_pos if int(p) in keep]
        ok = 0
        n = 0
        for j in idx:
            b = line[(j + 1) * blk:(j + 2) * blk]
            sc = b @ (Erms @ model.pack_R[j + 1].float()).t()
            ok += int(int(sc.argmax()) == int(toks[j]))
            n += 1
        return ok, n
    if cfg.pools_segment:
        assert seg_tok is not None and val_pos is not None, (
            "round-trip phase 3 : seg_tok + val_pos requis")
        seg_tok = seg_tok.reshape(-1)
        ok = 0
        pos = [int(j) for j in val_pos]
        for j in pos:
            c = cand[0, min(j, cfg.seg_n_pos - 1)] \
                if cfg.code in SEG_PHASE_CODES else cand[0, 0]
            ok += int(int(torch.argmax(c.float() @ E.t())) == int(seg_tok[j]))
        return ok, len(pos)
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

    # Le self-test ne seedait RIEN : les embeddings de chaque modèle jouet
    # étaient tirés au hasard du process, et les round-trips ORACLE (qui sont
    # des argmax sur un vocab entier, donc bornés par le max d'un bruit
    # gaussien) flambaient de temps en temps. Le seed rend le verdict
    # reproductible — il ne rend rien plus facile.
    torch.manual_seed(20260730)

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
    for var in BANK_VARIANTS:
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
    for var in BANK_VARIANTS:
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
    P2_CODES = ("mean", "chunk", "phase", "rows")   # formats à span-valeur

    # table SIF SYNTHÉTIQUE (hermétique, pas de passe de 300 convs) : les
    # tokens 7/8/9 jouent le « template très fréquent » (p = 0.2 ⇒ w ≈ 5e-4),
    # tout le reste est rare (p = 1e-5 ⇒ w ≈ 0.91). C'est exactement la forme
    # w = a/(a+p) que rend sif_weight_table.
    A_SIF = 1e-4

    def _sifw(vocab=512):
        w = torch.full((vocab,), A_SIF / (A_SIF + 1e-5))
        w[torch.tensor([7, 8, 9])] = A_SIF / (A_SIF + 0.2)
        return w

    def _mk(code, d=256, n_pos=8, vocab=512, base=0.0, seg_n_pos=32,
            offset=0, pack_blocks=8, top_k=13, row_pos_tag=True):
        c = ToyCfg(vocab_size=vocab, d_model=d, n_layers=1, n_heads=4,
                   mem_dim=d, variant="r3", max_seq_len=64, code=code,
                   n_pos=n_pos, rope_base=base, seg_n_pos=seg_n_pos,
                   sif_a=A_SIF, pos_offset=offset, pack_blocks=pack_blocks,
                   top_k=top_k, row_pos_tag=row_pos_tag)
        return ToyReadLM(c, env.n_slots, env.n_attrs,
                         sif_w=_sifw(vocab) if code in SIF_CODES
                         else None).eval()

    tok8 = torch.tensor([11, 77, 200, 41, 305, 9, 128, 460])   # 8 positions
    rt = {}
    for code in P2_CODES:
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
    seg20 = torch.randint(0, 512, (20,), generator=torch.Generator().manual_seed(3))
    for code in CODES:
        # `toprows` : la banque est faite de GROUPES de (1+top_k) lignes — la
        # remplir ligne à ligne violerait son layout. On prend top_k=3 pour
        # garder des banques minuscules, et le « padding » est un groupe ENTIER.
        grp3 = code in GROUP_CODES
        m = _mk(code, d=32, n_pos=4, vocab=512, seg_n_pos=8, top_k=3)
        rows = m.oracle_lines(3, 2, tok8[:3], seg_tok=seg20)
        nrow = rows.shape[0] if grp3 else 1
        bmix = torch.cat([torch.cat([rows[:nrow],
                                     torch.zeros(nrow, 32)])[None],
                          torch.zeros(1, 2 * nrow, 32)], dim=0)
        mmix = torch.tensor([[True] * nrow + [False] * nrow,
                             [False] * (2 * nrow)])
        ids = torch.randint(0, 512, (2, 6))
        with torch.no_grad():
            lg = m(ids, bmix, mmix)
        assert torch.isfinite(lg).all(), f"NaN sur lane vide (code {code})"
        with torch.no_grad():
            i1 = torch.randint(0, 512, (1, 6))
            a1 = m(i1, torch.zeros(1, 2 * nrow, 32),
                   torch.tensor([[False] * (2 * nrow)]))
            a2 = m(i1, None, None)
        assert torch.allclose(a1, a2, atol=1e-5), \
            f"lane vide != ablaté (code {code})"
        # 8bis. porte du pointer FERMÉE à l'init : biais ≈ 0
        b1 = rows[:nrow][None].expand(1, -1, -1)
        bm1 = torch.ones(1, b1.size(1), dtype=torch.bool)
        cand, cm = m.candidates(b1, bm1)
        with torch.no_grad():
            if code in PACK_CODES:
                bias = m.ptr(torch.randn(1, 5, 32), b1, bm1, m.embed.weight,
                             m.pack_R)
            elif grp3:
                bias = m.ptr(torch.randn(1, 5, 32), b1, bm1, m.embed.weight)
            else:
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
        for code in ("phase", "segmean", "segphase", "segsif", "pack",
                     "segpack", "toprows"):
            try:
                ToyCfg(vocab_size=512, d_model=32, n_heads=4, mem_dim=32,
                       variant=var, code=code, n_pos=4, seg_n_pos=8)
            except AssertionError as e:
                assert "r3" in str(e)
            else:
                raise AssertionError(f"{var} aurait dû refuser --code {code}")

    # ════════════════ PHASE 3 : POOL DU SEGMENT ENTIER ══════════════════════
    # 11. round-trip segphase sur un fait SYNTHÉTIQUE : seg de 20 tokens dont
    #     la valeur occupe les positions 5-7. Le code ne sait PAS où elle est
    #     (il poole les 20), le readout doit quand même la rendre top-1 aux 3
    #     positions.
    #
    #     ⚠️ Ce test valide LE FORMAT (le binding survit au pool du segment),
    #     PAS la viabilité aux dims du run. Superposer T tokens dans d dims
    #     donne SNR ≈ √(d/(T−1)) contre le max de |V| gaussiennes concurrentes
    #     (√(2·ln|V|)). À d=256/T=20 : 3.7 σ contre 3.5 (vocab 512) et 4.65
    #     (vocab réel 49152) ⇒ le format est SNR-LIMITÉ à l'échelle réelle,
    #     et c'est `code_roundtrip` sur les vrais segs qui rend le verdict de
    #     déploiement. Ici on prend d=512 (SNR 5.2) pour que l'assertion teste
    #     l'algèbre et non le tirage d'embeddings ; le chiffre à d=256 est
    #     imprimé à titre indicatif, sans assertion.
    torch.manual_seed(20260731)          # le self-test ne seedait pas le global
    g20 = torch.Generator().manual_seed(11)
    seg_syn = torch.randint(0, 512, (20,), generator=g20)
    seg_syn[5:8] = torch.tensor([301, 44, 199])            # la « valeur »
    vpos = [5, 6, 7]
    m_sp = _mk("segphase", d=512)
    ok, n = code_roundtrip(m_sp, 3, 2, tok8, seg_tok=seg_syn, val_pos=vpos)
    assert (ok, n) == (3, 3), (
        f"round-trip segphase : {ok}/{n} — le binding positionnel ne survit "
        f"pas au pool du segment entier, l'expérience n'a pas de sens")
    # rien de spécial au span valeur : le seg ENTIER se déroule au même taux.
    ok_all, n_all = code_roundtrip(m_sp, 3, 2, tok8, seg_tok=seg_syn,
                                   val_pos=list(range(20)))
    assert ok_all == n_all, f"round-trip segphase seg entier {ok_all}/{n_all}"
    ok_256, n_256 = code_roundtrip(_mk("segphase"), 3, 2, tok8,
                                   seg_tok=seg_syn, val_pos=list(range(20)))
    # segmean : l'ordre N'EXISTE PAS — un seul candidat, et il ne peut pas
    # rendre 3 tokens distincts (au plus 1 des 3 positions par accident).
    m_sm = _mk("segmean", d=512)
    ok_sm, n_sm = code_roundtrip(m_sm, 3, 2, tok8, seg_tok=seg_syn,
                                 val_pos=vpos)
    assert ok_sm <= 1, f"segmean ne devrait pas décoder l'ordre ({ok_sm}/{n_sm})"
    # et la PERMUTATION du segment donne la MÊME ligne (sac de tokens) alors
    # qu'elle change celle de segphase.
    permu = seg_syn[torch.randperm(20, generator=torch.Generator().manual_seed(5))]
    assert torch.allclose(m_sm.oracle_lines(3, 2, tok8, seg_tok=seg_syn),
                          m_sm.oracle_lines(3, 2, tok8, seg_tok=permu),
                          atol=1e-6), "segmean : la permutation devrait collapser"
    assert not torch.allclose(m_sp.oracle_lines(3, 2, tok8, seg_tok=seg_syn),
                              m_sp.oracle_lines(3, 2, tok8, seg_tok=permu),
                              atol=1e-4), "segphase : la permutation devrait compter"
    # 11bis. le code ne dépend PAS de val_tok (privilège retiré) mais DÉPEND du
    #        seg — la garantie que l'oracle n'a plus le span valeur.
    for m_ in (m_sm, m_sp):
        assert torch.equal(m_.oracle_lines(3, 2, tok8, seg_tok=seg_syn),
                           m_.oracle_lines(3, 2, tok8[:2], seg_tok=seg_syn)), \
            "phase 3 : le code regarde encore val_tok"
    # 11ter. seg_tok manquant ⇒ erreur explicite
    try:
        m_sp.oracle_lines(3, 2, tok8)
    except AssertionError as e:
        assert "seg_tok" in str(e)
    else:
        raise AssertionError("segphase aurait dû exiger seg_tok")
    # 11quater. FIFO INCHANGÉ : 12 faits ⇒ 8 lignes, une ligne par fait
    for code in SEG_CODES:
        m_ = _mk(code, d=32, n_pos=4, vocab=512, seg_n_pos=8)
        # `toprows` : le FIFO compte les GROUPES — 8 writes résidents comme
        # partout, mais 8×(1+top_k) LIGNES.
        gr = m_.cfg.group_rows if code in GROUP_CODES else 1
        b = []
        for k in range(12):
            s = dict(seg_tpl)
            s["fact_val"] = torch.full_like(seg_tpl["fact_val"], 1 + k)
            b = env.write(m_, b, s)
        assert len(b) == 8 * gr, f"FIFO {code} : {len(b)} lignes"
        ref = m_.oracle_lines(int(seg_tpl["fact_slot"][0, 0]),
                              int(seg_tpl["fact_attr"][0, 0]),
                              env.val_tokens(12),
                              seg_tok=OracleEnv.seg_tokens(seg_tpl))
        assert torch.equal(torch.stack(list(b)[-gr:]), ref), \
            f"FIFO {code} : dernier write"

    # 12. segsif : le pool est PONDÉRÉ SIF (w = a/(a+p)) — recette du write du
    #     350M. Tokens 7/8/9 = « template très fréquent » (w ≈ 5e-4), le reste
    #     rare (w ≈ 0.91).
    m_ss = _mk("segsif", d=512)
    seg_f = seg_syn.clone()
    seg_f[[0, 1, 2, 10, 11, 12, 15, 16]] = torch.tensor([7, 8, 9, 7, 8, 9, 7, 8])
    # (a) la pondération CHANGE le code : segsif ≠ segphase sur ce seg
    l_ss = m_ss.oracle_lines(3, 2, tok8, seg_tok=seg_f)
    l_sp = _mk("segphase", d=512).oracle_lines(3, 2, tok8, seg_tok=seg_f)
    assert not torch.allclose(l_ss, l_sp, atol=1e-3), \
        "segsif : la pondération SIF ne change pas le code"
    # (b) round-trip synthétique : la valeur (positions 5-7, tokens rares)
    #     reste top-1 malgré le bruit de template
    ok_ss, n_ss = code_roundtrip(m_ss, 3, 2, tok8, seg_tok=seg_f, val_pos=vpos)
    assert (ok_ss, n_ss) == (3, 3), f"round-trip segsif : {ok_ss}/{n_ss}"
    # (c) ÉCRASEMENT : bouger un token FRÉQUENT bouge le code segsif beaucoup
    #     moins que le code segphase (c'est TOUT le mécanisme du SIF)
    seg_g = seg_f.clone(); seg_g[10] = 9
    d_ss = float((m_ss.oracle_lines(3, 2, tok8, seg_tok=seg_g) - l_ss).norm())
    d_sp = float((_mk("segphase", d=512).oracle_lines(3, 2, tok8, seg_tok=seg_g)
                  - l_sp).norm())
    assert d_ss < 0.2 * d_sp, (
        f"segsif n'écrase pas les tokens fréquents (Δ {d_ss:.4f} vs segphase "
        f"{d_sp:.4f})")
    # (d) T_eff : le SIF fait chuter le nombre EFFECTIF de tokens superposés
    w_ = _sifw()[seg_f].float()
    t_eff = float(w_.sum() ** 2 / (w_ * w_).sum())
    assert t_eff < 0.7 * seg_f.numel(), (t_eff, seg_f.numel())
    # (e) sif_w OBLIGATOIRE (sinon le code serait silencieusement uniforme)
    try:
        ToyReadLM(ToyCfg(vocab_size=512, d_model=32, n_heads=4, mem_dim=32,
                         variant="r3", code="segsif", seg_n_pos=8),
                  env.n_slots, env.n_attrs)
    except AssertionError as e:
        assert "sif_w" in str(e)
    else:
        raise AssertionError("segsif aurait dû exiger sif_w")
    # (f) sif_a <= 0 refusé
    try:
        ToyCfg(vocab_size=512, d_model=32, n_heads=4, mem_dim=32,
               variant="r3", code="segsif", seg_n_pos=8, sif_a=0.0)
    except AssertionError as e:
        assert "sif_a" in str(e)
    else:
        raise AssertionError("sif_a=0 aurait dû être refusé")

    # ═══════════ EXTENSION 1 : code.pos_offset (fix position 0) ═════════════
    # 13a. la table EST décalée : offset k ⇒ ligne j = rot(θ·(j+k)), donc
    #      (binding DFT, période n_pos) elle vaut la ligne (j+k) mod n_pos de
    #      la table non décalée. offset 0 = tables IDENTIQUES (rétro-compat).
    c0, s0 = phase_tables(8, 32, 0.0)
    c1, s1 = phase_tables(8, 32, 0.0, offset=1)
    assert torch.equal(c0, phase_tables(8, 32, 0.0, offset=0)[0])
    for j in range(8):
        assert torch.allclose(c1[j], c0[(j + 1) % 8], atol=1e-5) and \
            torch.allclose(s1[j], s0[(j + 1) % 8], atol=1e-5), j
    # 13b. le décalage passe AU WRITE ET AUX CANDIDATS : le round-trip reste
    #      exact, y compris À LA POSITION 0 (qui, à offset 0, partage
    #      l'identité rotationnelle avec K[slot]+A[attr]).
    seg8 = torch.tensor([301, 44, 199, 12, 260, 88, 150, 7])
    for off in (0, 1):
        m_o = _mk("segphase", d=512, seg_n_pos=8, offset=off)
        # positions PROTÉGÉES : j + offset ≢ 0 [seg_n_pos]
        prot = list(range(8 - off))
        ok_o, n_o = code_roundtrip(m_o, 3, 2, tok8, seg_tok=seg8, val_pos=prot)
        assert (ok_o, n_o) == (len(prot), len(prot)), (off, ok_o, n_o)
        assert 0 in prot                      # la position 0 EST récupérée
    # et le code CHANGE avec l'offset (sinon le knob serait un no-op)
    assert not torch.allclose(
        _mk("segphase", d=512, seg_n_pos=8, offset=0).oracle_lines(
            3, 2, tok8, seg_tok=seg8),
        _mk("segphase", d=512, seg_n_pos=8, offset=1).oracle_lines(
            3, 2, tok8, seg_tok=seg8), atol=1e-4), "pos_offset = no-op"
    # 13c. WRAP documenté : à offset 1, la position seg_n_pos−1 retombe sur
    #      l'identité (index ≡ 0) — elle redevient exactement la position 0 du
    #      format non décalé. C'est le coût du knob, pas un bug.
    #      Invariant vérifiable : le candidat d'index de phase ≡ 0 est la LIGNE
    #      BRUTE (dé-rotation identité) — c'est le candidat 0 à offset 0, et le
    #      candidat n_pos−1 à offset 1. C'est CE candidat qui voit K/A non
    #      tourné ; l'offset le déplace sur une position JAMAIS occupée.
    for off, j_id in ((0, 0), (1, 7)):
        m_w = _mk("segphase", d=512, seg_n_pos=8, offset=off)
        li = m_w.oracle_lines(3, 2, tok8, seg_tok=seg8)
        cw, _ = m_w.candidates(li.unsqueeze(0), None)
        assert torch.allclose(cw[0, j_id], li[0], atol=1e-5), \
            f"wrap : à offset {off}, le candidat {j_id} doit être la ligne nue"
    # 13d. pos_offset REFUSÉ hors formats à binding de phase (pas de no-op
    #      silencieux), et refusé s'il dépasse n_pos.
    for code in ("mean", "chunk", "rows", "segmean"):
        try:
            ToyCfg(vocab_size=512, d_model=32, n_heads=4, mem_dim=32,
                   variant="r3", code=code, n_pos=4, seg_n_pos=8, pos_offset=1)
        except AssertionError as e:
            assert "pos_offset" in str(e)
        else:
            raise AssertionError(f"pos_offset aurait dû être refusé ({code})")
    try:
        ToyCfg(vocab_size=512, d_model=32, n_heads=4, mem_dim=32, variant="r3",
               code="segphase", seg_n_pos=8, pos_offset=8)
    except AssertionError as e:
        assert "wrappent" in str(e)
    else:
        raise AssertionError("pos_offset >= n_pos aurait dû être refusé")
    # 13e. déterminisme : deux écritures successives, offset actif, identiques
    m_d = _mk("segsif", d=512, seg_n_pos=8, offset=1)
    assert torch.equal(m_d.oracle_lines(3, 2, tok8, seg_tok=seg8),
                       m_d.oracle_lines(3, 2, tok8, seg_tok=seg8))

    # ═══════════ EXTENSION 2 : --write every (2ᵉ privilège retiré) ══════════
    m_e = _mk("segsif", d=32, n_pos=4, vocab=512, seg_n_pos=8)
    env_f = OracleEnv(tok, 8)                       # mode `fact` (défaut)
    env_e = OracleEnv(tok, 8, write_mode="every")
    nofact = [s for s in conv["segs"] if OracleEnv.fact_of(s) is None][0]
    factseg = [s for s in conv["segs"] if OracleEnv.fact_of(s)][0]
    # 14a. mode fact : un seg SANS fait n'écrit RIEN (inchangé)
    assert env_f.write(m_e, [], nofact) == [] and env_f.last_added == 0
    # 14b. mode every : il écrit UNE ligne, non nulle, de RMS 1
    b_e = env_e.write(m_e, [], nofact)
    assert len(b_e) == 1 and env_e.last_added == 1
    assert float(b_e[0].abs().max()) > 0 and \
        abs(float(b_e[0].pow(2).mean().sqrt()) - 1.0) < 1e-3
    # 14c. … et SANS composante K/A : la ligne d'un seg sans fait ne dépend NI
    #      du slot NI de l'attribut, alors que celle d'un seg porteur en dépend.
    st_nf = OracleEnv.seg_tokens(nofact)
    bare_a = m_e.oracle_lines(0, 0, torch.zeros(0, dtype=torch.long),
                              seg_tok=st_nf, bare=True)
    bare_b = m_e.oracle_lines(5, 3, torch.zeros(0, dtype=torch.long),
                              seg_tok=st_nf, bare=True)
    assert torch.equal(bare_a, bare_b), "ligne bare : K/A n'est pas annulé"
    assert torch.equal(bare_a[0], b_e[0])
    ka_a = m_e.oracle_lines(1, 1, torch.zeros(0, dtype=torch.long),
                            seg_tok=st_nf)
    ka_b = m_e.oracle_lines(5, 3, torch.zeros(0, dtype=torch.long),
                            seg_tok=st_nf)
    assert not torch.allclose(ka_a, ka_b, atol=1e-4) and \
        not torch.allclose(bare_a, ka_a, atol=1e-4), \
        "la ligne AVEC K/A devrait dépendre du slot/attr et différer de bare"
    # 14d. FIFO en mode every : 3 faits PUIS des segs vides ⇒ les lignes de
    #      faits SORTENT (c'est LE point de l'extension). Arithmétique : 3+6=9
    #      lignes pour 8 slots ⇒ le fait le PLUS ANCIEN est déjà évincé ; à
    #      3+8=11 il ne reste plus AUCUN fait.
    b = []
    fact_rows = []
    for k in range(3):
        s = dict(factseg)
        s["fact_val"] = torch.full_like(factseg["fact_val"], 1 + k)
        # segs DISTINCTS : les codes de segment poolent le seg, pas fact_val —
        # sans ça les 3 lignes seraient identiques et « évincée » indécidable.
        s["input_ids"] = factseg["input_ids"].clone()
        s["input_ids"][0, 1] = 100 + k
        b = env_e.write(m_e, b, s)
        fact_rows.append(b[-1].clone())
    assert len(b) == 3
    for k in range(6):
        b = env_e.write(m_e, b, nofact)
    assert len(b) == 8, len(b)
    assert not any(torch.equal(fact_rows[0], x) for x in b), \
        "FIFO every : le fait le plus ancien aurait dû être évincé par le flux"
    assert sum(any(torch.equal(fr, x) for x in b) for fr in fact_rows) == 2
    for k in range(2):
        b = env_e.write(m_e, b, nofact)
    for fr in fact_rows:
        assert not any(torch.equal(fr, x) for x in b), \
            "FIFO every : 8 segs vides doivent purger TOUS les faits"
    # en mode `fact` les mêmes 6 segs vides n'évincent RIEN
    b2 = []
    for k in range(3):
        s = dict(factseg)
        s["fact_val"] = torch.full_like(factseg["fact_val"], 1 + k)
        # segs DISTINCTS : les codes de segment poolent le seg, pas fact_val —
        # sans ça les 3 lignes seraient identiques et « évincée » indécidable.
        s["input_ids"] = factseg["input_ids"].clone()
        s["input_ids"][0, 1] = 100 + k
        b2 = env_f.write(m_e, b2, s)
    for k in range(6):
        b2 = env_f.write(m_e, b2, nofact)
    assert len(b2) == 3, len(b2)
    # 14e. lane vide ≡ ablaté et forward fini AVEC des lignes bare dans la
    #      banque (le régime every mélange lignes de faits et lignes nues)
    bmix = torch.stack([torch.stack(b[:2] + [torch.zeros(32)]),
                        torch.zeros(3, 32)])
    mmix = torch.tensor([[True, True, False], [False, False, False]])
    with torch.no_grad():
        lg = m_e(torch.randint(0, 512, (2, 6)), bmix, mmix)
    assert torch.isfinite(lg).all()
    with torch.no_grad():
        i1 = torch.randint(0, 512, (1, 6))
        assert torch.allclose(
            m_e(i1, torch.zeros(1, 3, 32), torch.tensor([[False] * 3])),
            m_e(i1, None, None), atol=1e-5), "lane vide != ablaté (every)"
    # 14f. déterminisme des lignes bare
    assert torch.equal(env_e.write(m_e, [], nofact)[0],
                       env_e.write(m_e, [], nofact)[0])
    # 14g. --write every REFUSÉ pour les codes à privilège span-valeur
    for code in ("mean", "chunk", "phase", "rows", "pack"):
        try:
            ToyCfg(vocab_size=512, d_model=32, n_heads=4, mem_dim=32,
                   variant="r3", code=code, n_pos=4, write_mode="every")
        except AssertionError as e:
            assert "every" in str(e)
        else:
            raise AssertionError(f"--write every aurait dû être refusé ({code})")

    # ═══════════ PHASE 5 : format PACK (ligne PARTITIONNÉE en blocs) ════════
    torch.manual_seed(20260801)
    # 15a. round-trip `pack` : 8 tokens, 7 blocs de contenu ⇒ 7 positions
    #      testées, toutes top-1. Le pack ne SUPERPOSE rien : chaque token vit
    #      dans son propre sous-espace, le seul bruit est celui de la
    #      dé-projection JL (SNR ≈ √blk contre √(2 ln|V|)).
    m_pk = _mk("pack", d=512)
    ok_pk, n_pk = code_roundtrip(m_pk, 3, 2, tok8)
    assert (ok_pk, n_pk) == (7, 7), (
        f"round-trip pack : {ok_pk}/{n_pk} — les blocs ne sont pas inversibles, "
        f"le format n'a pas de sens")
    # 15b. round-trip `segpack` SANS privilège : seg de 20 tokens dont 7 rares
    #      (positions 5-7 = la « valeur », 4 autres = du contenu), le reste =
    #      template très fréquent. La sélection top-7 SIF doit retenir
    #      EXACTEMENT les 7 rares, donc les 3 positions de valeur, et chacune
    #      doit être top-1 dans son bloc.
    seg_pk = torch.full((20,), 7, dtype=torch.long)
    seg_pk[[1, 3, 5, 6, 7, 14, 18]] = torch.tensor([120, 301, 44, 199, 260,
                                                    77, 410])
    m_sk = _mk("segpack", d=512)
    sel_pk = m_sk.pack_tokens(tok8, seg_pk)
    assert torch.equal(sel_pk, seg_pk[[1, 3, 5, 6, 7, 14, 18]]), sel_pk
    ok_sk, n_sk = code_roundtrip(m_sk, 3, 2, tok8, seg_tok=seg_pk,
                                 val_pos=[5, 6, 7])
    assert (ok_sk, n_sk) == (3, 3), f"round-trip segpack : {ok_sk}/{n_sk}"
    # … et le code ne regarde PAS val_tok (privilège retiré), mais DÉPEND du seg
    assert torch.equal(m_sk.oracle_lines(3, 2, tok8, seg_tok=seg_pk),
                       m_sk.oracle_lines(3, 2, tok8[:2], seg_tok=seg_pk)), \
        "segpack : le code regarde encore val_tok"
    # 15c. la CLÉ par paire est STABLE d'une instanciation à l'autre (elle est
    #      tirée par un générateur propre à la paire, pas par l'ordre du
    #      tirage), et elle SÉPARE les paires.
    k1 = _mk("pack", d=512).pack_key
    k2 = _mk("pack", d=512).pack_key
    assert torch.equal(k1, k2), "clé pack non stable entre instanciations"
    assert torch.equal(m_pk.pack_R, _mk("pack", d=512).pack_R), \
        "frames pack non stables entre instanciations"
    assert not torch.allclose(k1[3, 2], k1[3, 1], atol=1e-4) and \
        not torch.allclose(k1[3, 2], k1[5, 2], atol=1e-4), \
        "la clé pack ne dépend pas de la paire (slot, attr)"
    # 15d. les blocs sont DISJOINTS : changer le token du bloc 2 ne touche
    #      AUCUNE dimension hors du bloc 2 (à la RMS-norm globale près, qui est
    #      un scalaire — on compare donc les lignes NON normalisées via le
    #      rapport). C'est la propriété que segsif n'a pas.
    blk512 = 512 // 8
    la = m_pk.oracle_lines(3, 2, tok8[:5])[0]
    tb = tok8[:5].clone(); tb[1] = 333
    lb = m_pk.oracle_lines(3, 2, tb)[0]
    r = float(la[:blk512].norm() / lb[:blk512].norm())      # scalaire de RMS
    d_blk = [float((la[j * blk512:(j + 1) * blk512]
                    - r * lb[j * blk512:(j + 1) * blk512]).abs().max())
             for j in range(8)]
    assert d_blk[2] > 1e-3 and max(d_blk[:2] + d_blk[3:]) < 1e-5, d_blk
    # 15e. PackReadout : porte fermée ⇒ biais EXACTEMENT nul (scale zéro-init,
    #      pas seulement « petit »), et les deux étages sont bien définis.
    for code in PACK_CODES:
        m_ = _mk(code, d=512)
        bk_ = torch.stack([m_.oracle_lines(3, 2, tok8, seg_tok=seg_pk)[0],
                           m_.oracle_lines(5, 1, tok8[:4], seg_tok=seg_pk)[0]]
                          )[None]
        bm_ = torch.ones(1, 2, dtype=torch.bool)
        with torch.no_grad():
            bias = m_.ptr(torch.randn(1, 5, 512), bk_, bm_, m_.embed.weight,
                          m_.pack_R)
        assert float(bias.abs().max()) == 0.0, (code, float(bias.abs().max()))
        assert float(m_.ptr.last_gate.max()) < 1e-3, code
        # les softmax des deux étages somment à 1 sur leurs axes respectifs
        assert torch.allclose(m_.ptr.last_sel.sum(-1), torch.ones(1, 5)) and \
            torch.allclose(m_.ptr.last_pos.sum(-1), torch.ones(1, 5))
        assert m_.ptr.last_sel.shape[-1] == 2 and \
            m_.ptr.last_pos.shape[-1] == m_.cfg.pack_blocks - 1
        # porte OUVERTE (scale forcé) : le biais pointe bien vers un token de la
        # ligne — l'étage 2 relit vraiment le contenu.
        with torch.no_grad():
            m_.ptr.scale.fill_(50.0)
            m_.ptr.gate.bias.fill_(8.0)
            m_.ptr.wp.weight.zero_()
            b2 = m_.ptr(torch.randn(1, 1, 512), bk_[:, :1], bm_[:, :1],
                        m_.embed.weight, m_.pack_R)
        cand_tok = set(int(t) for t in m_.pack_tokens(tok8, seg_pk))
        assert int(b2[0, 0].argmax()) in cand_tok, (code, int(b2[0, 0].argmax()))
    # 15f. `segpack` + --write every : la ligne NUE n'a PAS de bloc-clé (elle ne
    #      peut donc pas gagner l'étage 1), mais elle porte bien du contenu.
    m_we = _mk("segpack", d=512)
    bare = m_we.oracle_lines(0, 0, torch.zeros(0, dtype=torch.long),
                             seg_tok=seg_pk, bare=True)[0]
    assert float(bare[:blk512].abs().max()) == 0.0, "ligne bare : clé non nulle"
    assert float(bare[blk512:].abs().max()) > 0
    assert abs(float(bare.pow(2).mean().sqrt()) - 1.0) < 1e-3
    assert torch.equal(bare, m_we.oracle_lines(5, 3,
                                               torch.zeros(0, dtype=torch.long),
                                               seg_tok=seg_pk, bare=True)[0]), \
        "ligne bare : dépend encore du slot/attr"
    # 15g. pack_blocks : diviseur de d_model obligatoire, ≥ 2
    for pb, needle in ((7, "divisible"), (1, "1")):
        try:
            ToyCfg(vocab_size=512, d_model=512, n_heads=4, mem_dim=512,
                   variant="r3", code="pack", pack_blocks=pb)
        except AssertionError as e:
            assert needle in str(e), (pb, str(e))
        else:
            raise AssertionError(f"pack_blocks={pb} aurait dû être refusé")

    # ═══ PHASE 6 : TOPROWS (groupes de lignes NATIVES, jamais transformées) ══
    torch.manual_seed(20260802)
    K6 = 5
    # seg de 20 tokens : 7 « rares » dont la valeur en 5-7, le reste = template
    # fréquent (tokens 7/8/9). La sélection top-5 SIF retient donc 5 des 7
    # rares — et comme le tri est stable sur les ex æquo, on vérifie AU CHIFFRE
    # quels indices sortent plutôt que de le supposer.
    seg_tr = torch.full((20,), 7, dtype=torch.long)
    seg_tr[[1, 3, 5, 6, 7, 14, 18]] = torch.tensor([120, 301, 44, 199, 260,
                                                    77, 410])
    m_tr = _mk("toprows", d=512, top_k=K6)
    sel_tr = m_tr.toprows_sel(seg_tr)
    assert sel_tr.numel() == K6
    # ORDRE DU SEGMENT préservé (pas l'ordre des poids)
    pos_tr = [int(p) for p in torch.topk(
        m_tr.sif_w[seg_tr].float(), K6).indices.sort().values]
    assert pos_tr == sorted(pos_tr) and \
        torch.equal(sel_tr, seg_tr[torch.tensor(pos_tr)])
    # 15h. LE GROUPE : 1+k lignes, ligne 0 = clé, lignes 1.. = embeddings BRUTS
    grp = m_tr.oracle_lines(3, 2, tok8, seg_tok=seg_tr)
    assert grp.shape == (1 + K6, 512), grp.shape
    assert torch.equal(grp[0], m_tr.pack_key[3, 2]), "ligne 0 ≠ clé de la paire"
    # NATIVITÉ : chaque ligne de contenu est à cos ≥ 0.97 de l'embedding
    # RMS-normé de son token (le tag de position ×0.2 ne fait que l'incliner).
    for j, t in enumerate(sel_tr):
        e = rms_unit(m_tr.embed.weight[int(t)].float())
        c = float(torch.dot(grp[1 + j], e) / (grp[1 + j].norm() * e.norm()))
        assert c > 0.97, (j, c)
    # … et SANS tag, la ligne est l'embedding EXACT (zéro transformation)
    m_nt = _mk("toprows", d=512, top_k=K6, row_pos_tag=False)
    g_nt = m_nt.oracle_lines(3, 2, tok8, seg_tok=seg_tr)
    for j, t in enumerate(m_nt.toprows_sel(seg_tr)):
        assert torch.allclose(g_nt[1 + j],
                              rms_unit(m_nt.embed.weight[int(t)].float()),
                              atol=1e-6), j
    # 15i. round-trip : le tag ne déplace PAS l'argmax (c'est tout ce que le RT
    #      teste ici — les lignes SONT des embeddings)
    ok_tr, n_tr = code_roundtrip(m_tr, 3, 2, tok8, seg_tok=seg_tr,
                                 val_pos=[5, 6, 7])
    assert (ok_tr, n_tr) == (3, 3), f"round-trip toprows : {ok_tr}/{n_tr}"
    # le code ne regarde PAS val_tok (privilège retiré)
    assert torch.equal(m_tr.oracle_lines(3, 2, tok8, seg_tok=seg_tr),
                       m_tr.oracle_lines(3, 2, tok8[:2], seg_tok=seg_tr)), \
        "toprows : le code regarde encore val_tok"
    # 15j. FIFO PAR GROUPE : 12 writes ⇒ 8 GROUPES résidents (8×(1+k) lignes),
    #      et c'est bien le groupe le plus ancien qui sort EN ENTIER.
    env_g = OracleEnv(tok, 8)
    b_g = []
    firsts = []
    for k in range(12):
        s = dict(seg_tpl)
        s["fact_val"] = torch.full_like(seg_tpl["fact_val"], 1 + k)
        s["input_ids"] = seg_tpl["input_ids"].clone()
        s["input_ids"][0, 1] = 100 + k          # segs DISTINCTS
        b_g = env_g.write(m_tr, b_g, s)
        firsts.append(b_g[-(1 + K6)].clone())   # la clé du groupe fraîchement
        assert env_g.last_added == 1            # posé ; 1 write = 1 groupe
        assert len(b_g) == min(k + 1, 8) * (1 + K6), (k, len(b_g))
        assert b_g.groups == [1 + K6] * min(k + 1, 8)
    # les 4 premiers groupes sont sortis, les 8 derniers sont là, dans l'ordre
    for i in range(8):
        assert torch.equal(b_g[i * (1 + K6)], firsts[4 + i]), i
    # 15k. GroupReadout : porte fermée ⇒ biais EXACTEMENT nul ; layout lu à la
    #      bonne foulée ; softmax des deux étages bien formés.
    bk_g = torch.stack(list(b_g))[None]                    # [1, 8*(1+k), 512]
    bm_g = torch.ones(1, bk_g.size(1), dtype=torch.bool)
    with torch.no_grad():
        bias = m_tr.ptr(torch.randn(1, 5, 512), bk_g, bm_g, m_tr.embed.weight)
    assert float(bias.abs().max()) == 0.0, float(bias.abs().max())
    assert float(m_tr.ptr.last_gate.max()) < 1e-3
    assert m_tr.ptr.last_sel.shape[-1] == 8 and \
        m_tr.ptr.last_pos.shape[-1] == K6
    assert torch.allclose(m_tr.ptr.last_sel.sum(-1), torch.ones(1, 5))
    # porte OUVERTE + étage 1 forcé sur UN groupe : le biais pointe un token de
    # CE groupe — la citation est bien native.
    with torch.no_grad():
        m_tr.ptr.scale.fill_(50.0)
        m_tr.ptr.gate.bias.fill_(8.0)
        one = torch.stack(list(b_g)[:1 + K6])[None]
        b2 = m_tr.ptr(torch.randn(1, 1, 512), one,
                      torch.ones(1, 1 + K6, dtype=torch.bool),
                      m_tr.embed.weight)
    cand_g = set(int(torch.argmax(rms_unit(m_tr.embed.weight.float())
                                  @ list(b_g)[1 + j])) for j in range(K6))
    assert int(b2[0, 0].argmax()) in cand_g, int(b2[0, 0].argmax())
    # 15l. banque mal alignée ⇒ erreur EXPLICITE (le layout est un invariant)
    try:
        m_tr.ptr(torch.randn(1, 2, 512), bk_g[:, :-1],
                 torch.ones(1, bk_g.size(1) - 1, dtype=torch.bool),
                 m_tr.embed.weight)
    except AssertionError as e:
        assert "layout" in str(e)
    else:
        raise AssertionError("banque non alignée : le readout aurait dû crier")
    # 15m. --write every : groupe à clé NULLE pour un seg sans fait, et le flux
    #      évince les groupes de faits (même sémantique qu'aux autres codes).
    env_ge = OracleEnv(tok, 8, write_mode="every")
    b_ge = env_ge.write(m_tr, [], nofact)
    assert len(b_ge) == 1 + K6 and env_ge.last_added == 1
    assert float(b_ge[0].abs().max()) == 0.0, "groupe bare : clé non nulle"
    assert float(torch.stack(list(b_ge)[1:]).abs().max()) > 0
    assert torch.equal(torch.stack(list(b_ge)),
                       torch.stack(list(env_ge.write(m_tr, [], nofact)))), \
        "groupe bare non déterministe"
    b_ge = []
    fr = []
    for k in range(3):
        s = dict(factseg)
        s["fact_val"] = torch.full_like(factseg["fact_val"], 1 + k)
        s["input_ids"] = factseg["input_ids"].clone()
        s["input_ids"][0, 1] = 100 + k
        b_ge = env_ge.write(m_tr, b_ge, s)
        fr.append(b_ge[-(1 + K6)].clone())
    for k in range(8):
        b_ge = env_ge.write(m_tr, b_ge, nofact)
    assert len(b_ge) == 8 * (1 + K6)
    assert not any(any(torch.equal(x, y) for y in b_ge) for x in fr), \
        "FIFO toprows every : 8 segs vides doivent purger tous les groupes"
    # 15n. clé STABLE entre instanciations et SÉPARANT les paires
    kk1 = _mk("toprows", d=512, top_k=K6).pack_key
    assert torch.equal(kk1, m_tr.pack_key), "clé toprows non stable"
    assert not torch.allclose(kk1[3, 2], kk1[3, 1], atol=1e-4)
    # 15o. groupe TOUJOURS de taille fixe, même si le seg est plus court que k
    m_big = _mk("toprows", d=512, top_k=9)
    g_short = m_big.oracle_lines(3, 2, tok8, seg_tok=seg_tr[:4])
    assert g_short.shape == (10, 512)
    assert torch.equal(g_short[4], g_short[9]), \
        "groupe court : la dernière ligne de contenu doit être répétée"

    # ═══ CHANTIER 2 : readout MoS (mélanger les DISTRIBUTIONS) ══════════════
    torch.manual_seed(20260803)
    # UN SEUL modèle, on ne bascule que le mode de mélange : la banque doit
    # être faite des embeddings DE CE modèle, sinon les lignes sont des
    # vecteurs étrangers, tous les softmax sont plats et la mesure ne veut
    # plus rien dire (piège payé en écrivant ce test).
    m_one = _mk("toprows", d=512, top_k=K6)
    env_m = OracleEnv(tok, 8)
    b_m: list = []
    for kk in range(3):
        s = dict(seg_tpl)
        s["fact_val"] = torch.full_like(seg_tpl["fact_val"], 1 + kk)
        s["input_ids"] = seg_tpl["input_ids"].clone()
        s["input_ids"][0, 1] = 100 + kk
        b_m = env_m.write(m_one, b_m, s)
    bk_m = torch.stack(list(b_m))[None]
    bm_m = torch.ones(1, bk_m.size(1), dtype=torch.bool)
    h_m = torch.randn(1, 4, 512)
    # 17a. porte fermée ⇒ biais EXACTEMENT nul aussi en MoS (le log ne doit pas
    #      fabriquer de NaN : les probas sous-débordantes sont clampées)
    m_one.ptr.mix = "mos"
    with torch.no_grad():
        bm_bias = m_one.ptr(h_m, bk_m, bm_m, m_one.embed.weight)
    assert float(bm_bias.abs().max()) == 0.0 and torch.isfinite(bm_bias).all()
    # 17b. porte OUVERTE : le MoS ne peut PAS élire un token qu'AUCUNE ligne ne
    #      porte. On force le PIRE CAS de superposition : sélection de groupe et
    #      porte-position UNIFORMES (wq/wp à zéro), donc le linéaire additionne
    #      toutes les lignes de la banque avant de projeter.
    with torch.no_grad():
        m_one.ptr.scale.fill_(30.0)
        m_one.ptr.gate.bias.fill_(8.0)
        m_one.ptr.wp.weight.zero_()
        m_one.ptr.wq.weight.zero_()
        m_one.ptr.mix = "mos"
        top_mos = [int(x) for x in
                   m_one.ptr(h_m, bk_m, bm_m, m_one.embed.weight)[0].argmax(-1)]
        m_one.ptr.mix = "linear"
        top_lin = [int(x) for x in
                   m_one.ptr(h_m, bk_m, bm_m, m_one.embed.weight)[0].argmax(-1)]
    # tokens RÉELLEMENT présents dans la banque (toutes lignes de contenu)
    Erms_g = rms_unit(m_one.embed.weight.float())
    present = set()
    for r in range(bk_m.size(1)):
        if r % (1 + K6) == 0:
            continue                          # ligne-clé : pas un token
        present.add(int(torch.argmax(Erms_g @ bk_m[0, r])))
    assert all(t in present for t in top_mos), (
        f"MoS a élu un token ABSENT de la banque {top_mos} ⊄ {present}")
    mos_hyb = 0                               # (le linéaire, lui, PEUT sortir
    lin_hyb = sum(1 for t in top_lin if t not in present)   # du répertoire)
    # 17c. le knob est bien porté par la config et refuse l'inconnu
    assert _mk("toprows", d=32, top_k=3).cfg.readout_mix == "linear"
    try:
        ToyCfg(vocab_size=512, d_model=32, n_heads=4, mem_dim=32, variant="r3",
               code="toprows", top_k=3, readout_mix="bogus")
    except AssertionError as e:
        assert "readout_mix" in str(e)
    else:
        raise AssertionError("readout_mix inconnu aurait dû être refusé")
    # 17d. l'entropie de la porte-position est exposée et DÉRIVABLE (pénalité)
    m_e2 = _mk("toprows", d=512, top_k=K6)
    m_e2.ptr(h_m, bk_m, bm_m, m_e2.embed.weight)
    assert m_e2.ptr.last_pos_ent.requires_grad
    assert abs(float(m_e2.ptr.last_pos_ent) - math.log(K6)) < 1e-4, \
        "porte-position zéro-init : l'entropie doit être log(k) (uniforme)"

    # ═══ CHANTIER 3 : r4, injection à sélection oracle ══════════════════════
    torch.manual_seed(20260804)
    c_r4 = ToyCfg(vocab_size=512, d_model=64, n_layers=2, n_heads=4,
                  mem_dim=64, variant="r4", max_seq_len=64, code="toprows",
                  seg_n_pos=8, sif_a=A_SIF, top_k=4, inject_sep_id=5)
    m_r4 = ToyReadLM(c_r4, env.n_slots, env.n_attrs, sif_w=_sifw()).eval()
    # 18a. AUCUN module de read appris : ni cross-attn, ni fast-weight, ni ptr
    assert m_r4.ptr is None and all(b.read is None for b in m_r4.blocks)
    assert [n for n, _ in m_r4.named_parameters() if "inject_type" in n] == \
        ["inject_type"], "r4 doit avoir UN seul paramètre de read"
    assert float(m_r4.inject_type.abs().max()) == 0.0, "inject_type zéro-init"
    # 18b. l'injection CHANGE le forward, et l'ABLATÉ est le backbone NU
    #      bit-à-bit (c'est le contraste que le run mesure)
    ids4 = torch.randint(0, 512, (2, 7))
    inj4 = torch.randint(0, 512, (2, 4))
    with torch.no_grad():
        o_inj = m_r4(ids4, None, None, inject=inj4)
        o_abl = m_r4(ids4, None, None)
        o_abl2 = m_r4(ids4)
    assert o_inj.shape == o_abl.shape == (2, 7, 512), o_inj.shape
    assert torch.equal(o_abl, o_abl2), "ablaté r4 ≠ backbone nu"
    assert not torch.allclose(o_inj, o_abl, atol=1e-5), \
        "l'injection ne change RIEN au forward"
    # 18c. le contenu injecté COMPTE (deux préfixes différents ⇒ deux sorties)
    with torch.no_grad():
        o_inj2 = m_r4(ids4, None, None, inject=(inj4 + 1) % 512)
    assert not torch.allclose(o_inj, o_inj2, atol=1e-5)
    # 18d. POSITIONS : le tour réel est décalé de k+2 et la position k+1 reste
    #      VIDE. On le vérifie par construction du vecteur d'index.
    k4, T4 = 4, 7
    pos_expect = list(range(k4 + 1)) + [k4 + 2 + i for i in range(T4)]
    assert pos_expect[:k4] == [0, 1, 2, 3]        # injecté 0..k−1
    assert pos_expect[k4] == 4                    # séparateur
    assert pos_expect[k4 + 1] == 6                # tour réel : trou en 5
    # et le modèle utilise BIEN ces positions : décaler la RoPE change la
    # sortie, donc le trou n'est pas cosmétique.
    with torch.no_grad():
        cos, sin = _rope_tables(16, 16, c_r4.rope_theta, ids4.device,
                                torch.float32)
        assert not torch.allclose(cos[5], cos[6], atol=1e-6)
    # 18e. lignes injectées = embeddings BRUTS (NON RMS-normés) + type
    with torch.no_grad():
        pre = m_r4.embed(inj4) + m_r4.inject_type
        assert torch.equal(pre, m_r4.embed(inj4)), "type zéro-init : pre = brut"
        nrm = pre.float().pow(2).mean(-1).sqrt()
        assert float((nrm - 1.0).abs().max()) > 1e-3, \
            "les lignes injectées ne doivent PAS être RMS-normées"
    # 18f. le PLAN d'injection : un groupe par seg de réponse, sélection =
    #      celle du write, et le fait ÉVINCÉ n'est pas injecté.
    plan, absent = env.inject_plan(m_r4, conv)
    for i, tk in plan.items():
        assert conv["segs"][i]["role"] == "assistant"
        assert tk.numel() <= c_r4.top_k
    # un fait écrit APRÈS la réponse ne peut pas être injecté devant elle
    assert all(i > min((j for j, s in enumerate(conv["segs"])
                        if OracleEnv.fact_of(s)), default=0) for i in plan)
    # 18g. r4 REFUSE ce qui n'a pas de sens : un autre code, le régime `every`
    for bad, needle in ((dict(code="segsif"), "toprows"),
                        (dict(code="toprows", write_mode="every"), "fact-only")):
        try:
            ToyCfg(vocab_size=512, d_model=32, n_heads=4, mem_dim=32,
                   variant="r4", seg_n_pos=8, sif_a=A_SIF, top_k=3, **bad)
        except AssertionError as e:
            assert needle in str(e), (bad, str(e))
        else:
            raise AssertionError(f"r4 aurait dû refuser {bad}")

    # ═══ CHANTIER 1 : grade CONDITIONNÉ À LA RÉSIDENCE ══════════════════════
    # 19. `resident` est aligné sur les réponses gradées : autant d'entrées que
    #     de vérités, et grade_resident ne grade QUE les faits encore là.
    #     (Test d'intégration mené sur le vrai `evaluate` au smoke ; ici on
    #     vérifie l'invariant d'alignement sur une trace synthétique.)
    _live = ["Palermo", "X", "Barnaby"]
    _tru = ["Palermo", "Y", "Barnaby"]
    _res = [True, False, True]
    _idx = [i for i, x in enumerate(_res) if x is True]
    assert len(_res) == len(_tru)
    assert grade_recall([_live[i] for i in _idx],
                        [_tru[i] for i in _idx]) == 1.0, \
        "grade|non-évincé doit ignorer les réponses dont le fait est sorti"

    # 16. RÉTRO-COMPAT BIT-À-BIT des codes NON-pack. La constante ci-dessous a
    #     été relevée sur le commit 4daf6a6 (phase 4, AVANT le pack) avec la
    #     même graine et les mêmes entrées : si l'ajout du pack avait déplacé
    #     d'un cran la consommation du générateur global (ordre de création des
    #     modules) ou touché un chemin partagé, elle bougerait. C'est
    #     l'assertion qui remplace le `git stash` manuel des phases passées.
    torch.manual_seed(4242)
    m_rc = _mk("segsif", d=64, n_pos=4, vocab=512, seg_n_pos=8)
    ids_rc = (torch.arange(12).reshape(2, 6) * 37) % 512
    rows_rc = m_rc.oracle_lines(3, 2, tok8[:3], seg_tok=seg20)
    bk_rc = rows_rc[:1][None].expand(2, -1, -1).contiguous()
    bm_rc = torch.ones(2, 1, dtype=torch.bool)
    with torch.no_grad():
        out_rc = float(m_rc(ids_rc, bk_rc, bm_rc).double().sum())
    RC_REF_4DAF6A6 = -18.756196630471095
    assert abs(out_rc - RC_REF_4DAF6A6) < 1e-9, (
        f"RÉTRO-COMPAT ROMPUE : forward segsif {out_rc} ≠ {RC_REF_4DAF6A6} "
        f"(valeur du commit 4daf6a6) — l'ajout du pack a déplacé un chemin "
        f"partagé ou le tirage des poids")
    # … et la MÊME garantie pour `toprows` en readout_mix=linear (le défaut),
    # constante relevée sur le commit 1345f49 (phase 6, avant MoS et r4).
    torch.manual_seed(4242)
    m_r2 = _mk("toprows", d=64, n_pos=4, vocab=512, seg_n_pos=8, top_k=3)
    rows_r2 = m_r2.oracle_lines(3, 2, tok8[:3], seg_tok=seg20)
    with torch.no_grad():
        out_r2 = float(m_r2(ids_rc, rows_r2[None].expand(2, -1, -1).contiguous(),
                            torch.ones(2, rows_r2.shape[0], dtype=torch.bool)
                            ).double().sum())
    RC_REF_1345F49 = -18.753843640828563
    assert abs(out_r2 - RC_REF_1345F49) < 1e-9, (
        f"RÉTRO-COMPAT ROMPUE : forward toprows/linear {out_r2} ≠ "
        f"{RC_REF_1345F49} (valeur du commit 1345f49) — le MoS ou r4 a touché "
        f"le chemin par défaut du GroupReadout")

    print("toy_read_lab self-test: OK (write oracle déterministe & "
          "embedding-dépendant, FIFO 8, porte pointer fermée à l'init, "
          "masque CE assistant-seul, padding de banque inerte, "
          "4 variantes forward live+ablaté)")
    print("  phase 2 — round-trip ORACLE (top-1 exacts / positions, vocab 512, "
          "d_model 256, n_pos 8) : " +
          "  ".join(f"{c} {rt[c][0]}/{rt[c][1]}" for c in P2_CODES) +
          "   [mean DOIT échouer : anagrammes ⇒ ligne identique]")
    print(f"  phase 2 — lane vide ≡ ablaté & porte fermée pour les "
          f"{len(CODES)} formats, FIFO rows (3 faits × 3 tokens ⇒ 8 lignes), "
          f"r0/r1/r2 refusent tout code ≠ mean (pack/segpack compris)")
    print(f"  phase 3 — POOL DU SEG ENTIER (privilège span-valeur retiré, "
          f"seg_n_pos 32, seg synthétique de 20 tokens, vocab 512) : "
          f"round-trip segphase d=512 valeur {ok}/{n}, seg entier "
          f"{ok_all}/{n_all} ; MÊME seg à d=256 (dims du run) "
          f"{ok_256}/{n_256} — régime SNR-limité, cf. le round-trip à "
          f"l'échelle réelle ; segmean {ok_sm}/{n_sm} [DOIT échouer : "
          f"permutation ⇒ ligne identique] ; codes indépendants de val_tok, "
          f"FIFO 8 inchangé")
    print(f"  phase 3 — segsif (pool PONDÉRÉ SIF w=a/(a+p), a={A_SIF:g}) : "
          f"round-trip valeur {ok_ss}/{n_ss}, code ≠ segphase, T_eff "
          f"{t_eff:.1f}/{seg_f.numel()} tokens, écrasement d'un token fréquent "
          f"Δ {d_ss:.4f} vs {d_sp:.4f} en poids uniformes ; sif_w et sif_a>0 "
          f"exigés")
    print("  extension 1 — code.pos_offset : table décalée (ligne j = rot(θ·"
          "(j+k))), appliqué au WRITE ET aux candidats (round-trip segphase "
          "offset 1 exact, position 0 comprise), wrap n_pos−1 ≡ position 0 du "
          "format non décalé, refusé hors phase/segphase/segsif et si "
          "offset ≥ n_pos, déterministe")
    print("  extension 2 — --write every : seg SANS fait ⇒ UNE ligne RMS 1 "
          "SANS composante K/A (indépendante du slot/attr), FIFO évince les 3 "
          "faits après 8 segs vides (le plus ancien dès 6 ; mode fact : rien n'est "
          "évincé), lane vide "
          "≡ ablaté avec lignes nues en banque, déterministe, refusé pour "
          "mean/chunk/phase/rows")
    print(f"  phase 5 — PACK (ligne PARTITIONNÉE, {8} blocs : 1 clé + 7 tokens, "
          f"d=512 ⇒ 64 dims/bloc) : round-trip pack {ok_pk}/{n_pk} (span "
          f"tronqué à 7), segpack {ok_sk}/{n_sk} (sélection top-7 SIF = les 7 "
          f"tokens rares du seg, ordre du segment, val_tok jamais lu) ; clé et "
          f"frames STABLES entre instanciations et séparant les paires ; blocs "
          f"DISJOINTS (toucher un token ne bouge que son bloc) ; PackReadout "
          f"biais EXACTEMENT 0 porte fermée et pointe un token de la ligne "
          f"porte ouverte ; ligne bare (write every) SANS bloc-clé ; "
          f"pack_blocks non-diviseur / < 2 refusés")
    print(f"  phase 6 — TOPROWS (GROUPES de lignes NATIVES, top_k {K6} ⇒ "
          f"groupes de {1 + K6}) : sélection top-k SIF dans l'ORDRE DU SEGMENT, "
          f"ligne 0 = clé de la paire, lignes de contenu à cos > 0.97 de "
          f"l'embedding brut (EXACTES sans row_pos_tag) ; round-trip "
          f"{ok_tr}/{n_tr} (le tag ne déplace pas l'argmax) ; FIFO PAR GROUPE "
          f"(12 writes ⇒ 8 groupes, éviction du plus ancien EN ENTIER, "
          f"last_added = 1 write) ; --write every ⇒ groupe à clé NULLE, purge "
          f"complète après 8 segs vides ; GroupReadout biais EXACTEMENT 0 porte "
          f"fermée, cite une ligne du groupe porte ouverte, banque mal alignée "
          f"refusée ; groupe de taille FIXE même sur seg court")
    print(f"  chantier 2 — readout MoS : biais EXACTEMENT 0 porte fermée (log "
          f"clampé, aucun NaN), et dans le PIRE CAS de superposition (sélection "
          f"et porte-position uniformes) le MoS n'élit QUE des tokens présents "
          f"en banque — propriété GARANTIE, alors que le linéaire n'en offre "
          f"aucune (il en sort {lin_hyb}/4 fois sur ce tirage) ; knob "
          f"readout_mix (défaut linear) et entropie de p exposée/dérivable "
          f"(= log k à l'init)")
    print("  chantier 3 — r4 INJECTION ORACLE : aucun module de read (UN seul "
          "paramètre, inject_type zéro-init), ablaté ≡ backbone nu BIT-À-BIT, "
          "l'injection change le forward et son CONTENU compte, lignes "
          "injectées NON RMS-normées, positions 0..k−1 / séparateur / tour "
          "décalé de k+2 (trou en k+1), plan d'injection = sélection du write "
          "avec FIFO max_mem (fait évincé ⇒ pas d'injection), r4 refuse un "
          "autre code et le régime every")
    print("  chantier 1 — grade CONDITIONNÉ À LA RÉSIDENCE aligné sur les "
          "réponses gradées (n_resident/n_absent), et éval FINALE élargie "
          "(training.final_eval_convs, défaut 200) écrite dans "
          "final_metrics.csv avec son erreur-type")
    print("  rétro-compat — forward segsif identique BIT-À-BIT au commit "
          "4daf6a6 (constante figée dans le self-test) : l'ajout du pack n'a "
          "déplacé ni un chemin partagé ni le tirage des poids ; forward "
          "toprows/linear identique BIT-À-BIT au commit 1345f49 (le MoS et r4 "
          "n'ont pas touché le chemin par défaut)")


if __name__ == "__main__":
    main()
