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
from .persona_chat_data import PersonaChatStream, fact_id_maps, grade_recall
from .streams import chat_stream_class

VARIANTS = ("r0", "r1", "r2", "r3", "r4", "r5")
# r4 = INJECTION À SÉLECTION ORACLE : AUCUN module de read appris. Le groupe
# toprows du fait interrogé est injecté en PRÉFIXE de pseudo-tokens et c'est le
# backbone NU qui doit copier. Bras « le stack natif sait-il copier ? ».
# r5 = r4 + RETRIEVER APPRIS : la sélection oracle est remplacée par un score
# appris sur les lignes-clés résidentes. VERDICT r4 (phase 7) : strate `code`
# 0.708 (n=106) — la citation exige le stack natif, il ne manque QUE la
# sélection. PRÉDICTION INSCRITE AVANT LE RUN : plafond de r5 = 0.708 ×
# recall@2 ; l'oracle des clés sépare à 100 %, donc un recall@2 < 0.9
# incriminerait l'apprentissage de W_q, pas la géométrie.
INJECT_VARIANTS = ("r4", "r5")
# variantes dotées d'un RETRIEVER appris (W_q sur les lignes-clés).
RETRIEVER_VARIANTS = ("r5",)
# les variantes qui LISENT UNE BANQUE (r4 n'en a pas : il lit une injection).
BANK_VARIANTS = ("r0", "r1", "r2", "r3")
# mélange des lignes par le readout de groupe (cf. GroupReadout) :
#   linear = superposition DANS l'espace d'embedding puis UNE projection
#   mos    = une distribution PAR LIGNE, puis mixture des DISTRIBUTIONS
READOUT_MIXES = ("linear", "mos")
CODES = ("mean", "chunk", "phase", "rows", "segmean", "segphase", "segsif",
         "pack", "segpack", "toprows", "tophid", "midhid")
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
# phase 10 : ce que l'injection dépose À L'ENTRAÎNEMENT devant un tour
# conditionné (cf. ToyCfg.cond_arm). Les TROIS conditions de LECTURE sont, elles,
# mesurées à l'éval sur CHAQUE run (cf. evaluate_cond) : live / shuf / none.
COND_ARMS = ("true", "shuffle", "none")
# phase 10 — OÙ les lignes entrent dans le stack (cf. ToyCfg.read_path).
READ_PATHS = ("entry", "kv")
# phase 10 — les QUATRE chemins de lecture exposés par `--read`, traduits en
# (variant, code|None, read_path). C'est la table qui rend la grille de la spec
# §2.4 lançable par un seul flag ; les axes `--tap`, `--age-rot` et `--m` sont
# ORTHOGONAUX à celui-ci.
#   seq_fw       lecture fast-weight SÉQUENTIELLE (r0) sur la MÊME banque de
#                groupes que les autres bras : la matrice plate
#                (max_mem·(1+m), d) est lue LIGNE À LIGNE, chaque ligne
#                engendrant sa transformation low-rank appliquée en série.
#                C'est littéralement le §2.4(b) (« chaque ligne = un sous-slot
#                d'entrée d, application séquentielle sur m×S sous-slots »).
#                ⚠️ COÛT : la boucle fait max_mem·(1+m) itérations PAR COUCHE —
#                à m=8 et max_mem=8 c'est 72 itérations × n_layers. C'est le
#                bras lent de la grille, et c'est intrinsèque au design.
#                Il n'a PAS de readout pointer (r0 n'en porte pas) : ce bras
#                mesure la lecture fast-weight NUE.
#   bank_xattn   lecture APPRISE sur la MÊME banque de groupes que l'injection
#                (r3 : cross-attn contenu + GroupReadout). C'est LUI le
#                comparateur non-injectif propre.
#   inject_entry pseudo-tokens en préfixe (r4, lecture α) — le bras de la ph.7-9.
#   kv_append    lignes appondues aux K/V des couches lectrices (r4 + kv,
#                lecture β) — sans ré-encodage, sans RoPE.
READ_MODES = {
    "seq_fw":       ("r0", None,   "entry"),
    "bank_xattn":   ("r3", None,   "entry"),
    "inject_entry": ("r4", None,   "entry"),
    "kv_append":    ("r4", None,   "kv"),
}
# formats PARTITIONNÉS (phase 5) : la ligne est un PACK de blocs disjoints —
# bloc 0 = clé dédiée, blocs 1..B−1 = un token chacun. Readout = PackReadout.
PACK_CODES = ("pack", "segpack")
# formats à BANQUE DE GROUPES (phase 6) : un write dépose 1+top_k LIGNES
# NATIVES — ligne 0 = clé dédiée, lignes 1.. = les embeddings BRUTS des tokens
# sélectionnés. Principe : NE JAMAIS TRANSFORMER. Le FIFO compte les GROUPES.
GROUP_CODES = ("toprows", "tophid", "midhid")
# ── phase 9 : la ligne stockée n'est plus un EMBEDDING D'ENTRÉE ─────────────
# `tophid` est `toprows` avec UNE différence, et une seule : la ligne de contenu
# est l'état caché APRÈS `norm_f` (l'espace que consomme `lm_head`, tying
# oblige) au lieu de l'embedding d'entrée du token. C'est le design
# `Banque(max_mem, mem_dim, d_model)` — on sélectionne des vecteurs après la
# dernière RMSNorm et on empile la matrice.
#
# POURQUOI L'A/B EST ISOLÉ À CE POINT : même clé oracle ligne 0, même sélection
# SIF, même `group_rows`, même `GroupReadout`, même layout d'injection, même
# `rms_unit` appliqué à la ligne. Ne bouge QUE le vecteur normé.
#
# CE QUE LE FROM-SCRATCH APPORTE, et que la sonde sur ckpt ne pouvait pas dire :
# `analysis/native_row_channel.py` a mesuré, sur un 350M entraîné SANS ce canal,
# qu'une ligne post-norm ne cite pas son propre token (rang médian 374). Mais
# `norm_f` et `lm_head` y ont été façonnés par un objectif qui ignore la banque.
# Entraîné DEPUIS ZÉRO, le modèle peut au contraire façonner cet espace pour
# qu'il porte la surface. C'est cette question-là, et elle seule, que la paire
# toprows/tophid tranche.
#
# ── phase 10 : `midhid` — le POINT DE PRÉLÈVEMENT descend à ~2/3 de profondeur
# `midhid` est `tophid` avec, encore une fois, UNE seule différence : la ligne
# de contenu est l'état du flux résiduel APRÈS `hid_tap_layers` blocs (défaut
# round(2/3·n_layers)), donc AVANT `norm_f` et avant la rotation vocabulaire
# que le tying impose au dernier étage. C'est le correctif ranké de la spec
# §2.4 (« le gist veut les CONCLUSIONS, pas la rotation vocabulaire ») et
# l'autre moitié du débat « prélever trop bas » : `toprows` prélève à la
# couche 0 (embedding d'entrée), `tophid` à la sortie, `midhid` entre les deux.
HID_CODES = ("tophid", "midhid")
# codes qui exigent la table SIF (pondération/sélection des tokens du segment).
SIF_CODES = ("segsif", "segpack", "toprows", "tophid", "midhid")
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
    # ── axe INJECTION (variantes r4 / r5) ───────────────────────────────────
    inject_sep_id: int = 0    # token du vocab posé ENTRE le préfixe injecté et
                              # le tour réel. Renseigné par main() (`<blank>`).
    retr_ce: float = 1.0      # r5 : coefficient de la CE AUXILIAIRE qui
                              # supervise le retriever (cible = l'index du
                              # groupe porteur du fait interrogé). Le retriever
                              # apprend PAR CE CANAL, pas par la loss LM.
    retr_topk: int = 2        # r5 : nombre de groupes injectés À L'ÉVAL (top-k
                              # DUR du retriever). À l'ENTRAÎNEMENT l'injection
                              # reste le groupe ORACLE (cf. ToyReadLM.forward).
    retr_train_groups: int = 0   # r5 : groupes injectés À L'ENTRAÎNEMENT.
                              # 0 = « suivre retr_topk » (résolu au
                              # __post_init__) — c'est le DÉFAUT, et il ferme
                              # le mismatch train/éval : v1 entraînait sur UN
                              # groupe et évaluait sur top-2, le circuit de
                              # copie se câblait sur un préfixe mono-groupe et
                              # le distracteur le cassait (recall@1 1.000 mais
                              # grade 0.067 contre 0.708 en r4).
                              # 1 = comportement v1 (rétro-compat bit-à-bit des
                              # runs 60/61). Si un fait n'a pas assez de voisins
                              # résidents, G est RÉDUIT pour cette réponse — on
                              # ne complète JAMAIS en répétant l'oracle.
    retr_train_order: str = "random"   # r5 : place de l'oracle dans le préfixe
                              # d'entraînement. `random` = ordre tiré au sort
                              # (le modèle ne doit pas apprendre « copie le
                              # groupe 1 ») ; `oracle_first` = oracle toujours
                              # en tête, ce qui REPRODUIT l'ordre de l'éval
                              # (les groupes y sont triés par score, et le vrai
                              # groupe sort premier — recall@1 mesuré à 1.000).
                              # ⚠️ CF. LE COMMENTAIRE DU YAML : le prompt de
                              # décodage ne contient PAS la question, donc en
                              # ordre aléatoire rien n'identifie le bon groupe
                              # à l'inférence.
    retr_detach: bool = True  # r5 : la CE du retriever ne remonte PAS dans le
                              # backbone (h_query est détaché). W_q est alors le
                              # SEUL paramètre que ce canal entraîne — c'est ce
                              # qui rend la prédiction lisible (« si recall@2 <
                              # 0.9, le goulot est W_q »). False = laisser la
                              # sélection remodeler le trunk, au risque de
                              # déranger le circuit de copie déjà acquis.
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
    # ── axe CHEMIN DE LECTURE (phase 10, spec §2.4 « VUE PLATE + ATTENTION »)
    read_path: str = "entry"  # OÙ les lignes entrent, pour les variantes
                              # d'injection :
                              #   `entry` (DÉFAUT, rétro-compat bit-à-bit de la
                              #     phase 7-9) = pseudo-tokens en PRÉFIXE, à
                              #     l'entrée du stack (lecture α). Les lignes
                              #     traversent TOUTES les couches et prennent
                              #     des positions RoPE.
                              #   `kv` = lignes appondues aux K/V des couches
                              #     LECTRICES, SANS ré-encodage ni RoPE
                              #     (lecture β). La ligne est découpée en
                              #     n_heads morceaux et sert de clé ET de
                              #     valeur. Zéro paramètre ajouté (le vecteur
                              #     de type reste le seul).
    # ── axe POINT DE PRÉLÈVEMENT (phase 10, `midhid`) ───────────────────────
    hid_tap: float = 2.0 / 3.0   # fraction de profondeur où `midhid` prélève
                              # l'état de la ligne. Le nombre de blocs traversés
                              # est round(hid_tap · n_layers), borné à
                              # [1, n_layers] (cf. hid_tap_layers). 1.0 = juste
                              # AVANT norm_f (le contrôle qui isole norm_f de la
                              # profondeur), le défaut 2/3 = le correctif ranké
                              # de la spec §2.4. Ignoré hors `midhid`.
    # ── axe ÂGE (phase 10, design ROTATION-PUIS-APLATISSEMENT) ──────────────
    age_rope: bool = False    # rote chaque LIGNE injectée par l'ÂGE EN WRITES
                              # de son slot (rot(θ·âge), même binding DFT que
                              # `phase`, table de max_mem entrées) AVANT
                              # d'ajouter le vecteur de type. La banque cible
                              # est (max_mem, m, d) VUE comme la matrice plate
                              # (max_mem·m, d) : aucune somme, aucune réduction
                              # — les lignes restent séparées et la rotation
                              # porte la provenance/récence. False (DÉFAUT) =
                              # rétro-compat bit-à-bit : le chemin d'injection
                              # ne voit pas une opération de plus.
    # ── axe CONDITIONNEMENT (phase 10) ──────────────────────────────────────
    cond: bool = False        # tâche de RÈGLE : le stream devient
                              # PersonaRuleStream et l'éval contrastive
                              # `evaluate_cond` s'ajoute (Δnll d'une
                              # continuation COHÉRENTE avec la règle contre une
                              # continuation INCOHÉRENTE). Ne retire RIEN : les
                              # convs de rappel restent dans le flux, donc la
                              # citation de la phase 9 continue d'être mesurée
                              # par `evaluate` sur le même run.
    cond_arm: str = "true"    # ce qui est injecté À L'ENTRAÎNEMENT devant les
                              # tours conditionnés (r4 seulement) :
                              #   true    = les groupes RÉSIDENTS de la vie
                              #   shuffle = ceux d'une AUTRE vie du lot
                              #             (contrôle « banque MÉLANGÉE »)
                              #   none    = rien (BORNE BASSE : backbone nu)
    cond_decoys: int = 1      # faits LEURRES plantés dans une vie-règle, en
                              # plus de la règle : le préfixe injecté fait donc
                              # 1 + cond_decoys groupes, la règle n'est pas
                              # toujours la plus récente, et le code d'âge a
                              # quelque chose à coder. 0 = un seul groupe (tous
                              # les âges valent 0 : `age_rope` devient un no-op
                              # mesurable, c'est le contrôle de la sonde).
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
        if self.variant in ("r3",) + INJECT_VARIANTS or \
                self.code in GROUP_CODES:
            # R3 : la banque VIT dans l'espace d'embedding — pas de projection.
            # R4 : pas de banque du tout, mais les tokens injectés vivent eux
            # aussi dans l'espace d'embedding.
            # CODES DE GROUPES (phase 10) : la banque est la matrice PLATE
            # (max_mem·group_rows, d) de la spec §2.4 — ses lignes sont des
            # vecteurs de pleine largeur, quel que soit le read qui la lit.
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
            assert self.code in GROUP_CODES, (
                f"--variant {self.variant} injecte un GROUPE de lignes "
                f"natives : il exige --code ∈ {GROUP_CODES} (reçu --code "
                f"{self.code})")
            assert self.write_mode == "fact", (
                f"--variant {self.variant} est un bras fact-only (le régime "
                f"`every` n'a pas de sens ici : la banque n'est lue que par "
                f"l'injection)")
        if self.variant in RETRIEVER_VARIANTS:
            assert self.retr_ce >= 0.0, self.retr_ce
            assert self.retr_topk >= 1, self.retr_topk
            # 0 = sentinelle « suivre retr_topk » : l'entraînement voit le même
            # nombre de groupes que l'éval, c'est TOUT le point du fix.
            if not self.retr_train_groups:
                self.retr_train_groups = self.retr_topk
            assert self.retr_train_groups >= 1, self.retr_train_groups
            assert self.retr_train_order in ("random", "oracle_first"), (
                f"retr_train_order inconnu {self.retr_train_order!r}")
        if self.code != "mean":
            # les nouveaux formats supposent banque == espace d'embedding et
            # pointer nu : c'est la définition de r3, on ne les porte pas
            # ailleurs (r0/r1/r2 restent le contrôle de la phase 1). r4 les
            # consomme autrement (injection), il est admis pour `toprows`.
            assert self.variant in ("r3",) + INJECT_VARIANTS or (
                self.variant == "r0" and self.code in GROUP_CODES), (
                f"--code {self.code} n'est supporté QUE par --variant r3 "
                f"(banque en espace d'embedding + pointer nu), les variantes "
                f"d'injection {INJECT_VARIANTS}, et — phase 10 — r0 sur les "
                f"codes de GROUPES (spec §2.4(b) : le hypernet fast-weight "
                f"applique une transformation par LIGNE de la matrice plate, "
                f"donc mem_dim = d et M = max_mem·group_rows sous-slots) ; "
                f"reçu --variant {self.variant}. Phase 1 = --code mean.")
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
        # ── phase 10 : prélèvement, âge, conditionnement ────────────────────
        assert self.read_path in READ_PATHS, (
            f"read_path inconnu {self.read_path!r} (∈ {READ_PATHS})")
        if self.read_path != "entry":
            assert self.variant in INJECT_VARIANTS, (
                f"read_path {self.read_path!r} décrit OÙ entrent les lignes "
                f"INJECTÉES : il exige une variante {INJECT_VARIANTS} (reçu "
                f"--variant {self.variant})")
        assert 0.0 < self.hid_tap <= 1.0, (
            f"hid_tap est une FRACTION de profondeur ∈ ]0, 1] "
            f"({self.hid_tap!r})")
        assert self.cond_arm in COND_ARMS, (
            f"cond_arm inconnu {self.cond_arm!r} (∈ {COND_ARMS})")
        assert self.cond_decoys >= 0, self.cond_decoys
        if self.age_rope:
            assert self.variant in INJECT_VARIANTS or \
                self.code in GROUP_CODES, (
                f"code.age_rope rote les LIGNES par l'âge de leur slot : il "
                f"exige soit une variante d'injection {INJECT_VARIANTS}, soit "
                f"un code de GROUPES lu depuis la banque {GROUP_CODES} (reçu "
                f"--variant {self.variant} --code {self.code}) — ailleurs il "
                f"serait silencieusement ignoré")
        if self.cond:
            assert self.variant in ("r0", "r3") + INJECT_VARIANTS, (
                f"--cond exige un bras qui LIT la banque de groupes : r0 "
                f"(fast-weight séquentiel), r3 (cross-attn + readout) ou "
                f"{INJECT_VARIANTS} (injection) ; reçu --variant "
                f"{self.variant}")
            assert self.code in GROUP_CODES, (
                f"--cond exige --code ∈ {GROUP_CODES} (reçu {self.code})")
            assert self.write_mode == "fact", (
                "--cond est un régime fact-only (le seg de RÈGLE est le seul "
                "porteur qu'il faut retenir)")
        if self.cond_arm != "true":
            assert self.cond and self.variant in INJECT_VARIANTS, (
                f"cond_arm {self.cond_arm!r} pilote ce que l'INJECTION dépose "
                f"à l'entraînement : il exige --cond et une variante "
                f"{INJECT_VARIANTS}")
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
    def hid_tap_layers(self) -> int:
        """Blocs traversés avant le prélèvement de la ligne (`midhid`).

        `tophid` prélève APRÈS `norm_f` (donc après les n_layers blocs ET la
        norme finale) ; `midhid` s'arrête à hid_tap_layers blocs, dans le flux
        RÉSIDUEL brut. hid_tap = 1.0 donne n_layers blocs SANS norm_f : c'est
        le contrôle qui sépare « la profondeur » de « la norme finale ».
        """
        return max(1, min(self.n_layers, int(round(self.hid_tap
                                                   * self.n_layers))))

    @property
    def cond_groups(self) -> int:
        """Groupes injectés devant un tour conditionné (règle + leurres)."""
        return 1 + self.cond_decoys

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

    def forward(self, x, pos=None, mem=None, mem_mask=None):
        """`pos` [T] : index RoPE EXPLICITES. None = 0..T−1 (chemin par défaut,
        bit-à-bit inchangé). La variante r4 s'en sert pour laisser un TROU de
        position entre le préfixe injecté et le tour réel.

        `mem` [B, S, d] (PHASE 10, `read_path='kv'`) : des LIGNES DE BANQUE
        appondues directement aux K et V de cette couche, SANS ré-encodage —
        pas de W_k, pas de W_v, pas de RoPE. La ligne est simplement découpée
        en n_heads morceaux de d_head, exactement comme un état de token le
        serait après projection. C'est la lecture β de la spec §2.4 : la banque
        entre là où l'attention la consomme, pas par l'entrée du stack.
        Le masque devient EXPLICITE (les S colonnes de mémoire sont visibles de
        TOUTES les positions, la partie T×T reste triangulaire) : `is_causal`
        ne sait pas exprimer « préfixe non causal + suffixe causal ».
        """
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
        if mem is None:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            return self.o(y.transpose(1, 2).reshape(B, T, d))
        S = mem.shape[1]
        km = mem.to(q.dtype).view(B, S, self.h, self.dh).transpose(1, 2)
        k = torch.cat([km, k], dim=2)
        v = torch.cat([km, v], dim=2)          # K ET V sont LA MÊME ligne
        am = torch.ones(T, S + T, dtype=torch.bool, device=x.device)
        am[:, S:] = torch.ones(T, T, dtype=torch.bool,
                               device=x.device).tril()
        am = am[None, None].expand(B, 1, T, S + T)
        if mem_mask is not None:
            am = am.clone()
            am[..., :S] &= mem_mask[:, None, None, :].to(torch.bool)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=am)
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


# ── r5 : RETRIEVER appris sur les lignes-clés ────────────────────────────────

class Retriever(nn.Module):
    """Score de SÉLECTION d'un groupe de banque, appris en SUPERVISÉ.

        score_g = (W_q · h_query) · clé_g / √d × exp(log_temp)

    `h_query` est l'état caché du DERNIER token du SEGMENT USER (la question),
    pris au forward de CE segment — jamais un token de la réponse. C'est le
    point critique du bras : en teacher-forcing, un état pris dans la réponse
    contiendrait déjà la valeur à retrouver, et le retriever apprendrait à lire
    la réponse au lieu de la question.

    W_q est ZÉRO-INIT : au step 0 tous les scores valent 0, le softmax est
    uniforme (CE = log G) et le gradient de la CE vaut h ⊗ clé ≠ 0 — le module
    n'est pas mort, il démarre simplement sans préférence.
    """

    def __init__(self, cfg: ToyCfg):
        super().__init__()
        d = cfg.d_model
        self.d = d
        self.wq = nn.Linear(d, d, bias=False)
        nn.init.zeros_(self.wq.weight)
        # température apprise (init 1.0) : la CE peut vouloir durcir le softmax
        # sans que W_q ait à gonfler sa norme.
        self.log_temp = nn.Parameter(torch.zeros(1))

    def forward(self, h, keys, key_mask=None):
        """h [n, d] ; keys [n, G, d] ; key_mask [n, G] → scores [n, G]."""
        q = self.wq(h)                                        # [n, d]
        sc = torch.einsum("nd,ngd->ng", q, keys) / math.sqrt(self.d)
        sc = sc * self.log_temp.exp()
        if key_mask is not None:
            sc = sc.masked_fill(~key_mask, float("-inf"))
        return sc


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

    def forward(self, x, bank, bank_mask, pos=None, mem=None, mem_mask=None):
        # `mem` n'entre QUE dans les couches LECTRICES (`read_layers`) : la
        # lecture β partage exactement le même budget de couches que les reads
        # appris de r0/r1/r3, sinon la comparaison porterait aussi sur « à
        # combien d'étages la banque parle ».
        x = x + self.attn(self.n1(x), pos,
                          mem if self.read_bank else None,
                          mem_mask if self.read_bank else None)
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
        # `tophid` injecte une ligne post-norm là où `toprows` injecte un
        # embedding : deux distributions d'échelle différentes (mesuré au 350M :
        # ‖h‖/‖E[id]‖ ≈ ×17). Un SCALAIRE appris, init 1.0, retire le confondant
        # sans qu'on ait à deviner le bon facteur — et sa valeur finale est
        # elle-même une mesure, donc elle est loggée.
        if cfg.code in HID_CODES:
            self.hid_scale = nn.Parameter(torch.ones(()))
        self.retr = (Retriever(cfg) if cfg.variant in RETRIEVER_VARIANTS
                     else None)
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
        # ── phase 10 : table de rotation par ÂGE ────────────────────────────
        # Même binding DFT que `phase` (base ≤ 0 ⇒ θ_i = 2π·(i mod n)/n), mais
        # indexé sur l'âge EN WRITES du slot (0 = write le plus récent). Elle
        # ne s'applique QU'AUX LIGNES INJECTÉES et ne touche jamais la banque
        # stockée : la matrice plate (max_mem·m, d) reste telle quelle, la
        # rotation est posée au moment où on l'aplatit dans le préfixe.
        # Buffers créés seulement si age_rope ⇒ state_dict inchangé sinon.
        if cfg.age_rope:
            c, s = phase_tables(max(cfg.max_mem, 1), cfg.d_model, cfg.rope_base)
            self.register_buffer("age_cos", c)                 # [max_mem, d/2]
            self.register_buffer("age_sin", s)

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
    def toprows_sel_idx(self, seg_tok: torch.Tensor) -> torch.Tensor:
        """Les POSITIONS des top_k tokens SIF, dans l'ordre du segment.

        Extrait de `toprows_sel` sans en changer une opération : `toprows`
        n'avait besoin que des tokens, `tophid` a besoin de savoir OÙ ils sont
        pour aller chercher l'état caché correspondant.
        """
        c = self.cfg
        st = seg_tok.to(self.embed.weight.device).reshape(-1)
        if st.numel() == 0:
            return st.new_empty(0, dtype=torch.long)
        w = self.sif_w[st].float()
        return torch.topk(w, min(c.top_k, st.numel())).indices.sort().values

    @torch.no_grad()
    def toprows_sel(self, seg_tok: torch.Tensor) -> torch.Tensor:
        """Les top_k tokens du segment au poids SIF le plus fort, DANS L'ORDRE
        DU SEGMENT (le tri par poids détruirait l'ordre). Aucun privilège : le
        write ne sait pas où est la valeur, il ne connaît que la table unigram.
        """
        st = seg_tok.to(self.embed.weight.device).reshape(-1)
        if st.numel() == 0:
            return st
        return st[self.toprows_sel_idx(seg_tok)]

    @torch.no_grad()
    def seg_hidden(self, seg_tok: torch.Tensor) -> torch.Tensor:
        """[T, d] — les états du segment APRÈS `norm_f`, banque DÉBRANCHÉE.

        C'est la ligne que le design `[max_mem, mem_dim, d_model]` stocke :
        exactement le tenseur que `lm_head` consomme (le forward rend `x`
        post-`norm_f` via `return_hidden`), donc exactement l'espace où le
        tying fait de `ligne @ Eᵀ` un biais pointeur.

        `bank=None` : le write ne lit pas la banque pour l'écrire. Sans ça le
        contenu d'un slot dépendrait des slots déjà présents et l'ordre
        d'écriture deviendrait un facteur caché du contenu.

        COÛT ASSUMÉ : un forward de plus par write. `toprows` lisait
        `embed.weight`, une table — ici il faut faire tourner le modèle. C'est
        un coût du HARNAIS jouet (l'env construit les banques hors du forward
        d'entraînement), pas du design : au 350M le write poole les états du
        tour que le forward vient déjà de calculer.
        """
        st = seg_tok.to(self.embed.weight.device).reshape(1, -1)
        if self.cfg.code != "midhid":
            _, h = self.forward(st, bank=None, bank_mask=None,
                                return_hidden=True)
            return h[0].float().detach()
        # ── phase 10 (`midhid`) : on s'arrête à hid_tap_layers blocs ────────
        # Le flux résiduel BRUT, avant `norm_f` : c'est là que vivent les
        # « conclusions » du segment, avant la rotation vocabulaire que le
        # tying impose au dernier étage. Les trois lignes ci-dessous sont
        # EXACTEMENT le préambule de `forward` sans banque ni injection (x =
        # embed(ids) puis la boucle de blocs) — le self-test 21b le vérifie
        # contre `forward(..., return_hidden=True)` à hid_tap = 1.0.
        x = self.embed(st)
        for blk in self.blocks[:self.cfg.hid_tap_layers]:
            x = blk(x, None, None, None)
        return x[0].float().detach()

    @torch.no_grad()
    def toprows_sel_fixed(self, seg_tok: torch.Tensor) -> torch.Tensor:
        """Les tokens sélectionnés, TOUJOURS top_k — la version « injection ».

        `toprows_sel` en rend min(top_k, |seg|) : un segment porteur plus court
        que top_k (mesuré : les segs vont de 12 à 26 tokens, donc ça arrive dès
        top_k = 13) donnait un groupe COURT, et empiler des plans de longueurs
        différentes dans un batch plantait au premier pas (job 58).

        Complétion : le DERNIER token est répété — exactement la convention du
        chemin banque (`toprows_rows`, « TAILLE FIXE : le groupe fait TOUJOURS
        1 + top_k lignes »). Un candidat en double est inerte.
        """
        t = self.toprows_sel(seg_tok)
        k = self.cfg.top_k
        n = int(t.numel())
        if n == k:
            return t
        if n == 0:                    # segment vide (jamais observé)
            return torch.zeros(k, dtype=torch.long,
                               device=self.embed.weight.device)
        return torch.cat([t, t[-1].repeat(k - n)])

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
        idx = self.toprows_sel_idx(seg_tok)
        n = int(idx.numel())
        if n:
            if c.code in HID_CODES:
                # LA SEULE DIFFÉRENCE DE L'A/B : l'état post-`norm_f` de la
                # position, au lieu de l'embedding d'entrée du token qui s'y
                # trouve. `rms_unit` est appliqué des DEUX côtés — sans quoi on
                # comparerait aussi deux échelles (‖h‖/‖E‖ ≈ ×17 au 350M) et le
                # bras perdrait pour une raison qui n'est pas la sienne.
                e = rms_unit(self.seg_hidden(seg_tok)[idx])    # [n, d], RMS 1
            else:
                st = seg_tok.to(dev).reshape(-1)
                e = rms_unit(self.embed.weight[st[idx]].float())  # [n, d], RMS 1
            if c.row_pos_tag:
                e = rms_unit(e + self.row_pos[:n].float() * c.oracle_ka_scale)
            out[1:1 + n] = e
            if n < c.top_k:            # complète le groupe (cf. docstring)
                out[1 + n:] = e[-1]
        return out.detach()

    @torch.no_grad()
    def tophid_rows_fixed(self, seg_tok: torch.Tensor) -> torch.Tensor:
        """[top_k, d] — les lignes de CONTENU à injecter, taille fixe.

        Le pendant vectoriel de `toprows_sel_fixed` : r4/r5 y transportaient des
        ID de tokens (le préfixe faisait `embed(inj)`), `tophid` transporte les
        lignes elles-mêmes puisqu'aucune table ne peut les reconstruire. Même
        convention de complétion : la dernière ligne est répétée.
        """
        k, d = self.cfg.top_k, self.cfg.d_model
        dev = self.embed.weight.device
        idx = self.toprows_sel_idx(seg_tok)
        n = int(idx.numel())
        if n == 0:                     # segment vide (jamais observé)
            return torch.zeros(k, d, device=dev)
        e = rms_unit(self.seg_hidden(seg_tok)[idx])
        if n < k:
            e = torch.cat([e, e[-1].expand(k - n, d)])
        return e[:k].detach()

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
    def forward(self, ids, bank=None, bank_mask=None, inject=None,
                return_hidden=False, inject_age=None, bank_age=None):
        """`inject` [B, k] (UN groupe, r4) ou [B, G, k] (G groupes, r5) : les
        tokens des groupes toprows injectés en PRÉFIXE de pseudo-tokens.

        Layout (spec) : chaque groupe pose ses k tokens PUIS un séparateur, donc
        le préfixe fait G·(k+1) positions RoPE contiguës (0..G·(k+1)−1), et le
        tour RÉEL démarre une position PLUS LOIN — la position G·(k+1) reste
        VIDE, trou délibéré qui marque la frontière. À G=1 c'est exactement le
        layout de r4 : injecté 0..k−1, séparateur k, tour réel à k+2.
        L'ordre des groupes est celui que l'appelant donne (r5 : score
        DÉCROISSANT du retriever).

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
        mem = None
        if self.cfg.age_rope and bank_age is not None and bank is not None \
                and bank.size(1) > 0:
            # PHASE 10, pendant BANQUE de la rotation d'âge : la matrice plate
            # (max_mem·m, d) est rotée ligne à ligne AVANT que le moindre read
            # ne la touche — fast-weight, cross-attn et readout voient tous la
            # MÊME banque rotée. La banque STOCKÉE, elle, n'est jamais modifiée
            # (invariant : le write est oracle et immuable).
            ag = bank_age.to(bank.device).long().clamp(
                0, self.age_cos.shape[0] - 1)                  # [B,M]
            bank = rot_pairs(bank, self.age_cos[ag].to(bank.dtype),
                             self.age_sin[ag].to(bank.dtype))
        if inject is not None:
            assert self.cfg.variant in INJECT_VARIANTS, self.cfg.variant
            B, T = ids.shape
            # `tophid` transporte des LIGNES ([B,(G,)k,d]), les autres des ID
            # ([B,(G,)k]) : le dispatch se fait sur le CODE, jamais sur le rang
            # du tenseur — à G=1 un [B,k,d] de lignes et un [B,G,k] d'ID ont le
            # même rang, et deviner ferait passer des embeddings pour des index.
            hid = self.cfg.code in HID_CODES
            want = 4 if hid else 3
            inj = inject if inject.dim() == want else inject.unsqueeze(1)
            G, k = inj.shape[1], inj.shape[2]
            pre = ((inj.to(x.dtype) * self.hid_scale) if hid
                   else self.embed(inj))                      # [B,G,k,d]
            if self.cfg.age_rope and inject_age is not None:
                # ROTATION-PUIS-APLATISSEMENT : chaque ligne du groupe g est
                # tournée de rot(θ·âge_g) AVANT d'être posée à plat dans le
                # préfixe. Toutes les lignes d'un groupe partagent l'âge de
                # leur slot (c'est une propriété du SLOT, pas de la ligne), et
                # la rotation est appliquée AVANT le vecteur de type — le type
                # dit « je suis une ligne de banque », l'âge dit « d'où je
                # viens » : les deux ne doivent pas se mélanger.
                ag = inject_age.to(ids.device).long().clamp_(
                    0, self.age_cos.shape[0] - 1)              # [B,G]
                ac = self.age_cos[ag].to(pre.dtype)[:, :, None]   # [B,G,1,d/2]
                asn = self.age_sin[ag].to(pre.dtype)[:, :, None]
                pre = rot_pairs(pre, ac, asn)
            pre = pre + self.inject_type                      # [B,G,k,d]
            if self.cfg.read_path == "kv":
                # ── LECTURE β : VUE PLATE + ATTENTION ───────────────────────
                # Pas de préfixe, pas de séparateur, pas de position RoPE : le
                # tenseur (G, k, d) est simplement APLATI en (G·k, d) et posé
                # aux K/V des couches lectrices. L'ordre des lignes n'a plus
                # aucune conséquence géométrique — ce qui porte la provenance,
                # c'est la rotation d'âge, et rien d'autre.
                mem = pre.reshape(B, G * k, -1)
            else:
                sep = self.embed(torch.full((B, G, 1),
                                            int(self.cfg.inject_sep_id),
                                            dtype=torch.long,
                                            device=ids.device))
                # [groupe0 | sép | groupe1 | sép | …] puis le tour, un cran plus
                # loin
                x = torch.cat([torch.cat([pre, sep], dim=2).reshape(
                    B, G * (k + 1), -1), x], dim=1)
                npre = G * (k + 1)
                pos = torch.cat([torch.arange(npre, device=ids.device),
                                 torch.arange(T, device=ids.device) + npre + 1])
        for blk in self.blocks:
            x = blk(x, bank, bank_mask, pos, mem)
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
        # r5 : l'état caché sert de QUERY au retriever (pris sur le segment
        # USER, cf. Retriever) — on le rend à la demande plutôt que de refaire
        # une passe de trunk.
        return (logits, x) if return_hidden else logits

    # ── décodage greedy (sans cache : préfixes courts) ──────────────────────
    @torch.no_grad()
    def greedy(self, prefix, bank, bank_mask, max_new: int, stop_id: int,
               inject=None, inject_age=None, bank_age=None):
        ids = prefix
        out = []
        for _ in range(max_new):
            lg = self.forward(ids[:, -self.cfg.max_seq_len:], bank, bank_mask,
                              inject=inject, inject_age=inject_age,
                              bank_age=bank_age)
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

# ── phase 10 : la tâche de CONDITIONNEMENT (règle en banque) ────────────────
#
# LA QUESTION. La phase 9 a mesuré la CITATION depuis des lignes d'états cachés
# injectées (0.434 from-scratch contre 0.708 pour l'embedding natif). Ce que
# personne n'a mesuré, c'est le CONDITIONNEMENT : des lignes d'états cachés
# injectées comme pseudo-tokens modulent-elles la DISTRIBUTION du modèle sans
# qu'il ait rien à citer ? C'est le test « une seule lecture, tout par le
# préfixe » de la spec §2.4 — s'il passe, le fast-weight read devient optionnel
# et la banque est une modalité au sens plein.
#
# LE DESIGN. Une vie-règle énonce UNE fois, tôt, une consigne de REGISTRE
# (« garde chaque réponse dans le registre {v} ») ; le seg de règle est le seg
# porteur, donc c'est LUI que le write oracle met en banque. Du filler pousse
# la règle hors de la fenêtre — et comme le labo forwarde SEG PAR SEG, la
# fenêtre du tour conditionné ne contient RIEN d'autre que le tour lui-même :
# la règle ne peut venir que de la banque. Les tours conditionnés posent
# ensuite une question quelconque et la réponse est rendue DANS le registre :
# même corps de phrase pour tous les registres, seuls changent un MARQUEUR
# d'ouverture et un marqueur de clôture.
#
# POURQUOI CE N'EST PAS DE LA CITATION. Le nom du registre n'apparaît JAMAIS
# dans la réponse. Le seul chemin de la banque à la loss passe par la
# distribution sur deux mots de template. Un modèle qui « copie » la ligne ne
# gagne rien ; un modèle qui la LIT gagne les marqueurs.
#
# LA MESURE. Paires contrastives : nll(réponse rendue dans le registre VRAI)
# contre nll(la MÊME réponse, même corps, même sujet, rendue dans un AUTRE
# registre). Le Δ ne peut venir que de la banque. Comme chaque registre est
# tiré uniformément et son concurrent aussi, la difficulté intrinsèque des
# gabarits se compense en espérance sur le jeu — et la borne basse `none`
# (aucune mémoire) mesure ce qu'il en reste.
COND_RULE_SLOT = "color"      # slot d'ACCUEIL de la règle : ses valeurs ont
                              # déjà un id dans `fact_id_maps()`, donc le write
                              # oracle et `pack_key[slot, attr]` marchent sans
                              # toucher aux maps d'identité (invariant du labo :
                              # une seule source, persona_chat_data).
COND_REGISTERS = ("crimson", "turquoise", "vermilion", "chartreuse")
# La clé du groupe est `pack_key[color, 0]` — IDENTIQUE pour les quatre
# registres. L'identité du registre ne vit donc QUE dans les lignes de contenu
# (les états cachés du seg de règle) : c'est bien le canal testé, pas la clé.
COND_MARK = {                 # (marqueur d'OUVERTURE, marqueur de CLÔTURE)
    "crimson":    ("Absolutely", "Cheers"),
    "turquoise":  ("Indeed", "Regards"),
    "vermilion":  ("Certainly", "Thanks"),
    "chartreuse": ("Naturally", "Onward"),
}
COND_RULE_TMPL = [
    "Style note: for this session, keep every reply in the {v} register.",
    # ASCII STRICT : le self-test hermétique tokenise 1 char = 1 ord() sur un
    # vocab de 512, un tiret cadratin (8212) le ferait sortir de la table.
    "One request: from here on, answer me in the {v} register please.",
    "Let us switch registers: use the {v} register for the rest of this chat.",
]
COND_Q_TMPL = ["Tell me about {c}.", "What do you make of {c}?",
               "Say something about {c}, would you?"]
# corps de réponse PARTAGÉ par les quatre registres : à index j fixé, deux
# registres produisent exactement la même phrase entre les deux marqueurs.
COND_TOPICS = ["gardening", "pottery", "birdwatching", "sailing", "baking",
               "astronomy", "cycling", "knitting", "origami", "geology",
               "beekeeping", "fencing", "calligraphy", "kayaking", "botany"]
COND_BODY = [
    "{c} is worth a little of our time.",
    "Let us look at {c} together.",
    "{c} comes up rather often around here.",
    "There is plenty to say about {c}.",
    "{c} makes for a good topic today.",
]


class PersonaRuleStream(PersonaChatStream):
    """PersonaChatStream + un troisième genre de conv : la VIE-RÈGLE.

    Les vies de rappel restent dans le flux telles quelles : la citation de la
    phase 9 (`evaluate`) continue d'être mesurée sur le MÊME run que le
    conditionnement (`evaluate_cond`). Le seul ajout est `p_rule`.

    Les valeurs de registre sont RETIRÉES du pool `color` : sans ça le même mot
    serait tantôt une règle de style, tantôt une valeur à citer, et les deux
    canaux se contamineraient.

    `cond_decoys` faits LEURRES (slots ≠ color) accompagnent la règle et sont
    écrits eux aussi : le préfixe injecté fait 1 + cond_decoys groupes, l'ordre
    des writes est tiré au sort, donc la règle n'est pas toujours la plus
    récente — c'est ce qui donne au code d'ÂGE (`code.age_rope`) quelque chose
    à coder, et au modèle quelque chose à sélectionner.
    """

    def __init__(self, tok, *, p_rule: float = 0.35, cond_decoys: int = 1,
                 cond_fillers: tuple = (2, 5), **kw) -> None:
        super().__init__(tok, **kw)
        self.p_rule = float(p_rule)
        self.cond_decoys = int(cond_decoys)
        self.cond_fillers = tuple(int(v) for v in cond_fillers)
        assert 0.0 <= self.p_rule <= 1.0, self.p_rule
        if COND_RULE_SLOT in self.slots:
            st, qs, ans, upd, pool = self.slots[COND_RULE_SLOT]
            sub = [v for v in pool if v not in COND_REGISTERS]
            if len(sub) >= 4:
                self.slots[COND_RULE_SLOT] = (st, qs, ans, upd, sub)
            else:                      # sous-pool trop maigre : slot retiré
                del self.slots[COND_RULE_SLOT]

    # ── pièces ──────────────────────────────────────────────────────────────
    def cond_answer(self, reg: str, j: int, topic: str) -> dict:
        """Réponse conditionnée + `cond_mask` sur les DEUX marqueurs.

        `cond_mask` isole les seuls tokens que le registre décide : le Δnll
        restreint à ce masque est la mesure NETTE (le corps de phrase, lui, est
        identique entre les deux membres de la paire et ne fait que diluer).
        """
        pre, suf = COND_MARK[reg]
        seg = self._assistant(f"{pre}. {COND_BODY[j].format(c=topic)} {suf}.")
        ids = seg["input_ids"][0].tolist()
        m = torch.zeros(len(ids))
        for w in (pre, suf):
            sp = self._val_span(ids, w)
            if sp is not None:
                m[sp[0]:sp[1]] = 1.0
        seg["cond_mask"] = m.unsqueeze(0)
        return seg

    def _conv_rule(self) -> dict:
        reg = self.rng.choice(COND_REGISTERS)
        rule = self._user_valued(self.rng.choice(COND_RULE_TMPL).format(v=reg),
                                 reg, slot=COND_RULE_SLOT)
        writes = [rule]
        used_slots, used_vals = {COND_RULE_SLOT}, {reg}
        for _ in range(self.cond_decoys):
            f = self._sample_fact(used_slots, used_vals)
            used_slots.add(f["slot"])
            used_vals.add(f["v"])
            writes.append(self._user_valued(
                self.rng.choice(f["st"]).format(v=f["v"], p=f["p"]), f["v"],
                slot=f["slot"], p=f["p"]))
        self.rng.shuffle(writes)
        rule_at = next(i for i, s in enumerate(writes) if s is rule)
        segs = list(writes)
        for _ in range(self.rng.randint(*self.cond_fillers)):
            segs += self._filler_pair()
        turns, alts = [], []
        for _ in range(self.rng.randint(*self.n_queries)):
            topic = self.rng.choice(COND_TOPICS)
            j = self.rng.randrange(len(COND_BODY))
            alt = self.rng.choice([r for r in COND_REGISTERS if r != reg])
            segs.append(self._user(self.rng.choice(COND_Q_TMPL).format(c=topic)))
            segs.append(self.cond_answer(reg, j, topic))
            turns.append(len(segs) - 1)
            alts.append(self.cond_answer(alt, j, topic))
        return {"kind": "rule", "segs": segs,
                # `truths` VIDE : `evaluate` (la citation) saute ces convs,
                # elles ne polluent aucune métrique de la phase 9.
                "info": {"truths": [], "queries": [], "ages": [],
                         "cond": {"reg": reg, "rule_at": rule_at,
                                  "turns": turns, "alts": alts}}}

    def next_conv(self) -> dict:
        r = self.rng.random()
        if r < self.p_smalltalk:
            return self._conv_smalltalk()
        if r < self.p_smalltalk + self.p_rule:
            return self._conv_rule()
        return self._conv_recall()


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
        plan = self.retr_plan(model, conv)
        out = {i: e["res"][e["gidx"]][2] for i, e in plan.items()
               if e["gidx"] is not None}
        return out, sum(1 for e in plan.values() if e["gidx"] is None)

    def cond_plan(self, model: ToyReadLM, conv: dict) -> dict:
        """PHASE 10 — {index du seg de RÉPONSE conditionnée → (lignes, âges)}.

        `lignes` : [G, top_k, d] pour les codes d'états cachés (`tophid`/
        `midhid`), [G, top_k] d'ID pour `toprows` — exactement les tenseurs que
        `ToyReadLM.forward` sait poser en préfixe multi-groupes.
        `âges` : [G] long, l'âge EN WRITES de chaque groupe (0 = le write le
        plus récent). C'est ce que `code.age_rope` rote.

        G est CONSTANT sur un run (1 + cfg.cond_decoys, cf. PersonaRuleStream),
        ce qui permet d'empiler les plans d'un lot sans regrouper par taille.
        Aucune sélection ici : TOUS les groupes résidents sont injectés, la
        règle parmi les leurres. Le bras ne s'offre donc AUCUN privilège de
        retrieval — l'oracle ne fait que rejouer le FIFO.

        Rend {} pour toute conv qui n'est pas une vie-règle : le chemin des
        phases 6-9 n'est pas touché.
        """
        cond = (conv.get("info") or {}).get("cond")
        if not cond:
            return {}
        turns = set(cond["turns"])
        hid = model.cfg.code in HID_CODES
        out, fifo, w = {}, [], 0
        for i, seg in enumerate(conv["segs"]):
            if i in turns and fifo:
                # âge 0 = le write LE PLUS RÉCENT (w−1), pas « zéro write
                # écoulé » : la table de rotation est indexée par le RANG de
                # récence, l'identité rot(0) revenant au plus frais.
                out[i] = (torch.stack([r for r, _ in fifo]),
                          torch.tensor([w - 1 - t for _, t in fifo],
                                       dtype=torch.long))
            f = self.fact_of(seg)
            if f is not None:
                st = self.seg_tokens(seg)
                fifo.append(((model.tophid_rows_fixed(st) if hid
                              else model.toprows_sel_fixed(st)), w))
                w += 1
                fifo = fifo[-self.max_mem:]
        return out

    def retr_plan(self, model: ToyReadLM, conv: dict) -> dict:
        """L'ÉTAT DE BANQUE vu par chaque segment de RÉPONSE gradé.

        {index de seg → {"uidx": index du segment USER qui pose la question,
                         "res": [(slot, attr, tokens)] des groupes RÉSIDENTS,
                         "gidx": index du groupe PORTEUR dans `res` — le write
                                 le PLUS RÉCENT du slot interrogé, celui dont
                                 la valeur est la vérité — ou None si le fait
                                 est déjà sorti du FIFO,
                         "pos":  TOUS les index de `res` portant ce slot}}

        POURQUOI `pos` : la clé d'un groupe est `pack_key[slot, attr]`, donc
        deux writes du MÊME slot (fait MIS À JOUR, p_update=0.15) ont des clés
        RIGOUREUSEMENT IDENTIQUES. Une CE mono-cible sur `gidx` serait alors
        non seulement inapprenable mais à gradient EXACTEMENT NUL (mesuré : les
        deux clés à cos 1.0 se compensent). La CE est donc MULTI-POSITIVE — elle
        pousse la masse sur l'ENSEMBLE des groupes du bon slot — et c'est la
        RÉCENCE, pas le score, qui départage à l'injection (cf. evaluate).

        C'est la source unique du FIFO rejoué : `inject_plan` (r4) n'en garde
        que les tokens du groupe porteur, et r5 s'en sert pour SCORER les clés
        résidentes (le retriever) puis injecter ses top-k.

        On ne matérialise PAS les lignes de la banque : la ligne-clé d'un groupe
        est exactement `model.pack_key[slot, attr]` (cf. toprows_rows, qui la
        pose telle quelle) et ses lignes de contenu sont exactement les tokens
        retenus. Aucun module ne lit le CONTENU de la banque en r4/r5 — le
        matérialiser serait un tenseur mort de 8×14×512.
        """
        truths = (conv.get("info") or {}).get("truths") or []
        q_slots = (conv.get("info") or {}).get("q_slots") or []
        a_idx = [i for i, s in enumerate(conv["segs"])
                 if s["role"] == "assistant"]
        graded = a_idx[-len(truths):] if truths else []
        qpos = {ix: qi for qi, ix in enumerate(graded)}
        plan: dict = {}
        fifo: list = []                # [(slot_id, attr_id, tokens)] résidents
        last_user = 0
        for i, seg in enumerate(conv["segs"]):
            if seg["role"] == "user":
                last_user = i
            qi = qpos.get(i)
            if qi is not None:
                sl = self.slot_ids.get(q_slots[qi]) if qi < len(q_slots) else None
                pos = [g for g, r in enumerate(fifo) if r[0] == sl]
                plan[i] = {"uidx": last_user, "res": list(fifo),
                           # le PLUS RÉCENT porte la vérité (les précédents sont
                           # des versions périmées du même slot)
                           "gidx": pos[-1] if pos else None, "pos": pos}
            f = self.fact_of(seg)
            if f is not None:
                # TAILLE FIXE : les plans d'un batch s'empilent, ils doivent
                # tous faire top_k (cf. toprows_sel_fixed).
                fifo.append((f[0], f[1],
                             model.tophid_rows_fixed(self.seg_tokens(seg))
                             if model.cfg.code in HID_CODES else
                             model.toprows_sel_fixed(self.seg_tokens(seg))))
                fifo = fifo[-self.max_mem:]
        return plan

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


def pad_bank_age(banks: list, device, group_rows: int = 1):
    """[B, M] long — l'ÂGE EN WRITES de la ligne m de la lane b.

    PHASE 10, pendant côté BANQUE de `cond_plan` : 0 = le write le PLUS
    RÉCENT, 1 = celui d'avant, etc. Toutes les lignes d'un même groupe
    partagent l'âge de leur slot (c'est une propriété du slot, pas de la
    ligne). Les lignes de PADDING prennent l'âge 0 — elles sont déjà inertes
    par le masque, la valeur n'a aucun effet.

    `GroupBank.groups` porte les tailles réelles ; une banque ordinaire (une
    ligne par write) retombe sur group_rows = 1 sans cas particulier.
    """
    M = max((len(b) for b in banks), default=0)
    out = torch.zeros(len(banks), max(M, 1), dtype=torch.long, device=device)
    for i, b in enumerate(banks):
        sizes = list(getattr(b, "groups", [group_rows] * (len(b) // max(
            group_rows, 1)))) or []
        ng = len(sizes)
        pos = 0
        for g, sz in enumerate(sizes):
            out[i, pos:pos + sz] = ng - 1 - g
            pos += sz
    return out


def bank_ages_for(model, banks: list, device):
    """[B, M] des âges, ou None si le bras ne rote rien (chemin par défaut)."""
    if not model.cfg.age_rope:
        return None
    return pad_bank_age(banks, device, model.cfg.group_rows)


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


# ── r5 : plomberie du retriever ──────────────────────────────────────────────

def query_hidden(model, segs, device, max_len):
    """h_query [n, d] : l'état caché du DERNIER token de chaque segment USER.

    Le forward est fait SANS banque ni injection : c'est la question seule qui
    doit décider quel groupe aller chercher. Et c'est le segment USER — jamais
    la réponse : en teacher-forcing, un état pris dans la réponse contiendrait
    déjà la valeur, et le retriever apprendrait à lire ce qu'il est censé
    retrouver.
    """
    X, _ = pad_segs(segs, device, max_len)
    lens = [min(int(s["input_ids"][0].numel()), max_len) for s in segs]
    _, h = model(X, None, None, return_hidden=True)
    idx = torch.tensor([n - 1 for n in lens], device=device)
    return h[torch.arange(len(segs), device=device), idx]


def train_group_pick(entry, n_groups: int, order: str = "random") -> list:
    """Index des groupes injectés devant UNE réponse À L'ENTRAÎNEMENT.

    L'ORACLE + (n_groups−1) DISTRACTEURS tirés au hasard parmi les autres
    groupes résidents. S'il n'y a pas assez de voisins, G est réduit pour cette
    réponse — jamais complété en répétant l'oracle : le préfixe multi-groupes
    accepte déjà des tailles variables, et un doublon apprendrait au modèle que
    deux groupes identiques sont une situation normale.

    `order='random'` : la place de l'oracle est tirée au sort, pour que le
    modèle apprenne à TROUVER le bon groupe plutôt qu'une position fixe.
    `order='oracle_first'` : l'oracle est toujours en tête — ce qui reproduit
    exactement l'ordre de l'éval (tri par score, le vrai groupe premier).

    Le tirage passe par le RNG GLOBAL de torch : à graine de training fixée, la
    séquence est reproductible, comme tout le reste du lab.
    """
    res_n = len(entry["res"])
    g = entry["gidx"]
    n = min(max(int(n_groups), 1), res_n)
    others = [x for x in range(res_n) if x != g]
    pick = [g]
    if n > 1 and others:
        idx = torch.randperm(len(others))[:n - 1]
        pick += [others[int(x)] for x in idx]
    if order == "random" and len(pick) > 1:
        pick = [pick[int(x)] for x in torch.randperm(len(pick))]
    return pick


def retr_scores(model, h, entries):
    """(scores [n, G], masque [n, G]) sur les groupes RÉSIDENTS de chaque
    entrée. Les banques de tailles différentes sont padées à droite et masquées
    à −inf. La clé d'un groupe est `pack_key[slot, attr]` — exactement la ligne
    0 que la banque matérialisée porterait (cf. retr_plan)."""
    n = len(entries)
    G = max(max(len(e["res"]) for e in entries), 1)
    keys = torch.zeros(n, G, model.cfg.d_model, device=h.device, dtype=h.dtype)
    mask = torch.zeros(n, G, dtype=torch.bool, device=h.device)
    for i, e in enumerate(entries):
        for g, (sl, at, _) in enumerate(e["res"]):
            keys[i, g] = model.pack_key[sl, at].to(h.dtype)
            mask[i, g] = True
    return model.retr(h, keys, mask), mask


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
    r5 = cfg.variant in RETRIEVER_VARIANTS
    # r5 : l'ÉTAT DE BANQUE par réponse (clés résidentes + index du vrai
    # groupe) ; r4 : seulement les tokens du groupe oracle.
    rplans = [env.retr_plan(model, c) for c in convs] if r5 else None
    plans = ([{i: e["res"][e["gidx"]][2] for i, e in p.items()
               if e["gidx"] is not None} for p in rplans] if r5
             else [env.inject_plan(model, c)[0] for c in convs] if r4 else None)
    # ── PHASE 10 : les tours CONDITIONNÉS reçoivent leur préfixe ────────────
    # `cond_plan` rend {} sur toute conv qui n'est pas une vie-règle, donc les
    # runs des phases 6-9 ne voient RIEN de plus. `ages` voyage à côté des
    # lignes (c'est lui que `code.age_rope` rote au moment de l'aplatissement).
    ages: list = [{} for _ in convs]
    if r4 and cfg.cond:
        cps = [env.cond_plan(model, c) for c in convs]
        if cfg.cond_arm == "shuffle":
            # CONTRÔLE BANQUE MÉLANGÉE, version ENTRAÎNEMENT : la vie i reçoit
            # le préfixe de la vie i+1 du lot. Les registres sont tirés
            # indépendamment, donc le préfixe est faux ~3 fois sur 4 et ne
            # porte JAMAIS d'information sur la règle de la vie courante.
            cps = [cps[(i + 1) % len(cps)] for i in range(len(cps))]
        if cfg.cond_arm != "none":
            for i, cp in enumerate(cps):
                for jj, (rows, ag) in cp.items():
                    # `shuffle` peut proposer un index de seg que la vie i n'a
                    # pas : on ne pose un préfixe QUE sur ses propres tours.
                    if jj in (conv_turns := set(
                            ((convs[i].get("info") or {}).get("cond")
                             or {}).get("turns") or [])):
                        plans[i][jj] = rows
                        ages[i][jj] = ag
                    del conv_turns
    # normalisation de la CE auxiliaire : une MOYENNE sur les réponses
    # supervisées du pas, pour qu'elle ne dépende pas du nombre de segs (la
    # loss LM est elle aussi normalisée par le total de tokens du pas).
    n_tgt = (sum(1 for p in rplans for e in p.values()
                 if e["gidx"] is not None) if r5 else 0)
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
        # Le drapeau « ce sous-lot est injecté » VOYAGE AVEC le sous-lot : le
        # relire d'après convs[sub[0]] ferait dépendre TOUT le lot de la
        # première conv (KeyError si elle seule n'a pas de plan à j, injections
        # perdues en silence dans l'autre sens).
        if r5:
            # ORACLE + DISTRACTEURS : le préfixe d'entraînement a le même
            # nombre de groupes que celui de l'éval. Comme G varie d'une
            # réponse à l'autre (banques plus ou moins remplies), les lanes
            # sont regroupées PAR TAILLE DE PRÉFIXE — un forward exige un
            # préfixe uniforme.
            picks = {}
            for i in lanes:
                e = rplans[i].get(j)
                if e is not None and e["gidx"] is not None:
                    picks[i] = train_group_pick(e, cfg.retr_train_groups,
                                                cfg.retr_train_order)
            bysize: dict = {}
            for i, p in picks.items():
                bysize.setdefault(len(p), []).append(i)
            subsets = [(v, True) for _, v in sorted(bysize.items())]
            rest = [i for i in lanes if i not in picks]
            if rest:
                subsets.append((rest, False))
        elif r4:
            # Un forward exige un préfixe UNIFORME. Sans la phase 10 tous les
            # plans font top_k et le regroupement rend EXACTEMENT les deux
            # sous-lots historiques ; avec elle, les tours conditionnés
            # apportent des plans [G, k(, d)] qui doivent voyager à part.
            byshape: dict = {}
            for i in lanes:
                pj = plans[i].get(j)
                if pj is not None:
                    byshape.setdefault(tuple(pj.shape), []).append(i)
            subsets = [(v, True) for _, v in sorted(byshape.items())]
            subsets.append(([i for i in lanes if j not in plans[i]], False))
        else:
            subsets = [(lanes, False)]
        for sub, has_inj in subsets:
            if not sub:
                continue
            segs = [convs[i]["segs"][j] for i in sub]
            X, W = pad_segs(segs, device, max_len)
            inj = None
            if has_inj and r5:
                inj = torch.stack([
                    torch.stack([rplans[i][j]["res"][g][2] for g in picks[i]])
                    for i in sub]).to(device)          # [n, G, k]
                if inj.shape[1] == 1:
                    # G=1 : on retombe sur le tenseur [n, k] de la v1, donc sur
                    # un forward BIT-À-BIT identique (cf. self-test 20e).
                    inj = inj[:, 0]
            elif has_inj:
                # toutes de MÊME longueur (top_k) : cf. toprows_sel_fixed
                inj = torch.stack([plans[i][j] for i in sub]).to(device)
            # PHASE 10 : les âges du préfixe voyagent AVEC le sous-lot. Ils
            # n'existent que pour les tours conditionnés (les réponses de
            # rappel des ph.7-9 n'ont pas d'âge ⇒ None ⇒ chemin inchangé).
            iage = None
            if has_inj and all(j in ages[i] for i in sub):
                iage = torch.stack([ages[i][j] for i in sub]).to(device)
            sub_banks = [banks[i] for i in sub]
            bank, bmask = pad_bank(sub_banks, device)
            bage = bank_ages_for(model, sub_banks, device)
            with torch.autocast(device.split(":")[0], dtype=torch.bfloat16,
                                enabled=amp):
                logits = model(X, bank, bmask, inject=inj, inject_age=iage,
                               bank_age=bage)
            s, n = seg_ce(logits, X, W)
            obj = s / total_w * scale_by if float(n) > 0 else None
            if obj is not None and ent_c > 0 and \
                    model.ptr.last_pos_ent is not None:
                # pousse la porte-position vers un choix DUR d'une ligne
                obj = obj + ent_c * model.ptr.last_pos_ent
            if r5 and has_inj and cfg.retr_ce > 0:
                # ── CE AUXILIAIRE : le retriever apprend en SUPERVISÉ ───────
                # (l'env connaît le groupe porteur), jamais par le canal LM.
                ent = [rplans[i][j] for i in sub]
                hq = query_hidden(model, [convs[i]["segs"][e["uidx"]]
                                          for i, e in zip(sub, ent)],
                                  device, max_len)
                if cfg.retr_detach:
                    hq = hq.detach()      # W_q = le SEUL apprenant de ce canal
                sc, _ = retr_scores(model, hq, ent)
                # CE MULTI-POSITIVE : −log Σ_{g ∈ positifs} p_g. Les writes
                # successifs d'un MÊME slot ont la même clé (cf. retr_plan) —
                # une cible unique y aurait un gradient exactement nul.
                sc = sc.float()
                pmask = torch.zeros_like(sc, dtype=torch.bool)
                for r, e in enumerate(ent):
                    pmask[r, e["pos"]] = True
                aux = (torch.logsumexp(sc, -1)
                       - torch.logsumexp(sc.masked_fill(~pmask, float("-inf")),
                                         -1)).sum()
                aux = cfg.retr_ce * aux / max(n_tgt, 1)
                obj = aux if obj is None else obj + aux
            if obj is not None:
                obj.backward()
            if float(n) > 0:
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
    r5 = model.cfg.variant in RETRIEVER_VARIANTS
    stream.rng = random.Random(seed)
    live_ans, abl_ans, truths_all, groups = [], [], [], []
    resident = []                        # fait encore en banque ? (aligné)
    hit_sel = []                         # r5 : le VRAI groupe est-il injecté ?
    r_at1 = r_at2 = r_den = 0            # r5 : recall@1 / recall@2 du retriever
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
        rplan = env.retr_plan(model, conv) if r5 else {}
        plan = ({} if r5 else env.inject_plan(model, conv)[0]) if r4 else {}
        for i, seg in enumerate(conv["segs"]):
            X = seg["input_ids"][:, :max_len].to(device)
            W = seg["loss_mask"][:, :max_len].to(device)
            b, bm = pad_bank([bank], device)
            bage = bank_ages_for(model, [bank], device)
            if i in graded:
                inj = None
                sel_ok = None
                if r5:
                    # ── SÉLECTION APPRISE : top-k DUR du retriever ──────────
                    # La query est la QUESTION (segment user), pas la réponse.
                    e = rplan.get(i)
                    if e is not None and e["res"]:
                        hq = query_hidden(model,
                                          [conv["segs"][e["uidx"]]],
                                          device, max_len)
                        sc, _ = retr_scores(model, hq, [e])
                        nk = min(model.cfg.retr_topk, len(e["res"]))
                        # tri STABLE sur les groupes pris du PLUS RÉCENT au plus
                        # ancien : à score ÉGAL — le cas EXACT de deux writes du
                        # même slot, qui partagent leur clé — c'est le plus
                        # récent qui passe devant, et c'est lui qui porte la
                        # vérité. La récence est le seul discriminant que la clé
                        # ne donne pas.
                        rec = torch.arange(len(e["res"]) - 1, -1, -1,
                                           device=sc.device)
                        order = torch.argsort(-sc[0][rec], stable=True)
                        top = [int(rec[o]) for o in order[:nk]]
                        # ordre = score DÉCROISSANT
                        inj = torch.stack([e["res"][g][2] for g in top]
                                          )[None].to(device)
                        if e["gidx"] is None:
                            n_absent += 1          # fait déjà sorti du FIFO
                        else:
                            r_den += 1
                            r_at1 += int(top[0] == e["gidx"])
                            r_at2 += int(e["gidx"] in top)
                            sel_ok = e["gidx"] in top
                elif r4:
                    tk = plan.get(i)
                    if tk is None:
                        n_absent += 1     # fait ÉVINCÉ : aucune injection
                    else:
                        inj = tk[None].to(device)
                with torch.autocast(device.split(":")[0], dtype=torch.bfloat16,
                                    enabled=amp):
                    lg_live = model(X, b, bm, inject=inj, bank_age=bage)
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
                                               inject=inj, bank_age=bage))
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
                hit_sel.append(sel_ok)
                if r5:
                    # « résident » en r5 = le VRAI groupe a été injecté (la
                    # sélection a réussi) — c'est la condition sous laquelle
                    # r4 avait mesuré 0.708.
                    resident.append(bool(sel_ok))
                elif r4:
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
    # ── r5 : LA DÉCOMPOSITION sélection vs copie ────────────────────────────
    # recall@k = le retriever a-t-il mis le vrai groupe dans ce qu'il injecte ;
    # grade|raté = ce que le modèle rend quand on lui a injecté les MAUVAIS
    # groupes (doit tomber au niveau ablaté, sinon il répond au prior).
    out["retr_r1"] = (r_at1 / r_den) if r_den else float("nan")
    out["retr_r2"] = (r_at2 / r_den) if r_den else float("nan")
    out["n_retr"] = r_den
    midx = [i for i, x in enumerate(hit_sel) if x is False]
    out["n_miss"] = len(midx)
    out["grade_miss"] = (
        grade_recall([live_ans[i] for i in midx],
                     [truths_all[i] for i in midx]) if midx else float("nan"))
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


@torch.no_grad()
def evaluate_cond(model, env, stream, seed, n_convs, device, max_len, amp):
    """PHASE 10 — le Δnll CONTRASTIF de conditionnement, en 3 lectures.

    Pour chaque tour conditionné on tient DEUX rendus de la même réponse (même
    corps, même sujet, même index de gabarit) : celui du registre VRAI et celui
    d'un AUTRE registre. On mesure nll(incohérent) − nll(cohérent) sous trois
    conditions de lecture, sur le MÊME modèle :

      live  la banque / le préfixe de CETTE vie          (le bras)
      shuf  ceux d'une AUTRE vie, de registre DIFFÉRENT  (contrôle d'artefact)
      none  aucune mémoire                               (borne basse)

    Un Δ positif en `live` qui S'EFFONDRE en `shuf` et en `none` est la seule
    signature qui prouve que le conditionnement passe PAR LA BANQUE. Un Δ
    positif partout mesurerait la difficulté intrinsèque des gabarits ; un Δ
    positif en `shuf` mesurerait une fuite du harnais.

    Deux échelles, toujours rendues ensemble :
      `dnll`  moyenne PAR TOKEN supervisé (dilué : le corps de phrase est
              identique entre les deux membres de la paire, il ne peut que
              rapprocher les deux nll)
      `mark`  restreint aux tokens de MARQUEUR (`cond_mask`) — les seuls que
              le registre décide. C'est la mesure NETTE.
      `acc`   2AFC : fraction des tours où nll(cohérent) < nll(incohérent) sur
              les marqueurs. Hasard = 0.500, lisible sans échelle.
    """
    model.eval()
    inj_arm = model.cfg.variant in INJECT_VARIANTS
    stream.rng = random.Random(seed)
    items: list = []
    guard = 0
    while len({id(x["conv"]) for x in items}) < n_convs and \
            guard < n_convs * 20:
        guard += 1
        conv = stream.next_conv()
        cond = (conv.get("info") or {}).get("cond")
        if not cond:
            continue
        cplan = env.cond_plan(model, conv) if inj_arm else {}
        bank: list = []
        turns = {t: q for q, t in enumerate(cond["turns"])}
        for i, seg in enumerate(conv["segs"]):
            q = turns.get(i)
            if q is not None:
                st = (("inj",) + cplan.get(i, (None, None)) if inj_arm
                      else ("bank", list(bank)))
                items.append({"conv": conv, "reg": cond["reg"], "state": st,
                              "coh": seg, "inc": cond["alts"][q]})
            if not inj_arm:
                bank = env.write(model, bank, seg)
    # ── appariement du contrôle MÉLANGÉ ────────────────────────────────────
    # Chaque item emprunte l'état d'un item de REGISTRE DIFFÉRENT (le premier
    # qui suit cycliquement). Un état de même registre ne serait pas un
    # contrôle : il porterait la bonne règle.
    shuf = []
    n = len(items)
    for i in range(n):
        j = next((k % n for k in range(i + 1, i + n)
                  if items[k % n]["reg"] != items[i]["reg"]), None)
        shuf.append(None if j is None else items[j]["state"])

    def _read(state):
        """(bank, bank_mask, inject, inject_age, bank_age) d'un état de
        lecture."""
        if state is None or state[0] is None:
            return None, None, None, None, None
        if state[0] == "bank":
            b, bm = pad_bank([state[1]], device)
            return b, bm, None, None, bank_ages_for(model, [state[1]], device)
        rows, ages = state[1], state[2]
        if rows is None:
            return None, None, None, None, None
        return None, None, rows[None].to(device), ages[None].to(device), None

    def _nll(seg, state):
        b, bm, inj, ag, bag = _read(state)
        X = seg["input_ids"][:, :max_len].to(device)
        W = seg["loss_mask"][:, :max_len].to(device)
        Mk = seg["cond_mask"][:, :max_len].to(device) * (W > 0).float()
        with torch.autocast(device.split(":")[0], dtype=torch.bfloat16,
                            enabled=amp):
            lg = model(X, b, bm, inject=inj, inject_age=ag, bank_age=bag)
        s_all, w_all = seg_ce(lg, X, W)
        s_mk, w_mk = seg_ce(lg, X, Mk)
        return (float(s_all), float(w_all), float(s_mk), float(w_mk))

    acc: dict = {}
    for cond_name in ("live", "shuf", "none"):
        num = den = mnum = mden = 0.0
        hit = tot = 0
        for i, it in enumerate(items):
            st = (it["state"] if cond_name == "live"
                  else shuf[i] if cond_name == "shuf" else None)
            if cond_name == "shuf" and st is None:
                continue
            ca, cw, cm, cmw = _nll(it["coh"], st)
            ia, iw, im, imw = _nll(it["inc"], st)
            if cw > 0 and iw > 0:
                num += ia / iw - ca / cw
                den += 1.0
            if cmw > 0 and imw > 0:
                mnum += im / imw - cm / cmw
                mden += 1.0
                hit += int(cm / cmw < im / imw)
                tot += 1
        acc[f"dnll_{cond_name}"] = num / den if den else float("nan")
        acc[f"mark_{cond_name}"] = mnum / mden if mden else float("nan")
        acc[f"acc_{cond_name}"] = hit / tot if tot else float("nan")
    acc["n"] = len(items)
    acc["n_convs"] = len({id(x["conv"]) for x in items})
    model.train()
    return acc


# ── nom de run (⇒ save_dir) ──────────────────────────────────────────────────

# ── PHASE 10 : nommage DÉTERMINISTE de la grille §2.4 ────────────────────────
# Un combo = un dossier, et le NOM du dossier se relit comme le combo. C'est ce
# qui rend les 36 runs agrégeables sans grepper un seul log.
GRID_READ = {("r0", "entry"): "seqfw",     ("r3", "entry"): "bankxattn",
             ("r4", "entry"): "injentry",  ("r4", "kv"): "kvappend",
             ("r5", "entry"): "r5entry",   ("r5", "kv"): "r5kv"}
GRID_TAP = {"toprows": "native", "tophid": "postnorm", "midhid": "mid",
            "mean": "pooled"}


def grid_name(cfg: ToyCfg) -> str:
    """`read-<mode>_rot-<on|off>_tap-<prov>_m<k>` (+ suffixe de bras).

    Le seed n'entre PAS dans le nom : la grille tourne à seed FIXE, un run par
    combo. Un balayage de graine, s'il vient, ajoutera son propre suffixe.
    """
    name = (f"read-{GRID_READ.get((cfg.variant, cfg.read_path), cfg.variant)}"
            f"_rot-{'on' if cfg.age_rope else 'off'}"
            f"_tap-{GRID_TAP.get(cfg.code, cfg.code)}"
            f"_m{cfg.top_k}")
    if cfg.cond_arm != ToyCfg.cond_arm:
        name += f"_arm-{cfg.cond_arm}"
    if cfg.cond_decoys != ToyCfg.cond_decoys:
        name += f"_dec{cfg.cond_decoys}"
    if cfg.write_mode != ToyCfg.write_mode:
        name += f"_w{cfg.write_mode}"
    return name


def grid_combos(reads=("seq_fw", "inject_entry", "kv_append"),
                rots=(False, True), taps=("postnorm", "mid"),
                ms=(1, 4, 8)) -> list:
    """La GRILLE COMPLÈTE, en clair. Un dict par combo, dans un ordre stable."""
    out = []
    for rd in reads:
        for tp in taps:
            for m in ms:
                for rot in rots:
                    out.append({"read": rd, "age_rot": bool(rot), "tap": tp,
                                "m": int(m)})
    return out


def _grid_cfg(combo: dict, base: dict) -> ToyCfg:
    v, c, rp = READ_MODES[combo["read"]]
    code = {"native": "toprows", "postnorm": "tophid",
            "mid": "midhid"}[combo["tap"]] if c is None else c
    return ToyCfg(**{**base, "variant": v, "code": code, "read_path": rp,
                     "age_rope": combo["age_rot"], "top_k": int(combo["m"]),
                     "cond": True})


def print_grid_manifest(config: str, base: dict, save_root: str,
                        fmt: str = "tsv", b_convs: int = 8) -> None:
    """Le MANIFESTE : une ligne par combo, commande exacte + coût estimé.

    Colonnes : run | read | rot | tap | m | lignes_lues | commande | note.
    `lignes_lues` = ce que le bras traverse VRAIMENT par forward — c'est le
    seul proxy honnête du coût, et il n'a pas la même unité selon le chemin :
      seq_fw     M = max_mem·(1+m) sous-slots, en boucle SÉQUENTIELLE et PAR
                 COUCHE : le coût est LINÉAIRE en M et il domine tout le reste.
      injentry   G·(1+m)+G positions de préfixe AJOUTÉES à la séquence :
                 l'attention est quadratique dessus.
      kvappend   G·m clés/valeurs de plus, sur les couches lectrices seulement,
                 sans allonger la séquence des queries.
    """
    rows = []
    for combo in grid_combos():
        cfg = _grid_cfg(combo, base)
        m = combo["m"]
        G = cfg.cond_groups
        if cfg.variant == "r0":
            load = cfg.max_mem * cfg.group_rows
            unit = "sous-slots seq"
        elif cfg.read_path == "kv":
            load = G * m
            unit = "K/V add."
        else:
            load = G * (m + 1)
            unit = "pos. préfixe"
        cmd = (f"python -m deepseek_v4_mini.toy_read_lab {config} --cond "
               f"--read {combo['read']} --tap {combo['tap']} --m {m}"
               + (" --age-rot" if combo["age_rot"] else ""))
        # COÛT : mesuré, pas deviné. `params` vient de param_report aux dims
        # de la config ; `rel` est le rapport de temps par pas RELEVÉ sur CPU à
        # dims RÉELLES (d512/L6, batch_convs=2, 2 pas) — c'est le RATIO entre
        # bras qui transfère au GPU, pas la seconde absolue.
        mod = ToyReadLM(cfg, 11, 12, sif_w=(torch.ones(cfg.vocab_size)
                                            if cfg.code in SIF_CODES else None))
        pr = param_report(mod)
        del mod
        # états AdamW fp32 (poids + grad + 2 moments) = 16 octets par paramètre
        opt_go = pr["total"] * 16 / 2 ** 30
        # terme DOMINANT d'activation : les logits [batch, T, V] en fp32, et
        # leur gradient. C'est lui qui décide, pas les poids.
        act_go = (b_convs * cfg.max_seq_len * cfg.vocab_size * 4 * 3) / 2 ** 30
        vram = opt_go + act_go
        rel = {"seq_fw": 2.0, "inject_entry": 1.0, "kv_append": 1.07}[
            combo["read"]] * (1.0 + 0.03 * (m - 1))
        note = ("boucle fast-weight de %d itérations × %d couches : le bras "
                "LENT (×2.0 relevé)" % (load, cfg.n_layers)
                if combo["read"] == "seq_fw" else "")
        if vram > 8.0:
            note = (note + " | ⚠️ NE TIENT PAS EN 8 Go "
                    f"({vram:.1f} Go estimés) — à lancer ailleurs, PAS à "
                    f"raboter en silence").strip(" |")
        rows.append({"run": grid_name(cfg), "read": combo["read"],
                     "rot": "on" if combo["age_rot"] else "off",
                     "tap": combo["tap"], "m": m,
                     "load": f"{load} {unit}",
                     "params_M": f"{pr['total']/1e6:.1f}",
                     "vram_Go_est": f"{vram:.2f}",
                     "cout_rel": f"{rel:.2f}",
                     "save_dir": os.path.join(save_root, grid_name(cfg)),
                     "cmd": cmd, "note": note})
    if fmt == "json":
        import json
        print(json.dumps(rows, indent=2, ensure_ascii=False))
        return
    cols = ["run", "read", "rot", "tap", "m", "load", "params_M",
            "vram_Go_est", "cout_rel", "save_dir", "cmd", "note"]
    print("\t".join(cols))
    for r in rows:
        print("\t".join(str(r[c]) for c in cols))


def run_name_for(cfg: ToyCfg) -> str:
    """Nom de dossier d'un run. INVARIANT : deux configs qui n'entraînent PAS
    la même chose ne doivent JAMAIS partager un nom — un run fini ne se fait pas
    écraser par un bras voisin.

    phase 1 → <variant>/ ; phase 2 → <variant>_<code>/ ; puis un suffixe par
    knob qui s'écarte de ce que porte déjà le dossier nu.

    ⚠️ `retr_topk` manquait à l'appel : `--retr-topk 1` retombait sur
    `r5_toprows` et aurait écrasé le run de référence (rattrapé avant
    déploiement). L'ancre de `retr_train_groups` n'est PAS le défaut de la
    dataclass mais 1 : le dossier `r5_toprows` existant contient un run entraîné
    à UN groupe injecté, donc tout entraînement à plusieurs groupes doit sortir
    ailleurs, y compris quand c'est devenu le défaut.
    """
    if cfg.cond:
        # PHASE 10 : le dossier se relit comme le combo de la grille §2.4.
        return grid_name(cfg)
    name = cfg.variant if cfg.code == "mean" else f"{cfg.variant}_{cfg.code}"
    if cfg.pos_offset:
        name += f"_o{cfg.pos_offset}"
    if cfg.code in PACK_CODES and cfg.pack_blocks != ToyCfg.pack_blocks:
        name += f"_b{cfg.pack_blocks}"         # sweep de partition = run à part
    if cfg.code in GROUP_CODES:
        if cfg.top_k != ToyCfg.top_k:
            name += f"_k{cfg.top_k}"           # sweep de k = run à part
        if not cfg.row_pos_tag:
            name += "_notag"
    if cfg.readout_mix != ToyCfg.readout_mix:
        name += f"_{cfg.readout_mix}"          # bras MoS = run à part
    if cfg.pos_entropy:
        name += f"_ent{cfg.pos_entropy:g}"
    if cfg.variant in RETRIEVER_VARIANTS:
        if cfg.retr_topk != ToyCfg.retr_topk:
            name += f"_topk{cfg.retr_topk}"
        if cfg.retr_train_groups != 1:         # ancre = le run r5_toprows fini
            name += f"_tg{cfg.retr_train_groups}"
        if cfg.retr_train_order != ToyCfg.retr_train_order:
            name += f"_{cfg.retr_train_order}"
        if cfg.retr_ce != ToyCfg.retr_ce:
            name += f"_ce{cfg.retr_ce:g}"
        if not cfg.retr_detach:
            name += "_nodetach"
    if cfg.write_mode == "every":
        name += "_wev"
    # ── phase 10 : chaque axe de la grille §2.4 entre dans le nom ───────────
    if cfg.read_path != ToyCfg.read_path:
        name += f"_{cfg.read_path}"
    if cfg.age_rope:
        name += "_age"
    if cfg.code == "midhid" and cfg.hid_tap != ToyCfg.hid_tap:
        name += f"_tap{cfg.hid_tap:g}"
    if cfg.cond:
        name += "_cond"
        if cfg.cond_arm != ToyCfg.cond_arm:
            name += f"_{cfg.cond_arm}"
        if cfg.cond_decoys != ToyCfg.cond_decoys:
            name += f"_dec{cfg.cond_decoys}"
    return name


# ── plomberie ────────────────────────────────────────────────────────────────

def build_tokenizer(name):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    add = [x for x in ("<think>", "<blank>") if x not in tok.get_vocab()]
    if add:
        tok.add_special_tokens({"additional_special_tokens": add})
    return tok


def persona_kwargs(raw, split, smoke, cond: bool = False):
    gen = dict((raw.get("persona") or {}).get("gen") or {})
    if cond:
        # kwargs PROPRES à PersonaRuleStream (bloc `cond:` du YAML). Ils ne
        # sont ajoutés QUE si la phase 10 est active : PersonaChatStream ne les
        # connaît pas et lèverait.
        gen.update((raw.get("cond") or {}).get("gen") or {})
    else:
        for k in ("p_rule", "cond_decoys", "cond_fillers"):
            gen.pop(k, None)
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
    ap.add_argument("--retr-ce", type=float, default=None, dest="retr_ce",
                    help="r5 : coefficient de la CE auxiliaire du retriever")
    ap.add_argument("--retr-topk", type=int, default=None, dest="retr_topk",
                    help="r5 : nombre de groupes injectés À L'ÉVAL (top-k dur)")
    ap.add_argument("--retr-train-groups", type=int, default=None,
                    dest="retr_train_groups",
                    help="r5 : groupes injectés À L'ENTRAÎNEMENT (oracle + "
                         "distracteurs) ; 0 = suivre --retr-topk, 1 = v1")
    ap.add_argument("--retr-train-order", choices=("random", "oracle_first"),
                    default=None, dest="retr_train_order",
                    help="r5 : place de l'oracle dans le préfixe d'entraînement")
    ap.add_argument("--final-eval-convs", type=int, default=None,
                    dest="final_eval_convs",
                    help="surcharge training.final_eval_convs (passe d'éval "
                         "élargie en fin de run ; 0 = désactivée)")
    # ── PHASE 10 : les quatre axes de la grille de la spec §2.4 ─────────────
    ap.add_argument("--read", choices=tuple(READ_MODES), default=None,
                    help="CHEMIN DE LECTURE (axe 1) : seq_fw = fast-weight "
                         "séquentiel (r0, impose --code mean — confond de "
                         "format DÉCLARÉ) ; bank_xattn = lecture apprise sur "
                         "la MÊME banque de groupes (r3) ; inject_entry = "
                         "pseudo-tokens en préfixe (r4, lecture α) ; "
                         "kv_append = lignes appondues aux K/V des couches "
                         "lectrices, sans ré-encodage (lecture β). Surcharge "
                         "--variant et code.read_path.")
    ap.add_argument("--age-rot", action="store_true", dest="age_rot",
                    help="AXE 2 : rote chaque ligne injectée par l'ÂGE en "
                         "writes de son slot (binding DFT, table max_mem). "
                         "Élémentaire, aucune matmul dédiée.")
    ap.add_argument("--tap", choices=("native", "postnorm", "mid"),
                    default=None,
                    help="AXE 3 — PROVENANCE de la ligne : native = "
                         "embedding d'entrée (--code toprows, ph.6-8) ; "
                         "postnorm = état après norm_f (--code tophid, ph.9) ; "
                         "mid = état à code.hid_tap de profondeur (--code "
                         "midhid, défaut 2/3). Surcharge --code.")
    ap.add_argument("--m", type=int, default=None, dest="m_rows",
                    help="AXE 4 : LIGNES par write (alias de --top-k / "
                         "code.top_k). C'est le cadran de budget de la spec "
                         "§2.4 une fois la largeur fixée à d_model.")
    ap.add_argument("--cond", action="store_true",
                    help="PHASE 10 : ajoute les vies-RÈGLE au flux et l'éval "
                         "CONTRASTIVE de conditionnement (les vies de rappel "
                         "restent, donc la citation de la ph.9 est mesurée sur "
                         "le même run)")
    ap.add_argument("--cond-arm", choices=COND_ARMS, default=None,
                    dest="cond_arm",
                    help="ce que l'injection dépose À L'ENTRAÎNEMENT devant un "
                         "tour conditionné : true (défaut) | shuffle (banque "
                         "MÉLANGÉE, from-scratch) | none (BORNE BASSE)")
    ap.add_argument("--cond-decoys", type=int, default=None,
                    dest="cond_decoys",
                    help="faits LEURRES plantés à côté de la règle (le préfixe "
                         "fait 1+d groupes). 0 ⇒ tous les âges valent 0 et "
                         "--age-rot devient un no-op mesurable.")
    ap.add_argument("--cond-eval-convs", type=int, default=None,
                    dest="cond_eval_convs",
                    help="vies-règle gradées par passe d'éval contrastive")
    ap.add_argument("--manifest", choices=("tsv", "json"), default=None,
                    help="n'entraîne RIEN : imprime le MANIFESTE de la grille "
                         "§2.4 (36 combos, commande exacte, save_dir et coût "
                         "estimé par combo) et sort. Exige la config.")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)

    if a.selftest:
        _selftest()
        return

    assert a.config, "config YAML requise (ou --selftest)"
    # ── PHASE 10 : les axes-raccourcis se traduisent AVANT toute validation ─
    read_path = "entry"
    if a.tap is not None:
        a.code = {"native": "toprows", "postnorm": "tophid",
                  "mid": "midhid"}[a.tap]
    if a.read is not None:
        v, c, read_path = READ_MODES[a.read]
        a.variant = v
        if c is not None:
            a.code = c
        elif a.tap is None and a.code == "mean":
            # un chemin de lecture sans provenance explicite : le défaut de la
            # grille est la provenance de la phase 9 (post-norm_f).
            a.code = "tophid"
    if a.code != "mean" and a.variant not in ("r3",) + INJECT_VARIANTS and \
            not (a.variant == "r0" and a.code in GROUP_CODES):
        raise SystemExit(
            f"--code {a.code} n'est supporté QUE par --variant r3 (banque en "
            f"espace d'embedding + pointer nu), les variantes d'injection "
            f"{INJECT_VARIANTS}, et — phase 10 — r0 sur les codes de GROUPES "
            f"({GROUP_CODES}, la matrice plate lue ligne à ligne par le "
            f"fast-weight, spec §2.4(b)). r1/r2 restent le contrôle de la "
            f"phase 1 : lance-les sans --code (mean).")
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
    for key, cast in (("retr_ce", float), ("retr_topk", int),
                      ("retr_detach", bool), ("retr_train_groups", int),
                      ("retr_train_order", str)):
        if key in cb:
            mc[key] = cast(cb[key])
    if a.retr_ce is not None:              # la CLI gagne sur le YAML
        mc["retr_ce"] = float(a.retr_ce)
    if a.retr_topk is not None:
        mc["retr_topk"] = int(a.retr_topk)
    if a.retr_train_groups is not None:
        mc["retr_train_groups"] = int(a.retr_train_groups)
    if a.retr_train_order is not None:
        mc["retr_train_order"] = a.retr_train_order
    if "pos_entropy" in cb:
        mc["pos_entropy"] = float(cb["pos_entropy"])
    # ── PHASE 10 : knobs YAML puis surcharges CLI ──────────────────────────
    for key, cast in (("hid_tap", float), ("age_rope", bool),
                      ("read_path", str), ("cond_arm", str),
                      ("cond_decoys", int)):
        if key in cb:
            mc[key] = cast(cb[key])
    if a.read is not None:
        mc["read_path"] = read_path
    if a.age_rot:
        mc["age_rope"] = True
    if a.m_rows is not None:
        mc["top_k"] = int(a.m_rows)
    if a.cond_arm is not None:
        mc["cond_arm"] = a.cond_arm
    if a.cond_decoys is not None:
        mc["cond_decoys"] = int(a.cond_decoys)
    mc["cond"] = bool(a.cond)
    if a.manifest:
        print_grid_manifest(
            a.config, {k: v for k, v in mc.items()
                       if k not in ("variant", "code", "read_path", "age_rope",
                                    "top_k", "cond")},
            t.get("save_dir", "./checkpoints/toy_read_lab"), a.manifest,
            b_convs=int(t.get("batch_convs", 8)))
        return
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
    # PHASE 10 : vies-règle gradées par passe contrastive (pas de décodage
    # greedy là-dedans — que des forwards teacher-forcés, donc c'est bon marché)
    cond_eval_convs = int(a.cond_eval_convs if a.cond_eval_convs is not None
                          else t.get("cond_eval_convs", 32))
    if a.smoke:
        mc.update(d_model=64, n_layers=2, n_heads=4, mem_dim=64, x_dim=0)
        steps, b_convs, eval_every, eval_convs, max_new = 2, 2, 1, 1, 8
        cond_eval_convs = 2

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
    # PHASE 10 : le stream gagne un troisième genre de conv, il ne perd rien.
    P = PersonaRuleStream if cfg.cond else chat_stream_class("persona")

    def pk(split, **over):
        return {**persona_kwargs(raw, split, a.smoke, cond=cfg.cond), **over}

    sif_w = None
    if cfg.code in SIF_CODES:
        # table SIF sur le split TRAIN (la vue du write). Le stream
        # d'entraînement reste surp OFF : le SIF n'entre QUE dans le code.
        sif_w = sif_weight_table(P, tok, pk("train"),
                                 cfg.vocab_size, cfg.sif_a,
                                 seed=int(t.get("seed", 0)))
    model = ToyReadLM(cfg, env.n_slots, env.n_attrs, sif_w=sif_w).to(device)

    run_name = run_name_for(cfg)
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
    if cfg.variant in RETRIEVER_VARIANTS:
        print(f"  RETRIEVER APPRIS (r5) : la sélection oracle de r4 est "
              f"remplacée par score_g = (W_q·h_query)·clé_g/√d — W_q "
              f"zéro-init, {cfg.d_model**2 + 1:,} params, SEUL module "
              f"appris du bras. h_query = état caché du DERNIER token du "
              f"SEGMENT USER (jamais la réponse : ce serait la fuite de "
              f"teacher-forcing). CE auxiliaire supervisée coef "
              f"{cfg.retr_ce:g}"
              + (" (h_query DÉTACHÉ : W_q apprend seul)" if cfg.retr_detach
                 else " (gradient RENVOYÉ dans le backbone)")
              + f". ENTRAÎNEMENT : {cfg.retr_train_groups} groupe(s) injecté(s) "
              f"— l'ORACLE + {cfg.retr_train_groups - 1} DISTRACTEUR(S) tirés "
              f"parmi les résidents, ordre {cfg.retr_train_order} (G réduit si "
              f"la banque est trop courte, JAMAIS complété par un doublon). "
              f"ÉVAL : top-{cfg.retr_topk} DUR du retriever, trié par score.",
              flush=True)
    if cfg.variant in RETRIEVER_VARIANTS and cfg.retr_train_groups > 1 and \
            cfg.retr_train_order == "random":
        print("  ⚠️ ORDRE ALÉATOIRE + PROMPT SANS QUESTION : le décodage gradé "
              "part de A_OPEN seul (la question n'est PAS dans le contexte, "
              "cf. evaluate/greedy). Avec plusieurs groupes en ordre "
              "aléatoire, RIEN n'identifie le bon groupe à l'inférence : le "
              "plafond attendu devient ~0.708/G au lieu de 0.708×recall@1. "
              "`code.retr_train_order: oracle_first` reproduit l'ordre de "
              "l'éval (le vrai groupe y sort premier, recall@1 mesuré 1.000) "
              "et lève la dégénérescence.", flush=True)
        print(f"  PRÉDICTION (inscrite avant le run) : plafond = 0.708 "
              f"(grade|vrai-groupe de r4, strate code) × recall@{cfg.retr_topk}. "
              f"L'oracle des clés sépare à 100 %, donc un recall@"
              f"{cfg.retr_topk} < 0.9 incrimine l'apprentissage de W_q, PAS la "
              f"géométrie.", flush=True)
    if cfg.variant in INJECT_VARIANTS:
        print(f"  INJECTION À SÉLECTION ORACLE : AUCUN module de read appris "
              f"(ni cross-attn, ni pointer) — le backbone NU lit un préfixe de "
              f"G×{cfg.top_k} pseudo-tokens (embeddings BRUTS, non normés, + "
              f"un vecteur de type appris), séparateur id {cfg.inject_sep_id} "
              f"après CHAQUE groupe, positions RoPE contiguës 0..G×"
              f"{cfg.top_k + 1}−1 puis tour réel un cran plus loin (trou "
              f"délibéré). "
              + (f"G = {cfg.retr_train_groups} à l'entraînement (oracle + "
                 f"distracteurs), {cfg.retr_topk} à l'éval."
                 if cfg.variant in RETRIEVER_VARIANTS else
                 "G = 1 (oracle) à l'entraînement et à l'éval.")
              + " ABLATÉ = le même tour SANS préfixe. "
              f"PRIVILÈGE DÉCLARÉ : la SÉLECTION du groupe est "
              + ("APPRISE à l'éval (r5) mais l'entraînement voit l'oracle"
                 if cfg.variant in RETRIEVER_VARIANTS else "l'oracle")
              + ", et l'injection est teacher-forcée à l'entraînement (aucun "
              "curriculum de copie in-context).", flush=True)
    if cfg.code in GROUP_CODES and cfg.readout_mix == "mos":
        print(f"  readout MoS : une distribution PAR LIGNE puis mixture "
              f"pondérée s·p (aucune superposition dans l'espace d'embedding, "
              f"donc aucun token hybride fabricable)"
              + (f" | pénalité d'entropie sur p : {cfg.pos_entropy:g}"
                 if cfg.pos_entropy else ""), flush=True)
    if cfg.cond or cfg.age_rope or cfg.read_path != "entry" or \
            cfg.code == "midhid":
        print(f"  PHASE 10 — grille §2.4 : lecture "
              + {("r0", "entry"): "seq_fw (fast-weight séquentiel, banque mean "
                                  "— CONFOND de format déclaré)",
                 ("r3", "entry"): "bank_xattn (cross-attn contenu + "
                                  "GroupReadout sur la banque de groupes)",
                 ("r4", "entry"): "inject_entry (pseudo-tokens en PRÉFIXE, α)",
                 ("r5", "entry"): "inject_entry + retriever appris",
                 ("r4", "kv"): "kv_append (lignes aux K/V des couches "
                               "lectrices, SANS ré-encodage ni RoPE, β)",
                 ("r5", "kv"): "kv_append + retriever appris",
                 }.get((cfg.variant, cfg.read_path), cfg.read_path)
              + f" | provenance "
              + {"toprows": "native (embedding d'entrée)",
                 "tophid": "postnorm (état après norm_f)",
                 "midhid": f"mid ({cfg.hid_tap_layers}/{cfg.n_layers} blocs, "
                           f"hid_tap {cfg.hid_tap:.3f}, AVANT norm_f)",
                 }.get(cfg.code, cfg.code)
              + f" | m = {cfg.top_k} ligne(s)/write | âge "
              + ("ROTÉ (DFT sur max_mem rangs de récence)" if cfg.age_rope
                 else "non codé"), flush=True)
    if cfg.cond:
        print(f"  CONDITIONNEMENT : {len(COND_REGISTERS)} registres "
              f"(clé de groupe IDENTIQUE pour tous ⇒ l'identité du registre "
              f"ne vit QUE dans les lignes de contenu), {cfg.cond_decoys} "
              f"leurre(s) ⇒ {cfg.cond_groups} groupes injectés, bras "
              f"d'entraînement `{cfg.cond_arm}`. Éval contrastive : "
              f"{cond_eval_convs} vies par palier, 3 lectures "
              f"(live / shuf / none). La règle n'est JAMAIS citable : son nom "
              f"n'apparaît dans aucune réponse.", flush=True)
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

    tr_stream = P(tok, seed=int(t.get("seed", 0)), **pk("train"))
    ev_stream = P(tok, seed=1234, **pk("eval"))
    tc_stream = P(tok, seed=4321, **pk("train"))
    # PHASE 10 : deux streams DÉDIÉS à l'éval contrastive (p_rule = 1.0 : on ne
    # paie pas de convs qu'`evaluate_cond` jetterait). Instances SÉPARÉES de
    # ev_stream/tc_stream, qui se font resemer par `evaluate`.
    cv_stream = (P(tok, seed=2468, **pk("eval", p_rule=1.0, p_smalltalk=0.0))
                 if cfg.cond else None)
    ct_stream = (P(tok, seed=8642, **pk("train", p_rule=1.0, p_smalltalk=0.0))
                 if cfg.cond else None)

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
            # `hid_scale` (phase 9) : le facteur que le bras tophid met sur la
            # ligne post-norm à l'injection. Il absorbe le confondant d'échelle
            # contre `embed` (×17 au 350M) et sa valeur EST une mesure — s'il
            # s'écarte franchement de 1, c'est que la ligne n'entrait pas dans
            # le stack à la bonne amplitude. Récupérable a posteriori dans le
            # state_dict, mais le voir bouger pendant le run vaut mieux.
            hs = getattr(model, "hid_scale", None)
            print(f"step {step:5d} | loss {loss:.4f} | gnorm {float(gn):.2f} "
                  f"| lr {base_lr*lr_at(step):.2e}"
                  + (f" | hid_scale {float(hs):.3f}" if hs is not None else "")
                  + f" | {time.time()-t0:.0f}s",
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
            cd = ct = None
            if cfg.cond:
                # ── PHASE 10 : Δnll CONTRASTIF, trois lectures ─────────────
                cd = evaluate_cond(model, env, cv_stream, 2468,
                                   cond_eval_convs, device, max_len, amp)
                ct = evaluate_cond(model, env, ct_stream, 8642,
                                   cond_eval_convs, device, max_len, amp)
                print(f"    COND held-out (n={cd['n']} tours / "
                      f"{cd['n_convs']} vies) Δnll marqueurs "
                      f"live {cd['mark_live']:+.4f} shuf {cd['mark_shuf']:+.4f} "
                      f"none {cd['mark_none']:+.4f} | 2AFC "
                      f"{cd['acc_live']:.3f}/{cd['acc_shuf']:.3f}/"
                      f"{cd['acc_none']:.3f} | Δnll tous tokens "
                      f"{cd['dnll_live']:+.4f}/{cd['dnll_shuf']:+.4f}/"
                      f"{cd['dnll_none']:+.4f}  || TRAIN marqueurs "
                      f"{ct['mark_live']:+.4f}/{ct['mark_shuf']:+.4f}/"
                      f"{ct['mark_none']:+.4f}", flush=True)
            print("    strates held-out : " + "  ".join(
                f"{g} {ev['grade_' + g]:.3f} (n={ev['n_' + g]})"
                for g in GROUPS) + f"  | porte pointer σ {ev['ptr_gate']:.4f}",
                flush=True)
            if cfg.variant in RETRIEVER_VARIANTS:
                # LA DÉCOMPOSITION du bras : sélection (recall) × copie
                # (grade | vrai groupe injecté). Le grade | RATÉ dit ce que le
                # modèle invente quand on lui injecte les mauvais groupes.
                print(f"    retriever : recall@1 {ev['retr_r1']:.3f} "
                      f"recall@{cfg.retr_topk} {ev['retr_r2']:.3f} "
                      f"(n={ev['n_retr']}) | grade | VRAI GROUPE injecté "
                      f"{ev['grade_resident']:.3f} (n={ev['n_resident']}) "
                      f"vs | RATÉ {ev['grade_miss']:.3f} "
                      f"(n={ev['n_miss']}) | plafond attendu "
                      f"{0.708 * ev['retr_r2']:.3f}", flush=True)
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
                               # colonnes de la PHASE 10 seulement sous --cond :
                               # le CSV des phases 1-9 reste octet-à-octet.
                               + ([f"cond_{p}_{c}" for p in
                                   ("mark", "dnll", "acc")
                                   for c in ("live", "shuf", "none")]
                                  + ["cond_n", "cond_mark_live_train"]
                                  if cfg.cond else [])
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
                           + ([_f(cd[f"{p}_{c}"]) for p in
                               ("mark", "dnll", "acc")
                               for c in ("live", "shuf", "none")]
                              + [cd["n"], _f(ct["mark_live"])]
                              if cfg.cond else [])
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
    fv = fc = None
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
        if cfg.variant in RETRIEVER_VARIANTS:
            print(f"    [final] retriever : recall@1 {fv['retr_r1']:.3f} "
                  f"recall@{cfg.retr_topk} {fv['retr_r2']:.3f} "
                  f"(n={fv['n_retr']}) | grade | VRAI GROUPE {fv['grade_resident']:.3f} "
                  f"(n={fv['n_resident']}) vs | RATÉ {fv['grade_miss']:.3f} "
                  f"(n={fv['n_miss']}) | plafond attendu "
                  f"{0.708 * fv['retr_r2']:.3f} (0.708 × recall@"
                  f"{cfg.retr_topk})", flush=True)
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
                    "n_resident", "n_absent", "age_evicted",
                    "retr_r1", "retr_r2", "n_retr", "grade_miss", "n_miss"] \
                + [c for g in GROUPS for c in
                   (f"grade_{g}", f"grade_{g}_abl", f"n_{g}")] + ["ptr_gate"]
            w.writerow(cols)

            def _g(x):
                return "" if x != x else f"{x:.4f}"
            w.writerow([final_eval_convs, _g(fv["grade_live"]), f"{se:.4f}",
                        _g(fv["grade_abl"]), _g(fv["dnll"]), fv["n"],
                        _g(fv["grade_resident"]), _g(fv["grade_resident_abl"]),
                        fv["n_resident"], fv["n_absent"],
                        _g(fv["age_evicted"]), _g(fv["retr_r1"]),
                        _g(fv["retr_r2"]), fv["n_retr"], _g(fv["grade_miss"]),
                        fv["n_miss"]]
                       + [v for g in GROUPS for v in
                          (_g(fv[f"grade_{g}"]), _g(fv[f"grade_{g}_abl"]),
                           fv[f"n_{g}"])] + [_g(fv["ptr_gate"])])
        print(f"  [final] écrit {fp}", flush=True)
    # ── PHASE 10 : ÉVAL CONTRASTIVE FINALE ÉLARGIE ─────────────────────────
    if cfg.cond and final_eval_convs > 0:
        fc = evaluate_cond(model, env, cv_stream, 2468, final_eval_convs,
                           device, max_len, amp)
        print(f"  [final] COND ({fc['n_convs']} vies, {fc['n']} tours) — Δnll "
              f"MARQUEURS live {fc['mark_live']:+.4f} | shuf "
              f"{fc['mark_shuf']:+.4f} | none {fc['mark_none']:+.4f} ; 2AFC "
              f"{fc['acc_live']:.3f} / {fc['acc_shuf']:.3f} / "
              f"{fc['acc_none']:.3f} (hasard 0.500) ; Δnll tous tokens "
              f"{fc['dnll_live']:+.4f} / {fc['dnll_shuf']:+.4f} / "
              f"{fc['dnll_none']:+.4f}", flush=True)
        print("    LECTURE DU VERDICT : live ≫ shuf ≈ none ⇒ le "
              "conditionnement passe PAR LA BANQUE (barreau 1 de l'échelle "
              "§2.4 franchi) ; live ≈ shuf ⇒ la sonde mesure un artefact ; "
              "live ≈ none ⇒ l'injection ne conditionne pas.", flush=True)
        fcp = os.path.join(save_dir, "cond_metrics.csv")
        with open(fcp, "w", newline="") as f:
            w = csv.writer(f)
            cols = [f"{p}_{c}" for p in ("mark", "dnll", "acc")
                    for c in ("live", "shuf", "none")] + ["n", "n_convs"]
            w.writerow(cols)
            w.writerow([f"{fc[c]:.5f}" if fc[c] == fc[c] else "" for c in
                        cols[:-2]] + [fc["n"], fc["n_convs"]])
        print(f"  [final] écrit {fcp}", flush=True)
    # ── RÉSULTAT PARSABLE : UN json par run, agrégeable sans grep ───────────
    # Il porte le COMBO (les quatre axes en clair) et les DEUX métriques du
    # run : conditionnement contrastif (avec ses contrôles shuf/none) et
    # citation (avec son bras ablaté). Écrit même si l'éval finale est coupée.
    import json
    res = {
        "run": run_name, "save_dir": save_dir, "steps": steps,
        "seed": int(t.get("seed", 0)), "device": device,
        "combo": {"read": next((k for k, v in READ_MODES.items()
                                if v[0] == cfg.variant
                                and v[2] == cfg.read_path), cfg.variant),
                  "age_rot": bool(cfg.age_rope),
                  "tap": GRID_TAP.get(cfg.code, cfg.code),
                  "m": cfg.top_k, "cond_arm": cfg.cond_arm,
                  "cond_decoys": cfg.cond_decoys,
                  "hid_tap_layers": cfg.hid_tap_layers,
                  "d_model": cfg.d_model, "n_layers": cfg.n_layers,
                  "max_mem": cfg.max_mem},
        "citation": ({"grade_live": fv["grade_live"],
                      "grade_abl": fv["grade_abl"], "dnll": fv["dnll"],
                      "grade_resident": fv["grade_resident"],
                      "n": fv["n"], "n_convs": final_eval_convs,
                      "strates": {g: fv[f"grade_{g}"] for g in GROUPS}}
                     if final_eval_convs > 0 else None),
        # Le CONDITIONNEMENT et ses DEUX contrôles, à plat : `live` est le
        # bras, `shuf` la banque MÉLANGÉE, `none` la borne SANS MÉMOIRE.
        "conditioning": (dict(fc) if cfg.cond and final_eval_convs > 0
                         else None),
    }
    def _clean(o):
        # NaN n'est PAS du JSON valide : une strate vide sort en `null`, pas en
        # un littéral que le moindre parseur strict refuserait. C'est le point
        # du fichier — être agrégeable sans précaution.
        if isinstance(o, dict):
            return {k: _clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_clean(v) for v in o]
        if isinstance(o, float) and o != o:
            return None
        return o

    rp = os.path.join(save_dir, "results.json")
    with open(rp, "w") as f:
        json.dump(_clean(res), f, indent=2, ensure_ascii=False, default=float)
    print(f"  écrit {rp} (résultat AGRÉGEABLE du combo)", flush=True)
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
    #     ⚠️ PHASE 10 : `toprows`/`tophid`/`midhid` sortent de cette liste pour
    #     r0 SEULEMENT — la banque de GROUPES est la matrice plate de la spec
    #     §2.4(b), que le hypernet fast-weight lit ligne à ligne (c'est le bras
    #     `--read seq_fw` de la grille). r1/r2 les refusent toujours, et r0 les
    #     refuse toujours pour tous les autres formats.
    for var in ("r0", "r1", "r2"):
        for code in ("phase", "segmean", "segphase", "segsif", "pack",
                     "segpack") + (() if var == "r0" else GROUP_CODES):
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
        assert tk.numel() == c_r4.top_k, (i, tk.numel())
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
    # 18h. RÉGRESSION job 58 — deux bugs payés au premier pas sur le rig :
    #   (i)  un seg porteur PLUS COURT que top_k rendait un plan court, et
    #        empiler [13] avec [12] plantait torch.stack ;
    #   (ii) le sous-lot injecté se relisait d'après la PREMIÈRE conv du lot.
    # Le test fait passer les deux par le VRAI train_step.
    m_r4b = ToyReadLM(ToyCfg(vocab_size=512, d_model=64, n_layers=2,
                             n_heads=4, mem_dim=64, variant="r4",
                             max_seq_len=256, code="toprows", seg_n_pos=8,
                             sif_a=A_SIF, top_k=6, inject_sep_id=5),
                      env.n_slots, env.n_attrs, sif_w=_sifw())
    short_seg = torch.tensor([11, 77, 200])          # 3 tokens < top_k 6
    raw_sel = m_r4b.toprows_sel(short_seg)
    fix_sel = m_r4b.toprows_sel_fixed(short_seg)
    assert raw_sel.numel() == 3 and fix_sel.numel() == 6
    assert torch.equal(fix_sel[:3], raw_sel) and \
        bool((fix_sel[3:] == raw_sel[-1]).all()), fix_sel
    # un seg DÉJÀ assez long n'est pas touché
    assert torch.equal(m_r4b.toprows_sel_fixed(seg20),
                       m_r4b.toprows_sel(seg20))
    # conv B : MÊMES segs mais AUCUN fait ⇒ aucun plan (le lot est MIXTE)
    convB = {"info": conv["info"], "segs": [dict(s) for s in conv["segs"]]}
    for s in convB["segs"]:
        s.pop("fact_slot", None)
    # conv C : TOUS les segs porteurs tronqués à 3 tokens ⇒ toute sélection est
    # courte, donc tout plan de cette conv DOIT avoir été complété.
    convC = {"info": conv["info"], "segs": [dict(s) for s in conv["segs"]]}
    for s in convC["segs"]:
        if OracleEnv.fact_of(s):
            for key in ("input_ids", "loss_mask", "attention_mask"):
                if key in s:
                    s[key] = s[key][:, :3]
    pA = env.inject_plan(m_r4b, conv)[0]
    pB = env.inject_plan(m_r4b, convB)[0]
    pC = env.inject_plan(m_r4b, convC)[0]
    assert pA and not pB, "le lot n'est pas mixte, le test ne prouve rien"
    assert all(t.numel() == 6 for t in pA.values())
    assert pC, "conv C sans plan : le cas du seg court n'est pas exercé"
    for t in pC.values():
        assert t.numel() == 6, ("plan d'un seg porteur court : la complétion "
                                "n'a pas eu lieu", t.numel())
        # ≤ 3 tokens réels complétés à 6 ⇒ la queue est forcément répétée
        assert int(t[-1]) == int(t[-2]), t
    # … et les trois traversent train_step ensemble SANS erreur (c'est le
    # crash du job 58), dans les DEUX ordres (le bug (ii) dépendait de qui
    # était en tête du lot).
    for order in ([conv, convB, convC], [convB, conv, convC]):
        m_r4b.zero_grad(set_to_none=True)
        lo = train_step(m_r4b, env, order, "cpu", 256, False, 1.0)
        assert lo == lo and lo > 0, lo               # ni NaN ni lot vide
    assert m_r4b.inject_type.grad is not None, \
        "aucun gradient n'a atteint l'injection : rien n'a été injecté"

    # ═══ PHASE 8 : r5, RETRIEVER APPRIS ═════════════════════════════════════
    torch.manual_seed(20260805)
    c_r5 = ToyCfg(vocab_size=512, d_model=64, n_layers=2, n_heads=4,
                  mem_dim=64, variant="r5", max_seq_len=256, code="toprows",
                  seg_n_pos=8, sif_a=A_SIF, top_k=4, inject_sep_id=5,
                  retr_topk=2)
    m_r5 = ToyReadLM(c_r5, env.n_slots, env.n_attrs, sif_w=_sifw())
    # 20a. le retriever est le SEUL module appris en plus (W_q zéro-init) et
    #      il n'y a toujours AUCUN read de contenu
    assert m_r5.ptr is None and all(b.read is None for b in m_r5.blocks)
    assert float(m_r5.retr.wq.weight.abs().max()) == 0.0, "W_q non zéro-init"
    assert sorted(n for n, _ in m_r5.named_parameters()
                  if "retr" in n or "inject" in n) == \
        ["inject_type", "retr.log_temp", "retr.wq.weight"]
    # 20b. LE PLAN : la query est un segment USER, ANTÉRIEUR à la réponse
    rp = env.retr_plan(m_r5, conv)
    assert rp, "aucune réponse gradée : le test ne prouve rien"
    for i, e in rp.items():
        assert conv["segs"][e["uidx"]]["role"] == "user", e["uidx"]
        assert e["uidx"] < i, (e["uidx"], i)
        if e["gidx"] is not None:
            assert 0 <= e["gidx"] < len(e["res"])
    # 20c. FUITE DE TEACHER-FORCING : changer les tokens de la RÉPONSE ne doit
    #      RIEN changer aux scores du retriever. C'est l'invariant critique.
    i0 = sorted(rp)[0]
    e0 = rp[i0]
    convL = {"info": conv["info"], "segs": [dict(s) for s in conv["segs"]]}
    convL["segs"][i0] = dict(conv["segs"][i0])
    convL["segs"][i0]["input_ids"] = torch.randint(
        0, 512, conv["segs"][i0]["input_ids"].shape)
    rpL = env.retr_plan(m_r5, convL)
    with torch.no_grad():
        m_r5.eval()
        h0 = query_hidden(m_r5, [conv["segs"][e0["uidx"]]], "cpu", 256)
        hL = query_hidden(m_r5, [convL["segs"][rpL[i0]["uidx"]]], "cpu", 256)
    assert torch.equal(h0, hL), \
        "h_query dépend de la RÉPONSE : fuite de teacher-forcing"
    # … et h_query dépend bien de la QUESTION (sinon la sonde serait vide)
    convQ = [dict(conv["segs"][e0["uidx"]])]
    convQ[0]["input_ids"] = torch.randint(
        0, 512, convQ[0]["input_ids"].shape)
    with torch.no_grad():
        hQ = query_hidden(m_r5, convQ, "cpu", 256)
    assert not torch.allclose(h0, hQ, atol=1e-6), \
        "h_query ne dépend pas de la question"
    # 20d. scores : une valeur par groupe RÉSIDENT, −inf sur le padding, et
    #      W_q zéro-init ⇒ scores tous nuls (softmax uniforme, CE = log G)
    ents = [e for e in rp.values() if e["res"]]
    if ents:
        with torch.no_grad():
            hh = query_hidden(m_r5, [conv["segs"][e["uidx"]] for e in ents],
                              "cpu", 256)
            sc, msk = retr_scores(m_r5, hh, ents)
        assert sc.shape == msk.shape and sc.shape[0] == len(ents)
        assert float(sc[msk].abs().max()) == 0.0, "W_q zéro-init : scores ≠ 0"
        assert bool(torch.isinf(sc[~msk]).all()) if (~msk).any() else True
        # top-k DUR : déterministe et TRIÉ par score décroissant
        e1 = ents[0]
        with torch.no_grad():
            m_r5.retr.wq.weight.normal_(0, 0.05)
            s1, _ = retr_scores(m_r5, hh[:1], [e1])
            s2, _ = retr_scores(m_r5, hh[:1], [e1])
        assert torch.equal(s1, s2), "scores non déterministes"
        nk = min(2, len(e1["res"]))
        t1 = s1[0].topk(nk).indices
        assert float(s1[0, t1[0]]) >= float(s1[0, t1[-1]]), "top-k non trié"
        assert torch.equal(t1, s2[0].topk(nk).indices)
        with torch.no_grad():
            m_r5.retr.wq.weight.zero_()
    # 20e. PRÉFIXE MULTI-GROUPES : [B,k] ≡ [B,1,k] BIT-À-BIT (donc r4 est
    #      inchangé par la généralisation), et 2 groupes changent la sortie.
    ids5 = torch.randint(0, 512, (2, 6))
    inj1 = torch.randint(0, 512, (2, 4))
    with torch.no_grad():
        m_r5.eval()
        o1 = m_r5(ids5, None, None, inject=inj1)
        o1b = m_r5(ids5, None, None, inject=inj1[:, None, :])
        o2 = m_r5(ids5, None, None,
                  inject=torch.stack([inj1, (inj1 + 1) % 512], dim=1))
    assert torch.equal(o1, o1b), "[B,k] ≢ [B,1,k] : r4 aurait bougé"
    assert o2.shape == o1.shape and not torch.allclose(o1, o2, atol=1e-5)
    # positions attendues : G·(k+1) contigus puis le tour un cran plus loin
    for G_ in (1, 2):
        npre_ = G_ * (4 + 1)
        pos_ = list(range(npre_)) + [npre_ + 1 + t for t in range(6)]
        assert pos_[npre_ - 1] == npre_ - 1 and pos_[npre_] == npre_ + 1, pos_
    # 20f. la CE auxiliaire ne fire QUE sur les réponses gradées
    convN = {"info": {"truths": [], "q_slots": []},
             "segs": [dict(s) for s in conv["segs"]]}
    assert not env.retr_plan(m_r5, convN), "conv sans vérité : plan non vide"
    m_r5.zero_grad(set_to_none=True)
    train_step(m_r5, env, [convN], "cpu", 256, False, 1.0)
    assert m_r5.retr.wq.weight.grad is None or \
        float(m_r5.retr.wq.weight.grad.abs().max()) == 0.0, \
        "la CE du retriever a fire sans réponse gradée"
    # … et elle fire bien dès qu'il y a un NÉGATIF à repousser. Il faut une
    # conv dont la banque contient, à la question, un groupe d'un AUTRE slot :
    # si tous les résidents portent le slot interrogé, la CE multi-positive
    # vaut 0 et un gradient nul est la BONNE réponse (rien à discriminer).
    st5 = PersonaChatStream(tok, seed=11)
    conv_neg = None
    for _ in range(80):
        cc = st5.next_conv()
        if not (cc["info"].get("truths") or []):
            continue
        pl = env.retr_plan(m_r5, cc)
        if any(e["gidx"] is not None and len(e["pos"]) < len(e["res"])
               for e in pl.values()):
            conv_neg = cc
            break
    assert conv_neg is not None, "pas de conv avec un négatif en banque"
    m_r5.zero_grad(set_to_none=True)
    lo5 = train_step(m_r5, env, [conv_neg, convN], "cpu", 256, False, 1.0)
    assert lo5 == lo5
    assert m_r5.retr.wq.weight.grad is not None and \
        float(m_r5.retr.wq.weight.grad.abs().max()) > 0, \
        "la CE du retriever n'a pas atteint W_q"
    # 20f-bis. CLÉS DUPLIQUÉES (fait MIS À JOUR : deux writes du même slot) —
    #      leurs clés sont RIGOUREUSEMENT identiques, donc aucun score ne peut
    #      les départager. Deux garanties : (i) la CE multi-positive les traite
    #      comme un bloc (gradient nul = rien à apprendre, PAS un bug) ;
    #      (ii) à score égal, l'injection prend le PLUS RÉCENT — celui qui porte
    #      la vérité.
    dup = [e for e in rp.values()
           if e["gidx"] is not None and len(e["pos"]) > 1]
    if dup:
        e_d = dup[0]
        ka = m_r5.pack_key[e_d["res"][e_d["pos"][0]][0],
                           e_d["res"][e_d["pos"][0]][1]]
        kb = m_r5.pack_key[e_d["res"][e_d["pos"][-1]][0],
                           e_d["res"][e_d["pos"][-1]][1]]
        assert torch.equal(ka, kb), "clés d'un même slot devenues différentes"
        assert e_d["gidx"] == e_d["pos"][-1], "le porteur n'est pas le + récent"
        # tri stable à récence prioritaire : scores TOUS ÉGAUX ⇒ le plus récent
        sc_eq = torch.zeros(1, len(e_d["res"]))
        rec = torch.arange(len(e_d["res"]) - 1, -1, -1)
        order = torch.argsort(-sc_eq[0][rec], stable=True)
        assert int(rec[order[0]]) == len(e_d["res"]) - 1, \
            "à score égal, l'injection doit préférer le groupe le PLUS RÉCENT"
    # 20g. retr_detach : par DÉFAUT le canal de sélection n'entraîne QUE W_q
    assert c_r5.retr_detach
    m_det = ToyReadLM(ToyCfg(**{**c_r5.__dict__, "retr_ce": 1.0}),
                      env.n_slots, env.n_attrs, sif_w=_sifw())
    m_det.zero_grad(set_to_none=True)
    # une conv SANS token supervisé côté LM ⇒ seul le canal retriever tire
    convM = {"info": conv_neg["info"],
             "segs": [dict(s) for s in conv_neg["segs"]]}
    for s in convM["segs"]:
        s["loss_mask"] = torch.zeros_like(s["loss_mask"])
    train_step(m_det, env, [convM], "cpu", 256, False, 1.0)
    assert float(m_det.retr.wq.weight.grad.abs().max()) > 0
    assert m_det.embed.weight.grad is None or \
        float(m_det.embed.weight.grad.abs().max()) == 0.0, \
        "retr_detach : la CE du retriever est remontée dans le backbone"
    # 20i. ENTRAÎNEMENT AVEC DISTRACTEUR (fix du mismatch train/éval)
    e_pick = next(e for e in env.retr_plan(m_r5, conv_neg).values()
                  if e["gidx"] is not None and len(e["res"]) >= 2)
    torch.manual_seed(99)
    p1 = train_group_pick(e_pick, 2)
    torch.manual_seed(99)
    p2 = train_group_pick(e_pick, 2)
    assert p1 == p2, "tirage des distracteurs non reproductible sous graine"
    assert len(p1) == 2 and e_pick["gidx"] in p1 and len(set(p1)) == 2, p1
    # l'ORACLE n'est pas toujours en tête (ordre aléatoire) …
    torch.manual_seed(7)
    firsts = {train_group_pick(e_pick, 2)[0] for _ in range(40)}
    assert len(firsts) >= 2 and e_pick["gidx"] in firsts, (
        f"l'oracle est toujours (ou jamais) en tête : {firsts}")
    # … sauf en `oracle_first`
    assert all(train_group_pick(e_pick, 2, "oracle_first")[0] == e_pick["gidx"]
               for _ in range(10))
    # G RÉDUIT quand la banque est trop courte (jamais de doublon)
    e_one = dict(e_pick)
    e_one["res"] = [e_pick["res"][e_pick["gidx"]]]
    e_one["gidx"] = 0
    assert train_group_pick(e_one, 3) == [0]
    assert len(train_group_pick(e_pick, 99)) == len(e_pick["res"])
    # le forward voit bien G groupes, et la CE aux reste sur le BON index
    m_dis = ToyReadLM(ToyCfg(**{**c_r5.__dict__, "retr_train_groups": 2}),
                      env.n_slots, env.n_attrs, sif_w=_sifw())
    assert m_dis.cfg.retr_train_groups == 2
    m_dis.zero_grad(set_to_none=True)
    ld = train_step(m_dis, env, [conv_neg, convN], "cpu", 256, False, 1.0)
    assert ld == ld and float(m_dis.retr.wq.weight.grad.abs().max()) > 0, \
        "la CE du retriever ne fire plus avec distracteur"
    # la cible de la CE est l'index dans `res` (donc INDÉPENDANTE de l'ordre du
    # préfixe injecté) — l'invariant qui rend le distracteur inoffensif pour le
    # retriever.
    e_chk = e_pick
    assert e_chk["gidx"] == e_chk["pos"][-1] and \
        e_chk["gidx"] < len(e_chk["res"])
    # 0 = sentinelle « suivre retr_topk »
    assert ToyCfg(**{**c_r5.__dict__, "retr_train_groups": 0,
                     "retr_topk": 3}).retr_train_groups == 3
    # 20j. NOMMAGE : deux configs qui n'entraînent PAS la même chose ne doivent
    #      JAMAIS partager un save_dir. `retr_topk` manquait — `--retr-topk 1`
    #      serait retombé sur `r5_toprows` et aurait écrasé le run de référence.
    # config de DÉPLOIEMENT (knobs aux défauts), pas celle du self-test
    _r5b = dict(vocab_size=512, d_model=64, n_heads=4, mem_dim=64,
                variant="r5", code="toprows", seg_n_pos=8, sif_a=A_SIF)

    def _n(**kw):
        return run_name_for(ToyCfg(**{**_r5b, **kw}))
    base = _n(retr_train_groups=1)          # = le run de référence fini
    assert base == "r5_toprows", base
    assert _n(retr_topk=1, retr_train_groups=1) == "r5_toprows_topk1"
    assert _n(retr_train_groups=2) == "r5_toprows_tg2"
    assert _n(retr_train_groups=2, retr_train_order="oracle_first") == \
        "r5_toprows_tg2_oracle_first"
    assert _n(retr_train_groups=1, retr_ce=0.5) == "r5_toprows_ce0.5"
    assert _n(retr_train_groups=1, retr_detach=False) == "r5_toprows_nodetach"
    # tous DISTINCTS deux à deux
    variants_n = [base, _n(retr_topk=1, retr_train_groups=1),
                  _n(retr_train_groups=2),
                  _n(retr_train_groups=2, retr_train_order="oracle_first"),
                  _n(retr_train_groups=1, retr_ce=0.5),
                  _n(retr_train_groups=1, retr_detach=False),
                  _n(retr_train_groups=3, retr_topk=3)]
    assert len(set(variants_n)) == len(variants_n), variants_n
    # et les conventions des phases précédentes n'ont pas bougé
    _b = dict(vocab_size=512, d_model=512, n_heads=4, mem_dim=512,
              variant="r3", seg_n_pos=8, sif_a=A_SIF)
    assert run_name_for(ToyCfg(**{**_b, "code": "segsif"})) == "r3_segsif"
    assert run_name_for(ToyCfg(**{**_b, "code": "toprows", "top_k": 8})) == \
        "r3_toprows_k8"
    assert run_name_for(ToyCfg(**{**_b, "code": "toprows",
                                  "readout_mix": "mos"})) == "r3_toprows_mos"
    assert run_name_for(ToyCfg(**{**_b, "code": "segsif",
                                  "write_mode": "every"})) == "r3_segsif_wev"
    assert run_name_for(ToyCfg(vocab_size=512, d_model=32, n_heads=4,
                               mem_dim=32, variant="r0")) == "r0"
    # 20h. r5 refuse ce qui n'a pas de sens (mêmes gardes que r4)
    for bad, needle in ((dict(code="segsif"), "toprows"),
                        (dict(code="toprows", write_mode="every"), "fact-only")):
        try:
            ToyCfg(vocab_size=512, d_model=32, n_heads=4, mem_dim=32,
                   variant="r5", seg_n_pos=8, sif_a=A_SIF, top_k=3, **bad)
        except AssertionError as e:
            assert needle in str(e), (bad, str(e))
        else:
            raise AssertionError(f"r5 aurait dû refuser {bad}")

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
    # ── phase 9 : `tophid` — la ligne est un ÉTAT POST-NORM, pas un embedding ─
    K9 = 5
    m_tr9 = _mk("toprows", d=512, top_k=K9)
    m_th9 = _mk("tophid", d=512, top_k=K9)
    # (a) la SÉLECTION est la même opération : refactoriser toprows_sel en
    #     toprows_sel_idx ne doit pas avoir déplacé un seul token.
    assert torch.equal(m_tr9.toprows_sel(seg20),
                       seg20.reshape(-1)[m_tr9.toprows_sel_idx(seg20)])
    assert torch.equal(m_tr9.toprows_sel_idx(seg20),
                       m_th9.toprows_sel_idx(seg20)), \
        "tophid doit sélectionner AUX MÊMES POSITIONS que toprows"
    g_tr = m_tr9.oracle_lines(3, 2, tok8[:3], seg_tok=seg20)
    g_th = m_th9.oracle_lines(3, 2, tok8[:3], seg_tok=seg20)
    # (b) même LAYOUT, même clé : seul le contenu des lignes 1.. change.
    assert g_tr.shape == g_th.shape == (1 + K9, 512), (g_tr.shape, g_th.shape)
    assert torch.equal(g_tr[0], g_th[0]), \
        "la ligne-clé doit être IDENTIQUE : l'A/B ne porte que sur le contenu"
    assert not torch.equal(g_tr[1:], g_th[1:]), \
        "les lignes de contenu doivent DIFFÉRER, sinon l'A/B ne mesure rien"
    # (c) les lignes post-norm sont normées comme celles de toprows (RMS 1) :
    #     sans ça le bras perdrait sur l'ÉCHELLE et pas sur son contenu.
    for g in (g_tr, g_th):
        r = g[1:].pow(2).mean(-1).sqrt()
        assert float((r - 1.0).abs().max()) < 1e-4, float(r.max())
    # (d) AUCUN gradient ne traverse la banque (invariant du design) — et
    #     `seg_hidden` a bien débranché la banque pour se calculer.
    assert not g_th.requires_grad
    assert m_th9.seg_hidden(seg20).shape == (int(seg20.numel()), 512)
    # (e) l'injection transporte des LIGNES, pas des ID, et son contenu compte.
    m_i9 = ToyReadLM(ToyCfg(vocab_size=512, d_model=64, n_layers=1, n_heads=4,
                            mem_dim=64, variant="r4", max_seq_len=64,
                            code="tophid", top_k=3, seg_n_pos=8, sif_a=A_SIF),
                     env.n_slots, env.n_attrs, sif_w=_sifw(512)).eval()
    rows9 = m_i9.tophid_rows_fixed(seg20)
    assert rows9.shape == (3, 64) and not rows9.requires_grad
    ids9 = torch.tensor([[5, 6, 7]])
    with torch.no_grad():
        o_nu = m_i9(ids9, None, None)
        o_a = m_i9(ids9, None, None, inject=rows9[None])           # [B,k,d]
        o_b = m_i9(ids9, None, None, inject=rows9[None, None])     # [B,1,k,d]
        o_c = m_i9(ids9, None, None, inject=rows9[None] * 0.5)
    assert torch.equal(o_a, o_b), "[B,k,d] et [B,1,k,d] doivent coïncider"
    assert not torch.equal(o_a, o_nu), "l'injection doit changer le forward"
    assert not torch.equal(o_a, o_c), "le CONTENU de la ligne doit compter"
    # (f) `hid_scale` existe, vaut 1.0 à l'init (⇒ la ligne entre telle quelle)
    #     et reçoit du gradient — sinon le confondant d'échelle reste non traité.
    assert float(m_i9.hid_scale) == 1.0
    assert not hasattr(m_tr9, "hid_scale"), \
        "toprows ne doit PAS gagner de paramètre : le bras resterait comparable"
    m_i9.zero_grad()
    m_i9(ids9, None, None, inject=rows9[None]).sum().backward()
    assert m_i9.hid_scale.grad is not None and \
        float(m_i9.hid_scale.grad.abs()) > 0, "hid_scale sans gradient"
    # (g) le NOMMAGE sépare les deux bras (sinon un run écrase l'autre).
    assert run_name_for(m_tr9.cfg).replace("r3", "r4") != \
        run_name_for(m_th9.cfg).replace("r3", "r4")
    # (top_k=3 ≠ défaut ⇒ suffixe _k3, cf. le sweep de k : c'est voulu)
    assert run_name_for(m_i9.cfg) == "r4_tophid_k3", run_name_for(m_i9.cfg)

    # ═══ PHASE 10 : provenance mi-tardive, âge roté, lecture β, règle ═══════
    # 21a. `midhid` : MÊME sélection, MÊME clé, MÊME layout que tophid/toprows —
    #      seul le vecteur normé change, et il change bien (l'A/B est isolé).
    K10 = 5
    m_mh = _mk("midhid", d=512, top_k=K10)
    assert torch.equal(m_mh.toprows_sel_idx(seg20), m_th9.toprows_sel_idx(seg20))
    g_mh = m_mh.oracle_lines(3, 2, tok8[:3], seg_tok=seg20)
    assert g_mh.shape == g_th.shape and torch.equal(g_mh[0], g_th[0]), \
        "midhid doit partager la CLÉ de tophid (l'A/B ne porte que le contenu)"
    assert not torch.equal(g_mh[1:], g_th[1:]), \
        "midhid ≡ tophid : le prélèvement n'a pas bougé"
    r = g_mh[1:].pow(2).mean(-1).sqrt()
    assert float((r - 1.0).abs().max()) < 1e-4, float(r.max())
    # 21b. le TAP est le bon étage : à hid_tap = 1.0 (n_layers blocs, AVANT
    #      norm_f) `seg_hidden` doit valoir la reconstruction explicite, et il
    #      doit DIFFÉRER de l'étage post-norm_f (sinon norm_f serait l'identité
    #      et tout le débat de la spec §2.4 serait vide).
    c_full = ToyCfg(vocab_size=512, d_model=64, n_layers=3, n_heads=4,
                    mem_dim=64, variant="r3", max_seq_len=64, code="midhid",
                    seg_n_pos=8, sif_a=A_SIF, top_k=3, hid_tap=1.0)
    m_full = ToyReadLM(c_full, env.n_slots, env.n_attrs,
                       sif_w=_sifw(512)).eval()
    assert m_full.cfg.hid_tap_layers == 3, m_full.cfg.hid_tap_layers
    with torch.no_grad():
        xx = m_full.embed(seg20.reshape(1, -1))
        for blk in m_full.blocks:
            xx = blk(xx, None, None, None)
        h_pre = xx[0].float()
        _, h_post = m_full.forward(seg20.reshape(1, -1), None, None,
                                   return_hidden=True)
    assert torch.allclose(m_full.seg_hidden(seg20), h_pre, atol=1e-5)
    assert not torch.allclose(h_pre, h_post[0].float(), atol=1e-3), \
        "pré-norm_f ≡ post-norm_f : le tap ne mesurerait rien"
    for tp, exp in ((2.0 / 3.0, 2), (0.34, 1), (0.5, 2)):
        assert ToyCfg(d_model=64, n_layers=3, n_heads=4, mem_dim=64,
                      variant="r3", code="midhid", hid_tap=tp,
                      sif_a=A_SIF).hid_tap_layers == exp, tp
    # 21c. ROTATION PAR ÂGE : buffers absents par défaut, âge 0 = IDENTITÉ,
    #      âges distincts ⇒ préfixes distincts, et rien ne bouge sans `ages`.
    def _mk_inj(**kw):
        # MÊME graine pour tous : les bras de la grille §2.4 doivent partager
        # leurs poids d'init, sinon un A/B mesurerait un tirage.
        torch.manual_seed(90210)
        c = ToyCfg(vocab_size=512, d_model=64, n_layers=2, n_heads=4,
                   mem_dim=64, variant="r4", max_seq_len=64, code="tophid",
                   top_k=3, seg_n_pos=8, sif_a=A_SIF, inject_sep_id=5,
                   max_mem=8, **kw)
        return ToyReadLM(c, env.n_slots, env.n_attrs, sif_w=_sifw(512)).eval()

    m_noage, m_age = _mk_inj(), _mk_inj(age_rope=True)
    assert not hasattr(m_noage, "age_cos") and hasattr(m_age, "age_cos")
    rows10 = m_age.tophid_rows_fixed(seg20)[None, None].expand(1, 2, 3, 64)
    ids10 = torch.tensor([[5, 6, 7]])
    z2 = torch.zeros(1, 2, dtype=torch.long)
    with torch.no_grad():
        o_none = m_age(ids10, None, None, inject=rows10)
        o_zero = m_age(ids10, None, None, inject=rows10, inject_age=z2)
        o_mix = m_age(ids10, None, None, inject=rows10,
                      inject_age=torch.tensor([[0, 1]]))
        o_off = m_noage(ids10, None, None, inject=rows10,
                        inject_age=torch.tensor([[0, 1]]))
        o_offn = m_noage(ids10, None, None, inject=rows10)
    assert torch.equal(o_none, o_zero), "âge 0 doit être rot(0) = IDENTITÉ"
    assert not torch.equal(o_zero, o_mix), "des âges distincts doivent séparer"
    assert torch.equal(o_off, o_offn), \
        "age_rope OFF : `inject_age` doit être IGNORÉ (rétro-compat)"
    # 21c-bis. PENDANT BANQUE de la rotation d'âge (bras seq_fw / bank_xattn) :
    #      `pad_bank_age` rend le RANG DE RÉCENCE par ligne (toutes les lignes
    #      d'un groupe partagent l'âge de leur slot), et la rotation change
    #      bien le forward — mesuré PORTE OUVERTE, puisque toutes les portes de
    #      read du labo sont fermées à l'init (convention de la phase 1).
    m_bk = _mk("tophid", d=64, top_k=3, vocab=512, seg_n_pos=8)
    m_bk.cfg.age_rope = True
    c_, s_ = phase_tables(m_bk.cfg.max_mem, 64, 0.0)
    m_bk.register_buffer("age_cos", c_)
    m_bk.register_buffer("age_sin", s_)
    bk = GroupBank()
    for _ in range(2):
        bk = env.write(m_bk, bk, seg_tpl)
    assert list(bk.groups) == [4, 4] and len(bk) == 8, (bk.groups, len(bk))
    ages_bk = pad_bank_age([bk], "cpu", m_bk.cfg.group_rows)
    assert ages_bk.tolist() == [[1, 1, 1, 1, 0, 0, 0, 0]], ages_bk.tolist()
    for blk in m_bk.blocks:
        if blk.read is not None:
            for _n, _p in blk.read.named_parameters():
                if "gate" in _n:
                    _p.data.fill_(1.0)
    b_bk, bm_bk = pad_bank([bk], "cpu")
    ids_bk = torch.tensor([[5, 6, 7]])
    with torch.no_grad():
        assert not torch.equal(m_bk(ids_bk, b_bk, bm_bk),
                               m_bk(ids_bk, b_bk, bm_bk, bank_age=ages_bk)), \
            "la rotation d'âge côté BANQUE ne change rien : plomberie morte"
        assert torch.equal(
            m_bk(ids_bk, b_bk, bm_bk),
            m_bk(ids_bk, b_bk, bm_bk, bank_age=torch.zeros_like(ages_bk))), \
            "âge 0 partout doit être l'IDENTITÉ côté banque aussi"
    # 21d. LECTURE β (`read_path='kv'`) : pas de préfixe, mémoire visible de
    #      TOUTE position, contenu comptant, ablaté ≡ backbone nu BIT-À-BIT,
    #      et la mémoire n'entre QUE dans les couches lectrices.
    m_kv = _mk_inj(read_path="kv")
    with torch.no_grad():
        k_nu1 = m_kv(ids10, None, None)
        k_nu2 = m_noage(ids10, None, None)
        k_a = m_kv(ids10, None, None, inject=rows10)
        k_b = m_kv(ids10, None, None, inject=rows10 * 0.5)
    assert torch.equal(k_nu1, k_nu2), \
        "read_path kv : le forward SANS injection doit être le backbone nu"
    assert not torch.equal(k_a, k_nu1) and not torch.equal(k_a, k_b), \
        "kv_append : l'injection et son CONTENU doivent compter"
    assert k_a.shape == k_nu1.shape, "kv_append ne doit rendre AUCUN préfixe"
    # la PREMIÈRE position voit déjà la mémoire (c'est tout l'intérêt de β :
    # aucune position n'est en amont de la banque)
    assert not torch.equal(k_a[:, 0], k_nu1[:, 0])
    m_kv1 = _mk_inj(read_path="kv", read_layers=[1])
    assert [b.read_bank for b in m_kv1.blocks] == [False, True]
    # 21e. LE STREAM : la règle est un seg PORTEUR (donc écrit), son nom
    #      n'apparaît JAMAIS dans une réponse, et la paire contrastive partage
    #      TOUT sauf les marqueurs.
    rs = PersonaRuleStream(tok, seed=11, p_rule=1.0, p_smalltalk=0.0,
                           cond_decoys=1)
    rconv = next(rs.next_conv() for _ in range(1))
    cnd = rconv["info"]["cond"]
    assert rconv["kind"] == "rule" and rconv["info"]["truths"] == [], \
        "une vie-règle ne doit RIEN offrir à grade_recall (ph.9 intacte)"
    wsegs = [i for i, s in enumerate(rconv["segs"])
             if OracleEnv.fact_of(s) is not None]
    assert len(wsegs) == 2 and cnd["rule_at"] < 2, wsegs
    assert cnd["turns"], "aucune réponse conditionnée"
    for q, ti in enumerate(cnd["turns"]):
        coh, inc = rconv["segs"][ti], cnd["alts"][q]
        txt = tok.decode(coh["input_ids"][0].tolist())
        assert cnd["reg"] not in txt, \
            f"le nom du registre FUIT dans la réponse : {txt!r}"
        assert float(coh["cond_mask"].sum()) > 0 and \
            float(inc["cond_mask"].sum()) > 0
        # même corps, mêmes marqueurs en NOMBRE, contenus DIFFÉRENTS
        assert not torch.equal(coh["input_ids"], inc["input_ids"])
    # les valeurs de registre sont RETIRÉES du pool `color`
    assert not (set(COND_REGISTERS) & set(rs.slots.get(COND_RULE_SLOT,
                                                       ((), (), (), (), ()))[4]))
    # 21f. `cond_plan` : G CONSTANT, âges = rangs de récence (0 = plus récent),
    #      et {} sur toute conv qui n'est pas une vie-règle.
    m_c = _mk_inj(cond=True, cond_decoys=1)
    cp = env.cond_plan(m_c, rconv)
    assert set(cp) == set(cnd["turns"]), (set(cp), cnd["turns"])
    for rws, ags in cp.values():
        assert rws.shape == (2, 3, 64), rws.shape
        assert sorted(ags.tolist()) == [0, 1], ags.tolist()
    assert env.cond_plan(m_c, conv) == {}, \
        "cond_plan doit être un no-op hors vie-règle (phases 6-9 intactes)"
    # 21g. `evaluate_cond` tourne, rend ses trois lectures, et `none` est
    #      EXACTEMENT le bras sans mémoire (Δ identique à un forward nu).
    ec = evaluate_cond(m_c, env, PersonaRuleStream(tok, seed=11, p_rule=1.0,
                                                   p_smalltalk=0.0),
                       11, 3, "cpu", 64, False)
    assert ec["n"] >= 3 and ec["n_convs"] == 3, ec
    for kk in ("mark_live", "mark_shuf", "mark_none", "acc_live", "dnll_none"):
        assert ec[kk] == ec[kk], f"{kk} NaN : la métrique ne sort pas"
    assert 0.0 <= ec["acc_live"] <= 1.0
    # 21h. NOMMAGE : les quatre axes séparent les dossiers (aucun run écrasé).
    _n10 = {run_name_for(_mk_inj(**kw).cfg) for kw in (
        {}, {"age_rope": True}, {"read_path": "kv"}, {"cond": True},
        {"cond": True, "cond_arm": "none"}, {"cond": True, "cond_decoys": 3},
        {"cond": True, "age_rope": True, "read_path": "kv"})}
    assert len(_n10) == 7, sorted(_n10)
    assert run_name_for(_mk_inj(cond=True, age_rope=True,
                                read_path="kv").cfg) == \
        "read-kvappend_rot-on_tap-postnorm_m3", _n10
    # la GRILLE : 36 combos, 36 noms DISTINCTS, et le nom se relit
    _base = dict(vocab_size=512, d_model=64, n_layers=2, n_heads=4,
                 mem_dim=64, max_seq_len=64, sif_a=A_SIF, seg_n_pos=8,
                 inject_sep_id=5, max_mem=8)
    _cmb = grid_combos()
    assert len(_cmb) == 36, len(_cmb)
    _gn = [grid_name(_grid_cfg(cc, _base)) for cc in _cmb]
    assert len(set(_gn)) == 36, sorted(_gn)
    assert "read-seqfw_rot-off_tap-mid_m4" in _gn, sorted(_gn)
    # 21i. les GARDE-FOUS : chaque axe refuse les combinaisons qui seraient
    #      silencieusement ignorées (le labo ne laisse pas passer un no-op).
    for bad in ({"variant": "r1", "code": "mean", "age_rope": True},
                {"variant": "r3", "read_path": "kv"},
                {"variant": "r1", "code": "mean", "cond": True},
                {"variant": "r4", "code": "tophid", "cond_arm": "none"},
                {"variant": "r3", "code": "midhid", "hid_tap": 0.0}):
        try:
            ToyCfg(vocab_size=512, d_model=64, n_layers=2, n_heads=4,
                   mem_dim=64, max_seq_len=64, sif_a=A_SIF, top_k=3,
                   **{"code": "tophid", **bad})
        except AssertionError:
            pass
        else:
            raise AssertionError(f"ToyCfg aurait dû refuser {bad}")

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
    print("  phase 8 — r5 RETRIEVER APPRIS : W_q zéro-init seul module ajouté, "
          "h_query = dernier token du segment USER (invariant vérifié : "
          "changer la RÉPONSE ne bouge pas h_query, changer la QUESTION oui), "
          "scores masqués à −inf hors résidents et top-k STABLE à récence "
          "prioritaire, préfixe multi-groupes ([B,k] ≡ [B,1,k] BIT-À-BIT ⇒ r4 "
          "intact), CE MULTI-POSITIVE (deux writes d'un même slot ont la MÊME "
          "clé — cible unique = gradient nul), ne fire que sur les réponses "
          "gradées, retr_detach n'envoie rien dans le backbone")
    print("  phase 8b — DISTRACTEUR à l'entraînement : oracle + distracteurs "
          "tirés au sort (reproductible sous graine), ordre aléatoire par "
          "défaut / oracle_first en option, G RÉDUIT si la banque est courte "
          "(jamais de doublon), retr_train_groups=1 ≡ v1 ; NOMMAGE : "
          "retr_topk/retr_train_groups/order/ce/detach entrent tous dans le "
          "save_dir, 7 configs r5 ⇒ 7 noms DISTINCTS (r5_toprows reste le run "
          "de référence à G=1)")
    print("  phase 9 — TOPHID (la ligne est l'état POST-`norm_f`, pas "
          "l'embedding d'entrée) : MÊME sélection SIF aux MÊMES positions, "
          "MÊME clé ligne 0, MÊME layout 1+top_k, MÊME RMS 1 — seul le vecteur "
          "normé change (l'A/B est isolé à ce point) ; banque detachée et "
          "`seg_hidden` calculé banque DÉBRANCHÉE ; injection par LIGNES "
          "([B,k,d] ≡ [B,1,k,d], contenu ET échelle comptent), `hid_scale` "
          "init 1.0 et dérivable, absent de toprows (bras comparables) ; "
          "r4_tophid ≠ r4_toprows au nommage")
    print(f"  phase 10 — GRILLE §2.4, quatre axes ORTHOGONAUX. "
          f"(a) PROVENANCE `midhid` : même sélection/clé/layout que tophid, "
          f"seul le vecteur change ; tap = round(hid_tap·n_layers) borné "
          f"[1,L], et pré-norm_f ≠ post-norm_f (le tap mesure quelque chose). "
          f"(b) ÂGE : buffers absents par défaut, rot(0) = IDENTITÉ, âges "
          f"distincts ⇒ préfixes distincts, `inject_age` IGNORÉ si age_rope "
          f"OFF. (c) LECTURE β `kv` : forward sans injection ≡ backbone nu "
          f"BIT-À-BIT, aucun préfixe rendu, la position 0 voit déjà la banque, "
          f"contenu comptant, mémoire limitée aux couches lectrices. "
          f"(d) RÈGLE : vie-règle = 1 règle + {1} leurre écrits, truths VIDE "
          f"(ph.9 intacte), nom du registre JAMAIS dans la réponse, registres "
          f"retirés du pool `color`, cond_plan à G constant et âges = rangs de "
          f"récence, no-op hors vie-règle ; evaluate_cond rend ses 3 lectures. "
          f"(e) 7 configs de la grille ⇒ 7 save_dir distincts, et 5 "
          f"combinaisons no-op sont REFUSÉES à la construction.", flush=True)
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
