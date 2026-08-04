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

from .data.math_school_data import A_OPEN, CLOSE as CLOSE_P11
from .infra.paths import load_yaml
from .data.persona_chat_data import (PET_TYPES, SIBLINGS, PersonaChatStream,
                                fact_id_maps, grade_recall)
from .data.streams import chat_stream_class

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
READ_PATHS = ("entry", "kv", "dual", "kvproj")
# ── phase 11 (spec §2.5) : les TROIS FAMILLES de métadonnées, sur les clés
# banque PROJETÉES de kvproj. Chaque famille a son bras de contrôle (`none`)
# et son steelman non-rotatif (biais scalaire pour l'âge, vecteur additif pour
# le tag et l'index local) — c'est ce qui rend S3/S5/S17 adjudicables.
BANK_ROTS = ("none", "age-log", "age-raw", "age-bias")
TAG_MODES = ("none", "rot", "add")
LOC_MODES = ("none", "rot", "add")
# environnements de la phase 11 (cf. ToyCfg.p11_env) + `life` (phase 12).
P11_ENVS = ("rule", "prov", "span", "life")
# les QUATRE examens (cf. ToyCfg.p11_exam et P11_EXAMS plus bas) : S3, S4, S5,
# S17 du registre §3.
P11_EXAM_NAMES = ("age", "ood", "tag", "locidx")
# canaux de PROVENANCE de l'env `prov` : qui a dit la ligne. Ordre FIGÉ — il
# indexe les plans de rotation (un plan 0/π par canal) et les vecteurs additifs.
CHANNELS = ("user", "self")
# ══ PHASE 12 — MAINTENANCE PROCÉDURALE (S6) ET DILUTION (S8) ════════════════
# Le RL est HORS SUJET ici (cadrage user 08-03) : rien de ce bloc n'apprend.
# La maintenance est un PLUG-IN PROCÉDURAL hors graphe (spec §2.3 : « append en
# tête + décalage + chute au bord », plus la PROPAGATION à budget), et le seul
# module appris reste le read kvproj.
#
#   fifo      aucune propagation — la BASELINE BASSE (p forcé à 0).
#   age       propage les plus VIEILLES encore vivantes. C'est le proxy que la
#             spec RÉPUDIE (auto-renforçant : ce qui a survécu devient plus dur
#             à déloger PARCE QUE ça a survécu). Sa présence au bakeoff est la
#             preuve qu'on l'a battu — ou pas.
#   attn-ema  EMA de la MASSE D'ATTENTION par ligne, lue GRATUITEMENT par la
#             sonde `bank_attn_probe` au forward de chaque tour : le LECTEUR
#             vote. Contrainte S2 (fuite non-causale) : la masse du tour t
#             n'entre dans les scores qu'au write du tour t+1, JAMAIS avant —
#             cf. le tampon `pend`/`arm` de RetentionStore.
#   coverage  couverture sémantique : meurt la ligne dont le plus proche voisin
#             (cosinus, dans l'espace des lignes) est LE PLUS PROCHE. La
#             redondance meurt, la singularité survit.
#   actr      activation ACT-R : Σ (Δ+1)^(−d) sur les usages (hits d'attention)
#             + la naissance, à décroissance temporelle.
RETENTION_SIGNALS = ("fifo", "age", "attn-ema", "coverage", "actr")
# S8 — de quoi la banque est REMPLIE quand on fait grandir max_mem. `foreign` =
# des groupes d'AUTRES vies du MÊME stream (des distracteurs RÉELS, pas du
# bruit gaussien), recalculés avec le modèle COURANT comme les vrais.
BANK_FILLS = ("none", "foreign")
# les examens de la phase 12 (registre §3 : S6 et S8).
P12_EXAM_NAMES = ("retention", "dilution")
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
#   dual_heads   GROUPE DE TÊTES DÉDIÉ à la banque dans les couches lectrices
#                (r4 + dual) : le groupe 1 fait l'attention normale sur le
#                contexte, le groupe 2 attend la BANQUE SEULE (ni RoPE ni
#                masque causal), les deux sorties fusionnent avant le FFN.
#                Répond au défaut structurel de kv_append — banque et contexte
#                s'y disputent la masse softmax dans la MÊME tête, invisible au
#                jouet mais risque n°1 au 350M à fenêtre longue. Coût assumé :
#                des paramètres dédiés (cf. BankHeads).
#   kv_proj      PROJECTIONS DÉDIÉES dans le softmax UNIFIÉ (r4 + kvproj) : K'
#                et V' viennent de W_k'/W_v' propres à la banque, mais entrent
#                dans les MÊMES têtes que le contexte. Avec `--bank-q`, les
#                lignes émettent en plus leurs propres requêtes et se
#                contextualisent de couche lectrice en couche lectrice.
#
# ── LE CARRÉ FACTORIEL 2×2 (design user) ────────────────────────────────────
# Deux axes ORTHOGONAUX se croisent sur trois de ces bras — c'est ce qui rend
# le dépouillement causal au lieu d'anecdotique :
#
#                     │ softmax UNIFIÉ        │ softmax SÉPARÉ
#   ──────────────────┼───────────────────────┼──────────────────────
#   projections       │ kv_append             │ (vide : sans projection
#   PARTAGÉES         │ (mesuré, 2AFC 1.000)  │  ni tête propre, il n'y
#                     │                       │  a rien à séparer)
#   projections       │ kv_proj               │ dual_heads
#   DÉDIÉES           │                       │
#
#   kv_proj vs kv_append  = l'effet PROJECTIONS (géométrie de clés), seul
#                           delta entre les deux.
#   kv_proj vs dual_heads = l'effet SOFTMAX PARTAGÉ : arbitrage par une
#                           distribution unique (la banque peut perdre) contre
#                           masse garantie (elle ne peut pas).
#   ±bank_q               = l'effet CONTEXTUALISATION des lignes.
# Départage sur le Δnll de CITATION et les MARGES aux marqueurs — PAS sur le
# 2AFC, saturé à 1.000 dès que le canal est ouvert.
READ_MODES = {
    "seq_fw":       ("r0", None,   "entry"),
    "bank_xattn":   ("r3", None,   "entry"),
    "inject_entry": ("r4", None,   "entry"),
    "kv_append":    ("r4", None,   "kv"),
    "dual_heads":   ("r4", None,   "dual"),
    "kv_proj":      ("r4", None,   "kvproj"),
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
    # ── TÊTES DÉDIÉES À LA BANQUE (phase 10, read_path='dual') ──────────────
    bank_heads: int = 2       # nombre de têtes du GROUPE BANQUE dans chaque
                              # couche lectrice. « Quelques têtes suffisent » :
                              # le groupe n'a qu'un ensemble de max_mem·(1+m)
                              # lignes à discriminer, pas une fenêtre entière.
    bank_head_dim: int = 0    # dim par tête du groupe banque. 0 = celle des
                              # têtes de contexte (d_model // n_heads), pour
                              # que la géométrie de clés soit comparable.
    # ── PROJECTIONS DÉDIÉES / SOFTMAX UNIFIÉ (read_path='kvproj') ────────────
    bank_q: bool = False      # `kvproj` : les lignes ÉMETTENT leurs propres
                              # requêtes (W_q' dédié) et attendent sur
                              # [banque ∪ contexte] ; leur état est PORTÉ de
                              # couche lectrice en couche lectrice, puis jeté à
                              # la sortie du stack. C'est le test « la ligne
                              # s'aiguise contre la question ». W_o' zéro-init
                              # ⇒ à l'init les lanes valent exactement les
                              # lignes brutes et le bras dégénère en `kvproj`
                              # nu. ⚠️ CE QUI EST LIVRÉ : les lanes sont un
                              # SIDE-STREAM (mise à jour résiduelle par
                              # ATTENTION seule, RMSNorm dédiée, PAS de MLP).
                              # Leur faire traverser le MLP en ferait une copie
                              # du flux de tokens, donc `inject_entry` sans
                              # RoPE — on perdrait le contraste que le bras est
                              # censé mesurer.
    # ── COMPOSITION DU GRADIENT de la boucle fast-weight (correctif terrain) ─
    fw_additive: bool = False  # stop-gradient sur l'ÉTAT PORTÉ de la boucle
                              # séquentielle de FastWeightRead : chaque
                              # itération lit y0 + (y − y0).detach() au lieu de
                              # y. Le FORWARD est numériquement IDENTIQUE (même
                              # valeur, autre graphe) ; seul le BACKWARD change
                              # — le produit de jacobiennes sur M×n_layers
                              # étages disparaît, chaque slot ne garde que le
                              # gradient LOCAL de son propre `upd`.
                              # POURQUOI : mesuré sur la ferme, le bras `seq_fw`
                              # multiplicatif diverge dans 5 cellules sur 6
                              # (gnorm pré-clip 1e4 → 4.7e9, loss 6.8-11.3,
                              # Δnll de citation NÉGATIF) ; seule survit la
                              # cellule à M minimal. grad_clip ne corrige que
                              # la norme, pas la direction. Cf. FastWeightRead.
                              # DÉFAUT OFF : le bras divergent EST une donnée
                              # de la grille, on ne le réécrit pas.
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
    # ══ PHASE 11 — MÉTADONNÉES PAR ROTATION SUR LES CLÉS BANQUE PROJETÉES ══
    # (spec §2.5 ; registre §3 S3/S4/S5/S17). TOUT ce bloc est OFF par défaut
    # et le forward est alors bit-à-bit celui de la phase 10.
    #
    # OÙ. Les trois familles s'appliquent à K' = W_K'·g — les clés DÉDIÉES de
    # `kvproj`, APRÈS la projection. C'est la différence de fond avec
    # `age_rope` (ph.10), qui rotait la LIGNE avant l'injection : une rotation
    # posée avant W_k ne survit pas à la projection (elle se fait mélanger par
    # une matrice apprise qui n'a aucune raison de commuter avec elle). Le
    # ph.10 `age_rope` est CONSERVÉ tel quel — c'est le bras historique de la
    # grille §2.4, il ne se réécrit pas.
    #
    # SUR QUELLES DIMS. Contrainte côté requête (§2.5) : kvproj partage le q du
    # backbone, RoPE compris, donc le score banque est (R(t)·W_Q x)ᵀ K'. Le
    # produit se fait paire à paire ⇒ les plans de métadonnées doivent viser
    # les paires QUASI STATIQUES du RoPE de tête (ω ≈ 0), sinon le code se
    # mélange à la position dans la fenêtre. `slow_rope_planes` fait le choix
    # EXPLICITEMENT (tri par fréquence) et `rot_drift_max` le refuse s'il ne
    # tient pas. Les trois familles occupent des plans DISJOINTS.
    bank_rot: str = "none"    # famille ÂGE (S3/S4) :
                              #   none     — AUCUNE rotation d'âge. C'est le
                              #              bras HoPE θ_âge=0, la baseline
                              #              NON NÉGOCIABLE à battre.
                              #   age-log  — φ(a) = A_ref·log(1+a)/log(1+A_ref)
                              #              puis rot(ω_p·φ(a)) sur age_planes
                              #              plans, ω APPRIS (init géométrique
                              #              entrelacé, sans bande haute).
                              #   age-raw  — idem mais φ(a) = a (le bras qui
                              #              doit s'effondrer en OOD, S4).
                              #   age-bias — AUCUNE rotation : un biais
                              #              scalaire de récence b(a) = w·a
                              #              (UN paramètre) sur le logit des
                              #              colonnes banque. Le fallback de la
                              #              règle S3 si la rotation perd.
    age_planes: int = 4       # plans de la famille âge (jouet : 4 ; le design
                              # 350M en prévoit 8, cf. veille). Échelles
                              # GÉOMÉTRIQUES entrelacées, jamais de bande
                              # haute fréquence.
    age_ref: int = 8          # A_cible de la compression log — l'horizon
                              # « normal » en writes (= max_mem au jouet).
    age_aug: bool = False     # S4 : à l'ENTRAÎNEMENT, l'âge de chaque lot est
                              # multiplié par un facteur log-uniforme dans
                              # [1, age_aug_max]. L'ORDRE des âges est
                              # préservé (c'est lui qui porte l'information),
                              # seule l'ÉCHELLE varie ⇒ le bras `raw` peut
                              # apprendre l'invariance qu'il n'a pas par
                              # construction. C'est le steelman du brut.
    age_aug_max: float = 8.0  # borne haute de l'augmentation. STRICTEMENT
                              # INFÉRIEURE aux échelles d'éval (10, 100) : une
                              # augmentation qui couvrirait l'éval ne
                              # mesurerait plus l'OOD.
    age_eval_scales: str = "1,10,100"   # S4 : l'éval contrastive est répétée
                              # avec les âges MULTIPLIÉS par chaque échelle
                              # (vies longues synthétiques : le fait ancien est
                              # toujours en banque, le compteur de writes a
                              # grandi). L'échelle 1 est le régime vu au train.
    tag_mode: str = "none"    # famille PROVENANCE (S5) : `none` (contrôle) |
                              # `rot` (UN PLAN 0/π PAR CANAL — jamais n angles
                              # sur un plan, cf. veille : la métrique cyclique
                              # inventerait des similarités entre canaux) |
                              # `add` (STEELMAN ADDITIF : un vecteur appris par
                              # canal AJOUTÉ sur les MÊMES dims réservées, donc
                              # à budget de dims identique).
    n_channels: int = 2       # canaux de provenance (user, self). Un plan
                              # 0/π par canal ⇒ n_channels plans réservés.
    prov_vals: str = "ref"    # POOL DE VALEURS de l'env `prov` : `ref` (legacy,
                              # codes XX-12345 — INVALIDÉ 08-04 comme
                              # instrument : la queue à 5 chiffres est hors de
                              # portée du circuit de copie du jouet, grade
                              # plancher 0 dans TOUS les bras, l'examen ne
                              # discrimine rien) | `span` (valeurs MESURÉES en
                              # tokens par span_value_pool, buckets courts —
                              # la copie est dans la plage prouvée de
                              # l'instrument : span grade 0,97).
    life_vals: str = "city"   # POOL DE VALEURS de l'env `life` : `city`
                              # (legacy, ~40 noms de villes — INVALIDÉ 08-04
                              # comme régime de données : répertoire FERMÉ
                              # d'entités à signature unique ⇒ le train se
                              # minimise par RECONNAISSANCE (villes mémorisées
                              # dans les poids, grade_train 0,80) et la copie
                              # ne se forme jamais (held-out 0,000 dans les 16
                              # cellules S6 ; sonde : la valeur est pourtant
                              # dans les lignes 73 % et r@1 0,75) | `span`
                              # (valeurs span_value_pool buckets 1-2 : 96
                              # valeurs COMPOSITIONNELLES ≤2 sous-tokens — le
                              # partage de morceaux ferme le raccourci de
                              # reconnaissance, et L≤2 tient dans m=4 sans
                              # confondant k).
    loc_mode: str = "none"    # famille INDEX LOCAL intra-span (S17) : `none` |
                              # `rot` (R_loc(j) sur loc_planes plans, j =
                              # index de la ligne DANS son write — borné par
                              # construction, donc aucun risque d'OOD) |
                              # `add` (embedding de position locale appris,
                              # additif, sur les mêmes dims réservées).
    loc_planes: int = 2       # plans de la famille index local.
    rot_drift_max: float = 0.5   # GARDE-FOU DE L'APPARIEMENT (radians) : dérive
                              # maximale tolérée de R(t) du RoPE backbone sur
                              # les plans choisis, sur toute la fenêtre
                              # (ω_max·(max_seq_len−1)). Au-delà, la
                              # construction ÉCHOUE : demander trop de plans
                              # fait déborder les métadonnées dans la bande
                              # rapide, où le code d'âge se mélangerait à la
                              # position du lecteur.
    p11_exam: str = ""        # EXAMEN DÉCLARÉ (age | ood | tag | locidx), vide
                              # = ce n'est pas une cellule de la phase 11.
                              # POURQUOI IL EST DÉCLARÉ ET NON DÉDUIT : le bras
                              # de CONTRÔLE de S3 (`agezero`) n'active AUCUNE
                              # métadonnée — sa config est exactement celle
                              # d'une cellule kvproj de la grille §2.4, et
                              # rien dans le modèle ne pourrait la distinguer.
                              # Sans champ déclaré, le contrôle retomberait sur
                              # le dossier de la ph.10 et l'examen n'aurait
                              # plus sa baseline dans son propre espace de
                              # noms. C'est aussi la cohérence qu'exige la
                              # règle S3 : la baseline doit être mesurée par le
                              # MÊME harnais que les bras qu'elle arbitre.
    p11_env: str = "rule"     # environnement de la phase 11 :
                              #   rule — la vie-règle de la ph.10 (S3/S4 :
                              #          conditionnement contrastif, appariable
                              #          aux 12 cellules kvproj du carré)
                              #   prov — vies à LOCUTEUR (S5) : deux faits du
                              #          MÊME attribut, l'un dit par l'user
                              #          l'autre par le modèle ; la question
                              #          cible le locuteur
                              #   span — valeurs multi-tokens de longueur
                              #          GRADUÉE 1..4 (S17), citation ORDONNÉE
                              #   life — PHASE 12 : vies LONGUES (T ≫ max_mem)
                              #          avec supersession tardive et
                              #          ré-évocation historique (S6)
    # ══ PHASE 12 — MAINTENANCE PROCÉDURALE (S6) ET DILUTION (S8) ═══════════
    # TOUT ce bloc est OFF par défaut et le forward est alors bit-à-bit celui
    # de la phase 11. RIEN ici n'apprend : la maintenance est un plug-in
    # procédural HORS GRAPHE (spec §2.3, §2.9).
    retention: str = "fifo"   # signal de rétention (cf. RETENTION_SIGNALS).
                              # `fifo` = aucune propagation (baseline basse).
    prop_budget: int = 0      # p — lignes PROPAGÉES par write (petit : 1-2).
                              # Les p entrées de la queue au meilleur score
                              # sont replacées en TÊTE, compteur de naissance
                              # PRÉSERVÉ (l'âge reste l'âge VRAI, cohérent avec
                              # la rotation d'âge de BankRot). 0 = FIFO nue.
    ema_beta: float = 0.9     # `attn-ema` : facteur de l'EMA de masse.
    actr_decay: float = 0.5   # `actr` : d de (Δ+1)^(−d).
    life_turns: int = 48      # env `life` : LONGUEUR CIBLE d'une vie en SEGS.
                              # Le nombre de writes en découle (≈ 2 + T/4) et
                              # doit rester ≫ max_mem — c'est toute la pression
                              # FIFO de S6.
    bank_fill: str = "none"   # S8 : `foreign` remplit la banque jusqu'à
                              # max_mem GROUPES avec des groupes d'autres vies
                              # du même stream. Les vrais résidents sont
                              # TOUJOURS placés EN DERNIER (les plus frais) et
                              # les remplisseurs portent des métadonnées NULLES
                              # (slot/val 0) : ils ne peuvent jamais devenir la
                              # cible du r@1, ils ne font que DILUER.
    fill_pool: int = 96       # réservoir de segments étrangers (FIFO).
    fill_refresh: int = 8     # les lignes étrangères sont RECALCULÉES avec le
                              # modèle courant tous les `fill_refresh` appels
                              # (≈ une fois par pas d'entraînement à 8 convs) :
                              # sans ça elles seraient des vecteurs PÉRIMÉS, et
                              # la dilution mesurerait la dérive des
                              # embeddings au lieu du nombre de lignes.
    p12_exam: str = ""        # EXAMEN DÉCLARÉ de la phase 12 (retention |
                              # dilution), vide = ce n'est pas une cellule ph.12.
                              # Même raison qu'au `p11_exam` : le bras de
                              # CONTRÔLE (`fifo`, `S8`) n'active rien et n'a
                              # rien d'autre pour se distinguer d'une cellule
                              # déjà lancée.
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
        if self.bank_q:
            assert self.read_path == "kvproj", (
                f"code.bank_q fait ÉMETTRE les lignes dans le softmax unifié "
                f"de `kvproj` : il n'a de sens que là (reçu read_path="
                f"{self.read_path!r}) — il serait un no-op silencieux")
        if self.read_path == "dual":
            assert self.bank_heads >= 1, self.bank_heads
            assert self.bank_head_dim >= 0, self.bank_head_dim
        elif self.bank_heads != ToyCfg.bank_heads or self.bank_head_dim:
            raise AssertionError(
                f"bank_heads/bank_head_dim ne pilotent QUE le groupe de têtes "
                f"dédié (read_path='dual') ; reçu read_path="
                f"{self.read_path!r} — ils seraient silencieusement ignorés")
        if self.fw_additive:
            assert self.uses_fw, (
                f"code.fw_additive change la COMPOSITION DU GRADIENT de la "
                f"boucle séquentielle de FastWeightRead : il n'a de sens que "
                f"pour les variantes qui en portent une ({'/'.join(v for v in VARIANTS if v in ('r0', 'r2'))} "
                f"— le bras `--read seq_fw` de la grille). Reçu --variant "
                f"{self.variant}, qui lit par ATTENTION : il n'y a aucune "
                f"boucle à rendre additive, le flag serait un no-op silencieux.")
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
        # ── phase 11 : les trois familles de rotations (spec §2.5) ─────────
        assert self.bank_rot in BANK_ROTS, (
            f"bank_rot inconnu {self.bank_rot!r} (∈ {BANK_ROTS})")
        assert self.tag_mode in TAG_MODES, (
            f"tag_mode inconnu {self.tag_mode!r} (∈ {TAG_MODES})")
        assert self.loc_mode in LOC_MODES, (
            f"loc_mode inconnu {self.loc_mode!r} (∈ {LOC_MODES})")
        assert self.p11_env in P11_ENVS, (
            f"p11_env inconnu {self.p11_env!r} (∈ {P11_ENVS})")
        assert self.p11_exam in ("",) + tuple(P11_EXAM_NAMES), (
            f"p11_exam inconnu {self.p11_exam!r} "
            f"(∈ {('',) + tuple(P11_EXAM_NAMES)})")
        if self.uses_p11_meta or self.p11_env != "rule":
            assert self.p11_exam or self.p12_exam, (
                "toute cellule des phases 11/12 DÉCLARE son examen "
                "(--p11-exam age|ood|tag|locidx, --p12-exam "
                "retention|dilution) : c'est lui qui nomme le dossier, et le "
                "bras de contrôle n'a rien d'autre pour se distinguer d'une "
                "cellule de la grille §2.4")
        if self.uses_p11_meta:
            # Les métadonnées vivent dans les projections K DÉDIÉES (§2.5 :
            # « jamais dans la bande RoPE du backbone ») et sur des lignes
            # INJECTÉES (c'est là que le tenseur de métadonnées existe).
            assert self.read_path == "kvproj", (
                f"les rotations de métadonnées (bank_rot/tag_mode/loc_mode) "
                f"s'appliquent à K' = W_K'·g, les clés DÉDIÉES de `kvproj` : "
                f"elles exigent --read kv_proj (reçu read_path="
                f"{self.read_path!r}) — ailleurs il n'y a pas de projection "
                f"dédiée où les héberger, et une rotation posée avant W_k ne "
                f"survit pas à la projection")
            assert self.variant in INJECT_VARIANTS, (
                f"les métadonnées de la phase 11 voyagent avec les lignes "
                f"INJECTÉES : variante {INJECT_VARIANTS} exigée (reçu "
                f"{self.variant})")
        assert self.age_planes >= 1 and self.loc_planes >= 1
        assert self.age_ref >= 1, self.age_ref
        assert self.age_aug_max > 1.0, self.age_aug_max
        assert self.n_channels >= 2, self.n_channels
        assert self.rot_drift_max > 0.0, self.rot_drift_max
        if self.age_aug:
            assert self.bank_rot in ("age-log", "age-raw"), (
                f"age_aug tire l'ÉCHELLE des âges au train : il n'a de sens "
                f"que pour une famille qui ROTE par l'âge (reçu bank_rot="
                f"{self.bank_rot!r}) — sur `none`/`age-bias` il serait un "
                f"no-op silencieux (le biais est linéaire en a, donc invariant "
                f"d'échelle à un facteur près appris)")
        if self.tag_mode != "none":
            assert self.p11_env == "prov", (
                f"le tag de PROVENANCE code un canal user/self : hors de l'env "
                f"`prov` toutes les lignes ont le canal 0 et le tag serait un "
                f"no-op silencieux (reçu p11_env={self.p11_env!r})")
        assert self.prov_vals in ("ref", "span"), (
            f"prov_vals inconnu {self.prov_vals!r} (∈ ref|span)")
        if self.prov_vals != "ref":
            assert self.p11_env == "prov", (
                f"prov_vals ne concerne que l'env `prov` (reçu "
                f"p11_env={self.p11_env!r}) — ailleurs il serait un no-op "
                f"silencieux")
        assert self.life_vals in ("city", "span"), (
            f"life_vals inconnu {self.life_vals!r} (∈ city|span)")
        if self.life_vals != "city":
            assert self.p11_env == "life", (
                f"life_vals ne concerne que l'env `life` (reçu "
                f"p11_env={self.p11_env!r}) — ailleurs il serait un no-op "
                f"silencieux")
        if self.loc_mode != "none":
            assert self.top_k >= 2, (
                f"l'index LOCAL distingue les lignes DANS un write : à "
                f"top_k={self.top_k} il n'y a qu'une ligne de contenu par "
                f"groupe, le code serait un no-op silencieux")
        if self.p11_env != "rule":
            assert self.variant in INJECT_VARIANTS and \
                self.code in GROUP_CODES, (
                f"les envs `prov`/`span`/`life` sont des envs de CITATION à "
                f"injection de tous les résidents : variante "
                f"{INJECT_VARIANTS} + code {GROUP_CODES} exigés")
            assert not self.cond, (
                "`prov`/`span`/`life` remplacent la vie-règle : --cond n'a pas "
                "de sens avec eux (deux tâches, deux évals, un seul stream)")
        # ── phase 12 : maintenance procédurale (S6) et dilution (S8) ───────
        assert self.retention in RETENTION_SIGNALS, (
            f"retention inconnu {self.retention!r} (∈ {RETENTION_SIGNALS})")
        assert self.bank_fill in BANK_FILLS, (
            f"bank_fill inconnu {self.bank_fill!r} (∈ {BANK_FILLS})")
        assert self.p12_exam in ("",) + tuple(P12_EXAM_NAMES), (
            f"p12_exam inconnu {self.p12_exam!r} "
            f"(∈ {('',) + tuple(P12_EXAM_NAMES)})")
        assert self.prop_budget >= 0, self.prop_budget
        assert 0.0 < self.ema_beta < 1.0, self.ema_beta
        assert self.actr_decay > 0.0, self.actr_decay
        assert self.life_turns >= 8, self.life_turns
        assert self.fill_pool >= 1 and self.fill_refresh >= 1
        if self.retention != "fifo":
            assert self.prop_budget >= 1, (
                f"le signal de rétention {self.retention!r} ne sert QU'À "
                f"choisir quoi PROPAGER : à prop_budget 0 rien n'est propagé "
                f"et le bras serait bit-à-bit la FIFO nue (no-op silencieux)")
        if self.prop_budget:
            assert self.retention != "fifo", (
                "`fifo` EST la baseline sans propagation : demander un budget "
                "de propagation sans signal n'a pas de sens (quelles lignes "
                "propager ?) — choisir --retention age|attn-ema|coverage|actr")
            assert self.prop_budget < self.max_mem, (
                f"prop_budget {self.prop_budget} >= max_mem {self.max_mem} : "
                f"la propagation immobiliserait TOUTE la banque, plus rien ne "
                f"pourrait chuter et la FIFO n'existerait plus")
        if self.uses_retention or self.bank_fill != "none":
            assert self.p12_exam, (
                "toute cellule de la phase 12 DÉCLARE son examen "
                "(--p12-exam retention|dilution)")
            assert not self.p11_exam, (
                "un run appartient à UN examen : p11_exam et p12_exam "
                "s'excluent (sinon deux espaces de noms revendiqueraient le "
                "même dossier)")
        if self.uses_retention:
            assert self.p11_env == "life", (
                f"la maintenance procédurale (§2.3) se juge sur des vies "
                f"LONGUES à supersession tardive : elle exige --p11-env life "
                f"(reçu {self.p11_env!r}) — ailleurs aucune ligne n'atteint "
                f"jamais le bord de la FIFO et le signal serait un no-op")
        if self.bank_fill != "none":
            assert self.variant in INJECT_VARIANTS and \
                self.code in GROUP_CODES, (
                f"le remplissage de banque (S8) pose des GROUPES entiers : "
                f"variante {INJECT_VARIANTS} + code {GROUP_CODES} exigés")
        if self.p11_env == "life":
            assert self.read_path == "kvproj", (
                f"la phase 12 se joue sur la forme ADOPTÉE du read (S1 : "
                f"kvproj) — reçu read_path={self.read_path!r}")
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

    # ── phase 11 : les familles actives, et les plans qu'elles réservent ────

    @property
    def uses_p11_meta(self) -> bool:
        """Une famille de métadonnées (§2.5) est-elle active ?

        `age-bias` en fait partie : il ne rote rien, mais il consomme le MÊME
        tenseur de métadonnées (l'âge par ligne) et le même chemin de code.
        """
        return (self.bank_rot != "none" or self.tag_mode != "none"
                or self.loc_mode != "none")

    @property
    def uses_bank_rot(self) -> bool:
        """Une famille tourne-t-elle VRAIMENT des plans de K' ?"""
        return (self.bank_rot in ("age-log", "age-raw")
                or self.tag_mode != "none" or self.loc_mode != "none")

    @property
    def rot_plane_budget(self) -> tuple:
        """(n_âge, n_canal, n_local) — plans RÉSERVÉS par famille, DISJOINTS.

        Le budget de dims est le MÊME pour un bras rotatif et son steelman
        additif (`add` réserve les mêmes plans qu'il n'aurait rotés) : sans
        ça l'A/B mesurerait la taille de la bande réservée, pas la forme du
        code.
        """
        na = self.age_planes if self.bank_rot in ("age-log", "age-raw") else 0
        nc = self.n_channels if self.tag_mode != "none" else 0
        nl = self.loc_planes if self.loc_mode != "none" else 0
        return na, nc, nl

    # ── phase 12 : la maintenance est-elle autre chose que la FIFO nue ? ────

    @property
    def uses_retention(self) -> bool:
        """Un signal de rétention PROPAGE-t-il vraiment quelque chose ?

        `fifo` à budget 0 est la baseline : elle ne « s'active » pas, elle EST
        le comportement historique (et le self-test le prouve bit-à-bit).
        """
        return self.retention != "fifo" or self.prop_budget > 0

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
        # ── PHASE 10, `read_path='kvproj'` : PROJECTIONS DÉDIÉES, SOFTMAX
        # UNIFIÉ (3ᵉ sommet du carré factoriel, cf. READ_MODES) ─────────────
        # Construites EN DERNIER et seulement pour ce chemin : les tirages de
        # poids des chemins existants ne bougent pas d'un bit.
        self.kvproj = cfg.read_path == "kvproj"
        self.bank_q = self.kvproj and cfg.bank_q
        # SONDE d'attention banque (phase 11, S5) : OFF par défaut et sans
        # aucun effet sur le forward — cf. `bank_attn_probe`.
        self.want_bank_attn = False
        self.last_bank_attn = None
        if self.kvproj:
            self.bk = nn.Linear(d, d, bias=False)
            self.bv = nn.Linear(d, d, bias=False)
            # BIAIS SCALAIRE PAR TÊTE sur les logits de banque, init 0. Avec S'
            # lignes qui concourent contre une fenêtre entière dans UN SEUL
            # softmax, la masse allouée à la banque est à la merci du rapport
            # S'/T — c'est précisément le défaut que `dual_heads` contourne en
            # séparant les softmax. Ici on le traite de face : un degré de
            # liberté appris, et sa VALEUR FINALE est une mesure (loggée dans
            # results.json) — positive = le modèle a dû REMONTER la banque
            # contre le contexte, négative = il a dû la faire taire.
            self.bank_bias = nn.Parameter(torch.zeros(h))
            # ── PHASE 11 : les métadonnées de ligne, sur K' (spec §2.5) ────
            # Construit APRÈS bank_bias et AVANT bq/bo : les configs de la
            # ph.10 (rot=none) ne créent rien et ne déplacent aucun tirage.
            self.rot = BankRot(cfg, h, self.dh, self.theta) \
                if cfg.uses_p11_meta else None
            if self.bank_q:
                # Les lignes ÉMETTENT aussi : elles attendent sur [banque ∪
                # contexte] et leur état est porté de couche lectrice en
                # couche lectrice. W_o' zéro-init ⇒ à l'init les lanes valent
                # EXACTEMENT les lignes brutes et le bras dégénère proprement
                # en `kvproj` sans bank-q.
                self.bq = nn.Linear(d, d, bias=False)
                self.bo = nn.Linear(d, d, bias=False)
                nn.init.zeros_(self.bo.weight)
                self.nb = RMSNorm(d)

    def forward(self, x, pos=None, mem=None, mem_mask=None,
                want_bank_out=False, mem_meta=None):
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
            y = self.o(y.transpose(1, 2).reshape(B, T, d))
            return (y, None) if want_bank_out else y
        S = mem.shape[1]
        if self.kvproj:
            # ── SOMMET « DÉDIÉES / UNIFIÉ » ────────────────────────────────
            # Seule différence avec `kv_append` : les lignes passent par W_k'
            # et W_v' au lieu d'entrer brutes. C'est LE delta qui isole
            # l'effet « géométrie de clés » — tout le reste (même softmax,
            # mêmes têtes, même masque, pas de RoPE sur les lignes) est
            # identique.
            hb = self.nb(mem) if self.bank_q else mem
            km = self.bk(hb).view(B, S, self.h, self.dh).transpose(1, 2)
            vm = self.bv(hb).view(B, S, self.h, self.dh).transpose(1, 2)
            # ── PHASE 11 : le codage des métadonnées frappe ICI, sur K' —
            # APRÈS W_K' (une rotation posée avant ne survit pas à la
            # projection) et JAMAIS sur V' (le contenu doit sortir intact).
            rbias = None
            if self.rot is not None and mem_meta is not None:
                km = self.rot(km, mem_meta)
                rbias = self.rot.logit_bias(mem_meta, q.dtype)
            k = torch.cat([km, k], dim=2)
            v = torch.cat([vm, v], dim=2)
            # masque FLOTTANT : il porte à la fois l'interdiction (−inf) et le
            # biais par tête sur les colonnes de banque.
            fm = torch.zeros(B, self.h, T, S + T, device=x.device,
                             dtype=q.dtype)
            fm[..., :S] += self.bank_bias.to(q.dtype)[None, :, None, None]
            if rbias is not None:
                # `age-bias` : le BIAIS SCALAIRE DE RÉCENCE, b(a) = w·a, sur
                # les colonnes banque. Un paramètre, aucune rotation — c'est le
                # fallback de la règle S3.
                fm[..., :S] += rbias[:, None, None, :]
            neg = torch.finfo(q.dtype).min
            causal = torch.ones(T, T, dtype=torch.bool,
                                device=x.device).tril()
            fm[..., S:] = torch.where(causal, fm.new_zeros(()),
                                      fm.new_full((), neg))
            if mem_mask is not None:
                fm[..., :S] = torch.where(
                    mem_mask[:, None, None, :].to(torch.bool),
                    fm[..., :S], fm.new_full((), neg))
            if self.want_bank_attn:
                # ── SONDE (S5) : la MASSE D'ATTENTION par ligne de banque ──
                # `scaled_dot_product_attention` ne rend pas ses poids. On
                # recalcule le softmax EXPLICITEMENT — même logits, même
                # masque flottant — et on ne garde que les S colonnes de
                # banque. Ce chemin ne touche PAS `y` : la sortie du modèle
                # reste celle du SDPA, la sonde n'est qu'un observateur.
                sc = (q.float() @ k.float().transpose(-1, -2)) / math.sqrt(
                    self.dh) + fm.float()
                self.last_bank_attn = torch.softmax(
                    sc, -1)[..., :S].detach()          # [B, h, T, S]
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=fm)
            y = self.o(y.transpose(1, 2).reshape(B, T, d))
            mem_out = None
            if self.bank_q and want_bank_out:
                # LES LIGNES ÉMETTENT : requêtes dédiées, visibilité TOTALE
                # (elles ne sont pas dans l'ordre du temps — ni causalité, ni
                # RoPE), sortie résiduelle sur l'état de lane.
                qb = self.bq(hb).view(B, S, self.h, self.dh).transpose(1, 2)
                fb = torch.zeros(B, self.h, S, S + T, device=x.device,
                                 dtype=q.dtype)
                fb[..., :S] += self.bank_bias.to(q.dtype)[None, :, None, None]
                if rbias is not None:
                    fb[..., :S] += rbias[:, None, None, :]
                if mem_mask is not None:
                    fb[..., :S] = torch.where(
                        mem_mask[:, None, None, :].to(torch.bool),
                        fb[..., :S], fb.new_full((), neg))
                yb = F.scaled_dot_product_attention(qb, k, v, attn_mask=fb)
                mem_out = mem + self.bo(
                    yb.transpose(1, 2).reshape(B, S, d))
            return (y, mem_out) if want_bank_out else y
        # ── `kv_append` : chemin HISTORIQUE, inchangé au bit près ──────────
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
        y = self.o(y.transpose(1, 2).reshape(B, T, d))
        return (y, None) if want_bank_out else y


class BankHeads(nn.Module):
    """PHASE 10 — un GROUPE DE TÊTES DÉDIÉ à la banque, à côté des têtes du
    contexte (design user, après le verdict de la grille).

    POURQUOI, ET CONTRE QUOI. `kv_append` (lecture β) fait entrer les lignes
    dans les K/V des têtes EXISTANTES : la banque et le contexte se disputent
    alors la masse softmax DANS la même tête, avec la MÊME requête. Le jouet ne
    peut pas le voir (fenêtres courtes, 2AFC déjà saturé à 1.000), mais c'est
    le risque n°1 au 350M à fenêtre longue — plus le contexte est long, plus la
    banque se fait diluer, sans qu'aucune métrique du jouet ne l'annonce. Les
    têtes dédiées garantissent la masse : le softmax du groupe 2 porte sur la
    banque SEULE, il ne peut pas être capté par le contexte.

    CE QUE LA TÊTE BANQUE N'A PAS, ET POURQUOI :
      * pas de RoPE — les lignes sont un ENSEMBLE, pas une séquence. Leur ordre
        dans la matrice plate est un artefact du FIFO, pas une information.
      * pas de masque causal — toute position du tour peut lire toute ligne.
      * la seule ancienneté qui compte est portée par le CODE D'ÂGE
        (`code.age_rope`), appliqué aux lignes AVANT d'arriver ici : la tête
        n'a rien à réinventer.

    RECUL ASSUMÉ. Contrairement à `kv_append`, ce bras a des paramètres
    dédiés (W_q/W_k/W_v/W_o) : on abandonne « aucune matmul dédiée » pour
    acheter la masse et une géométrie de clés propre. Le coût est borné —
    quelques têtes suffisent (cf. cfg.bank_heads / cfg.bank_head_dim).

    ── L'INIT, ET LE PIÈGE r3 ───────────────────────────────────────────────
    Le labo a déjà payé une fois le prix d'une porte MULTIPLICATIVE zéro-init
    (r3/r1 : `gate` scalaire à 0 ⇒ le gradient des paramètres INTERNES vaut
    g·(…) = 0, seule la porte bouge, et elle ne bouge que si quelque chose la
    tire — tête morte). Ici c'est la PROJECTION DE SORTIE `o` qui est
    zéro-initialisée, pas un scalaire :
      * forward au step 0 : la sortie est EXACTEMENT nulle ⇒ le bloc est
        bit-à-bit le backbone (critère habituel du labo) ;
      * backward au step 0 : ∂L/∂W_o = (∂L/∂y)·headᵀ avec `head` NON NUL (q/k/v
        sont initialisés normalement) ⇒ W_o reçoit du gradient DÈS LE PREMIER
        PAS et quitte zéro. C'est la différence structurelle avec la porte
        scalaire, et le self-test la mesure (W_o non nul après quelques pas,
        et le forward qui s'écarte du backbone).
    """

    def __init__(self, cfg: ToyCfg):
        super().__init__()
        d = cfg.d_model
        self.hb = int(cfg.bank_heads)
        self.dhb = int(cfg.bank_head_dim or (d // cfg.n_heads))
        inner = self.hb * self.dhb
        self.q = nn.Linear(d, inner, bias=False)
        self.k = nn.Linear(d, inner, bias=False)
        self.v = nn.Linear(d, inner, bias=False)
        self.o = nn.Linear(inner, d, bias=False)
        nn.init.zeros_(self.o.weight)      # cf. docstring : SORTIE, pas porte
        # SONDE d'attention banque : OFF par défaut, sans effet sur le forward
        # (cf. `bank_attn_probe`). Elle vit ici aussi depuis la ph.12 — sans
        # elle le bras `dual_heads` n'aurait pas de r@1, et la courbe de
        # dilution ne pourrait pas comparer les deux softmax sur la SÉLECTION.
        self.want_bank_attn = False
        self.last_bank_attn = None

    def forward(self, h, mem, mem_mask=None):
        """h [B,T,d] (déjà pré-normé par le bloc) × mem [B,S,d] → [B,T,d]."""
        B, T, _ = h.shape
        S = mem.shape[1]
        mm = mem.to(h.dtype)
        q = self.q(h).view(B, T, self.hb, self.dhb).transpose(1, 2)
        k = self.k(mm).view(B, S, self.hb, self.dhb).transpose(1, 2)
        v = self.v(mm).view(B, S, self.hb, self.dhb).transpose(1, 2)
        am = None
        if mem_mask is not None:
            am = mem_mask[:, None, None, :].to(torch.bool).expand(B, 1, T, S)
        if self.want_bank_attn:
            # softmax recalculé EN CLAIR pour l'observer — `y` reste celui du
            # SDPA, la sonde n'est qu'un observateur.
            sc = (q.float() @ k.float().transpose(-1, -2)) / math.sqrt(self.dhb)
            if am is not None:
                sc = sc.masked_fill(~am, torch.finfo(sc.dtype).min)
            self.last_bank_attn = torch.softmax(sc, -1).detach()
        # NI causal, NI RoPE : la banque est un ensemble (cf. docstring).
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=am)
        return self.o(y.transpose(1, 2).reshape(B, T, self.hb * self.dhb))


def slow_rope_planes(d_head: int, theta: float, n_planes: int, T: int
                     ) -> tuple:
    """(index des n_planes paires les PLUS LENTES du RoPE de tête, dérive).

    L'APPARIEMENT DE LA SPEC §2.5, rendu explicite. Le score banque de `kvproj`
    vaut (R(t)·W_Q x_t)ᵀ K' : la requête porte le RoPE du backbone, la clé
    banque ne l'annule pas. Le produit se faisant PAIRE À PAIRE, une
    métadonnée posée sur la paire i se lit à travers cos(ω_i·t − φ_méta) — le
    code se mélange à la position du lecteur dans la fenêtre, exactement ce que
    la spec interdit. La parade est de DOCKER les plans de métadonnées sur les
    paires quasi statiques (ω ≈ 0), où R(t) ne bouge pratiquement pas sur toute
    la fenêtre.

    Les fréquences RoPE ω_i = θ^(−2i/d_head) sont décroissantes en i : les
    paires lentes sont les DERNIÈRES. On ne s'appuie pas sur cette monotonie —
    on TRIE, pour que le choix reste correct si la forme des fréquences change.

    `dérive` = ω_max·(T−1) radians, la variation totale de l'angle de requête
    sur la plus rapide des paires retenues, sur toute la fenêtre. C'est le
    chiffre que `rot_drift_max` borne et que le self-test vérifie.
    """
    inv = 1.0 / (theta ** (torch.arange(0, d_head, 2).float() / d_head))
    assert n_planes <= inv.numel(), (
        f"{n_planes} plans demandés pour {inv.numel()} paires de dims de tête")
    order = torch.argsort(inv)                     # du plus LENT au plus rapide
    idx = order[:n_planes].sort().values           # ordre de dim, stable
    drift = float(inv[idx].max()) * max(T - 1, 0)
    return idx, drift


class BankRot(nn.Module):
    """PHASE 11 — les métadonnées de ligne, sur les CLÉS BANQUE PROJETÉES.

    Applique à K' = W_K'·g (dans `CausalSelfAttn`, chemin `kvproj`) les trois
    familles de la spec §2.5, sur des plans DISJOINTS et QUASI STATIQUES :

      ÂGE (S3/S4)          rot(ω_p·φ(a)) sur `age_planes` plans, ω_p APPRIS
                           (init géométrique entrelacé, sans bande haute —
                           veille LieRE/VideoRoPE). φ = log-comprimé
                           (`age-log`) ou brut (`age-raw`).
      PROVENANCE (S5)      UN PLAN 0/π PAR CANAL. Jamais n angles sur un seul
                           plan : à n=4, tag0·tag2 = −1 et tag0·tag1 = 0 — une
                           métrique cyclique parasite entre canaux qui n'ont
                           aucun ordre. `add` remplace la rotation par un
                           vecteur appris sur les MÊMES dims (steelman).
      INDEX LOCAL (S17)    R_loc(j) sur `loc_planes` plans, fréquences DFT
                           fixes (l'index est borné par top_k+1 ⇒ aucun OOD
                           possible). Argument structurel : entre deux lignes
                           d'un même span, « avancer d'un token » devient
                           l'opérateur CONSTANT R_loc(1) — le geste du circuit
                           de copie. `add` ne donne que des signatures.

    CE QUI N'EST PAS ROTÉ, ET POURQUOI : V' (le contenu doit sortir intact — la
    citation copie la valeur, pas une valeur tournée) et les requêtes de
    `bank_q` (S2 est un axe séparé ; le combiner ici mélangerait deux tests).

    INIT : rotations à angle non nul dès le pas 0 (une rotation N'EST PAS une
    porte — elle préserve la norme et le gradient des fréquences est non nul),
    vecteurs additifs à ZÉRO (le bras `add` démarre donc exactement sur le
    kvproj nu, comme le `rot` démarre sur des plans dont le code est déjà
    présent : les deux sont comparables et aucun n'a de gradient interne mort,
    cf. le piège de la porte scalaire).
    """

    def __init__(self, cfg: ToyCfg, n_heads: int, d_head: int, theta: float):
        super().__init__()
        na, nc, nl = cfg.rot_plane_budget
        self.na, self.nc, self.nl = na, nc, nl
        self.mode = cfg.bank_rot
        self.tag_mode, self.loc_mode = cfg.tag_mode, cfg.loc_mode
        self.age_ref = float(cfg.age_ref)
        self.n_loc = int(cfg.group_rows)          # index locaux possibles
        total = na + nc + nl
        # `age-bias` seul ne réserve AUCUN plan (il ne tourne rien) : pas
        # d'appariement à vérifier, et la garde ne doit pas s'inventer une
        # dérive sur un ensemble vide.
        idx = (slow_rope_planes(d_head, theta, total, cfg.max_seq_len)[0]
               if total else torch.zeros(0, dtype=torch.long))
        drift = (float(slow_rope_planes(d_head, theta, total,
                                        cfg.max_seq_len)[1]) if total else 0.0)
        assert drift <= cfg.rot_drift_max, (
            f"APPARIEMENT REFUSÉ (§2.5) : {total} plans de métadonnées "
            f"débordent de la bande quasi statique — la requête tourne de "
            f"{drift:.3f} rad sur la fenêtre (max toléré "
            f"{cfg.rot_drift_max}). Le code de métadonnée se mélangerait à la "
            f"position du lecteur. Réduire age_planes/loc_planes, ou assumer "
            f"le fallback « dé-roter q pour les colonnes banque ».")
        self.drift = drift
        # ATTRIBUTION : l'âge prend les plans LES PLUS LENTS (c'est lui qui a
        # la plus grande dynamique et le plus à perdre à une contamination),
        # puis le canal, puis l'index local (borné et à faible dynamique).
        self.register_buffer("age_idx", idx[:na])
        self.register_buffer("tag_idx", idx[na:na + nc])
        self.register_buffer("loc_idx", idx[na + nc:])
        self.n_pairs = d_head // 2
        if na:
            # ÉCHELLES GÉOMÉTRIQUES ENTRELACÉES, SANS BANDE HAUTE : λ_p de
            # ~4 tours à ~4·A_ref (veille). ω = 2π/λ, appris en LOG (donc
            # toujours > 0 et paramétré multiplicativement).
            lam = 4.0 * (cfg.age_ref ** (torch.arange(na).float()
                                         / max(na - 1, 1)))
            self.age_log_omega = nn.Parameter(
                torch.log(2.0 * math.pi / lam))
        if cfg.bank_rot == "age-bias":
            # LE FALLBACK DE LA RÈGLE S3 : b(a) = w·a sur le logit banque, UN
            # paramètre. Init 0 ⇒ bit-à-bit le kvproj nu au pas 0, et son
            # gradient est non nul dès le premier backward (c'est un terme
            # ADDITIF au logit, pas une porte multiplicative).
            self.age_bias_w = nn.Parameter(torch.zeros(1))
        if nc and cfg.tag_mode == "add":
            # STEELMAN ADDITIF : un vecteur appris PAR CANAL, sur les 2·nc dims
            # exactement réservées par le bras rotatif. Par TÊTE (le rotatif
            # agit lui aussi sur chaque tête).
            self.tag_add = nn.Parameter(torch.zeros(n_heads, cfg.n_channels,
                                                    2 * nc))
        if nl and cfg.loc_mode == "add":
            self.loc_add = nn.Parameter(torch.zeros(n_heads, self.n_loc,
                                                    2 * nl))

    def extra_repr(self) -> str:
        return (f"age={self.mode}({self.na}p) tag={self.tag_mode}({self.nc}p) "
                f"loc={self.loc_mode}({self.nl}p) drift={self.drift:.4f}rad")

    def phi_age(self, a: torch.Tensor) -> torch.Tensor:
        """Compression de l'âge AVANT rotation (§2.5).

        `age-log` : φ(a) = A_ref·log(1+a)/log(1+A_ref) — φ(A_ref) = A_ref, donc
        les deux bras coïncident dans la plage vue au train et ne divergent
        QU'EN OOD. C'est ce qui rend S4 lisible : à échelle 1 les deux bras
        sont dans le même régime, à ×100 seul le log tient.
        """
        if self.mode == "age-log":
            return (self.age_ref * torch.log1p(a)
                    / math.log1p(self.age_ref))
        return a

    def forward(self, km: torch.Tensor, meta: torch.Tensor) -> torch.Tensor:
        """km [B, H, S, dh] (clés banque projetées) × meta [B, S, 3] long
        (âge en writes, canal, index local dans le write) → clés codées."""
        B, H, S, dh = km.shape
        # Les ANGLES vivent en float32 quel que soit l'autocast : (a) sous AMP
        # la politique upcaste log1p ⇒ phi ressortait float32 et l'index-put
        # dans un ang bf16 plantait (jobs zzr103/104, 08-03 nuit) ; (b) des
        # phases mod 2π en bf16 (~8 bits de mantisse) perdraient la précision
        # que le code d'âge exige. cos/sin ∈ [−1,1] sont recastés vers le
        # dtype des clés à l'application seulement.
        ang = torch.zeros(B, S, self.n_pairs, device=km.device,
                          dtype=torch.float32)
        touched = False
        if self.na:
            phi = self.phi_age(meta[..., 0].float())               # [B,S] f32
            ang[..., self.age_idx] = (phi[..., None]
                                      * torch.exp(self.age_log_omega)
                                      .float()[None, None])
            touched = True
        if self.nc and self.tag_mode == "rot":
            # UN PLAN PAR CANAL, ANGLE 0 OU π. R(π)² = I ⇒ le code n'est pas
            # orienté (il ne distingue pas « self lit user » de l'inverse) —
            # assumé : le canal est un attribut de la ligne, pas une relation.
            ch = meta[..., 1].long()                               # [B,S]
            for c in range(self.nc):
                ang[..., self.tag_idx[c]] = math.pi * (ch == c).float()
            touched = True
        if self.nl and self.loc_mode == "rot":
            # FRÉQUENCES DFT sur l'index local : θ_p = 2π(p+1)/n_loc. Le
            # successeur est alors l'opérateur CONSTANT R_loc(1) sur chaque
            # plan, indépendamment du contenu — l'argument de la spec.
            j = meta[..., 2].float()
            w = (2.0 * math.pi
                 * (torch.arange(self.nl, device=km.device).float() + 1.0)
                 / float(self.n_loc))
            ang[..., self.loc_idx] = j[..., None] * w[None, None]
            touched = True
        if touched:
            cos, sin = torch.cos(ang), torch.sin(ang)
            # les plans NON touchés ont ang = 0 ⇒ cos 1 / sin 0 = identité
            km = rot_pairs(km, cos.to(km.dtype)[:, None],
                           sin.to(km.dtype)[:, None])
        if self.nc and self.tag_mode == "add":
            dims = torch.stack([2 * self.tag_idx, 2 * self.tag_idx + 1],
                               -1).reshape(-1)                     # [2·nc]
            v = self.tag_add[:, meta[..., 1].long()]      # [H,B,S,2nc]
            km = km.index_add(-1, dims, v.permute(1, 0, 2, 3).to(km.dtype))
        if self.nl and self.loc_mode == "add":
            dims = torch.stack([2 * self.loc_idx, 2 * self.loc_idx + 1],
                               -1).reshape(-1)
            v = self.loc_add[:, meta[..., 2].long().clamp(0, self.n_loc - 1)]
            km = km.index_add(-1, dims, v.permute(1, 0, 2, 3).to(km.dtype))
        return km

    def logit_bias(self, meta: torch.Tensor, dtype) -> torch.Tensor | None:
        """b(a) = w·a sur les colonnes banque (`age-bias`) — [B, S] ou None."""
        if self.mode != "age-bias":
            return None
        return (self.age_bias_w.to(dtype)
                * meta[..., 0].to(dtype))


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

    ── `cfg.fw_additive` (phase 10, correctif terrain) ──────────────────────
    VERDICT MESURÉ sur la ferme (6 premières cellules `seq_fw` de la grille
    §2.4, ~1500 steps) : le bras DIVERGE dans 5 cellules sur 6 — gnorm PRÉ-CLIP
    de 1e4 à 4.7e9, loss coincée entre 6.8 et 11.3, et Δnll de citation NÉGATIF
    (la banque NUIT contre le bras ablaté). Seule cellule saine :
    `rot-on_tap-mid_m1` (loss 3.59, gnorm 1.26) — c'est-à-dire celle qui a le
    MOINS d'itérations. Le `grad_clip` du harnais n'y peut rien : il rescale la
    NORME, mais la DIRECTION reste dominée par le produit de jacobiennes
    accumulé sur 16 à 72 itérations × n_layers couches.

    DIAGNOSTIC. La boucle est déjà résiduelle en FORWARD (`y = y + upd`), mais
    pas en BACKWARD : `upd_i` est calculé à partir de `y_{i-1}`, donc
    ∂y_M/∂θ_i traverse ∏_{j>i} (I + ∂upd_j/∂y). Ce produit est exactement ce
    qui explose — c'est la pathologie classique du RNN profond, ici avec M×L
    étages et aucune porte pour l'amortir.

    LE FIX. Stop-gradient sur l'ÉTAT PORTÉ : chaque itération lit
    `yin = y0 + (y - y0).detach()` au lieu de `y`.
      * `y0` GARDE son gradient  ⇒ le chemin REQUÊTE apprend encore (h → y0 →
        chaque upd_i), c'est lui qui porte « ce que la couche demande à la
        banque ».
      * l'ACCUMULÉ `(y - y0)` est coupé ⇒ plus AUCUN produit de jacobiennes.
      * chaque slot garde le gradient LOCAL de son propre `upd_i` par le terme
        additif final (`y - y0 = Σ_i upd_i`), donc fw_A/fw_B/la banque du slot
        i reçoivent toujours un signal — il est simplement devenu ADDITIF au
        lieu de MULTIPLICATIF.
    PROPRIÉTÉ CLÉ : `y0 + (y - y0).detach()` vaut NUMÉRIQUEMENT `y`. Le
    forward est donc identique bit-à-bit ; SEUL le backward change. C'est ce
    qui rend le flag comparable au bras multiplicatif à poids égaux.

    Défaut OFF : le bras multiplicatif divergent EST une donnée de la grille,
    on ne le réécrit pas rétroactivement.
    """

    def __init__(self, cfg: ToyCfg):
        super().__init__()
        d, r = cfg.d_model, cfg.fw_rank
        self.r, self.d = r, d
        self.additive = bool(cfg.fw_additive)
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
            # `yin` == `y` en VALEUR, au BIT PRÈS, et n'en diffère que par le
            # graphe. ⚠️ L'ÉCRITURE COMPTE : `y0 + (y − y0).detach()` a la
            # bonne sémantique mais PAS la bonne arithmétique — en flottant,
            # a + (b − a) ≠ b (mesuré : le self-test de bit-à-bit tombe).
            # `y.detach() + (y0 − y0.detach())` ajoute un tenseur EXACTEMENT
            # nul (x − x = 0 pour tout x fini) et porte le gradient de y0 :
            #   ∂yin/∂y0 = 1        ⇒ le chemin REQUÊTE apprend
            #   ∂yin/∂(Σ upd_j) = 0 ⇒ plus aucun produit de jacobiennes
            yin = (y.detach() + (y0 - y0.detach())) if self.additive else y
            zg = torch.einsum("brd,btd->btr", A[:, i, 0], yin) * ds
            zv = torch.einsum("brd,btd->btr", A[:, i, 1], yin) * ds
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
        # `mem` va aux K/V des têtes EXISTANTES (lecture β `kv`) ou à un GROUPE
        # DE TÊTES DÉDIÉ (`dual`) — jamais aux deux. Construit EN DERNIER : les
        # chemins existants tirent leurs poids dans le même ordre qu'avant.
        self.mem_in_attn = cfg.read_path in ("kv", "kvproj")
        self.bank_q = cfg.read_path == "kvproj" and cfg.bank_q
        self.bank_heads = (BankHeads(cfg)
                           if self.read_bank and cfg.read_path == "dual"
                           else None)

    def forward(self, x, bank, bank_mask, pos=None, mem=None, mem_mask=None,
                mem_meta=None):
        # `mem` n'entre QUE dans les couches LECTRICES (`read_layers`) : les
        # lectures β et « têtes dédiées » partagent exactement le même budget
        # de couches que les reads appris de r0/r1/r3, sinon la comparaison
        # porterait aussi sur « à combien d'étages la banque parle ».
        h1 = self.n1(x)
        use_mem = self.read_bank and self.mem_in_attn
        if self.bank_q and use_mem and mem is not None:
            # LES LANES BANQUE SONT PORTÉES : le bloc rend l'état mis à jour,
            # ToyReadLM le repasse à la couche lectrice suivante et le JETTE
            # après la dernière (cf. son forward).
            a, mem = self.attn(h1, pos, mem, mem_mask, want_bank_out=True,
                               mem_meta=mem_meta)
        else:
            a = self.attn(h1, pos, mem if use_mem else None,
                          mem_mask if use_mem else None,
                          mem_meta=mem_meta if use_mem else None)
        if self.bank_heads is not None and mem is not None:
            # FUSION AVANT LE FFN : les deux groupes de têtes lisent le MÊME
            # état pré-normé et leurs sorties se somment dans le résiduel.
            a = a + self.bank_heads(h1, mem, mem_mask)
        x = x + a
        if self.read is not None and bank is not None and bank.size(1) > 0:
            x = self.read(x, bank, bank_mask)
        x = x + self.mlp(self.n2(x))
        # tuple UNIQUEMENT quand des lanes existent : les chemins qui appellent
        # le bloc nu (prélèvement midhid, self-tests) reçoivent x brut — un
        # bloc bank_q à mem=None rendait (x, None) et le RMSNorm du bloc
        # suivant croquait le tuple (6 cellules bq×tap-mid du carré, 08-03).
        return (x, mem) if (self.bank_q and mem is not None) else x


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

    @torch.no_grad()
    def group_rows_batch(self, seg_toks: list) -> torch.Tensor:
        """PHASE 12 — les lignes de contenu de n segments, EN UN LOT.

        Le pendant BATCHÉ de `tophid_rows_fixed`/`toprows_sel_fixed`, pour le
        remplissage de banque de S8 : sans lui, remplir une banque de 64
        groupes coûterait 63 forwards batch-1 par plan.

        Le padding est à DROITE et l'attention est CAUSALE : une position non
        padée n'attend jamais une position padée, donc les états rendus sont
        ceux des segments isolés (aux réductions flottantes près, qui sont
        déterministes à composition de lot fixée). Les codes à ID ne
        forwardent rien — la sélection SIF est une table.
        """
        c = self.cfg
        dev = self.embed.weight.device
        segs = [t.reshape(-1)[:c.max_seq_len] for t in seg_toks]
        if c.code not in HID_CODES:
            return torch.stack([self.toprows_sel_fixed(t) for t in segs])
        n, T = len(segs), max(int(t.numel()) for t in segs)
        X = torch.zeros(n, max(T, 1), dtype=torch.long, device=dev)
        for i, t in enumerate(segs):
            X[i, :t.numel()] = t.to(dev)
        if c.code != "midhid":
            _, H = self.forward(X, bank=None, bank_mask=None,
                                return_hidden=True)
        else:
            H = self.embed(X)
            for blk in self.blocks[:c.hid_tap_layers]:
                H = blk(H, None, None, None)
        H = H.float()
        out = torch.zeros(n, c.top_k, c.d_model, device=dev)
        for i, t in enumerate(segs):
            idx = self.toprows_sel_idx(t)
            m = int(idx.numel())
            if m == 0:
                continue
            e = rms_unit(H[i][idx.to(dev)])
            out[i] = e[:c.top_k] if m >= c.top_k else torch.cat(
                [e, e[-1].expand(c.top_k - m, c.d_model)])
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
    def forward(self, ids, bank=None, bank_mask=None, inject=None,
                return_hidden=False, inject_age=None, bank_age=None,
                inject_chan=None):
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
        mem_meta = None
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
            if self.cfg.read_path in ("kv", "dual", "kvproj"):
                # ── LECTURE β : VUE PLATE + ATTENTION ───────────────────────
                # Pas de préfixe, pas de séparateur, pas de position RoPE : le
                # tenseur (G, k, d) est simplement APLATI en (G·k, d) et posé
                # aux K/V des couches lectrices. L'ordre des lignes n'a plus
                # aucune conséquence géométrique — ce qui porte la provenance,
                # c'est la rotation d'âge, et rien d'autre.
                mem = pre.reshape(B, G * k, -1)
                if self.cfg.uses_p11_meta:
                    # ── PHASE 11 : LE TENSEUR DE MÉTADONNÉES [B, G·k, 3] ───
                    # (âge en writes, canal, index local dans le write). Les
                    # trois sont des propriétés DE LA LIGNE — jamais de la
                    # position du lecteur (principe §2.5). L'âge et le canal
                    # sont des propriétés du SLOT (partagées par les lignes du
                    # groupe), l'index local est le rang de la ligne DANS son
                    # groupe, donc il se lit dans le layout lui-même.
                    dev = ids.device
                    ag = (inject_age.to(dev).long() if inject_age is not None
                          else torch.zeros(B, G, dtype=torch.long, device=dev))
                    ch = (inject_chan.to(dev).long() if inject_chan is not None
                          else torch.zeros(B, G, dtype=torch.long, device=dev))
                    lo = torch.arange(k, device=dev)[None, None].expand(B, G, k)
                    mem_meta = torch.stack(
                        [ag[..., None].expand(B, G, k),
                         ch[..., None].expand(B, G, k), lo], -1
                    ).reshape(B, G * k, 3)
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
        carry = self.cfg.read_path == "kvproj" and self.cfg.bank_q
        for blk in self.blocks:
            out = blk(x, bank, bank_mask, pos, mem, mem_meta=mem_meta)
            # `bank_q` : l'état des lanes banque VIT le temps du stack et est
            # jeté à la sortie (rien ne le réécrit dans la banque — l'invariant
            # « seule modification de la banque = l'append d'un write » tient).
            x, mem = out if (carry and isinstance(out, tuple)) else (out, mem)
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
               inject=None, inject_age=None, bank_age=None, inject_chan=None):
        ids = prefix
        out = []
        for _ in range(max_new):
            lg = self.forward(ids[:, -self.cfg.max_seq_len:], bank, bank_mask,
                              inject=inject, inject_age=inject_age,
                              bank_age=bank_age, inject_chan=inject_chan)
            nxt = int(lg[0, -1].argmax())
            if nxt == stop_id:
                break
            out.append(nxt)
            ids = torch.cat([ids, torch.tensor([[nxt]], device=ids.device)], 1)
        return out


class bank_attn_probe:
    """Contexte : allume la sonde de masse d'attention banque (phase 11, S5).

    Elle ne change RIEN au forward (cf. `CausalSelfAttn.forward`) — elle
    recalcule le softmax en clair pour l'observer. À la sortie du contexte,
    tout est éteint et les tampons libérés.

    Pourquoi une sonde plutôt qu'un score de retriever : dans `kvproj` il n'y a
    PAS de module de sélection — la sélection EST l'attention. Mesurer un
    « r@1 » sur autre chose que ses poids mesurerait un autre système.
    """

    def __init__(self, model):
        # `kvproj` : la sélection EST l'attention du softmax unifié.
        # `dual_heads` (ph.12) : elle est celle du groupe de têtes DÉDIÉ —
        # même geste, autre softmax. C'est exactement le contraste que la
        # courbe de dilution S8 doit pouvoir mesurer sur les deux bras.
        self.mods = [b.attn for b in model.blocks
                     if getattr(b.attn, "kvproj", False)]
        self.mods += [b.bank_heads for b in model.blocks
                      if getattr(b, "bank_heads", None) is not None]

    def __enter__(self):
        for m in self.mods:
            m.want_bank_attn, m.last_bank_attn = True, None
        return self

    def __exit__(self, *a):
        for m in self.mods:
            m.want_bank_attn, m.last_bank_attn = False, None
        return False

    def mass(self, tok_slice=slice(None)) -> torch.Tensor | None:
        """Masse d'attention MOYENNE par ligne de banque, [B, S].

        Moyennée sur les couches lectrices, les têtes et les positions
        demandées : c'est la lecture la plus neutre possible, aucune couche ni
        aucune tête n'étant privilégiée a priori.
        """
        got = [m.last_bank_attn for m in self.mods
               if m.last_bank_attn is not None]
        if not got:
            return None
        return torch.stack([a[:, :, tok_slice].mean((1, 2))
                            for a in got]).mean(0)


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
        elif ".read." in n or ".bank_heads." in n or \
                any(k in n for k in (".attn.bk.", ".attn.bv.", ".attn.bq.",
                                     ".attn.bo.", ".attn.bank_bias",
                                     ".attn.nb.")):
            # `kvproj` pose ses projections dédiées DANS le module d'attention :
            # sans ça elles seraient comptées comme de l'attention de contexte
            # et le budget « read » du bras paraîtrait nul.
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


# ══ PHASE 11 — LES DEUX ENVIRONNEMENTS NOUVEAUX (S5 et S17) ═════════════════
#
# CE QU'ILS CHANGENT AU LAYOUT, ET POURQUOI C'EST OBLIGATOIRE. Dans les phases
# 6-10, le tour gradé est décodé depuis `A_OPEN` SEUL : la question n'est dans
# AUCUN forward du tour de réponse (le labo forwarde seg par seg). C'était
# tenable tant que le préfixe injecté ne contenait qu'un candidat plausible —
# la ph.8 l'a d'ailleurs mesuré : à préfixe multi-groupes en ordre aléatoire,
# la copie tombe à 0.21 parce que RIEN n'identifie le bon groupe.
#
# S5 et S17 sont précisément des tâches où deux candidats coexistent et où
# c'est la QUESTION qui tranche (« qu'a dit l'utilisateur » vs « qu'ai-je
# dit »). Décoder sans la question rendrait la tâche INDÉCIDABLE et l'A/B
# rotation/additif ne mesurerait plus que du bruit. Les deux envs posent donc
# la question et la réponse dans UN SEUL SEG (question non supervisée, réponse
# supervisée) — au train comme à l'éval, isomorphes par construction (§4.3).
U_OPEN_P11 = "<|im_start|>user\n"

PROV_SELF_TMPL = [
    "Understood, I have logged your {p}reference as {v} on my side.",
    "Noted. On my end I am filing this under {v}.",
    "I will keep {v} as my own working label for it.",
]
PROV_Q = {
    # la question NOMME le canal — c'est tout le test : deux faits du MÊME
    # attribut, même clé oracle, seul le locuteur les sépare.
    "user": ["What did I tell you it was?",
             "Remind me of the value that I gave you."],
    "self": ["What did you file it under on your side?",
             "Remind me of the label that you chose yourself."],
}
PROV_A = {"user": ["You told me it was {v}.", "The value you gave me is {v}."],
          "self": ["I filed it under {v}.", "The label I chose is {v}."]}
PROV_SLOT = "ref"             # slot d'ACCUEIL : pool large (2500), valeurs
                              # arbitraires sans prior LM (strate `code`).
SPAN_SLOT = "ref"


def span_value_pool(tok, lengths=(1, 2, 3, 4, 5, 6), per_bucket: int = 48
                    ) -> dict:
    """{longueur en tokens → valeurs} — les valeurs de l'env `span` (S17).

    La longueur d'une valeur n'est PAS choisie, elle est MESURÉE : on engendre
    un large jeu de candidats (de 1 à ~12 caractères, du sigle nu au code
    segmenté « SV-19 62 1 »), on les tokenise avec LE tokenizer du run, et on
    les range par longueur RÉELLE de `" "+v`. Aucune hypothèse sur le BPE ne
    traîne dans le code — c'est ce qui rend l'env portable du stub hermétique
    (1 char = 1 token) au SmolLM2 du run.

    Les buckets vides sont simplement absents (avec le stub, la longueur 1 est
    inatteignable : « " "+v » fait au moins 2 caractères). Le stream exige
    seulement DEUX buckets non vides — en dessous, il n'y a plus de courbe
    « écart rot/add en fonction de la longueur » à tracer.
    """
    LET = "ABCDEFGHJKLMNPQRSTUVWXYZ"
    cands: list = []
    for a in LET:
        cands.append(a)                                   # 1 caractère
        for b in LET[:8]:
            n1, n2 = ord(b) % 90 + 10, ord(a) % 90 + 10
            d = (ord(a) + ord(b)) % 9 + 1
            cands += [a + b, f"{a}-{n1:02d}", f"{a}{b}-{n1:02d}",
                      f"{a}{b} {n1:02d}", f"{n1:02d}{n2:02d}", f"{n1}{n2}{d}",
                      f"{a}{b}{n1:02d}{n2:02d}",
                      f"{a}{b}-{n1:02d} {n2:02d}",
                      f"{a}{b}-{n1:02d} {n2:02d} {d}"]
    out: dict = {int(n): [] for n in lengths}
    for v in cands:
        n = len(tok(" " + v, add_special_tokens=False)["input_ids"])
        if n in out and len(out[n]) < per_bucket and v not in out[n]:
            out[n].append(v)
    return {n: vs for n, vs in out.items() if vs}


class Persona11Stream(PersonaChatStream):
    """Base des deux envs de la phase 11 : le tour gradé porte SA question.

    `_qa_seg` fabrique un seg unique [question user | ouverture assistant |
    réponse], la question NON supervisée (aucun gradient dessus, elle n'est là
    que pour être lue) et la réponse supervisée avec son `val_mask`. `q_len`
    dit où s'arrête le préfixe : c'est EXACTEMENT le prompt de décodage de
    l'éval, donc le décodage greedy voit ce que l'entraînement a vu.
    """

    def _qa_seg(self, q: str, a: str, v: str) -> dict:
        pieces = [(U_OPEN_P11, False), (q + "\n", False), (CLOSE_P11, False),
                  (A_OPEN, False), (a, True), ("\n", True), (CLOSE_P11, True)]
        seg = self._seg(pieces, "assistant")
        seg["q_len"] = sum(self._ids(p).numel() for p, _ in pieces[:4])
        return self._val_mask(seg, v)

    def _fact_channels(self, seg: dict, slot: str, v: str, p: str = "") -> dict:
        """Pose les canaux d'identité de fait sur un seg DÉJÀ construit.

        `fact_of` les lit pour décider qu'un seg est un WRITE : sans eux, un
        fait énoncé par le MODÈLE (canal `self`) n'entrerait jamais en banque —
        or l'asymétrie user/self est précisément ce que S5 mesure.
        """
        T = seg["input_ids"].size(1)
        seg["fact_slot"] = torch.full((1, T), self.slot_ids.get(slot, 0),
                                      dtype=torch.long)
        seg["fact_val"] = torch.full((1, T), self.val_ids.get(v, 0),
                                     dtype=torch.long)
        seg["fact_attr"] = torch.full((1, T), self.attr_ids.get(p, 0),
                                      dtype=torch.long)
        return seg


class PersonaProvStream(Persona11Stream):
    """S5 — vies à LOCUTEUR : deux faits du MÊME attribut, deux voix.

    Une vie pose DEUX valeurs pour le même slot : l'une énoncée par
    l'UTILISATEUR, l'autre par le MODÈLE lui-même (canal `self` — le
    self-write de la spec §2.3). Les deux writes ont donc :
      * la MÊME clé oracle `pack_key[slot, attr]` (rien à gagner par la clé),
      * un ORDRE TIRÉ AU SORT (rien à gagner par la récence),
      * des contenus également plausibles (rien à gagner par le prior LM).
    Seul le CANAL les sépare — c'est la seule information qui peut faire
    gagner un bras, et elle n'existe que si le read la code.

    La question nomme le canal ; la réponse cite la valeur de CE canal.
    """

    def __init__(self, tok, *, prov_fillers: tuple = (1, 3),
                 prov_vals: str = "ref", **kw) -> None:
        super().__init__(tok, **kw)
        self.prov_fillers = tuple(int(v) for v in prov_fillers)
        # `span` : valeurs MESURÉES en tokens (buckets 2-3, les deux plus
        # courts disponibles au-delà de 1) — la copie reste dans la plage
        # prouvée de l'instrument, et la longueur n'entre pas comme variable
        # confondante de l'examen tag (elle appartient à S17, pas S5).
        self.prov_pool = None
        if prov_vals == "span":
            sp = span_value_pool(tok)
            keep = [n for n in sorted(sp) if n >= 2][:2]
            self.prov_pool = [v for n in keep for v in sp[n]]
            assert len(self.prov_pool) >= 8, (
                f"pool prov `span` trop petit ({len(self.prov_pool)}) avec ce "
                f"tokenizer — buckets {keep} sur {sorted(sp)}")

    def _conv_prov(self) -> dict:
        pool = self.prov_pool or self.slots[PROV_SLOT][4]
        v_user, v_self = self.rng.sample(pool, 2)
        st = self.slots[PROV_SLOT][0]
        u_seg = self._user_valued(self.rng.choice(st).format(v=v_user, p=""),
                                  v_user, slot=PROV_SLOT, p="")
        u_seg["chan"] = 0                                  # CHANNELS[0] = user
        s_txt = self.rng.choice(PROV_SELF_TMPL).format(v=v_self, p="")
        s_seg = self._fact_channels(self._assistant_valued(s_txt, v_self),
                                    PROV_SLOT, v_self)
        s_seg["chan"] = 1                                  # CHANNELS[1] = self
        # ORDRE TIRÉ AU SORT : la récence ne doit RIEN prédire.
        writes = [u_seg, s_seg]
        self.rng.shuffle(writes)
        segs = list(writes)
        for _ in range(self.rng.randint(*self.prov_fillers)):
            segs += self._filler_pair()
        truths, chans, queries = [], [], []
        for _ in range(self.rng.randint(*self.n_queries)):
            c = self.rng.choice(list(CHANNELS))
            v = v_user if c == "user" else v_self
            q = self.rng.choice(PROV_Q[c])
            segs.append(self._qa_seg(q, self.rng.choice(PROV_A[c]).format(v=v),
                                     v))
            truths.append(v)
            chans.append(CHANNELS.index(c))
            queries.append(q)
        return {"kind": "prov", "segs": segs,
                "info": {"truths": truths, "queries": queries, "ages": [],
                         "q_slots": [PROV_SLOT] * len(truths),
                         "p11": {"turns": [i for i, s in enumerate(segs)
                                           if "q_len" in s],
                                 "chan": chans, "strate": [CHANNELS[c]
                                                           for c in chans]}}}

    def next_conv(self) -> dict:
        if self.rng.random() < self.p_smalltalk:
            return self._conv_smalltalk()
        return self._conv_prov()


class PersonaSpanStream(Persona11Stream):
    """S17 — valeurs multi-tokens de LONGUEUR GRADUÉE, citation ORDONNÉE.

    Une vie pose UN fait dont la valeur fait 1, 2, 3 ou 4 tokens (mesurés, cf.
    `span_value_pool`) et `span_decoys` faits leurres d'autres slots. La
    question est dans le seg gradé, la réponse doit rendre la valeur — et
    `grade_recall` exige la chaîne EXACTE, donc l'ORDRE des tokens.

    C'est le test de nécessité de la troisième famille (§2.5) : sans index
    local, un span multi-tokens est un SAC de lignes ; la prédiction inscrite
    est que l'écart rot/add CROÎT avec la longueur, parce que l'additif donne
    des signatures là où la rotation donne l'opérateur successeur R_loc(1).
    """

    def __init__(self, tok, *, span_decoys: int = 1,
                 span_fillers: tuple = (1, 3), **kw) -> None:
        super().__init__(tok, **kw)
        self.span_decoys = int(span_decoys)
        self.span_fillers = tuple(int(v) for v in span_fillers)
        self.span_pool = span_value_pool(tok)
        assert len(self.span_pool) >= 2, (
            f"env `span` : moins de deux longueurs de valeur atteignables avec "
            f"ce tokenizer ({ {k: len(v) for k, v in self.span_pool.items()} }) "
            f"— la courbe « écart par longueur » n'existerait pas")
        self.span_lens = sorted(self.span_pool)

    def _conv_span(self) -> dict:
        L = self.rng.choice(self.span_lens)
        v = self.rng.choice(self.span_pool[L])
        st, qs, ans, _, _ = self.slots[SPAN_SLOT]
        writes = [self._user_valued(self.rng.choice(st).format(v=v, p=""), v,
                                    slot=SPAN_SLOT, p="")]
        used_slots, used_vals = {SPAN_SLOT}, {v}
        for _ in range(self.span_decoys):
            f = self._sample_fact(used_slots, used_vals)
            used_slots.add(f["slot"])
            used_vals.add(f["v"])
            writes.append(self._user_valued(
                self.rng.choice(f["st"]).format(v=f["v"], p=f["p"]), f["v"],
                slot=f["slot"], p=f["p"]))
        self.rng.shuffle(writes)
        segs = list(writes)
        for _ in range(self.rng.randint(*self.span_fillers)):
            segs += self._filler_pair()
        q = self.rng.choice(qs).format(p="")
        segs.append(self._qa_seg(q, self.rng.choice(ans).format(v=v, p=""), v))
        return {"kind": "span", "segs": segs,
                "info": {"truths": [v], "queries": [q], "ages": [],
                         "q_slots": [SPAN_SLOT],
                         "p11": {"turns": [len(segs) - 1], "chan": [0],
                                 "strate": [f"L{L}"]}}}

    def next_conv(self) -> dict:
        if self.rng.random() < self.p_smalltalk:
            return self._conv_smalltalk()
        return self._conv_span()


# ══ PHASE 12 — L'ENV DES VIES LONGUES (S6) ══════════════════════════════════
# Le comportement CIBLE de la spec §2.3, littéralement : « I live in New York »
# au tour ~3 et « I moved to Austin » au tour ~T−10 coexistent, chacun avec son
# âge vrai. Trois questions, trois strates, et elles ne se confondent pas :
#
#   cur_pre    AVANT la supersession, « où j'habite » → la valeur ANCIENNE.
#              C'est le contrôle : tant que rien ne la supplante, elle doit
#              sortir. Une chute ici dit que la banque a simplement OUBLIÉ.
#   cur_post   APRÈS, « où j'habite » → la valeur NOUVELLE. Le temps que met
#              cette strate à basculer EST la métrique (a) « temps
#              d'adaptation » : elle se lit par horizon dw (writes écoulés
#              depuis la supersession).
#   hist       APRÈS, « où j'habitais AVANT Austin » → la valeur ANCIENNE
#              ENCORE. C'est la métrique (b) « survie du fait ancien utile »,
#              et c'est LA sonde que le kill-test n°5 exige : FIFO et récence y
#              scorent 0 PAR CONSTRUCTION (le bon slot est l'ANCIEN), donc
#              toute maintenance qui ne fait que rafraîchir le récent est
#              démasquée ici et nulle part ailleurs.
#
# La pression FIFO est RÉELLE : entre les deux, des faits distracteurs
# consomment des slots (un write = un slot) et du smalltalk du stream fait le
# remplissage. Le layout de read est celui de la ph.11 — la question vit DANS
# le seg gradé (§4.3 : train et éval isomorphes).
LIFE_SLOT = "city"            # le slot NY→Austin de la spec. Ses gabarits de
                              # supersession existent déjà (« Actually we just
                              # moved again, to {v} this time. »).
LIFE_HIST_Q = [
    "Where did I live before {n}?",
    "Which city was I living in before {n}?",
    "Before I moved to {n}, where was I living?",
]
LIFE_HIST_A = [
    "Before {n}, you lived in {v}.",
    "You lived in {v} before moving to {n}.",
]


class PersonaLifeStream(Persona11Stream):
    """S6 — vies LONGUES à supersession tardive et ré-évocation historique.

    Une vie tient en cinq temps : (1) l'ANCRE (`city` = v_old) tôt, (2) des
    blocs de distracteurs — un write de fait d'un autre slot + une paire de
    filler — qui font monter la pression FIFO, avec des questions courantes en
    chemin, (3) la SUPERSESSION (`city` = v_new) tard, (4) d'autres blocs, (5)
    des questions à plusieurs horizons : courante (→ v_new) et HISTORIQUE
    (→ v_old).

    `life_turns` est la longueur CIBLE en segs ; le nombre de blocs en découle
    et le nombre de WRITES (2 + n_blocs) doit rester ≫ max_mem — sinon rien
    n'atteint le bord de la FIFO et l'examen ne mesure rien. Le nombre réel de
    writes est rendu dans `info["life"]["n_writes"]` (mesuré, jamais décrété).
    """

    def __init__(self, tok, *, life_turns: int = 48, life_fillers: tuple = (1, 2),
                 life_vals: str = "city", **kw) -> None:
        super().__init__(tok, **kw)
        self.life_turns = int(life_turns)
        self.life_fillers = tuple(int(v) for v in life_fillers)
        # `span` (fix 08-04) : le pool `city` est un répertoire FERMÉ d'entités
        # à signature unique — le train se minimise par RECONNAISSANCE (villes
        # dans les poids) et la copie ne se forme jamais (16/16 cellules S6 à
        # grade held-out 0,000, valeur pourtant dans les lignes 73 %). Les
        # valeurs span buckets 1-2 sont COMPOSITIONNELLES (sous-tokens
        # partagés : la loss ne se minimise que par copie) et L≤2 tient dans
        # m=4 sans confondant k. Comme prov-sv, le pool n'est PAS splitté
        # train/éval : l'examen S6 mesure la LIAISON à travers la maintenance
        # (quel write survit), pas la généralisation d'émission — le contrôle
        # reste le bras ablaté.
        self.life_pool = None
        if life_vals == "span":
            sp = span_value_pool(tok)
            keep = [n for n in sorted(sp) if n >= 1][:2]
            self.life_pool = [v for n in keep for v in sp[n]]
            assert len(self.life_pool) >= 16, (
                f"pool life `span` trop petit ({len(self.life_pool)}) avec ce "
                f"tokenizer — buckets {keep} sur {sorted(sp)}")
            # les valeurs span doivent exister dans val_ids : fact_val du
            # write ET p11_target de l'env se résolvent par cette table
            # (même extension des deux côtés — cf. extend_val_ids_span).
            for n in sorted(sp):
                for v in sp[n]:
                    if v not in self.val_ids:
                        self.val_ids[v] = len(self.val_ids) + 1
        assert LIFE_SLOT in self.slots, (
            f"l'env `life` a besoin du slot {LIFE_SLOT!r} (supersession "
            f"outillée) — pool_split trop maigre ?")
        # blocs = distracteur + filler ; ~4 segs par bloc, moitié avant la
        # supersession, moitié après.
        self.n_blocks = max(4, (self.life_turns - 8) // 4)

    def _distractor(self, used_slots: set, used_vals: set) -> dict:
        """Un write de fait d'un AUTRE slot : il consomme un slot de la FIFO.

        Les slots se réutilisent quand le stock est épuisé (une vie longue en
        demande plus qu'il n'y en a) : c'est sans conséquence, seul le slot
        `city` est interrogé et c'est le write le plus récent de CE slot qui
        porte la vérité courante.
        """
        free = [s for s in self.slots if s not in used_slots and s != LIFE_SLOT]
        if not free:
            used_slots.clear()
            free = [s for s in self.slots if s != LIFE_SLOT]
        slot = self.rng.choice(free)
        used_slots.add(slot)
        st, _qs, _ans, _upd, pool = self.slots[slot]
        cand = [x for x in pool if x not in used_vals] or list(pool)
        v = self.rng.choice(cand)
        used_vals.add(v)
        p = self.rng.choice(PET_TYPES if slot == "pet"
                            else SIBLINGS if slot == "sibling" else [""])
        return self._user_valued(self.rng.choice(st).format(v=v, p=p), v,
                                 slot=slot, p=p)

    def _conv_life(self) -> dict:
        st, qs, ans, upd, pool = self.slots[LIFE_SLOT]
        pool = self.life_pool or pool
        v_old, v_new = self.rng.sample(pool, 2)
        used_slots, used_vals = {LIFE_SLOT}, {v_old, v_new}
        segs: list = []
        truths, strates, q_dw, turns = [], [], [], []
        nw = 0                       # writes ÉMIS (le compteur de la FIFO)
        w_super = None               # index du write de supersession

        def _fill():
            for _ in range(self.rng.randint(*self.life_fillers)):
                segs.extend(self._filler_pair())

        def _ask(q: str, a: str, v: str, strate: str):
            segs.append(self._qa_seg(q, a, v))
            turns.append(len(segs) - 1)
            truths.append(v)
            strates.append(strate)
            q_dw.append(None if w_super is None else nw - 1 - w_super)

        _fill()
        segs.append(self._user_valued(self.rng.choice(st).format(v=v_old, p=""),
                                      v_old, slot=LIFE_SLOT, p=""))
        nw += 1
        half = self.n_blocks // 2
        # ── AVANT la supersession : l'ancienne valeur est la valeur COURANTE
        for b in range(half):
            segs.append(self._distractor(used_slots, used_vals))
            nw += 1
            _fill()
            if b in (half // 2, half - 1):
                _ask(self.rng.choice(qs).format(p=""),
                     self.rng.choice(ans).format(v=v_old, p=""), v_old,
                     "cur_pre")
        # ── LA SUPERSESSION (gabarit de mise à jour du slot) ────────────────
        segs.append(self._user_valued(self.rng.choice(upd).format(v=v_new,
                                                                  p=""),
                                      v_new, slot=LIFE_SLOT, p=""))
        w_super = nw
        nw += 1
        # ── APRÈS : courante (→ v_new) ET historique (→ v_old) par horizon ─
        post = self.n_blocks - half
        hz = sorted({0, post // 4, post // 2, post - 1})
        for b in range(post):
            if b in hz:
                _ask(self.rng.choice(qs).format(p=""),
                     self.rng.choice(ans).format(v=v_new, p=""), v_new,
                     "cur_post")
                _ask(self.rng.choice(LIFE_HIST_Q).format(n=v_new),
                     self.rng.choice(LIFE_HIST_A).format(v=v_old, n=v_new),
                     v_old, "hist")
            segs.append(self._distractor(used_slots, used_vals))
            nw += 1
            _fill()
        return {"kind": "life", "segs": segs,
                "info": {"truths": truths, "queries": [], "ages": [],
                         "q_slots": [LIFE_SLOT] * len(truths),
                         "life": {"v_old": v_old, "v_new": v_new,
                                  "n_writes": nw, "w_super": w_super,
                                  "dw": q_dw, "n_segs": len(segs)},
                         "p11": {"turns": turns,
                                 "chan": [0] * len(turns),
                                 "strate": strates}}}

    def next_conv(self) -> dict:
        if self.rng.random() < self.p_smalltalk:
            return self._conv_smalltalk()
        return self._conv_life()


class RetentionStore:
    """PHASE 12 — la MAINTENANCE PROCÉDURALE de la spec §2.3, en un objet.

    UNE primitive : append en tête + décalage + chute au bord. PLUS la
    PROPAGATION à budget : les `prop` entrées de la queue au meilleur score de
    rétention sont replacées EN TÊTE, **compteur de naissance PRÉSERVÉ** —
    l'âge reste l'âge VRAI (cohérent avec la rotation d'âge de BankRot, qui
    code un âge, pas une position). La résurrection en découle : une ligne au
    bord peut repartir en tête, ancienneté reprise.

    ZÉRO PARAMÈTRE, ZÉRO GRADIENT (spec §2.9 : « tout le neuf est procédural »).
    Le seul signal qui regarde le modèle est `attn-ema`/`actr`, et il ne le
    regarde qu'à travers une SONDE d'attention qui n'altère pas le forward.

    ── LA CAUSALITÉ, PARCE QU'ELLE A DÉJÀ COÛTÉ UNE MESURE (S2) ──────────────
    Le rejeu d'entraînement est TEACHER-FORCÉ : tout le futur de la vie est
    disponible dans la structure. Un signal qui lirait la masse d'attention du
    tour COURANT au write du tour courant ferait exactement ce que le bank-q a
    fait au read — mesurer une fuite et l'appeler une lecture. Le tampon est
    donc à DEUX temps : `observe()` dépose dans `pend`, chaque `write()`
    COMMET `arm` (la masse du tour d'AVANT) puis arme `pend`. La masse du tour
    t entre donc dans les scores au write t+1, jamais avant, et le self-test le
    prouve.
    """

    def __init__(self, max_mem: int, signal: str = "fifo", prop: int = 0,
                 ema_beta: float = 0.9, actr_decay: float = 0.5) -> None:
        assert signal in RETENTION_SIGNALS, signal
        self.max_mem = int(max_mem)
        self.signal = signal
        # `fifo` NE PROPAGE PAS : c'est sa définition, pas une option.
        self.prop = 0 if signal == "fifo" else int(prop)
        self.ema_beta = float(ema_beta)
        self.actr_decay = float(actr_decay)
        self.rows: list = []      # index 0 = la PROCHAINE à chuter
        self.w = 0                # writes émis (le compteur d'âge)
        self.n_prop = 0           # télémétrie : propagations effectuées
        self.n_drop = 0           # télémétrie : lignes tombées au bord

    # ── le vote du lecteur ─────────────────────────────────────────────────
    def observe(self, mass) -> None:
        """`mass` [G] : la masse d'attention par GROUPE du tour qui vient
        d'être lu. Déposée en ATTENTE — elle n'entre dans aucun score avant le
        write SUIVANT."""
        if mass is None or self.signal not in ("attn-ema", "actr"):
            return
        m = [float(x) for x in mass]
        if len(m) != len(self.rows):
            return                       # vue désynchronisée : on ignore
        top = max(range(len(m)), key=lambda g: m[g])
        for g, e in enumerate(self.rows):
            e["pend"], e["pend_top"] = m[g], (g == top)

    def _commit(self) -> None:
        b = self.ema_beta
        for e in self.rows:
            if e["arm"] is not None:
                e["ema"] = b * e["ema"] + (1.0 - b) * e["arm"]
                if e["arm_top"]:
                    e["uses"].append(self.w)
            e["arm"], e["arm_top"] = e["pend"], e["pend_top"]
            e["pend"], e["pend_top"] = None, False

    # ── le score de rétention (plus haut = mieux gardé) ────────────────────
    def score(self, i: int) -> float:
        e = self.rows[i]
        if self.signal == "age":
            # LE PROXY RÉPUDIÉ : garder les plus VIEILLES. Auto-renforçant par
            # construction — sa présence au bakeoff est la preuve.
            return float(self.w - e["birth"])
        if self.signal == "attn-ema":
            return float(e["ema"])
        if self.signal == "actr":
            ts = [e["birth"]] + e["uses"]
            s = sum((self.w - t + 1.0) ** (-self.actr_decay) for t in ts)
            return math.log(max(s, 1e-12))
        if self.signal == "coverage":
            v = e["vec"]
            best = -1.0
            for j, o in enumerate(self.rows):
                if j != i:
                    best = max(best, float(torch.dot(v, o["vec"])))
            return -best              # le plus REDONDANT a le pire score
        return 0.0

    # ── la primitive ───────────────────────────────────────────────────────
    def write(self, rows, vec, chan: int = 0, slot: int = 0,
              val: int = 0) -> None:
        self._commit()
        if self.prop and len(self.rows) > 1:
            sc = [self.score(i) for i in range(len(self.rows))]
            # tri STABLE : (−score, index) ⇒ déterministe à état donné.
            keep = set(sorted(range(len(self.rows)),
                              key=lambda i: (-sc[i], i))[:self.prop])
            moved = [e for i, e in enumerate(self.rows) if i in keep]
            self.rows = [e for i, e in enumerate(self.rows)
                         if i not in keep] + moved
            self.n_prop += len(moved)
        self.rows.append({"rows": rows, "vec": vec, "birth": self.w,
                          "chan": int(chan), "slot": int(slot),
                          "val": int(val), "ema": 0.0, "uses": [],
                          "pend": None, "pend_top": False,
                          "arm": None, "arm_top": False})
        self.w += 1
        while len(self.rows) > self.max_mem:
            self.rows.pop(0)
            self.n_drop += 1

    # ── la vue lue par le read (même layout que p11_plan) ──────────────────
    def view(self) -> tuple:
        """(lignes [G,…], âges [G], canaux [G], slots [G], valeurs [G]).

        L'ÂGE est l'âge VRAI en writes — `w−1−naissance`, exactement la
        convention de `p11_plan` (0 = le write le plus récent) — et il n'est
        PAS la position : une ligne propagée est en tête avec son âge d'origine.
        C'est tout le point du compteur de naissance préservé.
        """
        rows = torch.stack([e["rows"] for e in self.rows])
        def _t(key):
            return torch.tensor([int(e[key]) for e in self.rows],
                                dtype=torch.long)
        ages = torch.tensor([self.w - 1 - e["birth"] for e in self.rows],
                            dtype=torch.long)
        return rows, ages, _t("chan"), _t("slot"), _t("val")


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


def extend_val_ids_span(val_ids: dict, tok) -> None:
    """Fix life-sv (08-05) : enregistre les valeurs de `span_value_pool` dans
    `val_ids`, ids DÉTERMINISTES appendés après la table de base (ordre =
    buckets croissants puis ordre du pool). Sans ça, `fact_val` vaut 0 côté
    stream et `p11_target` ne résout jamais la vérité côté env : le smoke a
    rendu resident_rate 0,0 et r@1 n_sel=0 alors que les writes se faisaient.
    À appeler des DEUX côtés (stream ET env) — chacun a sa copie de la table,
    et l'appariement des ids exige la même extension dans le même ordre."""
    sp = span_value_pool(tok)
    for n in sorted(sp):
        for v in sp[n]:
            if v not in val_ids:
                val_ids[v] = len(val_ids) + 1


class OracleEnv:
    """Rejoue une conv seg par seg et pose la banque à la place du modèle."""

    def __init__(self, tok, max_mem: int, write_mode: str = "fact",
                 span_vals: bool = False):
        self.tok = tok
        self.max_mem = max_mem
        assert write_mode in WRITE_MODES, write_mode
        self.write_mode = write_mode
        # nombre de lignes appendées par le DERNIER appel à write() (télémétrie
        # d'âge : « combien de writes séparent le fait de sa query »).
        self.last_added = 0
        slot_ids, val_ids, attr_ids = fact_id_maps()
        if span_vals:
            extend_val_ids_span(val_ids, tok)
        self.slot_ids = slot_ids
        self.val_ids = val_ids
        self.id2val = {i: v for v, i in val_ids.items()}
        # ── PHASE 12 (S8) : le RÉSERVOIR de groupes ÉTRANGERS ───────────────
        # Des segments porteurs d'AUTRES vies du même stream, gardés sous forme
        # de TOKENS (jamais de vecteurs : des lignes stockées vieilliraient
        # pendant que les embeddings bougent, et la courbe de dilution
        # mesurerait cette dérive). Les lignes sont recalculées par lots avec
        # le modèle COURANT, tous les `fill_refresh` appels.
        self.foreign_pool: list = []
        self._fill_rows = None
        self._fill_calls = 0
        self.last_life: dict = {}
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
                self._pool_add(model, st)
        # S8 : le remplissage travaille sur des tuples (lignes, âges, …) — le
        # plan de conditionnement en est un à deux champs, il passe tel quel.
        return self.fill_plan(model, out)

    def p11_plan(self, model: ToyReadLM, conv: dict) -> dict:
        """PHASE 11 — {index du seg gradé → (lignes, âges, canaux)}.

        Même mécanique que `cond_plan` (rejeu du FIFO, TOUS les résidents
        injectés, aucun privilège de retrieval) avec UN canal de plus : la
        PROVENANCE du write, lue sur le seg qui l'a produit (`seg["chan"]`,
        posé par l'env `prov` ; 0 partout ailleurs). C'est le seul endroit où
        le locuteur entre dans le système — il ne survit pas à la sélection
        top-k, exactement comme les balises de rôle du chat template au 350M
        (§2.5 : la rotation de provenance RÉIMPLÉMENTE le template dans
        l'espace où il n'existe plus).

        Rend {} hors des envs de la phase 11 : les chemins ph.6-10 ne voient
        rien.
        """
        p11 = (conv.get("info") or {}).get("p11")
        if not p11:
            return {}
        if conv.get("kind") == "life":
            # PHASE 12 : la FIFO nue est remplacée par le STORE (append +
            # décalage + chute + propagation). À `fifo`/prop 0 le store est
            # bit-à-bit le `fifo[-max_mem:]` de ci-dessous — self-testé.
            return self.life_plan(model, conv)
        turns = set(p11["turns"])
        hid = model.cfg.code in HID_CODES
        out, fifo, w = {}, [], 0
        for i, seg in enumerate(conv["segs"]):
            if i in turns and fifo:
                out[i] = (torch.stack([r for r, _, _, _ in fifo]),
                          torch.tensor([w - 1 - t for _, t, _, _ in fifo],
                                       dtype=torch.long),
                          torch.tensor([c for _, _, c, _ in fifo],
                                       dtype=torch.long),
                          torch.tensor([sl for _, _, _, sl in fifo],
                                       dtype=torch.long))
            f = self.fact_of(seg)
            if f is not None:
                st = self.seg_tokens(seg)
                fifo.append(((model.tophid_rows_fixed(st) if hid
                              else model.toprows_sel_fixed(st)), w,
                             int(seg.get("chan", 0)), int(f[0])))
                w += 1
                fifo = fifo[-self.max_mem:]
                self._pool_add(model, st)
        return self.fill_plan(model, out)

    # ══ PHASE 12 — LE STORE, LE VOTE DU LECTEUR ET LE REMPLISSAGE ══════════

    @staticmethod
    def _line_vec(model, rows) -> torch.Tensor:
        """Le vecteur qui REPRÉSENTE un write dans l'espace des lignes.

        Moyenne des lignes du groupe, normalisée : le produit scalaire entre
        deux `vec` EST alors le cosinus, ce dont `coverage` a besoin. Pour les
        codes à ID (`toprows`) les lignes sont des tokens : on passe par la
        table d'embeddings, le même espace que celui où elles seront injectées.
        """
        with torch.no_grad():
            v = (rows.float() if rows.dim() > 1
                 else model.embed.weight[rows.to(model.embed.weight.device)]
                 .float())
            v = v.reshape(-1, v.shape[-1]).mean(0)
            return (v / v.norm().clamp_min(1e-6)).cpu()

    @torch.no_grad()
    def _bank_vote(self, model, seg, store):
        """La MASSE D'ATTENTION par groupe, lue GRATUITEMENT (spec §2.4 : « la
        masse d'attention par ligne = un signal de rétention gratuit »).

        Un forward NO-GRAD du tour courant avec le store injecté, sous
        `bank_attn_probe` — la sonde recalcule le softmax en clair et
        n'altère RIEN. Le résultat part dans `observe()`, donc en ATTENTE : il
        n'entrera dans un score qu'au write SUIVANT.
        """
        if not store.rows:
            return None
        dev = model.embed.weight.device
        rows, ages, chans, _sl, _vl = store.view()
        r = rows[None].to(dev)
        X = seg["input_ids"][:, :model.cfg.max_seq_len].to(dev)
        with bank_attn_probe(model) as probe:
            model(X, None, None, inject=r, inject_age=ages[None].to(dev),
                  inject_chan=chans[None].to(dev))
            mass = probe.mass()
        if mass is None:
            return None
        G, k = r.shape[1], r.shape[2]
        return mass[0, :G * k].reshape(G, k).sum(-1)

    def life_plan(self, model: ToyReadLM, conv: dict) -> dict:
        """PHASE 12 (S6) — {index du seg gradé → (lignes, âges, canaux, slots,
        VALEURS)}, la banque telle que la MAINTENANCE la laisse.

        Cinquième champ par rapport à `p11_plan` : l'ID de VALEUR du write.
        Sans lui la cible du r@1 serait indécidable ici — les deux writes du
        slot interrogé (New York et Austin) ont le MÊME slot, et la question
        historique désigne l'ANCIEN. « Le plus récent du bon slot » (la
        convention ph.11) donnerait systématiquement la mauvaise réponse sur
        la strate qui compte le plus.

        Le rejeu est SÉQUENTIEL et le store est un objet d'ÉTAT : à chaque seg,
        (1) on photographie la banque si le tour est gradé, (2) on relève le
        vote du lecteur si le signal en demande un, (3) on écrit si le seg
        porte un fait. L'ordre 2-3 est ce qui rend le signal causal (cf.
        RetentionStore).
        """
        p11 = (conv.get("info") or {}).get("p11")
        if not p11:
            return {}
        cfg = model.cfg
        store = RetentionStore(self.max_mem, cfg.retention, cfg.prop_budget,
                               cfg.ema_beta, cfg.actr_decay)
        vote = cfg.retention in ("attn-ema", "actr")
        hid = cfg.code in HID_CODES
        turns = set(p11["turns"])
        out: dict = {}
        for i, seg in enumerate(conv["segs"]):
            if i in turns and store.rows:
                out[i] = store.view()
            f = self.fact_of(seg)
            if f is None:
                continue
            st = self.seg_tokens(seg)
            if vote:
                # LE VOTE PORTE SUR LE TOUR COURANT, LU CONTRE LA BANQUE
                # D'AVANT : la ligne qu'on s'apprête à écrire n'est pas encore
                # là, donc `attn-ema` ne peut pas dégénérer en récence.
                store.observe(self._bank_vote(model, seg, store))
            rws = (model.tophid_rows_fixed(st) if hid
                   else model.toprows_sel_fixed(st))
            store.write(rws, self._line_vec(model, rws),
                        chan=int(seg.get("chan", 0)), slot=int(f[0]),
                        val=int(f[2]))
            self._pool_add(model, st)
        self.last_life = {"n_writes": store.w, "n_prop": store.n_prop,
                          "n_drop": store.n_drop, "resident": len(store.rows)}
        return self.fill_plan(model, out)

    # ── S8 : le remplissage par des distracteurs RÉELS ─────────────────────

    def _pool_add(self, model, seg_tok) -> None:
        if model.cfg.bank_fill == "none":
            return
        self.foreign_pool.append(seg_tok.detach().cpu())
        if len(self.foreign_pool) > int(model.cfg.fill_pool):
            del self.foreign_pool[0]

    def _foreign_rows(self, model) -> torch.Tensor | None:
        """Les lignes du réservoir, RECALCULÉES avec le modèle courant.

        Un lot, pas un segment à la fois : le coût est celui d'UN forward de
        `fill_pool` segments courts tous les `fill_refresh` appels, contre
        `fill_pool` forwards à chaque plan sinon.
        """
        if not self.foreign_pool:
            return None
        stale = (self._fill_rows is None
                 or self._fill_rows.shape[0] != len(self.foreign_pool)
                 or self._fill_calls % int(model.cfg.fill_refresh) == 0)
        self._fill_calls += 1
        if stale:
            self._fill_rows = model.group_rows_batch(list(self.foreign_pool))
        return self._fill_rows

    def fill_plan(self, model: ToyReadLM, out: dict) -> dict:
        """S8 — complète CHAQUE entrée de plan jusqu'à `max_mem` GROUPES.

        Les VRAIS résidents restent EN DERNIER (les plus frais, l'ordre que le
        reste du labo suppose) et les remplisseurs portent des métadonnées
        NULLES (slot 0, valeur 0) : ils ne peuvent jamais devenir la cible du
        r@1, ils ne font que DILUER. C'est la définition opératoire de la
        courbe : même signal, plus de bruit, un seul cadran (S).

        No-op EXACT quand `bank_fill` vaut `none` (le dict est rendu tel quel).
        """
        if model.cfg.bank_fill == "none" or not out:
            return out
        pool = self._foreign_rows(model)
        if pool is None:
            return out
        S = self.max_mem
        res: dict = {}
        for i, ent in out.items():
            rows = ent[0]
            need = S - int(rows.shape[0])
            if need <= 0:
                res[i] = ent
                continue
            idx = torch.arange(need, device=pool.device) % pool.shape[0]
            fill = pool[idx].to(dtype=rows.dtype)
            rows = torch.cat([fill, rows], 0)
            # les remplisseurs sont PLUS VIEUX que tous les résidents : leur
            # âge continue la numérotation, l'ordre est donc cohérent.
            base = int(ent[1].max()) + 1 if ent[1].numel() else 0
            ages = torch.cat([torch.arange(need - 1, -1, -1) + base, ent[1]])
            zeros = torch.zeros(need, dtype=torch.long)
            res[i] = (rows, ages) + tuple(torch.cat([zeros, t])
                                          for t in ent[2:])
        return res

    def p11_target(self, conv: dict, seg_idx: int, qidx: int, ent) -> int | None:
        """Index, DANS LE PRÉFIXE INJECTÉ, du groupe qui porte la réponse.

        C'est la cible du r@1 de `evaluate_p11`, et elle se lit dans les
        métadonnées elles-mêmes :
          env `prov` — le groupe dont le CANAL est celui que la question nomme
            (les deux groupes partagent slot, attribut et donc clé oracle : le
            canal est LE seul discriminant, ce qui est tout le test) ;
          env `span` — le groupe du SLOT interrogé (les leurres sont d'autres
            slots).
        Le plus RÉCENT gagne en cas d'égalité — la même convention qu'ailleurs
        dans le labo. None = aucun groupe résident ne porte la réponse.
        """
        p11 = (conv.get("info") or {}).get("p11") or {}
        chans, slots = ent[2], ent[3]
        if conv.get("kind") == "life":
            # PHASE 12 : le groupe porteur est celui dont la VALEUR est la
            # vérité — pas « le plus récent du bon slot ». Les deux writes de
            # `city` partagent le slot ; c'est la question historique qui
            # désigne l'ANCIEN, et la convention ph.11 la raterait TOUJOURS.
            tr = (conv.get("info") or {}).get("truths") or []
            want = self.val_ids.get(tr[qidx]) if qidx < len(tr) else None
            vals = ent[4] if len(ent) > 4 else slots
            hit = [g for g in range(len(vals)) if int(vals[g]) == want]
            return hit[-1] if hit and want else None
        if conv.get("kind") == "prov":
            want = int(p11["chan"][qidx]) if qidx < len(p11["chan"]) else 0
            hit = [g for g in range(len(chans)) if int(chans[g]) == want]
        else:
            qs = (conv.get("info") or {}).get("q_slots") or []
            sid = self.slot_ids.get(qs[qidx]) if qidx < len(qs) else None
            hit = [g for g in range(len(slots)) if int(slots[g]) == sid]
        return hit[-1] if hit else None

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


def age_augment(cfg: ToyCfg, ages: torch.Tensor) -> torch.Tensor:
    """S4 — AUGMENTATION D'ÉCHELLE des âges, à l'ENTRAÎNEMENT seulement.

    Chaque lane du sous-lot tire un facteur LOG-UNIFORME dans [1, age_aug_max]
    et ses âges sont multipliés par lui. L'ORDRE est intact (c'est lui qui
    porte l'information « qui est plus vieux que qui ») ; seule l'ÉCHELLE du
    compteur varie. C'est le steelman du bras BRUT : sans compression, il ne
    peut tenir l'OOD qu'en apprenant cette invariance, et on lui donne les
    moyens de l'apprendre.

    Borne < échelles d'éval (10, 100) par construction (cf. le garde-fou de
    ToyCfg) : une augmentation qui couvrirait l'éval ne mesurerait plus rien.
    Tirage par le RNG GLOBAL de torch ⇒ reproductible à graine fixée.
    """
    if not cfg.age_aug or ages is None:
        return ages
    s = torch.exp(torch.rand(ages.shape[0], 1, device=ages.device)
                  * math.log(cfg.age_aug_max))
    return (ages.float() * s).round().long()


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
    chans: list = [{} for _ in convs]
    if r4 and cfg.p11_env != "rule":
        # ── PHASE 11 : les tours gradés reçoivent TOUS LES RÉSIDENTS ───────
        # Aucun privilège de sélection (le préfixe contient le bon groupe ET
        # ses concurrents), et le seg porte sa propre question : train et éval
        # voient EXACTEMENT le même layout (§4.3).
        # INDEXATION, pas dépaquetage : l'env `life` (ph.12) porte un cinquième
        # champ (les ID de valeur, cf. life_plan) et le dépaquetage strict
        # planterait sur lui.
        for i, c in enumerate(convs):
            for jj, ent in env.p11_plan(model, c).items():
                plans[i][jj] = ent[0]
                ages[i][jj] = ent[1]
                chans[i][jj] = ent[2]
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
                iage = age_augment(cfg, iage)
            ichan = None
            if has_inj and all(j in chans[i] for i in sub):
                ichan = torch.stack([chans[i][j] for i in sub]).to(device)
            sub_banks = [banks[i] for i in sub]
            bank, bmask = pad_bank(sub_banks, device)
            bage = bank_ages_for(model, sub_banks, device)
            with torch.autocast(device.split(":")[0], dtype=torch.bfloat16,
                                enabled=amp):
                logits = model(X, bank, bmask, inject=inj, inject_age=iage,
                               bank_age=bage, inject_chan=ichan)
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
def evaluate_cond(model, env, stream, seed, n_convs, device, max_len, amp,
                  age_scale: float = 1.0):
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
        if age_scale != 1.0:
            # ── S4, LA VIE LONGUE SYNTHÉTIQUE ──────────────────────────────
            # Le fait est TOUJOURS en banque (rien n'est évincé), mais le
            # COMPTEUR DE WRITES a grandi : c'est exactement l'OOD que le
            # design doit encaisser. L'ordre des âges est préservé, seule leur
            # échelle change — comme au train sous `age_aug`, mais 10× ou 100×
            # au-delà de ce que l'augmentation couvre.
            ages = (ages.float() * float(age_scale)).round().long()
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
        num = den = 0.0
        marges: list = []          # la MARGE aux marqueurs, tour par tour
        hit = tot = 0
        # ── LES ABSOLUS (règle durcie, audit 08-03) : la nll du rendu
        # COHÉRENT sous chaque condition de lecture. `nll_live` apparié entre
        # cellules est le juge ; le contraste et le delta ne disent rien du
        # PLANCHER, et c'est le plancher qui a masqué dual_heads au carré.
        abs_n = abs_w = abs_m = abs_mw = 0.0
        for i, it in enumerate(items):
            st = (it["state"] if cond_name == "live"
                  else shuf[i] if cond_name == "shuf" else None)
            if cond_name == "shuf" and st is None:
                continue
            ca, cw, cm, cmw = _nll(it["coh"], st)
            ia, iw, im, imw = _nll(it["inc"], st)
            abs_n += ca
            abs_w += cw
            abs_m += cm
            abs_mw += cmw
            if cw > 0 and iw > 0:
                num += ia / iw - ca / cw
                den += 1.0
            if cmw > 0 and imw > 0:
                marges.append(im / imw - cm / cmw)
                hit += int(cm / cmw < im / imw)
                tot += 1
        acc[f"dnll_{cond_name}"] = num / den if den else float("nan")
        acc[f"acc_{cond_name}"] = hit / tot if tot else float("nan")
        # ── LA MARGE, pas seulement l'ACCURACY ─────────────────────────────
        # Le 2AFC sature (1.000 dès que le canal est ouvert) et cesse alors de
        # départager les cellules. La MARGE, elle, garde de la dynamique : elle
        # dit de COMBIEN le bon rendu est préféré, pas seulement s'il l'est.
        # `mark_*` EST la marge moyenne (nom conservé — le format ne bouge
        # pas) ; l'erreur-type et la médiane s'y ajoutent pour la rendre
        # adjudicable (une moyenne sans dispersion n'arbitre rien à n~45).
        n_m = len(marges)
        acc[f"mark_{cond_name}"] = (sum(marges) / n_m) if n_m else float("nan")
        if n_m > 1:
            mu = acc[f"mark_{cond_name}"]
            var = sum((x - mu) ** 2 for x in marges) / (n_m - 1)
            acc[f"mark_se_{cond_name}"] = math.sqrt(var / n_m)
        else:
            acc[f"mark_se_{cond_name}"] = float("nan")
        acc[f"mark_med_{cond_name}"] = (sorted(marges)[n_m // 2] if n_m
                                        else float("nan"))
        acc[f"n_mark_{cond_name}"] = n_m
        acc[f"nll_{cond_name}"] = (abs_n / abs_w) if abs_w > 0 else float("nan")
        acc[f"nllmark_{cond_name}"] = ((abs_m / abs_mw) if abs_mw > 0
                                       else float("nan"))
    # ── PHASE 12 (S8) : le r@1 SOUS DILUTION ───────────────────────────────
    # Dans `kvproj` la sélection EST l'attention (cf. evaluate_p11) : on lit la
    # masse par groupe pendant la lecture `live`. La cible est le DERNIER
    # groupe — `fill_plan` place les vrais résidents en dernier et les
    # cellules de dilution tournent à cond_decoys 0, donc il y en a
    # exactement un. Hors remplissage la mesure n'aurait pas de cible définie :
    # elle reste NaN, et le CSV/JSON garde la même forme.
    acc["r_at1"], acc["n_sel"] = float("nan"), 0
    if inj_arm and model.cfg.bank_fill != "none":
        hit = tot = 0
        for it in items:
            st = it["state"]
            if st is None or st[0] != "inj" or st[1] is None:
                continue
            rows, ages = st[1], st[2]
            X = it["coh"]["input_ids"][:, :max_len].to(device)
            with bank_attn_probe(model) as probe, \
                    torch.autocast(device.split(":")[0], dtype=torch.bfloat16,
                                   enabled=amp):
                model(X, None, None, inject=rows[None].to(device),
                      inject_age=ages[None].to(device))
                mass = probe.mass()
            if mass is None:
                continue
            G, k = int(rows.shape[0]), int(rows.shape[1])
            per_g = mass[0, :G * k].reshape(G, k).sum(-1)
            hit += int(int(per_g.argmax()) == G - 1)
            tot += 1
        acc["r_at1"] = (hit / tot) if tot else float("nan")
        acc["n_sel"] = tot
    acc["n"] = len(items)
    acc["n_convs"] = len({id(x["conv"]) for x in items})
    acc["age_scale"] = float(age_scale)
    model.train()
    return acc


@torch.no_grad()
def evaluate_p11(model, env, stream, seed, n_convs, device, tok, stop_id,
                 max_new, max_len, amp):
    """PHASE 11 — l'éval des envs `prov` (S5) et `span` (S17).

    DEUX MESURES SÉPARÉES, et c'est le point (§3-S5) :

      `r_at1`  la SÉLECTION. Dans `kvproj` il n'y a pas de retriever : la
               sélection EST l'attention. On lit donc la MASSE D'ATTENTION par
               ligne de banque (sonde `bank_attn_probe`, qui n'altère pas le
               forward), on la somme PAR GROUPE, et on compte les tours où le
               groupe porteur sort en tête. C'est un r@1 sur le mécanisme
               réel, pas sur un module de substitution.
      `grade`  la COPIE. Décodage greedy depuis le préfixe [question | A_OPEN]
               — le MÊME que celui vu au train — et `grade_recall` exige la
               chaîne exacte, donc l'ordre des tokens.

    Les deux peuvent diverger, et c'est l'hypothèse à tester : une
    perturbation rotative des clés peut aider la sélection ET casser le circuit
    de copie, câblé sur le layout (ph.8). Un chiffre agrégé les confondrait.

    Bras ABLATÉ : le même tour SANS injection (backbone nu). Il borne ce que le
    prior LM et la question seule donnent.
    """
    model.eval()
    stream.rng = random.Random(seed)
    rows: list = []
    telem: list = []
    # [somme CE, poids, somme CE VALEUR, poids valeur] par bras de lecture
    nll = {"live": [0.0, 0.0, 0.0, 0.0], "abl": [0.0, 0.0, 0.0, 0.0]}
    done = guard = 0
    while done < n_convs and guard < n_convs * 20:
        guard += 1
        conv = stream.next_conv()
        p11 = (conv.get("info") or {}).get("p11")
        if not p11:
            continue
        done += 1
        truths = conv["info"]["truths"]
        strates = p11["strate"]
        plan = env.p11_plan(model, conv)
        # PHASE 12 : les horizons de la vie (writes écoulés depuis la
        # supersession) et la télémétrie de maintenance du rejeu.
        life = (conv.get("info") or {}).get("life") or {}
        dws = life.get("dw") or []
        if life:
            tel = dict(env.last_life)
            tel["n_graded"] = len(p11["turns"])
            telem.append(tel)
        qi = 0
        for i, seg in enumerate(conv["segs"]):
            if i not in set(p11["turns"]):
                continue
            ent = plan.get(i)
            tr = truths[qi] if qi < len(truths) else "?"
            st = strates[qi] if qi < len(strates) else "?"
            ch = p11["chan"][qi] if qi < len(p11["chan"]) else 0
            qi += 1
            dw = dws[qi - 1] if qi - 1 < len(dws) else None
            if ent is None:                 # rien de résident : rien à lire
                rows.append({"strate": st, "chan": ch, "hit": None,
                             "live": "", "abl": "", "truth": tr, "dw": dw,
                             "age": None, "res": 0})
                continue
            r, ag, cc = (t[None].to(device) for t in ent[:3])
            X = seg["input_ids"][:, :max_len].to(device)
            # ── LES DEUX TERMES ABSOLUS, APPARIÉS ──────────────────────────
            # Règle de protocole durcie (audit des absolus, 08-03 nuit) : le
            # Δnll INTRA-MODÈLE est déprécié comme juge — sur le carré il a
            # gonflé un bras (part bait) ET masqué l'autre (meilleur nll_live
            # puni pour son meilleur plancher ablaté). On émet donc nll_live et
            # nll_abl SÉPARÉMENT, sur le même segment et le même masque, et
            # c'est `nll_live` apparié qui adjuge (avec le grade et le r@1).
            W = seg["loss_mask"][:, :max_len].to(device)
            vm = seg.get("val_mask")
            Vm = (vm[:, :max_len].to(device) * (W > 0).float()
                  if vm is not None else None)
            with bank_attn_probe(model) as probe, \
                    torch.autocast(device.split(":")[0], dtype=torch.bfloat16,
                                   enabled=amp):
                lg_live = model(X, None, None, inject=r, inject_age=ag,
                                inject_chan=cc)
                mass = probe.mass()
                lg_abl = model(X, None, None)
            for tag, lg in (("live", lg_live), ("abl", lg_abl)):
                s, n = seg_ce(lg, X, W)
                nll[tag][0] += float(s)
                nll[tag][1] += float(n)
                if Vm is not None:
                    sv, nv = seg_ce(lg, X, Vm)
                    nll[tag][2] += float(sv)
                    nll[tag][3] += float(nv)
            hit = None
            # ── PHASE 12 : l'ÂGE VRAI (en writes) du groupe porteur, et sa
            # RÉSIDENCE. C'est la matière de la métrique (d) « horizon effectif
            # mesuré » : le plus grand âge encore CITABLE, pas le plus grand
            # âge présent.
            tgt0 = env.p11_target(conv, i, qi - 1, ent)
            age_t = None if tgt0 is None else int(ent[1][tgt0])
            if mass is not None:
                # somme PAR GROUPE (les lignes d'un groupe sont contiguës, cf.
                # l'aplatissement [G,k] → [G·k])
                G, k = r.shape[1], r.shape[2]
                per_g = mass[0, :G * k].reshape(G, k).sum(-1)
                # le groupe PORTEUR : celui dont le canal ET la position
                # correspondent à la question. En `prov` deux groupes existent,
                # un par canal ; en `span` c'est le groupe du slot interrogé,
                # qui est le seul dont le canal vaut 0 et la valeur la vérité —
                # on le repère par sa position dans le plan.
                tgt = tgt0
                if tgt is not None:
                    hit = int(int(per_g.argmax()) == tgt)
            pre = X[:, :int(seg.get("q_len", 0))]
            live = tok.decode(model.greedy(pre, None, None, max_new, stop_id,
                                           inject=r, inject_age=ag,
                                           inject_chan=cc))
            abl = tok.decode(model.greedy(pre, None, None, max_new, stop_id))
            rows.append({"strate": st, "chan": ch, "hit": hit,
                         "live": live, "abl": abl, "truth": tr, "dw": dw,
                         "age": age_t, "res": int(tgt0 is not None)})
    out = {"n": len(rows)}
    live = [r["live"] for r in rows]
    abl = [r["abl"] for r in rows]
    tru = [r["truth"] for r in rows]
    out["grade_live"] = grade_recall(live, tru) if rows else float("nan")
    out["grade_abl"] = grade_recall(abl, tru) if rows else float("nan")
    hits = [r["hit"] for r in rows if r["hit"] is not None]
    out["r_at1"] = (sum(hits) / len(hits)) if hits else float("nan")
    out["n_sel"] = len(hits)
    strat: dict = {}
    for r in rows:
        strat.setdefault(r["strate"], []).append(r)
    out["strates"] = {
        s: {"n": len(v), "grade": grade_recall([x["live"] for x in v],
                                               [x["truth"] for x in v]),
            "grade_abl": grade_recall([x["abl"] for x in v],
                                      [x["truth"] for x in v]),
            "r_at1": (sum(x["hit"] for x in v if x["hit"] is not None)
                      / max(sum(1 for x in v if x["hit"] is not None), 1))
            if any(x["hit"] is not None for x in v) else float("nan")}
        for s, v in sorted(strat.items())}
    # ── LES ABSOLUS (le juge) ET LE DELTA (déprécié, gardé pour continuité) ─
    for tag in ("live", "abl"):
        s, n, sv, nv = nll[tag]
        out[f"nll_{tag}"] = (s / n) if n > 0 else float("nan")
        out[f"nllval_{tag}"] = (sv / nv) if nv > 0 else float("nan")
    out["n_nll"] = nll["live"][1]
    out["n_nllval"] = nll["live"][3]
    # DÉPRÉCIÉ comme juge (audit 08-03) : il ne dit rien du PLANCHER. Rendu
    # parce que les campagnes antérieures le portent, jamais pour arbitrer.
    out["dnll_deprecated"] = out["nll_abl"] - out["nll_live"]
    if telem:
        out["life"] = life_metrics(rows, telem, env.max_mem)
    model.train()
    return out


def life_metrics(rows: list, telem: list, max_mem: int) -> dict:
    """PHASE 12 (S6) — LES QUATRE MÉTRIQUES DE PREMIÈRE CLASSE.

      (a) `t_adapt`     TEMPS D'ADAPTATION au fait NOUVEAU : la courbe
                        grade(cur_post) PAR HORIZON dw (writes écoulés depuis
                        la supersession), plus le premier horizon où elle
                        franchit 0.5. `None` = jamais franchi sur la plage
                        mesurée — un bras qui n'adopte jamais Austin.
      (b) `surv_hist`   SURVIE DU FAIT ANCIEN UTILE : le grade de la strate
                        `hist` (la ré-évocation historique APRÈS supersession).
                        C'est la sonde que FIFO et récence ratent PAR
                        CONSTRUCTION.
      (c) grade global et r@1 : déjà rendus par l'appelant.
      (d) `horizon_p90` HORIZON EFFECTIF MESURÉ : le 90ᵉ centile de l'âge (en
                        writes) des faits encore CITÉS CORRECTEMENT. Pas l'âge
                        des lignes présentes — l'âge de celles qui SERVENT
                        encore. `horizon_max` en est la borne observée.

    Plus la télémétrie de la maintenance elle-même (propagations et chutes par
    vie, résidence de la cible) : sans elle, un bras qui ne propage jamais et
    un bras qui propage la même ligne en boucle rendraient le même chiffre.
    """
    def _g(v):
        return (grade_recall([x["live"] for x in v], [x["truth"] for x in v])
                if v else float("nan"))

    def _ok(x):
        return grade_recall([x["live"]], [x["truth"]]) > 0.5

    out: dict = {}
    post = [x for x in rows if x["strate"] == "cur_post"
            and x["dw"] is not None]
    curve: dict = {}
    for x in post:
        curve.setdefault(int(x["dw"]), []).append(x)
    out["adapt_curve"] = {str(d): {"n": len(v), "grade": _g(v)}
                          for d, v in sorted(curve.items())}
    out["t_adapt"] = next((d for d in sorted(curve)
                           if _g(curve[d]) >= 0.5), None)
    out["surv_hist"] = _g([x for x in rows if x["strate"] == "hist"])
    out["surv_pre"] = _g([x for x in rows if x["strate"] == "cur_pre"])
    ages = sorted(int(x["age"]) for x in rows
                  if x["age"] is not None and _ok(x))
    out["horizon_p90"] = (ages[min(len(ages) - 1, int(0.9 * len(ages)))]
                          if ages else None)
    out["horizon_max"] = ages[-1] if ages else None
    out["n_cited"] = len(ages)
    res = [x["res"] for x in rows]
    out["resident_rate"] = (sum(res) / len(res)) if res else float("nan")
    n = max(len(telem), 1)
    for k in ("n_writes", "n_prop", "n_drop"):
        out[k + "_per_life"] = sum(t.get(k, 0) for t in telem) / n
    out["max_mem"] = int(max_mem)
    return out


def parse_scales(s) -> tuple:
    """« 1,10,100 » → (1.0, 10.0, 100.0). La première EST le régime du train."""
    if isinstance(s, (list, tuple)):
        vals = [float(v) for v in s]
    else:
        vals = [float(v) for v in str(s).replace(" ", "").split(",") if v]
    assert vals and vals[0] == 1.0, (
        f"age_eval_scales doit commencer par 1 (le régime vu au train) : {s!r}")
    return tuple(vals)


# ── nom de run (⇒ save_dir) ──────────────────────────────────────────────────

# ── PHASE 10 : nommage DÉTERMINISTE de la grille §2.4 ────────────────────────
# Un combo = un dossier, et le NOM du dossier se relit comme le combo. C'est ce
# qui rend les 36 runs agrégeables sans grepper un seul log.
GRID_READ = {("r0", "entry"): "seqfw",     ("r3", "entry"): "bankxattn",
             ("r4", "entry"): "injentry",  ("r4", "kv"): "kvappend",
             ("r4", "dual"): "dualheads", ("r4", "kvproj"): "kvproj",
             ("r5", "kvproj"): "r5kvproj",
             ("r5", "entry"): "r5entry",   ("r5", "kv"): "r5kv",
             ("r5", "dual"): "r5dual"}
GRID_TAP = {"toprows": "native", "tophid": "postnorm", "midhid": "mid",
            "mean": "pooled"}


def grid_name(cfg: ToyCfg) -> str:
    """`read-<mode>_rot-<on|off>_tap-<prov>_m<k>` (+ suffixe de bras).

    Le seed n'entre PAS dans le nom : la grille tourne à seed FIXE, un run par
    combo. Un balayage de graine, s'il vient, ajoutera son propre suffixe.
    """
    if cfg.p12_exam:
        return p12_name(cfg)
    if cfg.p11_exam:
        # PHASE 11 : ses cellules ont leur propre espace de noms — aucune ne
        # peut retomber sur un dossier de la grille §2.4 déjà lancée.
        return p11_name(cfg)
    name = (f"read-{GRID_READ.get((cfg.variant, cfg.read_path), cfg.variant)}"
            f"_rot-{'on' if cfg.age_rope else 'off'}"
            f"_tap-{GRID_TAP.get(cfg.code, cfg.code)}"
            f"_m{cfg.top_k}")
    if cfg.fw_additive:
        # Le bras multiplicatif divergent est une DONNÉE de la grille : les
        # cellules additives ne doivent jamais atterrir sur son dossier.
        name += "_fwadd"
    if cfg.bank_q:
        name += "_bq"
    if cfg.read_path == "dual" and (cfg.bank_heads != ToyCfg.bank_heads
                                    or cfg.bank_head_dim):
        name += f"_bh{cfg.bank_heads}"
        if cfg.bank_head_dim:
            name += f"x{cfg.bank_head_dim}"
    if cfg.cond_arm != ToyCfg.cond_arm:
        name += f"_arm-{cfg.cond_arm}"
    if cfg.cond_decoys != ToyCfg.cond_decoys:
        name += f"_dec{cfg.cond_decoys}"
    if cfg.write_mode != ToyCfg.write_mode:
        name += f"_w{cfg.write_mode}"
    return name


# ══ PHASE 11 — NOMMAGE ET GRILLE DES QUATRE EXAMENS ═════════════════════════
# Un examen = une question du registre §3, un jeu de bras, un save_dir par
# cellule. Le préfixe `p11-` isole les 22 cellules des 96 déjà lancées, et le
# ROOT de save_dir change aussi (checkpoints/toy_read_lab_p11) : deux barrières
# indépendantes contre l'écrasement d'un run fini.
P11_ARM = {"none": "agezero", "age-log": "agelog", "age-raw": "ageraw",
           "age-bias": "agebias"}


def p11_exam(cfg: ToyCfg) -> str:
    """L'examen DÉCLARÉ d'une config (cf. ToyCfg.p11_exam)."""
    return cfg.p11_exam


def p11_name(cfg: ToyCfg) -> str:
    """`p11-<examen>_<bras>_m<k>` — le nom se relit comme la cellule."""
    ex = p11_exam(cfg)
    if ex in ("age", "ood"):
        arm = P11_ARM[cfg.bank_rot] + ("-aug" if cfg.age_aug else "")
    elif ex == "tag":
        # `-sv` = pool de valeurs `span` (instrument v2, 08-04) — legacy `ref`
        # sans suffixe pour ne pas renommer les cellules déjà dépouillées.
        arm = "tag" + cfg.tag_mode + ("-sv" if cfg.prov_vals == "span" else "")
    else:
        arm = "loc" + cfg.loc_mode
    return f"p11-{ex}_{arm}_m{cfg.top_k}"


# Les bras de chaque examen, EN CLAIR. Chaque liste contient son CONTRÔLE (la
# baseline à battre) — sans lui l'examen ne tranche rien.
P11_EXAMS = {
    # S3 — la rotation d'âge doit BATTRE θ_âge=0 (contrôle HoPE, non
    # négociable) ; `age-bias` est le fallback prévu par la règle de décision.
    "age": {"env": "rule", "ms": (4, 8),
            "arms": [{"bank_rot": "none"}, {"bank_rot": "age-log"},
                     {"bank_rot": "age-raw"}, {"bank_rot": "age-bias"}]},
    # S4 — OOD d'âge. Les comparateurs NON augmentés sont les cellules m4 de
    # l'examen `age` (mêmes graine, mêmes steps, même env : appariement exact),
    # on ne les relance pas. Ne sont neuves que les deux augmentées.
    "ood": {"env": "rule", "ms": (4,),
            "arms": [{"bank_rot": "age-raw", "age_aug": True},
                     {"bank_rot": "age-log", "age_aug": True}]},
    # S5 — tag de provenance : rotatif contre additif contre rien.
    "tag": {"env": "prov", "ms": (4, 8),
            "arms": [{"tag_mode": "none"}, {"tag_mode": "rot"},
                     {"tag_mode": "add"}]},
    # S17 — index local intra-span : rotatif contre additif contre rien.
    "locidx": {"env": "span", "ms": (4, 8),
               "arms": [{"loc_mode": "none"}, {"loc_mode": "rot"},
                        {"loc_mode": "add"}]},
}


# ══ PHASE 12 — NOMMAGE ET GRILLE DES DEUX EXAMENS (S6, S8) ═════════════════
# Préfixe `p12-` et ROOT de save_dir distinct (checkpoints/toy_read_lab_p12) :
# deux barrières indépendantes, comme en ph.11, contre l'écrasement des ~118
# cellules déjà lancées.
P12_ARM = {"fifo": "fifo", "age": "age", "attn-ema": "attnema",
           "coverage": "cover", "actr": "actr"}


def p12_name(cfg: ToyCfg) -> str:
    """`p12-retention_<signal>[-p<budget>]_T<turns>` ou `p12-dilution_S<S>_m<k>`."""
    if cfg.p12_exam == "dilution":
        rd = "" if cfg.read_path == "kvproj" else f"-{cfg.read_path}"
        return (f"p12-dilution_S{cfg.max_mem}-{cfg.p11_env}{rd}"
                f"_m{cfg.top_k}")
    arm = P12_ARM[cfg.retention]
    if cfg.prop_budget:
        arm += f"-p{cfg.prop_budget}"
    # Les bras de ROTATION entrent dans le nom — sans quoi deux cellules qui ne
    # diffèrent que par bank_rot/tag/loc partagent le save_dir et la seconde
    # ÉCRASE la première (payé 08-04 : zzs115/116 agelog ont détruit les
    # results.json/ckpts des bras base zzs101/103 ; seuls les logs survivent).
    if cfg.bank_rot != "none":
        arm += "-" + P11_ARM[cfg.bank_rot] + ("-aug" if cfg.age_aug else "")
    if cfg.tag_mode != "none":
        arm += f"-tag{cfg.tag_mode}"
    if cfg.loc_mode != "none":
        arm += f"-loc{cfg.loc_mode}"
    # `-sv` = pool de valeurs `span` (fix régime de données 08-04) — legacy
    # `city` sans suffixe, comme prov-sv en ph.11.
    if cfg.life_vals == "span":
        arm += "-sv"
    return f"p12-retention_{arm}_T{cfg.life_turns}"


# Les cellules de chaque examen, EN CLAIR. `fifo` est la BASELINE BASSE de
# `retention` et `S8` l'appariement de `dilution` — sans eux, rien ne tranche.
P12_EXAMS = {
    # S6 — bakeoff de rétention. Trois blocs : budget 1 sur une vie courte
    # (les 5 signaux), budget 2 sur la MÊME vie (la FIFO nue y est
    # p-INDÉPENDANTE : sa cellule du bloc p1 SERT de baseline aux deux, on ne
    # la relance pas), puis budget 1 sur une vie DEUX FOIS plus longue.
    "retention": {
        "env": "life", "m": 4, "steps": 1500,
        "cells": [{"retention": s, "prop_budget": (0 if s == "fifo" else p),
                   "life_turns": T}
                  for (p, T) in ((1, 48), (2, 48), (1, 88))
                  for s in RETENTION_SIGNALS
                  if not (s == "fifo" and p == 2)]},
    # S8 — courbe de dilution à read kvproj FIGÉ (m=4). La banque est remplie
    # de distracteurs RÉELS jusqu'à S groupes ; S=1 est le PLAFOND (aucun
    # distracteur possible) et S=8 l'appariement avec le carré factoriel.
    # Deux envs parce que les deux moitiés de la question ne vivent pas au même
    # endroit : `rule` porte le CONDITIONNEMENT (2AFC + marges de marqueurs,
    # la mesure dont la ph.10 a prouvé qu'elle passe à ce tap), `span` porte la
    # CITATION (grade + r@1 par masse d'attention). La prédiction inscrite est
    # que la citation s'écroule AVANT le conditionnement.
    # BRAS `dual_heads` AUX GRANDS S (ajout 08-03 nuit, audit des absolus) :
    # dual_heads est requalifié « DOMINÉ COÛT-AJUSTÉ, candidat si le softmax
    # unifié plafonne à grande banque ». Mécanistiquement c'est exactement le
    # point de S8 : à S grand, le softmax UNIFIÉ partage sa masse entre
    # contexte et banque (la dilution même qu'on mesure), le softmax SÉPARÉ y
    # est immunisé PAR CONSTRUCTION. Si kvproj s'écroule à S=64 et pas
    # dual_heads, le rappel de candidature est automatique — AVANT de payer la
    # hiérarchie (S9). Streams et graine APPARIÉS aux cellules kvproj.
    "dilution": {
        "env": "rule", "m": 4, "steps": 3000,
        "cells": ([{"max_mem": S, "env": "rule"} for S in (1, 4, 8, 16, 64)]
                  + [{"max_mem": S, "env": "span"} for S in (8, 64)]
                  + [{"max_mem": S, "env": "rule", "read": "dual"}
                     for S in (16, 64)])},
}


def p12_combos(exam: str) -> list:
    """Les cellules d'un examen ph.12, dans un ordre stable."""
    assert exam in P12_EXAMS, f"examen inconnu {exam!r} (∈ {tuple(P12_EXAMS)})"
    spec = P12_EXAMS[exam]
    return [{"exam": exam, "env": c.get("env", spec["env"]),
             "m": spec["m"], "steps": spec["steps"],
             **{k: v for k, v in c.items() if k != "env"}}
            for c in spec["cells"]]


def _p12_cfg(combo: dict, base: dict) -> ToyCfg:
    """Config d'une cellule ph.12 : kvproj + tap postnorm, le SOMMET ADOPTÉ.

    Comme en ph.11, les axes tranchés sont FIXÉS et non rebalayés (S1 : kvproj,
    prélèvement postnorm). Ne varient que la maintenance (S6) ou S (S8).
    """
    kw = {k: v for k, v in combo.items()
          if k not in ("exam", "env", "m", "steps", "read")}
    rp = {"kvproj": "kvproj", "dual": "dual"}[combo.get("read", "kvproj")]
    if combo["exam"] == "dilution":
        kw["bank_fill"] = "foreign"
        # cond_decoys 0 : UN SEUL vrai groupe, tout le reste est du
        # remplissage. C'est la seule façon d'avoir un cadran PROPRE (S), au
        # lieu de mélanger « nombre de leurres de la vie » et « taille de la
        # banque ».
        if combo["env"] == "rule":
            kw["cond_decoys"] = 0
    return ToyCfg(**{**base, "variant": "r4", "code": "tophid",
                     "read_path": rp, "top_k": int(combo["m"]),
                     "p11_env": combo["env"], "p12_exam": combo["exam"],
                     "p11_exam": "", "cond": combo["env"] == "rule", **kw})


def p11_combos(exam: str) -> list:
    """Les cellules d'un examen, dans un ordre stable."""
    assert exam in P11_EXAMS, f"examen inconnu {exam!r} (∈ {tuple(P11_EXAMS)})"
    spec = P11_EXAMS[exam]
    return [{"exam": exam, "env": spec["env"], "m": int(m), **arm}
            for arm in spec["arms"] for m in spec["ms"]]


def _p11_cfg(combo: dict, base: dict) -> ToyCfg:
    """Config d'une cellule ph.11 : kvproj + tap postnorm, le SOMMET ADOPTÉ.

    Les axes déjà tranchés sont FIXÉS, pas rebalayés — `kv_proj` (S1) et le
    prélèvement postnorm (tap ≈ neutre au jouet, ph.10 §6). Ne varient que le
    bras de métadonnées et m.
    """
    kw = {k: v for k, v in combo.items() if k not in ("exam", "env", "m")}
    return ToyCfg(**{**base, "variant": "r4", "code": "tophid",
                     "read_path": "kvproj", "top_k": int(combo["m"]),
                     "p11_env": combo["env"], "p11_exam": combo["exam"],
                     "cond": combo["env"] == "rule", **kw})


def grid_combos(reads=("seq_fw", "inject_entry", "kv_append"),
                rots=(False, True), taps=("postnorm", "mid"),
                ms=(1, 4, 8), fw_additive: bool = False,
                bank_q: bool = False) -> list:
    """La GRILLE COMPLÈTE, en clair. Un dict par combo, dans un ordre stable."""
    out = []
    for rd in reads:
        for tp in taps:
            for m in ms:
                for rot in rots:
                    out.append({"read": rd, "age_rot": bool(rot), "tap": tp,
                                "m": int(m),
                                # le correctif ne s'applique QU'au bras à
                                # boucle : demander l'additif sur les bras
                                # attention n'aurait aucun sens (et ToyCfg le
                                # refuse).
                                "fw_additive": bool(fw_additive)
                                and rd == "seq_fw",
                                "bank_q": bool(bank_q) and rd == "kv_proj"})
    return out


# sous-ensembles nommés du manifeste (cf. --manifest-subset)
GRID_SUBSETS = {
    "all":            dict(),
    "seqfw":          dict(reads=("seq_fw",)),
    # LES 12 CELLULES DU CORRECTIF : mêmes axes rot/tap/m, gradient ADDITIF,
    # noms suffixés _fwadd ⇒ elles n'écrasent pas le bras multiplicatif.
    "seqfw-additive": dict(reads=("seq_fw",), fw_additive=True),
    "attn":           dict(reads=("inject_entry", "kv_append")),
    # LE NOUVEAU BRAS : 12 cellules, mêmes axes rot/tap/m. `all` reste à 36 —
    # c'est la grille DÉJÀ LANCÉE, on ne redéfinit pas un total en cours de
    # campagne (36 multiplicatives + 12 _fwadd + 12 dual = 60 au bout).
    "dual":           dict(reads=("dual_heads",)),
    "kvproj":         dict(reads=("kv_proj",)),
    "kvproj-bq":      dict(reads=("kv_proj",), bank_q=True),
}


def _grid_cfg(combo: dict, base: dict) -> ToyCfg:
    v, c, rp = READ_MODES[combo["read"]]
    code = {"native": "toprows", "postnorm": "tophid",
            "mid": "midhid"}[combo["tap"]] if c is None else c
    return ToyCfg(**{**base, "variant": v, "code": code, "read_path": rp,
                     "age_rope": combo["age_rot"], "top_k": int(combo["m"]),
                     "fw_additive": bool(combo.get("fw_additive")),
                     "bank_q": bool(combo.get("bank_q")),
                     "cond": True})


P11_NOTE = {
    "age": ("S3 — la rotation doit BATTRE le contrôle θ_âge=0 (`agezero`, "
            "HoPE) sur la CITATION ; sinon l'âge passe en biais scalaire "
            "(`agebias`, 1 paramètre)"),
    "ood": ("S4 — OOD d'âge : chaque run est ÉVALUÉ aux échelles 1/10/100. "
            "Comparateurs non augmentés = les cellules m4 de l'examen `age` "
            "(mêmes graine/steps/env, appariement exact — NE PAS relancer)"),
    "tag": ("S5 — A/B INÉDIT rotation vs additif sur un tag catégoriel : "
            "mesurer r@1 (masse d'attention par groupe) ET grade de copie "
            "SÉPARÉMENT, la rotation peut aider l'un et casser l'autre"),
    "locidx": ("S17 — prédiction PRÉENREGISTRÉE : l'écart rot−add CROÎT avec "
               "la longueur du span (l'additif donne des signatures, jamais "
               "l'opérateur successeur R_loc(1))"),
}


def print_p11_manifest(config: str, base: dict, save_root: str,
                       fmt: str = "tsv", b_convs: int = 8,
                       exam: str = "age") -> None:
    """Le manifeste d'un examen de la phase 11 (cf. print_grid_manifest)."""
    rows = []
    for combo in p11_combos(exam):
        cfg = _p11_cfg(combo, base)
        m = combo["m"]
        G = cfg.cond_groups if cfg.cond else 2
        mod = ToyReadLM(cfg, 11, 12, sif_w=torch.ones(cfg.vocab_size))
        pr = param_report(mod)
        rot = next((b.attn.rot for b in mod.blocks
                    if getattr(b.attn, "rot", None) is not None), None)
        del mod
        opt_go = pr["total"] * 16 / 2 ** 30
        act_go = (b_convs * cfg.max_seq_len * cfg.vocab_size * 4 * 3) / 2 ** 30
        vram = opt_go + act_go
        arm = (P11_ARM[cfg.bank_rot] + ("-aug" if cfg.age_aug else "")
               if exam in ("age", "ood") else
               "tag" + cfg.tag_mode if exam == "tag" else "loc" + cfg.loc_mode)
        flags = ""
        if cfg.bank_rot != "none":
            flags += f" --bank-rot {cfg.bank_rot}"
        if cfg.age_aug:
            flags += " --age-aug"
        if cfg.tag_mode != "none":
            flags += f" --tag {cfg.tag_mode}"
        if cfg.loc_mode != "none":
            flags += f" --locidx {cfg.loc_mode}"
        cmd = (f"python -m deepseek_v4_mini.toy_read_lab {config} "
               f"--read kv_proj --tap postnorm --m {m} "
               f"--p11-exam {exam} --p11-env {combo['env']}{flags}"
               + (" --cond" if cfg.cond else ""))
        note = P11_NOTE[exam]
        if rot is not None:
            note += (f" | plans {rot.na}âge/{rot.nc}canal/{rot.nl}local, "
                     f"dérive de requête {rot.drift:.3f} rad sur la fenêtre "
                     f"(max {cfg.rot_drift_max})")
        if vram > 8.0:
            note += f" | ⚠️ NE TIENT PAS EN 8 Go ({vram:.1f} Go estimés)"
        rows.append({"run": p11_name(cfg), "exam": exam, "arm": arm,
                     "env": combo["env"], "m": m,
                     "load": f"{G * m} clés projetées (softmax UNIFIÉ)",
                     "params_M": f"{pr['total'] / 1e6:.1f}",
                     "vram_Go_est": f"{vram:.2f}",
                     "cout_rel": f"{1.22 * (1.0 + 0.03 * (m - 1)):.2f}",
                     "save_dir": os.path.join(save_root, p11_name(cfg)),
                     "cmd": cmd, "note": note})
    if fmt == "json":
        import json
        print(json.dumps(rows, indent=2, ensure_ascii=False))
        return
    cols = ["run", "exam", "arm", "env", "m", "load", "params_M",
            "vram_Go_est", "cout_rel", "save_dir", "cmd", "note"]
    print("\t".join(cols))
    for r in rows:
        print("\t".join(str(r[c]) for c in cols))


P12_NOTE = {
    "retention": (
        "S6 — BAKEOFF de rétention, tout PROCÉDURAL (aucun RL, aucune "
        "politique apprise). Métriques de première classe : (a) temps "
        "d'adaptation au fait NOUVEAU par horizon dw, (b) survie du fait "
        "ANCIEN utile (strate `hist` — FIFO et récence y scorent 0 PAR "
        "CONSTRUCTION), (c) grade + r@1, (d) horizon effectif mesuré"),
    "dilution": (
        "S8 — KT8 au jouet : où la PLATE s'écroule. Banque remplie de "
        "distracteurs RÉELS (groupes d'autres vies, lignes recalculées avec "
        "le modèle courant) jusqu'à S groupes ; S=1 = plafond, S=8 = "
        "appariement du carré factoriel. C'est la baseline d'entrée de la "
        "hiérarchie (S9)"),
}


def print_p12_manifest(config: str, base: dict, save_root: str,
                       fmt: str = "tsv", b_convs: int = 8,
                       exam: str = "retention") -> None:
    """Le manifeste d'un examen de la phase 12 (cf. print_p11_manifest)."""
    rows = []
    for combo in p12_combos(exam):
        cfg = _p12_cfg(combo, base)
        m, S = combo["m"], cfg.max_mem
        G = (S if cfg.bank_fill != "none" else
             (cfg.cond_groups if cfg.cond else S))
        mod = ToyReadLM(cfg, 11, 12, sif_w=torch.ones(cfg.vocab_size))
        pr = param_report(mod)
        del mod
        opt_go = pr["total"] * 16 / 2 ** 30
        act_go = (b_convs * cfg.max_seq_len * cfg.vocab_size * 4 * 3) / 2 ** 30
        # les lignes de banque entrent aux K/V de CHAQUE couche lectrice : à
        # S=64 c'est 64·m clés de plus par tête, et le masque flottant du
        # softmax unifié est [B,h,T,S·m+T] — c'est LUI qui grandit avec S.
        msk_go = (b_convs * cfg.n_heads * cfg.max_seq_len
                  * (G * m + cfg.max_seq_len) * 2 * cfg.n_layers) / 2 ** 30
        vram = opt_go + act_go + msk_go
        if exam == "retention":
            arm = P12_ARM[cfg.retention] + (f"-p{cfg.prop_budget}"
                                            if cfg.prop_budget else "")
            flags = (f" --retention {cfg.retention}"
                     + (f" --prop {cfg.prop_budget}" if cfg.prop_budget else "")
                     + f" --life-turns {cfg.life_turns}")
        else:
            arm = f"S{S}-{cfg.p11_env}" + ("" if cfg.read_path == "kvproj"
                                           else f"-{cfg.read_path}")
            flags = (f" --max-mem {S} --bank-fill foreign"
                     + (" --cond-decoys 0" if cfg.cond else ""))
        rd = "kv_proj" if cfg.read_path == "kvproj" else "dual_heads"
        cmd = (f"python -m deepseek_v4_mini.toy_read_lab {config} "
               f"--read {rd} --tap postnorm --m {m} "
               f"--p12-exam {exam} --p11-env {combo['env']}{flags} "
               f"--steps {combo['steps']}"
               + (" --cond" if cfg.cond else ""))
        note = P12_NOTE[exam]
        if exam == "retention":
            note += (f" | vie ≈ {cfg.life_turns} segs ⇒ ≈ "
                     f"{2 + max(4, (cfg.life_turns - 8) // 4)} writes contre "
                     f"max_mem {S} (pression FIFO réelle)")
            if cfg.retention in ("attn-ema", "actr"):
                note += (" | UN forward no-grad de sonde par write (le vote "
                         "du lecteur), masse du tour t utilisée au write t+1")
        elif cfg.read_path == "dual":
            note += (f" | {G} groupes × {m} lignes = {G * m} clés dans un "
                     f"softmax SÉPARÉ ({cfg.bank_heads} têtes dédiées) — le "
                     f"bras immunisé PAR CONSTRUCTION au partage de masse")
        else:
            note += (f" | {G} groupes × {m} lignes = {G * m} clés projetées "
                     f"dans le softmax UNIFIÉ")
        if vram > 7.0:
            note += f" | ⚠️ NE TIENT PAS EN 7 Go ({vram:.1f} Go estimés)"
        rows.append({"run": p12_name(cfg), "exam": exam, "arm": arm,
                     "env": combo["env"], "S": S, "m": m,
                     "steps": combo["steps"],
                     "load": f"{G * m} clés projetées (softmax UNIFIÉ)",
                     "params_M": f"{pr['total'] / 1e6:.1f}",
                     "vram_Go_est": f"{vram:.2f}",
                     "save_dir": os.path.join(save_root, p12_name(cfg)),
                     "cmd": cmd, "note": note})
    if fmt == "json":
        import json
        print(json.dumps(rows, indent=2, ensure_ascii=False))
        return
    cols = ["run", "exam", "arm", "env", "S", "m", "steps", "load",
            "params_M", "vram_Go_est", "save_dir", "cmd", "note"]
    print("\t".join(cols))
    for r in rows:
        print("\t".join(str(r[c]) for c in cols))


def print_grid_manifest(config: str, base: dict, save_root: str,
                        fmt: str = "tsv", b_convs: int = 8,
                        subset: str = "all") -> None:
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
    for combo in grid_combos(**GRID_SUBSETS[subset]):
        cfg = _grid_cfg(combo, base)
        m = combo["m"]
        G = cfg.cond_groups
        if cfg.variant == "r0":
            load = cfg.max_mem * cfg.group_rows
            unit = "sous-slots seq"
        elif cfg.read_path == "dual":
            load = G * m
            unit = f"clés / {cfg.bank_heads} têtes dédiées"
        elif cfg.read_path == "kvproj":
            load = G * m
            unit = "clés projetées (softmax UNIFIÉ)"
        elif cfg.read_path == "kv":
            load = G * m
            unit = "K/V add."
        else:
            load = G * (m + 1)
            unit = "pos. préfixe"
        cmd = (f"python -m deepseek_v4_mini.toy_read_lab {config} --cond "
               f"--read {combo['read']} --tap {combo['tap']} --m {m}"
               + (" --age-rot" if combo["age_rot"] else "")
               + (" --fw-additive" if combo.get("fw_additive") else "")
               + (" --bank-q" if combo.get("bank_q") else ""))
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
        rel = {"seq_fw": 2.0, "inject_entry": 1.0, "kv_append": 1.07,
               "dual_heads": 1.18, "kv_proj": 1.22}[combo["read"]] * (
                   1.0 + 0.03 * (m - 1)) * (1.35 if combo.get("bank_q") else 1.0)
        note = ("boucle fast-weight de %d itérations × %d couches : le bras "
                "LENT (×2.0 relevé)" % (load, cfg.n_layers)
                if combo["read"] == "seq_fw" else "")
        if combo["read"] == "kv_proj":
            note = ("projections DÉDIÉES dans le softmax UNIFIÉ + biais de "
                    "logits appris par tête (init 0, sa valeur EST une mesure)"
                    + (" | bank-q : les lignes ÉMETTENT et se contextualisent "
                       "de couche lectrice en couche lectrice (side-stream, "
                       "sans MLP)" if combo.get("bank_q") else ""))
        if combo["read"] == "dual_heads":
            note = ("groupe de têtes DÉDIÉ : masse softmax garantie (la banque "
                    "ne se dispute rien avec le contexte), au prix de "
                    "paramètres — le seul bras de la grille qui en ajoute")
        if combo.get("fw_additive"):
            note += (" | gradient ADDITIF (stop-grad sur l'état porté) : "
                     "forward inchangé, plus de produit de jacobiennes")
        elif combo["read"] == "seq_fw":
            note += (" | gradient MULTIPLICATIF : DIVERGE sur la ferme dans "
                     "5 cellules sur 6 (gnorm pré-clip jusqu'à 4.7e9) — "
                     "cf. --manifest-subset seqfw-additive")
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
    if cfg.p12_exam:
        return p12_name(cfg)
    if cfg.p11_exam:
        return p11_name(cfg)
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
    if cfg.fw_additive:
        name += "_fwadd"
    if cfg.bank_q:
        name += "_bq"
    if cfg.read_path == "dual" and (cfg.bank_heads != ToyCfg.bank_heads
                                    or cfg.bank_head_dim):
        name += f"_bh{cfg.bank_heads}" + (f"x{cfg.bank_head_dim}"
                                          if cfg.bank_head_dim else "")
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
    ap.add_argument("--bank-heads", type=int, default=None, dest="bank_heads",
                    help="--read dual_heads : nombre de têtes du GROUPE BANQUE "
                         "par couche lectrice (défaut 2)")
    ap.add_argument("--bank-head-dim", type=int, default=None,
                    dest="bank_head_dim",
                    help="--read dual_heads : dim par tête du groupe banque "
                         "(0 = celle des têtes de contexte)")
    ap.add_argument("--bank-q", action="store_true", dest="bank_q",
                    help="--read kv_proj : les lignes de banque ÉMETTENT leurs "
                         "propres requêtes (W_q' dédié) et se contextualisent "
                         "de couche lectrice en couche lectrice (side-stream : "
                         "attention seule, pas de MLP ; état jeté en sortie de "
                         "stack). Suffixe _bq au save_dir.")
    ap.add_argument("--fw-additive", action="store_true", dest="fw_additive",
                    help="CORRECTIF TERRAIN du bras `--read seq_fw` : rend la "
                         "composition du gradient de la boucle fast-weight "
                         "ADDITIVE (stop-gradient sur l'état porté). Le "
                         "forward est numériquement INCHANGÉ, seul le backward "
                         "l'est — le produit de jacobiennes sur M×n_layers "
                         "étages disparaît. Défaut OFF (le bras multiplicatif "
                         "divergent est une donnée de la grille). Ajoute "
                         "_fwadd au save_dir. Refusé hors variantes à boucle.")
    # ── PHASE 11 : les trois familles de métadonnées (spec §2.5) ───────────
    ap.add_argument("--bank-rot", choices=BANK_ROTS, default=None,
                    dest="bank_rot",
                    help="famille ÂGE sur les clés banque projetées (S3/S4) : "
                         "none = le CONTRÔLE θ_âge=0 (HoPE, la baseline à "
                         "battre) | age-log = φ(a) log-comprimé, fréquences "
                         "apprises | age-raw = âge brut | age-bias = pas de "
                         "rotation, un biais scalaire de récence w·a sur le "
                         "logit banque (le fallback de la règle S3)")
    ap.add_argument("--age-aug", action="store_true", dest="age_aug",
                    help="S4 : au TRAIN, échelle des âges tirée log-uniforme "
                         "dans [1, age_aug_max] (l'ordre est préservé). "
                         "Steelman du bras brut face à l'OOD.")
    ap.add_argument("--age-eval-scales", default=None, dest="age_eval_scales",
                    help="S4 : échelles d'âge de l'éval contrastive, la "
                         "première étant 1 (le régime du train). Défaut "
                         "1,10,100.")
    ap.add_argument("--tag", choices=TAG_MODES, default=None, dest="tag_mode",
                    help="famille PROVENANCE (S5) : none | rot (un plan 0/π "
                         "PAR CANAL) | add (vecteur appris par canal sur les "
                         "MÊMES dims réservées). Exige --p11-env prov.")
    ap.add_argument("--prov-vals", choices=("ref", "span"), default=None,
                    dest="prov_vals",
                    help="pool de valeurs de l'env prov (S5) : ref (legacy, "
                         "codes XX-12345 — examen INVALIDÉ 08-04, copie hors "
                         "de portée) | span (valeurs mesurées en tokens, "
                         "buckets courts — l'instrument v2).")
    ap.add_argument("--life-vals", choices=("city", "span"), default=None,
                    dest="life_vals",
                    help="pool de valeurs de l'env life (S6) : city (legacy, "
                         "répertoire fermé d'entités — régime INVALIDÉ 08-04, "
                         "le train se minimise par reconnaissance et la copie "
                         "ne se forme jamais) | span (valeurs mesurées "
                         "buckets 1-2, compositionnelles, L≤2 — le fix).")
    ap.add_argument("--locidx", choices=LOC_MODES, default=None,
                    dest="loc_mode",
                    help="famille INDEX LOCAL intra-span (S17) : none | rot "
                         "(R_loc(j), opérateur successeur constant) | add "
                         "(embedding de position locale appris).")
    ap.add_argument("--p11-exam", choices=P11_EXAM_NAMES, default=None,
                    dest="p11_exam",
                    help="DÉCLARE l'examen de la cellule (age|ood|tag|locidx) "
                         "— c'est lui qui nomme le save_dir. Obligatoire dès "
                         "qu'une famille de métadonnées ou un env ph.11 est "
                         "actif : le bras de CONTRÔLE (agezero) n'active rien "
                         "et n'aurait sinon aucun moyen de se distinguer d'une "
                         "cellule de la grille §2.4.")
    ap.add_argument("--p11-env", choices=P11_ENVS, default=None,
                    dest="p11_env",
                    help="env de la phase 11 : rule (la vie-règle de la "
                         "ph.10) | prov (locuteur user/self, S5) | span "
                         "(valeurs de longueur graduée, S17)")
    # ── PHASE 12 : maintenance procédurale (S6) et dilution (S8) ──────────
    ap.add_argument("--retention", choices=RETENTION_SIGNALS, default=None,
                    help="SIGNAL DE RÉTENTION du plug-in procédural (S6) : "
                         "fifo (baseline basse, aucune propagation) | age (le "
                         "proxy RÉPUDIÉ, la baseline à battre) | attn-ema (le "
                         "lecteur vote, masse d'attention en EMA) | coverage "
                         "(la redondance meurt, la singularité survit) | actr "
                         "(activation ACT-R). Aucun gradient : c'est de la "
                         "maintenance, pas un module.")
    ap.add_argument("--prop", type=int, default=None, dest="prop_budget",
                    help="BUDGET DE PROPAGATION p : lignes de la queue "
                         "replacées en tête à chaque write, compteur de "
                         "naissance PRÉSERVÉ (l'âge reste l'âge vrai). 0 = "
                         "FIFO nue.")
    ap.add_argument("--life-turns", type=int, default=None, dest="life_turns",
                    help="env `life` : longueur CIBLE d'une vie en segs. Le "
                         "nombre de writes en découle et doit rester ≫ "
                         "max_mem (c'est toute la pression FIFO de S6).")
    ap.add_argument("--bank-fill", choices=BANK_FILLS, default=None,
                    dest="bank_fill",
                    help="S8 : `foreign` remplit la banque jusqu'à max_mem "
                         "GROUPES avec des groupes d'AUTRES vies du même "
                         "stream (distracteurs RÉELS, lignes recalculées avec "
                         "le modèle courant). Les vrais résidents restent en "
                         "dernier et les remplisseurs ont des métadonnées "
                         "nulles : ils ne peuvent que DILUER.")
    ap.add_argument("--max-mem", type=int, default=None, dest="max_mem",
                    help="S8 : taille S de la banque en GROUPES (surcharge "
                         "model.max_mem — c'est LE cadran de la courbe de "
                         "dilution)")
    ap.add_argument("--p12-exam", choices=P12_EXAM_NAMES, default=None,
                    dest="p12_exam",
                    help="DÉCLARE l'examen ph.12 (retention|dilution) — c'est "
                         "lui qui nomme le save_dir. Obligatoire dès qu'une "
                         "maintenance ou un remplissage est actif.")
    ap.add_argument("--manifest-subset", default="all",
                    choices=tuple(GRID_SUBSETS) + tuple(P11_EXAMS)
                    + tuple(P12_EXAMS),
                    dest="manifest_subset",
                    help="restreint le manifeste : all (36) | seqfw (les 12 "
                         "cellules fast-weight) | seqfw-additive (les MÊMES 12 "
                         "axes rot/tap/m avec --fw-additive, noms suffixés "
                         "_fwadd) | attn (les 24 cellules inject/kv) | dual "
                         "(les 12 cellules à TÊTES DÉDIÉES) | kvproj | "
                         "kvproj-bq ; PHASE 11 : age (8) | ood (2) | tag (6) "
                         "| locidx (6) ; PHASE 12 : retention (14) | "
                         "dilution (7)")
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
                      ("cond_decoys", int), ("fw_additive", bool),
                      ("bank_heads", int), ("bank_head_dim", int),
                      ("bank_q", bool)):
        if key in cb:
            mc[key] = cast(cb[key])
    if a.read is not None:
        mc["read_path"] = read_path
    if a.age_rot:
        mc["age_rope"] = True
    if a.m_rows is not None:
        mc["top_k"] = int(a.m_rows)
    if a.fw_additive:
        mc["fw_additive"] = True
    if a.bank_heads is not None:
        mc["bank_heads"] = int(a.bank_heads)
    if a.bank_head_dim is not None:
        mc["bank_head_dim"] = int(a.bank_head_dim)
    if a.bank_q:
        mc["bank_q"] = True
    if a.cond_arm is not None:
        mc["cond_arm"] = a.cond_arm
    if a.cond_decoys is not None:
        mc["cond_decoys"] = int(a.cond_decoys)
    mc["cond"] = bool(a.cond)
    # ── PHASE 11 : bloc YAML `p11:` puis surcharges CLI ────────────────────
    pb = dict(raw.get("p11") or {})
    for key, cast in (("bank_rot", str), ("age_planes", int), ("age_ref", int),
                      ("age_aug", bool), ("age_aug_max", float),
                      ("age_eval_scales", str), ("tag_mode", str),
                      ("prov_vals", str),
                      ("n_channels", int), ("loc_mode", str),
                      ("loc_planes", int), ("rot_drift_max", float),
                      ("p11_env", str), ("p11_exam", str)):
        if key in pb:
            mc[key] = cast(pb[key])
    for key, val in (("bank_rot", a.bank_rot), ("tag_mode", a.tag_mode),
                     ("prov_vals", a.prov_vals),
                     ("loc_mode", a.loc_mode), ("p11_env", a.p11_env),
                     ("p11_exam", a.p11_exam),
                     ("age_eval_scales", a.age_eval_scales)):
        if val is not None:
            mc[key] = val
    if a.age_aug:
        mc["age_aug"] = True
    # ── PHASE 12 : bloc YAML `p12:` puis surcharges CLI ────────────────────
    qb = dict(raw.get("p12") or {})
    for key, cast in (("retention", str), ("prop_budget", int),
                      ("ema_beta", float), ("actr_decay", float),
                      ("life_vals", str),
                      ("life_turns", int), ("bank_fill", str),
                      ("fill_pool", int), ("fill_refresh", int),
                      ("p12_exam", str)):
        if key in qb:
            mc[key] = cast(qb[key])
    for key, val in (("retention", a.retention),
                     ("prop_budget", a.prop_budget),
                     ("life_vals", a.life_vals),
                     ("life_turns", a.life_turns),
                     ("bank_fill", a.bank_fill), ("max_mem", a.max_mem),
                     ("p12_exam", a.p12_exam)):
        if val is not None:
            mc[key] = val
    if a.manifest and a.manifest_subset in P12_EXAMS:
        print_p12_manifest(
            a.config, {k: v for k, v in mc.items()
                       if k not in ("variant", "code", "read_path", "top_k",
                                    "cond", "cond_decoys", "p11_env",
                                    "p11_exam", "p12_exam", "max_mem",
                                    "retention", "prop_budget", "life_turns",
                                    "bank_fill")},
            t.get("save_dir", "./checkpoints/toy_read_lab"), a.manifest,
            b_convs=int(t.get("batch_convs", 8)), exam=a.manifest_subset)
        return
    if a.manifest and a.manifest_subset in P11_EXAMS:
        print_p11_manifest(
            a.config, {k: v for k, v in mc.items()
                       if k not in ("variant", "code", "read_path", "top_k",
                                    "cond", "p11_env", "p11_exam",
                                    "bank_rot", "age_aug", "tag_mode",
                                    "loc_mode")},
            t.get("save_dir", "./checkpoints/toy_read_lab"), a.manifest,
            b_convs=int(t.get("batch_convs", 8)), exam=a.manifest_subset)
        return
    if a.manifest:
        print_grid_manifest(
            a.config, {k: v for k, v in mc.items()
                       if k not in ("variant", "code", "read_path", "age_rope",
                                    "top_k", "cond")},
            t.get("save_dir", "./checkpoints/toy_read_lab"), a.manifest,
            b_convs=int(t.get("batch_convs", 8)),
            subset=a.manifest_subset)
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
        if any(mc.get(k, "none") != "none" for k in ("bank_rot", "tag_mode",
                                                     "loc_mode")):
            # PHASE 11 : à 4 têtes le smoke n'a que 8 paires par tête et la
            # bande quasi statique n'y loge pas 4 plans (dérive 3,19 rad sur la
            # fenêtre) — le garde-fou §2.5 refuserait, à raison. On élargit la
            # TÊTE (2 têtes de 32 dims) plutôt que d'assouplir le garde-fou :
            # un smoke qui passe en desserrant l'invariant ne prouve rien.
            mc["n_heads"] = 2
        steps, b_convs, eval_every, eval_convs, max_new = 2, 2, 1, 1, 8
        cond_eval_convs = 2

    torch.manual_seed(int(t.get("seed", 0)))
    tok = build_tokenizer(raw["tokenizer"])
    env = OracleEnv(tok, int(mc.get("max_mem", 8)), write_mode=a.write_mode,
                    span_vals=(mc.get("life_vals") == "span"))

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
    # PHASE 11 : `prov`/`span` REMPLACENT le stream (leurs tours gradés portent
    # leur question — cf. Persona11Stream).
    # PHASE 12 : `life` est le troisième env à question-dans-le-seg.
    P = ({"prov": PersonaProvStream, "span": PersonaSpanStream,
          "life": PersonaLifeStream}[cfg.p11_env]
         if cfg.p11_env != "rule" else
         PersonaRuleStream if cfg.cond else chat_stream_class("persona"))
    # kwargs du stream ph.11 : filtrés PAR PRÉFIXE (`prov_*` / `span_*`) —
    # les deux envs vivent dans le même bloc YAML mais n'acceptent pas les
    # mêmes arguments, et PersonaChatStream lèverait sur l'intrus.
    p11_gen = {k: v for k, v in (((raw.get("p11") or {}).get("gen") or {})
                                 | ((raw.get("p12") or {}).get("gen") or {})
                                 ).items()
               if k.startswith(f"{cfg.p11_env}_")}
    if cfg.p11_env == "life":
        # la longueur de vie est un AXE DE CELLULE (elle entre dans le nom du
        # dossier), donc elle vient de la config du modèle, pas des kwargs.
        p11_gen["life_turns"] = int(cfg.life_turns)
        # même statut que prov_vals : le pool de valeurs est un axe de
        # cellule (suffixe -sv).
        p11_gen["life_vals"] = cfg.life_vals
    if cfg.p11_env == "prov":
        # même statut : le pool de valeurs est un axe de cellule (suffixe -sv).
        p11_gen["prov_vals"] = cfg.prov_vals

    def pk(split, **over):
        return {**persona_kwargs(raw, split, a.smoke, cond=cfg.cond),
                **(p11_gen if cfg.p11_env != "rule" else {}), **over}

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
    if cfg.p11_exam:
        rot = next((b.attn.rot for b in model.blocks
                    if getattr(b.attn, "rot", None) is not None), None)
        print(f"  PHASE 11 — examen `{p11_exam(cfg)}` (spec §2.5, registre "
              f"§3) | env `{cfg.p11_env}` | âge `{cfg.bank_rot}`"
              + (f" (A_ref {cfg.age_ref}, augmentation d'échelle ON ≤ "
                 f"×{cfg.age_aug_max:g})" if cfg.age_aug else "")
              + f" | tag `{cfg.tag_mode}` | index local `{cfg.loc_mode}`",
              flush=True)
        if rot is not None:
            print(f"    plans sur K' (APRÈS W_K', jamais avant) : "
                  f"âge {list(map(int, rot.age_idx))} | canal "
                  f"{list(map(int, rot.tag_idx))} | local "
                  f"{list(map(int, rot.loc_idx))} — paires de tête TRIÉES par "
                  f"fréquence RoPE, dérive de la requête sur toute la fenêtre "
                  f"{rot.drift:.4f} rad (garde {cfg.rot_drift_max}) : c'est "
                  f"l'appariement §2.5 « plans de métadonnées sur les dims "
                  f"quasi statiques ».", flush=True)
        if cfg.bank_rot == "age-bias":
            print("    bras `age-bias` : AUCUNE rotation — un biais scalaire "
                  "de récence b(a) = w·a (1 paramètre) sur le logit des "
                  "colonnes banque. Sa valeur finale EST une mesure.",
                  flush=True)
        if cfg.p11_env != "rule":
            print(f"    layout : la QUESTION est DANS le seg gradé (non "
                  f"supervisée) et le décodage part de [question | A_OPEN] — "
                  f"au train comme à l'éval. Sans elle, deux candidats en "
                  f"banque rendraient la tâche indécidable (ph.8). "
                  f"TOUS les résidents sont injectés : aucun privilège de "
                  f"sélection.", flush=True)
    if cfg.p12_exam:
        print(f"  PHASE 12 — examen `{cfg.p12_exam}` (spec §2.3 / registre §3 "
              f"S6-S8) | env `{cfg.p11_env}` | banque {cfg.max_mem} groupes × "
              f"{cfg.top_k} lignes", flush=True)
        print(f"    MAINTENANCE PROCÉDURALE (hors graphe, ZÉRO paramètre) : "
              f"signal `{cfg.retention}`, budget de propagation "
              f"{cfg.prop_budget}"
              + (f" (EMA β {cfg.ema_beta:g})" if cfg.retention == "attn-ema"
                 else f" (décroissance {cfg.actr_decay:g})"
                 if cfg.retention == "actr" else "")
              + ". Append en tête + décalage + chute au bord ; les lignes "
              "propagées GARDENT leur compteur de naissance (l'âge reste "
              "l'âge vrai). AUCUN RL, aucune politique apprise de write ou de "
              "rétention — seul le read kvproj apprend.", flush=True)
        if cfg.retention in ("attn-ema", "actr"):
            print("    CAUSALITÉ (leçon S2) : la masse d'attention du tour t "
                  "est déposée EN ATTENTE et n'entre dans les scores qu'au "
                  "write du tour t+1 — jamais avant. Le rejeu est "
                  "teacher-forcé, donc tout signal qui lirait le tour courant "
                  "serait une fuite, pas une lecture.", flush=True)
        if cfg.p11_env == "life":
            print(f"    VIES LONGUES : ≈ {cfg.life_turns} segs, "
                  f"≈ {2 + max(4, (cfg.life_turns - 8) // 4)} writes contre "
                  f"max_mem {cfg.max_mem} — l'ancre est posée tôt, la "
                  f"supersession tard, et trois strates sont gradées : "
                  f"cur_pre (l'ancienne valeur tant qu'elle est courante), "
                  f"cur_post (la nouvelle, PAR HORIZON dw = le temps "
                  f"d'adaptation) et hist (la ré-évocation historique — la "
                  f"sonde que FIFO et récence ratent par construction).",
                  flush=True)
        if cfg.bank_fill != "none":
            print(f"    DILUTION (S8) : la banque est remplie jusqu'à "
                  f"{cfg.max_mem} GROUPES avec des groupes d'AUTRES vies du "
                  f"même stream (réservoir {cfg.fill_pool} segs, lignes "
                  f"RECALCULÉES avec le modèle courant tous les "
                  f"{cfg.fill_refresh} appels — des lignes périmées feraient "
                  f"mesurer la dérive des embeddings). Les vrais résidents "
                  f"restent EN DERNIER, les remplisseurs portent slot/valeur 0 "
                  f"et ne peuvent jamais être la cible du r@1.", flush=True)
    if cfg.uses_fw:
        n_sub = cfg.max_mem * (cfg.group_rows if cfg.code in GROUP_CODES else 1)
        print(f"  boucle fast-weight : {n_sub} sous-slots × {cfg.n_layers} "
              f"couches, gradient "
              + ("ADDITIF (stop-grad sur l'état porté : chaque slot ne rend "
                 "que le gradient LOCAL de son propre upd ; le forward est "
                 "numériquement IDENTIQUE au bras multiplicatif)"
                 if cfg.fw_additive else
                 "MULTIPLICATIF (produit de jacobiennes sur les "
                 f"{n_sub * cfg.n_layers} étages) — ⚠️ MESURÉ DIVERGENT sur "
                 f"5 cellules /6 de la grille (gnorm pré-clip 1e4 → 4.7e9, "
                 f"Δnll de citation NÉGATIF) ; --fw-additive est le correctif"),
              flush=True)
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
        if ((step + 1) % eval_every == 0 or last) and cfg.p11_env != "rule":
            # ── PHASE 11 (S5/S17) : SÉLECTION et COPIE, séparément ─────────
            # `evaluate` (ph.9) ne s'applique pas ici : son décodage part de
            # A_OPEN seul et son injection est le groupe ORACLE, deux choses
            # que ces envs ne font pas (ils injectent TOUS les résidents et
            # posent la question dans le seg). Un bras au layout jamais
            # entraîné ne mesure rien (leçon KT2).
            pv = evaluate_p11(model, env, ev_stream, 1234, eval_convs, device,
                              tok, stop_id, max_new, max_len, amp)
            pt = evaluate_p11(model, env, tc_stream, 4321, eval_convs, device,
                              tok, stop_id, max_new, max_len, amp)
            print(f"  [eval {step+1}] P11/{cfg.p11_env} HELD-OUT grade "
                  f"{pv['grade_live']:.3f} (abl {pv['grade_abl']:.3f}) | "
                  f"r@1 attention {pv['r_at1']:.3f} (n={pv['n_sel']}) "
                  f"| TRAIN grade {pt['grade_live']:.3f} r@1 "
                  f"{pt['r_at1']:.3f} | n={pv['n']}", flush=True)
            print("    strates : " + "  ".join(
                f"{s} grade {d['grade']:.3f} r@1 {d['r_at1']:.3f} (n={d['n']})"
                for s, d in pv["strates"].items()), flush=True)
            if pv.get("life"):
                lm = pv["life"]
                print(f"    S6 : t_adapt {lm['t_adapt']} | survie hist "
                      f"{lm['surv_hist']:.3f} (pre {lm['surv_pre']:.3f}) | "
                      f"horizon p90 {lm['horizon_p90']} max "
                      f"{lm['horizon_max']} | résidence "
                      f"{lm['resident_rate']:.3f} | writes/vie "
                      f"{lm['n_writes_per_life']:.1f} prop "
                      f"{lm['n_prop_per_life']:.1f} chutes "
                      f"{lm['n_drop_per_life']:.1f}", flush=True)
            with open(csv_path, "a", newline="") as f:
                w = csv.writer(f)
                if new_csv:
                    w.writerow(["step", "loss", "grade_live", "grade_abl",
                                "r_at1", "n_sel", "n", "grade_train",
                                "r_at1_train", "sec"])
                    new_csv = False
                w.writerow([step + 1, f"{loss:.5f}", f"{pv['grade_live']:.4f}",
                            f"{pv['grade_abl']:.4f}", f"{pv['r_at1']:.4f}",
                            pv["n_sel"], pv["n"], f"{pt['grade_live']:.4f}",
                            f"{pt['r_at1']:.4f}", f"{time.time()-t0:.0f}"])
            if pv["grade_live"] > best:
                best = pv["grade_live"]
                torch.save({"step": step + 1, "model": model.state_dict(),
                            "cfg": cfg.__dict__, "grade": best},
                           os.path.join(save_dir, "best.pt"))
            continue
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
    fp11 = None
    if final_eval_convs > 0 and cfg.p11_env != "rule":
        fp11 = evaluate_p11(model, env, ev_stream, 1234, final_eval_convs,
                            device, tok, stop_id, max_new, max_len, amp)
        se = math.sqrt(max(fp11["grade_live"] * (1 - fp11["grade_live"]), 1e-9)
                       / max(fp11["n"], 1))
        print(f"  [final] P11/{cfg.p11_env} ({final_eval_convs} convs) grade "
              f"{fp11['grade_live']:.3f} ± {se:.3f} (SE) | abl "
              f"{fp11['grade_abl']:.3f} | r@1 attention {fp11['r_at1']:.3f} "
              f"(n={fp11['n_sel']}) | n={fp11['n']}", flush=True)
        print(f"  [final] ABSOLUS (LE JUGE, audit 08-03) — nll_live "
              f"{fp11['nll_live']:.4f} vs nll_abl {fp11['nll_abl']:.4f} "
              f"(n={fp11['n_nll']:.0f} tokens) | VALEUR seule : live "
              f"{fp11['nllval_live']:.4f} vs abl {fp11['nllval_abl']:.4f} "
              f"(n={fp11['n_nllval']:.0f}). Le Δ intra-modèle "
              f"({fp11['dnll_deprecated']:+.4f}) est rendu pour continuité, "
              f"PAS pour arbitrer : il ne dit rien du plancher.", flush=True)
        for s, d in fp11["strates"].items():
            print(f"    [final] strate {s} : grade {d['grade']:.3f} "
                  f"(abl {d['grade_abl']:.3f}) r@1 {d['r_at1']:.3f} "
                  f"n={d['n']}", flush=True)
        if fp11.get("life"):
            lm = fp11["life"]
            print(f"  [final] S6 — (a) t_adapt {lm['t_adapt']} writes "
                  f"(courbe " + " ".join(
                      f"dw{d}:{v['grade']:.2f}(n={v['n']})"
                      for d, v in lm["adapt_curve"].items())
                  + f") | (b) survie hist {lm['surv_hist']:.3f} "
                  f"[contrôle cur_pre {lm['surv_pre']:.3f}] | (d) horizon "
                  f"effectif p90 {lm['horizon_p90']} max {lm['horizon_max']} "
                  f"writes (n citées {lm['n_cited']}, max_mem "
                  f"{lm['max_mem']}) | résidence de la cible "
                  f"{lm['resident_rate']:.3f} | maintenance : "
                  f"{lm['n_writes_per_life']:.1f} writes, "
                  f"{lm['n_prop_per_life']:.1f} propagations, "
                  f"{lm['n_drop_per_life']:.1f} chutes par vie", flush=True)
            print("    LECTURE (S6) : un horizon p90 > max_mem PROUVE la "
                  "propagation (une ligne plus vieille que la profondeur de "
                  "la FIFO est encore citée) ; survie hist ≈ 0 avec "
                  "t_adapt = 0 = le bras n'a fait que suivre la récence.",
                  flush=True)
        fpp = os.path.join(save_dir, "p11_metrics.csv")
        with open(fpp, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["strate", "n", "grade", "grade_abl", "r_at1",
                        "nll_live", "nll_abl", "nllval_live", "nllval_abl"])
            w.writerow(["ALL", fp11["n"], f"{fp11['grade_live']:.5f}",
                        f"{fp11['grade_abl']:.5f}", f"{fp11['r_at1']:.5f}",
                        f"{fp11['nll_live']:.5f}", f"{fp11['nll_abl']:.5f}",
                        f"{fp11['nllval_live']:.5f}",
                        f"{fp11['nllval_abl']:.5f}"])
            for s, d in fp11["strates"].items():
                w.writerow([s, d["n"], f"{d['grade']:.5f}",
                            f"{d['grade_abl']:.5f}", f"{d['r_at1']:.5f}",
                            "", "", "", ""])
        print(f"  [final] écrit {fpp}", flush=True)
    elif final_eval_convs > 0:
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
        print(f"  [final] ABSOLUS (LE JUGE, audit 08-03) — nll du rendu "
              f"COHÉRENT : live {fc['nll_live']:.4f} | shuf "
              f"{fc['nll_shuf']:.4f} | none {fc['nll_none']:.4f} ; aux "
              f"MARQUEURS live {fc['nllmark_live']:.4f} | shuf "
              f"{fc['nllmark_shuf']:.4f} | none {fc['nllmark_none']:.4f}. "
              f"C'est `nll_live` APPARIÉ entre cellules qui arbitre — le Δnll "
              f"intra-modèle est DÉPRÉCIÉ (il ne dit rien du plancher).",
              flush=True)
        if cfg.bank_fill != "none":
            print(f"  [final] S8 — r@1 sous DILUTION (S = {cfg.max_mem} "
                  f"groupes, 1 vrai + {cfg.max_mem - 1} distracteurs réels) : "
                  f"{fc['r_at1']:.3f} (n={fc['n_sel']}, hasard "
                  f"{1.0 / max(cfg.max_mem, 1):.3f})", flush=True)
        print("    LECTURE DU VERDICT : live ≫ shuf ≈ none ⇒ le "
              "conditionnement passe PAR LA BANQUE (barreau 1 de l'échelle "
              "§2.4 franchi) ; live ≈ shuf ⇒ la sonde mesure un artefact ; "
              "live ≈ none ⇒ l'injection ne conditionne pas.", flush=True)
        fcp = os.path.join(save_dir, "cond_metrics.csv")
        with open(fcp, "w", newline="") as f:
            w = csv.writer(f)
            # `nll_*`/`nllmark_*` = LES ABSOLUS (le juge depuis l'audit du
            # 08-03) ; `dnll_*`/`mark_*` restent des contrastes.
            cols = [f"{p}_{c}" for p in ("mark", "mark_se", "mark_med",
                                         "dnll", "acc", "nll", "nllmark")
                    for c in ("live", "shuf", "none")] + ["n", "n_convs"]
            w.writerow(cols)
            w.writerow([f"{fc[c]:.5f}" if fc[c] == fc[c] else "" for c in
                        cols[:-2]] + [fc["n"], fc["n_convs"]])
        print(f"  [final] écrit {fcp}", flush=True)
    # ── S4 : LA COURBE D'OOD D'ÂGE ─────────────────────────────────────────
    # Le MÊME modèle, la MÊME éval, les MÊMES vies — seul le compteur de
    # writes est multiplié. C'est la définition opératoire de « vie longue » :
    # le fait est toujours en banque, son âge a explosé. Le bras qui tient
    # l'échelle 100 gagne S4.
    food = None
    if cfg.cond and final_eval_convs > 0 and cfg.bank_rot != "none":
        scales = parse_scales(cfg.age_eval_scales)
        food = {"1.0": {k: fc[k] for k in ("mark_live", "acc_live",
                                           "dnll_live")}}
        for sc in scales[1:]:
            fo = evaluate_cond(model, env, cv_stream, 2468, final_eval_convs,
                               device, max_len, amp, age_scale=sc)
            food[str(sc)] = {k: fo[k] for k in ("mark_live", "acc_live",
                                                "dnll_live")}
            print(f"  [final] OOD âge ×{sc:g} — Δnll MARQUEURS live "
                  f"{fo['mark_live']:+.4f} | 2AFC {fo['acc_live']:.3f} | "
                  f"Δnll tous tokens {fo['dnll_live']:+.4f}", flush=True)
        print(f"    LECTURE (S4) : le bras qui garde sa marge à ×"
              f"{scales[-1]:g} tient l'OOD ; la prédiction inscrite est que "
              f"`age-raw` s'effondre et que `age-log` tient (l'augmentation "
              f"d'échelle au train, si elle est ON, est le steelman du brut).",
              flush=True)
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
                  "fw_additive": bool(cfg.fw_additive),
                  "hid_tap_layers": cfg.hid_tap_layers,
                  "d_model": cfg.d_model, "n_layers": cfg.n_layers,
                  "max_mem": cfg.max_mem},
        # ── LE BIAIS DE LOGITS APPRIS (`kv_proj`) : sa valeur EST une
        # mesure. Positif ⇒ le modèle a dû REMONTER la banque contre le
        # contexte dans le softmax unifié (le déséquilibre de masse était
        # réel) ; ~0 ⇒ le softmax unifié n'avait pas besoin d'aide ; négatif
        # ⇒ il a dû la faire TAIRE.
        "bank_logit_bias": ([[round(float(v), 5) for v in
                              b.attn.bank_bias.detach().cpu()]
                             for b in model.blocks
                             if getattr(b.attn, "kvproj", False)]
                            if cfg.read_path == "kvproj" else None),
        # ── PHASE 11 : le combo de métadonnées, ses mesures, sa courbe OOD ─
        "p11": ({"exam": p11_exam(cfg), "env": cfg.p11_env,
                 "bank_rot": cfg.bank_rot, "age_aug": bool(cfg.age_aug),
                 "age_ref": cfg.age_ref, "age_planes": cfg.age_planes,
                 "tag_mode": cfg.tag_mode, "prov_vals": cfg.prov_vals,
                 "loc_mode": cfg.loc_mode,
                 "loc_planes": cfg.loc_planes,
                 "drift_rad": next((round(float(b.attn.rot.drift), 6)
                                    for b in model.blocks
                                    if getattr(b.attn, "rot", None) is not None),
                                   None),
                 # les FRÉQUENCES APPRISES : leur dérive depuis l'init
                 # géométrique est une mesure (le modèle a-t-il voulu d'autres
                 # échelles que celles qu'on lui a données ?)
                 "age_omega": [[round(float(v), 6) for v in
                                torch.exp(b.attn.rot.age_log_omega.detach()
                                          .cpu())]
                               for b in model.blocks
                               if getattr(b.attn, "rot", None) is not None
                               and hasattr(b.attn.rot, "age_log_omega")],
                 "age_bias_w": [round(float(b.attn.rot.age_bias_w.detach()
                                            .cpu()), 6)
                                for b in model.blocks
                                if getattr(b.attn, "rot", None) is not None
                                and hasattr(b.attn.rot, "age_bias_w")],
                 "selection_citation": ({"grade_live": fp11["grade_live"],
                                         "grade_abl": fp11["grade_abl"],
                                         "r_at1": fp11["r_at1"],
                                         # LES ABSOLUS (le juge, audit 08-03)
                                         "nll_live": fp11["nll_live"],
                                         "nll_abl": fp11["nll_abl"],
                                         "nllval_live": fp11["nllval_live"],
                                         "nllval_abl": fp11["nllval_abl"],
                                         "n": fp11["n"], "n_sel": fp11["n_sel"],
                                         "strates": fp11["strates"]}
                                        if fp11 else None),
                 "ood_age": food}
                if cfg.p11_exam else None),
        # ── PHASE 12 : le combo de MAINTENANCE / DILUTION et ses mesures ───
        # Le même schéma que `p11` : le combo en clair (agrégeable sans
        # grepper un log) puis les métriques de première classe de S6, et pour
        # S8 les deux moitiés (conditionnement et citation) sous le SEUL
        # cadran qui varie, S.
        "p12": ({"exam": cfg.p12_exam, "env": cfg.p11_env,
                 "retention": cfg.retention, "prop_budget": cfg.prop_budget,
                 "ema_beta": cfg.ema_beta, "actr_decay": cfg.actr_decay,
                 "life_turns": cfg.life_turns, "life_vals": cfg.life_vals,
                 "bank_fill": cfg.bank_fill,
                 "S": cfg.max_mem, "m": cfg.top_k,
                 "cond_decoys": cfg.cond_decoys,
                 "retention_metrics": (fp11.get("life") if fp11 else None),
                 "selection_citation": ({"grade_live": fp11["grade_live"],
                                         "grade_abl": fp11["grade_abl"],
                                         "r_at1": fp11["r_at1"],
                                         "nll_live": fp11["nll_live"],
                                         "nll_abl": fp11["nll_abl"],
                                         "nllval_live": fp11["nllval_live"],
                                         "nllval_abl": fp11["nllval_abl"],
                                         "n": fp11["n"],
                                         "n_sel": fp11["n_sel"],
                                         "strates": fp11["strates"]}
                                        if fp11 else None)}
                if cfg.p12_exam else None),
        "citation": ({"grade_live": fv["grade_live"],
                      "grade_abl": fv["grade_abl"], "dnll": fv["dnll"],
                      "grade_resident": fv["grade_resident"],
                      "n": fv["n"], "n_convs": final_eval_convs,
                      "strates": {g: fv[f"grade_{g}"] for g in GROUPS}}
                     if fv is not None else None),
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
    from .data.persona_chat_data import PersonaChatStream, _StubTok

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
    # 21d-bis. TÊTES DÉDIÉES (`read_path='dual'`) : masse softmax garantie.
    m_du = _mk_inj(read_path="dual")
    assert [b.bank_heads is not None for b in m_du.blocks] == [True, True]
    assert [b.bank_heads for b in m_noage.blocks] == [None, None], \
        "les autres chemins de lecture ne doivent PAS gagner de têtes"
    assert float(m_du.blocks[0].bank_heads.o.weight.abs().max()) == 0.0, \
        "la projection de SORTIE des têtes banque doit être zéro-init"
    with torch.no_grad():
        d_nu = m_du(ids10, None, None)
        d_inj0 = m_du(ids10, None, None, inject=rows10)
    # (a) BIT-À-BIT le backbone À L'INIT, banque présente OU non : la sortie
    #     des têtes est exactement nulle, donc le résiduel est intact.
    #     ⚠️ La référence ne peut PAS être un autre `_mk_inj` : construire les
    #     BankHeads consomme du RNG, donc un modèle `entry` de même graine n'a
    #     pas les mêmes poids à partir du bloc 1. La bonne référence est CE
    #     modèle, têtes banque RETIRÉES — c'est exactement la propriété qu'on
    #     veut : présent-mais-nul ≡ absent.
    import copy as _copy
    m_du_ref = _copy.deepcopy(m_du)
    for _b in m_du_ref.blocks:
        _b.bank_heads = None
    with torch.no_grad():
        assert torch.equal(d_inj0, m_du_ref(ids10, None, None, inject=rows10)), \
            ("read_path dual : à l'init, têtes banque PRÉSENTES ≡ têtes "
             "ABSENTES, bit-à-bit")
        assert torch.equal(d_nu, m_du_ref(ids10, None, None))
    assert torch.equal(d_inj0, d_nu), \
        "à l'init les têtes banque doivent rendre EXACTEMENT zéro"
    assert d_inj0.shape == d_nu.shape, "dual ne doit rendre AUCUN préfixe"
    # (b) LE PIÈGE r3 : une porte MULTIPLICATIVE zéro-init serait morte (le
    #     gradient de ses paramètres internes vaut g·(…) = 0). Ici c'est la
    #     SORTIE qui est nulle : W_o reçoit du gradient dès le premier
    #     backward, donc la tête est VIVANTE. On le mesure, on ne le suppose
    #     pas.
    m_du.zero_grad()
    m_du(ids10, None, None, inject=rows10).float().pow(2).mean().backward()
    g_o = m_du.blocks[0].bank_heads.o.weight.grad
    assert g_o is not None and float(g_o.abs().max()) > 0, \
        "W_o sans gradient au step 0 : la tête banque est MORTE (piège r3)"
    # (c) et elle apprend VRAIMENT : quelques pas suffisent à la décoller de
    #     zéro et à écarter le forward du backbone.
    m_tr = _mk_inj(read_path="dual")
    opt_du = torch.optim.SGD(m_tr.parameters(), lr=0.5)
    tgt = torch.tensor([[6, 7, 8]])
    for _ in range(5):
        opt_du.zero_grad(set_to_none=True)
        lg = m_tr(ids10, None, None, inject=rows10)
        F.cross_entropy(lg.reshape(-1, lg.size(-1)), tgt.reshape(-1)).backward()
        opt_du.step()
    assert float(m_tr.blocks[0].bank_heads.o.weight.abs().max()) > 0, \
        "W_o est resté à zéro après 5 pas : la tête n'apprend pas"
    with torch.no_grad():
        assert not torch.equal(m_tr(ids10, None, None, inject=rows10),
                               m_tr(ids10, None, None)), \
            "après entraînement la banque doit CHANGER le forward"
        # le CONTENU compte
        assert not torch.equal(m_tr(ids10, None, None, inject=rows10),
                               m_tr(ids10, None, None, inject=rows10 * 0.5))
        # (d) LA BANQUE EST UN ENSEMBLE, PAS UNE SÉQUENCE : ni RoPE ni masque
        #     causal ⇒ permuter les lignes ne doit RIEN changer. C'est
        #     l'invariant qui distingue `dual` de `inject_entry` (dont le
        #     préfixe est positionné) et la raison pour laquelle l'ancienneté
        #     doit passer par le code d'âge et par lui seul.
        perm = rows10[:, :, torch.randperm(rows10.shape[2])]
        assert torch.allclose(m_tr(ids10, None, None, inject=rows10),
                              m_tr(ids10, None, None, inject=perm), atol=1e-5)
        # et la position 0 du tour voit déjà la banque (aucun masque causal)
        assert not torch.equal(m_tr(ids10, None, None, inject=rows10)[:, 0],
                               m_tr(ids10, None, None)[:, 0])
    # (e) budget de couches apparié : hors couches lectrices, pas de têtes
    m_du1 = _mk_inj(read_path="dual", read_layers=[1])
    assert [b.bank_heads is not None for b in m_du1.blocks] == [False, True]
    # (f) NON-RÉGRESSION des trois reads existants : à `mem` absent, le bloc
    #     `dual` est le MÊME calcul que les autres — et `entry`/`kv` n'ont pas
    #     bougé d'un bit (cf. d_nu ci-dessus et le self-test 21d).
    with torch.no_grad():
        assert torch.equal(_mk_inj(read_path="kv")(ids10, None, None),
                           _mk_inj()(ids10, None, None))
    # (g) knobs des têtes REFUSÉS hors dual (pas de no-op silencieux)
    for bad in ({"bank_heads": 4}, {"bank_head_dim": 16}):
        try:
            ToyCfg(vocab_size=512, d_model=64, n_layers=2, n_heads=4,
                   mem_dim=64, max_seq_len=64, sif_a=A_SIF, top_k=3,
                   variant="r4", code="tophid", **bad)
        except AssertionError as e:
            assert "dual" in str(e), str(e)
        else:
            raise AssertionError(f"ToyCfg aurait dû refuser {bad} hors dual")

    # 21d-ter. `kv_proj` — PROJECTIONS DÉDIÉES, SOFTMAX UNIFIÉ (3ᵉ sommet).
    m_kp = _mk_inj(read_path="kvproj")
    m_kq = _mk_inj(read_path="kvproj", bank_q=True)
    assert all(b.attn.kvproj for b in m_kp.blocks)
    assert not any(getattr(b.attn, "kvproj", False) for b in m_noage.blocks), \
        "les autres chemins ne doivent pas gagner de projections de banque"
    assert float(m_kp.blocks[0].attn.bank_bias.abs().max()) == 0.0, \
        "le biais de logits de banque doit être init 0"
    assert not hasattr(m_kp.blocks[0].attn, "bq"), \
        "sans --bank-q, aucune projection de requête de banque"
    assert float(m_kq.blocks[0].attn.bo.weight.abs().max()) == 0.0, \
        "W_o' des lanes banque doit être zéro-init"
    with torch.no_grad():
        p_nu = m_kp(ids10, None, None)
        p_inj = m_kp(ids10, None, None, inject=rows10)
        p_half = m_kp(ids10, None, None, inject=rows10 * 0.5)
        q_inj = m_kq(ids10, None, None, inject=rows10)
    # (a) BIT-À-BIT le backbone quand il n'y a PAS de banque. ⚠️ La référence
    #     ne peut pas être un autre `_mk_inj` : construire bk/bv consomme du
    #     RNG. La bonne référence est CE modèle, machinerie de banque RETIRÉE —
    #     c'est la propriété qu'on veut : sans banque, elle n'intervient pas.
    import copy as _cp0
    m_kp_ref = _cp0.deepcopy(m_kp)
    for _b in m_kp_ref.blocks:
        _b.attn.kvproj = False
        del _b.attn.bk, _b.attn.bv
    with torch.no_grad():
        assert torch.equal(p_nu, m_kp_ref(ids10, None, None)), \
            "kvproj sans banque doit être le backbone nu, bit-à-bit"
    assert p_inj.shape == p_nu.shape, "kvproj ne doit rendre AUCUN préfixe"
    # (b) la banque compte, et son CONTENU aussi. ⚠️ Contrairement à
    #     `dual_heads`, ce bras n'est PAS neutre à l'init avec banque — W_k'/
    #     W_v' sont tirés normalement, exactement comme `kv_append` fait entrer
    #     ses lignes brutes. C'est voulu : les deux sommets du haut du carré
    #     doivent partir du même régime.
    assert not torch.equal(p_inj, p_nu) and not torch.equal(p_inj, p_half)
    assert not torch.equal(p_inj[:, 0], p_nu[:, 0]), \
        "la position 0 doit déjà voir la banque (softmax unifié, pas de masque)"
    # (c) LE BIAIS DE LOGITS mord, et il est DÉRIVABLE (sinon la mesure
    #     annoncée dans results.json serait un zéro décoratif).
    with torch.no_grad():
        m_kp.blocks[0].attn.bank_bias.fill_(6.0)
        p_bias = m_kp(ids10, None, None, inject=rows10)
        m_kp.blocks[0].attn.bank_bias.zero_()
    assert not torch.equal(p_bias, p_inj), "le biais par tête ne change rien"
    m_kp.zero_grad()
    m_kp(ids10, None, None, inject=rows10).float().pow(2).mean().backward()
    gb_ = m_kp.blocks[0].attn.bank_bias.grad
    assert gb_ is not None and float(gb_.abs().max()) > 0, \
        "bank_bias sans gradient : il n'apprendrait jamais"
    # (d) BANK-Q : à l'init, W_o' nul ⇒ les lanes valent les lignes brutes ⇒
    #     le bras dégénère EXACTEMENT en kvproj (poids partagés : m_kq a des
    #     modules en plus, on compare donc m_kq à lui-même, lanes débranchées).
    import copy as _cp
    m_kq_ref = _cp.deepcopy(m_kq)
    for _b in m_kq_ref.blocks:
        _b.bank_q = False
    with torch.no_grad():
        assert torch.equal(q_inj, m_kq_ref(ids10, None, None,
                                           inject=rows10)), \
            "bank_q à l'init doit être bit-à-bit le kvproj nu (W_o' = 0)"
    # et les lanes APPRENNENT : quelques pas décollent W_o' et écartent le
    # forward du bras sans portage.
    opt_kq = torch.optim.SGD(m_kq.parameters(), lr=0.5)
    for _ in range(5):
        opt_kq.zero_grad(set_to_none=True)
        lg = m_kq(ids10, None, None, inject=rows10)
        F.cross_entropy(lg.reshape(-1, lg.size(-1)),
                        torch.tensor([[6, 7, 8]]).reshape(-1)).backward()
        opt_kq.step()
    assert float(m_kq.blocks[0].attn.bo.weight.abs().max()) > 0, \
        "W_o' est resté nul : les lanes banque n'apprennent pas"
    # (e) LA BANQUE RESTE UN ENSEMBLE : ni RoPE ni ordre sur les lignes ⇒
    #     permuter les lignes ne change pas la sortie du TOUR.
    with torch.no_grad():
        pm = rows10[:, :, torch.randperm(rows10.shape[2])]
        assert torch.allclose(m_kq(ids10, None, None, inject=rows10),
                              m_kq(ids10, None, None, inject=pm), atol=1e-5)
    # (f) l'état des lanes est JETÉ en sortie : rien ne le réécrit dans la
    #     banque (invariant « seule modif de la banque = l'append d'un write »).
    assert m_kq(ids10, None, None, inject=rows10).shape == p_nu.shape
    # (g) budget de couches apparié + refus du no-op
    m_kp1 = _mk_inj(read_path="kvproj", read_layers=[1])
    assert [b.read_bank for b in m_kp1.blocks] == [False, True]
    for bad in ({"read_path": "kv"}, {"read_path": "dual"},
                {"read_path": "entry"}):
        try:
            ToyCfg(vocab_size=512, d_model=64, n_layers=2, n_heads=4,
                   mem_dim=64, max_seq_len=64, sif_a=A_SIF, top_k=3,
                   variant="r4", code="tophid", bank_q=True, **bad)
        except AssertionError as e:
            assert "bank_q" in str(e), str(e)
        else:
            raise AssertionError(f"bank_q aurait dû être refusé avec {bad}")

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
    # ── 21g-bis. CORRECTIF TERRAIN : gradient ADDITIF de la boucle seq_fw ───
    # Le bras multiplicatif DIVERGE sur la ferme (5 cellules /6, gnorm pré-clip
    # jusqu'à 4.7e9). Le fix est un stop-gradient sur l'état porté. Ce qu'il
    # faut prouver ici : (a) le forward ne bouge pas D'UN BIT, (b) le backward,
    # lui, bouge, (c) sous ON les termes produits ont disparu, (d) le flag est
    # refusé là où il n'y a pas de boucle.
    _base10 = dict(vocab_size=512, d_model=64, n_layers=2, n_heads=4,
                   mem_dim=64, max_seq_len=64, sif_a=A_SIF, seg_n_pos=8,
                   inject_sep_id=5, max_mem=8)

    def _mk_fw(additive, M=8, top_k=3, seed=31337):
        torch.manual_seed(seed)
        c = ToyCfg(vocab_size=512, d_model=64, n_layers=2, n_heads=4,
                   mem_dim=64, variant="r0", max_seq_len=64, code="tophid",
                   top_k=top_k, seg_n_pos=8, sif_a=A_SIF, max_mem=M,
                   fw_additive=additive)
        return ToyReadLM(c, env.n_slots, env.n_attrs, sif_w=_sifw(512))

    def _fw_grad(model, M_lines, scale=1.0, seed=99):
        """(sortie, gnorm total) sur une banque tirée à `scale` donnée."""
        torch.manual_seed(seed)
        bk = torch.randn(2, M_lines, 64) * scale
        bm = torch.ones(2, M_lines, dtype=torch.bool)
        ids = (torch.arange(12).reshape(2, 6) * 41) % 512
        model.zero_grad(set_to_none=True)
        out = model(ids, bk, bm)
        out.float().pow(2).mean().backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1e30)
        return out.detach(), float(gn)

    # (a) FORWARD IDENTIQUE BIT-À-BIT — c'est ce qui rend les deux bras
    #     comparables à poids égaux (seul le graphe diffère).
    m_off, m_on = _mk_fw(False), _mk_fw(True)
    assert m_off.blocks[0].read.additive is False
    assert m_on.blocks[0].read.additive is True
    o_off, g_off = _fw_grad(m_off, 8)
    o_on, g_on = _fw_grad(m_on, 8)
    assert torch.equal(o_off, o_on), (
        "le forward DOIT être bit-à-bit identique : y0 + (y − y0).detach() "
        "vaut y en VALEUR, seul le graphe change")
    # (b) les GRADIENTS diffèrent (sinon le fix ne ferait rien)
    d_max = max(float((p.grad - q.grad).abs().max())
                for (_, p), (_, q) in zip(m_off.named_parameters(),
                                          m_on.named_parameters())
                if p.grad is not None and q.grad is not None)
    assert d_max > 0.0, "gradients identiques : le stop-grad n'a pas mordu"
    # (c) LES TERMES PRODUITS ONT DISPARU. Sur un M GRAND et une banque à forte
    #     échelle, le bras multiplicatif compose M jacobiennes et sa gnorm
    #     explose de plusieurs ordres ; l'additif reste borné. On exige un
    #     rapport franc, pas un epsilon.
    m_off_big, m_on_big = _mk_fw(False, M=64), _mk_fw(True, M=64)
    _, gb_off = _fw_grad(m_off_big, 64, scale=6.0)
    _, gb_on = _fw_grad(m_on_big, 64, scale=6.0)
    assert gb_on < gb_off / 100.0, (
        f"le gradient additif n'a pas amorti la boucle : additif {gb_on:.3e} "
        f"contre multiplicatif {gb_off:.3e} (rapport "
        f"{gb_off / max(gb_on, 1e-30):.1f}×)")
    assert gb_on == gb_on and gb_on < 1e6, f"gnorm additive non bornée {gb_on}"
    # et la MONOTONIE du fléau : à M=64 le multiplicatif est bien pire qu'à
    # M=8, alors que l'additif ne se dégrade pas dans les mêmes proportions.
    _, g8_off = _fw_grad(_mk_fw(False, M=8), 8, scale=6.0)
    _, g8_on = _fw_grad(_mk_fw(True, M=8), 8, scale=6.0)
    assert gb_off / max(g8_off, 1e-30) > gb_on / max(g8_on, 1e-30), (
        "la gnorm multiplicative doit se dégrader AVEC M plus vite que "
        f"l'additive (mult {g8_off:.2e}→{gb_off:.2e}, add "
        f"{g8_on:.2e}→{gb_on:.2e})")
    # (e) refusé là où il n'y a PAS de boucle (bras attention)
    for var, code in (("r1", "mean"), ("r3", "tophid"), ("r4", "tophid")):
        try:
            ToyCfg(vocab_size=512, d_model=64, n_layers=2, n_heads=4,
                   mem_dim=64, max_seq_len=64, sif_a=A_SIF, top_k=3,
                   variant=var, code=code, fw_additive=True)
        except AssertionError as e:
            assert "fw_additive" in str(e), str(e)
        else:
            raise AssertionError(
                f"fw_additive aurait dû être refusé pour {var}/{code}")
    # NOMMAGE : la cellule additive ne peut pas écraser la multiplicative
    assert grid_name(_mk_fw(True).cfg) != grid_name(_mk_fw(False).cfg)
    assert grid_name(_mk_fw(True).cfg).endswith("_fwadd"), \
        grid_name(_mk_fw(True).cfg)
    # le SOUS-ENSEMBLE de manifeste : 12 cellules, toutes additives, 12 noms
    _add = grid_combos(**GRID_SUBSETS["seqfw-additive"])
    assert len(_add) == 12 and all(c["fw_additive"] for c in _add)
    _addn = [grid_name(_grid_cfg(c, _base10)) for c in _add]
    assert len(set(_addn)) == 12 and all(n.endswith("_fwadd") for n in _addn)
    assert not (set(_addn) & {grid_name(_grid_cfg(c, _base10))
                              for c in grid_combos()}), \
        "une cellule additive écraserait une cellule multiplicative"
    # SOUS-ENSEMBLES `kvproj` / `kvproj-bq` : 12 + 12, tous distincts et sans
    # collision avec quoi que ce soit de déjà lancé. Le `_bq` doit séparer.
    _kp = [grid_name(_grid_cfg(c, _base10))
           for c in grid_combos(**GRID_SUBSETS["kvproj"])]
    _kq = [grid_name(_grid_cfg(c, _base10))
           for c in grid_combos(**GRID_SUBSETS["kvproj-bq"])]
    assert len(set(_kp)) == 12 and len(set(_kq)) == 12
    assert not (set(_kp) & set(_kq)), "bank_q ne sépare pas les dossiers"
    assert all(n.endswith("_bq") for n in _kq), _kq
    assert all("kvproj" in n for n in _kp), _kp
    # bank_q ne DÉBORDE pas sur les autres bras
    assert not any(c["bank_q"] for c in grid_combos(bank_q=True)
                   if c["read"] != "kv_proj")
    # SOUS-ENSEMBLE `dual` : 12 cellules, 12 noms, aucune collision avec les
    # 36 de la grille lancée ni avec les 12 additives.
    _dl = grid_combos(**GRID_SUBSETS["dual"])
    assert len(_dl) == 12, len(_dl)
    _dln = [grid_name(_grid_cfg(c, _base10)) for c in _dl]
    assert len(set(_dln)) == 12 and all("dualheads" in n for n in _dln), _dln
    _lance = {grid_name(_grid_cfg(c, _base10)) for c in grid_combos()}
    assert not (set(_dln) & (_lance | set(_addn))), \
        "une cellule dual écraserait une cellule déjà lancée"
    assert not ((set(_kp) | set(_kq)) & (_lance | set(_addn) | set(_dln))), \
        "une cellule kvproj écraserait une cellule déjà lancée"
    # et l'additif ne DÉBORDE PAS sur les bras attention
    assert not any(c["fw_additive"] for c in
                   grid_combos(fw_additive=True) if c["read"] != "seq_fw")

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
    _cmb = grid_combos()
    assert len(_cmb) == 36, len(_cmb)
    _gn = [grid_name(_grid_cfg(cc, _base10)) for cc in _cmb]
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

    # ══ 22. PHASE 11 — LES TROIS FAMILLES DE MÉTADONNÉES (spec §2.5) ═══════
    # Dimensions du bloc : d_model 64 / 2 TÊTES (⇒ 32 dims de tête, 16 paires
    # RoPE). Pourquoi 2 et pas 4 : à 8 paires la bande quasi statique ne loge
    # pas 4 plans (dérive 3,19 rad sur T=64) et le garde-fou refuserait — À
    # RAISON. On élargit la tête plutôt que d'assouplir l'invariant.
    def _mk11(**kw):
        torch.manual_seed(90210)
        c = ToyCfg(vocab_size=512, d_model=64, n_layers=2, n_heads=2,
                   mem_dim=64, variant="r4", max_seq_len=64, code="tophid",
                   top_k=3, seg_n_pos=8, sif_a=A_SIF, inject_sep_id=5,
                   max_mem=8, read_path="kvproj",
                   **{"p11_exam": "age", **kw})
        return ToyReadLM(c, env.n_slots, env.n_attrs, sif_w=_sifw(512)).eval()

    ids11 = torch.tensor([[3, 4, 5]])
    rows11 = torch.randn(1, 2, 3, 64)
    age11 = torch.tensor([[0, 1]])
    chan11 = torch.tensor([[0, 1]])
    m11_off = _mk11()
    with torch.no_grad():
        ref11 = m11_off(ids11, None, None, inject=rows11, inject_age=age11,
                        inject_chan=chan11)
    assert getattr(m11_off.blocks[0].attn, "rot", None) is None, \
        "aucune famille active ⇒ AUCUN module de rotation ne doit exister"
    # (a) APPARIEMENT §2.5 : les plans sont les paires les PLUS LENTES du RoPE
    #     de tête, et la requête ne tourne quasiment pas dessus sur toute la
    #     fenêtre. C'est LA condition qui rend le codage lisible côté banque.
    _dh, _th, _T = 32, 10000.0, 64
    idx11, drift11 = slow_rope_planes(_dh, _th, 4, _T)
    _inv = 1.0 / (_th ** (torch.arange(0, _dh, 2).float() / _dh))
    assert set(int(i) for i in idx11) == set(
        int(i) for i in torch.argsort(_inv)[:4]), \
        "les plans ne sont pas les paires les plus LENTES"
    assert drift11 < 0.35, f"dérive de requête trop grande : {drift11}"
    # le garde-fou MORD : demander une bande trop large ÉCHOUE à la
    # construction plutôt que de coder l'âge dans la bande rapide.
    try:
        _mk11(bank_rot="age-log", age_planes=12)
    except AssertionError as e:
        assert "APPARIEMENT REFUSÉ" in str(e), str(e)
    else:
        raise AssertionError("12 plans auraient dû être refusés (§2.5)")
    # (b) ÂGE. Rotation ⇒ le forward bouge ET l'âge COMPTE ; le bras `none`
    #     est le contrôle θ_âge=0 exact.
    m11_log, m11_raw = _mk11(bank_rot="age-log"), _mk11(bank_rot="age-raw")
    with torch.no_grad():
        o_log = m11_log(ids11, None, None, inject=rows11, inject_age=age11,
                        inject_chan=chan11)
        o_log2 = m11_log(ids11, None, None, inject=rows11,
                         inject_age=age11 * 10, inject_chan=chan11)
        o_zero = m11_log(ids11, None, None, inject=rows11,
                         inject_age=torch.zeros_like(age11),
                         inject_chan=chan11)
    assert not torch.equal(ref11, o_log), "la rotation d'âge ne mord pas"
    assert not torch.equal(o_log, o_log2), "l'ÂGE ne change rien : le code est mort"
    assert torch.allclose(ref11, o_zero, atol=1e-6), (
        "âge 0 partout ⇒ rot(0) = IDENTITÉ ⇒ le forward doit retomber sur le "
        "kvproj nu")
    # log vs brut : ILS COÏNCIDENT dans la plage vue au train (φ(A_ref)=A_ref)
    # et DIVERGENT en OOD — c'est ce qui rend S4 lisible.
    _rot = m11_log.blocks[0].attn.rot
    _a = torch.tensor([[0.0, 1.0, 8.0, 800.0]])
    _phi = _rot.phi_age(_a)
    assert abs(float(_phi[0, 2]) - 8.0) < 1e-4, float(_phi[0, 2])
    assert float(_phi[0, 3]) < 0.1 * 800.0, (
        f"la compression log doit écraser un âge de 800 writes (A_ref 8 ⇒ "
        f"φ ≈ 24, soit 3× l'horizon normal pour 100× le compteur) : "
        f"{float(_phi[0, 3]):.1f}")
    assert float(m11_raw.blocks[0].attn.rot.phi_age(_a)[0, 3]) == 800.0
    # fréquences APPRISES : géométriques à l'init, et dérivables.
    _om = torch.exp(_rot.age_log_omega.detach())
    assert _om.numel() == 4 and float(_om[0]) > float(_om[-1]), _om
    o = m11_log(ids11, None, None, inject=rows11, inject_age=age11)
    o.float().pow(2).mean().backward()
    assert float(m11_log.blocks[0].attn.rot.age_log_omega.grad.abs().max()) > 0, \
        "les fréquences d'âge ne reçoivent AUCUN gradient"
    # (c) LE FALLBACK `age-bias` : init 0 ⇒ bit-à-bit le kvproj nu, et il
    #     MORD + reçoit du gradient dès le premier backward (un terme additif
    #     au logit, jamais une porte multiplicative).
    m11_b = _mk11(bank_rot="age-bias")
    with torch.no_grad():
        assert torch.equal(ref11, m11_b(ids11, None, None, inject=rows11,
                                        inject_age=age11, inject_chan=chan11))
        m11_b.blocks[0].attn.rot.age_bias_w.fill_(1.0)
        assert not torch.equal(ref11, m11_b(ids11, None, None, inject=rows11,
                                            inject_age=age11)), \
            "le biais de récence ne mord pas sur la sortie"
        m11_b.blocks[0].attn.rot.age_bias_w.zero_()
    o = m11_b(ids11, None, None, inject=rows11, inject_age=age11)
    o.float().pow(2).mean().backward()
    assert float(m11_b.blocks[0].attn.rot.age_bias_w.grad.abs().max()) > 0, \
        "le biais de récence est un paramètre mort"
    assert sum(p.numel() for n, p in m11_b.named_parameters()
               if "age_bias_w" in n) == m11_b.cfg.n_layers, \
        "le biais de récence doit coûter UN paramètre par couche lectrice"
    # (d) TAG DE PROVENANCE. Rotation : un plan PAR CANAL, angle 0 ou π ⇒
    #     R(π)² = I (le code n'est pas orienté, assumé). Additif : MÊMES dims
    #     réservées, init 0 ⇒ démarre exactement sur le kvproj nu, et il
    #     apprend (le gradient d'un terme additif n'est pas nul à zéro).
    m11_tr = _mk11(tag_mode="rot", p11_env="prov")
    m11_ta = _mk11(tag_mode="add", p11_env="prov")
    with torch.no_grad():
        t_a = m11_tr(ids11, None, None, inject=rows11, inject_chan=chan11)
        t_b = m11_tr(ids11, None, None, inject=rows11,
                     inject_chan=1 - chan11)
        assert not torch.equal(t_a, t_b), "le canal ne change rien au forward"
        assert torch.equal(ref11, m11_ta(ids11, None, None, inject=rows11,
                                         inject_chan=chan11)), \
            "le steelman ADDITIF doit démarrer bit-à-bit sur le kvproj nu"
    assert m11_tr.blocks[0].attn.rot.tag_idx.numel() == \
        m11_ta.blocks[0].attn.rot.tag_idx.numel() == 2, \
        "rot et add doivent réserver le MÊME nombre de plans (budget apparié)"
    o = m11_ta(ids11, None, None, inject=rows11, inject_chan=chan11)
    o.float().pow(2).mean().backward()
    assert float(m11_ta.blocks[0].attn.rot.tag_add.grad.abs().max()) > 0, \
        "le vecteur de tag additif ne reçoit aucun gradient"
    # R(π) sur un plan : appliquer DEUX fois le même canal revient à l'identité
    _r = m11_tr.blocks[0].attn.rot
    _km = torch.randn(1, 2, 2, 32)
    _meta = torch.zeros(1, 2, 3, dtype=torch.long)
    _meta[..., 1] = 1
    assert torch.allclose(_r(_r(_km, _meta), _meta), _km, atol=1e-5), \
        "R(π)² doit valoir l'identité (un plan 0/π par canal)"
    # (e) INDEX LOCAL. rot : l'OPÉRATEUR SUCCESSEUR est CONSTANT — passer de
    #     j à j+1 est la MÊME rotation quel que soit j et quel que soit le
    #     contenu. C'est l'argument structurel contre l'additif (§2.5).
    m11_lr = _mk11(loc_mode="rot")
    _r = m11_lr.blocks[0].attn.rot
    _k0 = torch.randn(1, 2, 1, 32)
    def _at(j):
        mm = torch.zeros(1, 1, 3, dtype=torch.long)
        mm[..., 2] = j
        return _r(_k0, mm)
    _d = [(_at(j + 1) - _at(j)) for j in range(3)]
    # l'écart entre deux positions successives n'est PAS constant en valeur
    # (c'est une rotation, pas une translation) : ce qui est constant, c'est
    # l'OPÉRATEUR. On le vérifie en angle sur le plan local.
    _pl = int(_r.loc_idx[0])
    def _ang(x, j):
        z = x[0, 0, 0, 2 * _pl:2 * _pl + 2]
        return math.atan2(float(z[1]), float(z[0]))
    _da = [(_ang(_at(j + 1), j + 1) - _ang(_at(j), j)) % (2 * math.pi)
           for j in range(3)]
    assert max(_da) - min(_da) < 1e-4, (
        f"R_loc(1) doit être un opérateur CONSTANT : {_da}")
    m11_la = _mk11(loc_mode="add")
    with torch.no_grad():
        assert torch.equal(ref11, m11_la(ids11, None, None, inject=rows11)), \
            "l'index local ADDITIF doit démarrer bit-à-bit sur le kvproj nu"
        assert not torch.equal(ref11, m11_lr(ids11, None, None,
                                             inject=rows11)), \
            "l'index local rotatif ne mord pas"
    o = m11_la(ids11, None, None, inject=rows11)
    o.float().pow(2).mean().backward()
    assert float(m11_la.blocks[0].attn.rot.loc_add.grad.abs().max()) > 0
    # (f) LES TROIS FAMILLES SONT DISJOINTES (aucun plan partagé).
    m11_all = _mk11(bank_rot="age-log", tag_mode="rot", loc_mode="rot",
                    p11_env="prov", age_planes=2, loc_planes=2,
                    rot_drift_max=1.0)
    _r = m11_all.blocks[0].attn.rot
    _sets = [set(map(int, _r.age_idx)), set(map(int, _r.tag_idx)),
             set(map(int, _r.loc_idx))]
    assert len(set().union(*_sets)) == sum(len(s) for s in _sets), \
        "les familles se marchent dessus : les plans doivent être DISJOINTS"
    # (g) LA SONDE d'attention : elle N'ALTÈRE PAS le forward, et sa masse est
    #     une distribution sur les lignes de banque.
    with torch.no_grad():
        with bank_attn_probe(m11_log) as _p:
            _o1 = m11_log(ids11, None, None, inject=rows11, inject_age=age11,
                          inject_chan=chan11)
            _mass = _p.mass()
        _o2 = m11_log(ids11, None, None, inject=rows11, inject_age=age11,
                      inject_chan=chan11)
    assert torch.equal(_o1, _o2), "la sonde a changé le forward"
    assert _mass.shape == (1, 6) and float(_mass.sum()) <= 1.0 + 1e-4
    assert all(not b.attn.want_bank_attn for b in m11_log.blocks), \
        "la sonde doit s'éteindre à la sortie du contexte"
    # (h) LES GARDE-FOUS : chaque famille refuse ce qui serait un no-op.
    for bad in ({"bank_rot": "age-log", "read_path": "kv"},
                {"bank_rot": "age-log", "read_path": "entry"},
                {"tag_mode": "rot"},                      # env `rule`
                {"loc_mode": "rot", "top_k": 1},
                {"age_aug": True},                        # sans rotation d'âge
                {"age_aug": True, "bank_rot": "age-bias"},
                {"p11_env": "prov", "cond": True}):
        try:
            ToyCfg(vocab_size=512, d_model=64, n_layers=2, n_heads=2,
                   mem_dim=64, max_seq_len=64, sif_a=A_SIF,
                   **{"variant": "r4", "code": "tophid", "top_k": 3,
                      "read_path": "kvproj", "p11_exam": "age", **bad})
        except AssertionError:
            pass
        else:
            raise AssertionError(f"ToyCfg aurait dû refuser {bad}")
    # et l'EXAMEN NON DÉCLARÉ : une cellule ph.11 sans `p11_exam` retomberait
    # sur le nommage de la grille §2.4 — refusé à la construction.
    try:
        ToyCfg(vocab_size=512, d_model=64, n_layers=2, n_heads=2, mem_dim=64,
               max_seq_len=64, sif_a=A_SIF, variant="r4", code="tophid",
               top_k=3, read_path="kvproj", bank_rot="age-log")
    except AssertionError as e:
        assert "DÉCLARE son examen" in str(e), str(e)
    else:
        raise AssertionError("une cellule ph.11 sans examen déclaré est passée")
    # (i) LES ENVS. `prov` : deux writes du MÊME slot et du MÊME attribut (donc
    #     de clé oracle IDENTIQUE), un par canal, ordre tiré au sort ; la
    #     question NOMME le canal et vit DANS le seg gradé.
    ps = PersonaProvStream(tok, seed=11, p_smalltalk=0.0)
    pconv = ps.next_conv()
    assert pconv["kind"] == "prov"
    _w = [s for s in pconv["segs"] if OracleEnv.fact_of(s) is not None]
    assert len(_w) == 2 and {s["chan"] for s in _w} == {0, 1}, \
        "la vie `prov` doit écrire UN fait par canal"
    assert len({OracleEnv.fact_of(s)[:2] for s in _w}) == 1, (
        "les deux faits doivent partager slot ET attribut — sinon la CLÉ les "
        "sépare et le canal n'a plus rien à faire")
    _t = pconv["segs"][pconv["info"]["p11"]["turns"][0]]
    assert _t["q_len"] > 0 and float(_t["loss_mask"][0, :_t["q_len"]].sum()) == 0, \
        "la question doit être DANS le seg et NON supervisée"
    assert float(_t["loss_mask"][0, _t["q_len"]:].sum()) > 0
    # `span` : longueurs MESURÉES, jamais décrétées.
    ss = PersonaSpanStream(tok, seed=12, p_smalltalk=0.0)
    assert len(ss.span_pool) >= 2, ss.span_pool
    for L, vals in ss.span_pool.items():
        assert all(len(tok(" " + v, add_special_tokens=False)["input_ids"]) == L
                   for v in vals), f"bucket {L} mal mesuré"
    sconv = ss.next_conv()
    assert sconv["kind"] == "span" and len(sconv["info"]["truths"]) == 1
    assert sconv["info"]["p11"]["strate"][0].startswith("L")
    # (j) LE PLAN ph.11 : TOUS les résidents injectés, âges = rangs de récence,
    #     canaux portés, et la CIBLE du r@1 est bien le groupe du bon canal.
    m11_p = _mk11(tag_mode="rot", p11_env="prov")
    pl = env.p11_plan(m11_p, pconv)
    assert pl, "aucun tour gradé n'a de plan"
    _i, (_rows, _ages, _chans, _slots) = next(iter(pl.items()))
    assert _rows.shape[0] == 2 and set(map(int, _chans)) == {0, 1}
    assert sorted(map(int, _ages)) == [0, 1], "les âges = rangs de récence"
    _q0 = pconv["info"]["p11"]["chan"][0]
    _tgt = env.p11_target(pconv, _i, 0, (_rows, _ages, _chans, _slots))
    assert _tgt is not None and int(_chans[_tgt]) == _q0, (
        "la cible du r@1 doit être le groupe du canal que la question nomme")
    assert env.p11_plan(m11_p, rconv) == {}, \
        "p11_plan doit être un no-op hors des envs de la phase 11"
    # (k) AUGMENTATION D'ÉCHELLE : l'ORDRE est préservé, l'échelle varie, et
    #     elle est un NO-OP quand le flag est OFF (rétro-compat).
    _cfa = _mk11(bank_rot="age-raw", age_aug=True).cfg
    torch.manual_seed(0)
    _ag = torch.tensor([[0, 1, 2], [0, 3, 7]])
    _au = age_augment(_cfa, _ag)
    assert torch.equal(age_augment(_mk11().cfg, _ag), _ag), \
        "age_aug OFF doit être un no-op EXACT"
    for r in range(2):
        _o = [int(x) for x in _au[r]]
        assert _o == sorted(_o), f"l'ordre des âges doit être préservé : {_o}"
    assert float(_au.max()) >= float(_ag.max()), _au
    # (l) NOMMAGE : 22 cellules, 22 dossiers, AUCUNE collision avec les 96
    #     déjà lancées (36 grille + 12 _fwadd + 12 dual + 12 kvproj + 12 _bq +
    #     12 bank-q en file).
    _p11n, _p11c = [], 0
    for _ex in P11_EXAMS:
        _cc = p11_combos(_ex)
        _p11c += len(_cc)
        _p11n += [p11_name(_p11_cfg(c, _base10)) for c in _cc]
    assert _p11c == 22 and len(set(_p11n)) == 22, (len(_p11n), sorted(_p11n))
    _lancees = set()
    for _sub in GRID_SUBSETS.values():
        _lancees |= {grid_name(_grid_cfg(c, _base10))
                     for c in grid_combos(**_sub)}
    assert not (set(_p11n) & _lancees), (
        "une cellule ph.11 écraserait un run de la grille §2.4")
    assert all(n.startswith("p11-") for n in _p11n), _p11n
    # et le nom se relit : run_name_for == p11_name, et il PORTE l'examen.
    for _ex in P11_EXAMS:
        for c in p11_combos(_ex):
            _cf = _p11_cfg(c, _base10)
            assert run_name_for(_cf) == p11_name(_cf) and \
                p11_name(_cf).startswith(f"p11-{_ex}_") and \
                p11_name(_cf).endswith(f"_m{c['m']}"), p11_name(_cf)
    # LE CONTRÔLE DE S3 EST BIEN DANS L'ESPACE DE NOMS ph.11 : sans le champ
    # déclaré il retomberait sur `read-kvproj_rot-off_tap-postnorm_m4`,
    # c'est-à-dire sur une cellule DÉJÀ LANCÉE du carré factoriel.
    _ctl = _p11_cfg({"exam": "age", "env": "rule", "m": 4,
                     "bank_rot": "none"}, _base10)
    assert run_name_for(_ctl) == "p11-age_agezero_m4", run_name_for(_ctl)
    assert not _ctl.uses_p11_meta, (
        "le contrôle θ_âge=0 ne doit activer AUCUNE métadonnée — c'est sa "
        "définition (HoPE)")

    # ══ 23. PHASE 12 — MAINTENANCE PROCÉDURALE (S6) ET DILUTION (S8) ══════
    def _mk12(**kw):
        torch.manual_seed(90210)
        c = ToyCfg(vocab_size=512, d_model=64, n_layers=2, n_heads=2,
                   mem_dim=64, variant="r4", max_seq_len=128, code="tophid",
                   seg_n_pos=8, sif_a=A_SIF, inject_sep_id=5,
                   **{"p12_exam": "retention", "top_k": 3,
                      "read_path": "kvproj", **kw})
        return ToyReadLM(c, env.n_slots, env.n_attrs, sif_w=_sifw(512)).eval()

    def _row(i, d=64):
        torch.manual_seed(1000 + i)
        return torch.randn(3, d)

    def _vec(r):
        v = r.reshape(-1, r.shape[-1]).mean(0)
        return v / v.norm().clamp_min(1e-6)

    def _fill_store(sig, prop, n=12, mm=8, **kw):
        st = RetentionStore(mm, sig, prop, **kw)
        for i in range(n):
            r = _row(i)
            st.write(r, _vec(r), slot=i + 1, val=i + 1)
        return st

    # (a) LA FIFO NUE EST BIEN LA FIFO NUE. `fifo` à budget 0 doit reproduire
    #     EXACTEMENT le `fifo[-max_mem:]` des phases 6-11 : sans ça, la
    #     baseline basse du bakeoff ne serait pas la baseline historique et
    #     tout l'examen dériverait d'un cran.
    s_fifo = _fill_store("fifo", 0, n=12)
    assert [e["val"] for e in s_fifo.rows] == list(range(5, 13)), \
        [e["val"] for e in s_fifo.rows]
    assert s_fifo.n_prop == 0 and s_fifo.n_drop == 4
    _, _ag, _, _, _vl = s_fifo.view()
    assert list(map(int, _ag)) == [7, 6, 5, 4, 3, 2, 1, 0], list(map(int, _ag))
    # le budget est REFUSÉ sur `fifo` (et le signal sans budget aussi) : les
    # deux seraient des no-op silencieux.
    for bad in ({"retention": "fifo", "prop_budget": 1},
                {"retention": "age", "prop_budget": 0},
                {"retention": "age", "prop_budget": 8},
                {"retention": "age", "prop_budget": 1, "p11_env": "span"},
                {"bank_fill": "foreign", "p12_exam": ""},
                {"retention": "age", "prop_budget": 1, "p12_exam": ""},
                {"retention": "age", "prop_budget": 1, "p11_exam": "age"}):
        try:
            _mk12(**{"p11_env": "life", "p12_exam": "retention", **bad})
        except AssertionError:
            pass
        else:
            raise AssertionError(f"ToyCfg aurait dû refuser {bad}")

    # (b) LA PROPAGATION PRÉSERVE LA NAISSANCE — l'âge reste l'âge VRAI, pas
    #     la position. C'est l'invariant qui rend la rotation d'âge de BankRot
    #     cohérente avec la maintenance, et c'est aussi ce qui distingue une
    #     propagation d'une ré-écriture.
    s_age = _fill_store("age", 1, n=12)
    vals_age = [e["val"] for e in s_age.rows]
    assert 1 in vals_age, (
        "le signal `age` doit avoir SAUVÉ la plus vieille ligne du bord : "
        f"{vals_age}")
    e0 = next(e for e in s_age.rows if e["val"] == 1)
    assert e0["birth"] == 0, e0["birth"]
    _, ag2, _, _, vl2 = s_age.view()
    i0 = [int(v) for v in vl2].index(1)
    assert int(ag2[i0]) == 11, (
        f"une ligne propagée doit garder son âge VRAI (11), pas celui de sa "
        f"position ({int(ag2[i0])})")
    assert int(ag2[i0]) > s_age.max_mem, (
        "l'horizon effectif doit pouvoir DÉPASSER la profondeur de la FIFO — "
        "c'est la signature mesurable de la propagation")
    # 10 propagations et non 12 : les deux premiers writes n'ont rien (ou une
    # seule entrée) à propager — déplacer l'unique ligne serait un no-op.
    assert s_age.n_prop == 10 and len(s_age.rows) == 8, s_age.n_prop

    # (c) CAUSALITÉ DU VOTE — LE TEST SÉVÈRE (leçon S2 : le bank-q a été
    #     invalidé pour avoir lu le futur teacher-forcé). Une observation ne
    #     doit RIEN changer avant le write SUIVANT, et le tour t ne doit
    #     compter qu'au write t+1.
    sA = _fill_store("attn-ema", 1, n=6)
    sB = _fill_store("attn-ema", 1, n=6)
    boom = [0.0] * len(sB.rows)
    boom[0] = 1000.0                     # masse ÉNORME sur la plus vieille
    assert [sA.score(i) for i in range(len(sA.rows))] == \
        [sB.score(i) for i in range(len(sB.rows))]
    sB.observe(torch.tensor(boom))
    assert [sA.score(i) for i in range(len(sA.rows))] == \
        [sB.score(i) for i in range(len(sB.rows))], \
        "observe() a bougé un score AVANT le moindre write — fuite du futur"
    r6 = _row(6)
    sA.write(r6, _vec(r6), val=99)
    sB.write(r6, _vec(r6), val=99)
    assert [e["val"] for e in sA.rows] == [e["val"] for e in sB.rows], (
        "la masse du tour t a compté DÈS le write du tour t : c'est "
        "exactement la fuite non-causale qui a invalidé S2")
    assert [sA.score(i) for i in range(len(sA.rows))] == \
        [sB.score(i) for i in range(len(sB.rows))], (
        "au write du tour t la masse du tour t est seulement ARMÉE, jamais "
        "commise — sinon le signal lirait le tour courant")
    r7 = _row(7)
    sA.write(r7, _vec(r7), val=98)
    sB.write(r7, _vec(r7), val=98)
    assert [sA.score(i) for i in range(len(sA.rows))] != \
        [sB.score(i) for i in range(len(sB.rows))], \
        "au write t+1 la masse du tour t DOIT enfin être commise"
    assert [e["val"] for e in sA.rows] != [e["val"] for e in sB.rows], (
        "au write t+1 la masse du tour t DOIT enfin peser sur la propagation")
    # et les signaux qui ne votent pas ignorent la sonde ENTIÈREMENT.
    sC = _fill_store("coverage", 1, n=6)
    scC = [sC.score(i) for i in range(len(sC.rows))]
    sC.observe(torch.tensor(boom))
    sC._commit()
    assert [sC.score(i) for i in range(len(sC.rows))] == scC, \
        "`coverage` ne doit RIEN lire de la sonde"

    # (d) DÉTERMINISME à graine fixée : deux rejeux identiques ⇒ même store,
    #     signal par signal. Une maintenance non déterministe rendrait le
    #     bakeoff inarbitrable.
    for sig, p in (("fifo", 0), ("age", 1), ("attn-ema", 1),
                   ("coverage", 1), ("actr", 2)):
        a1 = [e["val"] for e in _fill_store(sig, p).rows]
        a2 = [e["val"] for e in _fill_store(sig, p).rows]
        assert a1 == a2, f"signal {sig} non déterministe : {a1} vs {a2}"

    # (e) COUVERTURE : la ligne REDONDANTE meurt, la singulière survit. On
    #     construit trois entrées quasi colinéaires et une orthogonale.
    s_cov = RetentionStore(4, "coverage", 1)
    base = torch.zeros(3, 64)
    base[:, 0] = 1.0
    for i in range(3):
        r = base.clone()
        r[:, 1] = 0.01 * i               # trois quasi-jumelles
        s_cov.write(r, _vec(r), val=i + 1)
    lone = torch.zeros(3, 64)
    lone[:, 5] = 1.0                     # la SINGULIÈRE
    s_cov.write(lone, _vec(lone), val=42)
    r5c = _row(5)
    s_cov.write(r5c, _vec(r5c), val=7)
    assert 42 in [e["val"] for e in s_cov.rows], (
        "la ligne SINGULIÈRE devait être propagée (son plus proche voisin est "
        f"le plus loin) : {[e['val'] for e in s_cov.rows]}")

    # (f) ACT-R : un USAGE récent bat une naissance récente. C'est la
    #     différence entre « activation » et « récence », et sans elle le bras
    #     ne serait qu'un `fifo` déguisé.
    s_ar = RetentionStore(8, "actr", 1)
    for i in range(4):
        r = _row(i)
        s_ar.write(r, _vec(r), val=i + 1)
    m = [0.0] * 4
    m[0] = 1.0                            # la plus VIEILLE est utilisée
    s_ar.observe(torch.tensor(m))
    r9 = _row(9)
    s_ar.write(r9, _vec(r9), val=9)       # arme
    r10 = _row(10)
    s_ar.write(r10, _vec(r10), val=10)    # commet ⇒ l'usage compte
    assert s_ar.rows[0]["uses"] or any(e["uses"] for e in s_ar.rows), \
        "aucun usage n'a été enregistré : le canal `actr` est mort"
    assert s_ar.score([e["val"] for e in s_ar.rows].index(1)) > \
        s_ar.score([e["val"] for e in s_ar.rows].index(2)), (
        "une ligne UTILISÉE doit battre une ligne du même âge jamais utilisée")

    # (g) L'ENV DES VIES LONGUES : la pression FIFO est RÉELLE, la question
    #     est DANS le seg gradé (train/éval isomorphes, §4.3), et les trois
    #     strates existent avec les bonnes vérités.
    ls = PersonaLifeStream(tok, seed=5, life_turns=48)
    lconv = next(c for c in (ls.next_conv() for _ in range(20))
                 if c["kind"] == "life")
    li = lconv["info"]["life"]
    assert li["n_writes"] > 8, (
        f"une vie doit écrire ≫ max_mem (sinon rien n'atteint le bord) : "
        f"{li['n_writes']}")
    assert li["n_writes"] == 2 + max(4, (48 - 8) // 4), li["n_writes"]
    st_l = lconv["info"]["p11"]["strate"]
    assert set(st_l) == {"cur_pre", "cur_post", "hist"}, set(st_l)
    for q, tr, s in zip(lconv["info"]["p11"]["turns"],
                        lconv["info"]["truths"], st_l):
        seg = lconv["segs"][q]
        assert seg.get("q_len", 0) > 0, "la question doit être DANS le seg"
        assert seg["loss_mask"][0, :seg["q_len"]].sum() == 0, (
            "la question ne doit PAS être supervisée")
        # la vérité de `hist` est l'ANCIENNE valeur — c'est toute la sonde.
        assert tr == (li["v_old"] if s in ("cur_pre", "hist")
                      else li["v_new"]), (s, tr)
    dws = [d for d, s in zip(li["dw"], st_l) if s == "cur_post"]
    assert dws == sorted(dws) and dws[0] == 0 and len(set(dws)) > 1, dws

    # (h) LA CIBLE DU r@1 EN VIE LONGUE : c'est le groupe de la VALEUR, pas
    #     « le plus récent du bon slot ». Les deux writes de `city` partagent
    #     le slot ; la convention ph.11 raterait TOUJOURS la strate `hist`.
    _sid = env.slot_ids["city"]
    _vo, _vn = env.val_ids[li["v_old"]], env.val_ids[li["v_new"]]
    _ent = (torch.zeros(2, 3, 64), torch.tensor([5, 0]), torch.zeros(2).long(),
            torch.tensor([_sid, _sid]), torch.tensor([_vo, _vn]))
    _qh = next(i for i, s in enumerate(st_l) if s == "hist")
    _qc = next(i for i, s in enumerate(st_l) if s == "cur_post")
    assert env.p11_target(lconv, 0, _qh, _ent) == 0, \
        "la question HISTORIQUE doit viser l'ANCIEN write"
    assert env.p11_target(lconv, 0, _qc, _ent) == 1, \
        "la question COURANTE doit viser le NOUVEAU write"

    # (i) LE PLAN DE VIE, DE BOUT EN BOUT : `fifo` perd l'ancre, la
    #     propagation la garde. C'est le bakeoff en miniature.
    m12 = _mk12(p11_env="life", retention="fifo", prop_budget=0)
    pl_f = env.p11_plan(m12, lconv)
    assert pl_f and all(len(e) == 5 for e in pl_f.values()), \
        "le plan `life` porte CINQ champs (les valeurs en plus)"
    _last = pl_f[max(pl_f)]
    assert _vo not in [int(v) for v in _last[4]], (
        "sous FIFO nue, l'ancre doit avoir CHUTÉ au dernier tour gradé — "
        "sinon la vie est trop courte et l'examen ne mesure rien")
    m12p = _mk12(p11_env="life", retention="age", prop_budget=1)
    pl_a = env.p11_plan(m12p, lconv)
    assert _vo in [int(v) for v in pl_a[max(pl_a)][4]], (
        "avec propagation, l'ancre doit SURVIVRE au dernier tour gradé")
    assert env.last_life["n_prop"] > 0 and env.last_life["n_drop"] > 0
    # les métadonnées de la banque ne bougent PAS le forward du bras nu :
    # `life` sans propagation ≡ la FIFO des phases précédentes.
    assert env.p11_plan(m12, pconv)[
        next(iter(env.p11_plan(m12, pconv)))][0].shape[0] >= 1

    # (j) LE VOTE PASSE PAR LA SONDE, ET LA SONDE N'ALTÈRE RIEN. Un bras qui
    #     vote doit rendre le MÊME forward qu'un bras qui ne vote pas.
    m12e = _mk12(p11_env="life", retention="attn-ema", prop_budget=1)
    _ids12 = torch.tensor([[3, 4, 5, 6]])
    _rw12 = torch.randn(1, 2, 3, 64)
    with torch.no_grad():
        _o1 = m12e(_ids12, None, None, inject=_rw12)
        with bank_attn_probe(m12e):
            _o2 = m12e(_ids12, None, None, inject=_rw12)
        _o3 = m12e(_ids12, None, None, inject=_rw12)
    assert torch.equal(_o1, _o2) and torch.equal(_o1, _o3), \
        "la sonde ph.12 altère le forward (ou ne s'éteint pas)"
    assert all(not b.attn.want_bank_attn for b in m12e.blocks)

    # (k) LE LOT DE LIGNES ÉTRANGÈRES ≡ segment par segment. Sans ça le
    #     remplissage de S8 injecterait autre chose que ce que le write
    #     produirait, et la dilution mesurerait un artefact de batching.
    _segs12 = [seg20[:12], seg20[:20], seg20[:16]]
    _bt = m12.group_rows_batch(_segs12)
    for _i, _s in enumerate(_segs12):
        assert torch.allclose(_bt[_i], m12.tophid_rows_fixed(_s), atol=1e-4), \
            f"group_rows_batch diverge du calcul segment par segment ({_i})"

    # (l) LE REMPLISSAGE (S8) : exactement S groupes, les VRAIS résidents EN
    #     DERNIER, les remplisseurs à métadonnées NULLES (donc jamais cible du
    #     r@1), et un NO-OP EXACT quand il est éteint.
    m12f = _mk12(p11_env="span", p12_exam="dilution", bank_fill="foreign",
                 max_mem=8, top_k=3)
    _fake = {0: (torch.randn(2, 3, 64), torch.tensor([1, 0]),
                 torch.zeros(2).long(), torch.tensor([9, 9]),
                 torch.tensor([4, 5]))}
    assert env.fill_plan(m12, dict(_fake)) == _fake or \
        list(env.fill_plan(m12, dict(_fake)).values())[0][0].shape[0] == 2, \
        "bank_fill=none doit être un NO-OP exact"
    env.foreign_pool = [seg20[:14 + i % 5] for i in range(10)]
    env._fill_rows = None
    _fl = env.fill_plan(m12f, dict(_fake))[0]
    assert _fl[0].shape[0] == 8, _fl[0].shape
    assert list(map(int, _fl[3][-2:])) == [9, 9] and \
        list(map(int, _fl[4][-2:])) == [4, 5], "les VRAIS doivent être EN DERNIER"
    assert set(map(int, _fl[3][:-2])) == {0} and \
        set(map(int, _fl[4][:-2])) == {0}, (
        "les remplisseurs doivent porter des métadonnées NULLES : sinon ils "
        "pourraient devenir la cible du r@1 et la dilution deviendrait un "
        "problème d'étiquetage")
    assert torch.equal(_fl[0][-2:], _fake[0][0]), \
        "le remplissage ne doit RIEN changer aux vraies lignes"
    assert list(map(int, _fl[1])) == [7, 6, 5, 4, 3, 2, 1, 0], \
        list(map(int, _fl[1]))
    env.foreign_pool, env._fill_rows = [], None

    # (m) NOMMAGE : 23 cellules ph.12 ⇒ 23 dossiers, ZÉRO collision avec les
    #     96 de la grille §2.4 ni les 22 de la ph.11.
    _p12n, _p12c = [], 0
    for _ex in P12_EXAMS:
        _cc = p12_combos(_ex)
        _p12c += len(_cc)
        _p12n += [p12_name(_p12_cfg(c, _base10)) for c in _cc]
    assert _p12c == 23 and len(set(_p12n)) == 23, (_p12c, sorted(_p12n))
    assert not (set(_p12n) & (_lancees | set(_p11n))), (
        "une cellule ph.12 écraserait un run déjà lancé")
    assert all(n.startswith("p12-") for n in _p12n), _p12n
    for _ex in P12_EXAMS:
        for c in p12_combos(_ex):
            _cf = _p12_cfg(c, _base10)
            assert run_name_for(_cf) == p12_name(_cf) and \
                p12_name(_cf).startswith(f"p12-{_ex}_"), p12_name(_cf)
    # la BASELINE BASSE vit bien dans l'espace de noms ph.12 (comme le
    # contrôle θ_âge=0 en ph.11) : sans le champ déclaré elle n'aurait rien
    # pour se distinguer d'une cellule de la grille.
    _b12 = _p12_cfg({"exam": "retention", "env": "life", "m": 4,
                     "steps": 1500, "retention": "fifo", "prop_budget": 0,
                     "life_turns": 48}, _base10)
    assert run_name_for(_b12) == "p12-retention_fifo_T48" and \
        not _b12.uses_retention

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
    print(f"  correctif seq_fw — GRADIENT ADDITIF (code.fw_additive, défaut "
          f"OFF) : le forward est identique BIT-À-BIT (l'écriture "
          f"`y.detach() + (y0 − y0.detach())` ajoute un zéro EXACT ; la forme "
          f"naïve `y0 + (y − y0).detach()` échouait ce test), les GRADIENTS "
          f"diffèrent, et le produit de jacobiennes a disparu : à M=64 sur "
          f"une banque à forte échelle, gnorm multiplicative {gb_off:.2e} "
          f"contre additive {gb_on:.2e} ({gb_off / max(gb_on, 1e-30):.0f}×), "
          f"et la dégradation avec M est bien plus forte côté multiplicatif "
          f"(M=8→64 : ×{gb_off / max(g8_off, 1e-30):.0f} contre "
          f"×{gb_on / max(g8_on, 1e-30):.1f}). Refusé sur r1/r3/r4 (aucune "
          f"boucle), suffixe _fwadd au save_dir, et le sous-ensemble de "
          f"manifeste `seqfw-additive` rend 12 cellules dont AUCUNE ne "
          f"collisionne avec le bras multiplicatif.", flush=True)
    print(f"  TÊTES DÉDIÉES (--read dual_heads) : groupe de "
          f"{ToyCfg.bank_heads} têtes par couche LECTRICE, attention sur la "
          f"banque SEULE (ni RoPE ni masque causal), fusion avec les têtes de "
          f"contexte AVANT le FFN. Vérifié : présent-mais-nul ≡ ABSENT "
          f"bit-à-bit à l'init (W_o zéro-init) ; W_o reçoit du gradient DÈS le "
          f"premier backward — c'est ce qui évite le piège r3 de la porte "
          f"multiplicative morte — et 5 pas suffisent à décoller la tête ; "
          f"la banque est un ENSEMBLE (permuter les lignes ne change rien, "
          f"invariant impossible pour inject_entry) et la position 0 la voit ; "
          f"têtes limitées aux couches lectrices ; knobs bank_heads/"
          f"bank_head_dim REFUSÉS hors dual ; 12 cellules `dual` sans "
          f"collision avec les 36 lancées ni les 12 _fwadd.", flush=True)
    print(f"  CARRÉ FACTORIEL — 3ᵉ sommet `--read kv_proj` (projections "
          f"DÉDIÉES / softmax UNIFIÉ) : sans banque ≡ backbone BIT-À-BIT ; la "
          f"banque et son CONTENU comptent, la position 0 la voit ; le biais "
          f"de logits par tête est init 0, MORD sur la sortie et reçoit du "
          f"gradient (donc sa valeur loggée est une vraie mesure) ; --bank-q "
          f"est bit-à-bit le kvproj nu à l'init (W_o' zéro-init) puis décolle "
          f"en 5 pas, les lignes restent un ENSEMBLE (permutation neutre) et "
          f"leur état est JETÉ en sortie de stack ; bank_q REFUSÉ hors kvproj. "
          f"Nommage : 12 cellules kvproj + 12 _bq, sans collision avec les 36 "
          f"lancées, les 12 _fwadd ni les 12 dual.", flush=True)
    print(f"  PHASE 11 — MÉTADONNÉES PAR ROTATION sur K' (spec §2.5). "
          f"(a) APPARIEMENT : les plans sont les paires les PLUS LENTES du "
          f"RoPE de tête (tri par fréquence, pas une convention d'index) et "
          f"la requête ne tourne que de {drift11:.4f} rad sur toute la "
          f"fenêtre ; une bande trop large est REFUSÉE à la construction. "
          f"(b) ÂGE : rot(0) ≡ identité (le kvproj nu au bit près), l'âge "
          f"MORD, log et brut COÏNCIDENT à A_ref et divergent en OOD "
          f"(a=800 ⇒ φ 24.3 contre 800), fréquences apprises géométriques et "
          f"DÉRIVABLES. (c) `age-bias` : UN paramètre par couche, init 0 ≡ "
          f"kvproj nu, mord et reçoit du gradient. (d) TAG : R(π)² = I "
          f"vérifié, le canal change le forward, le steelman ADDITIF réserve "
          f"le MÊME nombre de plans et démarre bit-à-bit sur le nu. "
          f"(e) INDEX LOCAL : R_loc(1) est un opérateur CONSTANT (écarts "
          f"d'angle égaux à 1e-4 près) — l'argument structurel contre "
          f"l'additif. (f) les trois familles occupent des plans DISJOINTS. "
          f"(g) la sonde d'attention n'altère pas le forward et s'éteint. "
          f"(h) 8 combinaisons no-op REFUSÉES, examen non déclaré compris. "
          f"(i) envs : `prov` écrit UN fait par canal à slot ET attribut "
          f"IDENTIQUES (la clé ne peut rien) et pose la question DANS le seg "
          f"gradé, NON supervisée ; `span` mesure ses longueurs au lieu de "
          f"les décréter. (j) p11_plan : tous les résidents, âges = rangs de "
          f"récence, canaux portés, cible du r@1 = le groupe du bon canal, "
          f"no-op hors ph.11. (k) augmentation d'échelle : ordre préservé, "
          f"no-op EXACT quand OFF. (l) 22 cellules ⇒ 22 dossiers, ZÉRO "
          f"collision avec les 96 de la grille §2.4, et le contrôle θ_âge=0 "
          f"vit bien dans l'espace de noms ph.11.", flush=True)
    print(f"  PHASE 12 — MAINTENANCE PROCÉDURALE (S6) ET DILUTION (S8), "
          f"ZÉRO paramètre, ZÉRO RL. (a) `fifo` à budget 0 EST la FIFO "
          f"historique (mêmes résidents, mêmes âges, 0 propagation) et les "
          f"7 combinaisons no-op sont REFUSÉES. (b) la PROPAGATION préserve "
          f"la NAISSANCE : l'ancre sauvée du bord garde l'âge 11 (> max_mem "
          f"8) — un horizon effectif au-delà de la profondeur EST la "
          f"signature de la propagation. (c) CAUSALITÉ (leçon S2) : une "
          f"observation ne bouge AUCUN score avant le write suivant, la "
          f"masse du tour t est armée au write t et commise au write t+1, et "
          f"les signaux non-votants ignorent la sonde. (d) les 5 signaux sont "
          f"déterministes à graine fixée. (e) `coverage` propage la ligne "
          f"SINGULIÈRE contre trois quasi-jumelles ; (f) `actr` fait battre "
          f"une ligne UTILISÉE contre une ligne du même âge jamais utilisée. "
          f"(g) l'env `life` écrit {li['n_writes']} fois contre max_mem 8, "
          f"pose sa question DANS le seg gradé (non supervisée) et rend les "
          f"trois strates avec les bonnes vérités (`hist` → l'ANCIENNE). "
          f"(h) la cible du r@1 y est le groupe de la VALEUR, pas le plus "
          f"récent du slot — la convention ph.11 raterait TOUJOURS `hist`. "
          f"(i) de bout en bout : sous FIFO nue l'ancre a CHUTÉ au dernier "
          f"tour gradé, avec propagation elle SURVIT. (j) la sonde n'altère "
          f"pas le forward et s'éteint. (k) le lot de lignes étrangères ≡ le "
          f"calcul segment par segment. (l) le remplissage S8 rend exactement "
          f"S groupes, vrais résidents EN DERNIER, remplisseurs à "
          f"métadonnées NULLES, no-op exact quand éteint. (m) 23 cellules "
          f"⇒ 23 dossiers, ZÉRO collision avec les 96 de la grille §2.4 ni "
          f"les 22 de la ph.11.", flush=True)
    print("  ABSOLUS (règle durcie, audit 08-03) — evaluate_p11 et "
          "evaluate_cond émettent désormais nll_live ET nll_abl (plus la "
          "nll VALEUR seule) : le Δnll intra-modèle reste rendu sous "
          "`dnll_deprecated`, jamais comme juge")
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
