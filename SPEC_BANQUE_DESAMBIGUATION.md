# Spécification de la banque proposée — levée des ambiguïtés

**Date** : 2026-08-02
**Auteur** : Kkuette
**Objet** : désambiguïser ce que `DESIGN v0.5` demande, avant l'implémentation d'E1.
**Statut** : cette note **corrige et précise** `DESIGN_EM1.html` v0.5.x, qui employait « rotation »
pour deux opérations distinctes sans les séparer. L'ambiguïté est de mon fait.

---

## 1. Les objets

| Objet | Shape | Ce que c'est | Espace |
|---|---|---|---|
| `x_j` | `(d_m,)` | une **ligne native** — un vecteur sélectionné dans `H` du segment, non transformé | espace d'embedding |
| `s_j` | scalaire entier | **indice d'écriture absolu** de la ligne `j` (compteur monotone, jamais réinitialisé) | — |
| `bank` | `(M, G, d_m)` | tenseur : M segments × G lignes par segment | — |
| `bank_flat` | `(N, d_m)`, N = M·G | la même chose aplatie **selon `n`** | — |
| `m̃` | `(d_m,)` | le code composite lu — **un seul vecteur** | — |

**Invariant** : `d_m` ne change jamais. L'aplatissement porte sur `n`, jamais sur `d`. L'entrée du
hypernet reste `d_m`.

---

## 2. Quatre opérations, quatre objets différents

C'est ici que la spec v0.5 était floue. **Chaque opération agit sur un objet précis et n'est pas
interchangeable avec les autres.**

| # | Opération | Agit sur | Rôle | Sans elle |
|---|---|---|---|---|
| **O1** | **sélection à l'écriture** | les `T` lignes de `H` du segment | choisir les `G` lignes à conserver | on écrit tout, ou un pooling |
| **O2** | **liaison** (rotation) | **la LIGNE `x_j`** | rendre les lignes **séparables dans la somme** | la somme est une moyenne : interférence non gérée |
| **O3** | **récupération** | les `N` entrées de la banque | choisir les `k` entrées à composer pour cette requête | on compose tout, borne de fidélité atteinte |
| **O4** | **déliaison + cleanup** | `m̃` | ré-extraire un élément et le projeter sur `E` | on lit une superposition brute |

**O1 ≠ O3.** La sélection à l'écriture (`topsum`, poids SIF, dans le segment, une fois) et la
récupération à la lecture (`retr_topk`, sur toute la banque, par requête) sont **deux mécanismes
distincts**, et l'architecture a besoin des **deux composés**. `topsum` n'implémente pas O3.

**O2 agit sur la ligne, pas sur la clé.** C'est le point critique — voir §4.

---

## 3. Les quatre ambiguïtés de ma spec, tranchées

### A1 — Le FIFO porte sur quoi ?
**Sur `M` (les segments), pas sur `N` (les lignes).** Une éviction retire un **groupe de G lignes**
d'un coup. Conséquence : la cohérence d'un segment est préservée, et `s_j` est constant à l'intérieur
d'un groupe. Un FIFO sur les lignes individuelles casserait un segment en morceaux d'âges différents
et rendrait `s_j` non informatif au niveau du groupe.

### A2 — `s_j` est absolu, jamais relatif à la position dans l'anneau
Compteur monotone global. **Une ligne n'est jamais re-tournée** quand l'anneau tourne. C'est la
requête qui est tournée par le compteur courant `t`. Un `s_j` = position dans l'anneau obligerait à
re-tourner tout le tenseur à chaque éviction, et ferait diverger train et éval si le compteur vit du
mauvais côté.

### A3 — Les rôles `ρ` ne sont testables que dans le cas à deux usages
`ρ` sépare **un code de règle** et **des lignes citables** *portés par le même vecteur*. Dans le
montage actuel du toy lab, une ligne ne porte **jamais** deux types de contenu — donc `ρ` n'a rien à
séparer et il est **structurellement** un no-op, indépendamment de `d` ou de `key_from_content`.

> **Conséquence pratique** : mesurer `ρ` aujourd'hui ne mesure rien de la proposition. Le bras
> `_role` est à **retirer de la vague E1**, pas à tester. Il redevient pertinent le jour où une ligne
> porte simultanément un code de règle et une valeur citable — c'est-à-dire au moment où le canal
> unique existe, pas avant.

### A4 — `k` est un paramètre de la récupération (O3), pas de la sélection d'écriture (O1)
La borne de fidélité `1/√(1+k/d_m)` porte sur le **nombre d'éléments liés dans un même `m̃`**,
c'est-à-dire sur O3. Un `k` d'écriture dans un segment de 12–26 tokens (le `topsum` du plan) est un
autre paramètre, avec une autre borne. **Les deux ne doivent pas partager un nom.**

---

## 4. La tension réelle — et le plan la tranche sans le dire

Le plan d'exécution pose, en 2d :

> « **Tourner la CLÉ et la REQUÊTE, jamais la ligne.** `GroupReadout` étage 2 fait
> `ligne @ rms_unit(E)ᵀ` avec zéro paramètre : une ligne tournée n'est plus un embedding et tout le
> design phase 6 tombe. »

**La contrainte est réelle et bien vue. Mais la décision supprime O2.**

Rotation sur la clé et la requête ⇒ le **score** dépend de l'âge (récence). Les lignes, elles, sont
sommées **non tournées** ⇒ la superposition n'est **pas** séparable. On obtient la récence et on perd
la liaison, qui était la raison d'être de l'opération.

Et l'étape 2a du même plan mesure, elle, **la version tournée sur la ligne** (« ligne = somme de k
embeddings liés par rotation, puis dé-rotation »). **Le plan mesure O2 à l'oracle et implémente autre
chose à l'entraînement.** C'est exactement l'écart qui produirait un résultat propre et non concluant.

### La résolution

Elle est mécanique, et elle ne casse pas phase 6 :

```
écriture :   r_j = R(s_j) · x_j                      # la ligne stockée EST tournée
somme    :   m̃  = Σ_j α_j · R(s_j) · x_j
déliaison:   R(−s_i) · m̃ = α_i·x_i + Σ_{j≠i} α_j·R(s_j−s_i)·x_j
                            └ signal ┘   └──── résidus déphasés ────┘
readout  :   ( R(−s_i) · m̃ ) @ rms_unit(E)ᵀ          # phase 6 intacte, sur le vecteur DÉLIÉ
```

`R` est orthogonale par blocs : `R(a)R(b) = R(a+b)`, `R(s)ᵀ = R(−s)`, norme préservée. Donc
`R(−s_i)·m̃` **est de nouveau dans l'espace d'embedding**, et `ligne @ rms_unit(E)ᵀ` s'applique tel
quel, sans paramètre. Le design phase 6 ne tombe pas — il s'applique **après la déliaison** au lieu
de s'appliquer directement.

**Ce qu'il faut pour ça** : connaître `s_i` au moment du readout. C'est gratuit — `s_j` est stocké
avec la ligne, et O3 (la récupération) désigne déjà l'entrée dominante. La boucle est la boucle
standard des VSA : *retrouver → délier → nettoyer*.

**Bénéfice secondaire** : la clé dérivée de la ligne tournée hérite de la rotation, donc
`⟨R(t)q, R(s)x⟩ = ⟨q, R(s−t)x⟩` — **la récence sort gratuitement de la même opération**. Une seule
rotation, appliquée à la ligne, donne O2 *et* la propriété de récence de 2d. Il n'y a pas à choisir.

---

## 5. Séparation ≠ récence — et pourquoi le DFT bat RoPE

Le plan rapporte : « la forme RoPE littérale est mesurée **inférieure** au binding DFT (3/8 positions
contre 8/8) ». **Ce n'est pas un argument contre la liaison — c'est la preuve que les deux mécanismes
optimisent des objectifs différents.**

| | Objectif | Schéma de fréquences | Bon pour |
|---|---|---|---|
| **Binding DFT** | **séparation exacte** — chaque position doit être discernable | fréquences réparties uniformément | distinguer N entrées les unes des autres |
| **RoPE littéral** | **distance graduée** — les proches doivent se ressembler plus que les lointains | schéma géométrique, l'essentiel de l'énergie dans les rotations lentes | préférence de récence continue |

La mesure 8/8 contre 3/8 teste la **séparation**. Le DFT gagne parce que c'est son métier. RoPE perd
parce qu'il n'a jamais été conçu pour ça.

> **Décision** : garder **le binding DFT existant** pour O2 (la séparation), et n'introduire un
> schéma de fréquences façon RoPE **que si** une préférence de récence graduée est explicitement
> voulue. Ma formulation v0.5.1 (« façon RoPE ») était une erreur de désignation : le mécanisme
> demandé est celui déjà en place, mieux mesuré.
>
> ⚠️ Et un mode d'échec à connaître si le schéma RoPE est un jour retenu : avec des rotations à taux
> uniforme, une entrée vieille de 1 000 écritures a une phase quasi aléatoire vis-à-vis de la requête
> et son score s'effondre **quel que soit son contenu**. Ce n'est pas un a priori doux, c'est un
> brouillage. Le schéma multi-fréquences de RoPE existe précisément pour éviter ça.

---

## 6. Ce que ça change au plan d'exécution, concrètement

| Item du plan | Modification |
|---|---|
| **2c — rôles ρ** | **Retirer de la vague.** Structurellement no-op tant qu'une ligne ne porte pas deux types de contenu (A3). Le self-test « ρ mesure sa propre inutilité » reste utile comme documentation, mais il ne faut pas planifier de run derrière. |
| **2d — rotation** | Appliquer la rotation **à la ligne** (`r_j = R(s_j)·x_j`), pas à la clé. Ajouter la **déliaison** `R(−s_i)` avant le readout `@ rms_unit(E)ᵀ` de `GroupReadout` étage 2. La récence s'obtient alors gratuitement, et 2a mesure enfin la même chose que ce qui est entraîné. |
| **2d — binding** | Réutiliser le **binding DFT existant**, pas une forme RoPE littérale. La mesure du dépôt tranche (§5). |
| **2b — `topsum`** | Sans changement, mais **renommer le paramètre** : c'est un `k` d'écriture (O1), à ne pas confondre avec le `k` de récupération (O3) auquel s'applique la borne `1/√(1+k/d_m)`. Deux noms distincts dans le CSV. |
| **2a — balayage oracle** | Sans changement — mais il devient le **contrôle** de 2d au lieu d'en être déconnecté, puisque les deux portent désormais sur la ligne. |
| **FIFO** | Documenter explicitement qu'il porte sur les **groupes**, pas sur les lignes (A1). |

---

## 7. Test d'acceptation de la compréhension

Trois assertions à vérifier avant de coder. Si l'une est fausse dans l'implémentation, l'expérience
mesurera autre chose que ce qui est demandé.

1. **`torch.equal(bank[i], bank[i])` à travers les écritures** — une ligne stockée n'est **jamais**
   re-tournée ni modifiée après son écriture, même quand l'anneau tourne.
2. **`R(−s_i) @ (R(s_i) @ x)` == `x`** au bit en float64, et l'écart imprimé en float32.
3. **Round-trip à `k=1`** : une ligne seule, liée puis déliée, doit donner **exactement** le résultat
   du chemin non lié actuel. Si c'est faux, la liaison introduit une perte là où elle devrait être
   une identité, et rien de ce qui suit n'est interprétable.

---

## 8. Vérifications faites en dépôt après cette note (2026-08-02)

Trois constats qui ne changent pas la spec mais réduisent le travail d'implémentation :

- **O2 existe déjà, sur l'axe *position*.** `oracle_lines` lie par `rot_pairs(ew[:n], sg_cos, sg_sin)`
  pour `segphase`/`segsif` et `candidates()` délie par `rot(−θj)` avant le readout. La boucle
  *lier → délier → nettoyer* est bâtie et mesurée : round-trip oracle **97,4 %** (`segsif`, d=512),
  **94,9 %** sur la strate `code`. Ce que cette note ajoute est l'**axe `s_j`**, qui se compose avec
  l'existant sans le remplacer, exactement par `R(a)R(b) = R(a+b)`.
- **La déliaison a DEUX points d'accrochage, pas un.** `candidates()` retourne la banque **inchangée**
  pour les `GROUP_CODES` (« la ligne EST le candidat »). La déliaison d'âge se compose donc dans
  `candidates()` pour `segphase`/`segsif`/`topsum`, mais elle doit vivre **dans `GroupReadout`** pour
  `toprows`, avant `u @ rms_unit(embed_w).t()`. Une seule implémentation aurait laissé `toprows`
  silencieusement non délié.
- **Le coût est négligeable** : `s_j` étant constant par groupe (A1), c'est une rotation par groupe,
  G ≤ 8.
