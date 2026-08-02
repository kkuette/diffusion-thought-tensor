# Addendum au plan d'exécution — retours de l'auteur de PLAN_EXPERIENCES.md

**Date** : 2026-08-02
**Objet** : réponse au plan d'exécution produit après vérification en dépôt.
**Statut** : le plan d'exécution **prévaut** sur `PLAN_EXPERIENCES.md` partout où ils divergent.

---

## A. Ce que je retire — mes prémisses fausses

Le plan d'exécution a travaillé sur le HEAD local (8 commits + fichiers non commités que je n'ai pas vus :
`rti*.py`, `recall_env.py`, `SPEC_MEMOIRE_V2.md`, kill-test 3). Trois de mes six expériences reposaient sur
des prémisses mesurées fausses. **À retirer explicitement de `PLAN_EXPERIENCES.md`** :

| Ce que j'écrivais | Ce qui est mesuré |
|---|---|
| **E2** : « la citation gratuite par injection post-norm, facteur 42 » | `PointerReadout` (`toy_read_lab.py:832`) implémente déjà l'identité exacte, et il est **mort** : 12 runs r3, grade held-out ≤ 0,100 contre 0,281 pour l'injection native. Et mon « facteur 42 » comparait à une tête `d_m × V` **qui n'existe nulle part dans le dépôt** — le vrai `rti_copy` coûte 0,61 M MACs/token, soit **moins que l'homme de paille que j'ai construit**. C'est mon erreur la plus grossière : j'ai chiffré une économie contre une implémentation imaginaire. |
| **E1** : « la rotation façon RoPE » | Mesurée **inférieure** au binding DFT déjà en place (3/8 positions contre 8/8), et `pos_offset` est nul (97,4 → 97,5 %). Ma proposition est une régression par rapport à l'existant. |
| **E4.1** : « les chiffres sont fatalement évincés par la fréquence marginale » | **Faux** : inclusion numérique pleine à 1.000, même avec le teacher `a`. Le vrai bug était `k` fixe contre longueur de tour. Mon raisonnement était juste dans l'abstrait et faux sur ce système. |

**Ce qui survit intact** : E0 (l'oracle, confirmé comme pivot), C1 (et il a déjà rendu son verdict), la
discipline de rapport, les règles méthodologiques du §1.

---

## B. E1 — la question a changé, et il faut la reformuler avant de dépenser la ferme

**C'est ma seule remarque de fond.**

Le plan d'exécution rapporte deux faits qui, mis ensemble, déplacent la question :

1. le round-trip **oracle** est à **94,9 %** sur la strate `code` à d=512 — *le code est inversible* ;
2. `r3` / `PointerReadout` plafonne à **≤ 0,100** en held-out — *le read appris ne l'exploite pas*.

Donc l'écart n'est **pas** dans la représentation. Il est dans ce que le read appris parvient à faire d'un
code déjà décodable à 95 %.

Or E1 tel que planifié (`topsum`, `age_rot`, `ρ`) travaille **sur le code** : rendre inversible ce qui l'est
déjà à 94,9 %. Le gain plafonne par construction à 5 points d'un intermédiaire, pendant que le vrai gap fait
**85 points**.

Le plan d'exécution applique déjà exactement le bon réflexe sur ρ — « un self-test mesure le no-op avant de
dépenser une carte ». **Je propose de généraliser ce réflexe à tout E1**, avec une sonde préalable :

> **E1.0 — plafond du readout appris, à code oracle.**
> Geler tout ; poser la banque oracle (celle qui donne 94,9 % en round-trip) ; **n'entraîner que le readout**,
> en supervision directe ligne → token. Mesurer le plafond.
>
> - **Plafond élevé** (≫ 0,100) ⇒ le read *peut* citer, il n'a jamais été **entraîné** à le faire. On est
>   dans la contribution n°3 du papier : rien n'émerge, il faut une pression dédiée. → la suite est une
>   question de **pression d'entraînement**, pas d'architecture, et `topsum`/`age_rot`/ρ sont hors sujet.
> - **Plafond bas** ⇒ c'est bien la **classe de fonctions** du readout. → la refonte du canal unique est
>   justifiée par la mesure, et E1 reprend tel que planifié.
>
> Coût : CPU ou une carte, minutes. Aucune dépense de vague ferme avant ce résultat.

Sans cette sonde, la vague E1 (5 variantes × 45 min) peut mesurer très proprement des variations de 2 points
sur un intermédiaire qui n'est pas le facteur limitant.

**Le reste de E1 est bon et je n'y touche pas** : le découplage 2a/2b (le balayage `k ∈ {16…256}` était
effectivement impossible avec `seg_n_pos: 32` et des segs de 12–26 tokens — je ne connaissais pas ces
chiffres), le `return_cos` (top-1 argmax et cosinus sont deux quantités différentes, reporter les deux est
juste), le contrôle de continuité `topsum ≡ segsif` bit-à-bit à `k ≥ |seg|`, et le correctif du suffixe `_k`
de `run_name_for` — celui-là aurait silencieusement écrasé des runs.

---

## C. E3 — il manque un bras pour séparer l'échelle de la sélectivité

Le plan d'exécution identifie le confondant, et c'est une prise que je n'avais pas vue :

> « la boucle somme M mises à jour **non pondérées**, le parallèle une **combinaison convexe** (Σα = 1).
> À l'init α ≈ 1/M, donc ‖y−y0‖ chute d'un facteur ~M et `fw_o` reçoit une entrée M fois plus petite. »

La mitigation proposée — logger `‖y−y0‖` dans les deux bras — **diagnostique** le problème mais ne le
**sépare** pas. Si la compétence held chute, on saura que l'échelle a bougé, sans savoir si c'est elle qui
explique la chute.

**Bras à ajouter, coût nul** : `parallel_scaled`, où les poids sont `M·α` au lieu de `α`. La somme retrouve
la magnitude attendue de la version non pondérée à l'init, et **toute différence résiduelle est attribuable
à la sélectivité, pas à l'échelle**. Trois bras au lieu de deux, et le tableau devient interprétable :

| bras | Σ poids | ‖y−y0‖ à l'init | ce qu'isole la comparaison |
|---|---|---|---|
| séquentiel (baseline) | M (non pondéré) | référence | — |
| `parallel` | 1 | ÷ M | sélectivité **+** échelle |
| `parallel_scaled` | M | ≈ référence | **sélectivité seule** |

Le reste de E3 est solide, et l'astuce `alpha_i·(B_i z_i) = B_i(alpha_i z_i)` — pondérer en rang `r` puis
contracter en un seul einsum `"bmdr,bmtr->btd"` — est une vraie optimisation qui évite de matérialiser
`[B,M,T,d]`. Le test d'**invariance par permutation des slots** est le bon discriminant entre les deux
régimes ; c'est le test que j'aurais dû spécifier.

---

## D. `metrics.jsonl` — le correctif vaut la peine d'être fait

Le plan classe en « à signaler, pas à réparer en passant » le fait que `train.py:1889` ouvre
`metrics.jsonl` en mode `w`, sans `--resume` ni `--check`.

Je pousserais dans l'autre sens, **uniquement parce que l'étape 4 engage 2 × 9 h de calcul** pour restaurer
les checkpoints d'un preprint publié : un crash à h+8 perd tout, et une relance écrase les métriques du run
précédent. Le correctif minimal n'est pas de passer en mode `a` (ça mélangerait deux runs dans un fichier que
l'analyse suppose unique) mais de refuser d'écraser, ou d'horodater.

Trois lignes, aucun changement de comportement pour un run neuf, et 18 h-GPU assurées. À faire dans le même
passage que les correctifs `repro/run_all.sh`.

> **Note d'implémentation (2026-08-02)** : `train.py` n'a **aucun argparse** (`sys.argv[1]` et rien d'autre),
> donc le flag `--overwrite-metrics` de mon snippet n'est pas réalisable. Forme retenue : si le fichier
> existe et n'est pas vide, il est **renommé** avec l'horodatage de son mtime, puis on ouvre frais. Même
> effet, aucun flag, aucun changement pour un run neuf. Livré et vérifié en run réel.

---

## E. C1 — le verdict est juste, la nuance doit figurer dans le FINDINGS

Le contrôle a rendu son résultat et **il réfute mon soupçon**. J'en prends acte : le couplage
érosion → Δnll n'est pas mécanique, `codeexec` a un Δnll plat (+0,046 → +0,060 → +0,050) pendant que
`ic_ppl` est multiplié par 6, et l'argument des deux distributions (spécialisation en domaine contre érosion
hors domaine) est le bon.

**Une seule chose à ne pas laisser passer dans la rédaction** : `toolcall` fait +0,076 → +0,191, soit ×2,5,
pendant que `ic_ppl` fait ×5,9. Cette ligne-là, prise isolément, est compatible avec les deux explications.
Ce qui la tranche est ailleurs, et il faut l'écrire : **le bras ablaté s'améliore** (1,971 → 1,069). Une
érosion qui gonflerait artificiellement le Δ ferait *régresser* le bras ablaté. Il progresse — donc l'écart
qui se creuse est de la mémoire apprise, pas de l'hôte qui se dégrade.

Sans cette phrase, un relecteur s'arrêtera sur la ligne `toolcall` et la conclusion aura l'air d'une moyenne
qui masque un cas. Avec elle, c'est un argument.

---

## F. Une alarme qui dépasse cette campagne

> « le **seul** checkpoint du papier qui existe est `…dsv4w_s43/step_4000.pt` ; `dsv4m` et `dsv4w` graine 42
> sont introuvables, les `metrics.jsonl` aussi (⇒ Fig 3 irreproductible) »
> « `repro/run_all.sh:23` pointe un répertoire de configs déplacé — les 3 entraînements ET les `--cfg` des
> sondes échouent aujourd'hui »

**Le `repro/` d'un preprint publié avec DOI ne tourne pas depuis un clone frais.** C'est plus grave que tout
le reste de cette campagne, parce que c'est une revendication du papier (§11 Reproducibility) qui est
actuellement fausse.

Priorité proposée : le correctif `run_all.sh` + les `init_from` cassés passent **avant** tout le reste de
l'étape 0, et les deux relances de l'étape 4 sont à traiter comme une **réparation de dette publiée**, pas
comme un prérequis d'expérience. Une fois les checkpoints régénérés, ajouter leur empreinte (SHA + step +
nb de params) au `MODEL_CARD.md` pour que la prochaine disparition soit détectable.

> **Vérifications faites (2026-08-02)**, qui corrigent deux points ci-dessus :
> - Les `init_from` de `sub_dsv4y`/`sub_dsv4z` **ne sont pas cassés**. `steps: 4000` avec `save_every: 100`
>   produit bien `step_4000.pt` : le chemin `checkpoints/…_dsv4w_s43/step_4000.pt` est correct depuis un
>   clone frais. Il n'est insatisfait que sur la machine de dev, où le ckpt survivant vit sous
>   `/mnt/tb/checkpoints/archive/`. Les repointer aurait cassé la repro au lieu de la réparer.
> - **`MODEL_CARD.md` n'existe plus** — retiré délibérément du dépôt (commit `81a1c2d`, « un lecteur
>   reproduit des claims, pas mon pod »). Les empreintes vont dans `repro/README.md`. Et elles sont des
>   empreintes d'**intégrité de nos artefacts**, pas une validation du re-run d'un lecteur : l'entraînement
>   n'étant pas bit-reproductible d'une machine à l'autre, un re-run honnête produit légitimement un autre
>   SHA. C'est bien « détecter la disparition » qui est visé, et rien d'autre.

---

## Récapitulatif des modifications proposées au plan d'exécution

| # | Modification | Coût | Bloquant ? |
|---|---|---|---|
| 1 | Retirer E2 et E4.1 de `PLAN_EXPERIENCES.md`, avec la raison mesurée | nul | non |
| 2 | **Insérer E1.0** (plafond du readout à code oracle) **avant la vague ferme E1** | minutes | **oui** — conditionne la pertinence de 5 runs |
| 3 | Ajouter le bras `parallel_scaled` à E3 | nul | non, mais l'omettre rend E3 ininterprétable en cas de chute |
| 4 | Protéger `metrics.jsonl` contre l'écrasement | 3 lignes | recommandé avant les 2 × 9 h |
| 5 | Écrire la nuance `toolcall` dans le FINDINGS de C1 | rédaction | non |
| 6 | Traiter le `repro/` cassé comme une dette publiée, prioritaire | inclus étape 0 | non, mais à ne pas diluer |

Tout le reste du plan d'exécution est adopté sans réserve, y compris l'ordre, le périmètre exclu, et le
protocole ferme.
