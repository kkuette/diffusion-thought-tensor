# Plan d'expériences — thought-bank

**Destinataire** : agent de code opérant dans `github.com/kkuette/thought-bank`
**Rédigé le** : 2026-08-02
**Auteur du plan** : Claude (session d'analyse externe — papier v0.2, code `main` + `perf/decode-dispatch`, FINDINGS complet)
**Décideur** : Kkuette (Tony Denion). Toute dépense GPU non triviale demande son GO explicite.

---

## ERRATA — 2026-08-02, après vérification en dépôt

> Ce plan a été rédigé sur `origin/perf/decode-dispatch`, **8 commits en arrière du HEAD local**.
> Son auteur n'a vu ni `rti.py`, ni `rti_copy.py`, ni `rti_learner.py`, ni `rti_policy.py`, ni
> `recall_env.py` (~4 500 lignes), ni `SPEC_MEMOIRE_V2.md`, ni l'entrée FINDINGS du kill-test 3.
> Trois propositions reposaient sur des prémisses **mesurées fausses**. Le texte d'origine est
> conservé ci-dessous pour la provenance ; il ne doit pas être exécuté tel quel.
>
> **Le plan d'exécution prévaut sur ce document partout où ils divergent** (décision de l'auteur,
> addendum du 2026-08-02).

| Section | Statut | Raison mesurée |
|---|---|---|
| **§2 E2** — citation gratuite par injection post-norm | **RETIRÉ** | `PointerReadout` ([toy_read_lab.py:832](deepseek_v4_mini/toy_read_lab.py:832)) implémente déjà l'identité exacte `logits = (x + g·scale·sel) @ Eᵀ`, et il est **mort** : 12 runs r3, grade held-out ≤ 0,100 contre 0,281 pour l'injection native. Le « facteur 42 » chiffrait une économie contre une tête `d_m × V` **qui n'existe nulle part dans le dépôt** ; le vrai `rti_copy` coûte 0,61 M MACs/token, soit **moins que l'homme de paille construit pour la comparaison**. |
| **§2 E4, point 1** — `val_mask` généralisé | **RETIRÉ** (la mesure survit, pas le raisonnement) | La prémisse « fréquence marginale élevée ⇒ les chiffres ne sont jamais sélectionnés ⇒ jamais citables » est fausse : inclusion pleine `numeric` **1.000**, même au `a` du teacher. Le top-k est un classement **intra-tour** — le rang décide, pas le poids absolu. Le vrai bug était **`k` fixe contre longueur de tour** (`code`, T=35, budget 13 : nom et constante évincés). Correctif livré le 2026-08-01, cf. FINDINGS. |
| **§2 E1, point 1** — rotation « façon RoPE » | **RETIRÉ** | Le binding DFT déjà en place sépare **8/8** positions contre **3/8** pour la forme RoPE littérale, et `pos_offset` ne change rien (97,4 → 97,5 %). La proposition est une régression par rapport à l'existant. Ce qui survit de E1 : l'axe **indice d'écriture absolu**, qui n'a pas d'antécédent — cf. plan d'exécution §2c. |

**Ce qui survit intact** : **E0** (l'oracle d'expressivité, confirmé comme pivot de tout le reste),
**C1** (qui a déjà rendu son verdict, et qui **réfute** le soupçon de son auteur), **E3**, **E5**, la
discipline de rapport et les règles méthodologiques du §1 — dont l'une, « un grade composite à 0 ne
localise rien », a servi à dédoubler la sonde E1.0 du plan d'exécution.

---

## 0. Lis ça en entier avant de toucher au code

### Ce qui existe et qui est acquis
Le dépôt contient un **preprint publié** (tag `V0.2.2-preprint`, DOI 10.5281/zenodo.21225721) et une campagne
expérimentale de trois semaines documentée dans `FINDINGS.md` (2 530 lignes) et `EXPERIMENTS.md`.
**Ne re-démontre rien de ce qui suit.**

- Le **Thought Bank** : banque FIFO lue **comme des poids** — chaque slot est développé par un hypernet en
  couche MLP low-rank, appliquée **séquentiellement** sur les M slots (`model.py::DualModalBlock._cross_modal`).
- Papier (3,08 M, d=128, 8 slots × 32 d, r=16) : binding d'une règle jamais entraînée à **0,79–1,00**
  (hasard 0,008) · TTT ne transfère **rien** à 138× le coût · **la politique mémoire est un comportement
  entraîné** (persévération 1,000 en structure fixe → 0,000 en structure randomisée).
- Mécanisme : **superposition redondante** — les 8 slots convergent vers ~le même vecteur (rang effectif
  1,13/8), donc évincer un slot retire une copie, pas le contenu.
- Le **rang-1 par produit externe n'a jamais dépassé le hasard** sur l'application de règle. La non-linéarité
  entre slots est nécessaire. **La séquentialité, elle, n'a jamais été testée.**
- 350 M / 97 M sur données réelles : d=768, L=12, `mem_dim=512`, `mem_read_rank=8`, SwiGLU, embeddings **liés**
  (`lm_head.weight = embed.weight`).

### La question ouverte, en une phrase
Le read fast-weight sait **appliquer** une règle. Il ne sait pas **citer** une valeur — c'est le mur
copie-argmax du rappel persona : canal ouvert en Δnll (jusqu'à +0,33), argmax fermé (grade 0,00, même sur le
split train). `toy_read_lab.py` existe précisément pour isoler cette question, avec un **write oracle**.

L'objectif de ce plan : déterminer si un **canal unique** peut porter les deux usages, sans ajouter un
second mécanisme.

### Ce qu'il ne faut PAS faire
1. **Ne pas brancher un second mécanisme** (tête pointeur indépendante, branche cross-attention parallèle)
   comme solution par défaut. Le journal établit que chaque capacité demande sa propre pression
   d'entraînement : deux mécanismes = deux politiques à installer = deux points fixes à briser.
2. **Ne pas attaquer la « page morte » par la capacité ou la profondeur.** Quatre tentatives ont échoué
   (entrées 07-13(3), 07-14, 07-14(2), 07-14(3)). Le symptôme — « contribue en milli-nats mais n'est jamais
   adressée » — est un problème de **séparation**, pas de taille.
3. **Ne pas créer de nouvelle tâche ni de nouvelle branche de travail.** Chaque expérience est un **drop-in
   sur le benchmark existant** pour que la comparaison reste exacte.
4. **Ne pas lancer d'entraînement sans GO.** E0 et E2 sont autonomes. E1, E3, E4, E5 coûtent des heures GPU.

---

## 1. Règles méthodologiques non négociables

Elles viennent du journal. Ce sont des leçons déjà payées ; ne les réapprends pas.

| Règle | Origine | Application |
|---|---|---|
| **Un grade composite à 0 ne localise rien.** | 07-27 : `grade_calls` = porte dure sur le nom × F1 des args affichait 0,00 aux 8 paliers, alors que le nom était bon 33 % du temps. Un mur a été déclaré à tort. | Toujours re-décoder en **séparant les termes** (nom seul / nom dans le menu / args) avant de conclure qu'un canal est fermé. |
| **La conversion nll → comportement se paie au decay.** | 07-26 et 07-27 : rien ne bouge à LR plein, tout se consolide pendant la décroissance ; le saut est à ×0,2 du LR de croisière. | **Aucune éval de mi-run à LR plein ne permet de conclure.** Utiliser l'escalier de LR (`wsd_decay_shape: stair`) comme **instrument de mesure** : chaque palier reçoit ses évals. |
| **Δnll est une différence : un meilleur hôte tire les deux bras.** | 07-26 : à step 750, codeexec nll 0,570 / ablaté 0,622 — le bras sans mémoire est déjà quasi parfait, il ne reste rien à gagner. | Ne jamais interpréter un Δnll sans reporter la **nll absolue des deux bras**. |
| **Toujours un bras ablaté.** | Partout dans le papier et le journal. | Une métrique sans bras ablaté n'est pas une mesure de mémoire. Le bras ablaté doit être **au plancher**. |
| **Annoncer explicitement ce qui n'est pas ablaté.** | 07-27 : « trois changements décidés ensemble et donc non ablatés ». | Si plusieurs variables bougent dans un run, l'écrire dans le rapport. Ne pas attribuer. |
| **Rang de banque faible ≠ pathologie.** | Papier §6 : rang effectif 1,13/8 avec écart d'ablation +4,6 nats. | Le rang doit être lu **conjointement avec l'écart d'ablation**. Ne pas diagnostiquer un effondrement représentationnel sur le rang seul. |

---

## 2. Les expériences

### E0 — Oracle d'expressivité `[AUTONOME · minutes · à faire en premier]`

**Question unique.** Le goulot est-il le **write** (son amortisation) ou le **read** (sa classe de fonctions) ?

**Principe.** Sur un checkpoint **totalement gelé**, on optimise directement le code de slot par descente de
gradient. Le résultat est la **borne supérieure** de ce que n'importe quel schéma d'écriture peut atteindre
avec ce read.

```
m* = argmin_m  CE(segment | banque_{t-1} ∪ {m}) + λ‖m‖²
```

Tout est gelé sauf `m` : **32 nombres** au 3 M, 512 au 350 M. Aucun poids n'est touché.

**Protocole.**
1. Charger le checkpoint du papier (seeds 42 et 43, cf. `repro/run_all.sh`).
2. Reprendre le protocole d'évaluation de `repro/` **à l'identique** — mêmes conversations, même split held,
   même mesure d'accuracy sur requêtes inédites.
3. Remplacer l'appel au write par : un slot initialisé (aléatoire ou à la sortie du write entraîné),
   `requires_grad=True`, puis K≈20–50 pas d'Adam sur la CE du segment de présentation. Le trunk, le hypernet
   et tous les autres slots restent gelés.
4. Évaluer les requêtes comme d'habitude, avec le slot optimisé en place.

**Quatre conditions, et la troisième est celle qui compte.**

| Condition | Lecture d'un résultat élevé | Lecture d'un résultat au hasard |
|---|---|---|
| Règles entraînées | sanité — doit être ≥ le write entraîné | bug de protocole, ne pas continuer |
| Règles held (aujourd'hui 0,79–1,00) | le write laisse de la marge → **E5 (P1) devient prioritaire** | le write est déjà optimal → **E1/E3 (le read) deviennent prioritaires** |
| **Soustraction** `y=(x−s) mod 128`, hors famille (papier §7 : hasard pour tous les bras) | **l'enveloppe était l'amortisation du write, pas la classe de fonctions du read** → §7 du papier est à réécrire, et c'est un résultat en soi | la frontière est bien le read — la revendication du papier se renforce **au niveau oracle**, ce qui est le niveau le plus fort possible |
| Rappel persona, **citation** (argmax, pas Δnll) | le code optimal permet de citer → le mur est le write | **le mur est la classe de fonctions du read** → la refonte du canal unique est justifiée par la mesure |

**Critère de succès de l'expérience elle-même** : la ligne « règles entraînées » est ≥ le write entraîné.
Sinon, le protocole est faux.

**Coût.** 3 M, segments de 13 tokens : ~5 GFLOPs par conversation, **quelques minutes** pour tout le
protocole. Sur le rappel persona au 350 M, compter 1 à 2 h GPU — demander le GO.

**Ce que ça débloque.** Tout le reste. E0 dit s'il faut investir dans le write ou dans le read, et la
quatrième ligne justifie (ou non) la refonte architecturale par une mesure au lieu d'une intuition.

**Livrable.** Une entrée `FINDINGS.md` au format du dépôt + le script sous `analysis/` avec sa commande de
reproduction.

---

### E1 — `toy_read_lab` phase 9 : liaison positionnelle + top-k `[GO REQUIS · heures GPU]`

**Question unique.** Une somme de lignes **liées par rotation** préserve-t-elle la séparabilité comme la
théorie le prédit, et permet-elle de citer ?

**Contexte.** `toy_read_lab.py` teste déjà quatre classes de read avec **write oracle**, dont `r3` : banque en
espace d'embeddings, `mem_dim` forcé à `d_model`, pointer readout nu (`biais = s_t @ embed^T`). La phase 6 a
fait passer la banque à une **FIFO de groupes de lignes natives** (le write sélectionne, il ne transforme
plus) et la phase 8 a remplacé la sélection oracle par un **retriever appris**.

**Ce qu'il faut ajouter.**

1. **Rotation par indice d'écriture absolu.** `[la forme « façon RoPE » est RETIRÉE — voir ERRATA ;
   l'axe indice d'écriture, lui, survit]` Chaque ligne est tournée par `R(s)` où `s` est son compteur
   d'écriture absolu (rotation par blocs de 2 dimensions, mécanique RoPE). La requête est tournée par `R(t)`,
   `t` = compteur courant. Alors `⟨R_t q, R_s row⟩ = ⟨q, R_{t−s} row⟩` — le score ne dépend que de
   **l'ancienneté**, sans un seul paramètre.
   *Détail qui compte pour un FIFO en anneau : la rotation étant absolue, les lignes n'ont **jamais** besoin
   d'être re-tournées quand l'anneau tourne. C'est la requête qui bouge.*
2. **Vecteurs de rôle** `ρ` (règle / clé / valeur) multipliés élément par élément avant la somme, pour que
   les deux usages occupent des sous-espaces séparables dans le même vecteur.
3. **Top-k avant la somme.** `k` est un hyperparamètre à balayer.

**Balayage obligatoire.** `k ∈ {16, 32, 64, 128, 256}`, mesurer la **fidélité de citation** et la comparer à
la courbe théorique `cos ≈ 1/√(1 + k/d_m)` :

| k | 16 | 32 | 64 | 128 | 256 | 512 |
|---|---|---|---|---|---|---|
| fidélité théorique (d_m=512) | 0,986 | 0,971 | 0,944 | 0,895 | 0,817 | 0,707 |

**C'est l'écart entre mesure et théorie qui est le résultat**, pas la valeur absolue. Si la mesure suit la
courbe, la liaison fait son travail et la borne est bien la superposition. Si elle est nettement en dessous,
la liaison ne sépare pas — chercher pourquoi (rotations trop corrélées, rôles qui ne séparent pas, cleanup
absent).

**Critère d'arrêt.** Si à k=16 la citation est déjà au plancher alors que le write est oracle, le problème
n'est pas la superposition : c'est le **cleanup**. Passer directement au point suivant.

**Le cleanup est obligatoire — ne pas l'oublier.** Une superposition sans mécanisme de nettoyage donne du
bruit. Dans ce montage, le dictionnaire de nettoyage naturel est la matrice d'embedding `E`, et c'est
exactement ce que fait `r3`. **`r3` est le cleanup.** Si `r3` ne fonctionne pas, la déliaison n'a pas de
cible et il faut un codebook appris — le noter comme un résultat, pas comme un échec d'implémentation.

**Ce que ça débloque.** La faisabilité du canal unique, mesurée dans l'instrument le plus contrôlé du dépôt
(write oracle, backbone vanilla sans mHC ni MoE ni CSA — si le mur tombe là, ce n'est pas l'archi exotique
qui le tenait).

---

### E2 — Canal citation gratuit par injection post-norm `[RETIRÉ — voir ERRATA]`

**Question unique.** Peut-on obtenir le biais pointeur sans tête dédiée ?

**Constat.** `model.py` a les **embeddings liés** :
```python
self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
self.lm_head.weight = self.embed.weight     # weight tying
```
L'espace d'entrée et l'espace de sortie sont donc **le même**. Ajouter un vecteur de contenu `c` au residual
stream **juste après `norm_out` et avant `lm_head`** produit exactement le biais `c·Eᵀ`, en réutilisant le
matmul de la tête.

```python
H_text = self.norm_out(H_text)
H_text = H_text + gate * proj_mem(c)        # ← injection ; proj_mem : d_m → d_model
logits = self.lm_head(H_text)               # contient désormais le biais pointeur, gratuitement
```

**Coût comparé, au 350 M (V=32 000, d=768, d_m=512).**

| Implémentation | MACs/token | % du modèle |
|---|---|---|
| tête pointeur séparée (`d_m × V`) | 16,38 M | 4,68 % |
| **injection post-norm** (projection `d_m → d`, `lm_head` réutilisée) | **0,39 M** | **0,11 %** |

**Facteur 42, résultat mathématiquement identique.**

**Vérification à produire.** Un test qui montre l'équivalence numérique entre les deux implémentations sur un
batch fixe (tolérance flottante). Si l'écart est non trivial, c'est la RMSNorm — injecter **après** `norm_out`,
pas avant, et le documenter.

**Nota bene à ne pas rater.** Ni l'injection dans le residual stream ni le biais de logits **n'ajoutent de
terme en O(n²)**. Le coût de l'attention est `n²·d` : il dépend du **nombre** de vecteurs et de leur largeur,
jamais de leur contenu. Ajouter la lecture mémoire n'ajoute aucune paire à attendre.

---

### E3 — Read parallèle (P2a) `[GO REQUIS · un run du benchmark papier, ~5 h sur 3090]`

**Question unique.** La composition **séquentielle** sur les slots fait-elle un travail que la
non-linéarité seule ne fait pas ?

**Modification.** Dans `DualModalBlock._cross_modal`, remplacer la boucle

```python
for i in range(M):
    z = act(einsum(A[:, i, 0], y))
    y = y + drop(einsum(Bm[:, i], z))       # ← chaîne de longueur M
```

par une somme pondérée parallèle :

```python
alpha = softmax(q @ (W_k @ bank).T / sqrt(d_k))          # (T, M) — l'adressage
y = y + sum_i alpha[:, i] * B_i @ act(A_i @ y)           # un einsum batché sur i
```

**Mêmes FLOPs, mêmes matrices, même hypernet, même non-linéarité.** La seule différence est que les M branches
deviennent indépendantes, et qu'elles sont pondérées par une récupération.

**Trois livrables, même run.**
1. **Compétence held** contre la baseline du papier, deux graines (42 et 43). Comparaison exacte, le
   benchmark ne change pas.
2. **`α` — le signal d'adressage**, qui n'existait pas. Reporter son **entropie normalisée par `log M`**.
   *C'est le diagnostic de distraction qui manque à la ligne page.* Proche de 1 ⇒ la tête n'adresse rien.
3. **Rang effectif de la banque en fin de conversation**, à comparer au 1,13/1,48 sur 8 du papier. Avec une
   pondération par récupération, la redondance n'est plus le seul moyen de survivre à l'éviction : le rang
   devrait **monter**. S'il ne monte pas, c'est une information.

**Critère d'arrêt.** Si la compétence held chute de plus de quelques points, la séquentialité fait un travail
réel — le noter et abandonner la direction « une seule itération ».

**Indice pré-existant qui va dans le sens de P2a.** Le papier §9 rapporte que la composition **interne**
`f(f(x))` est au hasard alors que le chaînage externe est à 0,961. Si les M couches séquentielles ne
délivrent pas la composition, leur justification principale est faible.

---

### E4 — Critère de sélection : `val_mask` généralisé + diversité `[GO REQUIS]`

**Question unique.** Comment sélectionner les G lignes à écrire sans perdre les classes citables fréquentes
(chiffres, identifiants, littéraux) ?

**Ce qui est déjà tranché — ne pas le refaire.** Décision du 21/07, verdict `analysis/freq_vs_surp*` : le
poids SIF `a/(a+p(token))` sur table unigram auto **bat la surprise nll² sur les deux axes** (persona et mix
14 sources), sans modèle de référence et **borné** (typos plafonnés).
Voir `persona_chat_data.py::_sif_table` et `toy_read_lab.py::sif_weight_table`.

**Mais ce verdict porte sur un poids de *pooling*, pas sur une *sélection*.** La raison de la victoire de SIF
ne transfère pas :

| | Pondération pour un pooling | Sélection de G lignes |
|---|---|---|
| Score aberrant | un typo à surprise 20 **empoisonne la moyenne** → un score borné est décisif | il occupe **un slot**, rien de plus. Le top-k est intrinsèquement borné |
| Ce qu'on garde | le signal central, robuste | **les exceptions** — ce qui ne se régénère pas |
| Chiffres | tolérable, dilués mais présents | **fatal** : fréquence marginale élevée ⇒ jamais sélectionnés ⇒ jamais citables |

Le bon critère n'est pas *« est-ce rare ? »* mais **« est-ce régénérable plus tard ? »**.

**La parade existe déjà dans le dépôt.** Le `val_mask` de `sft_sota_350m_valsif_stair` — « cible discriminante
= span des noms d'outils déclarés, repli pooling SIF partout ailleurs » — **est** la protection des classes
citables. Et c'est le run où la sélection s'est ouverte (10/30 contre 0/30 ablaté).

**À tester, du moins cher au plus cher.**
1. **`val_mask` généralisé** `[RETIRÉ — voir ERRATA : la mesure a été livrée le 2026-08-01, mais le
   raisonnement ci-dessous est réfuté ; le bug était k fixe vs longueur de tour, pas la fréquence]` :
   étendre des noms d'outils à toute classe citable — chiffres, identifiants,
   noms propres, littéraux de code. Un masque, pas un nouveau mécanisme.
2. **Winsoriser la surprise** au lieu de l'écarter : plafonner le score importe la vertu de bornage de SIF
   dans un critère contextuel. Une ligne de code.
3. **Erreur de prédiction contre la banque** plutôt que surprise brute : une ligne mérite l'écriture si elle
   n'est *pas déjà* récupérable depuis la banque. Déduplique par construction, gratuit si la lecture est déjà
   calculée. (C'est le critère de HOLA/HAM.)
4. **Diversité entre lignes retenues** (MMR : score − max de similarité aux déjà choisies).
   **Ce n'est pas une heuristique de compression.** La relecture d'une superposition s'écrit
   `S k_j = v_j + Σ_{i≠j} v_i ⟨k_i, k_j⟩` : le terme d'interférence **est exactement** la somme des produits
   scalaires croisés entre les lignes sélectionnées. Choisir des lignes peu corrélées minimise directement le
   bruit de relecture. La sélection par redondance et la capacité de superposition sont le même problème.

**Métrique.** Taux de citation par classe (chiffres / noms d'outils / prose), séparément. Une moyenne globale
masquerait exactement l'effet recherché.

---

### E5 — Le write comme argmin local (P1) `[GO REQUIS · reformulation]`

**Question unique.** Le point fixe *ignore-the-bank* disparaît-il si le write résout un objectif local ?

**Diagnostic.** Le cadre qui unifie la littérature mémoire est la régression au temps de test :
`M_t = argmin_M ℓ(M; k_t, v_t) + Ret(M, M_{t−1})`. L'attention linéaire est **un pas** de descente sur des
moindres carrés pondérés, la règle delta sa version streaming, Titans y ajoute du momentum. Toutes
**résolvent un objectif**, et c'est pourquoi aucune n'a besoin de bootstrap.

Le write actuel est `m = W_w · mean_pool(H_text)` — une projection linéaire d'un pooling. Il ne résout rien.
Son sens lui vient entièrement du gradient qui remonte par TBPTT. **Hypothèse : le teacher Fourier est la
prothèse de cet argmin absent.**

**Deux niveaux.**
- **E5a — teacher amortisé.** Remplacer le teacher Fourier par `m*` (celui de E0) recalculé en ligne. Même
  mécanique — blend annealé + distillation cosinus — mais la cible n'encode **aucune connaissance de domaine**.
  *Ce que ça achète* : la limitation « teacher spécifique » du papier (« le teacher est un code de Fourier, la
  géométrie naturelle de cette famille ; la généralité au-delà de la structure circulaire est non testée »)
  disparaît. Ne calculer `m*` que pendant la fenêtre de blend et sur ~10 % des segments.
- **E5b — write inférentiel.** La tête d'écriture ne produit plus le code mais son **initialisation** ; K pas
  de descente (K = 3 à 10) produisent le code écrit. C'est Titans, mais sur un code de 32–512 dimensions au
  lieu des poids d'un MLP.

**Critère décisif de E5b** : lancer **sans teacher, sans curriculum, sans anneal**, en drop-in sur le
benchmark du papier. Si le point fixe ignore-bank ne réapparaît pas, l'hypothèse est confirmée et une part
de « rien n'émerge » était « cette greffe-là était dure à entraîner ». Si le point fixe réapparaît,
**l'hypothèse est fausse — la retirer proprement du plan et le documenter**.

**Honnêteté à préserver.** E5b rend le write dépendant de K et η, et il cesse d'être « un seul forward pass »
— qui est une revendication du papier. La revendication devient « **pas de backward sur les poids** », ce qui
reste la distinction qui compte face au TTT, mais c'est une reformulation à assumer explicitement dans tout
écrit.

---

## 3. Contrôle transversal à ajouter quoi qu'il arrive

### C1 — Découpler Δnll de l'érosion de l'hôte `[AUTONOME · checkpoints existants]`

**Le problème.** Deux observations du journal se combinent mal.
- **07-27** : l'érosion hôte a doublé — `ic_ppl` codeparrot 19,5 → 115,7, fineweb 141 → 663. Différé
  (« l'érosion est un problème pour plus tard »).
- **07-26** : « Δnll est une différence, et un meilleur hôte tire les DEUX bras. À step 750, codeexec est à
  nll 0,570 / ablaté 0,622 : le bras sans mémoire est déjà quasi parfait, il ne reste rien à gagner. »

Mis ensemble : **plus l'hôte se dégrade, plus le Δnll a de place pour croître.** L'indicateur principal de
santé mémoire est mécaniquement favorisé par la dégradation qui a été différée.

**Ce n'est pas une accusation de résultat** — la sélection 10/30 contre 0/30 ablaté est une mesure de
comportement, pas de Δnll, et elle tient. Mais c'est exactement le couplage qu'un relecteur cherche.

**Le contrôle.** Tracer **Δnll contre `ic_ppl`** à travers les checkpoints (ils existent tous les 100 steps).
Une figure. Si Δnll croît quand l'hôte se dégrade, la courbe le dit et le récit est sous contrôle.
Coût : quelques minutes de CPU/GPU sur des checkpoints déjà écrits.

---

## 4. Ordre et dépendances

```
E0  (autonome, minutes)  ─────────────┬──► si le write laisse de la marge ──► E5
      oracle d'expressivité           │
                                      └──► si le read plafonne ────────────► E1, E3
E2  (autonome, minutes)  ──────────────────► prérequis d'implémentation de E1
      citation gratuite

C1  (autonome, minutes)  ──────────────────► indépendant, à faire tôt

E1  (GO) ──► si la liaison sépare ──────────► E4  ──► intégration 350 M
E3  (GO) ──► indépendant de E1, livre α ────► diagnostic page (5e hypothèse)
```

**À faire en premier, dans cet ordre, sans attendre de GO** : **E0**, puis **E2**, puis **C1**.
Les trois sont autonomes, coûtent des minutes, et E0 conditionne tout le reste.

---

## 5. Format de rapport attendu

Une entrée en tête de `FINDINGS.md`, **au format déjà en usage dans le dépôt** :

```markdown
## AAAA-MM-JJ — <titre : le résultat, pas le sujet>

**TL;DR.** <2-3 phrases, le chiffre qui porte>

### Setup
<config, checkpoint d'init, steps, matériel, durée réelle>
<liste explicite des variables qui bougent ensemble et ne sont donc PAS ablatées>

### Verdict
<tableau : métrique | bras mémoire | bras ablaté | baseline>

### Attribution honnête, et ce qui casse
<ce qui est attribuable à quoi ; ce qui a régressé ; ce qui n'est plus une sonde valide>

### Reproduire
<commande exacte depuis un clone frais>
```

**Exigences minimales de tout rapport.**
- Un **bras ablaté**, et il doit être au plancher.
- Les **nll absolues des deux bras**, pas seulement le Δ.
- Les **termes séparés** de toute métrique composite.
- Une **commande de reproduction** exacte.
- Ce qui **n'a pas** été ablaté, écrit noir sur blanc.

---

## 6. Références utiles (vérifiées sur arXiv)

| Sujet | Référence |
|---|---|
| Cadre unifiant (régression au temps de test) | arXiv 2501.12352 · Miras arXiv 2504.13173 |
| Le transformer comme DNC **sans état** — identifie « politiques d'écriture apprises, effacement, liens temporels » comme ce qui lui manque. **Formalise la contribution n°3 du papier.** | arXiv 2603.19272 |
| Token émis par le modèle déclenchant une mémoire latente (sans persistance) | VisMem, arXiv 2511.11007 |
| Banque latente persistante inter-sessions, cross-attention insérée *(auteur unique, non reproduit — antériorité conceptuelle seulement)* | arXiv 2603.22329 |
| Seuils de capacité `d² ≍ n log n` pour la récupération top-1 en mémoire linéaire | arXiv 2605.05189 |
| Context Saturation Gap — la plupart des benchmarks mémoire tiennent dans les fenêtres modernes | arXiv 2602.19320 |
| Profondeur de couche pour la récupération : la dernière couche est un mauvais choix | arXiv 2502.02013 (ICML 2025) · arXiv 2605.23033 (ICML 2026) |
| Alignement cross-modal à 0,3–0,5 de profondeur normalisée | arXiv 2606.03871 |
| Effondrement de la CoT latente sur traces longues — **objection à anticiper** contre la conjecture « banque = scratchpad latent » du §9. Réponse : la banque **n'est pas une chaîne**, une entrée est écrite puis relue, jamais réinjectée comme entrée de sa propre production. | arXiv 2607.16972 |
