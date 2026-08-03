# SPEC mémoire v2 — recentrage validé 2026-07-31

Statut : **VALIDÉ** (discussion 3 tours + critique adversariale, verdicts intégrés).
Companion : [EXPERIMENTS.md](EXPERIMENTS.md) (arbre), [FINDINGS.md](FINDINGS.md) (journal).
Remplace comme cap de phase : l'arc recall/RTI en tant que *fin* — le RTI devient l'aile
« citation » d'un mécanisme à deux régimes, pas le claim.

---

## 1. Le claim (v2)

> **À fenêtre de contexte bornée et appariée, un modèle avec banque (écriture en ligne,
> écrasement de slots, hors contexte) maintient le conditionnement comportemental ET le
> rappel exact à travers des resets de fenêtre, à coût de contexte nul — contre la
> compaction textuelle (qui paie son résumé en tokens de fenêtre) et le RAG sur transcript
> (qui paie son retrieval), le tout apparié en forwards totaux.**

Comptabilité — règles non négociables :

- L'appariement se fait en **forwards totaux** (ou FLOPs). Jamais en « tokens visibles » :
  cette comptabilité exclut le compute caché (writes, cascade, pseudo-tokens injectés)
  et rend le claim tautologique. Verdict de la critique du 07-31, accepté.
- La **fenêtre est bornée et identique** pour tous les bras. L'avantage revendiqué de la
  banque est structurel : elle est *hors* de la fenêtre. La compaction porte son résumé
  dans la fenêtre suivante (et le paie en tokens de fenêtre) ; le RAG paie ses tokens de
  retrieval ; la banque paie 0 token de fenêtre.
- Le **token RESET** ([mémoire projet : primitive du 07-27]) passe d'extension future à
  cœur du protocole : une « session » = plusieurs fenêtres séparées par des resets, la
  banque est le seul état qui traverse. Invariant conservé : la seule modification de la
  banque est l'écriture d'un write (désormais : sur un slot, avec écrasement).

Ce qu'on ne revendique **plus** (dégonflages actés) :

- ~~« Le modèle pense dans le latent »~~ — horizon, pas claim. Le curriculum Coconut est
  **abandonné** pour cette phase : (a) Coconut perd contre la CoT explicite à compute
  apparié (précédent publié), (b) il affame la sélection de surface en supprimant les
  tokens qu'elle échantillonne (contradiction interne relevée par la critique).
  Le bloc `<think>` reste **verbalisé**.
- ~~« Le gist encode un état de pensée »~~ — le contenu du gist n'a pas de canal
  d'apprentissage praticable à notre échelle (TBPTT inter-tours infaisable sur 24 Go ;
  un scalaire RL ne façonne pas un vecteur mem_dim). Le gist est une **compression SIF
  assumée comme telle**. Le RL apprend *quand* écrire et *quel slot* écraser — jamais
  *quoi* mettre dans le vecteur.
- ~~Le rappel exact comme claim de *performance* contre le RAG~~ — le RAG résout le
  verbatim mieux et moins cher en régime fenêtre non bornée. Le rappel est un claim de
  **mécanisme unifié sous fenêtre bornée** (même store qui conditionne et cite à travers
  les resets), démontré à budget total incluant les tokens de retrieval du RAG.

## 2. Le mécanisme

Un seul store, un seul geste d'écriture, deux régimes de lecture.

### 2.1 Store : espace fixe de S slots

- S fixe (cohérent avec `mem_seed_slots = max_mem` décidé sur perf/decode-dispatch ;
  le rebind CUDA-graphs suppose une banque de taille constante).
- S est un hyperparamètre **à sonder** (kill-test 8 : courbe de dilution S = 1/4/16/64) —
  deux pressions opposées : le rappel veut S grand, la lecture ambiante se dilue.

### 2.2 Slot structuré

```
slot = (clé, surface, gist)
```

- **clé** : pooling SIF procédural du segment (existant, `build_group` dans
  [rti.py](deepseek_v4_mini/rti.py)) — avec la **correction de pondération** du §3.1.
- **surface** : sélection top-k d'**embeddings natifs** de tokens réellement émis
  (jamais transformés par un réseau — labo jouet ph.7 : tout readout appris est mort,
  seule l'injection native cite). Matière première = le bloc `<think>` verbalisé.
- **gist** : compression SIF des états cachés du segment. Consommé par la lecture
  ambiante uniquement. Aucune promesse sur son contenu au-delà de ce que le teacher
  SIF a=1e-4 y met.

### 2.3 Écriture = choix de slot + écrasement

Ajouter et oublier sont le même geste : écrire sur un slot détruit son contenu
(surface + gist ensemble).

- **Amorçage : FIFO** (écraser le plus vieux). Politique dégénérée assumée, gratuite.
- **Cible : supersession apprise** (écraser le slot contredit), au GRPO — mais seulement
  après que le kill-test 6 a montré que le signal existe (§4).
- Actions RL existantes conservées : write Bernoulli, retrieve Plackett-Luce
  ([rti_policy.py](deepseek_v4_mini/rti_policy.py)) ; s'ajoute une tête de **choix de
  slot** (pointeur d'écrasement, initialisé FIFO).
- Piège documenté : l'amorçage FIFO installe un prior de récence que la supersession
  devra désapprendre (dsv4mini switch task : ces politiques ne s'apprennent pas « en
  passant »). D'où la sonde anti-récence obligatoire (§5.3).

### 2.4 Budget de la banque et points de prélèvement (ancrages 08-01 soir)

**Ancrage cross-modal** : la banque est une modalité — cible long-terme = vision/son/
texte écrivant dans le même mem_dim via des encodeurs de write par modalité. Pas un
chantier courant, mais aucun choix ne doit fermer cette voie (l'injection-comme-
modalité y est compatible ; seule la voie surface/pointeur est spécifique au texte,
la seule modalité où le verbatim existe).

**Budget** : trois cadrans, à ne tourner qu'après localisation de la perte —
mem_dim (512→d_model, déjà ranké phase 1), rang r du read fast-weight (l'expressivité
de la lecture peut plafonner avant la capacité du stockage), et **une MATRICE par write**
(précision user 08-01, affinée) : le write rend G ∈ R^{m×d} — m vecteurs de PLEINE
largeur d_model, la projection 768→512 disparaît. « mem_dim » se réinterprète : la
largeur est fixée à d (l'intégrité du vecteur de base est conservée), le cadran de
budget devient m (combien de vecteurs par write) — la seule compression restante est
la SÉLECTION/le pooling en m têtes, plus aucune perte en largeur. Conséquences :
(a) le slot devient HOMOGÈNE avec la surface — les deux composantes sont des listes
de vecteurs d-dim, seule change leur provenance (embeddings couche 0 vs états
mi-tardifs) — un seul format de slot, et les lignes de gist deviennent elles-mêmes
injectables comme pseudo-tokens — **l'INJECTION-COMME-MODALITÉ, À TESTER (user
08-01)** : une seule lecture (tout passe par le préfixe, le fast-weight devient
optionnel) serait la banque-comme-modalité au sens plein. Échelle de test :
(1) labo jouet d'abord — la ph.9 a montré que l'injection d'états post-norm CITE
à 0.434 from-scratch (vs 0.708 natif) ; la mesure manquante est le
CONDITIONNEMENT (Δnll contrastif avec lignes de gist injectées vs fast-weight vs
sans mémoire, from-scratch) ; (2) si le jouet passe : bras SFT 350M avec un
type-vec dédié « gist » (comme rti_type.vec) injectant m lignes mi-tardives à
côté des groupes surface, sonde persona Δnll ; (3) verdict phase 1. Piège connu :
les états mi-stack au préfixe couche 0 sont OOD pour un modèle gelé — le test
n'a de sens qu'entraîné (jouet from-scratch ou SFT avec type-vec).
(b) côté read fast-weight, chaque ligne = un sous-slot d'entrée d (hypernet
fw_A/fw_B passe de mem_dim→… à d→…) — trois agrégations candidates :
séquentielle m×S (l'actuelle, coût ×m), sélection au read, et **VUE PLATE +
ATTENTION (user 08-01, design retenu à tester — mécanique précisée)** :
le stockage reste le tenseur **(max_mem, mem_dim, d)** (mem_dim = lignes par
slot, pleine largeur d) ; à la lecture, une simple **VUE (max_mem·mem_dim, d)**
— reshape zéro-copie, l'axe slot étendu sur l'axe lignes, **AUCUNE
multiplication de matrices dédiée au read** (pas d'hypernet, pas de projection ;
la rotation par âge est élémentaire type RoPE, recalculée au write, cohérente
avec l'écrasement). Les lignes entrent dans l'attention telles quelles — seules
les projections K/V propres de la couche s'appliquent, comme à n'importe quel
token. Lignes séparées : zéro interférence, récence portée par la rotation.
Sous-choix d'implémentation à trancher (ph.10 / bras SFT) : (α) lignes en
pseudo-tokens à l'ENTRÉE (traversent le stack — requis pour la SURFACE, dont le
pointeur copy-head a besoin des positions au préfixe) vs (β) lignes appondues
en K/V aux couches lectrices (pas de ré-encodage — naturel pour le GIST, dont
les lignes sont déjà des états mi-tardifs) ; le mapping surface→(α), gist→(β)
recoupe exactement le principe de séparation. Coût honnête : max_mem·mem_dim
lignes dans l'attention = LE cadran de coût du gist, à budgéter comme le
préfixe surface. Garde-fou restant : autre classe de fonctions que le
fast-weight ⇒ A/B apparié (jouet ph.10, rotation par âge en variante) avant
d'adopter ;
(c) la sonde de localisation reste utile : si elle montre que la projection 768→512
tuait déjà l'information, ce design est directement vindiqué ; si c'était le pooling,
m est le cadran ; si c'était le rang r, élargir les lignes n'aurait rien rendu. **SONDE FAITE (08-01 soir, FINDINGS)** : verdict = le goulot est le POOLING appris
à une tête (rétention valeur 1.00 → 0.47 au pooling ; la moyenne uniforme des mêmes
états retient 0.675 — l'information était là, le pooling la jette) ; la projection
768→512 ne coûte rien (0.47 → 0.45) ; le read r=8 = second ordre (0.45 → 0.37) ;
slot/strate traversent intacts (le gist actuel suffit pour moduler). **Cadran
désigné : m têtes de pooling par write** — la matrice (m, d) est vindiquée sur
l'axe du NOMBRE de lignes ; la pleine largeur est gratuite mais n'était pas le
goulot ; r vient après. Le tap mi-stack reste à sonder (la dernière couche contient
encore la valeur linéairement). Contre-pression : la dilution ambiante (kill-test 8).

**Point de prélèvement du write (gist)** : le pooling actuel prend la DERNIÈRE couche
— l'étage le plus aligné output (logit-lens ; jouet ph.9 : canal post-norm 0.434 vs
0.708 pour l'embedding d'entrée, mort sous readout appris). La surface est déjà
prélevée couche 0 (ce qui la sauve) ; le gist veut les CONCLUSIONS (mi-tardif, ~2/3
de profondeur), pas la rotation vocabulaire. Correctif ranké : tap à ~2/3, ou
scalar-mix appris sur les couches (n_layers scalaires — le mix appris est lui-même
un diagnostic). Knob phase 1.

### 2.5 Deux lectures sur un seul store

| | Lecture ambiante | Lecture ciblée |
|---|---|---|
| Mécanisme | cascade, chaque forward ([cascade.py](deepseek_v4_mini/cascade.py)) | RTI + copy-head ([rti.py](deepseek_v4_mini/rti.py), [rti_copy.py](deepseek_v4_mini/rti_copy.py)) |
| Consomme | gist | surface |
| Fonction | module la distribution (registre, langue, règles) | restitue au token près |
| Invocation | toujours | à la demande (action RL) |

**Principe de séparation : ce qui doit être DIT passe par la surface ; ce qui doit
MODULER passe par le gist.** La lecture ambiante n'est jamais sollicitée pour verbaliser
(le « tour blanc » l'a montré : le prior natif prend).

Appuis empiriques du couple : conditionnement passe (persona Δnll +0.332 à β=0) ;
citation passe uniquement par l'injection native (jouet ph.7-8) + copy-head
(CE valeur 0.201 ON vs 2.212 OFF).

## 2bis. Mise à plat 2026-08-03 — l'architecture consolidée

Cette section intègre la journée du 08-03 : verdicts du run copy (chaîne
retrieve-then-inject-then-copy fermée au 350M), des kill-tests 1-2 (la banque bat
la compaction texte au rappel ; le mur = la sélection), de la grille jouet ph.10
(l'attention-read gagne, le fast-weight perd dans les deux régimes), du carré
factoriel (en cours), de la veille rotations (FINDINGS + fiche
ref-rotations-metadonnees-sota-2026), et de cinq itérations de design user sur le
write. Là où elle contredit §2.3-2.5, ELLE fait foi.

### 2bis.1 Le store : UN tenseur, des conventions

Banque = un seul tenseur `(max_mem, mem_dim, d)` + compteurs par slot (naissance,
usage). Aucune structure annexe. Les « régions » sont des conventions de position :

| région | où | règle |
|---|---|---|
| épinglée | slots exempts du décalage FIFO (budget propre) | lignes taguées system : jamais évincées — le prompt système devient banque-résident, survit au RESET à coût fenêtre nul |
| active | tête | la session vivante |
| froide | queue (vers max_mem−1) | évincée du slot de tête, pas encore sortie ; lisible, sauvable |
| morte | au-delà du bord | perdue définitivement |

Le gradient de température EST la position FIFO. La fenêtre de résurrection est
bornée par max_mem : capacité et sursis sont le même cadran.

### 2bis.2 Trois étages, trois natures

**Acquisition (frontière de tour, mécanique + une déviation apprise).**
Capture par défaut SYMÉTRIQUE user/self : chaque tour écrit sa sélection SIF
top-k (copy_mask, §3.1) — 0 forward, aucune politique. Argument décisif = le
RESET : sans self-writes, amnésie asymétrique de ses propres engagements
post-reset. Risque assumé = boucle fermée (le serve divergeait en ~5-10 tours) ;
défenses : tag de provenance (le read pondère « j'ai dit » ≠ « on m'a dit »),
rétention par usage (le junk s'affame), kill-test = horizon de divergence
self-writes ON/OFF. Le `<think>` ne porte plus la capture : il porte la SYNTHÈSE
(ce qui n'est verbatim dans aucun tour) et le marquage de salience — appris,
au-dessus du défaut mécanique.

**Maintenance (procédurale, zéro paramètre, zéro forward).**
Une seule primitive : append en tête + décalage + chute au bord. La survie est un
acte : PROPAGATION des lignes utiles de la queue vers la tête (budget par write,
compteur de naissance préservé). Le poids-par-ancienneté est RÉPUDIÉ (proxy
auto-renforçant) ; le signal de rétention est choisi par bakeoff ph.11 :
attention-EMA (le lecteur vote — gratuit dans les reads à projections dédiées) /
couverture sémantique (évincer la ligne au plus proche voisin) / activation
ACT-R (usage à décroissance) / baseline âge (à battre). Résurrection = propagation
depuis la queue avant le bord, ancienneté reprise.

**Lecture (apprise, attention — la seule classe de fonctions du design).**
Convergence de quatre lignes : l'empirique (grille ph.10 : 2AFC 1,000 attention
vs effondrement fast-weight), le codage (les rotations ne se décodent que dans un
produit q·k), la maintenance (la masse d'attention par ligne = le signal de
rétention, sous-produit gratuit), le cross-modal (l'attention sur un ensemble est
modality-agnostic). Deux ailes inchangées (§2.5) : surface = RTI α + copy-head
(prouvé 350M : grade 0,28 vs 0,00, p_copy val-only) ; gist = attention sur la vue
plate — le POINT dans le carré factoriel : {kv_append, kv_proj, dual_heads}
± bank-q, départagé sur Δnll citation + marges (le 2AFC sature).

### 2bis.3 Métadonnées = rotations sur plans réservés (K/V dédiées)

Tout ce qui décrit une ligne — quand, qui, quel canal — est une rotation
élémentaire ; le contenu reste le vecteur natif intact (norme préservée).
Contraintes actées par la veille (ref-rotations-metadonnees-sota-2026) :

- **Âge : LOG-COMPRIMÉ avant rotation** (φ(a) ∝ log(1+a)) — un âge non borné à
  fréquences fixes = le problème d'extension de contexte déguisé (Base of RoPE) ;
  8 plans, échelle géométrique, entrelacés, SANS bande haute fréquence
  (collisions VideoRoPE ; dims précoces inutilisables 2502.11276), fréquences
  apprises (LieRE). Résolution décroissante avec l'âge = la sémantique voulue.
- **Provenance/canal : UN PLAN 0/π PAR CANAL** (user, self, system ; puis
  modalités) — jamais n angles sur un plan (métrique cyclique parasite). π/2 si
  l'orientation « qui lit qui » doit compter. La rotation de provenance =
  réimplémentation du chat template dans l'espace où il n'existe plus (la
  sélection détruit les balises de rôle ; précédent : TS-RoPE, locuteur).
- **Jamais dans la bande RoPE du backbone** (guideline « textual priors ») —
  les plans vivent dans les projections K/V dédiées du read.
- **Contrôle NON NÉGOCIABLE (ph.11) : θ_âge = 0** — HoPE prouve que la rotation
  nulle sur l'axe long maximise le rappel ; si notre rotation d'âge ne bat pas ce
  bras sur la citation, l'âge passe en biais scalaire de récence.

### 2bis.4 Sélection : le mur, et ses trois réponses en couches

KT1 a localisé le mur : grade 0,717 quand le vrai groupe sort en tête, 0,000 sur
les miss, r@1 0,480 en banque multi-vies. Réponses, du court au long terme :
1. **GRPO retrieve** (cliquet prévu) — le crédit Plackett-Luce existe déjà ;
2. **Les métadonnées comme filtres de descente** : la requête choisit son axe —
   « qu'a dit l'utilisateur sur X » descend par le plan provenance, « qu'ai-je
   dit hier » par les plans d'âge ;
3. **La hiérarchie CSA/HCA sur la banque** (§2bis.5) : le niveau grossier est un
   retriever SOFT différentiable — le remplaçant candidat du top-k dur.

### 2bis.5 Trajectoire de scaling : la même attention, hiérarchique

Reprendre la structure CSA/HCA du backbone SUR la banque. La hiérarchie a DEUX
axes de localité — quand (blocs d'âge : récent ligne à ligne, passé résumé par
blocs, même geste que le log-âge) et qui (plans de canal) — et une requête choisit
sa descente. Coût de read sous-linéaire en S ⇒ max_mem de ~8 à ~10³ : la banque
passe de mémoire de travail à mémoire de session longue. Ordre des preuves :
le carré tranche l'attention PLATE à max_mem=8 → KT8 (courbe de dilution
S=1/4/16/64) dit où la plate s'écroule → la hiérarchie entre avec l'écroulement
comme baseline. Pitch : le backbone lit son contexte et sa mémoire avec la MÊME
structure — « attention is all you need », à condition de lui donner un deuxième
corpus qui survive à la fenêtre.

### 2bis.6 Ce qui apprend, ce qui n'apprend pas

| apprend (dans le graphe) | n'apprend pas (procédural, hors graphe) |
|---|---|
| le read (projections, biais de logits banque) | sélection SIF top-k + copy_mask |
| la copy-head | FIFO, propagation, résurrection, épinglage |
| le `<think>` (synthèse, salience) — GRPO | rotations (âge log, plans de canal) |
| le retrieve (GRPO, puis hiérarchie soft) | compteurs de naissance/usage |

Tout le neuf est procédural ; tout ce qui apprend est du transformer standard.

## 3. Corrections pré-run obligatoires (dettes identifiées par la critique)

### 3.1 Sélection de surface à priorité de span — RÉVISÉ par le kill-test 3 (08-01)

Le kill-test 3 a **renversé** le diagnostic initial (« biais chiffres ») : le top-k est
un classement intra-tour, et les chiffres y survivent très bien dans les tours courts
(numeric : inclusion pleine 1.000, même au a teacher). Le vrai bug est **le budget k
fixe contre la longueur du tour** : code (T=35) a une inclusion pleine de **0.000** —
les digits du nom du helper et la constante perdent contre les mots de template rares.
Verdict complet dans FINDINGS 2026-08-01.

Correctifs **LIVRÉS** (08-01, self-tests verts, re-mesure : inclusion pleine 1.000
sur les quatre strates) : (1) sélection à **priorité de span** au write —
`build_group(…, copy_mask)`, positions garanties, clé intouchée, fallback SIF pur
sans mask (même statut d'oracle que `fact_slot`, remplacé à terme par le write
`<think>`) ; (2) l'env marque **tous** les atomes citables via `copy_mask` (champ
`atoms` : pour `code`, nom ET constante ; `val_mask` inchangé = cible teacher) ;
(3) k reste 13 (max 11 atomes observés ; k=22 rejeté, +64 % de préfixe). Côté
**clés** : rien à corriger — rank-1 même-slot numeric 0.997 ; la faiblesse pref
(0.57-0.65) est le territoire du tie-break de récence (supersession by design), et
le centrage global ne change rien (testé).

### 3.2 Contrefactuel leave-one-overwrite

Le leave-one-slot existant ([rti_learner.py](deepseek_v4_mini/rti_learner.py)) mesure la
valeur de ce qui est **écrit**, pas de ce qui est **détruit**. Sans « rejouer en
préservant le slot écrasé », le choix de slot n'a pas de gradient et le GRPO dégénère en
FIFO. Extension du harnais requise ; SNR à mesurer (kill-test 6) avant de promettre la
supersession.

### 3.3 Normalisation du layout d'injection

Le circuit de copie est câblé sur le layout du préfixe (jouet ph.8 : copie 0.66 si
ordre = score, 0.21 si aléatoire). Le nombre de groupes injectés et leur ordre doivent
être **normalisés en dur** entre train et déploiement (k fixe, ordre = score, padding à
vide). La falaise se cartographie au kill-test 9.

### 3.4 Copy-head : le régime négatif

Le copy-head est un canal d'hallucination-par-copie : sur retrieval erroné ou slot
périmé, il copie la mauvaise valeur avec confiance. Le micro-train n'a jamais vu de cas
négatif ; le gate `log_alpha` n'est calibré sur rien. À entraîner avec des négatifs
(wrong-slot, slot-périmé) et à évaluer sur le **taux de copie indue** comme métrique de
première classe (kill-test 7).

## 4. Phase zéro — kill-tests avant d'engager le run

Éval seule sauf mention. Ordre d'exécution : **1 → 2 → 3** décident de la forme finale
du claim et du bug connu d'avance ; le reste gate les promesses une à une.

| # | Question | Protocole | Décision |
|---|---|---|---|
| 1 | La voie surface > compaction textuelle ? | Env recall ([recall_env.py](deepseek_v4_mini/recall_env.py)), 3 bras : A = RTI-inject des k tokens de surface ; B = les **mêmes** k tokens en texte dans le prompt ; C = OFF. Rappel exact + Δnll. | Si B ≥ A en fenêtre libre : l'avantage de A n'existe qu'en fenêtre bornée → le protocole RESET (§5.4) devient le seul terrain du claim surface. |
| 2 | Le précédent Coconut nous tue-t-il ? | OFF + think explicite, apparié en **forwards totaux** vs ON. | Si la CoT explicite gagne : confirme l'abandon du claim « budget visible » (déjà acté) et borne ce que l'ambiant peut revendiquer. |
| 3 | ~~Biais chiffres~~ **FAIT 08-01** | `analysis/rti_key_surface_bias.py`, 1171 énonciations, ckpt rti step_1000. | **Prédiction renversée** : numeric 1.000, le bug réel = k fixe vs longueur de tour (code 0.000, constante hors val_mask). Correctifs §3.1 actés. FINDINGS 2026-08-01. |
| 5 | La sonde de supersession discrimine-t-elle ? | Env « correction rétractée » : le bon slot est l'**ancien** (« en fait non, garde la v1 »). | FIFO/récence score 0 par construction — toute sonde sans ce cas est morte. |
| 6 | La supersession est-elle GRPO-able ? | Leave-one-overwrite sur ~100 épisodes, mesurer le SNR. | SNR nul → supersession hors phase, FIFO assumé dans le papier. |
| 7 | Copie indue | Injections wrong-slot / slot-périmé sur le ckpt copy, calibration `log_alpha`. | Avant toute démo de rappel. |
| 8 | Dilution ambiante | Courbe Δnll conditionnement vs slots distracteurs, S = 1/4/16/64. | Fixe S, borne le claim de conditionnement. |
| 9 | Falaise layout | Varier nombre de groupes / ordre à l'éval sur le ckpt copy. | Fixe la normalisation §3.3. |
| 10 | Exposure bias du write | Jouet : A/B scheduled-sampling sur les entrées du write, métrique = horizon de divergence. | Seul mécanisme candidat pour faire monter l'horizon (§5.5). |

(Le kill-test 4 de la critique — mémoire d'activations TBPTT inter-tours — est acté
sans le payer : verdict intégré au §1, gist = compression SIF assumée.)

## 5. Batterie d'évaluation

### 5.1 Aile conditionnement — paires contrastives Δnll

Banque contient un fait → Δnll entre continuation cohérente et incohérente :

- langue (« parle français » → nll(FR) vs nll(EN)) — barreau 1, signal massif ;
- registre (« 5 ans » → nll(simple) vs nll(technique)) — barreau 2 ;
- règle de contenu (« végétarien » → nll(sans viande) vs nll(steak)) — barreau 3,
  liaison profonde, là où ça casse en premier.

**Contrôles obligatoires** (faux positifs identifiés par la critique) :
gist **aléatoire** → Δ ≈ 0 exigé (sinon le shift est un artefact de perturbation) ;
contenu **inversé** → Δ s'inverse ; occupancy variable (§4.8).

### 5.2 Aile citation — rappel exact apparié

L'env recall et le harnais RL existants servent tels quels ([recall_env.py](deepseek_v4_mini/recall_env.py),
vies scriptées appariées = ancres GiGPO). S'ajoutent : les négatifs du §3.4 et le taux
de copie indue.

### 5.3 Oubli — la sonde qui discrimine

Supersession : après écrasement, la paire contrastive s'inverse **ET** l'env
« correction rétractée » (kill-test 5) sépare supersession apprise d'heuristique de
récence. Les deux, jamais la première seule.

### 5.4 Le protocole central : sessions à travers RESET

Le terrain où le claim vit. Sessions > fenêtre, resets aux frontières, fenêtre bornée
identique pour tous les bras :

- **banque** : traverse les resets, 0 token de fenêtre ;
- **compaction** : résumé textuel porté dans chaque nouvelle fenêtre (paie sa place) ;
- **RAG transcript** : BM25/embeddings sur l'historique, réinjection textuelle (paie son
  retrieval) ;
- **rien** : borne basse.

Métriques : les deux ailes (5.1, 5.2) mesurées *après* n resets, en fonction de n.
Appariement en forwards totaux.

### 5.5 Stabilité : horizon de divergence en boucle fermée

Métrique héritée (divergence en 5-10 tours au step ~309, invisible en teacher-forced).
Toutes les sondes 5.1-5.3 étant teacher-forcées, **aucun tableau vert ne vaut sans ce
chiffre**. Mécanisme candidat pour le faire monter : scheduled sampling sur les entrées
du write (kill-test 10, au jouet d'abord — tension avec la recette TF+distill à arbitrer
sur résultat).

## 6. Mapping vers l'existant

**Se garde tel quel** : write SIF teacher a=1e-4 ; cascade ; RTI + copy-head (aile
citation) ; harnais RL complet (CISPO, GiGPO, vies appariées, leave-one-slot) ; env
recall ; infra désagrégée workers/learner ; perf decode-dispatch (rebind, banque fixe) ;
recette TF+distill+anneal β→0 pour l'amorçage des lectures.

**Se modifie** : append → écrasement de slot (+ tête de choix de slot, init FIFO) ;
pondération de sélection clé/surface (§3.1) ; harnais + leave-one-overwrite (§3.2) ;
layout normalisé (§3.3) ; copy-head ré-entraîné avec négatifs (§3.4) ; slot devient
(clé, surface, gist) — le write actuel ne produit que l'équivalent du gist.

**Se gèle** : curriculum Coconut / CoT latente (horizon post-claim) ; GRPO recall
(déjà en pause @17) ; claim « budget visible » ; tout claim sur le *contenu* du gist.

**S'écrit** (nouveau, petit) : env RESET multi-fenêtres (§5.4) ; env correction
rétractée (§5.3) ; paires contrastives (§5.1) ; bras compaction et RAG (baselines).

## 7. Baselines du papier

| Baseline | Statut | Danger |
|---|---|---|
| Compaction textuelle | **tueuse n°1**, à intégrer partout | attaque les deux ailes ; ne survit au claim qu'en fenêtre bornée |
| RAG sur transcript | tueuse pour le rappel-performance | neutralisée par la reformulation §1 (mécanisme unifié, budget total) |
| DeltaNet / linear attention | objection reviewer n°1, engagement public | à **démontrer** contre un hybride (contrôlabilité : slots adressables, actions RL, supersession) — pas affirmer |
| RMT / compressive | dangereuse pour le gist seul | scinder les claims : notre différenciateur est l'aile surface, que RMT n'a pas |
| Steering vectors / prefix-tuning / TTT | couverte | démo banque vs TTT : 0.95/0.78 à 1/138ᵉ du coût |
| Engram / K3 / V4 (SOTA 2026) | faire-valoir | lookup statique, lecture seule — « Engram sépare le savoir, nous séparons l'état de session » |

## 8. Risques ouverts (assumés, pas résolus)

1. La supersession peut ne pas être GRPO-able (signal de destruction rare/retardé) —
   sortie : FIFO assumé, supersession = travail futur (kill-test 6 tranche).
2. Le conditionnement peut rester superficiel (langue passe, règles de contenu non) —
   l'échelle 5.1 le détecte barreau par barreau, au prix d'un Δnll.
3. L'horizon de divergence peut ne pas monter sans casser la recette TF+distill —
   arbitrage sur le résultat du kill-test 10, au jouet.
4. La dilution ambiante peut imposer un S petit qui étrangle le rappel — le couple
   (8) + protocole RESET donne le trade-off chiffré ; S différenciés par lecture en
   dernier recours (au prix du « un seul store » — à ne payer que contraint).
