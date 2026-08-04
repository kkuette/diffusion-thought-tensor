# SPEC mémoire v2 — édition lissée 2026-08-03

Statut : **VALIDÉ ET CONSOLIDÉ**. Cette édition fusionne le recentrage du 07-31
(critique adversariale acceptée), les verdicts du 08-01 (KT3, sonde de
localisation), et la journée du 08-03 (run copy, KT1-2, grille jouet ph.10,
carré factoriel partiel, veille rotations, cinq itérations de design du write).
L'historique des strates vit dans git et [FINDINGS.md](FINDINGS.md) ; companions :
[EXPERIMENTS.md](EXPERIMENTS.md) (arbre) et
[assets/diagramme_banque.html](assets/diagramme_banque.html) (le diagramme
d'architecture, tenu à jour avec les verdicts). Ce qui n'est pas encore tranché est
au **§3 (EN SUSPENS)** — une décision, un test, une règle.

Remplace comme cap de phase : l'arc recall/RTI en tant que *fin* — le RTI est
l'aile « citation » d'un mécanisme à deux régimes, pas le claim.

---

## 1. Le claim

> **À fenêtre de contexte bornée et appariée, un modèle avec banque (écriture en
> ligne, hors contexte) maintient le conditionnement comportemental ET le rappel
> exact à travers des resets de fenêtre, à coût de contexte nul — contre la
> compaction textuelle (qui paie son résumé en tokens de fenêtre) et le RAG sur
> transcript (qui paie son retrieval), le tout apparié en forwards totaux.**

Comptabilité — règles non négociables :

- Appariement en **forwards totaux** (ou FLOPs), jamais en « tokens visibles » :
  cette comptabilité exclut le compute caché (writes, cascade, pseudo-tokens
  injectés) et rend le claim tautologique.
- **Fenêtre bornée et identique** pour tous les bras. L'avantage revendiqué est
  structurel : la banque est *hors* de la fenêtre et paie 0 token de fenêtre.
- Le **token RESET** est le cœur du protocole : une « session » = plusieurs
  fenêtres séparées par des resets, la banque est le seul état qui traverse.
  Invariant : la seule modification de la banque est un write.

Ce que le claim PORTE, adossé à des mesures :

- **Le rappel exact en fait partie.** Mesuré : la banque bat la compaction
  textuelle au rappel à travers reset (KT1 : 0,343 vs 0,227, Δ apparié +0,115,
  CI [+0,089, +0,142]) ; la chaîne retrieve-then-inject-then-copy est fermée au
  350M (run copy : grade 0,28 vs 0,00 ablaté, p_copy val-only). Réserve : claim
  de **mécanisme unifié sous fenêtre bornée**, pas de performance contre le RAG
  en fenêtre non bornée (le RAG y résout le verbatim mieux et moins cher) —
  démontré à budget total incluant les tokens de retrieval du RAG.
- **Le conditionnement par construction.** La banque entre dans l'attention du
  modèle, donc elle l'influence — et c'est mesuré : 2AFC 1,000, marqueurs
  +2,5/+3,3 nats, shuf SOUS le hasard (spécifique au contenu) ; persona
  Δnll +0,332 à β=0.
- **Règle de protocole (KT1, durcie par l'audit des absolus 08-03)** : le
  pointeur discret gagne le rappel exact mais perd le nll de valeur — le Δnll
  ne juge pas la citation ; et l'audit a montré que le **Δnll INTRA-MODÈLE
  (abl − live) est DÉPRÉCIÉ comme juge tout court** : dans le même carré il a
  gonflé un gagnant (part bait +0,044 de kvproj) ET masqué un concurrent
  (dual_heads, meilleur en absolu, puni pour son meilleur plancher).
  **Adjudication = nll_live ABSOLU apparié + grade décodé + r@1** ; le Δnll
  reste une sonde de conditionnement (paires contrastives), jamais un juge
  d'architecture.

Ce qu'on ne revendique **plus ou pas encore** :

- ~~« Le modèle pense dans le latent »~~ — horizon, pas claim ; à distinguer du
  conditionnement-par-construction (influencé par la banque ≠ pense dedans).
  Curriculum Coconut **abandonné** pour cette phase : (a) Coconut perd contre la
  CoT explicite à compute apparié (précédent publié), (b) il affame la sélection
  de surface en supprimant les tokens qu'elle échantillonne. Le `<think>` reste
  **verbalisé**.
- ~~« Le gist encode un état de pensée »~~ — le contenu des lignes n'a pas de
  canal d'apprentissage praticable à notre échelle (TBPTT inter-tours infaisable
  sur 24 Go ; un scalaire RL ne façonne pas un vecteur). Les lignes sont une
  **sélection/compression SIF assumée comme telle**. Le RL apprend *quand*
  écrire et *quoi propager* — jamais *quoi* mettre dans le vecteur.

## 2. L'architecture

Un tenseur, des vecteurs natifs, des clés calculées, des métadonnées en
rotation ; trois étages — acquisition mécanique, maintenance procédurale,
lecture par attention.

### 2.1 Le store : UN tenseur constant, des conventions

```
banque = tenseur (max_mem, mem_dim, d), taille CONSTANTE
         profondeur = max_mem slots (FIFO), hauteur = mem_dim lignes par slot,
         largeur = d (= d_model, pleine largeur)
```

Plus des compteurs par slot (naissance, usage). Aucune structure annexe. Taille
constante = cohérent avec `mem_seed_slots = max_mem` (perf/decode-dispatch : le
rebind CUDA-graphs suppose une banque de taille constante). Les « régions » sont
des conventions de position :

| région | où | règle |
|---|---|---|
| épinglée | slots exempts du décalage FIFO (budget propre) | lignes taguées system : jamais évincées — le prompt système devient banque-résident, survit au RESET à coût fenêtre nul |
| active | tête | la session vivante |
| froide | queue (vers max_mem−1) | évincée du slot de tête, pas encore sortie ; lisible, sauvable |
| morte | au-delà du bord | perdue définitivement |

Le gradient de température EST la position FIFO. La fenêtre de résurrection est
bornée par max_mem : capacité et sursis sont le même cadran (KT8 le sonde,
§3-S8).

### 2.2 Lignes unifiées : des vecteurs de d, rien d'autre

~~slot = (clé, surface, gist)~~ — le slot structuré est REMPLACÉ. Chaque ligne
de largeur d est **littéralement un vecteur du modèle** — embedding natif
(couche 0) ou état caché côté sortie (mi-tardif/logit) — sélectionné par SIF
top-k, jamais transformé par un réseau (jouet ph.7 : tout readout appris est
mort, seule l'injection native cite). Tout est unifié sur d :

- **Plus de clé stockée.** La clé est une FONCTION, pas une donnée : calculée à
  la lecture par les projections K du read depuis le contenu de la ligne. Le
  pooling SIF de `build_group` survit uniquement comme score de sélection au
  write.
- **Plus de dichotomie surface/gist dans le FORMAT.** Les deux natures
  subsistent comme *provenance du prélèvement* (embeddings couche 0 vs états
  mi-tardifs) et *régime de lecture* (§2.6) — même type, même store, même
  largeur. Une ligne de gist est injectable comme pseudo-token au même titre
  qu'une ligne de surface (injection-comme-modalité, validée ph.10 :
  2AFC 1,000).
- **Les métadonnées ne sont pas des champs** : quand/qui/quel canal = rotations
  sur plans réservés (§2.5), le contenu reste intact (norme préservée).

**Budget** — verdicts de la sonde de localisation (08-01) : le goulot était le
POOLING appris à une tête (rétention valeur 1,00 → 0,47 ; la moyenne uniforme
des mêmes états retient 0,675 — l'information était là, le pooling la jetait) ;
la projection en largeur ne coûtait rien ; d'où la pleine largeur d et **m
lignes par write comme cadran de budget** (la seule compression restante est la
sélection). La grille ph.10 confirme : m est un cadran réel pour l'attention
(la citation croît m4→m8), pas pour le fast-weight (qui s'effondre avec m). Le
point de prélèvement gist (dernière couche vs ~2/3 de profondeur vs scalar-mix)
reste un knob phase 1 (§3-S10).

**Dimensionnement de mem_dim (précision 08-03)** : l'ancre n'est PAS seq_len
mais la statistique d'ATOMES PAR TOUR — KT3 : max 11 atomes citables observés,
k=13 suffit (inclusion pleine 1,000). Une fraction de seq_len (même /2)
produirait une banque massivement vide (2048/2 = 1024 lignes pour ~13 utiles).
Règle : mem_dim = quantile haut des atomes/tour (≈ k + marge propagation),
tenseur à taille CONSTANTE conservé (CUDA-graphs), et les lignes vides MASQUÉES
du softmax — le vide ne coûte alors ni dilution ni signal, seulement de la
mémoire. Chiffrage sur corpus réel = §3-S16.

**PRINCIPE DE SUFFISANCE (user 08-03, prime sur tout le reste du budget)** :
au tour t, la banque doit contenir TOUT le contexte encore intéressant de la
session — partie agent comprise — c'est-à-dire la STATISTIQUE SUFFISANTE pour
continuer. C'est le sens opérationnel du RESET (§6.4) : après reset la banque
est tout ce qui reste ; ce qui n'y est pas est perdu. Conséquence : la
capacité se dimensionne sur le STOCK utile à l'instant t, pas sur le flux par
tour — et le chiffrage S16 rend le principe exigeant (chat dense : ~30 atomes
nouveaux/tour × 100 tours ≈ 3000 pour 128 lignes à 8×16). La banque ne peut
pas tenir l'union de tout-ce-qui-fut-intéressant : elle tient
ce-qui-est-ENCORE-intéressant, un ensemble que la maintenance FABRIQUE —
l'oubli (propagation/rétention S6) n'est pas une fuite, c'est le mécanisme qui
rend la suffisance tenable ; le `<think>`-synthèse compresse quand le flux
déborde k ; les self-writes sont non négociables (les engagements de l'agent
font partie de la statistique suffisante). Mesure du stock réel = S18.

**Comptabilité de capacité** : l'accumulation k×T des tours ne s'empile PAS
dans mem_dim — elle consomme des SLOTS (un tour = un write = un slot FIFO).
Factorisation propre des cadrans : mem_dim = largeur d'UN tour (invariant en
T), max_mem = horizon EN TOURS, capacité totale = max_mem × k lignes utiles
(celle que la hiérarchie S9 étend vers ~10³). Les overlaps étirent l'horizon
au-delà de max_mem tours : un tour dont la tranche delta est entièrement
redondante (dédup Δn, §2.3) n'écrit rien et ne consomme pas de slot — horizon
effectif = max_mem / taux-de-write — et la propagation recycle par-dessus les
lignes utiles. Le taux de write et le taux de dédup mesurés sur corpus réel
font partie de S16.

**Ancrage cross-modal** : la banque est une modalité — cible long-terme =
vision/son/texte écrivant dans le même store via des encodeurs de write par
modalité. Aucun choix ne doit fermer cette voie ; l'attention sur un ensemble de
lignes est modality-agnostic, seule la voie surface/pointeur est spécifique au
texte (la seule modalité où le verbatim existe).

### 2.3 Trois étages, trois natures

**Acquisition (frontière de tour, mécanique + une déviation apprise).**
Capture par défaut SYMÉTRIQUE user/self : chaque tour écrit sa sélection SIF
top-k (copy_mask, §4.1) — 0 forward, aucune politique. Argument décisif = le
RESET : sans self-writes, amnésie asymétrique de ses propres engagements
post-reset.

Le sélecteur mécanique N'EST PAS le SIF seul (précision 08-03, sur objection
user) : le biais SIF contre les chiffres est mesuré (w̄ chiffres 0,02 vs médian
0,156 ; KT3 : inclusion code 0,000 avant fix) et les tokens numériques sont
précisément les plus citables. En env synthétique le fix est le copy_mask
d'oracle (§4.1) ; en déploiement réel, son transport = un **copy_mask
PROCÉDURAL** : sélection = SIF top-k ∪ détecteur de spans (chiffres,
identifiants, chaînes citées), zéro paramètre. Mesure S16 : sur ultrachat
(pauvre en chiffres) la part numérique hors-SIF n'est que ~2 % — le biais
porte sur les domaines numériques/code/outils, pas sur le chat-essai.

Mécanique du curseur Δn (précision 08-03) : la matrice d'états `(n, d)` grandit
pendant le décodage, et on conserve une COPIE de n (le curseur du dernier
write). La provenance se lit dans le PAS de croissance — un saut Δn > 1 = un
bloc arrivé en prefill (tour user ; system pour le tout premier bloc), +1 par
step = token émis par le modèle. Le tag user/self/system (§2.5) est donc dérivé
mécaniquement du décodage lui-même, aucun parsing de template. La sélection ne
porte QUE sur la tranche delta `[n_prev, n)` : les nouveaux vecteurs sont
scorés SIF puis COMPARÉS À LA BANQUE pour décider quoi ajouter (la redondance
avec l'existant s'écarte, la nouveauté entre) — **jamais tout n contre la
banque entière**. Coût O(Δn·S), incrémental par construction. Le curseur est un
état de FENÊTRE : il repart à zéro au RESET, la banque non. Risque assumé = boucle fermée (le serve divergeait en ~5-10 tours) ;
défenses : tag de provenance (le read pondère « j'ai dit » ≠ « on m'a dit »),
rétention par usage (le junk s'affame), et le kill-test horizon de divergence
self-writes ON/OFF (§3-S13). Le `<think>` ne porte plus la capture : il porte
la SYNTHÈSE (ce qui n'est verbatim dans aucun tour) et le marquage de
salience — appris, au-dessus du défaut mécanique.

**Maintenance (procédurale, zéro paramètre, zéro forward).**
Une seule primitive : append en tête + décalage + chute au bord. La survie est
un acte : PROPAGATION des lignes utiles de la queue vers la tête (budget par
write, compteur de naissance préservé). Résurrection = propagation depuis la
queue avant le bord, ancienneté reprise. Le comportement cible : « I live in
New York » au tour 1 et « I moved to Austin » au tour 100 coexistent, chacun
avec son âge vrai. Le poids-par-ancienneté est RÉPUDIÉ (proxy auto-renforçant :
ce qui survit devient plus dur à déloger parce qu'il a survécu) ; le signal de
rétention est EN SUSPENS, choisi par bakeoff (§3-S6). Remplace l'ancien design
« écrasement + tête de choix de slot » : la supersession n'est plus un choix de
destruction mais un effet émergent — le fait nouveau entre par append, l'ancien
survit s'il est utile. Piège hérité toujours valable (dsv4mini switch task) :
ces politiques ne s'apprennent pas « en passant » — la sonde anti-récence
(§6.3) reste obligatoire.

**Lecture (apprise, attention — la seule classe de fonctions du design).**
Convergence de quatre lignes : l'empirique (grille ph.10 : 2AFC 1,000 attention
vs effondrement fast-weight dans les deux régimes — le multiplicatif diverge
dès m>1, l'additif s'effondre avec m), le codage (les rotations ne se décodent
que dans un produit q·k), la maintenance (la masse d'attention par ligne = un
signal de rétention gratuit), le cross-modal (l'attention sur un ensemble est
modality-agnostic). À la lecture, le store s'aplatit en VUE
`(max_mem·mem_dim, d)` — reshape zéro-copie — et les lignes entrent dans
l'attention des couches lectrices. Le fast-weight read est retiré du design.

### 2.4 Le read : forme tranchée par le carré factoriel

Carré 2×2 : projections {partagées, dédiées} × softmax {unifié, séparé} —
kv_append (partagées/unifié), dual_heads (dédiées/séparé), kv_proj
(dédiées/unifié) ± bank-q. **Verdict (08-03, ailes COMPLÈTES + AUDIT DES
ABSOLUS ; bank-q = S2 INVALIDÉ pour fuite non-causale, voir §3-S2)** — le
départage initial au Δnll citation a été RE-JUGÉ en nll_live absolu apparié
(règle §1, l'audit ayant montré que le delta gonfle ET masque) :

- **kvproj ADOPTÉ (S1 tranché, confirmé en absolu)** : nll_live −0,165 ± SE
  0,053 (t 3,1) vs kv_append, part bait du delta +0,044 non significative ;
  l'exploitation du budget m est réelle en absolu (3,86 → 3,35 de m1 à m8).
  Double justification : performance mesurée ET hébergement des rotations
  (§2.5 — une rotation appliquée avant W_k ne survit pas à la projection ; les
  dims partagées sont occupées par la sémantique du backbone). Prix : ~2d² par
  couche lectrice. Fallback dims-étendues retiré.
- **dual_heads : DOMINÉ COÛT-AJUSTÉ (verdict corrigé par l'audit)**. Le delta
  le disait nul — artefact : son plancher ablaté plus bas le punissait. En
  ABSOLU il est le meilleur à chaque m (m8 : 3,298 vs kvproj 3,346), mais
  l'avance sur kvproj est dans le bruit (+0,048, SE ~0,05), le coût est double
  (4 matrices dédiées + seconde passe d'attention) et les paramètres ne sont
  pas appariés. kvproj reste le choix ; dual_heads redevient un candidat
  seulement si un jour le softmax unifié plafonne à grande banque (KT8).
- **La compétition de masse softmax n'est pas une contrainte réelle** : le
  `bank_logit_bias` appris de kvproj reste ≈ 0 (moy +0,004 à m1, léger
  amortissement −0,012/−0,019 à m4/m8).
- L'age-rot du jouet (DFT brute) reste ~neutre dans kvproj (Δcit +0,055,
  Δmark −0,144, n=6) — le design réel (log + fréquences apprises) se juge en
  ph.11 (S3-S4), pas ici.
- Réserve : verdict à max_mem=8 — KT8 dira s'il tient quand la banque grandit.

### 2.5 Métadonnées = rotations sur plans réservés (K/V dédiées)

Tout ce qui décrit une ligne — quand, qui, quel canal — est une rotation
élémentaire ; le contenu reste le vecteur natif intact. Contraintes actées par
la veille (fiche ref-rotations-metadonnees-sota-2026) :

- **Âge : LOG-COMPRIMÉ avant rotation** (φ(a) ∝ log(1+a)) — un âge non borné à
  fréquences fixes = le problème d'extension de contexte déguisé (Base of
  RoPE) ; 8 plans, échelle géométrique, entrelacés, SANS bande haute fréquence
  (collisions VideoRoPE ; dims précoces inutilisables), fréquences apprises
  (LieRE). Résolution décroissante avec l'âge = la sémantique voulue (3 vs 4
  tours compte, 5000 vs 5001 non).
- **Provenance/canal : UN PLAN 0/π PAR CANAL** (user, self, system ; puis
  modalités) — jamais n angles sur un plan (métrique cyclique parasite). π/2 si
  l'orientation « qui lit qui » doit compter. La rotation de provenance =
  réimplémentation du chat template dans l'espace où il n'existe plus (la
  sélection top-k détruit les balises de rôle ; précédent : TS-RoPE, locuteur).
  Le tag system + épinglage (§2.1) = le prompt système banque-résident.
- **Position LOCALE intra-span (troisième famille, précision 08-03)** : sans
  ordre interne, un span multi-tokens devient un sac (« New York » = {New,
  York}) — cassant pour la citation. On encode l'INDEX LOCAL de la ligne dans
  son write (0..mem_dim−1, borné ⇒ aucun risque d'OOD, fréquences standard).
  Jamais la position absolue de fenêtre : elle meurt au RESET et le « quand »
  est déjà porté par les plans d'âge. Test de nécessité = §3-S17.
  La relativité vit LIGNE↔LIGNE, pas lecteur↔ligne : vu du token lecteur (sans
  coordonnée locale), R_loc(j) est une signature de phase qui DISTINGUE les
  lignes ; entre lignes, « avancer d'un token dans le span » devient un
  opérateur CONSTANT R_loc(1), indépendant du contenu — exactement le geste
  d'un circuit de copie (induction heads sur RoPE, précédent backbone). C'est
  l'argument structurel contre l'index ADDITIF (S17) : l'additif donne des
  signatures, jamais l'opérateur successeur. (Les scores ligne-ligne pleinement
  relatifs R(j_a − j_b) restent disponibles pour la contextualisation au WRITE
  — le bank-q au read est retiré, verdict S2 : fuite non-causale.)
- **Jamais dans la bande RoPE du backbone** — les plans vivent dans les
  projections K dédiées de kvproj : la dédicace est un MOYEN d'hébergement, pas
  une fin (§2.4).
- **Contrainte côté REQUÊTE (08-03, lue dans le code)** : kvproj partage le q
  du backbone, ROPE COMPRIS — le score banque est (R(t)·W_Q x_t)ᵀ(W_K' g_r),
  la rotation R(t) n'étant pas annulée côté banque. Le produit se faisant dim à
  dim, les plans de métadonnées doivent viser les dims QUASI STATIQUES (ω ≈ 0)
  du RoPE backbone côté requête, sinon le code d'âge se mélange à la position
  dans la fenêtre (cos(ωt − φ_âge)). Le corollaire espéré — « les clés banque
  se dockent naturellement dans la bande lente, HoPE retrouvé par
  l'implémentation, W_K' l'apprend seul » — a été **MESURÉ 08-03 (nuit) et
  RÉFUTÉ** : sur les 15 ckpts kvproj, l'énergie de W_K' par bande de fréquence
  est indiscernable de l'uniforme (moitié lente 0,5000 ± 0,0005). À fenêtre
  courte, la contamination ne coûte rien, donc rien ne pousse le modèle à s'en
  protéger ⇒ **la prescription se tient PAR CONSTRUCTION**
  (`slow_rope_planes` + garde de dérive `rot_drift_max`, livrés en ph.11), et
  la mesure est à refaire en fenêtre LONGUE au 350M
  ([analysis/kvproj_wk_spectrum.py](deepseek_v4_mini/analysis/kvproj_wk_spectrum.py)).
  Fallback si contamination avérée :
  DÉ-ROTER q pour les colonnes banque (score banque = qᵀk' nu, deux attentions
  fusionnées par log-sum-exp — exact, prix = perte du SDPA unique). JAMAIS de
  rotation de la banque par position absolue de fenêtre : (R(t)q)ᵀR(p)k' =
  qᵀR(p−t)k' — le parasite devient relatif au lieu de disparaître, et la
  position de fenêtre (qui meurt au RESET) rentre par la fenêtre. Principe :
  toute rotation côté banque est fonction de propriétés DE LA LIGNE (âge,
  canal, index local), jamais de la position du token lecteur.
- Trois examens EN SUSPENS avant d'acter (§3-S3/S4/S5) : le contrôle θ_âge=0
  (HoPE : la rotation nulle sur l'axe long maximise le rappel — si l'âge rotatif
  ne le bat pas, l'âge passe en biais scalaire de récence), l'OOD d'âge, et le
  tag rotatif vs additif (le circuit de copie est câblé au layout, ph.8 — une
  perturbation rotative des clés peut le casser là où l'additif non).

### 2.6 Deux régimes de lecture sur un seul store

| | Lecture ambiante | Lecture ciblée |
|---|---|---|
| Mécanisme | attention sur la vue plate (kvproj), chaque forward | RTI + copy-head ([rti.py](deepseek_v4_mini/rti.py), [rti_copy.py](deepseek_v4_mini/rti_copy.py)) |
| Consomme | lignes de nature gist | lignes de nature surface |
| Fonction | module la distribution (registre, langue, règles) | restitue au token près |
| Invocation | toujours | à la demande (action RL) |

**Principe de séparation : ce qui doit être DIT passe par la surface ; ce qui
doit MODULER passe par le gist.** La lecture ambiante n'est jamais sollicitée
pour verbaliser (le « tour blanc » l'a montré : le prior natif prend). Appuis :
conditionnement (persona Δnll +0,332 à β=0 ; 2AFC 1,000 ph.10) ; citation
uniquement par injection native + copy-head (CE valeur 0,201 ON vs 2,212 OFF ;
chaîne fermée 350M au run copy). Nuance ph.10 : les lignes d'états cachés
conditionnent parfaitement mais ne citent pas (grade 0,000 partout) — la
séparation est confirmée des deux ailes.

### 2.7 Sélection : le mur, et ses trois réponses en couches

KT1 a localisé LE mur du système : grade 0,717 quand le vrai groupe sort en
tête, 0,000 sur les miss, r@1 0,480 en banque multi-vies. Réponses, du court au
long terme :

1. **GRPO retrieve** (cliquet prévu) — le crédit Plackett-Luce existe déjà ;
2. **Les métadonnées comme filtres de descente** : la requête choisit son axe —
   « qu'a dit l'utilisateur sur X » descend par le plan provenance, « qu'ai-je
   dit hier » par les plans d'âge ;
3. **La hiérarchie CSA/HCA sur la banque** (§2.8) : le niveau grossier est un
   retriever SOFT différentiable — remplaçant candidat du top-k dur.

### 2.8 Trajectoire de scaling : la même attention, hiérarchique

Reprendre la structure CSA/HCA du backbone SUR la banque. Deux axes de
localité — quand (blocs d'âge : récent ligne à ligne, passé résumé par blocs,
même geste que le log-âge) et qui (plans de canal) — et une requête choisit sa
descente. Coût de read sous-linéaire ⇒ max_mem de ~8 à ~10³ : la banque passe
de mémoire de travail à mémoire de session longue. Ordre des preuves : le carré
tranche l'attention PLATE à max_mem=8 → KT8 (dilution S=1/4/16/64) dit où la
plate s'écroule → la hiérarchie entre avec l'écroulement comme baseline. Pitch :
le backbone lit son contexte et sa mémoire avec la MÊME structure — « attention
is all you need », à condition de lui donner un deuxième corpus qui survive à
la fenêtre.

### 2.9 Ce qui apprend, ce qui n'apprend pas

| apprend (dans le graphe) | n'apprend pas (procédural, hors graphe) |
|---|---|
| le read (projections kvproj, fréquences d'âge) | sélection SIF top-k sur la tranche Δn + dédup contre la banque ; curseur n ; tag par pas de croissance |
| la copy-head | FIFO, propagation, résurrection, épinglage |
| le `<think>` (synthèse, salience) — GRPO | rotations (âge log, plans de canal) |
| le retrieve (GRPO, puis hiérarchie soft) | compteurs de naissance/usage |

Tout le neuf est procédural ; tout ce qui apprend est du transformer standard.

## 3. EN SUSPENS — registre des décisions ouvertes

Chaque ligne : une décision, LE test qui la tranche, la règle appliquée au
verdict. Rien d'autre n'est ouvert.

| # | Décision | Test | État | Règle de décision |
|---|---|---|---|---|
| S1 | kvproj vs kv_append (coût de la dédicace K/V) | 12 cellules kvproj du carré + audit des absolus (`p10_abs_nll.py`, 42/42 harnais OK) | **TRANCHÉ 08-03 : kvproj ADOPTÉ, confirmé en ABSOLU** | nll_live −0,165 (t 3,1) apparié vs kv_append, part bait +0,044 ns ; exploitation de m réelle (3,86→3,35) ; dual_heads re-jugé : meilleur absolu mais DOMINÉ COÛT-AJUSTÉ (+0,048 sur kvproj à m8 = bruit, double coût, params non appariés) ; règle durcie : adjudication = nll_live absolu + grade + r@1, plus jamais le Δnll intra-modèle |
| S2 | ±bank-q (les lignes se contextualisent au read) | 12 cellules bank-q du carré | **MESURE INVALIDÉE 08-03** (7/12 suffisent) | fuite NON-CAUSALE : les lanes voient tout le segment teacher-forcé (futur compris) et le réinjectent aux K/V des couches suivantes — citation +1,45 (fuite, pas lecture) ET conditionnement effondré 2AFC 0,89→0,52 avec m (le vrai circuit s'atrophie, cousin du bug boundary_step GRPO 07-27). Verdict de design : la contextualisation banque-lit-banque appartient à l'étage WRITE (frontière de tour, passé seul) — bank-q au read est retiré ; re-test éventuel = lanes mises à jour aux frontières de tour uniquement |
| S3 | rotation d'âge vs rien vs biais scalaire | ph.11 : contrôle θ_âge=0 (HoPE), NON NÉGOCIABLE | **NON-RÉSULTAT 08-04 sur `rule`** (8 cellules finies) : acc 1,0 plafond partout, marges agezero ≥ tous les bras rotés — l'env n'a AUCUNE charge temporelle, la rotation n'y a rien à coder (objection user 08-04). Verdict final = zzs115/116 (ph.12, `age-log` × {fifo, attn-ema} sur env `life`, A/B exacts vs zzs101/103). **08-04 : premier A/B `life` AVEUGLE sur la citation** (bug régime de données S6) — sauvé des logs, SUGGESTIF pro-rotation non tranchant : fifo agelog r@1 0,754 vs 0,635 (hist 0,375 vs 0,283, nll_live meilleur), attn-ema 0,625 vs 0,602 mais hist inversé. Verdict reporté sur zzt115/116 vs zzt101/103 (pool span, en file 08-05) | la rotation doit BATTRE θ_âge=0 sur la citation, sinon âge = biais scalaire de récence |
| S4 | OOD d'âge (compression) | ph.11 : train ≤A_train, éval 10×/100× ; bras {brut, log-comprimé, brut+augmentation} | **TRANCHÉ 08-04 : l'AUGMENTATION est le porteur, pas le log.** Sans aug : agelog m8 s'effondre à ×10 (marge 3,01→1,82, acc 0,90), ageraw m8 fléchit (2,50). Avec aug (zzr201/202) : raw ET log tiennent (acc 1,0, marges plates ×1/10/100). Rotation sans payload = perturbation parasite ; l'aug apprend au read à l'annuler | le bras qui tient l'OOD gagne ; prédiction log RÉFUTÉE — l'aug d'échelle devient OBLIGATOIRE dès qu'un plan d'âge existe |
| S5 | tag provenance rotatif vs additif | ph.11 : A/B direct, mesurer r@1 ET taux de copie SÉPARÉMENT | **EXAMEN INVALIDE 08-04** : les 6 cellules `prov` rendent des métriques GELÉES step 0→3000 (grade 0,000 train ET éval, r@1 identique au bit dans les 6 bras : argmax de masse TOUJOURS sur le groupe self = 159/299) alors que `span` (même harnais) monte à 0,97. **Cause (sonde 08-04)** : pool `ref` = codes XX-12345, la queue à 5 chiffres dégénère en répétitions au décodage (classe d'échec KT3/RTI) ⇒ grade tout-ou-rien plancher 0, zéro discrimination — le routage de canal marche pourtant (template du bon canal + lettres copiées). **Re-run posté** : `zzr311-316` avec `--prov-vals span` (valeurs mesurées, buckets 2-3) ; surveiller l'argmax de masse collé au groupe self | **v2 PRÉLIMINAIRE 08-04 (zzr311-316 finies, n=1)** : instrument réparé (grades 0,66-0,98) ; la rotation ne casse PAS la copie (m8 : rot 0,933 ≈ none 0,943, add 0,980 à ~2σ) ; l'argmax de masse est disqualifié comme sonde de routage à m8 (user r@1 0,000 avec grade user 1,000 — la masse colle au self, la valeur passe quand même) ; MAIS verdict « tag ≈ none » borné au DOMAINE layout-régulier : dans `prov` le canal est inférable de l'âge/layout (rime S17). GARDE (user 08-04) : le domaine réel du tag = épinglage (ligne system ancienne par construction : âge et position FIFO muets), post-RESET, canaux entrelacés irréguliers (user/assistant/tool/system), et surtout la SÉLECTION (filtre de descente §2.7, le mur r@1 0,480) + l'escompte du canal self en boucle fermée (anti-auto-empoisonnement) — pas la citation. Coût quasi nul (0 param, norme préservée) ; prix réel = discipline d'aug (nll rot 0,0826 vs none 0,0768, cohérent S4). CONTRÔLE DÉCISIF avant d'économiser : env à canal DÉCORRÉLÉ du layout (ordre d'écriture irrégulier ou slot épinglé ancien) — là le tag porte le routage seul ou meurt | si la rotation casse le circuit de copie là où l'additif non → additif sur dims réservées |
| S6 | signal de rétention (maintenance) | ph.12 : bakeoff FIFO nue / âge (baseline à battre) / attention-EMA / couverture sémantique / ACT-R — vies longues avec ré-évocation tardive | **PREMIER TIR AVEUGLE 08-04** : 16 cellules finies, grade held-out 0,000 PARTOUT — régime de données invalide, pas la maintenance (grade_train 0,80, r@1 transfère ; sonde : valeur dans les lignes 73 %, sorties = villes TRAIN au prior). Cause = pool `city` fermé (~40 entités à signature unique) ⇒ le train se minimise par RECONNAISSANCE, la copie ne se forme jamais (leçon dsv4mini « répertoire fermé », rejouée). Fix : `--life-vals span` (buckets 1-2, 96 valeurs compositionnelles L≤2) + val_ids étendue des deux côtés ; **re-run zzt101-116 EN FILE 08-05**. Indices r@1 sous réserve : T88 fifo 0,675 > cover ≈ actr > attnema ≈ age | métrique de première classe = temps d'adaptation aux faits nouveaux ET survie des faits anciens utiles |
| S7 | politique `<think>` (synthèse + salience) | GRPO ph.2 (cliquet SFT/RL) | après phase 11 | suivi = Δnll + toolcall, jamais requote (sonde morte) |
| S8 | max_mem (dilution vs rappel) | KT8 : courbe 2AFC conditionnement vs S = 1/4/8/16/64, banque remplie de distracteurs réels | **TRANCHÉ 08-04** : acc S1 0,773 / S4 0,943 / S8 0,807 / S16 0,557 / S64 0,500 (hasard) — l'attention plate s'écroule entre S8 et S16, morte à S64 ; les bras dual ne repoussent rien (S16 0,568, S64 0,489) : mur ARCHITECTURAL, robuste inter-reads | max_mem = 8 pour la greffe ; au-delà de ~S8, la hiérarchie S9/§2.8 n'est pas une option mais la seule voie — l'écroulement mesuré est sa baseline d'entrée |
| S9 | hiérarchie CSA/HCA sur la banque | bras jouet post-KT8 | après S8 | n'entre que si la plate s'écroule ; battre la plate à S égal |
| S10 | point de prélèvement gist (dernière couche vs ~2/3 vs scalar-mix) | knob phase 1 (sonde tap mi-stack) | à lancer | le mix appris est lui-même un diagnostic |
| S11 | KT2 propre (banque vs CoT explicite à forwards appariés) | re-run avec bras `<think>` ENTRAÎNÉ (le KT2 du 08-03 est non concluant : layout jamais vu au train) | après un SFT think | si la CoT gagne à budget apparié, l'ambiant borne ses revendications |
| S12 | supersession/propagation GRPO-able | KT6 : SNR du contrefactuel leave-one sur ~100 épisodes | à lancer | SNR nul → maintenance purement procédurale assumée dans le papier |
| S13 | self-writes : boucle fermée maîtrisable | horizon de divergence self-writes ON/OFF (+ KT10 scheduled sampling comme levier) | à lancer | si OFF >> ON, la capture self recule vers `<think>`-seul |
| S14 | copie indue (négatifs copy-head) | KT7 : wrong-slot / slot-périmé, calibration `log_alpha` | avant toute démo de rappel | taux de copie indue = métrique de première classe |
| S15 | falaise layout | KT9 : varier nombre/ordre de groupes à l'éval sur le ckpt copy | à lancer | fixe la normalisation §4.3 |
| S16 | mem_dim (taille de slot) | `analysis/s16_memdim_stats.py` (recall_env exact + ultrachat proxy SIF∪numérique v2) | **FAIT 08-03** | recall_env : ≤10 tokens/tour (mem_dim 16 confirmé, seq_len/2 réfuté), write 0,234 ⇒ horizon ×4,27 ; ultrachat : assistant p50 38 candidats ≫ k=13 (les tours longs saturent la sélection ⇒ rôle du `<think>`-synthèse), write 0,944 ⇒ en chat dense la dédup n'étire pas l'horizon — la charge est sur la propagation (S6) ; part numérique hors-SIF ~2 % sur ultrachat MAIS le biais chiffres porte sur code/outils (KT3) ⇒ sélecteur déployé = SIF ∪ copy_mask procédural (§2.3), à re-mesurer sur corpus agentique |
| S17 | position locale intra-span nécessaire ? | ph.11 : citation de valeurs multi-tokens avec/sans rotation d'index local | **TRANCHÉ 08-04 : ÉCONOMISÉE.** locnone m8 grade 0,97 jusqu'à L6 (r@1 0,995) ; locrot 0,95, locadd 0,95 — aucun bras n'apporte rien. La chute m4 sur L4-L6 (0,37-0,55, identique dans les 3 bras) est la LARGEUR de slot (k=4 lignes < 5-6 tokens), pas l'ordre : l'ordre vient gratuitement du layout d'injection. Le zéro held-out de `rule` = non-isomorphie train/éval (leçon ph.8), pas l'index manquant | si le sans-ordre casse les spans multi-tokens → la troisième famille entre ; sinon économisée — VERDICT : économisée ; le vrai cadran des spans longs = k (largeur de slot, cf. S16) |
| S18 | STOCK utile à l'instant t (principe de suffisance §2.2) | `analysis/s18_stock_utile.py` : stock vivant = atomes vus ≤ t qui resurgissent > t (borne basse lexicale) | **v1 FAITE 08-03** (courte session) | ultrachat : stock p50 ~22, pic p90 62, 1,8 % des convs > 128 lignes ⇒ suffisance TENABLE à 8×16 ; stock ≪ flux (l'essentiel ne resurgit jamais — l'écart que la rétention S6 exploite) ; résurgence p50 2 tours ; RESTE : re-mesure sur sessions LONGUES (convs p50 = 6 tours seulement) — c'est elle qui arbitrera max_mem grand vs hiérarchie S9 |
| S19 | read_layers (placement du read kvproj dans la cascade 350M) | sonde 350M : bras {toutes les couches / tiers tardif / HCA seules / CSA seules / bloc 0 contrôle} à params appariés | OUVERTE (08-04) — le jouet ne peut pas l'arbitrer (attention dense, 6 couches) : TOUS les verdicts jouets (carré, audit, ph.11/12) sont en régime toutes-couches (`read_layers` vide). Coût 350M : ~2d²/couche lectrice ≈ 14 % des params si toutes ; les colonnes banque (S=128) sont gratuites en FLOPs | battre toutes-couches à params moindres ; défaut si non tranchée = toutes les couches (le régime prouvé) |
| S20 | la cascade a-t-elle encore un emploi ? (spec v3 → v2) | ablation pendant la phase de GREFFE : cascade ON/OFF à coût apparié, UNE FOIS le read kvproj en place | OUVERTE (08-04, note user) — la spec v2 a vidé ses deux rôles : le write est une SÉLECTION (ph.7 : tout write-réseau est mort, la cascade ne fabrique plus le gist) et la lecture ambiante est portée par kvproj/S19 (la cascade n'en est plus le « précurseur »). Ce qui reste = un point de prélèvement (S10). NE PAS l'arracher du ckpt 10B pour la greffe — elle y est cuite ; la décision porte sur le PROCHAIN from-scratch (qui révise aussi le mid-training : ChatML + instructions + données à dépendance mémoire via protocole RESET + corpus agentique + curriculum de longueur) | si elle ne paie plus ses FLOPs avec kvproj actif → supprimée du from-scratch (bonus : chemin de décodage simplifié, cf. chantier perf) ; sinon elle garde sa place et S10 se lit dedans |
| S21 | le RTI a-t-il encore un emploi ? (retriever dur + injection préfixe + copy-head vs kvproj seul) | A/B pendant la phase de GREFFE, kvproj actif : citation kvproj-seul vs kvproj+copy-head, sur DEUX pools de valeurs — tokens fréquents (facile) vs chaînes dures (codes XX-NNNNN, chiffres longs — la classe d'échec KT3/RTI/S5) ; adjudication nll_live absolu + grade + r@1 | OUVERTE (08-04, question user « pourquoi garder le RTI si kvproj fait tout ? » — principe : si on peut simplifier, il faut le faire) — le jouet cite déjà par kvproj seul (grade 0,97 `span` ph.11, sans préfixe ni copy-head) et §2.8 nomme déjà la hiérarchie « remplaçant candidat du top-k dur » ⇒ le rôle RETRIEVE est absorbable ; ce qui retient = le rôle POINTEUR : le chemin soft re-décode la valeur par la lm_head et dégénère sur les chaînes arbitraires (S5 : lettres copiées, queues 5 chiffres → répétitions ; 350M : ` mosaic`→` mask`), la copy-head garantit le token-exact par construction (mélange d'ids aux logits). Indice gratuit avant la greffe = zzr311-316 : si les valeurs `span` faciles copient proprement en `prov`, l'intuition kvproj-seul se renforce | GARDE (note user 08-04) : le pool dur se lit EN CROISANT le knob S10 (point de prélèvement : les embeddings couche 0 sont fidèles au re-décodage mais auto-similaires sur les chiffres — l'échec S5 est un échec d'AVANCEMENT entre lignes quasi identiques, pas de fidélité ; un tap mi-stack/scalar-mix les désambiguïse, au prix connu ph.10 que les états cachés purs ne citent pas → tester le MIX) ; et si kvproj-seul échoue partout, bras de contrôle « + estampille intra-span » avant de conclure — S17 l'a économisée sur des spans de MOTS, hors du domaine des chaînes auto-similaires. Ne pas condamner kvproj-seul sur un seul réglage. si kvproj-seul tient le pool DUR au 350M → tout l'étage RTI tombe (retriever W_q, préfixe 28 pseudo-tokens, copy-head, layout §4.3) : il ne reste qu'un store + une attention ; sinon la copy-head garde son emploi de pointeur et seul le retriever dur meurt (absorbé par S19/hiérarchie §2.8) |

## 4. Corrections pré-run (état)

### 4.1 Sélection de surface à priorité de span — LIVRÉ (08-01)

KT3 a renversé le diagnostic initial (« biais chiffres ») : le vrai bug était le
budget k fixe contre la longueur du tour (code T=35 : inclusion pleine 0,000).
Correctifs livrés et re-mesurés (inclusion pleine 1,000 sur les quatre
strates) : sélection à priorité de span (`build_group(…, copy_mask)`), l'env
marque tous les atomes citables, k = 13. Le fallback SIF pur garde le statut
d'oracle, remplacé à terme par le write `<think>`.

### 4.2 Contrefactuel leave-one — REQUIS pour S12

Le leave-one-slot existant ([rti_learner.py](deepseek_v4_mini/rti_learner.py))
mesure la valeur de ce qui est écrit, pas de ce qui est perdu. Sans « rejouer en
préservant la ligne évincée », le choix de propagation n'a pas de gradient et le
GRPO dégénère en FIFO. Extension du harnais requise avant KT6.

### 4.3 Normalisation du layout d'injection — RÈGLE PERMANENTE

Le circuit de copie est câblé sur le layout du préfixe (ph.8 : copie 0,66 si
ordre = score, 0,21 si aléatoire). Nombre de groupes et ordre normalisés EN DUR
entre train et déploiement (k fixe, ordre = score, padding à vide) ; **train et
éval isomorphes obligatoire** (leçon KT2 : un bras au layout jamais entraîné ne
mesure rien). Falaise cartographiée au KT9 (S15).

### 4.4 Copy-head : le régime négatif — REQUIS pour S14

Canal d'hallucination-par-copie : sur retrieval erroné ou ligne périmée, il
copie la mauvaise valeur avec confiance. Le micro-train n'a jamais vu de
négatif ; `log_alpha` n'est calibré sur rien. À entraîner avec wrong-slot /
slot-périmé avant toute démo.

## 5. Kill-tests — verdicts et restants

| # | Question | État / verdict |
|---|---|---|
| 1 | Surface > compaction textuelle ? | **FAIT 08-03** : OUI au rappel (0,343 vs 0,227, Δ+0,115), NON au nll (+0,223) — le claim surface vit sous fenêtre bornée ; Δnll ne juge plus la citation (règle §1). Mur localisé = sélection (0,717 hit / 0,000 miss). |
| 2 | Le précédent Coconut nous tue-t-il ? | **FAIT 08-03, NON CONCLUANT** : bras think jamais entraîné (gap de layout). Re-run = S11. |
| 3 | Biais chiffres | **FAIT 08-01, renversé** : bug réel = k fixe vs longueur de tour. Correctifs §4.1 livrés. |
| 5 | Sonde de supersession discriminante | env « correction rétractée » (le bon slot est l'ANCIEN) — FIFO/récence score 0 par construction ; toute sonde sans ce cas est morte. À lancer (gate de S6). |
| 6 | Propagation GRPO-able ? | = S12. |
| 7 | Copie indue | = S14. |
| 8 | Dilution ambiante | = S8. |
| 9 | Falaise layout | = S15. |
| 10 | Exposure bias du write | jouet : A/B scheduled-sampling sur les entrées du write, métrique = horizon de divergence. Levier de S13. |

(KT4 — mémoire d'activations TBPTT inter-tours — acté sans le payer : gist =
compression SIF assumée, §1.)

## 6. Batterie d'évaluation

### 6.1 Aile conditionnement — paires contrastives Δnll

Banque contient un fait → Δnll entre continuation cohérente et incohérente :
langue (barreau 1, signal massif) ; registre (barreau 2) ; règle de contenu
(barreau 3, liaison profonde, casse en premier). Contrôles obligatoires : gist
aléatoire → Δ ≈ 0 exigé ; contenu inversé → Δ s'inverse ; occupancy variable.
Le 2AFC de la grille ph.10 (live/shuf/none) est le format validé au jouet.

### 6.2 Aile citation — rappel exact apparié

Env recall et harnais RL existants tels quels ([recall_env.py](deepseek_v4_mini/recall_env.py),
vies scriptées appariées = ancres GiGPO) + négatifs §4.4 + taux de copie indue.
Métrique = grade de rappel, jamais Δnll (règle §1).

### 6.3 Oubli — la sonde qui discrimine

Après un fait nouveau, la paire contrastive s'inverse **ET** l'env « correction
rétractée » (KT5) sépare politique apprise d'heuristique de récence. Les deux,
jamais la première seule. S'y ajoute (design 08-03) : la survie du fait ANCIEN
encore utile (NY reste citable après Austin) — l'oubli correct est sélectif,
pas total.

### 6.4 Le protocole central : sessions à travers RESET

Le terrain où le claim vit. Sessions > fenêtre, resets aux frontières, fenêtre
bornée identique : **banque** (traverse, 0 token) vs **compaction** (résumé
porté dans chaque fenêtre) vs **RAG transcript** (réinjection payée) vs
**rien** (borne basse). Métriques : les deux ailes, mesurées après n resets, en
fonction de n. Appariement en forwards totaux. KT1 en est la première
instance ; le protocole complet ajoute n > 1.

### 6.5 Stabilité : horizon de divergence en boucle fermée

Métrique héritée (divergence en 5-10 tours au step ~309, invisible en
teacher-forced). Toutes les sondes 6.1-6.3 étant teacher-forcées, **aucun
tableau vert ne vaut sans ce chiffre**. Doublé par S13 (self-writes ON/OFF).

## 7. Mapping vers l'existant

**Se garde tel quel** : write SIF teacher a=1e-4 ; RTI + copy-head (aile
citation) ; harnais RL complet (CISPO, GiGPO, vies appariées, leave-one-slot) ;
env recall ; infra désagrégée workers/learner ; perf decode-dispatch (rebind,
banque fixe) ; recette TF+distill+anneal β→0 pour l'amorçage des lectures.

**Se modifie** : slot → lignes UNIFIÉES de largeur d (§2.2 — le write actuel
produit des lignes de nature gist, il lui manque la sélection d'embeddings
natifs couche 0) ; read fast-weight → attention kvproj sur la vue plate
(§2.3-2.4 ; la cascade actuelle est le précurseur de la lecture ambiante) ;
écrasement/choix-de-slot → append FIFO + propagation (§2.3) ; harnais +
leave-one (§4.2) ; layout normalisé (§4.3) ; copy-head ré-entraîné avec
négatifs (§4.4).

**Se gèle** : curriculum Coconut / CoT latente (horizon post-claim) ; GRPO
recall (en pause @17) ; claim « budget visible » ; tout claim sur le *contenu*
des lignes ; fast-weight read (perdant ph.10, retiré).

**S'écrit** (nouveau, petit) : env RESET multi-fenêtres n > 1 (§6.4) ; env
correction rétractée (§6.3) ; paires contrastives (§6.1) ; bras compaction et
RAG (baselines) ; rotations âge/canal dans kvproj (après S1, S3-S5) ;
propagation procédurale + compteurs (après S6).

## 8. Baselines du papier

| Baseline | Statut | Danger |
|---|---|---|
| Compaction textuelle | **tueuse n°1** — première manche jouée (KT1 : battue au rappel sous reset, gagne le nll) | attaque les deux ailes ; ne survit au claim qu'en fenêtre bornée |
| RAG sur transcript | tueuse pour le rappel-performance | neutralisée par la reformulation §1 (mécanisme unifié, budget total) |
| CoT explicite appariée | tueuse pour l'ambiant | S11 (le KT2 du 08-03 ne compte pas) |
| DeltaNet / linear attention | objection reviewer n°1, engagement public | à **démontrer** contre un hybride (contrôlabilité : lignes adressables, actions RL) — renforcée par ph.10 : notre fast-weight interne a perdu contre l'attention |
| RMT / compressive | dangereuse pour le gist seul | scinder les claims : notre différenciateur est l'aile surface, que RMT n'a pas |
| Steering vectors / prefix-tuning / TTT | couverte | démo banque vs TTT : 0,95/0,78 à 1/138ᵉ du coût |
| Engram / K3 / V4 (SOTA 2026) | faire-valoir | lookup statique, lecture seule — « Engram sépare le savoir, nous séparons l'état de session » |

## 9. Risques ouverts (assumés, pas résolus)

1. La propagation apprise peut ne pas être GRPO-able (signal rare/retardé) —
   sortie : maintenance purement procédurale assumée (S12 tranche).
2. Le conditionnement peut rester superficiel (langue passe, règles de contenu
   non) — l'échelle 6.1 le détecte barreau par barreau.
3. L'horizon de divergence peut ne pas monter sans casser la recette
   TF+distill — arbitrage sur le résultat de KT10, au jouet.
4. La dilution peut imposer un max_mem petit qui étrangle le rappel — S8 donne
   le trade-off chiffré ; la hiérarchie (S9) est la sortie par le haut, des
   stores différenciés par lecture le dernier recours (au prix du « un seul
   store » — à ne payer que contraint).
5. Les rotations peuvent casser le circuit de copie (câblé layout) — S5 le
   mesure séparément sur r@1 et taux de copie avant d'acter.
