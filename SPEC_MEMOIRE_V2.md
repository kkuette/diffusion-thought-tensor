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

### 2.4 Deux lectures sur un seul store

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
