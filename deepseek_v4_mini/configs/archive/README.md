# Configs archivées — arcs clos

Ces configs ne sont plus des points d'entrée du programme courant. Elles
restent versionnées parce que `FINDINGS.md`, `README.md` et les sondes de
`deepseek_v4_mini/analysis/` les citent nommément : ce sont les pièces
justificatives des verdicts, pas du code mort.

Rien ici n'est à relancer tel quel. Pour le programme courant, voir les
configs à la racine de `configs/` (labo jouet `toy_read_lab*`, phase 1 SIF
`v350_*`, aile citation `sft_recall_350m_copy/rti`, RL `rl_*`) et
`configs/farm/` (ablations 350M).

## `dsv4mini/` — l'arc jouet (2026-06-30 → 2026-07-09)

Répertoire fermé, tâches synthétiques : règles continues multi-tours
(`multiturn_rule_*`, familles K=1/K=2, held-out, horizon, switch, joint),
gist latent (`multiturn_gist*`, `gist.yaml`), persistance de banque
(`code_persist*`), rappel adressable (`synth_recall.yaml`), et les jouets de
mise au point (`tiny`, `small`, `cpu_*`).

C'est l'arc du papier dsv4mini : la banque porte le gist, le write généralise,
le read ne généralise qu'au-delà d'un seuil de diversité. Il est **clos** — le
verdict « transport = recognition à répertoire fermé » interdit d'en tirer des
conclusions de capacité.

Les sondes de repro du papier (`analysis/` : ttt_demo*, switch_probe_k2,
superposition_probe) pointent ici et restent exécutables ; les autres sondes
historiques ont été retirées le 2026-08-05 (résultats conservés dans
`analysis/README.md`, fichiers dans git + archive NAS).

## `mechanism/` — l'arc mécanisme natif v2/v3 (2026-07-09 → 2026-07-16)

From-scratch natif sur du vrai code/texte : `code_defer_native_v1..v2d`,
les greffes SmolLM2 (`sft_smollm_v*`), et sous `farm/` les balayages du rig
(v2b capacité/init, v2e diversité, v2f adressage, v2g carry, v2h stack,
v3 cascade lite/deep/reach, validations 135M).

Arc **clos le 2026-07-16** sur un lot ferme tout vert (adressage réel
−0.41..−0.54, capdeep horizon 2049+ steps, reach-back > deep, cross-modal
dans les 2 sens, curriculum warm-restart validé) : le stack compose, cap mis
sur la phase 2 / 350M.

## `phase2_350m/` — arcs persona / SOTA / smokes de bring-up (2026-07-17 → 2026-07-30)

Rangées le 2026-08-05 (mise au propre sous SPEC_MEMOIRE_V2). Trois familles :

- **persona** (`sft_persona_350m*` : base, disc, surp, sif_hot) — arc persona
  CLOS 07-24 (Δnll +0,332 à β=0 mais lookup fermé).
- **SOTA/agentique** (`sft_sota_350m*`, `sft_sota_fromsif_*`,
  `sft_sota_smoke_97m`) — arc data SOTA + tools ; la seule survivante,
  `sft_sota_350m_valsif_stair.yaml`, reste en racine comme lignée d'`init_from`
  de `rl_disagg_350m.yaml`.
- **recall table** (`sft_recall_350m_table`, `_table2`) — bras morts (verdict
  SFT recall 07-30 : la classe de fonctions du read) ; `_stair` reste en racine
  (init_from de `rl_recall_350m.yaml`), `_copy`/`_rti` sont l'aile citation
  vivante.
- **smokes de bring-up** (`v350_compile_smoke`, `v350_sif_repass_smoke`,
  `v350_sync_val`, `v350_sif_warmcache`, `code_defer_native_350m_mix/_fineweb`)
  — ponctuels, remplacés par `v350_fastpath_smoke.yaml`.

Les configs restées en racine parce qu'elles sont l'ancêtre d'un ckpt encore
référencé : `sft_persona_350m_sif_repass(.yaml/_rearm)`, `v350_sif_repass`,
`sft_recall_350m_stair`, `sft_sota_350m_valsif_stair`.
