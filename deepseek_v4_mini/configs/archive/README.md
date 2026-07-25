# Configs archivées — arcs clos

Ces configs ne sont plus des points d'entrée du programme courant. Elles
restent versionnées parce que `FINDINGS.md`, `README.md` et les sondes de
`deepseek_v4_mini/analysis/` les citent nommément : ce sont les pièces
justificatives des verdicts, pas du code mort.

Rien ici n'est à relancer tel quel. Pour le programme courant, voir les
configs à la racine de `configs/` (phase 1 SIF `v350_*`, SFT `sft_sota_*`,
RL `rl_*`) et `configs/farm/` (ablations 350M en cours).

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

Les sondes de `analysis/` (ttt_demo, gate_probe, switch_probe_k2,
superposition_probe, …) pointent ici et restent exécutables.

## `mechanism/` — l'arc mécanisme natif v2/v3 (2026-07-09 → 2026-07-16)

From-scratch natif sur du vrai code/texte : `code_defer_native_v1..v2d`,
les greffes SmolLM2 (`sft_smollm_v*`), et sous `farm/` les balayages du rig
(v2b capacité/init, v2e diversité, v2f adressage, v2g carry, v2h stack,
v3 cascade lite/deep/reach, validations 135M).

Arc **clos le 2026-07-16** sur un lot ferme tout vert (adressage réel
−0.41..−0.54, capdeep horizon 2049+ steps, reach-back > deep, cross-modal
dans les 2 sens, curriculum warm-restart validé) : le stack compose, cap mis
sur la phase 2 / 350M.
