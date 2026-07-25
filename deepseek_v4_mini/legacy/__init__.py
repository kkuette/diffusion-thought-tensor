"""Arc clos : la greffe SmolLM2 (2026-07-07 → 2026-07-09).

Voie « real data » d'avant le from-scratch natif : prendre un SmolLM2 gelé et
lui greffer la banque (write head + read fast-weights), plutôt que d'entraîner
un hôte à partir de zéro. L'arc est clos — le programme courant est natif
(`code_defer_native`) — mais le code reste versionné et exécutable : ses
verdicts sont cités dans FINDINGS.md et ses configs vivent dans
`configs/archive/mechanism/sft_smollm_v*.yaml`.

Rangé ici surtout pour lever une ambiguïté de nommage : `sft_train.py` n'est
PAS le SFT de la phase 2. La phase 2 passe par le bloc `chat:` de
`code_defer_native` (configs `sft_sota_*`, `sft_persona_*`).

Correspondance des commandes (le déplacement date du 2026-07-25) :

    python -m deepseek_v4_mini.sft_train    → deepseek_v4_mini.legacy.sft_train
    python -m deepseek_v4_mini.code_train   → deepseek_v4_mini.legacy.code_train

Les modules : `smollm_graft` (module de greffe), `code_train` (trainer
bi-optimiseur hôte AdamW + greffe Muon), `sft_train` (SFT complet sur tâches
verbales), `verbal_tasks` (générateurs de règles + tours UltraChat),
`bank_viz` (figures de banque, appelé par sft_train seulement).
"""
