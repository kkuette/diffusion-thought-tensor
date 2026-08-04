#!/usr/bin/env bash
# dsv6 — enchaîne les self-tests hermétiques du paquet.
#
# Contrat : un module qui expose un self-test le lance quand on l'invoque SANS
# argument (`python -m deepseek_v4_mini.<mod>`). Avec un argument il fait son
# travail normal (config, sous-commande). Tout nouveau module qui suit ce
# contrat s'ajoute à MODULES ci-dessous.
#
# Aucun de ces tests ne touche au réseau ni au NAS : ils tournent sur CPU avec
# des données synthétiques, et c'est le geste à faire avant tout bring-up de pod.
#
#   scripts/selftest.sh                  # tout
#   scripts/selftest.sh rl_rewards       # un sous-ensemble
#   PY=/chemin/python scripts/selftest.sh
set -uo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-python}
TIMEOUT=${TIMEOUT:-900}

MODULES=(
  infra.paths               # expansion ${TB_ROOT} dans les configs
  infra.sched               # schedules LR/β : identiques au calcul historique
  infra.runtime             # ddp/tf32/compile/grad_checkpoint (grads identiques)
  infra.ckpt                # save atomique + resume : les RNG reprennent la SUITE
  core.mhc                 # RMSNorm + Sinkhorn : les deux garde-fous NaN du stack
  core.moe                 # routage, loss d'équilibrage, garde-fou sqrt(softplus)
  infra.muon                # Newton-Schulz + le piège √cols du rms_match
  core.memory              # write : FIFO, gates, pad-safety
  core.cascade             # débordement en 2 temps, read sans dilution d'init
  core.attention           # cache KV incrémental = recompute complet (float64)
  core.model               # forward, portage de banque, mem_read_layers
  infra.decode              # génération unique : batch B>1 == ligne-à-ligne
  infra.decode_graphs       # runner CUDA graphs : eager CPU == generate, capturabilité
  code_defer_native   # ALIGNEMENT DES CIBLES (loaders ne décalent pas, la loss si)
  data.streams             # registre nom→classe : tout résout, alias RL compris
  rl.exec_sandbox        # bac à sable d'exécution (le juge de l'env code)
  rl.rl_rewards          # extraction + grading des récompenses RL
  rl.rl_lives            # snapshot/restore/fork de banque, mixer d'envs
  rl_disagg           # GRPO désagrégé : hub, staleness, groupes de workers
  data.math_school_data    # stream leçons/drills
  data.persona_chat_data   # stream persona (recall, octaves d'âge)
  infra.cfg_schema          # schéma des configs : clé inconnue = arrêt (anti-dérive ast)
  data.chat_mix            # mix pondéré de streams derrière le contrat mono-stream
  data.chat_batch          # B lanes chat en lockstep par tour (padding, pas de ligne vide)
  rl.rti                 # retrieve-then-inject : groupe, FIFO, retriever, préfixe, pilote
  rl.rti_copy            # tête de copie aux logits : mélange, oracle, micro-train ON/OFF
  data.recall_env          # vies de rappel scriptées/appariées + reward vérifiable
  rl.rti_policy          # les 3 ACTIONS RL du bras rti : write, Plackett-Luce, décodage
  rl.rti_learner         # passe 2 rti : ratios on-policy, CISPO, GiGPO, leave-one-slot
)

[ $# -gt 0 ] && MODULES=("$@")

$PY -c "import torch" 2>/dev/null || {
  echo "!! torch introuvable pour '$PY' — active l'env (conda activate diffusion-thought)"
  echo "   ou passe PY=/chemin/vers/python"; exit 2; }

fail=0
for m in "${MODULES[@]}"; do
  printf "%-20s " "$m"
  out=$(timeout "$TIMEOUT" $PY -m "deepseek_v4_mini.$m" 2>&1); rc=$?
  if [ $rc -eq 0 ]; then
    echo "OK   $(echo "$out" | tail -1 | cut -c1-70)"
  else
    fail=1
    [ $rc -eq 124 ] && echo "TIMEOUT (${TIMEOUT}s)" || echo "ÉCHEC rc=$rc"
    echo "$out" | tail -20 | sed 's/^/    | /'
  fi
done

echo
[ $fail -eq 0 ] && echo "=== self-tests: TOUT VERT (${#MODULES[@]} modules) ===" \
                || echo "=== self-tests: AU MOINS UN ÉCHEC ==="
exit $fail
