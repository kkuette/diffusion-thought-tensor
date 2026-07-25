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
  streams             # registre nom→classe : tout résout, alias RL compris
  exec_sandbox        # bac à sable d'exécution (le juge de l'env code)
  rl_rewards          # extraction + grading des récompenses RL
  rl_lives            # snapshot/restore/fork de banque, mixer d'envs
  rl_disagg           # GRPO désagrégé : hub, staleness, groupes de workers
  math_school_data    # stream leçons/drills
  persona_chat_data   # stream persona (recall, octaves d'âge)
  chat_mix            # mix pondéré de streams derrière le contrat mono-stream
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
