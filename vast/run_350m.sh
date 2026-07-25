#!/usr/bin/env bash
# dsv6 — TEMPLATE du vrai run 350M sur Vast.ai (préemption-safe, sync HF Hub).
# NE PAS lancer avant : (a) verdict sweep A100 (batch/GPU), (b) verdicts
# rehearsals 110/111 (composition + depth), (c) config 350M finale committée.
#
# Boucle de vie : restore depuis HF -> trainer --resume -> sidecar hf_sync
# pousse last.pt + jalons pendant le run. Préemption/reboot => l'onstart
# rejoue tout, le run reprend au dernier checkpoint SAUVÉ (= vérifié dehors).
#
# Variables (via `vastai create instance ... --env '-e X=...'`) :
#   HF_TOKEN   (OBLIGATOIRE, write ; jamais en clair dans l'onstart)
#   HF_REPO    défaut kkuette/dsv6-350m (repo privé, créé si absent)
#   CFG        défaut deepseek_v4_mini/configs/v350_phase1_10b.yaml
set -uo pipefail

WORK=/workspace
REPO_URL=${REPO_URL:-https://github.com/kkuette/thought-bank.git}
BRANCH=${BRANCH:-sft-persona-350m}
HF_REPO=${HF_REPO:-kkuette/dsv6-350m}
CFG=${CFG:-deepseek_v4_mini/configs/v350_phase1_10b.yaml}
# Les chemins de config sont relatifs à ${TB_ROOT} (défaut "." = dépôt local).
# Sur le pod, la racine est le volume persistant.
export TB_ROOT=$WORK

[ -n "${HF_TOKEN:-}" ] || { echo "HF_TOKEN manquant — abandon avant de brûler du GPU"; exit 1; }

mkdir -p $WORK && cd $WORK
export HF_HOME=$WORK/hf
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if [ ! -d thought-bank ]; then
  git clone --depth 1 -b "$BRANCH" "$REPO_URL" thought-bank
fi
cd thought-bank
git pull --ff-only || true
pip install -q -r requirements.txt huggingface_hub

# SAVE_DIR = ce que la config déclare (avec ${TB_ROOT} étendu), pas une valeur
# recopiée à la main : le sidecar hf_sync doit synchroniser le dossier où le
# trainer écrit VRAIMENT, sinon il pousse un dossier vide.
SAVE_DIR=$(python -c "import sys; from deepseek_v4_mini.paths import load_yaml; print(load_yaml(sys.argv[1])['training']['save_dir'])" "$CFG")
[ -n "$SAVE_DIR" ] || { echo "save_dir illisible dans $CFG — abandon avant de brûler du GPU"; exit 1; }
echo "=== save_dir (depuis la config) : $SAVE_DIR ==="
mkdir -p "$SAVE_DIR"

# 1. restaurer le dernier checkpoint sauvé (no-op au premier boot)
python vast/hf_sync.py --restore --repo "$HF_REPO" --dir "$SAVE_DIR"

# 2. sidecar de sync (survit au trainer, tué à la fin du script)
python vast/hf_sync.py --repo "$HF_REPO" --dir "$SAVE_DIR" > $WORK/hf_sync.log 2>&1 &
SYNC_PID=$!
trap 'kill $SYNC_PID 2>/dev/null' EXIT

# 3. le run (le trainer écrit last.pt + jalons save_every dans SAVE_DIR,
#    qui est lu de la config plus haut : plus rien à recopier à la main)
python -m deepseek_v4_mini.code_defer_native "$CFG" --resume 2>&1 | tee $WORK/train.log
RC=${PIPESTATUS[0]}

# 4. laisser le sidecar pousser le dernier état avant de rendre la main
sleep 150
echo "=== RUN TERMINÉ rc=$RC — vérifier hf_sync.log (dernier SAUVÉ) avant de détruire l'instance ==="
