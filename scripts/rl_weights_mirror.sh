#!/usr/bin/env bash
# Miroir local des poids du learner RL (NAS → disque local du rig).
# Un seul transfert NAS par publication ; les N workers du rig lisent la
# copie locale via TB_WEIGHTS_MIRROR=<dst_root> (le worker lit
# <dst_root>/weights/). Tourne sur l'hôte du rig, hors queue (pas de GPU).
#
# Usage : rl_weights_mirror.sh <src_root_nfs> <dst_root_local> [poll_s]
# Ex.   : nohup scripts/rl_weights_mirror.sh /mnt/tb/rl/disagg_350m \
#             /var/tmp/tb_mirror/disagg_350m 15 >/var/tmp/tb_mirror.log 2>&1 &
#
# Sûr par construction : le learner publie atomiquement (tmp+rename puis
# LATEST), donc un fichier nommé par LATEST est complet ; ici on copie en
# .tmp puis rename, et LATEST local n'avance qu'après — les workers ne
# voient jamais une copie partielle. Si le NAS élague le fichier pendant la
# copie (miroir très en retard), cp échoue et on retente au tour suivant.
set -u
SRC="${1:?src_root (NFS)}/weights"
DST="${2:?dst_root (local)}/weights"
POLL="${3:-15}"
KEEP=3

mkdir -p "$DST"
echo "mirror: $SRC -> $DST (poll ${POLL}s, keep $KEEP)"
while true; do
  name=$(cat "$SRC/LATEST" 2>/dev/null || true)
  if [ -n "${name:-}" ] && [ ! -f "$DST/$name" ] && [ -s "$SRC/$name" ]; then
    if cp "$SRC/$name" "$DST/.tmp.$name" 2>/dev/null; then
      mv "$DST/.tmp.$name" "$DST/$name"
      printf '%s' "$name" >"$DST/.LATEST.tmp" && mv "$DST/.LATEST.tmp" "$DST/LATEST"
      echo "mirror: $name ($(date '+%F %H:%M:%S'))"
      ls "$DST"/step_*.pt 2>/dev/null | head -n -"$KEEP" | xargs -r rm -f --
    else
      rm -f "$DST/.tmp.$name"
    fi
  fi
  sleep "$POLL"
done
