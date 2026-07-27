#!/usr/bin/env bash
# Installe le dashboard de la ferme en service, sur la machine qui monte le
# share (VM data par défaut). Usage: sudo ./setup_dashboard.sh [port]
# Idempotent : relançable, il réécrit l'unité et redémarre le service.
#
# Le service lance le script DU REPO. Le dashboard a déjà été patché à la main
# sur la VM sans commit une fois (voir setup_rig.sh) : la version déployée avait
# divergé et personne ne savait laquelle tournait. Mettre à jour = git pull ici,
# puis `systemctl restart tb-dashboard`.
set -euo pipefail

PORT="${1:-8787}"
MNT=/mnt/tb
REPO=/opt/thought-bank

[ -d "$REPO" ] || { echo "repo absent : git clone https://github.com/kkuette/thought-bank $REPO"; exit 1; }
mountpoint -q "$MNT" || { echo "$MNT n'est pas monté"; exit 1; }

# python3 système suffit : farm_dashboard.py et rl_status.py sont stdlib pur,
# pas de torch, pas de venv (c'est la contrainte qui les garde déployables ici).
cat > /etc/systemd/system/tb-dashboard.service <<EOF
[Unit]
Description=thought-bank dashboard de la ferme (port ${PORT})
After=remote-fs.target
Requires=remote-fs.target
StartLimitIntervalSec=0
[Service]
Environment=TB_MNT=${MNT}
ExecStart=/usr/bin/python3 ${REPO}/scripts/farm/farm_dashboard.py ${PORT} ${MNT}
Restart=always
RestartSec=10
[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable --now tb-dashboard.service
systemctl restart tb-dashboard.service
sleep 1
systemctl --no-pager --lines=5 status tb-dashboard.service || true
echo
echo "dashboard : http://$(hostname -I | awk '{print $1}'):${PORT}"
