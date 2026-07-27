# Ferme de rigs — mode d'emploi

Architecture : `server0` (Unraid) exporte le share NFS `llm_research` ; chaque rig
(Debian 12 natif) monte le share sur `/mnt/tb` et fait tourner un worker par GPU.
Pas de scheduler : la file est un répertoire, un job est un fichier bash.

## 1. Côté Unraid (fait une fois)

- `Settings → NFS → Enable`.
- Share `llm_research` : `Export: Yes`, `Security: Private`, règle :
  `<SOUS_RESEAU>/24(sec=sys,rw,no_root_squash,insecure)`
  — `insecure` requis pour les clients WSL2 (NAT = ports source >1024) ; sans
  effet pour les rigs natifs.
- Unraid sert du NFSv3 (v4 refusé) → monter avec `vers=3,nolock`.
- Le share sur pool cache SSD de préférence (l'array shfs est lent en écriture).

## 2. Côté rig (une fois par rig, depuis Debian 12 fraîche)

```bash
git clone https://github.com/kkuette/thought-bank /opt/thought-bank
sudo /opt/thought-bank/scripts/farm/setup_rig.sh <IP_NAS> llm_research
sudo reboot
# après reboot :
nvidia-smi                # les 6 cartes doivent apparaître
for i in 0 1 2 3 4 5; do sudo systemctl enable --now tb-worker@$i; done
systemctl status tb-agent # activé par setup_rig.sh : alimente le dashboard
```

Côté VM data, une fois : `sudo /opt/thought-bank/scripts/farm/setup_dashboard.sh`.

## 3. Lancer des jobs

Un job = un fichier `*.job` (script bash) déposé dans `/mnt/tb/queue/`.
Il s'exécute sur UN GPU (déjà isolé via `CUDA_VISIBLE_DEVICES`), cwd = repo,
venv actif, variables `TB_MNT`, `WORKER` disponibles. Priorité = ordre
lexicographique du nom (`00_urgent.job` passe avant `50_sweep.job`).

Exemple — une seed du 97M v2c :

```bash
cat > /mnt/tb/queue/10_v2c_97m_seed44.job <<'EOF'
cd deepseek_v4_mini
python code_defer_native.py \
  --config configs/archive/mechanism/code_defer_native_v2c_varlen.yaml \
  --seed 44 \
  --out $TB_MNT/checkpoints/v2c_97m_s44 \
  --log $TB_MNT/runs/v2c_97m_s44
EOF
```

Cycle de vie : `queue/*.job` → `queue/running/<worker>__nom.job` →
`queue/done/` ou `queue/failed/`. Log worker : `runs/<nom>.workerlog`.
Rejouer un échec : `mv queue/failed/x.job queue/`.

## 4. Surveiller

Dashboard : `http://<IP_VM_DATA>:8787` (service `tb-dashboard`, installé par
`setup_dashboard.sh`). Il agrège les `status/*.json` des rigs (service
`tb-agent`), la file de jobs, et **les runs RL de `rl/*/`**.

Un bandeau collant en haut répond à « ça va ? » sans scroller : une pastille par
nœud, une par run RL vivant avec son step, la file, les échecs. Le reste est en
cartes, sur une grille qui se réorganise avec la largeur de l'écran ; l'état de
santé se lit à la bordure gauche de chaque carte.

Pretraining : dernier step, `ic`, s/step, et le dernier GAP par domaine.

RL désagrégé : le learner ne passe pas par la file (il tourne à la main sur la
3090) — il est lu via ses JSONL, pas via un log. La carte du run donne le step
et sa progression, cinq tuiles (reward, ce, write%, lag, s/step) avec
micro-courbe et flèche de tendance, puis le **découpage par environnement** —
dont le `grade` (taux d'appels justes, pass-rate exec) qui est le vrai suivi, le
reward agrégé mélangeant des échelles incompatibles.

La flèche de tendance porte une polarité : un reward qui monte est vert, une
entropie croisée qui monte est rouge, `write%` reste neutre (la politique
d'écriture est ce que le RL doit trouver, pas une note).

Les workers RL n'apparaissent qu'**une fois**, dans la carte du run, avec le nom
de leur job. Un job `rl_worker` qui n'a écrit aucune métrique n'est pas masqué
pour autant : il reste dans « jobs actifs », puisque c'est justement celui qu'il
faut voir.

Un worker RL n'imprime quasiment rien sur stdout : ne pas juger sa santé à son
`.workerlog`. Son horloge, c'est son JSONL, et le seuil de silence est calé sur
son débit (un step learner 350M dure ~13 min : le seuil fixe de 300 s des jobs
de pretraining y voyait une panne permanente).

Si le dashboard devient injoignable, la page **garde le dernier instantané** et
l'annonce dans le bandeau, au lieu de s'effacer au moment où on en a besoin.

En terminal, même lecture sans navigateur :

```bash
python scripts/farm/rl_status.py --tb /mnt/tb          # tous les runs
python scripts/farm/rl_status.py --tb /mnt/tb 350m     # un seul
```

Déployer une modif du dashboard (jamais de copie hors du repo) :

```bash
ssh <VM_DATA> 'git -C /opt/thought-bank pull && sudo systemctl restart tb-dashboard'
```

Le front vit dans `scripts/farm/web/` (preact + htm, vendorés dans
`web/vendor/` : la VM data n'a pas de réseau, et il n'y a ni npm ni build).
Les fichiers sont relus à chaque requête, donc **un changement de CSS ou de JS
ne demande qu'un `git pull`** — le `systemctl restart` ne sert que si
`farm_dashboard.py` ou `rl_status.py` change. Contrôles avant de pousser :

```bash
python scripts/farm/rl_status.py && python scripts/farm/farm_dashboard.py --selftest && node --check scripts/farm/web/dashboard.mjs
```

## 5. Pré-tokenisation (VM `data`, CPU, sans GPU)

VM Debian dédiée : NFS monté, venv `/opt/tb-venv` (torch CPU), repo
`/opt/thought-bank`, `HF_HOME=/mnt/tb/data/hf_cache`, SSH par clé uniquement.
Usage :

```bash
ssh <user>@<IP_VM_DATA>
export TB_REPO=/opt/thought-bank
/opt/tb-venv/bin/python /mnt/tb/scripts/prebuild_data.py \
  --cache-dir /mnt/tb/data_cache \
  /opt/thought-bank/deepseek_v4_mini/configs/<config>.yaml
```

(`TB_REPO` : le script est lancé depuis la copie NAS, hors repo — la variable
pointe l'import `deepseek_v4_mini`.) L'alternative Docker-sur-Unraid
(`prebuild_data.sh`) reste disponible mais la VM est le chemin par défaut.

### Ancienne méthode (server0, Docker, CPU)

`prebuild_data.py` rejoue la construction de données du trainer (mêmes clés de
cache md5, y compris les tokens spéciaux `<think>`/`<blank>`) et remplit le
cache partagé `${TB_ROOT}/data_cache/` (TB_ROOT = le montage, exporté par gpu_worker.sh). Validé : 4/4 cache hits sur la config v2c.

Depuis le terminal Unraid :

```bash
/mnt/user/llm_research/scripts/prebuild_data.sh \
  deepseek_v4_mini/configs/archive/mechanism/code_defer_native_v2c_varlen.yaml
```

**Convention** : toute config destinée à la ferme déclare
`data.cache_dir: ${TB_ROOT}/data_cache` (le défaut du trainer est un `data_cache/`
local qui raterait le cache partagé). Le cache des corpus actuels est déjà
semé (16 entrées, ~1 Go). Le HF cache partagé (`data/hf_cache`) évite aussi
les re-téléchargements.

## Notes matérielles

- Risers PCIe x1 : sans effet (jobs mono-GPU, données lues en séquentiel sur NFS).
- Power limit 150 W/carte via `gpu-powerlimit.service` (boot).
- 8 GB VRAM : OK 97M (~3,7 GB/conv batchée), 135M à valider en batch 1 ragged,
  350M impossible (A100 spot requis).
- Premier job de validation sur UNE carte avant toute campagne : run 97M 500 steps
  (VRAM réelle, débit vs 3090 attendu ~55-60 %, stabilité riser sous charge).
