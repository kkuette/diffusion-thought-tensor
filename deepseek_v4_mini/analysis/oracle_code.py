"""dsv6 — E0 : oracle d'expressivité. Le goulot est-il le WRITE ou le READ ?

Sur un checkpoint TOTALEMENT gelé, on optimise directement le code de slot par
descente de gradient. 32 nombres bougent (mem_dim), rien d'autre — ni le trunk,
ni l'hypernet, ni les autres slots. Le résultat est une BORNE SUPÉRIEURE : ce
qu'aucun schéma d'écriture ne peut dépasser avec ce read.

    m* = argmin_m  CE(cible | banque_{t-1} ∪ {m}) + λ‖m‖²

DEUX barreaux, parce qu'en un seul nombre la mesure ne localise rien (règle du
journal : « un grade composite à 0 ne localise rien ») :

  pres   cible = le SEGMENT DE PRÉSENTATION lui-même (positions y_j, j>=1 —
         la convention de supervision du tour 0, train.py::_build_synthetic_rule).
         C'est la borne d'un write AUTO-SUPERVISÉ : il n'a accès qu'à ce que la
         présentation contient, ce qui est exactement la situation du write réel.
  query  cible = les RÉPONSES AUX REQUÊTES inédites. C'est la borne du READ tout
         court : si même ce code-là échoue, la règle n'est pas exprimable dans
         la classe de fonctions du read, et aucun write ne sauvera l'affaire.

L'écart pres → query est lui-même un résultat : il dit si l'objectif
auto-supervisé est le bon signal d'entraînement, ce qui décide de E5a (remplacer
le teacher Fourier par m* amorti).

Chaque barreau part de DEUX initialisations — le code du write entraîné, et du
bruit — parce qu'un optimum atteint depuis le write seul ne prouve pas qu'on a
trouvé mieux que lui : on peut n'avoir jamais quitté son bassin.

Trois conditions (les trois premières lignes du tableau E0 de PLAN_EXPERIENCES) :

  train  règles VUES à l'entraînement. SANITÉ : l'oracle doit être >= le write
         entraîné. S'il ne l'est pas, le protocole est faux et on s'arrête là.
  held   règles jamais vues (le 0,79-1,00 du papier). Oracle >> write ⇒ le write
         laisse de la marge ⇒ E5. Oracle ≈ write ⇒ le write est déjà optimal ⇒
         la frontière est le read, et la revendication du papier se renforce
         AU NIVEAU ORACLE, qui est le niveau le plus fort possible.
  sub    y=(s−x)%128, HORS famille (papier §7 : hasard pour tous les bras).
         Oracle au-dessus du hasard ⇒ l'enveloppe était l'amortisation du write,
         pas la classe de fonctions du read ⇒ le §7 du papier est à réécrire.

Le protocole d'éval est celui de ttt_demo.py, recopié à l'identique (mêmes
conversations, même RNG CPU, même argmax sur le dernier logit) : sans ça la
comparaison au bras `bank` ne veut rien dire.

Repro :
    PYTHONPATH=. python deepseek_v4_mini/analysis/oracle_code.py <ckpt> --cfg <cfg>
    ... --train-pool     # règles VUES (sanité)
    ... --sub            # famille soustraction, hors famille
"""
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, ".")
from deepseek_v4_mini.infra.config import ThoughtBankConfig
from deepseek_v4_mini.core.model import ThoughtBankLM
from deepseek_v4_mini.train import _rule_space
from deepseek_v4_mini.infra.paths import load_yaml

torch.manual_seed(0)
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT = (sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else
        "checkpoints/multiturn_rule_k2_inter_s128_dsv4m/final.pt")
CFG = "deepseek_v4_mini/configs/archive/dsv4mini/multiturn_rule_k2_inter_s128_dsv4m.yaml"
if "--cfg" in sys.argv:
    CFG = sys.argv[sys.argv.index("--cfg") + 1]
USE_TRAIN = "--train-pool" in sys.argv
S, m, K, SYM_OFF = 128, 6, 2, 3
KEY_OFF = SYM_OFF + S
N_CONV, TURNS = 64, 8

# Optimisation du code. LR balayé comme ttt_demo balaie le sien : un oracle qui
# perd sur un mauvais LR ne mesure pas une classe de fonctions, il mesure un LR.
OPT_LRS = (0.03, 0.1, 0.3)
# Budget de pas. 50 suffit depuis l'init `write` (le code est déjà bon) mais PAS
# depuis `noise` : à 50 pas le meilleur point est encore le dernier et la CE
# chute encore — on mesurerait la convergence d'Adam, pas la borne. `--steps N`.
OPT_STEPS = 200
if "--steps" in sys.argv:
    OPT_STEPS = int(sys.argv[sys.argv.index("--steps") + 1])
EVAL_EVERY = max(1, OPT_STEPS // 20)
# λ‖m‖² : les codes du write passent par norm_write = RMSNorm(mem_dim), donc
# RMS ≈ 1 ; les codes Fourier du teacher aussi (train.py:527). Le rôle de λ est
# d'empêcher un code de partir à l'infini là où le read sature — pas de piloter.
# DÉFAUT 0 : pour une BORNE, tout terme qui éloigne du minimum de la CE ne peut
# que la rabaisser, et le RMS du m* retenu est rapporté, ce qui rend le contrôle
# d'échelle observable sans le payer. `--lambda X` remet le terme de la spec.
LAMBDA = 0.0
if "--lambda" in sys.argv:
    LAMBDA = float(sys.argv[sys.argv.index("--lambda") + 1])

raw = load_yaml(CFG)
cfg = ThoughtBankConfig.from_yaml(CFG)
model = ThoughtBankLM(cfg)
sd = torch.load(CKPT, map_location="cpu", weights_only=False)
model.load_state_dict(sd["model"])
model.eval().to(DEV)
model.requires_grad_(False)                 # TOUT est gelé. Seul m bougera.
P = sum(p.numel() for p in model.parameters())

_units, _n, TRAIN, HELD, _apply = _rule_space(raw["data"])
POOL = torch.tensor(TRAIN if USE_TRAIN else HELD)
FAM = "HELD (règles fraîches)"
if USE_TRAIN:
    FAM = "TRAIN (règles vues — sanité)"
if "--sub" in sys.argv:
    _apply = lambda rid, x: (rid - x) % S
    POOL = torch.arange(1, S)
    FAM = "SUBTRACTION y=(s−x)%S — HORS famille"

print(f"loaded step {sd['step']} | params {P/1e6:.2f}M | device {DEV.type} "
      f"| mem_dim {cfg.mem_dim} | pool = {FAM}")


# ── conversations : IDENTIQUES à ttt_demo.py (RNG CPU, même ordre) ───────────
def make_convs():
    convs = []
    for _ in range(N_CONV):
        ctxs = []
        for k in range(K):
            while True:
                s = int(POOL[int(torch.randint(0, len(POOL), (1,)))])
                if not ctxs or s != ctxs[0][0]:
                    break
            perm = torch.randperm(S).tolist()
            ctxs.append((s, perm[:m], perm[m:]))
        convs.append(ctxs)
    return convs


CONVS = make_convs()


def pres_rows(k):
    rows = []
    for ctxs in CONVS:
        s, ex, _ = ctxs[k]
        row = [KEY_OFF + k]
        for xi in ex:
            row += [SYM_OFF + xi, SYM_OFF + _apply(s, xi)]
        rows.append(row)
    return torch.tensor(rows)


def query_batch(k, idx):
    rows, ys = [], []
    for ctxs in CONVS:
        s, _, unseen = ctxs[k]
        q = unseen[idx % len(unseen)]
        rows.append([KEY_OFF + k, SYM_OFF + q])
        ys.append(SYM_OFF + _apply(s, q))
    return torch.tensor(rows), torch.tensor(ys)


@torch.no_grad()
def eval_queries(mdl, mem0=None, carry=False):
    """TURNS tours de requête ; carry=True enfile la banque (bras banque)."""
    mem, hits, tot = mem0, 0, 0
    for t in range(TURNS):
        k, idx = t % K, t // K
        xq, y = query_batch(k, idx)
        out = mdl(xq.to(DEV), init_mem=mem if carry else mem0, compute_logits=True)
        if carry:
            mem = out["mem_bank"]
        hits += int((out["logits"][:, -1].argmax(-1).cpu() == y).sum())
        tot += y.numel()
    return hits / tot


# ── bras de référence : le WRITE ENTRAÎNÉ ────────────────────────────────────
with torch.no_grad():
    mem_w = None
    for k in range(K):
        mem_w = model(pres_rows(k).to(DEV), init_mem=mem_w,
                      compute_logits=False)["mem_bank"]
    acc_write = eval_queries(model, mem_w, carry=True)
    acc_abl = eval_queries(model, None, carry=False)
M_TOT = mem_w.size(1)
rms_w = float(mem_w[:, -K:].pow(2).mean().sqrt())
print(f"\nwrite entraîné  acc={acc_write:.3f}   (banque {M_TOT} slots = "
      f"{M_TOT-K} seeds + {K} écrits, RMS des écrits {rms_w:.3f})")
print(f"ablaté          acc={acc_abl:.3f}   (hasard = {1/S:.3f})")

# Cibles pré-calculées, une fois : les requêtes servent de cible au barreau
# `query` ET de mesure aux deux barreaux — elles doivent être les mêmes.
QUERIES = [query_batch(t % K, t // K) for t in range(TURNS)]
PRES = [pres_rows(k).to(DEV) for k in range(K)]


def pres_loss(bank):
    """CE du SEGMENT de présentation, positions y_j pour j>=1.

    j=0 est exclu : la première paire ne peut pas être prédite, la règle n'est
    pas encore connue. C'est la convention de supervision du tour 0 dans
    train.py::_build_synthetic_rule (« loss on applied x-pos j>=1 ») — s'en
    écarter mesurerait un autre objectif que celui du write.
    """
    tot = 0.0
    for k in range(K):
        row = PRES[k]
        out = model(row, init_mem=bank, compute_logits=True, write=False)
        # row = [key, x0, y0, x1, y1, ...] ; y_j est en position 2j+2, prédit
        # depuis la position 2j+1. j>=1 ⇒ positions cibles 4, 6, ..., 2m.
        pos = torch.arange(1, m, device=DEV) * 2 + 1        # logits
        tot = tot + F.cross_entropy(
            out["logits"][:, pos].reshape(-1, out["logits"].size(-1)),
            row[:, pos + 1].reshape(-1))
    return tot / K


def query_loss(bank):
    """CE des RÉPONSES aux requêtes inédites — la borne du read tout court."""
    tot = 0.0
    for xq, y in QUERIES:
        out = model(xq.to(DEV), init_mem=bank, compute_logits=True, write=False)
        tot = tot + F.cross_entropy(out["logits"][:, -1], y.to(DEV))
    return tot / len(QUERIES)


def optimise(target: str, init: str, lr: float):
    """Optimise les K slots ÉCRITS, tout le reste gelé. Rend le MEILLEUR point
    de la trajectoire, step 0 INCLUS.

    Pourquoi le meilleur de la trajectoire et pas le dernier : la quantité
    revendiquée est une BORNE — « existe-t-il un code qui atteint X ? ». Prendre
    le dernier point mesurerait la convergence d'Adam sur une CE, pas la borne.
    Et l'objectif optimisé est une CE alors que la mesure est un argmax : baisser
    la CE peut déplacer quelques argmax dans le mauvais sens, ce qui rendait un
    majorant plus petit que le write — absurde pour un majorant.

    Step 0 est dans la recherche parce que le code du write EST un code
    atteignable : la borne est ainsi >= le write PAR CONSTRUCTION quand on part
    de lui, et le critère de sanité devient ce qu'il aurait toujours dû être —
    non pas « l'oracle bat-il le write ? » (trivial) mais « de COMBIEN ».

    Sélectionner le point sur la métrique qu'on rapporte est optimiste, et c'est
    assumé : on ne revendique pas la généralisation de cette sélection, on
    calcule un max sur des codes. À écrire dans le rapport.
    """
    frozen = mem_w[:, :-K].detach().clone()          # seeds : intouchés
    if init == "write":
        slots = mem_w[:, -K:].detach().clone()
    else:                                             # bruit à l'échelle du write
        g = torch.Generator(device="cpu").manual_seed(1234)
        slots = (torch.randn(mem_w[:, -K:].shape, generator=g).to(DEV)
                 * rms_w)
    slots.requires_grad_(True)
    opt = torch.optim.Adam([slots], lr=lr)
    loss_fn = pres_loss if target == "pres" else query_loss
    ce0 = None
    best = (-1.0, 0, 0.0, 0.0)                        # acc, step, ce, rms
    for step in range(OPT_STEPS + 1):
        bank = torch.cat([frozen, slots], dim=1)
        ce = loss_fn(bank)
        if ce0 is None:
            ce0 = float(ce)
        if step % EVAL_EVERY == 0 or step == OPT_STEPS:
            with torch.no_grad():
                acc = eval_queries(model, bank.detach(), carry=True)
                if acc > best[0]:
                    best = (acc, step, float(ce),
                            float(slots.pow(2).mean().sqrt()))
        if step == OPT_STEPS:
            break
        reg = LAMBDA * slots.pow(2).sum(-1).mean()
        opt.zero_grad()
        (ce + reg).backward()
        opt.step()
    return best[0], ce0, best[2], best[3], best[1]


print(f"\noracle ({OPT_STEPS} pas d'Adam sur {K}×{cfg.mem_dim} nombres, "
      f"λ={LAMBDA:g}, tout le reste gelé) :")
print(f"  {'cible':<6} {'init':<6} {'lr':>5} {'acc*':>7} {'@step':>6} "
      f"{'CE début→acc*':>16} {'RMS m*':>8}")
best = {}
t0 = time.perf_counter()
for target in ("pres", "query"):
    for init in ("write", "noise"):
        for lr in OPT_LRS:
            acc, ce0, ce1, rms, at = optimise(target, init, lr)
            key = (target, init)
            if acc > best.get(key, (-1,))[0]:
                best[key] = (acc, lr, ce0, ce1, rms, at)
            print(f"  {target:<6} {init:<6} {lr:>5g} {acc:>7.3f} {at:>6} "
                  f"{ce0:>7.3f}→{ce1:<8.3f} {rms:>8.3f}")
dt = time.perf_counter() - t0

print(f"\n== E0 — {FAM} == ({dt:.0f}s)")
print(f"  ablaté (plancher)          {acc_abl:.3f}   hasard {1/S:.3f}")
print(f"  write entraîné             {acc_write:.3f}")
for (target, init), (acc, lr, _, ce1, rms, at) in sorted(best.items()):
    tag = {"pres": "borne d'un write auto-supervisé",
           "query": "BORNE DU READ"}[target]
    print(f"  oracle {target:<5} init={init:<6} {acc:.3f}   (lr={lr:g}, "
          f"@{at}, CE {ce1:.3f}, RMS {rms:.3f})  {tag}")

# Les deux inits ne répondent PAS à la même question, et les confondre par un
# max rend le tableau illisible :
#   • init=write  — SANITÉ. Le best-of-trajectoire inclut le pas 0, qui EST le
#     write entraîné : ce bras ne peut donc jamais passer sous lui, et son seul
#     contenu informatif est « l'optimisation ne casse pas ce qui marchait ».
#     Quand le write est déjà au plafond, ce bras vaut le plafond, point.
#   • init=noise  — MESURE. Un code choisi librement, qui n'a jamais vu la
#     présentation. C'est lui qui dit ce que l'objectif porte comme signal.
bq_n, bp_n = best[("query", "noise")][0], best[("pres", "noise")][0]
bq_w = best[("query", "write")][0]
print()
if USE_TRAIN and bq_w < acc_write - 1e-9:
    print("  ⚠ SANITÉ EN ÉCHEC : l'oracle est SOUS le write entraîné sur les")
    print("    règles vues. Le protocole est faux — ne pas lire les autres")
    print("    conditions tant que ce n'est pas réparé.")
elif acc_write >= 0.999:
    print(f"  ⚠ MÉTRIQUE SATURÉE : le write entraîné est à {acc_write:.3f}. Il n'y a")
    print("    pas de marge à mesurer au-dessus — E0 ne peut PAS trancher la")
    print("    question « le write laisse-t-il de la place ? » sur cette tâche.")
    print("    Ce qui reste mesurable est le bras init=noise, qui ne dit pas")
    print("    combien il reste à gagner mais QUEL OBJECTIF porte le signal :")
    print(f"      query (oracle, non calculable au write)  {bq_n:.3f}")
    print(f"      pres  (auto-supervisé, calculable)       {bp_n:.3f}   "
          f"hasard {1/S:.3f}")
    print(f"      écart query − pres : {bq_n - bp_n:+.3f}")
    if bp_n < 4 * (1 / S):
        print("    ⇒ l'objectif auto-supervisé est QUASI VIDE : optimiser la CE de")
        print("      la présentation ne produit pas un code qui répond aux requêtes.")
        print("      La présentation CONTIENT les paires en contexte, donc la banque")
        print("      y est presque libre. E5a, qui proposait exactement cette cible")
        print("      pour remplacer le teacher Fourier, est RÉFUTÉ par ce chiffre.")
else:
    print(f"  écart oracle-query − write : {bq_n - acc_write:+.3f}  (init=noise)")
    print(f"  écart oracle-pres  − write : {bp_n - acc_write:+.3f}  (init=noise)")
    print(f"  écart query − pres         : {bq_n - bp_n:+.3f}   "
          "(> 0 ⇒ l'objectif auto-supervisé n'est pas le bon signal ⇒ E5a)")
