"""dsv6 — que porte une LIGNE NATIVE ? (banque [M, mem_dim] → [M, mem_dim, d_model])

Le slot n'est plus un code compressé : c'est une MATRICE de `mem_dim` lignes
sélectionnées après la dernière RMSNorm ([model.py:540](../model.py:540)),
complétée à zéro. Le `thought_head` (pooling appris T→1) et la projection
`d_model → mem_dim` disparaissent tous les deux du chemin de surface.

Ce script mesure ce que porte une telle ligne, parce que les deux chemins de
lecture DÉJÀ dans le dépôt consomment des espaces DIFFÉRENTS :

  • côté ENTRÉE — `inject` ([model.py:512](../model.py:512)), le préfixe de
    `rti.build_prefix`, qui pose `embed_w[ids] + type_vec`. C'est le chemin qui
    a gagné au jouet ph.7 (code 0.708 contre 0.000 pour tout readout appris) —
    mais il y injectait des embeddings d'ENTRÉE, pas des états post-norm.
  • côté LOGITS — la géométrie de E2 : `lm_head(norm_out(H) + g·row)` ajoute
    exactement `g·(row @ Eᵀ)` (vérifié au bit dans `postnorm_equiv.py`). Là une
    ligne post-norm est chez elle par construction : c'est l'espace que
    `lm_head` consomme, et le tying `lm_head.weight = embed.weight` fait de
    `row @ Eᵀ` un vrai biais pointeur.

Trois questions, dans l'ordre où elles peuvent tuer le design :

  1. CONTENU (décisive). `h_t` est l'état qui PRÉDIT le token `t+1`. La surface
     qu'on veut citer est le token `x_t` qui est À la position t. Une ligne
     post-norm encode-t-elle la surface sélectionnée, ou la prédiction
     suivante ? Si c'est la prédiction, la banque de surface stocke un décalage
     d'un token et il faut stocker `h_{t-1}` — un correctif d'une ligne, mais
     seulement si on le sait avant le run.
  2. RANG DE CITATION. Où se classe `x_t` dans `h_t @ Eᵀ` ? C'est exactement ce
     que le canal logit peut faire ressortir ; un rang médian élevé le condamne
     sans appel.
  3. ÉCHELLE. ‖h_t‖ contre ‖E[id]‖. Un écart de facteur k rend l'injection
     CÔTÉ ENTRÉE hors distribution par l'échelle seule — réparable par un
     rescale, mais il faut le chiffre avant de l'appeler un détail.

Aucune de ces trois n'est réglée par les mesures existantes : le jouet ph.7-8
injectait `embed_w[ids]` (discret, lossless, borné au vocabulaire) ; ici la
ligne est continue et vit dans l'espace de SORTIE.

Repro (CPU, quelques minutes) :
    PYTHONPATH=. python deepseek_v4_mini/analysis/native_row_channel.py
    PYTHONPATH=. python deepseek_v4_mini/analysis/native_row_channel.py \
        --ckpt <autre.pt> --cfg <autre.yaml>
"""
from __future__ import annotations

import argparse
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, ".")
from deepseek_v4_mini.infra.config import ThoughtBankConfig
from deepseek_v4_mini.core.model import ThoughtBankLM
from deepseek_v4_mini.infra.paths import load_yaml

torch.manual_seed(0)

CKPT = "/mnt/tb/checkpoints/v350_sft_recall_rti/step_1000.pt"
CFG = "deepseek_v4_mini/configs/sft_recall_350m_rti.yaml"

# Les quatre strates de l'env recall, en clair : ce sont les surfaces que la
# banque doit restituer au token près. `code` est celle qui a fait 0.000 partout
# où un readout appris a été essayé.
TEXTS = [
    "The user's name is Barnaby Whitlock and he lives in Trondheim.",
    "Remember the access code: HQR-719. Do not share it.",
    "def compute_mosaic_offset(grid, seed=8317):\n    return (grid * seed) % 4096\n",
    "Her favourite colour is vermilion and she was born in 1987.",
    "The server is at 192.168.1.21 and the port is 8787.",
]


def load(ckpt: str, cfg_path: str):
    # Les configs 350M sont IMBRIQUÉES (`model:`), celles du papier sont plates ;
    # `from_yaml` refuse explicitement les premières plutôt que de rendre un
    # config tout par défaut (config.py:273). On accepte les deux formes.
    raw = load_yaml(cfg_path)
    cfg = (ThoughtBankConfig(**raw["model"]) if "model" in raw
           else ThoughtBankConfig.from_yaml(cfg_path))
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    # `vocab_size` du YAML est un placeholder : train.py le fixe au moment du run
    # depuis le tokenizer (SmolLM2 49152 + `<think>` + `<blank>`). On le reprend
    # du checkpoint, seule source qui ne peut pas mentir.
    cfg.vocab_size = int(sd["model"]["embed.weight"].shape[0])
    model = ThoughtBankLM(cfg)
    missing, unexpected = model.load_state_dict(sd["model"], strict=False)
    if missing or unexpected:
        print(f"  (state_dict : {len(missing)} manquants, {len(unexpected)} en trop)")
    model.eval()
    return cfg, model, sd.get("step")


@torch.no_grad()
def post_norm_rows(model, ids: torch.Tensor) -> torch.Tensor:
    """[T, d] — la sortie de `norm_out`, c'est-à-dire LA LIGNE qu'on stockerait.

    Capturée par hook plutôt que recalculée : c'est le tenseur exact que
    `model.py:540` produit, et le seul moyen d'être sûr qu'on ne mesure pas une
    reconstruction approchée du chemin réel.
    """
    grabbed = []
    h = model.norm_out.register_forward_hook(lambda m, i, o: grabbed.append(o))
    try:
        model(ids, compute_logits=True, write=False)
    finally:
        h.remove()
    assert grabbed, "hook sur norm_out jamais déclenché"
    return grabbed[-1][0]                      # [T, d] — lane 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=CKPT)
    ap.add_argument("--cfg", default=CFG)
    ap.add_argument("--tokenizer", default="HuggingFaceTB/SmolLM2-135M")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)

    cfg, model, step = load(args.ckpt, args.cfg)
    E = model.lm_head.weight                                   # [V, d]
    tied = model.lm_head.weight is model.embed.weight
    print(f"ckpt step {step} | d_model {cfg.d_model} | V {E.size(0)} | "
          f"tying {'OUI' if tied else 'NON (le canal logit N’EST PAS un pointeur)'}")
    assert tied, "sans tying, `row @ Eᵀ` n'est pas un biais pointeur"

    Ef = E.detach().float()
    En = F.normalize(Ef, dim=-1)
    e_norm = Ef.norm(dim=-1)

    cur_cos, nxt_cos, cur_rank, h_norm, cur_top1 = [], [], [], [], []
    prev_rank, prev_hit, hard_rank = [], [], []
    for txt in TEXTS:
        ids = torch.tensor([tok(txt)["input_ids"]])
        H = post_norm_rows(model, ids).detach().float()        # [T, d]
        x = ids[0]                                             # token À la position t
        logits = H @ Ef.t()                                    # [T, V] = ce que lit lm_head
        y = logits.argmax(-1)                                  # token PRÉDIT par h_t
        Hn = F.normalize(H, dim=-1)
        # t < T-1 seulement : la dernière ligne n'a pas de « suivant » observé.
        for t in range(H.size(0) - 1):
            cur_cos.append(float(Hn[t] @ En[x[t]]))
            nxt_cos.append(float(Hn[t] @ En[y[t]]))
            # rang de x_t (la SURFACE) dans le classement que la ligne induit
            cur_rank.append(int((logits[t] > logits[t, x[t]]).sum()) + 1)
            cur_top1.append(int(y[t] == x[t]))
            h_norm.append(float(H[t].norm()))
        # Le correctif évident au décalage : stocker la ligne PRÉCÉDENTE, celle
        # qui prédit x_t. Elle cite x_t exactement quand le modèle l'avait déjà
        # prédit — donc jamais sur ce qu'une banque de rappel sert à porter.
        for t in range(1, H.size(0)):
            r = int((logits[t - 1] > logits[t - 1, x[t]]).sum()) + 1
            prev_rank.append(r)
            prev_hit.append(int(y[t - 1] == x[t]))
            if y[t - 1] != x[t]:               # token NON prédit = le citable
                hard_rank.append(r)

    def stat(v, name, fmt="{:.3f}"):
        s = sorted(v)
        n = len(s)
        med = s[n // 2]
        print(f"  {name:<34} médiane {fmt.format(med):>10}   "
              f"p10 {fmt.format(s[n//10]):>10}   p90 {fmt.format(s[9*n//10]):>10}")

    print(f"\n== 1. CONTENU — la ligne porte-t-elle la SURFACE ou la PRÉDICTION ? "
          f"(n={len(cur_cos)} positions)")
    stat(cur_cos, "cos(h_t, E[x_t])  SURFACE en t")
    stat(nxt_cos, "cos(h_t, E[argmax])  PRÉDICTION")
    frac = sum(cur_top1) / len(cur_top1)
    print(f"  argmax(h_t @ Eᵀ) == x_t : {frac:.3f} "
          f"({'la ligne pointe sur elle-même' if frac > 0.5 else 'la ligne pointe AILLEURS'})")

    print("\n== 2. RANG DE CITATION — où se classe x_t dans h_t @ Eᵀ ?")
    stat(cur_rank, "rang de x_t (1 = citable direct)", "{:.0f}")
    top10 = sum(r <= 10 for r in cur_rank) / len(cur_rank)
    print(f"  x_t dans le top-10 de sa propre ligne : {top10:.3f}")

    print("\n== 2bis. LE CORRECTIF DE DÉCALAGE, ET POURQUOI IL NE SUFFIT PAS")
    print("  On stocke h_{t-1} (la ligne qui PRÉDIT x_t) au lieu de h_t :")
    stat(prev_rank, "rang de x_t dans h_{t-1} @ Eᵀ", "{:.0f}")
    print(f"  cite x_t au rang 1 : {sum(prev_hit)/len(prev_hit):.3f} "
          f"— c'est la PRÉCISION du modèle, pas une propriété de la banque.")
    if hard_rank:
        stat(hard_rank, "…restreint aux tokens NON prédits", "{:.0f}")
        t10 = sum(r <= 10 for r in hard_rank) / len(hard_rank)
        print(f"  ces tokens dans le top-10 : {t10:.3f}   (n={len(hard_rank)})")
        print("  Or ce sont EXACTEMENT ceux qu'une banque de rappel doit porter :")
        print("  un nom, un code, une constante sont citables parce qu'ils sont")
        print("  IMPRÉVISIBLES. Une ligne post-norm est une prédiction ; là où la")
        print("  prédiction tombe juste, la banque était inutile.")

    print("\n== 3. ÉCHELLE — injection CÔTÉ ENTRÉE")
    stat(h_norm, "‖h_t‖ (la ligne stockée)")
    stat(e_norm.tolist(), "‖E[id]‖ (ce qu'attend `inject`)")
    ratio = (sorted(h_norm)[len(h_norm) // 2]
             / float(e_norm.median()))
    print(f"  rapport des médianes : ×{ratio:.1f}"
          f"{'  ← hors distribution par l’échelle seule' if ratio > 3 or ratio < 0.33 else ''}")

    print("\n== VERDICT")
    if frac > 0.5:
        print("  La ligne post-norm pointe MAJORITAIREMENT sur son propre token :")
        print("  le canal logit peut citer la surface sans décalage, et le stockage")
        print("  de `h_t` est le bon indice.")
    else:
        print("  La ligne post-norm pointe sur le token SUIVANT, pas sur le sien, et")
        print("  décaler d'un cran ne répare rien : la ligne décalée cite bien, mais")
        print("  seulement là où le modèle prédisait déjà — c'est-à-dire là où la")
        print("  banque ne sert à rien.")

    print("\n  CE QUE CETTE MESURE NE DIT PAS (et qu'il ne faut pas lui faire dire).")
    print("  Elle lit la ligne AVEC `lm_head`. C'est le seul readout GRATUIT (tying),")
    print("  et c'est tout l'argument d'économie du canal post-norm. Elle ne dit pas")
    print("  que x_t est absent de h_t : le stream porte le token courant, mais")
    print("  `norm_out`+`lm_head` sont entraînés à le SUPPRIMER (ne pas répéter).")
    print("  Une sonde linéaire apprise le retrouverait sans doute — sauf qu'à ce")
    print("  moment on a re-payé un readout appris, et c'est précisément la classe")
    print("  de fonctions mesurée MORTE en held-out (r3, ≤ 0.100 contre 0.281).")
    print("\n  LE VRAI ARBITRAGE, alors. Le RTI actuel stocke des ID de tokens et")
    print("  injecte `E[id]` : lossless par construction, zéro readout. La ligne")
    print("  post-norm est plus RICHE (elle porte le contexte) mais exige un décodage")
    print("  que l'ID n'exige pas. Le design ne se justifie que si ce surplus de")
    print("  contexte paie plus que le readout ne coûte — et le seul précédent")
    print("  chiffré qu'on ait sur ce readout est négatif.")
    print("\n  (Mesures géométriques. Le test FONCTIONNEL — injecter la ligne en")
    print("   préfixe et comparer la distribution à celle de E[x_t] — est le pas")
    print("   suivant, et lui seul dit si le stack la lit à l'entrée.)")


if __name__ == "__main__":
    main()
