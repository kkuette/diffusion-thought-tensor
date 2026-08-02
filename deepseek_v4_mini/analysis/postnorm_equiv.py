"""dsv6 — E2 : l'injection post-norm est-elle un canal de citation gratuit ?

La proposition (PLAN_EXPERIENCES §2 E2) : les embeddings étant LIÉS
(`lm_head.weight = embed.weight`, model.py:467), l'espace d'entrée et l'espace
de sortie sont le même. Ajouter un vecteur de contenu `c` au residual stream
JUSTE APRÈS `norm_out` et AVANT `lm_head` produit donc exactement le biais
pointeur `c·Eᵀ` en réutilisant le matmul de la tête, au lieu de payer une tête
dédiée `d_m × V`. Facteur 42 annoncé.

Ce que ce script mesure, et pourquoi il ne mesure pas ce que la proposition
croyait :

  1. L'identité algébrique est VRAIE et on la vérifie au bit près (float64) et
     à l'ULP près (float32). Injecter APRÈS `norm_out` est bien la condition —
     le contre-exemple pré-norm est rendu, et l'écart n'y est pas petit : la
     RMSNorm est non linéaire, elle re-normalise ce qu'on vient d'ajouter.

  2. Le « facteur 42 » compare à une tête pointeur `d_m × V` **qui n'existe
     nulle part dans le dépôt**. La tête de citation réellement implémentée est
     `rti_copy.CopyHead`, un pointer-generator sur les POSITIONS DU PRÉFIXE
     (P ≤ 28 colonnes), qui ne matérialise jamais un [B,T,V] : elle coûte
     0,61 M MACs/token, soit MOINS que l'homme de paille de la comparaison.

  3. Surtout, l'injection post-norm produit un BIAIS ADDITIF aux logits — la
     classe de fonctions de `PointerReadout` (toy_read_lab.py:832), qui fait
     `logits = (x + g·scale·sel) @ Eᵀ`, c'est-à-dire l'identité EXACTE de la
     proposition, en r3 sans même la projection. Cette classe est MESURÉE MORTE :
     12 runs r3 à d=512, grade held-out entre 0.000 et 0.100, contre 0.281
     (strate code 0.708) pour l'injection de pseudo-tokens natifs. Un biais
     additif peut faire gagner un token qu'AUCUNE ligne ne porte (HAB-719 →
     HQR-719) ; le mélange normalisé de `rti_copy` ne le peut pas, par
     construction. L'économie de 0,22 M MACs/token s'achèterait au prix de cette
     propriété-là.

Le livrable de ce script n'est donc pas une amélioration : c'est la rétractation
de E2, chiffrée ici plutôt que par transitivité avec les runs r3.

Repro :
    PYTHONPATH=. python deepseek_v4_mini/analysis/postnorm_equiv.py
"""
from __future__ import annotations

import sys

import torch

sys.path.insert(0, ".")
from deepseek_v4_mini.config import ThoughtBankConfig
from deepseek_v4_mini.model import ThoughtBankLM

torch.manual_seed(0)

# Dimensions RÉELLES du 350M (sft_recall_350m_copy.yaml) — pour la table de MACs.
D_MODEL, MEM_DIM = 768, 512
V = 49154                    # len(SmolLM2-135M) = 49152, + <think> + <blank>
TOP_K, GROUPS = 13, 2
P = GROUPS * (TOP_K + 1)     # 28 colonnes de préfixe (rti.group_prefix)


def equivalence(dtype: torch.dtype) -> tuple[float, float]:
    """(écart post-norm, écart pré-norm) en max|Δ| sur les logits.

    Modèle jouet mais MÉCANIQUE RÉELLE : le `norm_out` et le `lm_head` du dépôt,
    avec le weight tying asserté. C'est l'opération de E2 isolée, rien d'autre.
    """
    cfg = ThoughtBankConfig(vocab_size=257, d_model=64, n_layers=2, n_heads=2,
                            d_head=32, d_ff=128, mem_dim=32, max_mem=4)
    mdl = ThoughtBankLM(cfg).to(dtype).eval()
    assert mdl.lm_head.weight is mdl.embed.weight, "weight tying attendu"
    E = mdl.lm_head.weight                                  # [V, d]

    B, T = 3, 7
    H = torch.randn(B, T, cfg.d_model, dtype=dtype)         # residual pré-norm
    c = torch.randn(B, T, cfg.mem_dim, dtype=dtype)         # contenu mémoire
    proj = torch.nn.Linear(cfg.mem_dim, cfg.d_model, bias=False).to(dtype)
    g = 0.37                                                # la porte, un scalaire

    with torch.no_grad():
        Hn = mdl.norm_out(H)
        base = mdl.lm_head(Hn)
        # (a) injection APRÈS norm_out, lm_head réutilisée
        inj = mdl.lm_head(Hn + g * proj(c))
        # (b) tête pointeur dédiée : le biais c·Eᵀ ajouté aux logits
        bias = base + g * (proj(c) @ E.t())
        post = float((inj - bias).abs().max())
        # (c) contre-exemple : injecter AVANT la norme
        pre = float((mdl.lm_head(mdl.norm_out(H + g * proj(c))) - bias).abs().max())
    return post, pre, float(base.abs().max())


def macs() -> None:
    lm = D_MODEL * V
    ded = MEM_DIM * V                       # tête pointeur dédiée (l'homme de paille)
    pn = MEM_DIM * D_MODEL                  # injection post-norm (proj seule)
    # rti_copy.CopyHead : W_c d×d, scores q·h_préfixe sur P colonnes, porte h·w_g,
    # agrégation par classe d'égalité d'id sur P². Aucun matmul en V.
    cp = D_MODEL * D_MODEL + P * D_MODEL + D_MODEL + P * P
    print(f"\n== MACs par token (d={D_MODEL}, d_m={MEM_DIM}, V={V}, P={P}) ==")
    for name, m, note in (
            ("lm_head (référence)", lm, ""),
            ("tête pointeur d_m × V", ded, "N'EXISTE PAS dans le dépôt"),
            ("injection post-norm", pn, "la proposition E2"),
            ("rti_copy.CopyHead", cp, "ce qui est RÉELLEMENT implémenté"),
    ):
        print(f"  {name:<24} {m/1e6:>7.2f} M   {100*m/lm:>5.2f} % du lm_head   {note}")
    print(f"\n  « facteur 42 » annoncé   : {ded/pn:.0f}× — mais contre une tête inexistante.")
    print(f"  face à ce qui existe     : {cp/pn:.1f}× ({(cp-pn)/1e6:.2f} M MACs/token"
          f" = {100*(cp-pn)/lm:.2f} % du lm_head)")
    print(f"  rti_copy coûte DÉJÀ moins que l'homme de paille : "
          f"{cp/1e6:.2f} M contre {ded/1e6:.2f} M.")


def main() -> None:
    print("== équivalence numérique : lm_head(norm_out(H) + g·proj(c))"
          "  ==  logits + g·(proj(c)·Eᵀ) ==")
    for dt, tol in ((torch.float64, 1e-9), (torch.float32, 1e-3)):
        post, pre, scale = equivalence(dt)
        ok = "OK" if post < tol else "ÉCHEC"
        print(f"  {str(dt):<16} post-norm max|Δ| = {post:.3e}  [{ok}]"
              f"   | échelle des logits {scale:.2f}")
        print(f"  {'':<16} PRÉ-norm  max|Δ| = {pre:.3e}"
              f"   ← la RMSNorm est non linéaire : elle re-normalise ce qu'on"
              f" vient d'ajouter")
        assert post < tol, (dt, post)
    print("\n  L'identité tient, et la condition est bien « APRÈS norm_out » :"
          " l'écart pré-norm")
    print("  n'est pas un arrondi, il est de l'ordre des logits eux-mêmes.")

    macs()

    print("\n== VERDICT ==")
    print("  L'algèbre de E2 est juste. Sa conclusion ne l'est pas, pour deux")
    print("  raisons indépendantes :")
    print("   1. le gain est chiffré contre une tête d_m × V qui n'existe nulle")
    print("      part ; ce qui est implémenté coûte déjà moins qu'elle ;")
    print("   2. un biais ADDITIF aux logits est la classe de fonctions de")
    print("      PointerReadout (toy_read_lab.py:832) — l'identité exacte de la")
    print("      proposition — et elle est mesurée MORTE : 12 runs r3, grade")
    print("      held-out <= 0.100 contre 0.281 pour l'injection native.")
    print("  E2 échangerait 0.22 M MACs/token contre la normalisation du")
    print("  mélange, c'est-à-dire contre la seule propriété qui empêche de")
    print("  citer un token qu'aucune ligne ne porte. RETIRÉ.")


if __name__ == "__main__":
    main()
