"""Une seule boucle de génération, pour tout le dépôt.

Il y en avait trois — `code_defer_native._greedy`, `rl_disagg.decode_lb`,
`chat_cli._decode` — nées par copier-coller et divergentes depuis :

  * seule celle de `rl_disagg` protégeait l'autocast par `out.is_cuda` (les
    deux autres lèvent sur CPU) ;
  * seule celle de `chat_cli` savait échantillonner (température, top-p) ;
  * **toutes les trois câblaient B=1 en dur** via `if int(nt) == stop_id`, qui
    lève dès qu'on décode deux lignes à la fois. C'est précisément ce qui
    empêchait de batcher les G rollouts d'un groupe GRPO — le décodage étant
    le goulot déclaré de `rl_disagg`.

Ce module remplace les trois par `generate()`, avec un masque `done` PAR LIGNE.

Le décodage n'écrit pas dans la banque (`write=False`) : la banque produite par
le forward de décodage était de toute façon jetée, et le write est un pool
attentionnel sur toute la séquence — le seul morceau non cachable de manière
incrémentale, donc autant ne pas le payer à chaque token.
"""
from __future__ import annotations

import torch


def _pick(logits, temp: float, top_p: float, generator=None):
    """Token suivant depuis les logits de dernière position : greedy si
    temp == 0, sinon température + noyau (top-p)."""
    if temp <= 0:
        return logits.argmax(-1, keepdim=True)
    probs = torch.softmax(logits.float() / temp, dim=-1)
    if top_p < 1.0:
        sp, si = probs.sort(dim=-1, descending=True)
        keep = sp.cumsum(-1) - sp < top_p       # garde au moins le token de tête
        sp = sp * keep
        idx = torch.multinomial(sp / sp.sum(-1, keepdim=True), 1,
                                generator=generator)
        return si.gather(-1, idx)
    return torch.multinomial(probs, 1, generator=generator)


@torch.no_grad()
def generate(model, prefix, *, bank=None, layer_banks=None, max_new: int = 48,
             stop_id=None, amp: bool = False, temp: float = 0.0,
             top_p: float = 1.0, generator=None, write: bool = False,
             use_cache: bool = False):
    """Génère au plus `max_new` tokens après `prefix`, depuis la banque COURANTE.

    prefix : [B, T]. Renvoie (gen, lens) :
      * `gen`  [B, n] les tokens générés, n ≤ max_new ;
      * `lens` [B]    longueur utile de chaque ligne, `stop_id` INCLUS quand il
        a été émis. Ce qui suit `lens[i]` sur la ligne i est du remplissage
        produit pendant que les autres lignes finissaient — le lire serait une
        erreur.

    L'arrêt est par ligne : la boucle ne s'interrompt que lorsque TOUTES les
    lignes ont émis `stop_id` (ou au plafond `max_new`), et une ligne finie
    n'influence plus le résultat des autres — chaque ligne est indépendante,
    seule sa propre colonne compte dans son forward.
    """
    B = prefix.size(0)
    dev = prefix.device
    out = prefix
    done = torch.zeros(B, dtype=torch.bool, device=dev)
    lens = torch.full((B,), max_new, dtype=torch.long, device=dev)

    # Avec cache : le préfixe passe une fois, puis chaque token ne traverse le
    # stack qu'UNE fois au lieu de refaire tout le préfixe (l'attention relit
    # ses blocs compressés déjà fermés ; tout le reste est positionwise).
    #
    # OFF PAR DÉFAUT, et ce n'est pas de la prudence rituelle. Le cache est
    # exact — `attention._selftest` le vérifie en float64 à 1e-13 près,
    # frontières de bloc comprises — mais « exact » se paie en ULP, et CE
    # modèle amplifie les ULP : le routage MoE est un top-k DUR, donc une
    # différence de 1e-16 sur deux experts quasi ex æquo bascule l'expert, et
    # la sortie change de O(0.1) sur les logits. Mesuré ici (97M jouet, poids
    # NON entraînés, donc pire cas) : erreur logits médiane 3e-1 pour un écart
    # top1−top2 de 4e-2, soit un token différent presque à chaque pas.
    #
    # C'est la même pathologie que le recompute du grad_checkpoint (voir
    # `runtime.build_fwd`, gc_save_topk) : rien n'est faux, c'est un autre
    # échantillon aussi légitime. Mais l'activer par défaut rendrait les évals
    # incomparables aux runs passés sans que rien ne le signale. À mesurer sur
    # un checkpoint ENTRAÎNÉ (routage décisif ⇒ bascules rares) avant de
    # basculer le défaut ; le RL, lui, peut l'activer tout de suite : ses
    # rollouts sont des échantillons, et c'est là que le gain est.
    cache = model.make_cache() if (use_cache and hasattr(model, "make_cache")) else None
    fed = prefix

    n = 0
    for i in range(max_new):
        with torch.autocast("cuda", dtype=torch.bfloat16,
                            enabled=amp and out.is_cuda):
            o = model(fed, init_mem=bank, layer_banks=layer_banks, write=write,
                      **({"cache": cache} if cache is not None else {}))
        nt = _pick(o["logits"][:, -1], temp, top_p, generator)   # [B,1]
        out = torch.cat([out, nt], dim=1)
        fed = nt if cache is not None else out
        n = i + 1
        if stop_id is not None:
            hit = (nt.squeeze(1) == stop_id) & ~done
            lens = torch.where(hit, torch.full_like(lens, n), lens)
            done = done | hit
            if bool(done.all()):
                break

    return out[:, prefix.size(1):], lens.clamp(max=n)


def trim(gen, lens):
    """Les lignes de `generate` coupées à leur longueur utile."""
    return [gen[i, :int(lens[i])] for i in range(gen.size(0))]


# ── self-test ────────────────────────────────────────────────────────────────

def _selftest() -> None:
    """L'invariant qui compte : décoder 4 lignes ensemble donne EXACTEMENT ce
    que donne chacune décodée seule. C'est la propriété qui rend légitime le
    batch des G rollouts d'un groupe GRPO — sans elle, batcher change les
    récompenses, donc le gradient.
    """
    import torch.nn as nn

    torch.manual_seed(0)
    V = 23

    class Toy(nn.Module):
        """Un modèle jouet mais au bon CONTRAT : dict en sortie, kwargs du
        trainer acceptés, et une dépendance réelle au préfixe entier (somme
        cumulée) pour qu'un mélange entre lignes se voie."""

        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(V, 12)
            self.head = nn.Linear(12, V)
            self.calls = 0

        def forward(self, x, init_mem=None, layer_banks=None, write=True,
                    **kw):
            self.calls += 1
            h = self.emb(x).cumsum(dim=1)
            o = {"logits": self.head(h), "balance_loss": torch.zeros(())}
            o["mem_bank"] = init_mem if not write else (
                init_mem if init_mem is not None else torch.zeros(x.size(0), 0, 4))
            return o

    m = Toy().eval()
    STOP = 7
    prefixes = [torch.randint(0, V, (1, 5)) for _ in range(4)]

    # ── ligne à ligne (le comportement historique, B=1) ──────────────────────
    solo = []
    for p in prefixes:
        g, l = generate(m, p, max_new=12, stop_id=STOP)
        solo.append(g[0, :int(l[0])])

    # ── les mêmes, batchées ────────────────────────────────────────────────
    batched, lens = generate(m, torch.cat(prefixes, 0), max_new=12, stop_id=STOP)
    got = trim(batched, lens)
    for i, (a, b) in enumerate(zip(solo, got)):
        assert torch.equal(a, b), \
            f"ligne {i} : le batch change la génération\n  seule={a}\n  batch={b}"

    # ── l'arrêt est bien PAR LIGNE, et lens inclut le stop ──────────────────
    class Stopper(nn.Module):
        """Ligne i émet STOP au (i+1)-ème token — les lignes finissent à des
        moments différents, ce qui est exactement le cas que `int(nt)` cassait."""

        def __init__(self):
            super().__init__()
            self.T0 = None

        def forward(self, x, init_mem=None, layer_banks=None, write=True, **kw):
            B, T = x.shape
            if self.T0 is None:
                self.T0 = T
            k = T - self.T0                       # combien de tokens déjà générés
            lg = torch.zeros(B, T, V)
            for i in range(B):
                nxt = STOP if k >= i else (i + 1)
                lg[i, -1, nxt] = 10.0
            return {"logits": lg, "mem_bank": init_mem,
                    "balance_loss": torch.zeros(())}

    s = Stopper()
    g, lens = generate(s, torch.zeros(4, 3, dtype=torch.long), max_new=9,
                       stop_id=STOP)
    assert lens.tolist() == [1, 2, 3, 4], f"arrêt par ligne cassé : {lens.tolist()}"
    for i in range(4):
        assert int(g[i, int(lens[i]) - 1]) == STOP, "lens n'inclut pas le stop"
    assert g.size(1) == 4, \
        f"la boucle doit s'arrêter dès que TOUTES les lignes ont fini (vu {g.size(1)})"

    # ── sans stop_id : on va jusqu'au plafond, toutes lignes utiles ─────────
    g, lens = generate(m, prefixes[0], max_new=6, stop_id=None)
    assert g.shape == (1, 6) and lens.tolist() == [6]

    # ── temp=0 déterministe ; temp>0 pilotable par un generator ─────────────
    a, _ = generate(m, prefixes[0], max_new=8, stop_id=None, temp=0.0)
    b, _ = generate(m, prefixes[0], max_new=8, stop_id=None, temp=0.0)
    assert torch.equal(a, b), "greedy non déterministe"
    g1 = torch.Generator().manual_seed(1234)
    g2 = torch.Generator().manual_seed(1234)
    s1, _ = generate(m, prefixes[0], max_new=8, stop_id=None, temp=0.9,
                     top_p=0.95, generator=g1)
    s2, _ = generate(m, prefixes[0], max_new=8, stop_id=None, temp=0.9,
                     top_p=0.95, generator=g2)
    assert torch.equal(s1, s2), "même graine ⇒ même échantillon"

    # ── le décodage n'écrit PAS dans la banque par défaut ───────────────────
    seen = {}

    class Spy(Toy):
        def forward(self, x, init_mem=None, layer_banks=None, write=True, **kw):
            seen["write"] = write
            return super().forward(x, init_mem, layer_banks, write, **kw)

    generate(Spy().eval(), prefixes[0], max_new=2, stop_id=None)
    assert seen["write"] is False, "le décodage paie un write qu'il jette"

    # ── CPU : l'autocast cuda ne doit pas être armé (le piège des 2 copies) ──
    g, _ = generate(m, prefixes[0], max_new=3, stop_id=None, amp=True)
    assert g.shape == (1, 3), "amp=True sur CPU doit passer sans lever"

    # ── le cache KV rend la MÊME séquence que le recompute complet ──────────
    # Sur le vrai stack, en float64 : c'est la précision qui isole la question
    # « le cache est-il correct ? » de la question « le routage MoE
    # bascule-t-il sur un ULP ? ». La première se teste ici, la seconde est une
    # propriété du modèle (voir le commentaire de use_cache).
    from .config import ThoughtBankConfig
    from .model import ThoughtBankLM

    # top_k_experts == n_experts et top_k_csa ≥ nb de blocs : AUCUN top-k dur ne
    # tranche ici. C'est délibéré — sinon le test ne mesurerait pas le cache
    # mais la sensibilité chaotique du routage MoE aux ULP, qui est une
    # propriété du modèle et se documente ailleurs (cf. use_cache).
    torch.manual_seed(0)
    cfg = ThoughtBankConfig(vocab_size=61, d_model=32, n_layers=3, n_heads=2,
                            d_head=8, csa_m=3, hca_m=5, top_k_csa=64, n_win=4,
                            d_latent_q=8, n_groups=1, n_experts=2, n_shared=1,
                            top_k_experts=2, d_ff=32, mem_dim=16, max_mem=3,
                            mem_seed_slots=2, mem_read_rank=4, sinkhorn_iters=5)
    real = ThoughtBankLM(cfg).double().eval()
    bank64 = torch.randn(2, 3, 16, dtype=torch.float64)
    for T0 in (1, 2, 3, 5, 6, 11):                  # autour des deux m
        pr = torch.randint(0, 61, (2, T0))
        no_c, l1 = generate(real, pr, bank=bank64, max_new=13, use_cache=False)
        wi_c, l2 = generate(real, pr, bank=bank64, max_new=13, use_cache=True)
        assert torch.equal(no_c, wi_c) and torch.equal(l1, l2), (
            f"cache ≠ recompute pour un préfixe de {T0} (float64)\n"
            f"  sans cache : {no_c}\n  avec cache : {wi_c}")

    print("decode self-test: OK (batch B=4 IDENTIQUE au ligne-à-ligne, arrêt "
          "par ligne + lens incluant le stop, plafond sans stop, greedy "
          "déterministe et échantillonnage reproductible, write=False par "
          "défaut, amp inerte sur CPU, cache KV = recompute complet en float64 "
          "sur le vrai stack pour 6 longueurs de préfixe)")


if __name__ == "__main__":
    _selftest()
