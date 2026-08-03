"""dsv6 — SPEC_MEMOIRE_V2 §2.4 : OÙ meurt l'information dans la chaîne du gist ?

Diagnostic bon marché exigé par la spec avant de tourner l'un des trois cadrans
de budget (mem_dim / rang r du read / matrice par write). La chaîne du write
« gist » enchaîne trois compressions, et la spec dit qu'on ne tourne un cadran
qu'APRÈS avoir localisé la perte :

    tour (T×768)  ──pooling attentionnel `write_ctx_q`──▶  h_ctx (768)      (i)
    h_ctx  ──`thought_head` 768→512 + norm + gate α·p──▶  m stocké (512)    (ii)
    m  ──hypernet fw_A/fw_B (rang r=8) → delta sur une requête──▶  Δ (768)  (iii)

Sonde linéaire (ridge closed-form) des attributs ÉTIQUETÉS du tour écrit, aux
trois étages. L'étage où la sonde décroche désigne le cadran (spec §2.4) :

    faible dès (i)              → le POOLING est le goulot (T×768→768, ~50:1)
                                  ⇒ matrice par write (G ∈ R^{k×mem_dim})
    décroche entre (i) et (ii)  → la projection 768→512 coupe ⇒ élargir mem_dim
    forte en (ii), faible (iii) → la LECTURE plafonne ⇒ élever le rang r

Un quatrième étage de contrôle, `mean768`, est mesuré : le pooling par MOYENNE
uniforme des mêmes états de dernière couche. Il sépare deux causes très
différentes d'une perte à l'étage (i) — « il n'y a plus rien à pooler dans la
dernière couche » (mean768 faible aussi) contre « le pooling APPRIS jette »
(mean768 nettement au-dessus de pooled768). C'est le contrôle qui décide si le
correctif est le tap à ~2/3 de profondeur (spec §2.4, point de prélèvement) ou
bien les k têtes de pooling.

Trois attributs, tous sur les segs d'énonciation RÉELS de `RecallEnvStream`
(le chemin d'entraînement, pas une reconstruction) :

  slot     identité du slot interrogeable (12 classes, 4 strates) — ridge
           one-hot, split par SURFACE (aucun énoncé du test n'est vu au train).
  strate   persona / numeric / code / preference (4 classes) — idem.
  valeur   identité de LA VALEUR. Le vocabulaire des valeurs est OUVERT (≈1
           énoncé par valeur : classifier serait vide de sens, et un split par
           valeur rendrait toute classification impossible par construction).
           Formulation honnête et bien posée : ridge de l'étage vers
           l'EMBEDDING moyen des tokens de la valeur, puis RÉCUPÉRATION par
           cosinus parmi 8 candidats — évaluée sur des valeurs JAMAIS vues au
           train (split par valeur, strict). Chance = 1/8 = 0.125. On mesure
           donc « l'étage porte-t-il de quoi pointer la bonne valeur », pas de
           la mémorisation d'un répertoire fermé.

Étage (iii), choix documenté : le vecteur stocké est mis SEUL dans une banque
(M=1) et `DualModalBlock._cross_modal` est appliqué à un état de requête FIXE,
partagé par tous les slots — les états de la couche lue pour un tour de
question neutre (« What did I tell you earlier? »), capturés à l'entrée du
`_cross_modal` de la couche `--read-layer` (défaut : la dernière). La feature
est le delta moyen sur les positions de la requête (768). Isoler le slot est
délibéré : à M=8 le delta d'un slot dépend de la composition séquentielle avec
les 7 autres, et on mesurerait la dilution ambiante (kill-test 8) au lieu de
l'expressivité du read. C'est donc une borne SUPÉRIEURE optimiste sur (iii).

Choix du checkpoint : il faut un ckpt où le chemin write→banque→read
fast-weight a été ENTRAÎNÉ. Le run copy en cours contourne la banque
(cascade_depth 0) et ne convient pas. Défaut retenu :
`v350_sft_persona_sif_repass_rearm/step_550.pt` — le plus abouti des repass SIF
persona (550 steps contre 150 pour `v350_sft_persona_sif_repass` ;
`v350_phase2_stack` est vide), banque active dans sa cfg embarquée
(mem_dim=512, max_mem=8, mem_read_rank=8, mem_read_swiglu, mem_write_gate).

CPU uniquement (la 3090 est prise) : le script force `CUDA_VISIBLE_DEVICES=""`
et `map_location="cpu"`.

Premier passage (2026-08-03, ckpt ci-dessus, N=300, CPU) : slot et strate
traversent la chaîne quasi intacts (0.95/1.00 en mean768 → 0.80/0.98 après le
read) ; la VALEUR tombe de 0.675 à 0.383 AU POOLING (rétention relative 0.47),
reste plate à travers 768→512 (0.372) et ne perd qu'un peu au read (0.327).
Cadran désigné : la MATRICE PAR WRITE (k têtes de pooling) — pas mem_dim. Stable
en seed (0/1) et en couche de lecture (6/11).

Usage (racine du repo) :
    PYTHONPATH=. python deepseek_v4_mini/analysis/gist_local_probe.py \
        --ckpt /mnt/tb/checkpoints/v350_sft_persona_sif_repass_rearm/step_550.pt \
        [--n 300] [--read-layer -1] [--seed 0]
"""

import argparse
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""          # avant tout import torch

import dataclasses
from collections import defaultdict

import torch

from deepseek_v4_mini.config import ThoughtBankConfig
from deepseek_v4_mini.model import ThoughtBankLM
from deepseek_v4_mini.recall_env import RecallEnvStream, slot_id_map

STAGES = ["mean768", "pooled768", "stored512", "read768"]


# ── modèle ───────────────────────────────────────────────────────────────────

def load_model(ckpt: str):
    """Modèle CPU + cfg embarquée. Rend (model, cfg, step)."""
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    raw = sd["cfg"]
    raw = raw if isinstance(raw, dict) else vars(raw)
    keep = {f.name for f in dataclasses.fields(ThoughtBankConfig)}
    cfg = ThoughtBankConfig(**{k: v for k, v in raw.items() if k in keep})
    model = ThoughtBankLM(cfg)
    missing, unexpected = model.load_state_dict(sd["model"], strict=False)
    assert not [k for k in missing if not k.startswith("rti")], missing[:8]
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, cfg, int(sd.get("step", -1))


# ── données : les énonciations réelles de l'env, étiquetées ──────────────────

def collect(stream, n: int) -> list:
    """[{ids, slot, strate, value, text}] — segs d'énonciation du chemin réel.

    Les segs sont reconstruits par le MÊME `_user_fact` que `RecallEnvStream.segs`
    (mêmes ids, même val_mask) ; on repasse par les turns du script uniquement
    pour récupérer la VALEUR et la strate, que le seg ne porte pas."""
    inv = {v: k for k, v in slot_id_map().items()}
    out, life = [], 0
    while len(out) < n:
        sc = stream.script(life)
        life += 1
        by_fid = {f["fid"]: f for f in sc["facts"]}
        for t in sc["turns"]:
            if t["kind"] != "fact":
                continue
            f = by_fid[t["fid"]]
            seg = stream._user_fact(t["user"], f["value"], f["slot_id"],
                                    f.get("atoms"))
            out.append({"ids": seg["input_ids"][0],
                        "val_mask": seg["val_mask"][0].bool(),
                        "slot": inv[int(f["slot_id"])],
                        "strate": f["stratum"], "value": f["value"],
                        "text": t["user"]})
            if len(out) >= n:
                break
    return out


# ── capture des trois étages ─────────────────────────────────────────────────

def query_state(model, stream, layer: int, bank0: torch.Tensor,
                question: str) -> torch.Tensor:
    """État de requête FIXE : l'entrée du `_cross_modal` de la couche `layer`
    pour un tour de question neutre, sous la banque de référence."""
    seg = stream._user(question)
    blk = model.blocks[layer]
    grabbed = {}
    orig = blk._cross_modal

    def spy(h, bank):
        grabbed["h"] = h.detach().clone()
        return orig(h, bank)

    blk._cross_modal = spy
    try:
        with torch.no_grad():
            model(seg["input_ids"], init_mem=bank0, compute_logits=False)
    finally:
        blk._cross_modal = orig
    return grabbed["h"]                                    # [1, Tq, d_model]


def features(model, items: list, bank0: torch.Tensor, h_q: torch.Tensor,
             layer: int, batch: int = 8) -> dict:
    """Rend {étage: [N, D]} float32."""
    ts = model.thought_stream
    blk = model.blocks[layer]
    feats = {s: [] for s in STAGES}
    for i0 in range(0, len(items), batch):
        chunk = items[i0:i0 + batch]
        lens = [it["ids"].numel() for it in chunk]
        T = max(lens)
        ids = torch.zeros(len(chunk), T, dtype=torch.long)
        pad = torch.zeros(len(chunk), T)
        for b, it in enumerate(chunk):
            ids[b, :lens[b]] = it["ids"]
            pad[b, :lens[b]] = 1.0
        with torch.no_grad():
            # attention causale + right-padding ⇒ les positions réelles ne
            # voient jamais le pad ; pad_mask protège quand même le pool.
            H = model(ids, init_mem=bank0[:len(chunk)], compute_logits=False,
                      pad_mask=pad, write=False)["hidden"]     # [B, T, d]
            for b, it in enumerate(chunk):
                h = H[b:b + 1, :lens[b]]                       # [1, T_b, d]
                feats["mean768"].append(h.mean(dim=1)[0].float())
                w = torch.softmax(ts.write_ctx_q(h).squeeze(-1), dim=-1)
                feats["pooled768"].append((w.unsqueeze(-1) * h).sum(1)[0].float())
                m = ts._new_thought(h, bank0[:1])              # [1, 1, mem_dim]
                feats["stored512"].append(m[0, 0].float())
                delta = blk._cross_modal(h_q, m) - h_q         # [1, Tq, d]
                feats["read768"].append(delta.mean(dim=1)[0].float())
    return {k: torch.stack(v) for k, v in feats.items()}


# ── sondes linéaires (ridge closed-form, torch pur) ──────────────────────────

LAMBDAS = [1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3, 1e4]


def _fit(X: torch.Tensor, Y: torch.Tensor, lam: float) -> torch.Tensor:
    """W = (XᵀX + λI)⁻¹ XᵀY, en float64, biais inclus (colonne de 1)."""
    X = torch.cat([X, torch.ones(X.size(0), 1, dtype=X.dtype)], dim=1)
    G = X.T @ X
    G = G + lam * torch.eye(G.size(0), dtype=X.dtype)
    return torch.linalg.solve(G, X.T @ Y)


def _apply(X: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
    return torch.cat([X, torch.ones(X.size(0), 1, dtype=X.dtype)], dim=1) @ W


def _standardise(Xtr: torch.Tensor, Xte: torch.Tensor):
    mu = Xtr.mean(0, keepdim=True)
    sd = Xtr.std(0, keepdim=True).clamp_min(1e-6)
    return (Xtr - mu) / sd, (Xte - mu) / sd


def ridge_clf(Xtr, ytr, Xte, yte, n_cls: int) -> float:
    """Accuracy test d'une ridge one-hot ; λ choisi sur un split interne 80/20."""
    Xtr, Xte = _standardise(Xtr.double(), Xte.double())
    Y = torch.zeros(len(ytr), n_cls, dtype=torch.float64)
    Y[torch.arange(len(ytr)), ytr] = 1.0
    cut = max(1, int(0.8 * len(ytr)))
    perm = torch.randperm(len(ytr))
    a, b = perm[:cut], perm[cut:]
    best, best_lam = -1.0, LAMBDAS[0]
    for lam in LAMBDAS:
        W = _fit(Xtr[a], Y[a], lam)
        acc = (_apply(Xtr[b], W).argmax(1) == ytr[b]).double().mean().item()
        if acc > best:
            best, best_lam = acc, lam
    W = _fit(Xtr, Y, best_lam)
    return (_apply(Xte, W).argmax(1) == yte).double().mean().item()


def ridge_retrieval(Xtr, Ztr, Xte, Zte, n_cand: int, draws: int,
                    gen: torch.Generator) -> float:
    """Ridge vers l'embedding de la valeur, évaluée en RÉCUPÉRATION top-1 parmi
    `n_cand` candidats (valeurs de test, jamais vues au train)."""
    Xtr, Xte = _standardise(Xtr.double(), Xte.double())
    Ztr, Zte = Ztr.double(), Zte.double()
    cut = max(1, int(0.8 * len(Xtr)))
    perm = torch.randperm(len(Xtr), generator=gen)
    a, b = perm[:cut], perm[cut:]
    best, best_lam = -1.0, LAMBDAS[0]
    for lam in LAMBDAS:
        W = _fit(Xtr[a], Ztr[a], lam)
        P = torch.nn.functional.normalize(_apply(Xtr[b], W), dim=-1)
        sim = P @ torch.nn.functional.normalize(Ztr[b], dim=-1).T
        acc = (sim.argmax(1) == torch.arange(len(b))).double().mean().item()
        if acc > best:
            best, best_lam = acc, lam
    W = _fit(Xtr, Ztr, best_lam)
    P = torch.nn.functional.normalize(_apply(Xte, W), dim=-1)
    Z = torch.nn.functional.normalize(Zte, dim=-1)
    n, hits, tot = len(Xte), 0, 0
    for _ in range(draws):
        for i in range(n):
            pool = torch.randperm(n, generator=gen)[:n_cand + 1]
            pool = pool[pool != i][:n_cand - 1]
            cand = torch.cat([torch.tensor([i]), pool])
            hits += int((P[i] @ Z[cand].T).argmax().item() == 0)
            tot += 1
    return hits / max(tot, 1)


# ── splits ───────────────────────────────────────────────────────────────────

def group_split(keys: list, frac: float, gen: torch.Generator):
    """Split par GROUPE (aucun groupe des deux côtés). Rend (idx_tr, idx_te)."""
    groups = sorted(set(keys))
    perm = torch.randperm(len(groups), generator=gen).tolist()
    cut = max(1, int(frac * len(groups)))
    test = {groups[i] for i in perm[:cut]}
    tr = [i for i, k in enumerate(keys) if k not in test]
    te = [i for i, k in enumerate(keys) if k in test]
    return torch.tensor(tr), torch.tensor(te)


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/mnt/tb/checkpoints/"
                    "v350_sft_persona_sif_repass_rearm/step_550.pt")
    ap.add_argument("--n", type=int, default=300, help="segs d'énonciation")
    ap.add_argument("--read-layer", type=int, default=-1,
                    help="couche dont le _cross_modal sert à l'étage (iii)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--n-cand", type=int, default=8, help="candidats valeur")
    ap.add_argument("--draws", type=int, default=20)
    ap.add_argument("--question", default="What did I tell you earlier?")
    ap.add_argument("--tokenizer", default="HuggingFaceTB/SmolLM2-135M")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    gen = torch.Generator().manual_seed(args.seed)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    stream = RecallEnvStream(tok, seed=args.seed)

    model, cfg, step = load_model(args.ckpt)
    layer = args.read_layer % cfg.n_layers
    items = collect(stream, args.n)

    # Banque de référence : UN tirage seed_bank, partagé par tous les segments —
    # les étages doivent différer par le TOUR, pas par le bruit de la banque.
    with torch.no_grad():
        bank0 = model.thought_stream.seed_bank(args.batch, torch.device("cpu"),
                                               torch.float32)
    h_q = query_state(model, stream, layer, bank0[:1], args.question)

    n_strat = defaultdict(int)
    for it in items:
        n_strat[it["strate"]] += 1
    print(f"ckpt {args.ckpt} (step {step})")
    print(f"  d_model={cfg.d_model} mem_dim={cfg.mem_dim} max_mem={cfg.max_mem} "
          f"rang r={cfg.mem_read_rank} swiglu={cfg.mem_read_swiglu} "
          f"write_gate={cfg.mem_write_gate} n_layers={cfg.n_layers}")
    print(f"  {len(items)} énonciations {dict(sorted(n_strat.items()))}, "
          f"étage (iii) : couche {layer}, requête « {args.question} » "
          f"(Tq={h_q.size(1)}), banque M=1")

    F = features(model, items, bank0, h_q, layer, args.batch)

    # étiquettes
    slots = sorted({it["slot"] for it in items})
    strates = sorted({it["strate"] for it in items})
    y_slot = torch.tensor([slots.index(it["slot"]) for it in items])
    y_str = torch.tensor([strates.index(it["strate"]) for it in items])

    # cible valeur : embedding moyen des tokens de la valeur (span val_mask ;
    # repli sur la tokenisation nue si le span est introuvable)
    emb = model.embed.weight.detach()
    Z, keep = [], []
    for i, it in enumerate(items):
        vm = it["val_mask"]
        vids = (it["ids"][vm] if vm.any() else
                torch.tensor(tok(" " + it["value"],
                                 add_special_tokens=False)["input_ids"]))
        if vids.numel() == 0:
            continue
        Z.append(emb[vids].mean(0).float())
        keep.append(i)
    Z = torch.stack(Z)
    keep = torch.tensor(keep)

    surf = [it["text"] for it in items]
    tr_s, te_s = group_split(surf, 0.3, gen)          # split par SURFACE
    vals = [items[i]["value"] for i in keep.tolist()]
    tr_v, te_v = group_split(vals, 0.3, gen)          # split par VALEUR

    maj_slot = max((y_slot[te_s] == c).double().mean().item()
                   for c in range(len(slots)))
    maj_str = max((y_str[te_s] == c).double().mean().item()
                  for c in range(len(strates)))

    print(f"\n  split surface : {len(tr_s)} train / {len(te_s)} test "
          f"(énoncés disjoints) ; split valeur : {len(tr_v)} / {len(te_v)} "
          f"(valeurs disjointes)")
    print("\n── sondes linéaires (ridge) : accuracy test par attribut × étage ──")
    hdr = (f"{'attribut':<22}{'chance':>8}" +
           "".join(f"{s:>12}" for s in STAGES))
    print(hdr)
    print("-" * len(hdr))

    rows = []
    accs = {}
    for name, y, n_cls, chance in [
            (f"slot ({len(slots)} cl.)", y_slot, len(slots), maj_slot),
            (f"strate ({len(strates)} cl.)", y_str, len(strates), maj_str)]:
        r = [ridge_clf(F[s][tr_s], y[tr_s], F[s][te_s], y[te_s], n_cls)
             for s in STAGES]
        accs[name] = r
        rows.append((name, chance, r))
    name = f"valeur@{args.n_cand} (OOD)"
    r = [ridge_retrieval(F[s][keep][tr_v], Z[tr_v], F[s][keep][te_v], Z[te_v],
                         args.n_cand, args.draws, gen) for s in STAGES]
    accs[name] = r
    rows.append((name, 1.0 / args.n_cand, r))

    for name, chance, r in rows:
        print(f"{name:<22}{chance:>8.3f}" + "".join(f"{v:>12.3f}" for v in r))

    # lecture du verdict — rapports d'étage, normalisés à la chance
    print("\n── rétention relative (au-dessus de la chance, réf. mean768) ──")
    print(f"{'attribut':<22}" + "".join(f"{s:>12}" for s in STAGES))
    for name, chance, r in rows:
        base = max(r[0] - chance, 1e-9)
        print(f"{name:<22}" + "".join(f"{(v - chance) / base:>12.2f}"
                                      for v in r))

    print("\nLecture (spec §2.4) : (i)=pooled768 vs contrôle mean768 → le pooling ; "
          "(i)→(ii) → mem_dim ; (ii)→(iii) → le rang r du read.")


if __name__ == "__main__":
    main()
