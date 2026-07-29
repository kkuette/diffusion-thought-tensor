"""ChatBatchStream — B conversations chat en lockstep par index de tour.

POURQUOI. Le step SFT du 350M est à ~250× du régime compute-bound : 386M params
sur ~150 tokens font ~0,5 TFLOP (≈10 ms de 3090) et coûtent 2,7 s. Ce qui le
limite est le NOMBRE D'OPS ÉMISES, pas les FLOPs — et le compte d'ops aten d'un
forward ne dépend pas du batch (mesuré : 11 462 à B=1 contre 11 486 à B=4,
+0,2 %). Faire tourner G convs en séquence (`grad_accum`) paie donc G fois le
dispatch ; les faire tourner en B lanes le paie une fois.

Le stream fichiers a déjà son chemin batché (`code_data.next_conv_batch`) mais il
évite le padding par construction : il ne prend que des chunks PLEINS, tous de
longueur L. Les convs chat, elles, sont ragged deux fois — longueur de tour ET
nombre de tours. Ce module reprend donc le patron de
`train._build_multiturn_loader` (arc dsv4mini) :

    chaque lane tire ses convs indépendamment et se RECHARGE quand elle est vide,
    donc aucune ligne n'est jamais entièrement du padding ; chaque `next_conv_batch`
    rend `slots` tours successifs, chacun collationné [B, L] avec right-padding.

LOCKSTEP DES ÉVICTIONS. `CascadeMemory` porte le batch en dim 0 mais son
branchement de débordement (`len(p0) == M`) est UNIQUE pour tout le batch : une
éviction tombe pour toutes les lanes ou aucune. C'est satisfait ici par
construction — chaque lane écrit exactement un gist par slot de tour, donc les
compteurs d'écriture restent égaux entre lanes. C'est ce qui rend le batching
chat compatible avec `cascade_depth > 0`, là où le chemin fichiers exige
`pack_convs` (profondeur constante K).

CONTREPARTIE ASSUMÉE. Avec `no_reset_files: 0` la « vie unique » devient B vies
parallèles, une par lane (décision user 2026-07-25).

CE QUE LE PADDING COÛTE, MESURÉ. Le forward n'est PAS invariant au right-padding,
mais pas pour la raison qu'on croit : la causalité est stricte (masque de bloc
`j < t//m`, le bloc de la requête est exclu) et vérifiée exhaustivement. L'écart
vient du top-k DUR de l'indexeur Lightning : son `F.relu` produit des scores
EXACTEMENT égaux (souvent 0), et quand le nombre de blocs valides dépasse
`top_k_csa`, `topk` tranche ces ex æquo par disposition mémoire — qui dépend de
la longueur paddée. Mesuré en float64 : scores identiques à 1e-17, marge entre le
8ᵉ et le 9ᵉ score EXACTEMENT nulle, et la sélection change (blocs {0,1,2,3,4,6,7,8}
contre {0,1,2,4,5,6,7,8}) ⇒ ~3e-2 sur l'état caché de cette position. C'est la
même pathologie que le top-k du routage MoE sous recompute (voir runtime.py) :
le batching padé n'est donc pas bit-identique à B=1, exactement comme
`decode_cache` ou `compile` ne le sont pas. La position perturbée est celle dont
la cible est un pad, donc `loss_mask` l'exclut déjà de la CE.

PAVAGE DE LA FENÊTRE (`tile: true`, proposition user 2026-07-25). La banque est
DÉTACHÉE à chaque pull, donc `slots` EST la fenêtre TBPTT : un write ne reçoit de
gradient que des réponses du MÊME appel. Avec le refill naïf ci-dessus, une lane
se recharge n'importe où dans la fenêtre, donc ~1 session sur 3 est coupée par la
frontière de pull et ses tours tardifs perdent le gradient vers ses écritures
précoces. Un couple d'âge `a` ne survit alors qu'avec probabilité `max(0, 1-a/s)`
(offset uniforme) — 96,9 % à l'âge 1 mais **31,2 % à l'âge 22**, c'est-à-dire que
le régime le plus cher à apprendre est celui qui reçoit le moins de gradient.

`tile: true` remplit chaque lane avec des sessions ENTIÈRES dont les tailles
sommeent EXACTEMENT à `slots` (8+8+16, 4+4+24…) : plus aucune session à cheval,
donc **100 % des couples intra-session** sont couverts, tous âges confondus.
Mesuré sur 1500 sessions du mix prod : tailles médiane 10 / p90 16 / max 30, et
un pavage exact à 32 réussit **100 % du temps avec un pool de 16 sessions**
(91,8 % à 8). Le pool est le seul état entre deux fenêtres (les files de lane
sont vidées exactement), d'où `state_dict()` — le round-trip rng du ckpt ne
suffit pas à le reconstruire.
C'est l'analogue chat de `pack_convs` : profondeur CONSTANTE par pull.
`convs_per_session` chaînait déjà les convs, mais sans les aligner sur la
frontière de pull — c'était la moitié manquante.

SPLIT DES TOURS LONGS (`max_seg_tok`). Un tour peut dépasser la longueur de
travail (mesuré : médiane 88 tokens mais jusqu'à 996), et c'est le pire slot qui
fixe le pic VRAM. `max_seg_tok: N` coupe tout seg de plus de N tokens en segs
consécutifs — un tour long devient plusieurs tours, donc plusieurs écritures,
exactement comme un doc devient plusieurs chunks côté fichiers. On NE jette PAS
les convs qui en contiennent : les tours longs SONT les cas exec/tool (bloc de
code, schéma d'outil), les jeter biaiserait la distribution contre le régime
qu'on cherche à apprendre. Contrepartie exacte : la CE se décale à l'intérieur
d'un seg (`m2 = lmask[:, 1:]`), donc chaque frontière de split coûte UN token
supervisé — 1 sur 996, et c'est le même prix que les frontières de chunk.

Le stream s'utilise comme n'importe quel autre (registre `streams.CHAT_STREAMS`) :

  chat:
    stream: chat_batch
    gen:
      batch: 4                 # lanes en parallèle
      slots: 8                 # tours par appel = profondeur du graphe
      tile: true               # pavage exact de la fenêtre (défaut false)
      pool: 16                 # sessions gardées d'avance (croît si besoin)
      pool_max: 0              # plafond de croissance (0 = 16 x pool)
      max_seg_tok: 512         # coupe les tours plus longs (défaut 0 = jamais)
      stream: chat_mix         # le stream enveloppé, n'importe lequel
      gen: {streams: [...]}    # ses kwargs

`next_conv` / `rng` / `grade_conv` DÉLÈGUENT au stream enveloppé : l'éval
(`evaluate_math`, B=1, conv par conv) et le round-trip rng du checkpoint
continuent de marcher sans le savoir.

Self-test CPU hermétique :  python -m deepseek_v4_mini.chat_batch
"""
from __future__ import annotations

import torch

# Clés par-token à padder. `input_ids` prend le pad_id, tout le reste prend 0 :
# loss_mask/surp_w à 0 = aucune contribution à la CE ni au pooling du teacher,
# val_mask à 0 = le seg n'est pas porteur sur ces positions.
_ZERO_PAD_KEYS = ("loss_mask", "attention_mask", "val_mask", "surp_w",
                  # identité du fait (teacher value_table) : [1,T] long,
                  # 1-based — le pad 0 vaut « pas de fait » par construction
                  "fact_slot", "fact_val", "fact_attr")


class ChatBatchStream:
    def __init__(self, tok, *, seed: int = 0, batch: int = 2, slots: int = 8,
                 stream: str = "chat_mix", gen: dict | None = None,
                 tile: bool = False, pool: int = 16, pool_max: int = 0,
                 max_seg_tok: int = 0, _inner=None) -> None:
        assert int(batch) >= 1, f"batch doit être >= 1, vu {batch}"
        assert int(slots) >= 1, f"slots doit être >= 1, vu {slots}"
        assert int(max_seg_tok) >= 0, f"max_seg_tok >= 0, vu {max_seg_tok}"
        # Un pool d'une seule session ne laisse aucun choix au pavage : il
        # dégénérerait en refill naïf sans le dire.
        assert not tile or int(pool) >= 2, f"tile exige pool >= 2, vu {pool}"
        self.B = int(batch)
        self.slots = int(slots)
        self.tile = bool(tile)
        self.pool_n = int(pool)                    # pas de croissance du pool
        self.pool_max = int(pool_max or 16 * self.pool_n)
        self.max_seg_tok = int(max_seg_tok)
        if _inner is not None:                     # injection de test
            self.inner = _inner
        else:
            from .streams import build_chat_stream
            self.inner = build_chat_stream(stream, tok, seed=seed, gen=gen or {})
        # Le contrat de checkpoint et l'éval passent par le stream enveloppé.
        self.rng = self.inner.rng
        self.pad_id = int(getattr(tok, "pad_token_id", None) or 0) if tok else 0
        self._q: list[list] = [[] for _ in range(self.B)]
        # Pavage : sessions tirées d'avance (partagées), et reste d'une session
        # à cheval par lane (fallback quand aucun pavage exact n'existe).
        self._pool: list[list] = []
        self._pool_now = self.pool_n               # cible courante (croissante)
        self._carry: list[list] = [[] for _ in range(self.B)]
        self.n_straddle = 0                        # diagnostic : pavages ratés

    # ── délégation : l'éval et le ckpt ne voient pas l'enveloppe ─────────────

    def next_conv(self) -> dict:
        return self.inner.next_conv()

    def grade_conv(self, conv: dict, texts) -> float:
        return self.inner.grade_conv(conv, texts)

    # ── état sérialisable (le pool NE se reconstruit pas depuis le rng) ──────

    def state_dict(self) -> dict:
        """Les sessions déjà tirées mais pas encore consommées.

        Le ckpt sauve l'état rng du stream enveloppé, ce qui suffit à rejouer les
        tirages À VENIR — mais pas à retrouver ce qui a déjà été tiré et attend
        dans le pool, le carry ou une file de lane. Sans ça le resume repart au
        milieu d'une session différente. Les tenseurs sont minuscules (~16
        sessions x ~10 segs x ~100 tokens).
        """
        # COPIES : renvoyer les listes vives ferait muter l'instantané par les
        # appels suivants (les segs eux-mêmes ne sont jamais modifiés en place).
        return {"pool": [list(s) for s in self._pool],
                "carry": [list(c) for c in self._carry],
                "q": [list(q) for q in self._q],
                "pool_now": self._pool_now,
                "n_straddle": self.n_straddle}

    def load_state_dict(self, st: dict) -> None:
        self._pool = list(st.get("pool") or [])
        carry = st.get("carry") or []
        self._carry = [list(carry[b]) if b < len(carry) else []
                       for b in range(self.B)]
        q = st.get("q") or []
        self._q = [list(q[b]) if b < len(q) else [] for b in range(self.B)]
        # La taille atteinte par le pool fait partie de l'état : repartir de
        # pool_n rejouerait toute la phase de croissance (et ses sessions
        # coupées) à chaque reprise.
        self._pool_now = max(self.pool_n,
                             min(self.pool_max, int(st.get("pool_now",
                                                           self.pool_n))))
        self.n_straddle = int(st.get("n_straddle", 0))

    # ── découpage des tours plus longs que max_seg_tok ───────────────────────

    def _split_seg(self, s: dict) -> list:
        """Un seg de L > max_seg_tok devient ceil(L/max_seg_tok) segs consécutifs.

        Toute valeur tenseur de forme [1, L] est tranchée (détection par FORME,
        pas par liste de clés : une clé par-token oubliée serait une corruption
        silencieuse) ; le reste — `role`, scalaires — est recopié tel quel.
        """
        C = self.max_seg_tok
        if C <= 0:
            return [s]
        L = int(s["input_ids"].size(1))
        if L <= C:
            return [s]
        per_tok = [k for k, v in s.items()
                   if torch.is_tensor(v) and v.dim() == 2 and v.size(1) == L]
        assert "input_ids" in per_tok, "input_ids doit être [1, L]"
        out = []
        for a in range(0, L, C):
            piece = {k: v for k, v in s.items() if k not in per_tok}
            for k in per_tok:
                piece[k] = s[k][:, a:min(a + C, L)].contiguous()
            out.append(piece)
        return out

    def _draw_session(self) -> list:
        """Une session du stream enveloppé, tours longs déjà découpés."""
        segs: list = []
        for s in self.inner.next_conv()["segs"]:
            segs.extend(self._split_seg(s))
        return segs

    # ── pavage exact de la fenêtre ──────────────────────────────────────────

    @staticmethod
    def _subset_to(sizes: list, target: int) -> list | None:
        """Indices d'un sous-ensemble de `sizes` sommant EXACTEMENT à `target`.

        Programmation dynamique sur les sommes atteignables : `target` est petit
        (= `slots`) et `sizes` court (= `pool`), donc c'est immédiat. Renvoie None
        si aucun sous-ensemble ne tombe juste.
        """
        reach: dict = {0: ()}
        for i, v in enumerate(sizes):
            if v > target:
                continue
            for tot in sorted(reach, reverse=True):
                nt = tot + v
                if nt <= target and nt not in reach:
                    reach[nt] = reach[tot] + (i,)
            if target in reach:
                return list(reach[target])
        return None

    def _fill_lane_tiled(self, b: int, need: int) -> list:
        """Exactement `need` segs pour la lane `b`, sans couper de session."""
        q: list = []
        # Reste d'une session laissée à cheval par un pavage impossible au tour
        # précédent : on le solde d'abord, sinon elle resterait coupée pour
        # toujours.
        if self._carry[b]:
            take = self._carry[b][:need]
            self._carry[b] = self._carry[b][len(take):]
            q.extend(take)
        while len(q) < need:
            room = need - len(q)
            # POOL AUTO-CROISSANT. Le subset-sum consomme préférentiellement les
            # petites sessions, donc un pool de taille fixe DÉRIVE vers les
            # grandes et finit par ne plus tomber juste : mesuré sur le mix prod,
            # 30 % d'échecs à pool 16, 21 % à 32, 11 % à 64, 0 % à 128. Plutôt
            # que de coder en dur un 128 qui ne survivrait pas à un autre mix, on
            # agrandit le pool tant qu'aucun pavage exact n'existe.
            idx = None
            while True:
                while len(self._pool) < self._pool_now:
                    self._pool.append(self._draw_session())
                idx = self._subset_to([len(s) for s in self._pool], room)
                if idx is not None or self._pool_now >= self.pool_max:
                    break
                self._pool_now = min(self.pool_max,
                                     self._pool_now + self.pool_n)
            if idx is None:
                # Aucun pavage exact possible avec ce pool (ex. pool de tailles
                # {6, 9} et room 32). On prend la plus grande session qui entre,
                # sinon la plus petite du pool en la laissant à cheval.
                fits = [i for i, s in enumerate(self._pool) if len(s) <= room]
                if fits:
                    idx = [max(fits, key=lambda i: len(self._pool[i]))]
                else:
                    i = min(range(len(self._pool)),
                            key=lambda i: len(self._pool[i]))
                    sess = self._pool.pop(i)
                    q.extend(sess[:room])
                    self._carry[b] = sess[room:]
                    self.n_straddle += 1
                    break
            for i in sorted(idx, reverse=True):
                q.extend(self._pool.pop(i))
        assert len(q) == need, (len(q), need)
        return q

    # ── chemin batché ───────────────────────────────────────────────────────

    def _refill(self, b: int) -> None:
        """Une lane n'est JAMAIS vide au moment du pop : sinon on produirait une
        ligne toute-pad, ce qui casserait le lockstep des écritures (la lane
        n'écrirait pas son gist du tour) en plus de gâcher du calcul."""
        while not self._q[b]:
            self._q[b].extend(self._draw_session())

    def next_conv_batch(self, slots: int | None = None) -> list:
        """`slots` tours consécutifs, chacun [B, L] right-padé.

        Renvoie la même forme que `next_conv()["segs"]` : une liste de segs que
        le trainer déroule dans l'ordre, un write par seg et par lane.
        """
        n = self.slots if slots is None else int(slots)
        if self.tile:
            # La fenêtre précédente a été consommée exactement, donc on la
            # remplit d'un coup : c'est ce qui garantit qu'aucune session ne
            # chevauche la frontière de pull.
            for b in range(self.B):
                room = max(0, n - len(self._q[b]))
                if room:
                    self._q[b].extend(self._fill_lane_tiled(b, room))
        out = []
        for _ in range(n):
            rows = []
            for b in range(self.B):
                if not self.tile:
                    self._refill(b)
                rows.append(self._q[b].pop(0))
            out.append(self._collate(rows))
        return out

    def _collate(self, rows: list) -> dict:
        L = max(int(r["input_ids"].size(1)) for r in rows)
        B = len(rows)
        seg: dict = {}

        ids = torch.full((B, L), self.pad_id, dtype=torch.long)
        pad_mask = torch.zeros(B, L, dtype=torch.bool)
        for b, r in enumerate(rows):
            t = int(r["input_ids"].size(1))
            ids[b, :t] = r["input_ids"][0]
            pad_mask[b, :t] = True
        seg["input_ids"] = ids
        # pad_mask : True = position RÉELLE (convention de model.forward, qui la
        # passe telle quelle au write ; memory._new_thought masque le pooling).
        seg["pad_mask"] = pad_mask

        for key in _ZERO_PAD_KEYS:
            if not any(key in r for r in rows):
                continue
            ref = next(r[key] for r in rows if key in r)
            buf = torch.zeros(B, L, dtype=ref.dtype)
            for b, r in enumerate(rows):
                if key in r:
                    t = int(r[key].size(1))
                    buf[b, :t] = r[key][0]
            seg[key] = buf

        # `role` est un scalaire par seg et diffère d'une lane à l'autre : on le
        # garde au pluriel pour que rien ne le compare à une chaîne par accident.
        seg["roles"] = tuple(r.get("role") for r in rows)
        return seg


# ── self-test (hermétique : stub de stream, aucun tokenizer) ─────────────────

def _selftest() -> None:
    """Ce que le batching chat peut casser en silence : une ligne toute-pad (la
    lane n'écrit pas son gist ⇒ lockstep des évictions rompu ⇒ cascade
    corrompue), du padding qui fuit dans la CE ou dans le pooling du teacher, et
    du contenu réel décalé par la collation."""
    import random

    class Stub:
        """Convs ragged : nombre de tours ET longueur de tour variables."""

        def __init__(self, seed):
            self.rng = random.Random(seed)
            self.made = 0

        def next_conv(self):
            n_segs = self.rng.randint(1, 4)
            segs = []
            for _ in range(n_segs):
                t = self.rng.randint(3, 11)
                self.made += 1
                segs.append({
                    "input_ids": torch.arange(
                        self.made * 1000, self.made * 1000 + t).unsqueeze(0),
                    "loss_mask": torch.ones(1, t) * 0.5,
                    "attention_mask": torch.ones(1, t, dtype=torch.long),
                    "surp_w": torch.full((1, t), 0.25),
                    "role": "assistant",
                })
            return {"segs": segs, "info": {}, "kind": "stub"}

        def grade_conv(self, conv, texts):
            return 0.5

    B, S = 4, 6
    st = ChatBatchStream(None, batch=B, slots=S, _inner=Stub(0))
    assert st.B == B and st.rng is st.inner.rng, "rng doit déléguer (ckpt)"
    segs = st.next_conv_batch()
    assert len(segs) == S, f"{S} slots attendus, vu {len(segs)}"

    n_pad_tot = n_tok_tot = 0
    for s in segs:
        ids, pm = s["input_ids"], s["pad_mask"]
        assert ids.shape == pm.shape and ids.size(0) == B, ids.shape
        L = ids.size(1)
        # AUCUNE ligne toute-pad : c'est l'invariant qui tient le lockstep
        assert bool(pm.any(dim=1).all()), "ligne toute-pad = lane sans write"
        # au moins une lane atteint L (sinon on pad tout le monde pour rien)
        assert int(pm.sum(dim=1).max()) == L, "L doit être le max des longueurs"
        # les masques sont nuls EXACTEMENT sur les pads
        for key in ("loss_mask", "attention_mask", "surp_w"):
            v = s[key]
            assert v.shape == ids.shape, (key, v.shape)
            assert float(v[~pm].abs().sum()) == 0.0, f"{key} non nul sur un pad"
            assert bool((v[pm] != 0).all()), f"{key} nul sur une position réelle"
        assert s["input_ids"][~pm].eq(st.pad_id).all(), "pad_id attendu"
        assert len(s["roles"]) == B
        n_pad_tot += int((~pm).sum()); n_tok_tot += pm.numel()

    # le contenu réel est intact et dans l'ordre : les ids du stub sont des
    # plages contiguës, donc un décalage de collation se verrait tout de suite
    for s in segs:
        for b in range(B):
            t = int(s["pad_mask"][b].sum())
            row = s["input_ids"][b, :t]
            assert torch.equal(row, torch.arange(int(row[0]), int(row[0]) + t)), \
                "contenu réel décalé par la collation"

    # lockstep : chaque lane a consommé EXACTEMENT un seg par slot
    st2 = ChatBatchStream(None, batch=3, slots=5, _inner=Stub(1))
    before = st2.inner.made
    got = st2.next_conv_batch()
    produced = sum(int(g["pad_mask"][b].sum() > 0) for g in got for b in range(3))
    assert produced == 3 * 5, produced
    assert st2.inner.made > before

    # une lane courte se recharge sans jamais produire de ligne vide
    st3 = ChatBatchStream(None, batch=2, slots=40, _inner=Stub(2))
    for s in st3.next_conv_batch():
        assert bool(s["pad_mask"].any(dim=1).all())

    # délégation de next_conv : l'éval reste à B=1
    solo = st.next_conv()
    assert solo["segs"][0]["input_ids"].size(0) == 1, "next_conv doit rester B=1"
    assert st.grade_conv(solo, []) == 0.5

    # slots=1 et batch=1 sont des cas limites légitimes
    st4 = ChatBatchStream(None, batch=1, slots=1, _inner=Stub(3))
    one = st4.next_conv_batch()
    assert len(one) == 1 and one[0]["input_ids"].size(0) == 1
    for bad in (dict(batch=0), dict(slots=0)):
        try:
            ChatBatchStream(None, _inner=Stub(4), **bad)
        except AssertionError:
            pass
        else:
            raise AssertionError(f"{bad} aurait dû lever")

    # ── max_seg_tok : un tour long devient plusieurs tours, sans rien perdre ──

    class Long:
        """Une seule conv, un seul seg très long : le découpage doit être exact."""

        def __init__(self, L):
            self.rng = random.Random(0)
            self.L = L

        def next_conv(self):
            L = self.L
            return {"segs": [{
                "input_ids": torch.arange(L).unsqueeze(0),
                "loss_mask": torch.arange(1, L + 1).float().unsqueeze(0),
                "surp_w": torch.full((1, L), 0.5),
                "role": "assistant",
            }], "info": {}, "kind": "long"}

        def grade_conv(self, conv, texts):
            return 0.0

    C, LL = 7, 30                                   # 30 = 4x7 + 2 -> 5 morceaux
    stL = ChatBatchStream(None, batch=1, slots=5, max_seg_tok=C, _inner=Long(LL))
    pieces = stL._split_seg(stL.inner.next_conv()["segs"][0])
    assert len(pieces) == 5, len(pieces)
    assert [int(p["input_ids"].size(1)) for p in pieces] == [7, 7, 7, 7, 2]
    # contenu et masques recollés = l'original, à l'identique
    assert torch.equal(torch.cat([p["input_ids"] for p in pieces], 1),
                       torch.arange(LL).unsqueeze(0)), "split : ids perdus"
    assert torch.equal(torch.cat([p["loss_mask"] for p in pieces], 1),
                       torch.arange(1, LL + 1).float().unsqueeze(0)), \
        "split : loss_mask désaligné"
    assert all(p["role"] == "assistant" for p in pieces), "split : role perdu"
    # aucune clé par-token laissée entière (ce serait une corruption muette)
    for p in pieces:
        n = int(p["input_ids"].size(1))
        for k, v in p.items():
            if torch.is_tensor(v) and v.dim() == 2:
                assert v.size(1) == n, f"{k} non tranché ({v.size(1)} vs {n})"
    # max_seg_tok=0 (défaut) ne touche à rien
    st0 = ChatBatchStream(None, batch=1, slots=2, _inner=Long(LL))
    assert len(st0._split_seg(st0.inner.next_conv()["segs"][0])) == 1

    # ── tile : la fenêtre est pavée par des sessions ENTIÈRES ────────────────

    class Sized:
        """Sessions de tailles connues. L'identité (session, position dans la
        session, taille) est ENCODÉE dans les input_ids — `sid*10000 + pos*100 +
        n` — donc lisible après collation sans rien ajouter à `_collate`."""

        def __init__(self, sizes):
            self.rng = random.Random(0)
            self.sizes = list(sizes)
            self.i = 0
            self.sid = 0

        def next_conv(self):
            n = self.sizes[self.i % len(self.sizes)]
            self.i += 1
            self.sid += 1
            segs = [{
                "input_ids": torch.full(
                    (1, 3), self.sid * 10000 + j * 100 + n, dtype=torch.long),
                "loss_mask": torch.ones(1, 3),
                "role": "assistant",
            } for j in range(n)]
            return {"segs": segs, "info": {}, "kind": "sized"}

        def grade_conv(self, conv, texts):
            return 0.0

    def _decode(v):
        v = int(v)
        return v // 10000, (v % 10000) // 100, v % 100      # sid, pos, n

    W = 12
    NB = 3
    stT = ChatBatchStream(None, batch=NB, slots=W, tile=True, pool=8,
                          _inner=Sized([4, 5, 6, 8, 3, 7, 12, 2]))
    seen: dict = {}                            # (lane, sid) -> [(slot, pos, n)]
    for w in range(6):                         # plusieurs fenêtres d'affilée
        segs = stT.next_conv_batch()
        assert len(segs) == W
        # chaque lane a consommé W segs : la file est vidée EXACTEMENT, ce qui
        # est la condition même du « aucune session à cheval »
        assert all(not q for q in stT._q), "file de lane non vidée"
        for slot, s in enumerate(segs):
            assert bool(s["pad_mask"].all()), "tous les segs font 3 tokens ici"
            for b in range(NB):
                sid, pos, n = _decode(s["input_ids"][b, 0])
                seen.setdefault((b, sid), []).append((w * W + slot, pos, n))
    for (b, sid), got in seen.items():
        n = got[0][2]
        # une session apparaît ENTIÈRE et contiguë, jamais coupée par une
        # frontière de fenêtre
        assert [p for _, p, _ in got] == list(range(n)), \
            f"session {sid} lane {b} incomplète : {[p for _, p, _ in got]}/{n}"
        sl = [x for x, _, _ in got]
        assert sl == list(range(sl[0], sl[0] + n)), \
            f"session {sid} lane {b} non contiguë : {sl}"
        assert sl[0] // W == sl[-1] // W, \
            f"session {sid} lane {b} à cheval sur deux fenêtres : {sl}"
    assert stT.n_straddle == 0, f"{stT.n_straddle} pavage(s) raté(s) sur ce pool"

    # pavage IMPOSSIBLE : pool de tailles {5} et fenêtre 12 -> 5+5 puis à cheval
    stX = ChatBatchStream(None, batch=1, slots=12, tile=True, pool=4,
                          _inner=Sized([5]))
    for _ in range(3):
        got = stX.next_conv_batch()
        assert len(got) == 12, len(got)          # la fenêtre est TOUJOURS pleine
    assert stX.n_straddle > 0, "le fallback à cheval devrait avoir servi"

    # tile exige un pool >= 2
    try:
        ChatBatchStream(None, batch=1, slots=4, tile=True, pool=1,
                        _inner=Sized([2]))
    except AssertionError:
        pass
    else:
        raise AssertionError("tile avec pool=1 aurait dû lever")

    # ── round-trip d'état : le resume ne doit pas repartir ailleurs ──────────

    a = ChatBatchStream(None, batch=2, slots=W, tile=True, pool=8,
                        _inner=Sized([4, 5, 6, 8, 3, 7, 12, 2]))
    a.next_conv_batch()                          # du pool est déjà tiré
    snap = a.state_dict()
    rng_snap = a.inner.rng.getstate()
    ref = a.next_conv_batch()
    b2 = ChatBatchStream(None, batch=2, slots=W, tile=True, pool=8,
                         _inner=Sized([4, 5, 6, 8, 3, 7, 12, 2]))
    b2.inner.rng.setstate(rng_snap)
    b2.inner.i, b2.inner.sid = a.inner.i, a.inner.sid   # l'état du stream enveloppé
    b2.load_state_dict(snap)
    got = b2.next_conv_batch()
    assert len(got) == len(ref)
    for s1, s2 in zip(ref, got):
        assert torch.equal(s1["input_ids"], s2["input_ids"]), \
            "resume : la fenêtre reprise diffère"
    # et SANS l'état : le pool déjà tiré est perdu, donc la fenêtre change —
    # c'est précisément ce que state_dict existe pour éviter.
    b3 = ChatBatchStream(None, batch=2, slots=W, tile=True, pool=8,
                         _inner=Sized([4, 5, 6, 8, 3, 7, 12, 2]))
    b3.inner.rng.setstate(rng_snap)
    b3.inner.i, b3.inner.sid = a.inner.i, a.inner.sid
    assert not torch.equal(b3.next_conv_batch()[0]["input_ids"],
                           ref[0]["input_ids"]), \
        "sans state_dict la reprise devrait diverger (sinon le test ne teste rien)"

    print(f"chat_batch self-test: OK (B={B}×{S} slots, aucune ligne toute-pad, "
          f"masques nuls exactement sur les pads, contenu réel intact, lockstep "
          f"1 write/lane/slot, next_conv délègue en B=1, "
          f"{100 * n_pad_tot / n_tok_tot:.0f}% de padding sur ce tirage ; "
          f"split exact + tile sans session coupée + état sérialisé)")


if __name__ == "__main__":
    _selftest()
