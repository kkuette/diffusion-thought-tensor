"""Runner CUDA graphs pour le pas de décodage — EXPÉRIMENTAL, opt-in.

Le décodage caché du 386M reste launch-bound après les flags eager (~12.4k ops
aten/token, FINDINGS 2026-07-27) : le levier qui absorbe TOUT le compte d'ops
est de rejouer le pas de décodage comme un CUDA graph. Ce module prépare cette
capture ; le chrono et la validation GPU attendent un GPU libre (règle
un-run-par-GPU).

Ce que la capture exige, et comment c'est obtenu :

  * shapes CONSTANTES à chaque pas → mode « largeur pleine » des AttnCache
    statiques (`enter_full_width`) : toutes les largeurs valent `cap`, la
    comptabilité des blocs fermés passe par des tenseurs device (index_copy_ /
    index_fill_ / add_) — le python ne tourne pas au replay ;
  * aucun scalaire python baké → la position RoPE est INJECTÉE via deux
    buffers statiques (attention._ROPE_OVERRIDE) remplis avant chaque replay ;
  * aucune shape data-dependent → exige les flags decode_fuse +
    decode_dense_moe + decode_static_cache, et B == 1 (le gate du MoE dense) ;
  * une banque EXPLICITE (init_mem non None) : un torch.rand de seed_bank
    capturé rejouerait la même « randomness » à chaque token.

CE QUE LE MODE LARGEUR PLEINE CHANGE (classe ULP, comme decode_cache — c'est
pour ça que le runner est un OBJET à construire explicitement, pas un flag de
config) : le top-k CSA reçoit `cap` candidats au lieu de nbf, ses ex æquo
(relu ⇒ 0.0 exacts) peuvent se départager autrement ; le softmax HCA élargi
s'associe autrement (~5e-15). Rien n'est faux — c'est un autre échantillon
aussi légitime (cf. le commentaire de decode.generate) — mais ce n'est PAS
bit-identique au décodage eager : à réserver aux rollouts RL, pas aux évals
comparées.

La capture par phase : après `decode_fuse` etc., la seule dépendance de
FORME au temps qui reste est le calendrier de fermeture des blocs — périodique
de période lcm(csa_m, hca_m) (16 sur le 386M). Un graph par phase
`pos % lcm`, capturés dans l'ordre au premier cycle post-warmup (le chaînage
des activations intermédiaires entre graphs passe par le pool mémoire partagé,
d'où l'ordre). Toute erreur de capture ⇒ fallback eager BRUYANT, jamais
silencieux (piège WSL2 connu).
"""
from __future__ import annotations

import math
import warnings

import torch

from . import attention
from .attention import _rope_at_cached


class GraphDecodeRunner:
    """Décodage greedy B=1 token-par-token, pas rejoué par CUDA graph.

    Usage :
        runner = GraphDecodeRunner(model, bank)          # exige les 3 flags
        gen = runner.decode(prefix, max_new=64, stop_id=eos)

    Sur CPU (ou si la capture échoue) : eager pur en mode statique NORMAL,
    strictement identique à decode.generate — c'est le self-test. La bascule
    largeur pleine + graphs n'arme que sur CUDA, après `warmup` tokens eager
    (fenêtres pleines, k CSA saturé, régime de fermeture établi).
    """

    def __init__(self, model, bank: torch.Tensor, *, layer_banks=None,
                 warmup: int = 32) -> None:
        cfg = model.cfg
        missing = [f for f in ("decode_fuse", "decode_dense_moe",
                               "decode_static_cache")
                   if not bool(getattr(cfg, f, False))]
        if missing:
            raise ValueError(
                f"GraphDecodeRunner exige les flags {missing} (shapes fixes / "
                f"zéro sync) — les poser dans la config du modèle")
        if bank is None:
            raise ValueError(
                "GraphDecodeRunner exige une banque EXPLICITE : le torch.rand "
                "de seed_bank capturé dans un graph rejouerait la même "
                "« randomness » à chaque token")
        self.model = model
        self.bank = bank
        self.layer_banks = layer_banks
        self.warmup = int(warmup)
        self.lcm = math.lcm(int(cfg.csa_m), int(cfg.hca_m))
        self.d_head = int(cfg.d_head)
        # une génération entière doit tenir dans les caches statiques
        self.max_pos = int(cfg.max_seq_len)

        self.cache = None       # posé au premier step (préfixe)
        self.pos = 0
        self.graphs: dict[int, tuple] = {}   # phase -> (graph, in_buf, out_buf)
        # Le juge est le DEVICE DU MODÈLE (via la banque), pas
        # cuda.is_available() : sur une machine dont le GPU est occupé par un
        # autre run, un décodage CPU ne doit JAMAIS créer de contexte CUDA
        # (torch.cuda.CUDAGraph en crée un — ~centaines de Mo de VRAM chez le
        # voisin, règle un-run-par-GPU).
        self.eager_only = not (bank.is_cuda and torch.cuda.is_available())
        self._rope_bufs = None
        self._full = False

    # ── un pas ───────────────────────────────────────────────────────────────

    @torch.no_grad()
    def step(self, fed: torch.Tensor) -> torch.Tensor:
        """fed [1, S] (préfixe au premier appel, puis [1, 1]) → logits [1,1,V]."""
        assert fed.size(0) == 1, "B == 1 (gate du MoE dense, buffers statiques)"
        if self.cache is None:
            self.cache = self.model.make_cache()
            assert self.cache and self.cache[0].cap is not None
        out = None
        if self.pos + fed.size(1) > self.max_pos:
            raise RuntimeError(f"génération > max_seq_len={self.max_pos} : "
                               f"les caches statiques sont dimensionnés dessus")

        if self.eager_only or self.pos + fed.size(1) <= self.warmup or fed.size(1) > 1:
            out = self._eager(fed)
        else:
            if not self._full:
                self._enter_full()
            phase = self.pos % self.lcm
            if phase not in self.graphs:
                out = self._capture(phase, fed)
            if out is None:
                out = self._replay(phase, fed)
        self.pos += fed.size(1)
        return out

    def decode(self, prefix: torch.Tensor, *, max_new: int = 48,
               stop_id: int | None = None):
        """Greedy, une ligne. Rend (gen [1, n], len utile) — contrat trim-é
        de decode.generate, sans la machinerie multi-lignes."""
        toks = []
        fed = prefix
        for _ in range(max_new):
            nt = self.step(fed)["logits"][:, -1].argmax(-1, keepdim=True)
            toks.append(nt)
            fed = nt
            if stop_id is not None and int(nt) == stop_id:
                break
        gen = torch.cat(toks, dim=1)
        return gen, gen.size(1)

    # ── chemins internes ─────────────────────────────────────────────────────

    def _eager(self, fed):
        return self.model(fed, init_mem=self.bank, layer_banks=self.layer_banks,
                          write=False, cache=self.cache)

    def _enter_full(self):
        for c in self.cache:
            c.enter_full_width()
        half = self.d_head // 2
        dev = self.bank.device
        self._rope_bufs = (torch.zeros(1, half, device=dev),
                           torch.zeros(1, half, device=dev))
        self._full = True

    def _fill_rope(self):
        cos, sin = _rope_at_cached(self.pos, 1, self.d_head, self.bank.device)
        self._rope_bufs[0].copy_(cos)
        self._rope_bufs[1].copy_(sin)

    def _capture(self, phase, fed):
        """Capture le graph de cette phase — et rend la sortie du pas (la
        capture EXÉCUTE le pas). Échec ⇒ fallback eager définitif, bruyant."""
        try:
            in_buf = fed.clone()
            self._fill_rope()
            attention._ROPE_OVERRIDE = self._rope_bufs
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            # pool partagé entre les phases : les activations intermédiaires
            # d'une phase (hist, fenêtres) sont relues par la suivante aux
            # mêmes adresses — d'où la capture DANS L'ORDRE du premier cycle
            pool = getattr(self, "_pool", None)
            with torch.cuda.graph(g, pool=pool):
                out = self._eager(in_buf)
            torch.cuda.synchronize()    # faire surfacer ICI toute erreur de
            self._pool = g.pool() if pool is None else pool   # capture différée
            self.graphs[phase] = (g, in_buf, out)
            return out
        except Exception as e:                      # noqa: BLE001 — piège WSL2
            import traceback
            warnings.warn(f"capture CUDA graph échouée (phase {phase}, "
                          f"pos {self.pos}) : {e} — fallback eager DÉFINITIF, "
                          f"le gain graphs est perdu\n"
                          f"{traceback.format_exc()}", stacklevel=2)
            self.eager_only = True
            self.graphs.clear()
            attention._ROPE_OVERRIDE = None
            try:                        # l'état CUDA peut être poisseux après
                torch.cuda.synchronize()  # un échec de capture : purger avant
            except Exception:             # de continuer en eager
                pass
            return self._eager(fed)
        finally:
            if self.eager_only:
                attention._ROPE_OVERRIDE = None

    def _replay(self, phase, fed):
        g, in_buf, out = self.graphs[phase]
        in_buf.copy_(fed)
        self._fill_rope()
        g.replay()
        return out

    def close(self):
        """Rend la main proprement (l'override RoPE est module-level)."""
        attention._ROPE_OVERRIDE = None


# ── self-test (CPU-only : la capture réelle attend un GPU libre) ─────────────

def _selftest() -> None:
    """Ce qui est prouvable sans GPU :
      1. le runner refuse les configs sans flags, et l'absence de banque ;
      2. sur CPU il dégrade en eager et rend EXACTEMENT decode.generate ;
      3. l'injection RoPE est bit-identique au calcul en place ;
      4. en mode largeur pleine, le POINT DE CAPTURE est atteignable : les
         shapes et le compte d'ops de chaque pas sont CONSTANTS par phase sur
         deux cycles lcm — le proxy CPU de la capturabilité ;
      5. la comptabilité device des blocs (idx/index_copy_) ferme les mêmes
         blocs que la comptabilité python.
    """
    from collections import Counter

    from torch.utils._python_dispatch import TorchDispatchMode

    from .config import ThoughtBankConfig
    from .decode import generate
    from .model import ThoughtBankLM

    torch.manual_seed(0)
    kw = dict(vocab_size=61, d_model=32, n_layers=2, n_heads=2, d_head=8,
              csa_m=3, hca_m=5, top_k_csa=2, n_win=4, d_latent_q=8,
              n_groups=1, n_experts=2, n_shared=1, top_k_experts=1, d_ff=32,
              mem_dim=16, max_mem=3, mem_seed_slots=2, mem_read_rank=4,
              sinkhorn_iters=5, max_seq_len=128)
    cfg_off = ThoughtBankConfig(**kw)
    cfg_on = ThoughtBankConfig(**kw, decode_fuse=True, decode_dense_moe=True,
                               decode_static_cache=True)

    # 1. refus parlants
    m_off = ThoughtBankLM(cfg_off).double().eval()
    bank = torch.randn(1, 3, 16, dtype=torch.float64)
    try:
        GraphDecodeRunner(m_off, bank)
        raise SystemExit("un modèle sans flags aurait dû être refusé")
    except ValueError as e:
        assert "decode_fuse" in str(e)
    m_on = ThoughtBankLM(cfg_on).double().eval()
    m_on.load_state_dict(m_off.state_dict())
    try:
        GraphDecodeRunner(m_on, None)
        raise SystemExit("bank=None aurait dû être refusé")
    except ValueError as e:
        assert "seed_bank" in str(e)

    # 2. CPU ⇒ eager, tokens == generate (bit, float64, top-k durs compris)
    pr = torch.randint(0, 61, (1, 5))
    runner = GraphDecodeRunner(m_on, bank)
    assert runner.eager_only or torch.cuda.is_available()
    g1, n1 = runner.decode(pr, max_new=17)
    g2, l2 = generate(m_on, pr, bank=bank, max_new=17, use_cache=True)
    assert torch.equal(g1, g2[:, :n1]) and n1 == int(l2[0]), \
        f"runner eager ≠ generate\n  runner  : {g1}\n  generate: {g2}"
    runner.close()

    # 3. injection RoPE : buffers remplis = calcul en place, au bit
    from .attention import AttnCache, CompressedSparseAttention, _mk
    mod = _mk(CompressedSparseAttention, torch.float64, csa_m=3, top_k=2)
    H = torch.randn(1, 11, 24, dtype=torch.float64)
    c1, c2 = AttnCache(), AttnCache()
    with torch.no_grad():
        o1 = mod.forward_cached(H[:, :2], c1)
        o2 = mod.forward_cached(H[:, :2], c2)
        for i in range(2, H.size(1)):
            a = mod.forward_cached(H[:, i:i + 1], c1)
            cos, sin = _rope_at_cached(c2.pos, 1, 8, H.device)
            attention._ROPE_OVERRIDE = (cos.clone(), sin.clone())
            try:
                b = mod.forward_cached(H[:, i:i + 1], c2)
            finally:
                attention._ROPE_OVERRIDE = None
            assert torch.equal(a, b), f"injection RoPE ≠ calcul en place (pos {i})"

    # 4. largeur pleine : ops et shapes CONSTANTS par phase sur 2 cycles —
    #    le proxy CPU de « ce pas est capturable dans un CUDA graph »
    class ShapeProbe(TorchDispatchMode):
        def __init__(self):
            super().__init__()
            self.sig = []

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            out = func(*args, **(kwargs or {}))
            shp = tuple(tuple(t.shape) for t in
                        ((out,) if isinstance(out, torch.Tensor) else
                         tuple(x for x in out if isinstance(x, torch.Tensor))
                         if isinstance(out, (tuple, list)) else ()))
            self.sig.append((func.overloadpacket.__name__, shp))
            return out

    lcm = math.lcm(3, 5)
    runner2 = GraphDecodeRunner(m_on, bank, warmup=4)
    runner2.step(pr)                                # préfixe (eager)
    fed = torch.randint(0, 61, (1, 1))
    while runner2.pos < 4 + 2:                      # sort du warmup en eager…
        runner2.step(fed)
    # …puis on force la bascule largeur pleine SANS CUDA (test structurel)
    if not runner2._full:
        runner2._enter_full()
        runner2.eager_only = True                   # pas de capture sur CPU
    sigs = {}
    for _ in range(2 * lcm):
        probe = ShapeProbe()
        runner2._fill_rope()
        attention._ROPE_OVERRIDE = runner2._rope_bufs
        try:
            with probe:
                runner2.step(fed)
        finally:
            attention._ROPE_OVERRIDE = None
        ph = (runner2.pos - 1) % lcm
        if ph in sigs:
            assert sigs[ph] == probe.sig, (
                f"phase {ph} : la signature d'ops/shapes change entre deux "
                f"cycles — pas capturable")
        else:
            sigs[ph] = probe.sig
    assert len(sigs) == lcm
    runner2.close()

    # 5. la comptabilité device ferme les mêmes blocs que la python
    idxs = [int(c.idx) for c in runner2.cache]
    # en largeur pleine le miroir python `count` est encore tenu à jour en
    # eager : les deux doivent coïncider
    assert idxs == [c.count for c in runner2.cache], (idxs,
        [c.count for c in runner2.cache])

    print(f"decode_graphs self-test: OK (refus sans flags/banque, eager CPU == "
          f"generate au bit, injection RoPE bit-identique, largeur pleine : "
          f"signature ops+shapes CONSTANTE par phase sur 2×lcm={2 * lcm} pas "
          f"(proxy de capturabilité), comptabilité de blocs device == python ; "
          f"la capture réelle attend un GPU libre)")


if __name__ == "__main__":
    _selftest()
