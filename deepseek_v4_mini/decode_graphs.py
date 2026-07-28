"""Runner CUDA graphs pour le pas de décodage — EXPÉRIMENTAL, opt-in.

Le décodage caché du 386M reste launch-bound après les flags eager (~12.4k ops
aten/token, FINDINGS 2026-07-27) : le levier qui absorbe TOUT le compte d'ops
est de rejouer le pas de décodage comme un CUDA graph. Validé sur rig
(2026-07-27) : 135,3 → 23,4 ms/token, et le reliquat était le python de la
boucle (argmax host, remplissage RoPE, copies) — d'où le graph ÉTENDU :

  graph = [ tête RoPE ] → forward → [ argmax → chaînage → comptabilité ]

  * tête : cos/sin lus DANS le graph par index_select sur un compteur de
    position device (`_pos_dev`) — plus aucun remplissage host par token ;
  * queue : argmax des logits, le token produit est recopié dans le buffer
    d'entrée partagé (`_in_buf`) — le graph suivant le consomme tel quel —,
    journalisé dans `_out_toks[_wptr]`, et les compteurs device avancent ;
  * `_chain(fed, n)` : n × `graph.replay()`, ZÉRO aller-retour host — le mode
    de croisière de `decode()` quand stop_id est None.

Ce que la capture exige, et comment c'est obtenu :

  * shapes CONSTANTES à chaque pas → mode « largeur pleine » des AttnCache
    statiques (`enter_full_width`) : toutes les largeurs valent `cap`, la
    comptabilité des blocs fermés passe par des tenseurs device (index_copy_ /
    index_fill_ / add_) — le python ne tourne pas au replay ;
  * aucun scalaire python baké → position et RoPE vivent en tenseurs device
    (tête de graph), l'injection passe par attention._ROPE_OVERRIDE ;
  * aucune shape data-dependent → exige les flags decode_fuse +
    decode_dense_moe + decode_static_cache, et B == 1 (le gate du MoE dense) ;
  * une banque EXPLICITE (init_mem non None) : un torch.rand de seed_bank
    capturé rejouerait la même « randomness » à chaque token ;
  * la capture ENREGISTRE sans exécuter (sémantique cudaStreamBeginCapture) :
    chaque capture est suivie d'un replay immédiat qui exécute réellement le
    pas (sorties + mutations de cache + compteurs).

Bras bf16 (`amp="bf16"`) : autocast CUDA avec cache_enabled=False (la
contrainte documentée pour capturer sous autocast). Même statut opt-in que le
reste — un rollout RL le prend, une éval comparée non.

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

import contextlib
import math
import warnings

import torch

from . import attention
from .attention import _rope_at_cached


class GraphDecodeRunner:
    """Décodage greedy token-par-token, pas rejoué par CUDA graph.

    B est FIXÉ au constructeur (la shape de la banque) et baké dans les
    buffers et les graphs : un runner par largeur de groupe. À B > 1 le gate
    du MoE dense est élargi à B pendant le mode graphs (attribut runtime
    `dense_max_bt`, classe ULP — le chemin eager bit-exact reste BT==1).

    Usage :
        runner = GraphDecodeRunner(model, bank)          # exige les 3 flags
        gen = runner.decode(prefix, max_new=64, stop_id=eos)
        runner.rebind(bank2)         # décodage suivant : graphs CONSERVÉS,
        gen2 = runner.decode(...)    # warmup et captures jamais repayés

    Sur CPU (ou si la capture échoue) : eager pur en mode statique NORMAL,
    strictement identique à decode.generate — c'est le self-test. La bascule
    largeur pleine + graphs n'arme que sur CUDA, après `warmup` pas MONO-token
    eager (fenêtres pleines, kernels S=1 chauffés — indispensable sous
    autocast : une première exécution de GEMM pendant la capture alloue du
    workspace cublas, interdit).

    `amp="bf16"` : forward sous autocast bf16 (cache_enabled=False, la forme
    capturable). Classe ULP assumée — les logits changent, le greedy peut
    dévier. `chain=False` désactive le mode croisière `_chain` (diagnostic).
    """

    def __init__(self, model, bank: torch.Tensor, *, layer_banks=None,
                 warmup: int = 32, amp: str | None = None,
                 chain: bool = True) -> None:
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
        if amp not in (None, "bf16"):
            raise ValueError(f"amp={amp!r} : seul 'bf16' (ou None) est supporté")
        self.model = model
        # le runner POSSÈDE ses buffers de banque : les graphs (et le mémo
        # fw_A/fw_B) voient ces adresses-là, et rebind() y recopie l'état du
        # décodage suivant — jamais de réallocation
        self.bank = bank.detach().clone()
        self.B = int(bank.size(0))
        self.layer_banks = None if layer_banks is None else [
            None if x is None else x.detach().clone() for x in layer_banks]
        self.warmup = int(warmup)
        self.amp = amp
        self.chain = bool(chain)
        self.lcm = math.lcm(int(cfg.csa_m), int(cfg.hca_m))
        self.d_head = int(cfg.d_head)
        # une génération entière doit tenir dans les caches statiques
        self.max_pos = int(cfg.max_seq_len)

        self.cache = None       # posé au premier step (préfixe)
        self.pos = 0
        self._singles = 0       # pas mono-token déjà servis en eager (warmup)
        self.graphs: dict[int, tuple] = {}   # phase -> (graph, out)
        # Le juge est le DEVICE DU MODÈLE (via la banque), pas
        # cuda.is_available() : sur une machine dont le GPU est occupé par un
        # autre run, un décodage CPU ne doit JAMAIS créer de contexte CUDA
        # (torch.cuda.CUDAGraph en crée un — ~centaines de Mo de VRAM chez le
        # voisin, règle un-run-par-GPU).
        self.eager_only = not (bank.is_cuda and torch.cuda.is_available())
        if self.amp and self.eager_only:
            warnings.warn("amp='bf16' ignoré : le modèle n'est pas sur CUDA "
                          "(autocast bf16 n'arme qu'avec les graphs)",
                          stacklevel=2)
        self._rope_bufs = None
        self._full = False

    # ── un pas ───────────────────────────────────────────────────────────────

    @torch.no_grad()
    def step(self, fed: torch.Tensor) -> torch.Tensor:
        """fed [1, S] (préfixe au premier appel, puis [1, 1]) → logits [1,1,V]."""
        assert fed.size(0) == self.B, \
            f"B={fed.size(0)} ≠ {self.B} : B est baké dans les buffers du runner"
        if self.cache is None:
            self.cache = self.model.make_cache()
            assert self.cache and self.cache[0].cap is not None
        out = None
        if self.pos + fed.size(1) > self.max_pos:
            raise RuntimeError(f"génération > max_seq_len={self.max_pos} : "
                               f"les caches statiques sont dimensionnés dessus")

        graphs_ok = not (self.eager_only or fed.size(1) > 1
                         or self._singles < self.warmup)
        if graphs_ok and not self._full:
            # armement (ou RÉ-armement post-rebind) : n'entrer en largeur
            # pleine que si l'état eager reproduit les largeurs hist bakées
            # dans les graphs — sinon rester en singles eager statiques
            graphs_ok = self._enter_ready()
            if graphs_ok:
                self._enter_full()
        if not graphs_ok:
            out = self._eager(fed)
            if fed.size(1) == 1:
                self._singles += 1
        else:
            phase = self.pos % self.lcm
            if phase not in self.graphs:
                out = self._capture(phase, fed)
            if out is None:
                out = self._replay(phase, fed)
        self.pos += fed.size(1)
        return out

    def decode(self, prefix: torch.Tensor, *, max_new: int = 48,
               stop_id: int | None = None, chunk: int = 16):
        """Greedy batché — même contrat que decode.generate : rend
        (gen [B, n], lens [B]), `stop_id` INCLUS dans lens, ce qui suit
        `lens[i]` est du remplissage produit pendant que les autres lignes
        finissaient. L'arrêt est par ligne, la boucle s'arrête quand TOUTES
        ont émis stop_id (ou au plafond).

        Mode croisière : dès que les lcm phases sont capturées, le reste part
        en `_chain` — replays purs, argmax dans le graph. Avec stop_id, la
        chaîne avance par tranches de `chunk` tokens et scanne le JOURNAL une
        fois par tranche (une sync par tranche, pas par token) ; le dépassement
        au-delà du stop est du remplissage, comme dans generate."""
        B, dev = self.B, prefix.device
        toks = []
        fed = prefix
        done = torch.zeros(B, dtype=torch.bool, device=dev)
        lens = torch.full((B,), max_new, dtype=torch.long, device=dev)
        n = 0
        while n < max_new:
            if (self.chain and self._full and not self.eager_only
                    and len(self.graphs) == self.lcm and fed.size(1) == 1):
                k = max_new - n if stop_id is None else min(chunk, max_new - n)
                seg = self._chain(fed, k)               # [B, k]
                toks.append(seg)
                fed = seg[:, -1:]
                if stop_id is not None:
                    hits = seg == stop_id
                    hit_any = hits.any(1) & ~done
                    first = hits.int().argmax(1)        # 1er hit de la tranche
                    lens = torch.where(hit_any, n + first + 1, lens)
                    done = done | hit_any
                    n += k
                    if bool(done.all()):
                        break
                else:
                    n += k
                continue
            out = self.step(fed)
            nt = out["logits"][:, -1].argmax(-1, keepdim=True)
            toks.append(nt)
            fed = nt
            n += 1
            if stop_id is not None:
                hit = (nt.squeeze(1) == stop_id) & ~done
                lens = torch.where(hit, torch.full_like(lens, n), lens)
                done = done | hit
                if bool(done.all()):
                    break
        gen = torch.cat(toks, dim=1)
        return gen, lens.clamp(max=n)

    # ── chemins internes ─────────────────────────────────────────────────────

    def _autocast(self):
        if self.amp is None or self.eager_only:
            return contextlib.nullcontext()
        # cache_enabled=False : la contrainte documentée pour capturer sous
        # autocast (le cache de recast d'autocast ne survit pas à un graph)
        return torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False)

    def _eager(self, fed):
        with self._autocast():
            return self.model(fed, init_mem=self.bank,
                              layer_banks=self.layer_banks,
                              write=False, cache=self.cache)

    def _enter_ready(self) -> bool:
        """Les largeurs hist sont-elles en RÉGIME ÉTABLI à cette position ?

        `hist_len` est un int python baké comme taille de slice à la capture
        (attention.py, mode full) — la SEULE dépendance de forme au temps qui
        ne soit pas un tenseur device. En régime établi elle vaut
        `m + (pos % m)` (atteint dès pos ≥ m) : une fonction PÉRIODIQUE de la
        position, la condition pour qu'un graph capturé à la phase p soit
        rejouable à toute position ultérieure ≡ p (mod lcm). Armement ET
        ré-armement post-rebind attendent cette condition en singles eager —
        sinon un graph lirait au-delà du rempli, faux en silence (un préfixe
        court post-rebind, le cas a_open des rollouts, y passerait)."""
        cfg = self.model.cfg
        for li, c in enumerate(self.cache):
            m_l = int(cfg.csa_m) if li % 2 == 0 else int(cfg.hca_m)
            w = c.hist_len if c.full else (
                0 if c.hist is None else c.hist.size(1))
            if w != m_l + (self.pos % m_l):
                return False
        return True

    def _enter_full(self):
        cfg = self.model.cfg
        for li, c in enumerate(self.cache):
            # hist_cap = 2m selon la parité de la couche (même règle que
            # model.make_cache) — le cache ne connaît pas son m ; à la
            # ré-entrée (rebind) enter_full_width recopie l'état dans les
            # sbufs EXISTANTS, les adresses que les graphs connaissent
            m_l = int(cfg.csa_m) if li % 2 == 0 else int(cfg.hca_m)
            c.enter_full_width(hist_cap=2 * m_l)
        # le gate du MoE dense s'élargit à B — attribut runtime, PAS un flag de
        # config : le chemin eager bit-exact (BT==1) reste intact partout
        # ailleurs, l'élargissement est classe ULP comme le reste du runner
        for mod in self.model.modules():
            if hasattr(mod, "dense_max_bt"):
                mod.dense_max_bt = self.B
        dev = self.bank.device
        if self._rope_bufs is None:     # ── première fois : allouer ──────────
            half = self.d_head // 2
            self._rope_bufs = (torch.zeros(1, half, device=dev),
                               torch.zeros(1, half, device=dev))
            # table RoPE forcée à pleine capacité PUIS référencée : la tête de
            # graph la lit par index_select — si _rope_at_cached la
            # reconstruisait plus tard, le graph pointerait sur l'ancien
            # storage. Capacité ≥ max_pos + garde-fou step() sur max_pos ⇒
            # jamais de reconstruction.
            _rope_at_cached(self.max_pos - 1, 1, self.d_head, dev)
            self._rope_tab = attention._ROPE_TABLES[(str(dev), self.d_head)]
            # buffers du graph étendu : entrée partagée entre toutes les
            # phases (le tail de l'une nourrit la tête de la suivante),
            # journal des tokens produits, compteurs device
            self._in_buf = torch.zeros(self.B, 1, dtype=torch.long, device=dev)
            self._out_toks = torch.zeros(self.max_pos, self.B,
                                         dtype=torch.long, device=dev)
            self._wptr = torch.zeros(1, dtype=torch.long, device=dev)
            self._pos_dev = torch.tensor([self.pos], dtype=torch.long,
                                         device=dev)
        else:                           # ── ré-entrée (rebind) : en place ────
            self._wptr.zero_()
            self._pos_dev.fill_(self.pos)
        self._full = True

    def _graph_head(self):
        """Tête de graph : cos/sin de la position courante, lus depuis la
        table par le compteur device — remplace le remplissage host par token."""
        self._rope_bufs[0].copy_(self._rope_tab[0].index_select(0, self._pos_dev))
        self._rope_bufs[1].copy_(self._rope_tab[1].index_select(0, self._pos_dev))

    def _graph_tail(self, out):
        """Queue de graph : argmax, chaînage token→token, comptabilité device."""
        nt = out["logits"][:, -1].argmax(-1)            # [B]
        self._in_buf.copy_(nt.unsqueeze(1))             # le graph suivant lit ça
        self._out_toks.index_copy_(0, self._wptr, nt.unsqueeze(0))
        self._wptr.add_(1)
        self._pos_dev.add_(1)

    def _capture(self, phase, fed):
        """Capture le graph étendu de cette phase, puis le REJOUE aussitôt.

        La capture ENREGISTRE sans exécuter (sémantique cudaStreamBeginCapture) :
        sans le replay immédiat, la sortie rendue serait de la mémoire jamais
        écrite et — bien pire — les mutations de cache de ce pas (fenêtres,
        fermetures de bloc) n'auraient jamais lieu. Échec ⇒ fallback eager
        définitif, bruyant."""
        try:
            self._in_buf.copy_(fed)
            attention._ROPE_OVERRIDE = self._rope_bufs
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            # pool partagé entre les phases : les activations intermédiaires
            # d'une phase (hist, fenêtres) sont relues par la suivante aux
            # mêmes adresses — d'où la capture DANS L'ORDRE du premier cycle
            pool = getattr(self, "_pool", None)
            with torch.cuda.graph(g, pool=pool):
                self._graph_head()
                out = self._eager(self._in_buf)
                self._graph_tail(out)
            g.replay()                  # exécuter RÉELLEMENT le pas capturé
            torch.cuda.synchronize()    # faire surfacer ICI toute erreur de
            self._pool = g.pool() if pool is None else pool   # capture différée
            self.graphs[phase] = (g, out)
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
            # TOUJOURS rendre l'override : il ne sert qu'à l'ENREGISTREMENT
            # (les replays ne font pas tourner le python, la capture suivante
            # le repose). Le laisser posé empoisonnait tout forward eager
            # ultérieur du même process — les singles post-rebind lisaient la
            # RoPE de la DERNIÈRE position du décodage précédent (job 35 :
            # divergence au 1er single sur préfixe court, invisible sur
            # préfixe long qui n'a aucun single).
            attention._ROPE_OVERRIDE = None

    def _replay(self, phase, fed):
        g, out = self.graphs[phase]
        # redondant en chaîne (le tail du graph précédent a déjà posé le token)
        # mais nécessaire en pilotage python — même valeur dans les deux cas
        self._in_buf.copy_(fed)
        g.replay()
        return out

    @torch.no_grad()
    def _chain(self, fed: torch.Tensor, n: int) -> torch.Tensor:
        """n tokens en replays PURS — zéro aller-retour host. Le premier replay
        consomme `fed`, chaque tail nourrit la tête suivante ; les tokens
        sortent de `_out_toks` (une lecture device, la sync arrive quand
        l'appelant matérialise). Exige les lcm phases capturées."""
        if self.pos + n > self.max_pos:
            raise RuntimeError(f"chaîne de {n} tokens > max_seq_len="
                               f"{self.max_pos} depuis pos {self.pos}")
        self._in_buf.copy_(fed)
        self._wptr.zero_()
        for _ in range(n):
            self.graphs[self.pos % self.lcm][0].replay()
            self.pos += 1
        return self._out_toks[:n].t().clone()           # [B, n]

    @torch.no_grad()
    def rebind(self, bank: torch.Tensor, layer_banks=None) -> None:
        """Réarme le runner pour un NOUVEAU décodage sans repayer warmup ni
        captures : l'état neuf est recopié DANS les objets que les graphs
        connaissent, jamais réalloué.

          * banque (et layer_banks) : `copy_` dans les buffers possédés —
            mêmes shapes/dtype/device exigés (un runner = une signature) ;
          * caches : `AttnCache.reset()` (comp/dead/idx/sbufs préservés) ;
          * A/Bm mémoïsés (entrées des graphs, bakés à la capture) :
            `_fw_refresh` recalcule et recopie en place ;
          * gate MoE dense restauré à 1 : les singles post-rebind empruntent
            le chemin eager statique NORMAL (le même qu'un runner frais),
            `_enter_full` ré-élargira à la bascule.

        Les graphs, le pool, les buffers RoPE et le compteur de warmup
        survivent — c'est tout l'intérêt. La bascule largeur pleine ne
        réarme qu'aux positions où les largeurs hist eager coïncident avec
        les valeurs bakées (voir `_enter_ready`)."""
        if (bank.shape != self.bank.shape or bank.dtype != self.bank.dtype
                or bank.device != self.bank.device):
            raise ValueError(
                f"rebind : banque {tuple(bank.shape)}/{bank.dtype} ≠ buffer "
                f"{tuple(self.bank.shape)}/{self.bank.dtype} — un runner est "
                f"baké sur UNE signature, en construire un autre")
        if (layer_banks is None) != (self.layer_banks is None) or (
                layer_banks is not None
                and (len(layer_banks) != len(self.layer_banks)
                     or any((a is None) != (b is None)
                            for a, b in zip(layer_banks, self.layer_banks)))):
            raise ValueError(
                "rebind : structure de layer_banks ≠ celle du constructeur")
        self.bank.copy_(bank)
        if layer_banks is not None:
            for buf, nb in zip(self.layer_banks, layer_banks):
                if buf is not None:
                    buf.copy_(nb)
        if self.cache is not None:
            for c in self.cache:
                c.reset()
        self.pos = 0
        self._full = False
        # ceinture : aucun override RoPE ne doit survivre jusqu'aux singles
        # eager du prochain décodage (la leçon du job 35)
        attention._ROPE_OVERRIDE = None
        for mod in self.model.modules():
            if hasattr(mod, "dense_max_bt"):
                mod.dense_max_bt = 1
            if hasattr(mod, "_fw_refresh"):
                mod._fw_refresh()

    def close(self):
        """Rend la main proprement : override RoPE (module-level) et gate du
        MoE dense (attribut runtime élargi à B en mode graphs) restaurés."""
        attention._ROPE_OVERRIDE = None
        for mod in self.model.modules():
            if hasattr(mod, "dense_max_bt"):
                mod.dense_max_bt = 1


# ── self-test (CPU-only : la capture réelle attend un GPU libre) ─────────────

def _selftest() -> None:
    """Ce qui est prouvable sans GPU :
      1. le runner refuse les configs sans flags, l'absence de banque, un amp
         inconnu — et prévient qu'amp n'arme pas sur CPU ;
      2. sur CPU il dégrade en eager et rend EXACTEMENT decode.generate ;
      3. l'injection RoPE est bit-identique au calcul en place ;
      4. le PROGRAMME DU GRAPH ÉTENDU (tête RoPE → forward → queue argmax/
         chaînage) a des ops et shapes CONSTANTS par phase sur deux cycles
         lcm — le proxy CPU de la capturabilité ;
      5. la comptabilité device des blocs (idx/index_copy_) ferme les mêmes
         blocs que la comptabilité python ;
      6. le chaînage émulé (le programme du graph rejoué en boucle depuis
         _in_buf) rend EXACTEMENT les tokens du pilotage python en largeur
         pleine, et ses compteurs device tombent juste.
    """
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
    try:
        GraphDecodeRunner(m_on, bank, amp="fp16")
        raise SystemExit("amp='fp16' aurait dû être refusé")
    except ValueError as e:
        assert "bf16" in str(e)
    with warnings.catch_warnings(record=True) as wlog:
        warnings.simplefilter("always")
        r_amp = GraphDecodeRunner(m_on, bank, amp="bf16")
    assert any("ignoré" in str(x.message) for x in wlog), \
        "amp sur CPU doit prévenir qu'il n'arme pas"
    assert isinstance(r_amp._autocast(), contextlib.nullcontext), \
        "amp sur CPU (eager_only) doit rester un nullcontext"

    # 2. CPU ⇒ eager, tokens == generate (bit, float64, top-k durs compris).
    #    À B=2 aussi : en eager le gate MoE dense reste BT==1 (il ne s'élargit
    #    qu'en mode graphs), le chemin est donc bit-exact face à generate.
    for Bt in (1, 2):
        bank_t = bank if Bt == 1 else torch.randn(Bt, 3, 16, dtype=torch.float64)
        pr = torch.randint(0, 61, (Bt, 5))
        runner = GraphDecodeRunner(m_on, bank_t)
        assert runner.eager_only or torch.cuda.is_available()
        g1, l1 = runner.decode(pr, max_new=17)
        g2, l2 = generate(m_on, pr, bank=bank_t, max_new=17, use_cache=True)
        assert torch.equal(g1, g2) and torch.equal(l1, l2), \
            f"runner eager ≠ generate (B={Bt})\n  runner  : {g1}\n  generate: {g2}"
        # …et le contrat stop_id (arrêt par ligne, lens stop-inclus,
        # remplissage après) : un stop tiré DES tokens générés garantit le
        # hit. Le second décodage passe par rebind() — le chemin eager
        # post-rebind doit être EXACTEMENT generate, comme un runner frais.
        runner.rebind(bank_t)
        sid = int(g2[Bt - 1, 7])
        g3, l3 = runner.decode(pr, max_new=17, stop_id=sid)
        g4, l4 = generate(m_on, pr, bank=bank_t, max_new=17, stop_id=sid,
                          use_cache=True)
        assert torch.equal(g3, g4) and torch.equal(l3, l4), \
            f"contrat stop_id ≠ generate (B={Bt}, stop={sid})\n" \
            f"  runner  : {g3} lens={l3}\n  generate: {g4} lens={l4}"
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

    # 4. largeur pleine : le programme du graph ÉTENDU (tête → forward →
    #    queue) a des ops et shapes CONSTANTS par phase sur 2 cycles — le
    #    proxy CPU de « ce pas est capturable dans un CUDA graph »
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
    pr4 = torch.randint(0, 61, (1, 5))
    with torch.no_grad():
        runner2.step(pr4)                           # préfixe (eager)
        fed = torch.randint(0, 61, (1, 1))
        for _ in range(4):                          # warmup mono-token eager
            runner2.step(fed)
        # …puis bascule largeur pleine SANS CUDA (test structurel) : on rejoue
        # à la main le programme que _capture mettrait dans le graph
        runner2._enter_full()
        runner2._in_buf.copy_(fed)
        sigs = {}
        for _ in range(2 * lcm):
            probe = ShapeProbe()
            attention._ROPE_OVERRIDE = runner2._rope_bufs
            try:
                with probe:
                    runner2._graph_head()
                    out = runner2._eager(runner2._in_buf)
                    runner2._graph_tail(out)
            finally:
                attention._ROPE_OVERRIDE = None
            ph = runner2.pos % lcm
            runner2.pos += 1
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

    # 6. chaînage émulé == pilotage python (largeur pleine, greedy, au bit) :
    #    deux runners jumeaux, même préfixe, même warmup ; l'un piloté par
    #    argmax python, l'autre par le programme du graph bouclé sur _in_buf.
    #    À B=2 : exerce les buffers batchés ET le gate MoE dense élargi
    #    (dense_max_bt=B posé par _enter_full — les deux bras l'utilisent).
    for B2 in (1, 2):
        bank2 = torch.randn(B2, 3, 16, dtype=torch.float64)
        ra = GraphDecodeRunner(m_on, bank2, warmup=2)
        rb = GraphDecodeRunner(m_on, bank2, warmup=2)
        pr2 = torch.randint(0, 61, (B2, 4))
        with torch.no_grad():
            f = None
            for r in (ra, rb):
                o = r.step(pr2)
                f = o["logits"][:, -1].argmax(-1, keepdim=True)
                for _ in range(2):
                    o = r.step(f)
                    f = o["logits"][:, -1].argmax(-1, keepdim=True)
                r._enter_full()
            assert all(mod.dense_max_bt == B2
                       for mod in m_on.modules()
                       if hasattr(mod, "dense_max_bt")), \
                "_enter_full doit élargir le gate MoE dense à B"
            # stabilité d'adresses : l'état inter-pas vit dans des buffers
            # STABLES — c'est ce que les graphs captureront comme interface
            addrs = [(id(c.wk), id(c.wv), id(c.hist), id(c.idx))
                     for c in rb.cache]
            pos0 = int(rb._pos_dev)
            n = 9
            toks_a, fa = [], f.clone()
            for _ in range(n):                      # bras python
                o = ra._eager(fa)
                fa = o["logits"][:, -1].argmax(-1, keepdim=True)
                toks_a.append(fa)
            rb._in_buf.copy_(f)                     # bras « graph émulé »
            rb._wptr.zero_()
            for _ in range(n):
                attention._ROPE_OVERRIDE = rb._rope_bufs
                try:
                    rb._graph_head()
                    o = rb._eager(rb._in_buf)
                    rb._graph_tail(o)
                finally:
                    attention._ROPE_OVERRIDE = None
        assert addrs == [(id(c.wk), id(c.wv), id(c.hist), id(c.idx))
                         for c in rb.cache], (
            "l'état inter-pas a changé d'objet en mode largeur pleine — la "
            "couture du cycle des graphs se rouvre (wk/wv/hist doivent être "
            "mis à jour EN PLACE)")
        got = rb._out_toks[:n].t()                  # [B, n]
        want = torch.cat(toks_a, dim=1)
        assert torch.equal(want, got), (
            f"chaînage émulé ≠ pilotage python (B={B2})\n"
            f"  python : {want}\n  chaîne : {got}")
        assert int(rb._wptr) == n and int(rb._pos_dev) == pos0 + n, \
            "compteurs device du chaînage faux"
        ra.close(); rb.close()
        assert all(mod.dense_max_bt == 1 for mod in m_on.modules()
                   if hasattr(mod, "dense_max_bt")), \
            "close() doit restaurer le gate MoE dense"

    # 7. rebind : le programme du graph (émulé) survit au changement de
    #    décodage — mêmes adresses partout, garde de régime établi qui retient
    #    la bascule sur préfixe court, tokens == runner FRAIS au bit
    def _argmax(o):
        return o["logits"][:, -1].argmax(-1, keepdim=True)

    def _drive_full(r, n):
        """le programme du graph étendu, bouclé — l'émulation CPU des replays"""
        for _ in range(n):
            attention._ROPE_OVERRIDE = r._rope_bufs
            try:
                r._graph_head()
                o = r._eager(r._in_buf)
                r._graph_tail(o)
            finally:
                attention._ROPE_OVERRIDE = None
            r.pos += 1

    for B3 in (1, 2):
        torch.manual_seed(31 + B3)
        bank_a = torch.randn(B3, 3, 16, dtype=torch.float64)
        bank_b = torch.randn(B3, 3, 16, dtype=torch.float64)
        ra = GraphDecodeRunner(m_on, bank_a, warmup=2)
        pr_a = torch.randint(0, 61, (B3, 6))
        with torch.no_grad():
            f = _argmax(ra.step(pr_a))
            for _ in range(2):
                f = _argmax(ra.step(f))
            assert ra._enter_ready(), "régime établi attendu ici (pos ≥ m)"
            ra._enter_full()
            ra._in_buf.copy_(f)
            ra._wptr.zero_()
            _drive_full(ra, lcm + 3)                    # « décodage 1 »
        addrs = [(id(c.comp), id(c.dead), id(c.idx), *map(id, c.sbufs))
                 for c in ra.cache]
        io_ids = (id(ra._in_buf), id(ra._out_toks), id(ra._wptr),
                  id(ra._pos_dev), id(ra.bank))
        memo_ids = [(id(b._fw_memo[2]), id(b._fw_memo[3]))
                    for b in m_on.blocks if b._fw_memo is not None]
        assert memo_ids, "le mémo fw doit être chaud (entrées des graphs)"

        # sur GPU, _capture laisse l'override RoPE posé entre deux captures —
        # le simuler AVANT le rebind : les singles eager du décodage 2 ne
        # doivent pas lire la RoPE de la dernière position du décodage 1
        # (job 35 : divergence au 1er single sur préfixe court)
        attention._ROPE_OVERRIDE = ra._rope_bufs
        ra.rebind(bank_b)                               # banque NEUVE
        assert attention._ROPE_OVERRIDE is None, \
            "rebind doit rendre l'override RoPE (singles eager empoisonnés)"
        assert all(mod.dense_max_bt == 1 for mod in m_on.modules()
                   if hasattr(mod, "dense_max_bt")), \
            "rebind doit restaurer le gate MoE (singles = chemin normal)"
        pr_b = torch.randint(0, 61, (B3, 2))            # préfixe COURT (< m)
        with torch.no_grad():
            f = _argmax(ra.step(pr_b))
            n_sing = 0
            while not ra._enter_ready():
                f = _argmax(ra.step(f))
                n_sing += 1
                assert n_sing < 4 * lcm, "garde de régime établi : divergence"
            assert n_sing >= 1, "préfixe court : la garde devait RETENIR"
            ra._enter_full()
            assert addrs == [(id(c.comp), id(c.dead), id(c.idx),
                              *map(id, c.sbufs)) for c in ra.cache] \
                and io_ids == (id(ra._in_buf), id(ra._out_toks), id(ra._wptr),
                               id(ra._pos_dev), id(ra.bank)) \
                and memo_ids == [(id(b._fw_memo[2]), id(b._fw_memo[3]))
                                 for b in m_on.blocks
                                 if b._fw_memo is not None], \
                "le rebind a réalloué un objet vu par les graphs"
            ra._in_buf.copy_(f)
            ra._wptr.zero_()
            n2 = lcm + 2
            _drive_full(ra, n2)                         # « décodage 2 »
            got = ra._out_toks[:n2].t().clone()

        # référence : runner FRAIS sur bank_b, MÊME calendrier d'armement —
        # le décodage post-rebind doit lui être identique au bit
        rf = GraphDecodeRunner(m_on, bank_b, warmup=0)
        with torch.no_grad():
            f2 = _argmax(rf.step(pr_b))
            for _ in range(n_sing):
                f2 = _argmax(rf.step(f2))
            rf._enter_full()
            rf._in_buf.copy_(f2)
            rf._wptr.zero_()
            _drive_full(rf, n2)
            want = rf._out_toks[:n2].t()
        assert torch.equal(f, f2), \
            f"singles post-rebind ≠ runner frais (B={B3})"
        assert torch.equal(want, got), (
            f"décodage post-rebind ≠ runner frais (B={B3})\n"
            f"  frais  : {want}\n  rebind : {got}")
        ra.close()
        rf.close()

    # …et rebind refuse une mauvaise signature
    r_sig = GraphDecodeRunner(m_on, torch.randn(1, 3, 16, dtype=torch.float64))
    for bad in (torch.randn(2, 3, 16, dtype=torch.float64),
                torch.randn(1, 4, 16, dtype=torch.float64),
                torch.randn(1, 3, 16)):
        try:
            r_sig.rebind(bad)
            raise SystemExit(f"rebind aurait dû refuser {tuple(bad.shape)}/"
                             f"{bad.dtype}")
        except ValueError as e:
            assert "signature" in str(e)
    try:
        r_sig.rebind(torch.randn(1, 3, 16, dtype=torch.float64),
                     layer_banks=[None])
        raise SystemExit("rebind aurait dû refuser des layer_banks inattendus")
    except ValueError as e:
        assert "layer_banks" in str(e)
    r_sig.close()

    print(f"decode_graphs self-test: OK (refus sans flags/banque/amp inconnu, "
          f"eager CPU == generate au bit à B∈{{1,2}}, injection RoPE "
          f"bit-identique, programme du graph étendu : signature ops+shapes "
          f"CONSTANTE par phase sur 2×lcm={2 * lcm} pas, comptabilité de "
          f"blocs device == python, chaînage émulé == pilotage python au bit "
          f"à B∈{{1,2}} avec état inter-pas à ADRESSES STABLES (couture du "
          f"cycle fermée) + gate MoE dense élargi/restauré ; REBIND : "
          f"décodage 2 == runner frais au bit à B∈{{1,2}}, toutes adresses "
          f"(caches, buffers I/O, banque, mémo A/Bm) inchangées, garde de "
          f"régime établi qui retient la bascule sur préfixe court, refus "
          f"des mauvaises signatures ; la capture réelle attend un GPU libre)")


if __name__ == "__main__":
    _selftest()
