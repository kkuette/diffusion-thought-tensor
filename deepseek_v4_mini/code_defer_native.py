"""dsv6 — FROM-SCRATCH native bank as cross-chunk long-context memory on real code.

Pivot from the graft (code_train.py): grafting the bank onto a pretrained-frozen
host hit the hard ignore-bank tension — a host trained WITHOUT a bank has no slot
for the read (make it consume => generalization drifts; bound it => it carries
nothing; between => blow-up). See memory dsv6-bank-code-memory-defer §VERDICT.

Here the bank is NATIVE (ThoughtBankLM, read/write inside every DualModalBlock),
co-adapted with the model from init — no graft to force. Same deferred structure:
per code chunk (seq_len tokens):
  (1) in-context forward on [chunk, <think>]: next-token LM loss (+ MoE balance) —
      the model learns to emit <think> after context; the per-token write fills
      the bank as it reads the chunk;
  (2) deferred forward on defer_len <blank> tokens (NO context in-window): position
      i predicts the i-th token of the NEXT chunk from the BANK ALONE.
Dual loss L = L_incontext + defer_weight * L_defer over K chunks, bank carried
(TBPTT = whole conversation). Teacher (optional): distill the last bank slot toward
a random projection of the mean-pooled chunk gist, β anneals 1->0.

Success = deferred GAP > 0 (carried beats init_mem=None), STABLE across the anneal
and WSD decay (not the graft's spike-then-crash), while in-context ppl stays sane.

    PYTHONUNBUFFERED=1 python -m deepseek_v4_mini.code_defer_native \
        deepseek_v4_mini/configs/archive/mechanism/code_defer_native_v1.yaml
"""
import os, sys, math, time, yaml, json
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer

from .config import ThoughtBankConfig
from .model import ThoughtBankLM
from .muon import Muon, _split_muon_params
from .code_data import CodeChunkStream
from .cascade import CascadeMemory, default_layer_map
from .cfg_schema import ConfigError, check as check_cfg
from .ckpt import find_resume, load_bank, restore_train_state, save_bank, save_train_state
from .decode import generate
from .runtime import build_fwd, compile_model, enable_tf32, init_ddp
from .sched import (STAIR_END, STAIR_N, beta_at, describe as describe_sched,
                    lr_scale)
from .paths import load_yaml


def _fill(x_ref, tok_id, width):
    """[B, width] tensor filled with tok_id, on x_ref's device/dtype."""
    return torch.full((x_ref.size(0), width), tok_id, dtype=x_ref.dtype, device=x_ref.device)


def _append(x, tok_id):
    return torch.cat([x, _fill(x, tok_id, 1)], dim=1)


def _ic_loss(model, xt, bank, balw, amp, layer_banks=None):
    """In-context next-token CE on xt (=[chunk, <think>]) + MoE balance. Returns
    (loss, new_bank, ce_detached)."""
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
        o = model(xt, init_mem=bank, layer_banks=layer_banks)
    lg = o["logits"].float()
    ce = F.cross_entropy(lg[:, :-1].reshape(-1, lg.size(-1)), xt[:, 1:].reshape(-1))
    loss = ce + balw * o["balance_loss"].float()
    return loss, o["mem_bank"], float(ce.detach())


def _chat_loss(model, x, lmask, bank, balw, amp, layer_banks=None,
               pad_mask=None, m_any=None):
    """Chat-templated segment: next-token CE restricted to supervised positions
    (loss_mask marks the assistant answer + closing <|im_end|>; template/user
    tokens are masked). No <think> append — the template carries its own stop.
    Returns (loss, new_bank, ce_detached_or_None, ce_lane_or_None). User segs
    (mask all-zero) still forward (their WRITE is the point) but contribute no CE.

    `pad_mask` (True = position réelle) n'existe qu'au batch chat : il ne sert
    qu'au write, dont le pooling attentionnel doit ignorer les pads.

    `m_any` remplace un `float(m.sum()) > 0` qui SYNCHRONISAIT le device au
    milieu du forward — l'appelant le sait déjà côté CPU (loss_mask est un
    tenseur CPU avant son .to(device)), et à ~40 segs/step ces synchros
    empêchaient tout recouvrement CPU/GPU dans un régime déjà launch-bound.
    Laissé à None, le test se fait ici comme avant (au prix de la synchro) :
    seul le chemin chaud du trainer passe la valeur.

    Les deux CE retournées sont des TENSEURS détachés, pas des floats : les
    convertir ici coûtait une synchro par seg. L'appelant les accumule et ne
    matérialise qu'au log.
    """
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
        o = model(x, init_mem=bank, layer_banks=layer_banks, pad_mask=pad_mask)
    lg = o["logits"].float()
    loss = balw * o["balance_loss"].float()
    ce_t = ce_lane = None
    m2 = lmask[:, 1:]                                   # targets = positions 1..T-1
    if m_any is None:
        m_any = bool(m2.any())
    if m_any:
        m = m2.reshape(-1)
        ce_tok = F.cross_entropy(lg[:, :-1].reshape(-1, lg.size(-1)),
                                 x[:, 1:].reshape(-1), reduction="none")
        # Normalisation GLOBALE au batch (décision user 2026-07-25) : le poids
        # d'une lane dépend donc de sa longueur et de ses poids SIF.
        ce = (ce_tok * m).sum() / m.sum().clamp_min(1e-6)
        loss = loss + ce
        ce_t = ce.detach()
        # …d'où cette diagnostique PAR LANE, hors gradient : c'est elle qui reste
        # comparable au `chat` d'un run B=1 quand on change le batch.
        #
        # La moyenne ne porte que sur les lanes SUPERVISÉES. Un tour utilisateur
        # (masque tout-zéro) donne 0/ε = 0 ; le compter aurait dilué la moyenne
        # d'autant de zéros — mesuré, ça donnait 1.885 au lieu de ~2.96, soit
        # exactement le tiers de lanes non supervisées de ce mix. En B=1 ces segs
        # ne rentrent pas dans la moyenne (ce is None), donc les inclure ici
        # aurait rendu la diagnostique NON comparable — c'est-à-dire inutile.
        if x.size(0) > 1:
            w_lane = m2.sum(1)                              # [B] masse supervisée
            per = (ce_tok.view_as(m2) * m2).sum(1) / w_lane.clamp_min(1e-6)
            has = (w_lane > 0).to(per.dtype)
            ce_lane = ((per * has).sum() / has.sum().clamp_min(1.0)).detach()
    return loss, o["mem_bank"], ce_t, ce_lane


def _greedy(model, prefix, bank, max_new, stop_id, amp, use_cache=False,
            pool=None):
    """Greedy-decode max_new tokens after prefix from the CURRENT bank (reads
    only). Une seule ligne : l'éval décode conv par conv. Voir decode.generate
    pour la boucle (et pour le décodage batché, qui sert au RL).

    `pool` (opt-in `training.decode_graphs`) : décodage par CUDA graphs via un
    _GraphPool — ~14x sur le pas de décodage (bench 2026-07-28). Classe ULP
    (largeur pleine, cf. decode_graphs) : les DEUX bras d'un palier passent par
    le même chemin, la comparaison reste interne. Le bras ablaté (bank None)
    reste eager : GraphDecodeRunner exige une banque explicite, et ce bras est
    décodé UNE fois par palier de toute façon."""
    if pool is not None and bank is not None:
        out = pool.decode(prefix, bank, max_new, stop_id)
        if out is not None:
            return out
    gen, lens = generate(model, prefix, bank=bank, max_new=max_new,
                         stop_id=stop_id, amp=amp, use_cache=use_cache)
    return gen[:, :int(lens[0])]


class _GraphPool:
    """Pool de GraphDecodeRunner pour les décodages d'éval — un runner par
    forme de banque (B=1, slots 5..max_mem sans cascade), armement payé à la
    première rencontre de chaque forme puis rebind. Vie = UN appel
    d'evaluate_math (close() en sortie) : pas de VRAM résidente entre paliers,
    et les poids ayant bougé entre deux paliers, on ne rejoue jamais des
    graphs capturés sous d'anciens A/Bm sans rebind. Tout échec bascule
    l'éval en repli eager BRUYANT et définitif (piège WSL2 : jamais
    silencieux)."""

    def __init__(self, model):
        self.model = model
        self.pool = {}
        self.dead = False

    def decode(self, prefix, bank, max_new, stop_id):
        if self.dead:
            return None
        try:
            from .decode_graphs import GraphDecodeRunner
            key = (tuple(bank.shape), str(bank.dtype))
            r = self.pool.get(key)
            if r is None:
                r = self.pool[key] = GraphDecodeRunner(self.model, bank)
            else:
                r.rebind(bank)
            gen, lens = r.decode(prefix, max_new=max_new, stop_id=stop_id)
            return gen[:, :int(lens[0])]
        except Exception:
            import traceback
            print(f"decode_graphs KO — repli eager pour le reste du palier\n"
                  f"{traceback.format_exc()}", flush=True)
            self.dead = True
            self.close()
            return None

    def close(self):
        for r in self.pool.values():
            r.close()
        self.pool.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _age_bucket(age):
    for name, hi in (("<=4", 4), ("5-8", 8), ("9-16", 16)):
        if age <= hi:
            return name
    return ">16"


AGE_BUCKETS = ("<=4", "5-8", "9-16", ">16")


@torch.no_grad()
def evaluate_math(model, stream, tok, device, amp, n_conv, max_new=24,
                  use_cache=False, decode=True, graphs=False):
    """Chat eval (math_school | persona): canonical segments advance the bank
    (teacher-forced writes). Only the GRADED assistant turns (the last
    len(truths) — the answers to memory queries) are greedy-decoded TWICE —
    live bank vs ABLATED (None) — and graded by the generator's verifiers.
    grade_live - grade_ablated per kind = the working-memory efficacy figure
    (memory dsv6-grpo-m2-integre). Probe fine ajoutée (verdict run 5i : le
    grade exact-match est aveugle tant que le canal n'existe pas) : nll
    teacher-forcée du tour gradé live vs ablatée — Δnll>0 = la banque aide,
    sensible bien avant l'exact-match. Ventilée par âge (writes fait→réponse)
    quand le stream fournit info.ages. Bank-only (no cascade), like evaluate().
    Kinds sans truths (smalltalk) = contrôles : nll seule, pas de décodage.
    Returns {kind: {...}} + clé "_by_age" (à pop avant itération par kind).

    LE BRAS ABLATÉ EST DÉCODÉ UNE SEULE FOIS PAR APPEL. Son décodage part d'un
    préfixe constant (`a_open`) avec `init_mem=None`, en greedy, à poids gelés :
    il rend donc EXACTEMENT la même sortie aux ~27 tours gradés d'un palier
    (vérifié : 8 redécodages, une seule sortie distincte). Le recalculer coûtait
    36 % de l'éval chat — mesuré 2026-07-25, evaluate_math 958 → 611 s sur un
    palier de 1076 s, 4401 → 3150 forwards, dont 4242 passés dans les 54
    décodages à 218 ms le forward d'UN token : le décodage est du pur coût de
    lancement. Ce n'est pas une approximation : le bras ablaté est par
    construction « ce que le modèle répond sans aucune mémoire », une réponse et
    non une par question. Effet de bord bienvenu : il ne jitte plus d'un tour à
    l'autre sous l'effet des ULP du top-k MoE (mémoire dsv6-topk-dur-amplifie).

    `decode=False` saute les deux décodages et ne facture que la sonde Δnll (un
    forward ablaté par tour gradé). C'est le levier de cadence : le grade
    exact-match est aveugle tant que le canal n'existe pas, la sonde Δnll ne
    l'est pas — voir `chat.decode_every` côté trainer. Les clés grade/grade_abl
    sont alors normalisées par `n_dec` = 0 et le trainer ne les imprime pas."""
    from .math_school_data import A_OPEN, grade_conv
    from .persona_chat_data import grade_recall
    grade = getattr(stream, "grade_conv", grade_conv)   # persona ships its own
    model.eval()
    a_open = torch.tensor(tok(A_OPEN, add_special_tokens=False)["input_ids"],
                          dtype=torch.long, device=device).unsqueeze(0)
    stop_id = tok.convert_tokens_to_ids("<|im_end|>")
    # kind -> [nll_sum, nll_n, gl, ga, n, ans_nll_live, ans_nll_abl, n_ans, n_dec]
    agg = {}
    by_age = {}                       # bucket -> [n, dg_sum, dnll_sum, n_dg]
    abl_txt1 = None                   # le bras ablaté, décodé une fois (docstring)
    pool = _GraphPool(model) if (graphs and decode
                                 and next(model.parameters()).is_cuda) else None
    for _ in range(n_conv):
        conv = stream.next_conv()
        info = conv.get("info", {})
        truths = info.get("truths", []) or []
        ages = info.get("ages", []) or []
        a_idx = [i for i, s in enumerate(conv["segs"])
                 if s["role"] == "assistant"]
        graded = set(a_idx[-len(truths):]) if truths else set()
        bank = None
        live_txt, abl_txt = [], []
        nll_s, nll_n = 0.0, 0
        qi = 0
        a = agg.setdefault(conv["kind"], [0.0, 0, 0.0, 0.0, 0, 0.0, 0.0, 0, 0])
        for i, s in enumerate(conv["segs"]):
            x = s["input_ids"].to(device)
            lmask = s["loss_mask"].to(device)
            if i in graded and decode:
                if abl_txt1 is None:              # constant sur tout l'appel
                    abl_txt1 = tok.decode(_greedy(
                        model, a_open, None, max_new, stop_id, amp,
                        use_cache)[0].tolist())
                live_txt.append(tok.decode(_greedy(
                    model, a_open, bank, max_new, stop_id, amp, use_cache,
                    pool=pool)[0].tolist()))
                abl_txt.append(abl_txt1)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                o = model(x, init_mem=bank)
            m = lmask[:, 1:].reshape(-1)
            if float(m.sum()) > 0:
                lg = o["logits"].float()
                ce = F.cross_entropy(lg[:, :-1].reshape(-1, lg.size(-1)),
                                     x[:, 1:].reshape(-1), reduction="none")
                nll = float((ce * m).sum() / m.sum())
                nll_s += nll; nll_n += 1
                if i in graded:
                    with torch.autocast("cuda", dtype=torch.bfloat16,
                                        enabled=amp):
                        oa = model(x, init_mem=None)
                    lga = oa["logits"].float()
                    cea = F.cross_entropy(
                        lga[:, :-1].reshape(-1, lga.size(-1)),
                        x[:, 1:].reshape(-1), reduction="none")
                    nll_a = float((cea * m).sum() / m.sum())
                    a[5] += nll; a[6] += nll_a; a[7] += 1
                    if qi < len(ages):
                        b = by_age.setdefault(_age_bucket(ages[qi]),
                                              [0, 0.0, 0.0, 0])
                        b[0] += 1
                        b[2] += nll_a - nll
                        if decode and qi < len(truths):
                            b[1] += (grade_recall([live_txt[-1]], [truths[qi]])
                                     - grade_recall([abl_txt[-1]], [truths[qi]]))
                            b[3] += 1
                    qi += 1
            bank = o["mem_bank"]
        a[0] += nll_s; a[1] += nll_n
        if decode:
            a[2] += grade(conv, live_txt)
            a[3] += grade(conv, abl_txt)
            a[8] += 1
        a[4] += 1
    if pool is not None:
        pool.close()                  # pas de VRAM graphs résidente entre paliers
    model.train()
    out = {k: {"nll": v[0] / max(v[1], 1), "grade": v[2] / max(v[8], 1),
               "grade_abl": v[3] / max(v[8], 1), "n": v[4], "n_dec": v[8],
               "ans_nll": v[5] / max(v[7], 1),
               "ans_nll_abl": v[6] / max(v[7], 1), "n_ans": v[7]}
           for k, v in agg.items()}
    out["_by_age"] = {k: {"n": v[0], "dgrade": v[1] / max(v[3], 1),
                          "dnll": v[2] / v[0], "n_dg": v[3]}
                      for k, v in by_age.items()}
    return out


@torch.no_grad()
def evaluate(model, stream, device, think_id, blank_id, defer_len, n_conv, balw, amp,
             delta=None):
    model.eval()
    mdt = next(model.parameters()).dtype
    ic_loss = ic_n = 0.0
    d_car = d_res = d_car0 = d_res0 = dn = 0.0
    cont = cont0 = 0.0
    # GAP by conversation depth: hop-1 (first pair, i==0) vs deep (i>=4)
    c1 = r1 = n1 = 0.0
    cd = rd = nd_deep = 0.0
    for _ in range(n_conv):
        segs = stream.next_conv(); bank = None
        dstate = delta.init_state(1, device) if delta is not None else None
        for i, s in enumerate(segs):
            x = s["input_ids"].to(device)
            bank_in = bank                                # carried bank BEFORE this chunk's write
            xt = _append(x, think_id)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                o = model(xt, init_mem=bank)
            bank = o["mem_bank"]
            if delta is not None:                         # B4: carry = delta state
                dstate = delta.update(dstate, model.embed.weight[x])
                bank = delta.to_bank(dstate, mdt)
            lg = o["logits"].float()
            ic_loss += float(F.cross_entropy(lg[:, :-1].reshape(-1, lg.size(-1)),
                                             xt[:, 1:].reshape(-1))); ic_n += 1
            if i < len(segs) - 1:
                nxt = segs[i + 1]["input_ids"][:, :defer_len].to(device)   # [B, M]
                dl = nxt.size(1)                       # ragged: remainder chunk may be < defer_len
                # turn-0 CEILING: predict the SAME next-chunk tokens with chunk N in-window
                # (teacher-forced continuation). cont vs defer_car = cost of routing the
                # info through the bank instead of attention (user's turn0-vs-turn1 diff).
                ctx = torch.cat([x, nxt[:, :dl - 1]], dim=1)               # [B, L+M-1]
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                    oc = model(ctx, init_mem=bank_in)
                pc = oc["logits"].float()[:, x.size(1) - 1: x.size(1) - 1 + dl]  # [B,M,V]
                cont += float(F.cross_entropy(pc.reshape(-1, pc.size(-1)), nxt.reshape(-1)))
                cont0 += float(F.cross_entropy(pc[:, 0], nxt[:, 0]))
                # turn-1 DEFERRED: same targets from the bank ALONE (carried vs reset ablation)
                di = _fill(x, blank_id, dl)
                lall_m = {}
                for mode in ("car", "res"):
                    mem = bank if mode == "car" else None
                    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                        od = model(di, init_mem=mem)
                    lg = od["logits"].float()                              # [B, M, V]
                    lall = F.cross_entropy(lg.reshape(-1, lg.size(-1)), nxt.reshape(-1))
                    l0 = F.cross_entropy(lg[:, 0], nxt[:, 0])              # pos0 = pure bank
                    lall_m[mode] = float(lall)
                    if mode == "car": d_car += float(lall); d_car0 += float(l0)
                    else:             d_res += float(lall); d_res0 += float(l0)
                dn += 1
                lc, lr = lall_m["car"], lall_m["res"]         # by conversation depth
                if i == 0:  c1 += lc; r1 += lr; n1 += 1
                if i >= 4:  cd += lc; rd += lr; nd_deep += 1
    model.train()
    dnc = max(dn, 1)
    return {"ic_ppl": math.exp(ic_loss / max(ic_n, 1)),
            "defer_car": d_car / dnc, "defer_res": d_res / dnc,
            "defer_gap": (d_res - d_car) / dnc,
            "defer_car0": d_car0 / dnc, "defer_res0": d_res0 / dnc,
            "defer_gap0": (d_res0 - d_car0) / dnc,
            "cont": cont / dnc, "cont0": cont0 / dnc,                       # turn-0 ceiling
            "headroom": (d_car - cont) / dnc,                              # bank-only vs full-context
            "headroom0": (d_car0 - cont0) / dnc,
            "gap_hop1": (r1 - c1) / max(n1, 1),                            # GAP at depth-1 (first pair)
            "gap_deep": (rd - cd) / max(nd_deep, 1),                       # GAP at depth>=4
            "n_deep": nd_deep}


@torch.no_grad()
def evaluate_by_depth(model, stream, device, think_id, blank_id, defer_len,
                      depths, n_per, amp, delta=None):
    """GAP as a function of conversation DEPTH d = #chunks written into the bank
    before the deferred prediction. For each d: write d chunks (carry the bank),
    then predict the (d+1)-th chunk's opening from the bank ALONE (carried) vs reset
    (init_mem=None). Depth is CONTROLLED via conv_at_depth (not sampled), so buckets
    are populated with equal n — the reliable 'does memory hold as the conversation
    deepens?' curve. Returns {d: {'gap': .., 'car': .., 'res': .., 'n': ..}}."""
    model.eval()
    out = {}
    for d in depths:
        gv = cv = rv = 0.0; n = 0
        for _ in range(n_per):
            segs = stream.conv_at_depth(d + 1)          # d writes + 1 target
            if segs is None:
                break
            bank = None
            dstate = delta.init_state(1, device) if delta is not None else None
            for j in range(d):
                x = segs[j]["input_ids"].to(device)
                xt = _append(x, think_id)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                    o = model(xt, init_mem=bank)
                bank = o["mem_bank"]
                if delta is not None:
                    dstate = delta.update(dstate, model.embed.weight[x])
                    bank = delta.to_bank(dstate, next(model.parameters()).dtype)
            nxt = segs[d]["input_ids"][:, :defer_len].to(device)
            dl = nxt.size(1)
            di = _fill(nxt, blank_id, dl)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                oc = model(di, init_mem=bank)
                orr = model(di, init_mem=None)
            V = oc["logits"].size(-1)
            car = float(F.cross_entropy(oc["logits"].float().reshape(-1, V), nxt.reshape(-1)))
            res = float(F.cross_entropy(orr["logits"].float().reshape(-1, V), nxt.reshape(-1)))
            gv += res - car; cv += car; rv += res; n += 1
        nn = max(n, 1)
        out[d] = {"gap": gv / nn, "car": cv / nn, "res": rv / nn, "n": n}
    model.train()
    return out


def dry_run(cfg_path: str) -> None:
    """`--check` : tout ce qui peut être su d'une config SANS louer de GPU.

    Valide le schéma, construit le modèle sur CPU, puis imprime ce qu'on veut
    relire une dernière fois avant un bring-up : les chemins APRÈS expansion de
    ${TB_ROOT} (un cache mal pointé, c'est 25 min de re-tokenisation facturées),
    la table de schedule, et le mix de données. N'écrit rien, ne construit aucun
    stream, ne touche pas au réseau au-delà du tokenizer.
    """
    raw = load_yaml(cfg_path)
    check_cfg(raw, "code_defer_native")
    t, d = raw["training"], raw["data"]
    print(f"config: {cfg_path}  →  schéma OK")

    tok = AutoTokenizer.from_pretrained(raw["tokenizer"])
    add = [x for x in ("<think>", "<blank>") if x not in tok.get_vocab()]
    if add:
        tok.add_special_tokens({"additional_special_tokens": add})
    mcfg = dict(raw["model"]); mcfg["vocab_size"] = len(tok)
    cfg = ThoughtBankConfig(**mcfg)
    model = ThoughtBankLM(cfg)
    print(f"modèle: {model.num_params():,} params | d_model {cfg.d_model} "
          f"n_layers {cfg.n_layers} n_experts {cfg.n_experts} "
          f"mem_dim {cfg.mem_dim} max_mem {cfg.max_mem} | vocab {cfg.vocab_size}")

    steps = int(t["steps"]); warmup = int(t.get("warmup_steps", 100))
    tf = raw.get("teacher") or {}
    tf_a0, tf_a1 = (int(v) for v in tf.get("anneal", [200, 1000]))
    for line in describe_sched(
            steps=steps, warmup=warmup, wsd=bool(t.get("wsd_decay", True)),
            decay_start=int(t.get("wsd_decay_start", int(steps * 0.66))),
            shape=str(t.get("wsd_decay_shape", "linear")),
            floor=float(t.get("wsd_floor", 0.0)),
            stair_n=int(t.get("wsd_stair_n", STAIR_N)),
            stair_end=float(t.get("wsd_stair_end", STAIR_END)),
            muon_lr=float(t.get("muon_lr", 3e-3)), lr=float(t.get("lr", 3e-4)),
            tf_on=bool(tf.get("enabled", False)), tf_a0=tf_a0, tf_a1=tf_a1):
        print(line)

    # Les chemins sont RAPPORTÉS en entier avant de conclure : un bring-up veut
    # voir tous les problèmes d'un coup, pas les découvrir un par un.
    problems: list[str] = []
    print("chemins résolus (${TB_ROOT} expansé) :")
    for label, path in (("save_dir", t["save_dir"]),
                        ("cache_dir", d.get("cache_dir", "data_cache")),
                        ("metrics_file", t.get("metrics_file")),
                        ("init_from", t.get("init_from")),
                        ("bank_init", t.get("bank_init"))):
        if path is None:
            continue
        if "${" in str(path):
            problems.append(f"{label}: variable non résolue dans {path!r} — poser "
                            f"la variable d'environnement, sinon le run écrit à "
                            f"un chemin littéral")
            print(f"  {label:<13} {path}  [VARIABLE NON RÉSOLUE]")
            continue
        exists = os.path.exists(path)
        required = label in ("init_from", "bank_init")
        note = "existe" if exists else "ABSENT" if required else "sera créé"
        if required and not exists:
            problems.append(f"{label}: {path} introuvable — le run échouerait "
                            f"après avoir alloué le GPU "
                            f"(TB_ROOT={os.environ.get('TB_ROOT', '<non posé, = .>')})")
        extra = ""
        if label == "cache_dir" and exists:
            n = len([p for p in os.listdir(path) if p.startswith("chunks_")])
            extra = f" — {n} fichier(s) de chunks en cache" if n else \
                    " — VIDE : le run tokenisera (long)"
        print(f"  {label:<13} {path}  [{note}]{extra}")

    srcs = d.get("sources") or [{"dataset": d.get("dataset", "?"), "weight": 1.0}]
    print(f"données: seq_len {d['seq_len']} | chunks/conv {d['chunks_per_conv']} | "
          f"batch {d['batch_size']} | {len(srcs)} source(s)")
    for s in srcs:
        print(f"  - {s.get('dataset', '?'):<45} poids {s.get('weight', 1.0)}")
    chat = raw.get("chat") or {}
    if chat:
        print(f"chat: stream {chat.get('stream', 'math_school')} | "
              f"p_chat {chat.get('p_chat', 0.5)} | poids {chat.get('weight', 1.0)} "
              f"| max_new {chat.get('max_new', 24)}")
    depth = int(t.get("cascade_depth", 0) or 0)
    if depth > 0:
        cmap = ([int(v) for v in t.get("cascade_map")] if t.get("cascade_map")
                else default_layer_map(cfg.n_layers, depth))
        print(f"banque: max_mem {cfg.max_mem} | cascade_depth {depth} | "
              f"map effective {cmap}"
              + ("" if t.get("cascade_map") else "  (défaut, pas dans la config)"))
    else:
        print(f"banque: max_mem {cfg.max_mem} | pas de cascade")

    if problems:
        print(f"\n--check: {len(problems)} problème(s) — le run ne partirait pas "
              f"proprement :")
        for p in problems:
            print(f"  ✗ {p}")
        raise SystemExit(1)
    print("\n--check: OK — rien n'a été écrit.")


def main(cfg_path: str, resume: bool = False) -> None:
    raw = load_yaml(cfg_path)
    check_cfg(raw, "code_defer_native")   # une clé mal orthographiée s'arrête ICI,
    #                                       pas après N heures de GPU au mauvais schedule
    t = raw["training"]; d = raw["data"]

    # DDP (opt-in via torchrun), tf32, compile, grad-checkpoint : voir runtime.py
    # — le CONTEXTE d'exécution, séparé de l'objectif d'entraînement. Tous ces
    # leviers restent opt-in et OFF par défaut : une config existante produit
    # exactement le même run.
    ddp = init_ddp()
    ddp_world, ddp_rank, device = ddp.world, ddp.rank, ddp.device
    amp = bool(t.get("amp", False))            # native MoE/sinkhorn: fp32 by default
    enable_tf32(bool(t.get("tf32", False)))

    # `training.seed` pilotait les streams de données (random de Python) mais PAS
    # torch : l'init du modèle était tirée de l'entropie de l'OS, donc deux runs
    # de la MÊME config partaient de poids différents et aucun résultat n'était
    # reproductible depuis sa config seule (mesuré 2026-07-25 : deux
    # constructions du 97M donnent des poids sans rapport). Un dépôt public dont
    # le livrable est « un lecteur reproduit des claims » ne peut pas se le
    # permettre. Toutes les rangs partagent la graine : l'init est identique
    # partout, et le broadcast rank0 plus bas devient une ceinture-bretelles.
    torch.manual_seed(int(t.get("seed", 0)))

    tok = AutoTokenizer.from_pretrained(raw["tokenizer"])
    add = [x for x in ("<think>", "<blank>") if x not in tok.get_vocab()]
    if add:
        tok.add_special_tokens({"additional_special_tokens": add})
    think_id = tok.convert_tokens_to_ids("<think>")
    blank_id = tok.convert_tokens_to_ids("<blank>")

    mcfg = dict(raw["model"]); mcfg["vocab_size"] = len(tok)
    cfg = ThoughtBankConfig(**mcfg)
    model = ThoughtBankLM(cfg).to(device)
    print(f"native ThoughtBankLM {model.num_params():,} params | d_model {cfg.d_model} "
          f"n_layers {cfg.n_layers} n_experts {cfg.n_experts} mem_dim {cfg.mem_dim} "
          f"max_mem {cfg.max_mem} | <think>={think_id} <blank>={blank_id} vocab {cfg.vocab_size}",
          flush=True)

    # init_from: CONTINUED pretraining — model weights from a finished run's
    # checkpoint, fresh optimizer/schedule/data (unlike --resume, which restores
    # the full training state of THIS run). Used by the var-chunk phase (v2c).
    init_from = t.get("init_from")
    if init_from:
        ck0 = torch.load(init_from, map_location="cpu")
        model.load_state_dict(ck0["model"])
        print(f"init_from: model weights <- {init_from} (step {ck0.get('step', '?')})",
              flush=True)

    # `base` is the eager module — the source of truth for state_dict / named params
    # / optimizer grouping (torch.compile wraps in OptimizedModule and prefixes keys
    # with `_orig_mod.`, which would break checkpoint format and the name-based Muon
    # groups below). `model` is what we CALL forward on; when compiled it shares the
    # same Parameter tensors as `base`, so the optimizer built from `base` still
    # drives it. Opt-in; graph breaks (MoE/sinkhorn/einsum) measured in the sweep.
    base = model
    if bool(t.get("compile", False)):
        model = compile_model(model, cache_dir=t.get("compile_cache_dir"))

    grad_ckpt = bool(t.get("grad_checkpoint", False))
    _fwd = build_fwd(model, grad_checkpoint=grad_ckpt,
                     save_topk=bool(t.get("gc_save_topk", True)))

    # decode_cache : cache KV pour les décodages de l'ÉVAL (le seul endroit où
    # ce trainer décode). N'accélère PAS l'entraînement — un forward
    # d'entraînement voit la séquence entière d'un coup, il n'y a rien à
    # cacher. Mais l'éval pèse ~21% du mur sur ce run et elle est dominée par
    # ses décodages : ×1,6 mesuré sur le 350M au step 200. Voir decode.generate
    # pour ce que ça change (rien en arithmétique exacte, un token ici ou là
    # via le top-k du routage — soit le bruit que ce GPU produit déjà d'un
    # appel à l'autre, mesuré 5/6 séquences reproductibles SANS cache).
    decode_cache = bool(t.get("decode_cache", False))
    if decode_cache:
        print("decode_cache: ON (cache KV aux décodages d'éval)", flush=True)
    # decode_graphs (opt-in) : les décodages du bras LIVE d'evaluate_math
    # passent par CUDA graphs (_GraphPool) — exige les 3 flags decode_* dans
    # le bloc model. Classe ULP (cf. decode_graphs.py) : cohérent PAR RUN,
    # ne pas comparer un grade graphs à un grade cache d'un autre run.
    decode_graphs = bool(t.get("decode_graphs", False))
    if decode_graphs:
        missing = [f for f in ("decode_fuse", "decode_dense_moe",
                               "decode_static_cache")
                   if not bool(getattr(cfg, f, False))]
        assert not missing, \
            f"training.decode_graphs exige les flags model {missing}"
        print("decode_graphs: ON (CUDA graphs aux décodages d'éval live)",
              flush=True)

    # B4 (backlog 2026-07-13) : canal DeltaNet inter-tours À LA PLACE du carry
    # de banque — modèle strictement inchangé, seul le canal inter-chunks change
    # (o["mem_bank"] ignoré, l'état delta est porté et présenté en pseudo-banque).
    # Config : delta_channel: {d_k: 64}. Voir delta_channel.py.
    dc_cfg = t.get("delta_channel")
    delta = None
    if dc_cfg:
        from .delta_channel import DeltaChannel
        _dk = int(dc_cfg.get("d_k", 64)) if isinstance(dc_cfg, dict) else 64
        delta = DeltaChannel(cfg.d_model, cfg.max_mem, cfg.mem_dim, d_k=_dk).to(device)
        print(f"delta channel ON: d_k {_dk} d_v {delta.d_v} "
              f"({sum(p.numel() for p in delta.parameters()):,} params) — "
              f"carry inter-chunks = état delta, o['mem_bank'] ignoré", flush=True)

    # DDP: le seed commun ci-dessus suffit en principe à donner la même init
    # partout ; le broadcast reste, gratuit, comme garantie dure que la
    # all-reduce manuelle garde tous les rangs bit-identiques dès le step 1
    # (une différence d'ordre de construction suffirait à décaler un rang).
    # (tf_proj is Generator-seeded: already identical everywhere.)
    if ddp_world > 1:
        with torch.no_grad():
            for p in base.parameters():
                torch.distributed.broadcast(p.data, src=0)
            if delta is not None:
                for p in delta.parameters():
                    torch.distributed.broadcast(p.data, src=0)

    # teacher: distill the last bank slot toward a fixed random projection of the
    # mean-pooled chunk gist (a target the write CAN produce), β anneals 1->0.
    tf_cfg = raw.get("teacher", {}) or {}
    tf_on = bool(tf_cfg.get("enabled", False))
    tf_dw = float(tf_cfg.get("distill_weight", 2.0))
    tf_a0, tf_a1 = (int(v) for v in tf_cfg.get("anneal", [200, 1000]))
    # target: 'chunk' (défaut) = proj du gist moyen du chunk (hash de contenu) ;
    # 'value' = proj de l'embedding des tokens VALEUR (val_mask) = code propre
    # DISCRIMINANT par valeur (recette 47M nn.Embedding(rule_id) : 0.03→0.99).
    # En mode 'value' le teacher ne tire QUE les segs porteurs de valeur ; les
    # autres writes (filler, question, ack) restent natifs.
    # 'surprisal' = généralisation label-free de 'value' : pooling pondéré par
    # la nll^alpha d'un LM de référence gelé (surp_w posé par le générateur,
    # clé gen.surprisal_ref) — les tokens imprévisibles (l'information) dominent
    # la cible, les templates pèsent ~0. Marche sur tout corpus, tous les segs.
    # (La pondération EST du SIF quand `gen.surprisal_mode: sif` : 'surprisal'
    #  nomme le pooling, pas la formule des poids.)
    # 'value_sif' = 'value' LÀ OÙ un val_mask existe, repli 'surprisal' partout
    # ailleurs. Nécessaire dès que la cible discriminante ne concerne qu'une
    # partie du mix : en 'value' pur le teacher ne tire QUE les segs porteurs, ce
    # qui priverait ici 75 % du mix (sota + exec) de teacher. Motivation : le run
    # `fromsif_exec_tiled` laisse toolcall à grade 0.00 aux quatre paliers gradés
    # pendant que la nll tombe de 26 % — la sélection ne s'ouvre pas toute seule,
    # et à 47M c'est un teacher discriminant qui l'avait ouverte (0.03 → 0.99).
    # 'value_table' = teacher APPRIS à codes COMPOSÉS (recette 47M complète,
    # train.py:2126-2178 — la pièce qui manquait au run recall_stair, verdict
    # 07-29 : canal Δnll ouvert, copie argmax fermée). Trois nn.Embedding
    # (slot interrogeable / attribut chien-chat-sœur-frère / valeur) sommées
    # puis RMS-normées = le code d'un fait porte sa LIAISON, pas la valeur
    # seule (contrainte user 07-29 : « le chien peut être un chat, le nom peut
    # être la race »). Le blend β injecte ce code NON détaché dans la banque
    # que les segs suivants lisent : la CE de la tâche traverse le read jusqu'aux
    # tables — le codebook s'organise pour être ce que le read sait sélectionner
    # ET décompresser. La distill (write→code DÉTACHÉ) reste un pull annealé.
    tf_target = str(tf_cfg.get("target", "chunk"))
    assert tf_target in ("chunk", "value", "surprisal", "value_sif",
                         "value_table"), tf_target
    tf_proj = None
    tf_tables = tf_topt = None
    if tf_on:
        g = torch.Generator(device="cpu").manual_seed(1789)
        tf_proj = (torch.randn(cfg.d_model, cfg.mem_dim, generator=g) / cfg.d_model ** 0.5).to(device)
        if tf_target == "value_table":
            assert not (torch.distributed.is_available()
                        and torch.distributed.is_initialized()), \
                "value_table : tables locales, pas de réduction DDP implémentée"
            from .persona_chat_data import fact_id_maps
            _sm, _vm_ids, _am = fact_id_maps()
            tf_tables = nn.ModuleDict({
                "slot": nn.Embedding(len(_sm) + 1, cfg.mem_dim, padding_idx=0),
                "attr": nn.Embedding(len(_am) + 1, cfg.mem_dim, padding_idx=0),
                "val": nn.Embedding(len(_vm_ids) + 1, cfg.mem_dim,
                                    padding_idx=0),
            }).to(device).float()
            # Init SIF du codebook (2026-07-30) : le run table v1 a montré le
            # conflit de géométries — V aléatoire vs write ancré dans le pooling
            # SIF de la phase 1, équilibre distill à ~0.40 et read aveugle aux
            # writes nus (Δnll held-out ≈ 0 à β=0). Réconciliation : V[val]
            # démarre à RMSnorm(pool(embed(tokens du val)) @ tf_proj) — la
            # géométrie que le write produit NATIVEMENT (dist ~0.25 au stair) —
            # et K/A deviennent de petits offsets de liaison (×0.2). Le code
            # composé ≈ « du SIF organisé » : la distill ne serre plus que la
            # liaison, le read organise à partir d'un point atteignable.
            with torch.no_grad():
                tf_tables["slot"].weight.mul_(0.2)
                tf_tables["attr"].weight.mul_(0.2)
                _emb = model.embed.weight.detach().float()
                for _v, _i in _vm_ids.items():
                    _ids = tok(" " + _v, add_special_tokens=False)["input_ids"]
                    if not _ids:
                        continue
                    _c = _emb[torch.tensor(_ids, device=device)].mean(0) @ tf_proj.float()
                    _c = _c / _c.pow(2).mean().clamp_min(1e-12).sqrt()
                    tf_tables["val"].weight[_i] = _c
            print(f"value_table init SIF: V <- pool(embed)@proj "
                  f"({len(_vm_ids)} vals, ' '+val), K/A x0.2", flush=True)
            tf_topt = torch.optim.AdamW(tf_tables.parameters(),
                                        lr=float(tf_cfg.get("table_lr", 1e-3)))
        _tdesc = {"value": "proj embed valeur (discriminant)",
                  "surprisal": "proj pooling pondéré nll ref (label-free)",
                  "value_sif": "valeur si val_mask, sinon pooling pondéré",
                  "value_table": "codes APPRIS composés slot+attr+valeur "
                                 "(liaison), organisés via le read",
                  "chunk": "proj gist chunk"}[tf_target]
        print(f"teacher ON: distill_w {tf_dw}, anneal [{tf_a0},{tf_a1}], "
              f"target={tf_target} ({_tdesc})"
              + (f" | tables {len(_sm)}+{len(_am)}+{len(_vm_ids)} ids, "
                 f"lr {float(tf_cfg.get('table_lr', 1e-3)):g}"
                 if tf_tables is not None else ""), flush=True)

    def _beta(s):
        return beta_at(s, enabled=tf_on, anneal_start=tf_a0, anneal_end=tf_a1)

    L, K = int(d["seq_len"]), int(d["chunks_per_conv"])
    defer_len = int(d.get("defer_len", 16))
    sd = dict(seq_len=L, chunks_per_conv=K, batch=int(d["batch_size"]),
              n_files=int(d.get("n_files", 800)),
              dataset=d.get("dataset", "codeparrot/codeparrot-clean-valid"),
              data_dir=d.get("data_dir", ""), stream_cap=int(d.get("stream_cap", 60000)),
              cache_dir=d.get("cache_dir", "data_cache"),
              content_key=d.get("content_key", "content"),
              config_name=d.get("config_name", ""),
              min_chunks=int(d.get("min_chunks", 1)),
              stream_skip=int(d.get("stream_skip", 0)),
              sources=d.get("sources"),
              var_chunk=d.get("var_chunk"),
              surprisal_mode=d.get("surprisal_mode", "none"),
              sif_a=float(d.get("sif_a", 1e-4)),
              pack_convs=bool(d.get("pack_convs", False)),
              pack_same_source=bool(d.get("pack_same_source", False)),
              seed=int(t.get("seed", 0)))
    # DDP: per-rank seed offset => each rank samples different convs (random
    # sampling with per-rank RNG — no distributed sampler needed). Rank0 builds
    # the tokenized cache alone first (concurrent misses race on the .tmp
    # rename), the barrier releases the others onto a guaranteed cache hit.
    train_seed = sd["seed"] + 9973 * ddp_rank
    # depth_sync (opt-in): rank-invariant anchor/m rng in next_conv_batch, so all
    # ranks run the same conv depth per step (a DDP step lasts as long as the
    # deepest rank — independent draws cost ~2.6x the mean step time).
    depth_sync = bool(d.get("depth_sync", False))
    if ddp_world > 1 and ddp_rank != 0:
        torch.distributed.barrier()
    train_stream = CodeChunkStream(tok, split="train", **{**sd, "seed": train_seed},
                                   depth_sync_seed=sd["seed"] if depth_sync else None)
    eval_stream  = CodeChunkStream(tok, split="held",
                                   **{**sd, "batch": 1, "surprisal_mode": "none"})  # eval = batch=1 paths
    if ddp_world > 1 and ddp_rank == 0:
        torch.distributed.barrier()               # cache built — release the other ranks
    print(f"corpus: train {train_stream.n_chunk} chunks / held {eval_stream.n_chunk} | "
          f"seq_len {L}  K {K}  defer_len {defer_len}", flush=True)
    # per-domain eval views: on a weighted mix, GAP/depth are reported PER SOURCE
    # (a blended number would hide "the bank works on code but not on web text").
    eval_views = ([(nm, eval_stream.source_stream(i))
                   for i, nm in enumerate(eval_stream.src_names)]
                  if len(eval_stream.src_files) > 1 else [("", eval_stream)])
    # eval_sources (opt-in) : restreint l'éval per-source à des ANCRES — sur un
    # mix 14 sources l'éval complète re-teste tout le corpus à chaque palier
    # (14x8 convs + depth), ce qui domine le wall-clock d'un SFT court. None
    # (défaut) = toutes les sources (comportement historique).
    ev_srcs = t.get("eval_sources")
    if ev_srcs:
        eval_views = [(nm, es) for nm, es in eval_views if nm in ev_srcs]

    # ── chat mode (opt-in `chat:` block — phase 2 SFT, marche 2) ─────────────
    # Chat-templated convs (math school) mixed into the conv stream at p_chat:
    # a chat conv occupies one grad-accum slot and RIDES the same life carry
    # (no_reset/cascade) as code convs — cross-domain interleaving for free.
    # Segments carry loss_mask (CE on assistant answers only); defer/addr/
    # reach never fire on them (no defer_tgt). Absent block => bit-identical.
    chat_cfg = raw.get("chat") or {}
    chat_stream = chat_eval = None
    chat_B = 1
    p_chat = float(chat_cfg.get("p_chat", 0.5))
    chat_w = float(chat_cfg.get("weight", 1.0))
    chat_eval_convs = int(chat_cfg.get("eval_convs", 24))
    chat_max_new = int(chat_cfg.get("max_new", 24))
    # decode_every : le grade exact-match ne se paie qu'un palier sur N. Le
    # décodage greedy EST l'éval (mesuré 2026-07-25 sur ce mix : 958 s des
    # 1076 s d'un palier, 4242 forwards d'UN token à 218 ms pièce), alors que
    # la sonde Δnll — la seule sensible tant que le canal n'est pas ouvert —
    # coûte un forward par tour gradé. 1 (défaut) = comportement historique.
    chat_decode_every = int(chat_cfg.get("decode_every", 1))
    assert chat_decode_every >= 1, f"decode_every >= 1, vu {chat_decode_every}"
    if chat_cfg:
        from .streams import chat_stream_class
        sname = chat_cfg.get("stream", "math_school")
        _ChatStream = chat_stream_class(sname)
        gen_kw = dict(chat_cfg.get("gen", {}) or {})
        chat_stream = _ChatStream(tok, seed=train_seed + 1, **gen_kw)
        # chat.eval_gen (opt-in) : bloc gen COMPLET de remplacement pour le
        # stream d'ÉVAL — pas un merge (les gen sont imbriqués). Cas d'usage :
        # pool_split train/eval du stream persona — l'éval en palier porte
        # alors sur des valeurs JAMAIS vues du train (rappel, pas recognition).
        eval_kw = dict(chat_cfg.get("eval_gen") or {}) or gen_kw
        chat_eval = _ChatStream(tok, seed=1234, **eval_kw)
        print(f"chat mode ON: {sname} p_chat {p_chat} weight {chat_w} "
              f"eval_convs {chat_eval_convs} (masked-CE SFT convs in the "
              f"life carry)", flush=True)
        # chat_batch : B lanes en lockstep par tour (chat_batch.py). Le batch ne
        # vient PAS de data.batch_size — celui-là reste le batch du stream
        # FICHIERS, et les asserts ragged qui l'entourent (no_reset, interleave,
        # cascade) restent donc satisfaits tels quels. Les segs arrivent en
        # [B, L] right-padés avec un pad_mask ; l'éval reste conv par conv
        # (chat_batch délègue next_conv).
        chat_B = int(getattr(chat_stream, "B", 1))
        if chat_B > 1:
            assert delta is None, \
                "chat batché + delta_channel : delta.init_state prend train_stream.B " \
                "(le batch FICHIERS), il porterait un état de la mauvaise largeur"
            print(f"chat batché: {chat_B} lanes x {chat_stream.slots} tours par "
                  f"tirage (ops/step / {chat_B} a FLOPs quasi constants ; "
                  f"{chat_B} vies de banque parallèles)", flush=True)

    # single native optimizer: Muon (2-D weights) + bundled AdamW (embed/norm/1-D)
    lr = float(t.get("lr", 3e-4)); muon_lr = float(t.get("muon_lr", 3e-3))
    wd = float(t.get("weight_decay", 0.01)); balw = float(cfg.balance_loss_weight)
    muon_p, adam_p = _split_muon_params(base)
    if delta is not None:
        # B4 : les ~50k params du canal delta vont dans le bundle AdamW (module
        # neuf, pas de piège √cols Muon à gérer) ; état optimiseur sauvé avec.
        adam_p = adam_p + list(delta.parameters())
    # Per-module lr_scale: legacy Muon scaling (update * √cols) ties the per-matrix
    # update RMS to SHAPE (≈ √(cols/rows)), so changing mem_dim silently rescales the
    # effective lr of every mem_dim-shaped matrix: 64→512 made the read hypernet
    # (fw_A/fw_B, [.., mem_dim]) ~2.8x FASTER and the write head (thought_head/
    # write_gate, [mem_dim, ..]) ~2.8x SLOWER at fixed muon_lr — the v2 GAP collapse.
    # Restore the mem_dim=64-validated per-module effective RMS via group lr scales.
    ref_dim = float(t.get("muon_ref_mem_dim", 64))
    s_read  = (ref_dim / cfg.mem_dim) ** 0.5          # cols = mem_dim grew → scale down
    s_write = (cfg.mem_dim / ref_dim) ** 0.5          # rows = mem_dim grew → scale up
    names = {id(p): n for n, p in base.named_parameters()}
    g_read  = [p for p in muon_p if ("fw_A" in names[id(p)] or "fw_B" in names[id(p)])]
    g_write = [p for p in muon_p if ("thought_head" in names[id(p)] or "write_gate" in names[id(p)])]
    ids = {id(p) for p in g_read} | {id(p) for p in g_write}
    g_rest  = [p for p in muon_p if id(p) not in ids]
    groups = [{"params": g_rest},
              {"params": g_read,  "lr_scale": s_read},
              {"params": g_write, "lr_scale": s_write}]
    opt = Muon(groups, lr=muon_lr, momentum=0.95, nesterov=True, ns_steps=10, wd=wd,
               adam_params=adam_p, adam_lr=lr, adam_wd=wd,
               adam_fused=bool(t.get("adam_fused", False)))
    print(f"optimizer: Muon lr {muon_lr} ({sum(p.numel() for p in muon_p):,}) "
          f"+ AdamW lr {lr} ({sum(p.numel() for p in adam_p):,}) | "
          f"lr_scale read {s_read:.3f} ({sum(p.numel() for p in g_read):,}) "
          f"write {s_write:.3f} ({sum(p.numel() for p in g_write):,}) "
          f"[ref mem_dim {ref_dim:.0f}]", flush=True)
    _G, _B = int(t.get("grad_accum", 1)), train_stream.B
    print(f"grad_accum {_G} x batch {_B} (effective batch = {_G * _B} convs"
          f"{', batched full-chunk windows' if _B > 1 else ''}) "
          f"| K {K} (conv depth up to max_mem)", flush=True)

    steps = int(t["steps"]); warmup = int(t.get("warmup_steps", 100))
    grad_accum = int(t.get("grad_accum", 1))          # convs per optimizer step (effective batch)
    # no_reset_files N > 1: chain N consecutive files into one bank lifetime — the bank
    # is carried (detached) across file boundaries instead of reset, so every file after
    # the first STARTS with the previous file's gists in its slots (dirty-bank regime).
    # Boundary defer stays masked for free: batch=1 derives defer targets within-file only.
    no_reset_files = int(t.get("no_reset_files", 1))
    # no_reset_files == 0 : UNE vie infinie — la banque n'est JAMAIS reset, ni entre
    # convs ni entre steps d'optimiseur ; elle évolue en continu sur tout le run
    # (décision user 2026-07-20, run 5c). Non sauvegardée dans les ckpts : un resume
    # repart banque vide.
    nrf_never = (no_reset_files == 0)
    if nrf_never:
        assert int(d["batch_size"]) == 1, "no_reset_files=0 requires batch_size 1 (ragged mode)"
    if no_reset_files > 1:
        assert int(d["batch_size"]) == 1, "no_reset_files requires batch_size 1 (ragged mode)"
        assert grad_accum % no_reset_files == 0, "grad_accum must be a multiple of no_reset_files"
    # interleave_files F > 1 (idea G): each conv = F files' chunks randomly interleaved
    # in ONE bank lifetime (same total chunk budget as next_conv). Trains content-based
    # selection without the no-reset boundary confound (v2d probes 2026-07-11: no-reset
    # learned "off-topic last write => new file" and collapses on mid-file distractors).
    # scalar = fixed F; [lo, hi] = F sampled U[lo, hi] per conv (subject-count diversity)
    _ilv = t.get("interleave_files", 1)
    interleave_files = (tuple(int(v) for v in _ilv)
                        if isinstance(_ilv, (list, tuple)) else int(_ilv))
    ilv_on = (max(interleave_files) if isinstance(interleave_files, tuple)
              else interleave_files) > 1
    if ilv_on:
        assert int(d["batch_size"]) == 1, "interleave_files requires batch_size 1 (ragged mode)"
        # D+G (2026-07-12): no_reset_files>1 + interleave = the carry is INTERLEAVED
        # — bank lives across groups of mixed-thread convs. The carry/init logic is
        # sampler-independent, so the combination is free; boundary defers stay
        # within-file (per-seg defer_tgt). Probes must check whether the v2d
        # boundary heuristic partially returns (group boundaries correlate with
        # "all previous threads dead" until pages make old threads resumable).
    # G2 (2026-07-12): addressed defers — cue (file label 50% / raw chunk opening
    # 50%) + blanks toward a NON-last live stream; trains the content/label-
    # addressed read that the blank defer's recency convention never exercises.
    addr_prob = float(t.get("addr_prob", 0.0))
    addr_label = bool(t.get("addr_label", False))
    addr_max = int(t.get("addr_max", 2))    # cap/conv: each addr forward = a full
    #                                         read graph held until backward (8 GB!)
    if addr_prob > 0 or addr_label:
        assert ilv_on, "addr_prob/addr_label require interleave_files (multi-thread bank)"
    # B2 (backlog 2026-07-13) : resets ANNONCÉS — un marqueur tokenisé
    # <<RESET:SOON>> (pattern file_label_ids : texte, pas de token spécial,
    # vocab inchangé) est préfixé aux `reset_announce_chunks` derniers chunks
    # d'une VIE de banque avec prob `reset_announce` (0.5 = 50/50 annoncé/
    # surprise). On MESURE seulement (probe resetcue : la politique d'écriture
    # bouge-t-elle quand la mort est annoncée ?) — standing warning : aucune
    # perte/reward attachée à l'annonce.
    ra_prob = float(t.get("reset_announce", 0.0))
    ra_chunks = int(t.get("reset_announce_chunks", 3))
    ra_ids = (torch.tensor(tok("<<RESET:SOON>>")["input_ids"], dtype=torch.long)
              if ra_prob > 0 else None)
    # Cascade v3 (spec user 2026-07-12, débordement en 2 temps × fractale max_mem) :
    # cascade_depth = nombre de niveaux au-dessus de la banque vive (0 = off,
    # 1 = v3-lite page, 4 = complet). cascade_map[i] = niveau lu par la couche i
    # (0 = banque vive) ; défaut : les `depth` dernières couches lisent 1..depth.
    cascade_depth = int(t.get("cascade_depth", 0) or 0)
    cascade_map = None
    if cascade_depth > 0:
        _cmap = t.get("cascade_map")
        cascade_map = ([int(v) for v in _cmap] if _cmap else
                       default_layer_map(cfg.n_layers, cascade_depth))
        assert len(cascade_map) == cfg.n_layers and max(cascade_map) <= cascade_depth
        assert not bool(getattr(cfg, "mem_write_gate_merge", False)), \
            "cascade: gate_merge réordonne les slots, la capture d'éviction suppose FIFO pur"
        # ragged batch=1 (alignement conv historique) OU batché AVEC pack_convs :
        # à profondeur constante K les B vies sont en lockstep — l'éviction du
        # slot j tombe au même tour partout, CascadeMemory porte le batch tel
        # quel ([B,D] slots). Batché SANS pack refusé (m variable => vies
        # désalignées dans le batch).
        assert train_stream.B == 1 or getattr(train_stream, "pack", False), \
            "cascade batchée : exige pack_convs (vies alignées, K constant)"
        _seed_slots = int(getattr(cfg, "mem_seed_slots", cfg.max_mem))
    if delta is not None:
        assert cascade_depth == 0, "delta_channel remplace le canal — pas de cascade"
        assert not tf_on, "delta_channel: teacher incompatible (manipule les slots)"
    # OPTION 2 (verdict page 2026-07-13, FINDINGS d70b595) : reach-back
    # SUPERVISÉ. L'émergence est réfutée (2 seeds : ablater la page ne change
    # rien) mais la cible existe (early évincé −0.37..−0.72 vs reset via
    # résidus de superposition) et la page est gratuite quand non lue —
    # recette v2f : créer le mécanisme par SFT. Un POOL de cibles est porté à
    # travers la vie carried : chaque seg écrit y dépose (cue = ouverture du
    # chunk, label compris si addr_label ; tgt = son defer_tgt même-fichier) ;
    # dans les convs SUIVANTES, avec prob reach_prob, un defer adressé vise
    # une entrée dont les slots ont quitté la banque vive (âge >= max_mem
    # writes) — la seule route vers la cible est la page (ou les résidus).
    # STRATIFICATION par âge (réserve user 2026-07-13 : les cibles du bloc le
    # plus mergé pourraient ne pas apprendre) : perte loggée par strate
    #   s1 [M, 2M)  ~ page p0 fraîche      s2 [2M, 4M) ~ page mergée (p1)
    #   s3 [4M, ∞)  ~ au-delà de la page à depth 1 = DÉTRUIT (contrôle
    #                 négatif : ne DOIT pas s'améliorer si le read lit la page)
    # Même garde VRAM que G2 : chaque forward reach = un graphe de read complet
    # jusqu'au backward => cap reach_max par conv.
    reach_prob = float(t.get("reach_prob", 0.0))
    reach_max = int(t.get("reach_max", 2))
    reach_cue_len = int(t.get("reach_cue_len", 16))
    if reach_prob > 0:
        assert cascade_depth > 0, "reach_prob: il faut une page (cascade_depth >= 1)"
        assert int(t.get("no_reset_files", 1)) > 1, \
            "reach_prob: le pool vit dans le carry (no_reset_files > 1)"
    # ── Optimisations budget-compute (2026-07-23, tous OPT-IN, défauts =
    # comportement historique bit-identique). Contexte pod 10B : 8 boucles data
    # single-core à 99% (host-bound — B24→B32 était gratuit), m moyen ~3.9 vs
    # K=8 => coûts fixes par step (Muon + all-reduce + host) amortis sur moitié
    # moins de tokens que le pire cas dimensionnant la VRAM.
    #   prefetch        : thread producteur qui tire les convs batchées EN AVANCE
    #                     (queue depth 2, pin_memory) — le host tourne pendant le
    #                     compute GPU au lieu de le sérialiser. Chemin batché PUR.
    #                     NB resume : la rng du stream sauvée court <=2 convs en
    #                     avance du step exécuté (les dumps nan portent les segs
    #                     eux-mêmes, la repro d'incident n'en dépend pas).
    #   chunk_budget N  : accumulation DYNAMIQUE — on enchaîne des convs (banque
    #                     reset entre elles, distribution d'entraînement INTACTE)
    #                     jusqu'à N chunks par step avant l'opt.step => step time
    #                     uniforme au pire-cas VRAM, ~m̄/N fois moins d'opt-steps
    #                     par token (grads normalisés par le nb de convs, =
    #                     sémantique grad_accum). depth_sync requis : la suite
    #                     des m est rank-invariante => même nb de convs partout,
    #                     DDP reste en lockstep. tokens/step change (~N/m̄ x) :
    #                     re-checker lr/schedules au moment de config.
    #   allreduce_bf16  : all-reduce des buckets de grads en bf16 (÷2 le volume
    #                     NCCL). Perte de précision ~1 ULP bf16 sur la somme de
    #                     W grads fp32 — à valider par A/B court avant un long run.
    prefetch = bool(t.get("prefetch", False))
    chunk_budget = int(t.get("chunk_budget", 0) or 0)
    ar_bf16 = bool(t.get("allreduce_bf16", False))
    if prefetch:
        assert train_stream.B > 1, "prefetch: chemin batché (batch_size > 1) uniquement"
        assert (chat_stream is None and ra_prob == 0.0 and not ilv_on
                and no_reset_files == 1), \
            "prefetch: chemin batché PUR (pas de chat/reset_announce/interleave/no_reset)"
    if chunk_budget:
        assert train_stream.B > 1 and depth_sync, \
            "chunk_budget: chemin batché + depth_sync requis (lockstep du nb de convs)"
        assert grad_accum == 1, "chunk_budget remplace grad_accum (laisser grad_accum: 1)"
        assert chat_stream is None and ra_prob == 0.0, \
            "chunk_budget: chemin batché pur (pas de chat/reset_announce)"
    lam = float(t.get("defer_weight", 1.0))
    wsd = bool(t.get("wsd_decay", True)); wsd_floor = float(t.get("wsd_floor", 0.0))
    decay_start = int(t.get("wsd_decay_start", int(steps * 0.66)))
    # decay shape over the decay window (p = fraction of the window elapsed):
    #   linear : 1-p (legacy)
    #   step   : DeepSeek-V2/V3 — x0.316 immediately at decay_start, x0.1 for the last
    #            quarter (same 3:1 phase ratio as their 60%/90%-of-total boundaries).
    #            Leaves the read-destroying full-LR zone in ONE step.
    #   1-sqrt : Hägele et al. 2024 WSD-cooldown winner — fast early drop, long low tail
    #   cosine : Chinchilla/LLaMA classic
    #   stair  : `wsd_stair_n` paliers géométriques jusqu'à `wsd_stair_end`. Le
    #            run `fromsif_exec_tiled` n'a rien vu bouger AVANT le decay ; on
    #            donne donc plusieurs baisses au lieu d'une rampe, chaque palier
    #            assez long pour recevoir ses évals — l'escalier n'est pas qu'un
    #            schedule, c'est la mesure « à quel LR le canal s'ouvre ».
    decay_shape = str(t.get("wsd_decay_shape", "linear"))
    stair_n = int(t.get("wsd_stair_n", STAIR_N))
    stair_end = float(t.get("wsd_stair_end", STAIR_END))
    log_every, eval_every = int(t.get("log_every", 20)), int(t.get("eval_every", 200))
    eval_depths = list(t.get("eval_depths", []) or [])   # [] => depth-stratified eval OFF
    eval_depth_convs = int(t.get("eval_depth_convs", 8))
    # step == steps déclenche l'éval même avec eval_every énorme : sur un mix
    # large ça bloque ~40 min de GPU en fin de run (et le rang 0 seul en DDP).
    skip_final_eval = bool(t.get("skip_final_eval", False))
    save_every, save_dir = int(t.get("save_every", 500)), t["save_dir"]
    metrics_file = t.get("metrics_file"); os.makedirs(save_dir, exist_ok=True)
    if metrics_file: os.makedirs(os.path.dirname(metrics_file), exist_ok=True)
    if ddp_rank != 0:
        metrics_file = None                       # IO (metrics/tb/eval/save) = rank0 only
    writer = None
    if metrics_file:
        from torch.utils.tensorboard import SummaryWriter
        tb_dir = os.path.join(os.path.dirname(metrics_file), "tb")
        writer = SummaryWriter(tb_dir); print(f"tensorboard → {tb_dir}", flush=True)

    def set_lr(step):
        # Le calcul vit dans sched.py (mêmes valeurs, self-test à l'appui) : le
        # dry-run --check imprime la table depuis la MÊME source que le trainer.
        f = lr_scale(step, steps=steps, warmup=warmup, wsd=wsd,
                     decay_start=decay_start, shape=decay_shape, floor=wsd_floor,
                     stair_n=stair_n, stair_end=stair_end)
        for gp in opt.param_groups:
            gp["lr"] = muon_lr * f * gp.get("lr_scale", 1.0)
        ad = getattr(opt, "_adam", None)
        if ad:
            for gp in ad.param_groups: gp["lr"] = lr * f
        return muon_lr * f

    # Sauvegarde / reprise : voir ckpt.py (écriture atomique, RNG des streams,
    # la banque comme artefact séparé).
    def _save_ck(step, path):
        save_train_state(path, step=step, model=base, cfg=cfg, opt=opt,
                         delta=delta, ema_ic=ema_ic, ema_d=ema_d,
                         train_stream=train_stream, eval_stream=eval_stream,
                         chat_stream=chat_stream,
                         extra=(None if tf_tables is None else
                                {"tf_tables": tf_tables.state_dict(),
                                 "tf_topt": tf_topt.state_dict()}))

    def _save_bank(step, path):
        save_bank(path, step=step, bank=bank_carry, casc=casc_carry,
                  n_evict=nev_carry, w_total=wt_carry)

    def _load_bank(path, tag):
        return load_bank(path, tag, device)

    _bank_loaded, _casc_loaded, _nev_loaded, _wt_loaded = None, None, 0, 0
    _bi = t.get("bank_init")                 # chemin explicite : seed la vie avec
    if _bi:                                  # une banque venue d'un autre run
        _bank_loaded, _casc_loaded, _nev_loaded, _wt_loaded = \
            _load_bank(_bi, "bank_init")

    start_step = 0; ema_ic = ema_d = ema_a = ema_chat = ema_lane = None
    ema_reach = [None, None, None]              # EMA de perte par strate d'âge
    if resume:
        _rs, _rp, _done = find_resume(save_dir)
        if _done:
            print(f"resume: {os.path.join(save_dir, 'final.pt')} exists — "
                  f"training already complete, nothing to do.", flush=True)
            return
        if _rp is not None:
            start_step = _rs
            ck = torch.load(_rp, map_location="cpu", weights_only=False)
            ema_ic, ema_d = restore_train_state(
                ck, model=base, opt=opt, delta=delta,
                train_stream=train_stream, eval_stream=eval_stream,
                chat_stream=chat_stream, ddp_rank=ddp_rank,
                train_seed=train_seed, base_seed=sd["seed"],
                start_step=start_step)
            if tf_tables is not None and ck.get("tf_tables") is not None:
                tf_tables.load_state_dict(ck["tf_tables"])
                if ck.get("tf_topt") is not None:
                    tf_topt.load_state_dict(ck["tf_topt"])
                print("resume: tables teacher value_table restaurées",
                      flush=True)
            _bp = os.path.join(save_dir, f"bank_step_{start_step}.pt")
            if os.path.exists(_bp):
                _bank_loaded, _casc_loaded, _nev_loaded, _wt_loaded = \
                    _load_bank(_bp, "resume")
            print(f"resume: restored {_rp} @step {start_step} "
                  f"(opt {'yes' if ck.get('opt') else 'NO — old-format ck'})", flush=True)
        else:
            print("resume: no checkpoint found, starting fresh.", flush=True)

    _pf_q = None
    if prefetch:
        # Producteur UNIQUE de train_stream à partir d'ici (l'ordre des tirages
        # rng est préservé : queue FIFO, un seul thread). Démarré APRÈS le
        # resume pour produire depuis l'état rng restauré. Daemon : meurt avec
        # le process (fin de run / préemption).
        import threading, queue as _pyqueue
        _pf_q = _pyqueue.Queue(maxsize=2)

        def _pf_worker():
            while True:
                segs_ = train_stream.next_conv_batch(defer_len)
                for s_ in segs_:
                    for k_, v_ in s_.items():
                        if torch.is_tensor(v_):
                            s_[k_] = v_.pin_memory()
                _pf_q.put(segs_)

        threading.Thread(target=_pf_worker, daemon=True).start()
        print(f"prefetch: ON (queue 2, pin_memory, H2D non_blocking)", flush=True)
    if chunk_budget:
        print(f"chunk_budget: {chunk_budget} chunks/step (accumulation dynamique, "
              f"grads /= nb convs)", flush=True)

    model.train(); t0 = time.time()
    _win_data = 0.0; _win_chunks = 0    # fenêtre log_every : temps d'attente data + chunks vus
    _tf_gnorm = 0.0                     # value_table : dernier grad-norm des tables
    # carries hoisted out of the step loop: with nrf_never they persist across
    # optimizer steps (une vie = le run entier) ; sinon ils sont reset par step.
    bank_carry = _bank_loaded
    casc_carry, nev_carry = _casc_loaded, _nev_loaded
    dstate_carry = None
    rpool_carry, wt_carry = [], _wt_loaded
    for step in range(start_step + 1, steps + 1):
        _t_step0 = time.time()
        lr_now = set_lr(step)
        opt.zero_grad(set_to_none=True)
        if tf_topt is not None:
            tf_topt.zero_grad(set_to_none=True)
        ic_v = d_v = a_v = 0.0; ic_cnt = d_cnt = a_cnt = 0; distill_v = 0.0; distill_n = 0
        # dist ventilée porteur/filler : porteur = seg avec val_mask (fait
        # énoncé/màj). Si fait descend et fill reste ~1.0 = le write imite le
        # contenu ; les deux plats = le write n'imite rien (lever distill_w/α).
        dist_c = dist_f = 0.0; dist_cn = dist_fn = 0
        chat_v = 0.0; chat_cnt = 0
        # ce/lane : la CE normalisée DANS chaque lane puis moyennée — hors
        # gradient, c'est la seule quantité comparable au `chat` d'un run B=1
        # (la loss, elle, normalise globalement sur le batch).
        lane_v = 0.0; lane_cnt = 0
        _step_convs = []                     # trace repro nan-guard (voir plus bas)
        reach_v = [0.0, 0.0, 0.0]; reach_cnt = [0, 0, 0]
        # gradient accumulation: G independent conversations (batch=1 each, bank reset
        # between them) summed into one optimizer step => effective batch = G files,
        # variance reduced without padding/GPU-batching the ragged chunks.
        if not nrf_never:
            bank_carry = None
            casc_carry, nev_carry = None, 0
            dstate_carry = None
            rpool_carry, wt_carry = [], 0
        n_conv = 0; step_chunks = 0; data_t = 0.0
        while (step_chunks < chunk_budget) if chunk_budget else (n_conv < grad_accum):
            _g = n_conv
            _t_d = time.time()
            is_chat = (chat_stream is not None
                       and train_stream.rng.random() < p_chat)
            if _pf_q is not None:
                # data_t mesure l'ATTENTE réelle (0 si le producteur est en
                # avance) ; H2D non_blocking depuis la mémoire pinnée, ordonné
                # sur le stream par défaut donc sûr vis-à-vis des forwards.
                segs = [{k: (v.to(device, non_blocking=True)
                             if torch.is_tensor(v) else v) for k, v in s.items()}
                        for s in _pf_q.get()]
            else:
                segs = (chat_stream.next_conv_batch() if is_chat and chat_B > 1
                        else chat_stream.next_conv()["segs"] if is_chat
                        else train_stream.next_conv_batch(defer_len) if train_stream.B > 1
                        else train_stream.next_conv_interleaved(
                            interleave_files, defer_len,
                            label=addr_label, addr_prob=addr_prob, addr_max=addr_max)
                        if ilv_on else train_stream.next_conv())
            data_t += time.time() - _t_d
            # chat batché : un seg = B tours réels. On les compte tous, sinon
            # `chunks/step` (et le tokens/step qu'on en déduit) chuterait d'un
            # facteur B alors que le travail est identique.
            step_chunks += len(segs) * (chat_B if (is_chat and chat_B > 1) else 1)
            # B2 : la vie se termine à la fin de la DERNIÈRE conv du groupe
            # no_reset ((_g+1) % nrf == 0 ; nrf=1 => chaque conv est une vie).
            if (ra_ids is not None and not is_chat
                    and (_g + 1) % max(no_reset_files, 1) == 0
                    and train_stream.rng.random() < ra_prob):
                for s_ in segs[-ra_chunks:]:
                    s_["input_ids"] = torch.cat(
                        [ra_ids.unsqueeze(0), s_["input_ids"]], dim=1)
            if (nrf_never and bank_carry is not None) or (
                    no_reset_files > 1 and _g % no_reset_files != 0):
                bank = bank_carry                     # dirty start: previous file's gists
                casc, n_evict = casc_carry, nev_carry
                if cascade_depth and casc is None:    # banque chargée sans cascade
                    casc = CascadeMemory(cascade_depth, cfg.max_mem)
                dstate = dstate_carry
                reach_pool, w_total = rpool_carry, wt_carry
            else:
                bank = None
                casc = CascadeMemory(cascade_depth, cfg.max_mem) if cascade_depth else None
                n_evict = 0
                dstate = delta.init_state(_B, device) if delta is not None else None
                reach_pool, w_total = [], 0           # le pool meurt avec la vie
            total = 0.0
            reach_n = 0                               # cap VRAM par conv
            # trace repro nan-guard : entrées de la conv AVANT forward (segs =
            # tenseurs CPU du générateur, banque d'entrée = ref détachée) — si
            # le guard grad-norm trip en fin de step, on dumpe tout le step
            _step_convs.append({
                "segs": segs,
                "bank_in": None if bank is None else bank.detach(),
                "casc": None if casc is None else casc.state_dict(),
                "n_evict": n_evict})
            for i, s in enumerate(segs):
                x = s["input_ids"].to(device)
                chat_seg = "loss_mask" in s          # chat segs: no <think>,
                xt = x if chat_seg else _append(x, think_id)  # masked CE
                if casc is not None and bank is None:
                    # seed explicite : les niveaux profonds lisent None (vide),
                    # jamais la banque vive par accident au premier chunk
                    bank = model.thought_stream.seed_bank(
                        x.size(0), device, next(model.parameters()).dtype)
                # capture d'éviction AVANT le write : FIFO pur => le slot 0 de la
                # banque pleine est celui qui déborde vers la page (grain slot,
                # spec « débordement en 2 temps » — les seeds ne descendent pas)
                pre0 = (bank[:, 0].detach()
                        if casc is not None and bank.size(1) >= cfg.max_mem else None)
                lb = casc.layer_banks(bank, cascade_map) if casc is not None else None
                if chat_seg:
                    _lm = s["loss_mask"]
                    # décidé côté CPU, AVANT le .to(device) : cf. _chat_loss
                    _any = bool(_lm[:, 1:].any())
                    _pm = s.get("pad_mask")
                    loss, bank, ce, ce_lane = _chat_loss(
                        _fwd, xt, _lm.to(device), bank, balw, amp, lb,
                        pad_mask=None if _pm is None else _pm.to(device),
                        m_any=_any)
                    loss = chat_w * loss
                    if ce is not None:
                        chat_v = chat_v + ce; chat_cnt += 1
                    if ce_lane is not None:
                        lane_v = lane_v + ce_lane; lane_cnt += 1
                    ce = None
                else:
                    loss, bank, ce = _ic_loss(_fwd, xt, bank, balw, amp, lb)
                if delta is not None:
                    # B4 : le write du modèle reste actif DANS le chunk (même
                    # forward), mais le carry inter-chunks = l'état delta
                    dstate = delta.update(dstate, model.embed.weight[x])
                    bank = delta.to_bank(dstate, next(model.parameters()).dtype)
                if pre0 is not None:
                    n_evict += 1
                    if n_evict > _seed_slots:
                        casc.push_slot(pre0)
                total = total + loss
                if ce is not None:
                    ic_v += ce; ic_cnt += 1
                # teacher par SEG (run 6) : le blend s'applique à CHAQUE seg —
                # chat inclus — plus seulement aux segs à cible defer. Le slot
                # fraîchement écrit est tiré vers un gist prédictible du seg :
                # le CE de la réponse peut alors trouver le routage read→réponse
                # (recette anti point-fixe 47M). target=value (run 7-resume) :
                # cible = proj de l'embedding des tokens VALEUR (val_mask), code
                # discriminant par valeur, et NE tire que les segs porteurs.
                beta = _beta(step)
                # ── target value_table : codes appris composés (liaison) ────
                # Ne fire que sur les segs d'énonciation/update (fact_slot > 0
                # sur au moins une lane). gist NON détaché dans le blend : la
                # CE des segs suivants remonte au travers du read jusqu'aux
                # tables. Distill = pull du write vers le code DÉTACHÉ (le
                # teacher n'apprend pas de la distill — 47M verbatim). Gating
                # PAR LANE : en chat batché les lanes non porteuses (filler,
                # sota) gardent leur write natif.
                if tf_target == "value_table":
                    fs = s.get("fact_slot")
                    if (tf_on and beta > 0.0 and fs is not None
                            and int(fs.sum()) > 0):
                        sid = fs.to(device).max(dim=1).values          # [B]
                        vid = s["fact_val"].to(device).max(dim=1).values
                        aid = s["fact_attr"].to(device).max(dim=1).values
                        gist = (tf_tables["slot"](sid) + tf_tables["attr"](aid)
                                + tf_tables["val"](vid))               # fp32
                        gist = gist / gist.pow(2).mean(-1, keepdim=True) \
                            .clamp_min(1e-12).sqrt()
                        carrier = sid > 0                              # [B]
                        w0 = bank[:, -1]
                        per = 1.0 - F.cosine_similarity(
                            w0.float(), gist.detach(), dim=1)
                        distill = (per * carrier.float()).sum() \
                            / carrier.sum().clamp_min(1)
                        total = total + tf_dw * beta * distill
                        distill_v += float(distill.detach()); distill_n += 1
                        dist_c += float(distill.detach()); dist_cn += 1
                        blended = torch.where(
                            carrier.unsqueeze(1),
                            beta * gist.to(w0.dtype) + (1.0 - beta) * w0,
                            w0).unsqueeze(1)
                        bank = torch.cat([bank[:, :-1], blended], dim=1)
                # Cible PAR SEG. 'value_sif' prend le val_mask quand le seg en
                # porte un (ici : le span des noms d'outils déclarés) et retombe
                # sur le pooling pondéré sinon — les deux autres modes gardent
                # exactement leur comportement historique (bit à bit).
                vmask = (s.get("val_mask")
                         if tf_target in ("value", "value_sif") else None)
                surpw = (s.get("surp_w")
                         if tf_target in ("surprisal", "value_sif") else None)
                fire = tf_on and beta > 0.0 and (
                    tf_target == "chunk" or vmask is not None
                    or surpw is not None)
                if fire:
                    with torch.no_grad():
                        emb = model.embed.weight[x].float()          # [B,T,D]
                        alt = None
                        if surpw is not None:
                            sw = surpw.to(device).unsqueeze(-1).float()  # [B,T,1]
                            alt = (emb * sw).sum(dim=1) / sw.sum(dim=1).clamp_min(1e-6)
                        if vmask is not None:
                            vm = vmask.to(device).unsqueeze(-1).float()  # [B,T,1]
                            vs = vm.sum(dim=1)                           # [B,1]
                            pooled = (emb * vm).sum(dim=1) / vs.clamp_min(1.0)
                            # LE REPLI EST PAR LANE, pas par seg : en chat batché
                            # les lanes portent des sessions différentes, donc un
                            # même seg peut être un schéma d'outil en lane 0 et un
                            # tour sota en lane 2, où `val_mask` est un pad de
                            # zéros. Sans ce `where`, ces lanes-là recevaient une
                            # cible NULLE — cosine 0, distill 1, et le blend
                            # tirait leur slot vers zéro. (Bug présent aussi en
                            # 'value' pur ; il n'avait jamais mordu, ce mode
                            # n'ayant tourné que sur un mix 100 % porteur.)
                            pooled = torch.where(
                                vs > 0, pooled,
                                alt if alt is not None else emb.mean(dim=1))
                        elif alt is not None:
                            pooled = alt
                        else:
                            pooled = emb.mean(dim=1)
                        gist = pooled @ tf_proj.float()
                        gist = gist / gist.pow(2).mean(-1, keepdim=True).clamp_min(1e-12).sqrt()
                    w0 = bank[:, -1]
                    distill = (1.0 - F.cosine_similarity(w0.float(), gist, dim=1)).mean()
                    total = total + tf_dw * beta * distill
                    distill_v += float(distill.detach()); distill_n += 1
                    # porteur = le val_mask a de la MASSE, pas seulement la clé :
                    # en chat batché `_ZERO_PAD_KEYS` la crée dès qu'une lane en
                    # a une, donc tester `is not None` compterait tout le batch
                    # comme porteur et la ventilation ne dirait plus rien.
                    if vmask is not None and float(vmask.sum()) > 0:
                        dist_c += float(distill.detach()); dist_cn += 1
                    else:
                        dist_f += float(distill.detach()); dist_fn += 1
                    blended = (beta * gist.to(w0.dtype) + (1.0 - beta) * w0).unsqueeze(1)
                    bank = torch.cat([bank[:, :-1], blended], dim=1)
                # deferred target: batched segs carry their own defer_tgt (incl. the
                # LAST turn's external successor, -100-padded); batch=1 derives it
                # from the next in-conv chunk as before.
                nxt = s.get("defer_tgt")
                if nxt is None and not chat_seg and i < len(segs) - 1:
                    nxt = segs[i + 1]["input_ids"][:, :defer_len]
                if nxt is not None and bool((nxt != -100).any()):
                    nxt = nxt.to(device)
                    dl = nxt.size(1)                   # ragged: remainder chunk may be < defer_len
                    di = _fill(x, blank_id, dl)
                    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                        od = _fwd(di, init_mem=bank,
                                  layer_banks=casc.layer_banks(bank, cascade_map)
                                  if casc is not None else None)
                    lg = od["logits"].float()
                    dloss = F.cross_entropy(lg.reshape(-1, lg.size(-1)), nxt.reshape(-1),
                                            ignore_index=-100)
                    total = total + lam * dloss; d_v += float(dloss.detach()); d_cnt += 1
                    # deferred forward's own write is discarded (do NOT carry od bank)
                # G2 addressed defer: [cue, blanks] toward a NON-last stream; loss
                # only on the blank positions (cue is context, not supervision)
                ac, at = s.get("addr_cue"), s.get("addr_tgt")
                if ac is not None and bool((at != -100).any()):
                    ac, at = ac.to(device), at.to(device)
                    di = torch.cat([ac, _fill(ac, blank_id, at.size(1))], dim=1)
                    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                        oa = _fwd(di, init_mem=bank,
                                  layer_banks=casc.layer_banks(bank, cascade_map)
                                  if casc is not None else None)
                    lga = oa["logits"].float()[:, ac.size(1):]
                    aloss = F.cross_entropy(lga.reshape(-1, lga.size(-1)),
                                            at.reshape(-1), ignore_index=-100)
                    total = total + lam * aloss
                    a_v += float(aloss.detach()); a_cnt += 1
                    # addressed forward's write is discarded too
                # OPTION 2 : defer reach-back vers une entrée du pool dont les
                # slots ont quitté la banque vive (âge >= max_mem writes) —
                # même format que l'addr defer ([cue, blanks], perte sur les
                # blanks, write jeté), mais la cible vient d'une conv PASSÉE
                # de la vie : la page (ou ses résidus) est le seul pont.
                if reach_prob > 0 and reach_n < reach_max:
                    M_ = cfg.max_mem
                    elig = [e for e in reach_pool if w_total - e["w"] >= M_]
                    if elig and train_stream.rng.random() < reach_prob:
                        e = train_stream.rng.choice(elig)
                        age = w_total - e["w"]
                        sb = 0 if age < 2 * M_ else (1 if age < 4 * M_ else 2)
                        rc = e["cue"].to(device)
                        rt = e["tgt"].to(device)
                        di = torch.cat([rc, _fill(rc, blank_id, rt.size(1))], dim=1)
                        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                            orr = _fwd(di, init_mem=bank,
                                       layer_banks=casc.layer_banks(bank, cascade_map)
                                       if casc is not None else None)
                        lgr = orr["logits"].float()[:, rc.size(1):]
                        rloss = F.cross_entropy(lgr.reshape(-1, lgr.size(-1)),
                                                rt.reshape(-1), ignore_index=-100)
                        total = total + lam * rloss
                        reach_v[sb] += float(rloss.detach()); reach_cnt[sb] += 1
                        reach_n += 1
                # dépôt au pool APRÈS usage (une entrée n'est jamais éligible
                # dans sa propre conv : âge < M garanti par m_total <= K = M)
                if reach_prob > 0:
                    w_total += 1                      # ce seg vient d'écrire 1 gist
                    tg = s.get("defer_tgt")
                    if tg is not None and bool((tg != -100).any()):
                        reach_pool.append({"cue": s["input_ids"][:, :reach_cue_len],
                                           "tgt": tg, "w": w_total})
                        if len(reach_pool) > 64:      # borne mémoire (tenseurs CPU 16+16 tok)
                            reach_pool.pop(0)
            # nan-guard (incident run 5e step 33 : chat nan -> opt.step sur grads
            # NaN = poids morts ; le pod phase 1 avait son guard, ce chemin non).
            # Conv non finie => pas de backward, et la banque de cette conv ne
            # rentre JAMAIS dans le carry (en never-reset un carry NaN est éternel).
            tot_ok = bool(torch.isfinite(total).all()) if torch.is_tensor(total) \
                else math.isfinite(total)
            if not tot_ok:
                # dump repro : la conv fautive + son état d'entrée, pour rejouer
                # offline GC on/off (hypothèse user : recompute du checkpoint qui
                # diverge du forward sur le routage MoE => grads incohérents)
                _dp = os.path.join(save_dir, f"nan_conv_step{step}_g{_g}.pt")
                try:
                    torch.save({"step": step, "g": _g,
                                "segs": [{k: (v.cpu() if torch.is_tensor(v) else v)
                                          for k, v in s.items()} for s in segs],
                                "bank_in": None if bank_carry is None
                                else bank_carry.detach().cpu(),
                                "casc": None if casc is None else casc.state_dict(),
                                "n_evict": n_evict}, _dp)
                except Exception as e:
                    _dp = f"dump raté: {e}"
                # les poids du MOMENT du nan (une seule fois par run) : sans eux
                # le repro depuis le dernier ckpt peut ne pas trigger
                _wp = os.path.join(save_dir, "nan_weights.pt")
                if not os.path.exists(_wp):
                    torch.save({"step": step, "model": base.state_dict()}, _wp)
                print(f"[nan-guard] step {step} conv {_g}: loss non finie, "
                      f"conv sautée (carry préservé) — repro {_dp}", flush=True)
                continue
            # mean over the accumulated convs ; en mode chunk_budget le nb de
            # convs n'est connu qu'en fin de step => division des grads après.
            (total / (1.0 if chunk_budget else grad_accum)).backward()
            if no_reset_files > 1 or nrf_never:
                # graph freed per file; the carried bank is data, not gradient path
                # guard local (supersède le nan_to_num du pod) : banque non
                # finie => carry RESET vie neuve, sinon carry propre
                nb = bank.detach() if bank is not None else None
                if nb is not None and not bool(torch.isfinite(nb).all()):
                    print(f"[nan-guard] step {step} conv {_g}: banque non finie, "
                          f"carry RESET (vie neuve)", flush=True)
                    bank_carry, casc_carry, nev_carry = None, None, 0
                    rpool_carry, wt_carry = [], 0
                else:
                    bank_carry = nb
                    casc_carry, nev_carry = casc, n_evict
                    if reach_prob > 0:
                        rpool_carry, wt_carry = reach_pool, w_total
                if delta is not None:
                    dstate_carry = dstate.detach()
            n_conv += 1
        if chunk_budget and n_conv > 1:
            # sémantique grad_accum restaurée : grads = moyenne sur les convs
            _gs = [p.grad for p in base.parameters() if p.grad is not None]
            if delta is not None:
                _gs += [p.grad for p in delta.parameters() if p.grad is not None]
            torch._foreach_div_(_gs, float(n_conv))
        _win_data += data_t; _win_chunks += step_chunks
        _prof = os.environ.get("TB_DDP_PROF")
        if _prof:
            torch.cuda.synchronize(); _t_bwd = time.time()
        if ddp_world > 1:
            # manual grad sync: average across ranks BEFORE clip so every rank
            # computes the same norm and the same update (ranks stay identical).
            # Buckets of ~64MB: one flat all-reduce per bucket instead of one
            # NCCL call per tensor (the MoE makes that hundreds of tiny calls).
            from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
            grads = [p.grad for p in base.parameters() if p.grad is not None]
            if delta is not None:
                grads += [p.grad for p in delta.parameters() if p.grad is not None]
            bucket, nbytes = [], 0
            for g_ in grads + [None]:
                if g_ is None or nbytes + g_.numel() * g_.element_size() > 64 << 20:
                    if bucket:
                        flat = _flatten_dense_tensors(bucket)
                        if ar_bf16:
                            f16 = flat.to(torch.bfloat16)
                            torch.distributed.all_reduce(f16)
                            flat = f16.to(flat.dtype)
                        else:
                            torch.distributed.all_reduce(flat)
                        flat.div_(ddp_world)
                        for b_, s_ in zip(bucket, _unflatten_dense_tensors(flat, bucket)):
                            b_.copy_(s_)
                    bucket, nbytes = [], 0
                if g_ is not None:
                    bucket.append(g_); nbytes += g_.numel() * g_.element_size()
        if _prof:
            torch.cuda.synchronize(); _t_ar = time.time()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(),
                                            float(t.get("grad_clip", 1.0)))
        # DDP-safe (pod 10B) : après l'all-reduce les grads sont identiques sur
        # tous les ranks => ce check prend la même branche partout (pas de désync).
        if not bool(torch.isfinite(gn)):
            # LE cas intéressant pour l'hypothèse GC : loss finie mais grads NaN
            # (recompute qui diverge du forward). Dump du step complet + poids
            # (encore sains puisque le step est sauté).
            if ddp_rank == 0:
                # dump rank0 seulement : les convs diffèrent par rank, 8 writes
                # concurrents du même fichier = corruption
                _dp = os.path.join(save_dir, f"nan_gradstep_{step}.pt")
                try:
                    torch.save({"step": step,
                                "convs": [{"segs": [{k: (v.cpu() if torch.is_tensor(v)
                                                         else v) for k, v in s.items()}
                                                    for s in c["segs"]],
                                           "bank_in": None if c["bank_in"] is None
                                           else c["bank_in"].cpu(),
                                           "casc": c["casc"],
                                           "n_evict": c["n_evict"]}
                                          for c in _step_convs]}, _dp)
                except Exception as e:
                    _dp = f"dump raté: {e}"
                _wp = os.path.join(save_dir, "nan_weights.pt")
                if not os.path.exists(_wp):
                    torch.save({"step": step, "model": base.state_dict()}, _wp)
                # diagnostic pod 10B : quels modules portent les grads non finis
                bad = [n for n, p in model.named_parameters()
                       if p.grad is not None and not torch.isfinite(p.grad).all()]
                print(f"[nan-guard] step {step}: grad norm non finie ({len(bad)} "
                      f"tenseurs: {bad[:6]}), opt.step SAUTÉ — repro {_dp}", flush=True)
            opt.zero_grad(set_to_none=True)
            if tf_topt is not None:
                tf_topt.zero_grad(set_to_none=True)
        else:
            opt.step()
            if tf_topt is not None:
                # norme de grad des tables AVANT le step : la sonde « le read
                # vote le codebook » (0.0 = le blend ne fire pas / graphe coupé)
                _tf_gnorm = float(torch.nn.utils.clip_grad_norm_(
                    tf_tables.parameters(), 1.0e9))
                tf_topt.step()
        if _prof:
            torch.cuda.synchronize()
            print(f"[prof step {step}] fwd+bwd {_t_bwd - _t_step0:.2f}s  "
                  f"allreduce {_t_ar - _t_bwd:.2f}s  clip+opt {time.time() - _t_ar:.2f}s  "
                  f"data-wait {data_t:.2f}s  chunks {step_chunks} ({n_conv} convs)",
                  flush=True)
        ic_v /= max(ic_cnt, 1); d_v /= max(d_cnt, 1)
        if ic_v == ic_v:   # un batch NaN isolé ne doit pas polluer l'EMA à vie
            ema_ic = ic_v if ema_ic is None else 0.95 * ema_ic + 0.05 * ic_v
        if d_v == d_v:
            ema_d  = d_v  if ema_d  is None else 0.95 * ema_d  + 0.05 * d_v
        if a_cnt:
            a_v /= a_cnt
            ema_a = a_v if ema_a is None else 0.95 * ema_a + 0.05 * a_v
        if chat_cnt:
            # UNE seule matérialisation par step (chat_v est un tenseur accumulé
            # sur le device : le convertir par seg coûtait une synchro par seg).
            chat_v = float(chat_v) / chat_cnt
            ema_chat = chat_v if ema_chat is None else 0.95 * ema_chat + 0.05 * chat_v
        if lane_cnt:
            lane_v = float(lane_v) / lane_cnt
            ema_lane = lane_v if ema_lane is None else 0.95 * ema_lane + 0.05 * lane_v
        for _s in range(3):
            if reach_cnt[_s]:
                rv = reach_v[_s] / reach_cnt[_s]
                ema_reach[_s] = (rv if ema_reach[_s] is None
                                 else 0.9 * ema_reach[_s] + 0.1 * rv)
        if step % log_every == 0:
            addr_s = f"addr {ema_a:.3f}  " if ema_a is not None else ""
            if ema_chat is not None:
                addr_s = f"chat {ema_chat:.3f}  " + addr_s
            if ema_lane is not None:                  # batch chat : cf. lane_v
                addr_s = f"ce/lane {ema_lane:.3f}  " + addr_s
            if reach_prob > 0 and any(v is not None for v in ema_reach):
                # s1 ~ page p0, s2 ~ page mergée, s3 ~ détruit (contrôle : ne
                # doit PAS baisser si c'est bien la page qui est lue)
                addr_s += "reach " + "/".join(
                    "—" if v is None else f"{v:.2f}" for v in ema_reach) + "  "
            mem_s = (f"mem {torch.cuda.memory_allocated()/2**30:.1f}/"
                     f"{torch.cuda.max_memory_allocated()/2**30:.1f}G  "
                     if torch.cuda.is_available() else "")
            # probes utiles seulement : ic/defer masqués quand le step n'en a
            # pas vu (SFT pur p_chat=1.0 : ils étaient affichés à 0.000 fixe),
            # distill affiché dès que le teacher contribue (il n'était QUE dans
            # tensorboard pendant que β pilotait la moitié de la loss)
            ic_s = (f"ic {ema_ic:.3f} (ppl {math.exp(ema_ic):.1f})  "
                    if ic_cnt else "")
            d_s = f"defer {ema_d:.3f}  " if d_cnt else ""
            dist_s = (f"dist {distill_v / distill_n:.3f}  " if distill_n else "")
            if dist_cn or dist_fn:
                dist_s += ("[fait " + (f"{dist_c / dist_cn:.3f}" if dist_cn else "—")
                           + "/fill " + (f"{dist_f / dist_fn:.3f}" if dist_fn else "—")
                           + "]  ")
            _n_log = min(log_every, step - start_step)
            print(f"step {step:5d}  {ic_s}{d_s}"
                  f"{addr_s}{dist_s}β {_beta(step):.2f}  lr {lr_now:.2e}  {mem_s}"
                  f"{(time.time()-t0)/max(step - start_step, 1):.2f}s/step  "
                  f"chunks {_win_chunks / max(_n_log, 1):.1f}/step  "
                  f"data {_win_data / max(_n_log, 1):.2f}s", flush=True)
            if writer is not None:
                writer.add_scalar("train/ic_loss", ema_ic, step)
                writer.add_scalar("train/ic_ppl", math.exp(ema_ic), step)
                writer.add_scalar("train/defer_loss", ema_d, step)
                if ema_a is not None:
                    writer.add_scalar("train/addr_loss", ema_a, step)
                if ema_chat is not None:
                    writer.add_scalar("train/chat_loss", ema_chat, step)
                for _s, _v in enumerate(ema_reach):
                    if _v is not None:
                        writer.add_scalar(f"train/reach_s{_s + 1}", _v, step)
                writer.add_scalar("sched/lr", lr_now, step)
                writer.add_scalar("sched/beta", _beta(step), step)
                # perf : chunks/step + attente data (fenêtre log_every) — pour
                # régresser s/step = intercept + pente*m et chiffrer le host-bound
                writer.add_scalar("perf/chunks_per_step", _win_chunks / max(_n_log, 1), step)
                writer.add_scalar("perf/data_wait_s", _win_data / max(_n_log, 1), step)
                writer.add_scalar("perf/s_per_step",
                                  (time.time() - t0) / max(step - start_step, 1), step)
                if distill_n:
                    writer.add_scalar("train/distill", distill_v / distill_n, step)
                if dist_cn:
                    writer.add_scalar("train/distill_fait", dist_c / dist_cn, step)
                if dist_fn:
                    writer.add_scalar("train/distill_fill", dist_f / dist_fn, step)
                if tf_topt is not None:
                    writer.add_scalar("train/table_gnorm", _tf_gnorm, step)
            _win_data = 0.0; _win_chunks = 0
        if (step % eval_every == 0
                or (step == steps and not skip_final_eval)) and ddp_rank == 0:
            # eval_depth_sources (mix large, ex. divmix 13 sources) : la courbe
            # par profondeur coûte 4x le GAP top-level (eval_depths x
            # eval_depth_convs convs PAR source) et ne sert de comparaison que
            # sur les ancres — la restreindre à cette liste ramène l'éval de
            # 13x40 à 13x8 + 2x32 convs. None (défaut) = toutes les sources.
            depth_srcs = t.get("eval_depth_sources")
            if cascade_depth and casc is not None:
                print(f"[cascade @{step}] {casc.stats()} (dernière conv du step)")
            for src_name, es in eval_views:
                tag = f" [{src_name}]" if src_name else ""
                pfx = f"{src_name}/" if src_name else ""
                # eval sur `base` (non-compile) : les convs d'eval ont des largeurs
                # B=1 toutes differentes => un graphe dynamo par shape, premier
                # eval bloque >13 min a step 500 (pod 45191495). Eager = 1-2 min.
                m = evaluate(base, es, device, think_id, blank_id, defer_len,
                             int(t.get("eval_convs", 8)), balw, amp, delta=delta)
                print(f"[eval @{step}]{tag} ic_ppl {m['ic_ppl']:.1f} | defer car {m['defer_car']:.3f} "
                      f"res {m['defer_res']:.3f} GAP {m['defer_gap']:+.3f} GAP0 {m['defer_gap0']:+.3f} "
                      f"| ceil(t0) {m['cont']:.3f} headroom {m['headroom']:+.3f} "
                      f"| GAP hop1 {m['gap_hop1']:+.3f} deep(>=4) {m['gap_deep']:+.3f} (n={int(m['n_deep'])})",
                      flush=True)
                if metrics_file:
                    with open(metrics_file, "a") as f:
                        f.write(json.dumps({"step": step, "source": src_name, **m}) + "\n")
                if writer is not None:
                    for k, v in m.items():
                        writer.add_scalar(f"eval/{pfx}{k}", v, step)
                if eval_depths and (depth_srcs is None or not src_name
                                    or src_name in depth_srcs):
                    bd = evaluate_by_depth(base, es, device, think_id, blank_id,
                                           defer_len, eval_depths, eval_depth_convs, amp,
                                           delta=delta)
                    curve = "  ".join(f"d{d}:{bd[d]['gap']:+.3f}(n{bd[d]['n']})" for d in eval_depths)
                    print(f"[eval @{step}]{tag} GAP by depth (writes→predict next): {curve}", flush=True)
                    if metrics_file:
                        with open(metrics_file, "a") as f:
                            f.write(json.dumps({"step": step, "source": src_name, "gap_by_depth":
                                {str(d): bd[d] for d in eval_depths}}) + "\n")
                    if writer is not None:
                        for d in eval_depths:
                            writer.add_scalar(f"eval_depth/{pfx}gap_d{d}", bd[d]["gap"], step)
            if chat_eval is not None:
                chat_eval.rng.seed(1234)          # same conv set every eval
                dec_on = (step % (eval_every * chat_decode_every) == 0
                          or step == steps)
                mm = evaluate_math(model, chat_eval, tok, device, amp,
                                   chat_eval_convs, chat_max_new,
                                   use_cache=decode_cache, decode=dec_on,
                                   graphs=decode_graphs)
                by_age = mm.pop("_by_age", {})
                for kind in sorted(mm):
                    v = mm[kind]
                    if v["n_ans"]:
                        gp = (f"grade {v['grade']:.2f} abl {v['grade_abl']:.2f} "
                              f"Δg {v['grade'] - v['grade_abl']:+.2f} | "
                              if v["n_dec"] else "grade — | ")
                        print(f"[math @{step}] {kind:10s} nll {v['nll']:.3f} "
                              f"{gp}ans nll "
                              f"{v['ans_nll']:.3f} abl {v['ans_nll_abl']:.3f} "
                              f"Δnll {v['ans_nll_abl'] - v['ans_nll']:+.3f} "
                              f"(n={v['n']})", flush=True)
                    else:                     # contrôle sans truths (smalltalk)
                        print(f"[math @{step}] {kind:10s} nll {v['nll']:.3f} "
                              f"(n={v['n']}, contrôle)", flush=True)
                if by_age:
                    curve = "  ".join(
                        (f"{b}: Δg {by_age[b]['dgrade']:+.2f} "
                         if by_age[b]["n_dg"] else f"{b}: ")
                        + f"Δnll {by_age[b]['dnll']:+.3f} (n{by_age[b]['n']})"
                        for b in AGE_BUCKETS if b in by_age)
                    print(f"[math @{step}] recall par âge (writes fait→réponse)"
                          f" : {curve}", flush=True)
                if metrics_file:
                    with open(metrics_file, "a") as f:
                        f.write(json.dumps({"step": step, "math": mm,
                                            "math_by_age": by_age}) + "\n")
                if writer is not None:
                    for kind, v in mm.items():
                        writer.add_scalar(f"eval_math/{kind}/nll", v["nll"], step)
                        if v["n_dec"]:        # sinon on écrirait un 0 factice
                            writer.add_scalar(f"eval_math/{kind}/grade",
                                              v["grade"], step)
                            writer.add_scalar(f"eval_math/{kind}/grade_abl",
                                              v["grade_abl"], step)
                        if v["n_ans"]:
                            writer.add_scalar(
                                f"eval_math/{kind}/ans_dnll",
                                v["ans_nll_abl"] - v["ans_nll"], step)
        if (step % save_every == 0 or step == steps) and ddp_rank == 0:
            _save_ck(step, os.path.join(save_dir,
                     "final.pt" if step == steps else f"step_{step}.pt"))
            if nrf_never:
                _save_bank(step, os.path.join(save_dir,
                           "bank_final.pt" if step == steps
                           else f"bank_step_{step}.pt"))
    if ddp_world > 1:
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()
    print("done.", flush=True)


# ── self-test ────────────────────────────────────────────────────────────────

def _selftest() -> None:
    """L'alignement des cibles, épinglé.

    La convention du dépôt : les loaders NE pré-décalent PAS, c'est la loss qui
    aligne (`logits[:, :-1]` contre `x[:, 1:]`). Un décalage en trop est
    invisible — le modèle apprend juste une tâche décalée d'un cran et la loss
    descend quand même. On construit donc un modèle PARFAIT sauf à une position
    connue, et on vérifie que la loss accuse exactement celle-là.
    """
    V, B, T = 30, 2, 8
    torch.manual_seed(0)
    x = torch.randint(0, V, (B, T))

    class _Oracle:
        """Prédit le token suivant parfaitement, sauf aux positions `broken`
        (indices de la CIBLE, donc des positions dans x)."""

        def __init__(self, broken=()):
            self.broken = set(broken)

        def __call__(self, ids, init_mem=None, layer_banks=None, **kw):
            b, t = ids.shape
            lg = torch.full((b, t, V), -20.0)
            for i in range(t - 1):
                tgt = ids[:, i + 1]                    # ce que la position i doit prédire
                if (i + 1) in self.broken:
                    tgt = (tgt + 1) % V                # ...et qu'elle rate
                lg[torch.arange(b), i, tgt] = 20.0
            lg[:, -1] = 0.0                            # dernière position : hors cible
            return {"logits": lg, "mem_bank": init_mem,
                    "balance_loss": torch.zeros(())}

    # ── _ic_loss : un oracle parfait doit coûter ~0 ─────────────────────────
    _, _, ce = _ic_loss(_Oracle(), x, None, 0.0, False)
    assert ce < 1e-3, (f"oracle parfait, CE {ce:.4f} — la loss ne s'aligne pas sur "
                       f"logits[:, :-1] vs x[:, 1:]")
    # …et un oracle cassé partout doit coûter cher (la loss regarde bien qqch)
    _, _, ce_bad = _ic_loss(_Oracle(broken=range(1, T)), x, None, 0.0, False)
    assert ce_bad > 10.0, ce_bad

    # ── le décalage est EXACT, pas seulement cohérent ───────────────────────
    # Casser la cible p doit se voir ; le reste doit rester propre.
    p = 3
    _, _, ce_p = _ic_loss(_Oracle(broken={p}), x, None, 0.0, False)
    assert ce_p > 1.0, f"cible {p} cassée mais CE {ce_p:.4f} : la loss l'a manquée"
    assert abs(ce_p - 40.0 / (T - 1)) < 1.0, \
        f"CE {ce_p:.3f} : une seule position fautive sur {T - 1} attendue"

    # ── la loss d'équilibrage MoE entre bien avec son poids ─────────────────
    class _Bal(_Oracle):
        def __call__(self, ids, **kw):
            o = super().__call__(ids, **kw)
            o["balance_loss"] = torch.tensor(2.0)
            return o

    loss, _, _ = _ic_loss(_Bal(), x, None, 0.5, False)
    assert abs(float(loss) - 1.0) < 1e-2, f"CE~0 + 0.5*2.0 attendu, vu {float(loss)}"

    # ── _chat_loss : le masque sélectionne les positions supervisées ────────
    mask = torch.zeros(B, T)
    mask[:, p] = 1.0                                   # on ne supervise QUE la cible p
    _, _, ce_on, _ = _chat_loss(_Oracle(broken={p}), x, mask, None, 0.0, False)
    assert ce_on > 10.0, \
        (f"CE {ce_on:.4f} : le masque doit tomber sur la position cassée — "
         f"c'est ici qu'un off-by-one sur lmask[:, 1:] se voit")
    _, _, ce_off, _ = _chat_loss(_Oracle(broken={p}), x, mask, None, 0.0, False)
    other = torch.zeros(B, T); other[:, p + 1] = 1.0
    _, _, ce_other, _ = _chat_loss(_Oracle(broken={p}), x, other, None, 0.0, False)
    assert ce_other < 1e-3, \
        f"CE {ce_other:.4f} : une position saine a été facturée pour sa voisine"
    assert ce_off == ce_on                              # déterminisme

    # segment tout-masqué (tour utilisateur) : il FORWARD mais ne coûte rien
    loss_u, _, ce_u, _ = _chat_loss(_Oracle(broken=range(1, T)), x,
                                    torch.zeros(B, T), None, 1.0, False)
    assert ce_u is None and float(loss_u) == 0.0, \
        "un segment sans supervision doit passer sans CE (son write est le but)"
    # m_any court-circuite le test (chemin chaud) SANS changer le résultat :
    # c'est tout l'intérêt, la synchro par seg disparaît mais la CE est la même
    _, _, ce_fast, _ = _chat_loss(_Oracle(broken={p}), x, mask, None, 0.0, False,
                                  m_any=True)
    assert ce_fast == ce_on, "m_any explicite doit donner la MÊME CE"
    _, _, ce_skip, _ = _chat_loss(_Oracle(broken=range(1, T)), x,
                                  torch.zeros(B, T), None, 0.0, False, m_any=False)
    assert ce_skip is None, "m_any=False doit sauter la CE comme le test interne"

    # ── batch chat : la CE par lane est la diagnostique, pas la loss ─────────
    # La loss normalise GLOBALEMENT (somme des poids du batch) ; avec des lanes
    # de longueurs différentes les deux quantités DIVERGENT, et c'est justement
    # pourquoi on logge la seconde. Ici lane 0 a 1 position supervisée, lane 1
    # en a 2 : la CE globale pèse la lane 1 deux fois plus, la CE par lane non.
    xb = x[:1].repeat(2, 1)
    mb = torch.zeros(2, T)
    mb[0, p] = 1.0
    mb[1, p] = 1.0; mb[1, p + 1] = 1.0
    _, _, ce_g, ce_l = _chat_loss(_Oracle(broken={p}), xb, mb, None, 0.0, False)
    assert ce_l is not None, "B>1 doit produire la diagnostique par lane"
    # lane 0 : CE(p) ; lane 1 : (CE(p) + CE(p+1))/2 avec CE(p+1)~0 => moitié
    assert abs(float(ce_l) - 0.75 * float(ce_on)) < 1e-3 * float(ce_on), \
        f"CE par lane {float(ce_l):.3f} : moyenne des lanes attendue"
    # une lane NON supervisée (tour utilisateur) ne doit PAS entrer dans la
    # moyenne : sinon elle y verse un 0 et la diagnostique n'est plus comparable
    # à un run B=1, où ces segs sont simplement absents de la moyenne.
    mb2 = torch.zeros(2, T); mb2[0, p] = 1.0            # lane 1 = tout masquée
    _, _, _, ce_l2 = _chat_loss(_Oracle(broken={p}), xb, mb2, None, 0.0, False)
    assert abs(float(ce_l2) - float(ce_on)) < 1e-3 * float(ce_on), \
        (f"CE par lane {float(ce_l2):.3f} != {float(ce_on):.3f} : une lane sans "
         f"supervision a été comptée comme un 0")
    assert abs(float(ce_g) - 2 / 3 * float(ce_on)) < 1e-3 * float(ce_on), \
        f"CE globale {float(ce_g):.3f} : 2 positions sur 3 pondérées à parts égales"
    assert float(ce_g) != float(ce_l), \
        "si les deux normalisations coïncident, le test ne prouve rien"

    # ── helpers de remplissage ──────────────────────────────────────────────
    ref = torch.zeros(B, 3, dtype=torch.long)
    assert _fill(ref, 7, 4).shape == (B, 4) and (_fill(ref, 7, 4) == 7).all()
    assert _append(ref, 9).shape == (B, 4) and (_append(ref, 9)[:, -1] == 9).all()

    # ── strates d'âge du reach-back ─────────────────────────────────────────
    got = [_age_bucket(a) for a in (0, 4, 5, 8, 9, 16, 17, 1000)]
    assert got == ["<=4", "<=4", "5-8", "5-8", "9-16", "9-16", ">16", ">16"], got
    assert set(got) <= set(AGE_BUCKETS), "strate hors du vocabulaire déclaré"
    # les bornes sont INCLUSIVES : c'est ce que supposent les tables de FINDINGS
    assert _age_bucket(4) != _age_bucket(5) and _age_bucket(16) != _age_bucket(17)

    print("code_defer_native self-test: OK (alignement des cibles épinglé sur "
          "_ic_loss ET _chat_loss — position fautive localisée au bon index, "
          "masque tout-zéro sans CE, m_any sans effet sur la CE, batch chat : "
          "CE globale != CE par lane, poids de balance, helpers, strates d'âge)")


if __name__ == "__main__":
    flags = {"--resume", "--check"}
    args = [a for a in sys.argv[1:] if a not in flags]
    if not args:                       # contrat selftest.sh : sans argument = self-test
        _selftest()
    elif "--check" in sys.argv[1:]:
        dry_run(args[0])
    else:
        main(args[0], resume="--resume" in sys.argv[1:])
