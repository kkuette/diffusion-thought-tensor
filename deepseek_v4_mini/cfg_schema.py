"""Schéma des configs : une clé inconnue arrête le run au lieu de le fausser.

Le problème. Un trainer lit sa config par `t.get("wsd_decay_start", 0)`. Une
faute de frappe dans le YAML ne lève rien : la valeur par défaut s'applique, le
run part avec un autre schedule, et ça ne se voit qu'au dépouillement — après
avoir payé les heures GPU. `code_defer_native` lit à lui seul 48 clés
`training:`, 17 `data:`, 6 `chat:`, 4 `teacher:` de cette façon.

Ce module déclare, par point d'entrée, les clés qui existent, et `check()` refuse
tout le reste en suggérant la clé voisine. Il s'appelle en tête de `main()`,
juste après `load_yaml` (`paths.load_yaml` reste l'entonnoir unique).

Ce qu'il ne re-vérifie PAS, parce que c'est déjà strict ailleurs :
  - `model:`   → `ThoughtBankConfig(**raw["model"])` lève sur clé inconnue ;
  - `chat.gen` → `Cls(tok, seed=..., **gen)` lève sur kwarg inconnu.
Ces chemins sont déclarés DELEGATED : le schéma dit qui les garde, il ne les
duplique pas.

Anti-dérive. Le self-test relit le SOURCE des trainers en `ast`, extrait les
clés réellement lues (`t.get("…")`, `t["…"]`, y compris via `self.r`), et exige
que le schéma vaille exactement ça pour `code_defer_native` (et couvre au moins
ça pour les autres entrées). Ajouter une clé au trainer sans la déclarer ici
fait échouer `scripts/selftest.sh` — le schéma ne peut pas mentir en silence.

    python -m deepseek_v4_mini.cfg_schema           # self-test + valide les configs
    python -m deepseek_v4_mini.cfg_schema --dump    # schéma tel qu'extrait du source
"""
from __future__ import annotations

import ast
import difflib
from pathlib import Path

_PKG = Path(__file__).resolve().parent


class ConfigError(ValueError):
    """Config refusée : clé/section inconnue, ou clé obligatoire absente."""


# ── schémas ──────────────────────────────────────────────────────────────────
# entrée → section → (obligatoires, optionnelles). "" = la racine du YAML.
# Une section absente de ce dict n'est pas vérifiée en profondeur (voir
# DELEGATED, qui dit pourquoi). Les listes viennent de l'extraction ast
# ci-dessous : `--dump` régénère, le self-test compare.

SCHEMAS: dict[str, dict[str, tuple[set[str], set[str]]]] = {
    "code_defer_native": {
        "": ({"tokenizer", "model", "data", "training"}, {"chat", "teacher"}),
        "training": (
            {"save_dir", "steps"},
            {"adam_fused", "addr_label", "addr_max", "addr_prob", "allreduce_bf16",
             "amp", "bank_init", "cascade_depth", "cascade_map", "chunk_budget",
             "compile", "compile_cache_dir", "decode_cache", "decode_graphs",
             "defer_weight", "delta_channel",
             "eval_convs", "eval_depth_convs", "eval_depth_sources", "eval_depths",
             "eval_every", "eval_sources", "gc_save_topk", "grad_accum",
             "grad_checkpoint", "grad_clip", "init_from", "interleave_files",
             "log_every", "lr", "metrics_file", "muon_lr", "muon_ref_mem_dim",
             "no_reset_files", "prefetch", "reach_cue_len", "reach_max",
             "reach_prob", "reset_announce", "reset_announce_chunks", "save_every",
             "seed", "skip_final_eval", "tf32", "warmup_steps", "weight_decay",
             "wsd_decay", "wsd_decay_shape", "wsd_decay_start", "wsd_floor",
             "wsd_stair_end", "wsd_stair_n"},
        ),
        "data": (
            {"batch_size", "chunks_per_conv", "seq_len"},
            {"cache_dir", "config_name", "content_key", "data_dir", "dataset",
             "defer_len", "depth_sync", "min_chunks", "n_files", "pack_convs",
             "pack_same_source", "sif_a", "sources", "stream_cap", "stream_skip",
             "surprisal_mode", "var_chunk"},
        ),
        "chat": (set(),
                 {"decode_every", "eval_convs", "eval_gen", "gen", "max_new",
                  "p_chat", "stream", "weight"}),
        "teacher": (set(), {"anneal", "distill_weight", "enabled", "target"}),
    },
    "rl_disagg": {
        "": ({"tokenizer", "model", "data", "rl"}, set()),
        "data": ({"chunks_per_conv", "envs", "seq_len"},
                 {"cache_dir", "defer_len", "var_chunk"}),
        "rl": (
            {"disagg", "init_from", "steps"},
            {"amp", "cascade_depth", "cascade_map", "clip_high", "clip_low",
             "decode_cache", "decode_graphs", "explore_floor", "grad_clip",
             "group_size",
             "groups_per_step", "kl_coef",
             "lambda_write", "lr", "max_episodes_per_life", "max_new",
             "max_resample", "min_reward_std", "n_lives_per_worker", "save_every",
             "seed", "temp", "think_floor", "think_id", "think_nmax",
             "train_scope"},
        ),
        "rl.disagg": (
            {"root"},
            {"keep_weights", "lives_save_every", "max_groups", "max_lag",
             "max_pending", "poll_s", "publish_every", "xdom_every"},
        ),
    },
    "rl_defer_grpo_lives": {
        "": ({"tokenizer", "model", "data", "rl"}, set()),
        "data": ({"chunks_per_conv", "envs", "seq_len"},
                 {"cache_dir", "defer_len", "var_chunk"}),
        "rl": (
            {"init_from", "steps"},
            {"cascade_depth", "cascade_map", "clip_high", "clip_low", "eval_convs",
             "eval_every", "grad_clip", "group_size", "groups_per_step", "kl_coef",
             "lambda_write", "log_every", "lr", "max_episodes_per_life",
             "max_resample", "metrics_file", "min_reward_std", "n_lives",
             "save_dir", "save_every", "seed", "temp", "train_scope"},
        ),
    },
    "rl_defer_grpo": {
        "": ({"tokenizer", "model", "data", "rl"}, set()),
        "rl": (
            {"init_from", "steps"},
            {"clip_high", "clip_low", "eval_convs", "eval_every", "grad_clip",
             "group_size", "groups_per_step", "kl_coef", "lambda_write",
             "log_every", "lr", "max_resample", "metrics_file", "min_reward_std",
             "save_dir", "save_every", "seed", "temp", "think_id", "train_scope"},
        ),
    },
}

# Sous-blocs volontairement non vérifiés ici, avec leur gardien réel.
DELEGATED: dict[str, str] = {
    "model":            "ThoughtBankConfig(**raw['model']) — lève sur clé inconnue",
    "chat.gen":         "Cls(tok, seed=..., **gen) — lève sur kwarg inconnu",
    "data.sources":     "liste de sources, chacune consommée par code_data._load_source",
    "data.envs":        "liste d'envs RL, chacune consommée par rl_lives.build_envs",
    "data.var_chunk":   "bloc passé tel quel à CodeChunkStream",
    "training.delta_channel": "bloc passé tel quel à DeltaChannel",
}


def entry_for(raw: dict) -> str:
    """Point d'entrée qui consomme cette config, déduit de sa forme."""
    rl = raw.get("rl")
    if not isinstance(rl, dict):
        return "code_defer_native"
    if "disagg" in rl:
        return "rl_disagg"
    if "max_episodes_per_life" in rl or "n_lives" in rl:
        return "rl_defer_grpo_lives"
    return "rl_defer_grpo"


def _suggest(key: str, known) -> str:
    near = difflib.get_close_matches(key, sorted(known), n=1, cutoff=0.7)
    return f" — vouliez-vous {near[0]!r} ?" if near else ""


def _dig(raw: dict, dotted: str):
    """Sous-dict à `dotted` ('' = racine), ou None s'il est absent/pas un dict."""
    node = raw
    for part in (p for p in dotted.split(".") if p):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node if isinstance(node, dict) else None


def check(raw: dict, entry: str) -> None:
    """Valide `raw` contre le schéma de `entry`. Lève ConfigError au premier
    problème, en nommant la section, la clé et la voisine la plus proche."""
    try:
        schema = SCHEMAS[entry]
    except KeyError:
        raise ConfigError(
            f"point d'entrée inconnu {entry!r} "
            f"(connus : {' | '.join(sorted(SCHEMAS))})") from None

    for dotted, (required, optional) in schema.items():
        where = f"section {dotted!r}" if dotted else "racine de la config"
        node = _dig(raw, dotted)
        if node is None:
            if dotted == "" or dotted in schema[""][0]:
                raise ConfigError(f"{where} absente ou vide")
            continue                        # section optionnelle non fournie
        known = required | optional
        for key in node:
            if key not in known:
                path = f"{dotted}.{key}" if dotted else key
                raise ConfigError(
                    f"clé inconnue {path!r} ({where}){_suggest(key, known)}")
        missing = sorted(required - set(node))
        if missing:
            raise ConfigError(
                f"clé(s) obligatoire(s) absente(s) dans {where} : "
                f"{', '.join(missing)}")


# ── extraction ast : ce que le source lit VRAIMENT ───────────────────────────
# Objectif : rendre la dérive impossible. On suit, par fonction, les variables
# liées à un morceau de config (`t = raw["training"]`), y compris quand elles
# transitent par un attribut (`self.r, self.d = r, d` dans rl_disagg), et on
# ignore celles dont le nom est réutilisé pour autre chose (`for r in recs`).

ENTRY_SOURCE: dict[str, tuple[str, dict[str, str]]] = {
    # entrée → (fichier, {nom de variable racine: section qu'elle porte})
    "code_defer_native":   ("code_defer_native.py",   {"raw": ""}),
    "rl_disagg":           ("rl_disagg.py",           {"raw": ""}),
    "rl_defer_grpo_lives": ("rl_defer_grpo_lives.py", {"raw": ""}),
    "rl_defer_grpo":       ("rl_defer_grpo.py",       {"raw": ""}),
}


def _target_names(node) -> list[str]:
    """Noms liés par une cible d'affectation ('x', 'self.x', tuples inclus)."""
    if isinstance(node, ast.Name):
        return [node.id]
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
        return [f"{node.value.id}.{node.attr}"]
    if isinstance(node, (ast.Tuple, ast.List)):
        return [n for e in node.elts for n in _target_names(e)]
    return []


Env = dict          # nom → [(ligne, section ou None)], trié par ligne


def _section_at(env: Env, name: str, line: int) -> str | None:
    """Section portée par `name` à cette ligne : la dernière liaison en vigueur.

    Résoudre par ligne plutôt que par nom global est ce qui rend l'extraction
    utilisable sur du vrai code : `code_defer_native.main` lie `d = raw["data"]`
    ligne 310 puis réutilise `d` comme variable de boucle ligne 1401. Les deux
    coexistent sans qu'on ait à renommer quoi que ce soit.
    """
    sec = None
    for ln, s in env.get(name, ()):
        if ln > line:
            break
        sec = s
    return sec


def _source_section(node, env: Env, line: int | None = None) -> str | None:
    """Section portée par une expression, si elle dérive d'une variable suivie.

    `raw["training"]` → 'training' ; `r.get("disagg")` → 'rl.disagg' ; `r` → 'rl'.
    """
    line = getattr(node, "lineno", line) if line is None else line
    if line is None:
        return None
    if isinstance(node, ast.Name):
        return _section_at(env, node.id, line)
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
        return _section_at(env, f"{node.value.id}.{node.attr}", line)
    if isinstance(node, ast.BoolOp):                   # `raw.get("chat") or {}`
        for v in node.values:
            s = _source_section(v, env, line)
            if s is not None:
                return s
        return None
    if isinstance(node, ast.Call):                     # X.get("k") / dict(X)
        f = node.func
        # `raw = load_yaml(cfg)` : l'entonnoir unique des configs (paths.py)
        # produit la racine — sans ça, il écraserait la graine `raw`.
        if (isinstance(f, ast.Name) and f.id == "load_yaml") or \
                (isinstance(f, ast.Attribute) and f.attr == "load_yaml"):
            return ""
        if isinstance(f, ast.Attribute) and f.attr == "get" and node.args:
            base = _source_section(f.value, env, line)
            if base is not None and isinstance(node.args[0], ast.Constant) \
                    and isinstance(node.args[0].value, str):
                return f"{base}.{node.args[0].value}".lstrip(".")
        if isinstance(f, ast.Name) and f.id == "dict" and len(node.args) == 1:
            return _source_section(node.args[0], env, line)
        return None
    if isinstance(node, ast.Subscript):                # X["k"]
        base = _source_section(node.value, env, line)
        if base is not None and isinstance(node.slice, ast.Constant) \
                and isinstance(node.slice.value, str):
            return f"{base}.{node.slice.value}".lstrip(".")
    return None


def _scope_env(scope, seed: dict[str, str]) -> Env:
    """Liaisons de config du scope, datées, résolues en une passe source-ordre.

    Une liaison vers autre chose qu'une expression de config est enregistrée
    comme None : elle éteint la variable à partir de sa ligne.
    """
    env: Env = {k: [(0, v)] for k, v in seed.items()}
    binds = []
    for n in ast.walk(scope):
        if isinstance(n, ast.Assign):
            binds += [(n.lineno, t, n.value) for t in n.targets]
        elif isinstance(n, (ast.For, ast.AsyncFor)):
            binds.append((n.lineno, n.target, None))
        elif isinstance(n, ast.comprehension):
            binds.append((getattr(n.target, "lineno", 0), n.target, None))
    for line, tgt, val in sorted(binds, key=lambda b: b[0]):
        pairs = [(tgt, val)]
        if isinstance(tgt, (ast.Tuple, ast.List)) and isinstance(
                val, (ast.Tuple, ast.List)) and len(tgt.elts) == len(val.elts):
            pairs = list(zip(tgt.elts, val.elts))      # a, b = x, y
        for sub_t, sub_v in pairs:
            sec = _source_section(sub_v, env, line) if sub_v is not None else None
            for nm in _target_names(sub_t):
                env.setdefault(nm, []).append((line, sec))
    return env


def sections_read(entry: str) -> dict[str, tuple[set[str], set[str]]]:
    """Clés réellement lues par le source de `entry` : {section: (req, opt)}."""
    fname, seed = ENTRY_SOURCE[entry]
    tree = ast.parse((_PKG / fname).read_text())
    req: dict[str, set[str]] = {}
    opt: dict[str, set[str]] = {}

    scopes = [tree] + [n for n in ast.walk(tree)
                       if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef,
                                         ast.ClassDef))]
    # Les configs portées par un attribut (`self.r, self.d = r, d` dans
    # rl_disagg) sont liées dans __init__ et relues dans d'autres méthodes :
    # on les prend au scope module, où toutes les méthodes sont visibles.
    module_env = _scope_env(tree, seed)
    attrs = {k: v for k, v in module_env.items() if "." in k}
    for scope in scopes:
        env = _scope_env(scope, seed) if scope is not tree else module_env
        if scope is not tree:
            env = {**attrs, **env}
        for n in ast.walk(scope):
            if isinstance(n, ast.Subscript) and isinstance(n.slice, ast.Constant) \
                    and isinstance(n.slice.value, str):
                sec = _source_section(n.value, env)
                if sec is not None:
                    req.setdefault(sec, set()).add(n.slice.value)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) \
                    and n.func.attr == "get" and n.args \
                    and isinstance(n.args[0], ast.Constant) \
                    and isinstance(n.args[0].value, str):
                sec = _source_section(n.func.value, env)
                if sec is not None:
                    opt.setdefault(sec, set()).add(n.args[0].value)

    out = {}
    for sec in set(req) | set(opt):
        r = req.get(sec, set())
        out[sec] = (r, opt.get(sec, set()) - r)
    return out


# ── self-test ────────────────────────────────────────────────────────────────

def _fmt(schema) -> str:
    lines = []
    for sec in sorted(schema):
        r, o = schema[sec]
        lines.append(f"  {sec or '<racine>'}:")
        lines.append(f"    obligatoires : {sorted(r)}")
        lines.append(f"    optionnelles : {sorted(o)}")
    return "\n".join(lines)


def _selftest() -> None:
    import sys
    import yaml

    # 1. le schéma déclaré == ce que le trainer principal lit, exactement.
    read = sections_read("code_defer_native")
    decl = SCHEMAS["code_defer_native"]
    for sec in set(decl) | {s for s in read if s in decl}:
        r_dec, o_dec = decl[sec]
        r_read, o_read = read.get(sec, (set(), set()))
        assert r_dec == r_read, (
            f"[code_defer_native/{sec}] obligatoires : déclaré {sorted(r_dec)} "
            f"≠ lu {sorted(r_read)}")
        assert o_dec == o_read, (
            f"[code_defer_native/{sec}] optionnelles : "
            f"en trop {sorted(o_dec - o_read)} / manquantes {sorted(o_read - o_dec)}")

    # 2. les autres entrées : le schéma doit COUVRIR ce que le source lit
    #    (l'extracteur ne voit pas tout quand la config transite par des
    #    structures intermédiaires ; il ne doit jamais voir plus que déclaré).
    for entry in ("rl_disagg", "rl_defer_grpo_lives", "rl_defer_grpo"):
        for sec, (r_read, o_read) in sections_read(entry).items():
            if sec not in SCHEMAS[entry] or sec in DELEGATED:
                continue
            known = set().union(*SCHEMAS[entry][sec])
            extra = (r_read | o_read) - known
            assert not extra, f"[{entry}/{sec}] lues mais non déclarées : {sorted(extra)}"

    # 3. check() : accepte le bon, refuse l'inconnu, exige l'obligatoire.
    good = {"tokenizer": "tok", "model": {"d_model": 8}, "training": {"steps": 1,
            "save_dir": "/tmp/x", "wsd_decay_start": 3}, "data": {"seq_len": 8,
            "chunks_per_conv": 2, "batch_size": 1}}
    check(good, "code_defer_native")
    for broken, needle in (
            ({**good, "training": {**good["training"], "wsd_decay_strat": 3}},
             "wsd_decay_start"),                       # la suggestion doit tomber juste
            ({**good, "trainning": {}}, "trainning"),  # section inconnue
            ({k: v for k, v in good.items() if k != "data"}, "data"),
    ):
        try:
            check(broken, "code_defer_native")
        except ConfigError as e:
            assert needle in str(e), f"message peu utile : {e}"
        else:
            raise AssertionError(f"aurait dû lever : {needle}")
    try:
        check(good, "nope")
    except ConfigError:
        pass
    else:
        raise AssertionError("entrée inconnue : aurait dû lever")

    # 4. toutes les configs actives passent (le test que la CI rejoue).
    cfgs = sorted((_PKG / "configs").glob("*.yaml"))
    bad = []
    for p in cfgs:
        raw = yaml.safe_load(p.read_text()) or {}
        try:
            check(raw, entry_for(raw))
        except ConfigError as e:
            bad.append(f"{p.name} [{entry_for(raw)}] : {e}")
    if bad:
        print("\n".join("  ✗ " + b for b in bad), file=sys.stderr)
        raise AssertionError(f"{len(bad)}/{len(cfgs)} configs actives invalides")

    print(f"cfg_schema self-test: OK ({len(SCHEMAS)} entrées, "
          f"{sum(len(s) for s in SCHEMAS.values())} sections, "
          f"{len(cfgs)} configs actives valides, anti-dérive ast vert)")


if __name__ == "__main__":
    import sys
    if "--dump" in sys.argv:
        for entry in ENTRY_SOURCE:
            print(f"\n=== {entry} (lu dans le source)")
            print(_fmt(sections_read(entry)))
    else:
        _selftest()
