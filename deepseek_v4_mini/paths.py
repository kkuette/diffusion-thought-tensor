"""Expansion des variables d'environnement dans les configs YAML.

Les chemins de campagne (cache de données, checkpoints, metrics) pointaient
un montage de machine en dur. Une config comme ça ne démarre pas
chez quelqu'un d'autre, alors que le but du dépôt est que les claims soient
reproductibles. Elles s'écrivent désormais relativement à `${TB_ROOT}`, qui
vaut `.` par défaut : sans rien configurer, un run écrit dans le dépôt.

    data:    {cache_dir: ${TB_ROOT}/data_cache}
    logging: {save_dir: ${TB_ROOT}/checkpoints/v350_phase1}

    $ python -m deepseek_v4_mini.code_defer_native cfg.yaml      # → ./data_cache
    $ TB_ROOT=/mnt/big_volume python -m deepseek_v4_mini.code_defer_native cfg.yaml

Syntaxe : `${VAR}` et `${VAR:-défaut}`. Une variable inconnue et sans défaut
est laissée telle quelle plutôt qu'effacée — un chemin visiblement faux vaut
mieux qu'un run qui écrit silencieusement au mauvais endroit.
"""
from __future__ import annotations

import os
import re

_VAR = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")

# Variables ayant un défaut implicite : pas besoin d'écrire ${TB_ROOT:-.}
_DEFAULTS = {"TB_ROOT": "."}


def expand_str(s: str) -> str:
    def sub(m: re.Match) -> str:
        name, default = m.group(1), m.group(2)
        val = os.environ.get(name)
        if val is not None:
            return val
        if default is not None:
            return default
        if name in _DEFAULTS:
            return _DEFAULTS[name]
        return m.group(0)                     # inconnue : laissée visible
    return _VAR.sub(sub, s)


def expand(obj):
    """Expansion récursive sur toute chaîne d'une structure YAML."""
    if isinstance(obj, str):
        return expand_str(obj)
    if isinstance(obj, dict):
        return {k: expand(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [expand(v) for v in obj]
    return obj


def load_yaml(path):
    """yaml.safe_load + expansion. Le point d'entrée unique des configs."""
    import yaml
    with open(path) as f:
        return expand(yaml.safe_load(f) or {})


if __name__ == "__main__":
    import os as _os
    _os.environ.pop("TB_ROOT", None)
    assert expand_str("${TB_ROOT}/data_cache") == "./data_cache"
    _os.environ["TB_ROOT"] = "/vol"
    assert expand_str("${TB_ROOT}/data_cache") == "/vol/data_cache"
    assert expand_str("${NOPE:-fallback}/x") == "fallback/x"
    assert expand_str("${TOTALLY_UNSET}/x") == "${TOTALLY_UNSET}/x"
    assert expand({"a": ["${TB_ROOT}/c", 3], "b": {"d": "${TB_ROOT}"}}) == \
        {"a": ["/vol/c", 3], "b": {"d": "/vol"}}
    _os.environ.pop("TB_ROOT")
    assert expand(None) is None and expand(7) == 7
    print("paths self-test: OK (défaut local, override env, var inconnue "
          "préservée, récursion)")
