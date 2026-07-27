#!/usr/bin/env python
"""Importe tout le paquet et rapporte ce qui casse.

Le moins cher des tests : il attrape les imports morts, les cycles et les
symboles renommés ailleurs — la classe de panne qui, sur un pod, se manifeste
après l'allocation du GPU. Aucun réseau, aucun GPU, ~2 s.

    python scripts/import_smoke.py
"""
from __future__ import annotations

import importlib
import pkgutil
import sys
import traceback
from pathlib import Path

# Lancé par chemin (`python scripts/import_smoke.py`), sys.path[0] est scripts/,
# pas la racine du dépôt : sans ça le paquet est introuvable en CI.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> int:
    import deepseek_v4_mini as pkg

    names = [m.name for m in pkgutil.walk_packages(pkg.__path__, pkg.__name__ + ".")]
    failures = []
    for name in sorted(names):
        try:
            importlib.import_module(name)
        except Exception:
            failures.append((name, traceback.format_exc()))

    for name, tb in failures:
        print(f"\n=== ÉCHEC {name}\n{tb}", file=sys.stderr)
    verdict = "TOUT VERT" if not failures else f"{len(failures)} ÉCHEC(S)"
    print(f"import smoke: {verdict} ({len(names)} modules)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
