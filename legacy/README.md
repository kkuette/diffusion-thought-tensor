# `legacy/` — code d'avant le programme

Ce qui est ici n'est pas mort : c'est daté. Rien n'y est un point d'entrée du
programme courant, et rien n'y est à relancer tel quel.

## `thought_lm_minimal/` — l'ancêtre (2025-09)

Premier jet du « thought tensor » sur un LM autorégressif, antérieur à
`deepseek_v4_mini` et sans lien de code avec lui : paquet autonome, ses
propres configs, son propre `pyproject.toml`. Aucun module du programme
courant ne l'importe, aucune de ses conclusions n'est citée dans FINDINGS.md.

Il est gardé parce que c'est le point de départ historique de l'idée
(cf. la note d'origine « diffusion thought tensor »). Ses commandes tournent
depuis son propre dossier :

```bash
cd legacy/thought_lm_minimal && python -m thought_lm.train configs/default.yaml
```

## La greffe SmolLM2

Elle vit dans `deepseek_v4_mini/legacy/` (et pas ici) parce qu'elle importe
encore le paquet — `mhc`, `memory`, `code_data`, `muon`. Voir le docstring de
`deepseek_v4_mini/legacy/__init__.py` pour la table de correspondance des
commandes.
