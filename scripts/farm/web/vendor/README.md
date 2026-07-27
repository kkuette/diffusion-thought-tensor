# Dépendance tierce vendorée

Un seul fichier, et c'est la seule dépendance tierce du dépôt. Il est **commité**
volontairement : la VM data qui sert le dashboard n'a pas d'accès réseau, et le dépôt
n'a ni npm, ni bundler, ni étape de build.

| | |
|---|---|
| fichier | `htm-preact-standalone-3.1.1.mjs` |
| source  | `https://unpkg.com/htm@3.1.1/preact/standalone.module.js` |
| taille  | 13 194 octets |
| sha256  | `72284e8e9079c87817145df1110f74e8a2aa040b2fc384922e18dfcb46fc1fd7` |
| contenu | preact 10 + preact/hooks + htm, en un module ES |
| exports | `html, render, h, Component, createContext, useState, useReducer, useEffect, useLayoutEffect, useRef, useImperativeHandle, useMemo, useCallback, useContext, useDebugValue, useErrorBoundary` |

Le sha256 est **revérifié par le self-test** de `farm_dashboard.py` : une substitution
ou une édition accidentelle casse la CI au lieu de partir en production.

Audité à l'entrée : le paquet ne contient ni `fetch`, ni `XMLHttpRequest`, ni
`WebSocket`, ni `eval`/`new Function`, ni accès à `localStorage` ou aux cookies. La
seule URL qu'il porte est le namespace SVG `http://www.w3.org/2000/svg`.

## Le remplacer

```bash
V=3.1.1
curl -sL "https://unpkg.com/htm@$V/preact/standalone.module.js" \
  -o scripts/farm/web/vendor/htm-preact-standalone-$V.mjs
sha256sum scripts/farm/web/vendor/htm-preact-standalone-$V.mjs
```

Puis reporter la nouvelle version et le nouveau sha256 dans `VENDOR` /
`VENDOR_SHA256` de [`../../farm_dashboard.py`](../../farm_dashboard.py), dans l'import
de [`../dashboard.mjs`](../dashboard.mjs), et ici.
