"""Registre unique des streams de conversation.

Le mapping nom→classe existait en trois exemplaires (chat_mix, le bloc `chat:`
de code_defer_native, la construction d'envs de rl_disagg) et ils avaient
divergé : `code_exec` et `tool_session` n'étaient atteignables depuis le
trainer qu'en passant par `chat_mix`. Ici il n'y en a plus qu'un ; ajouter un
stream = ajouter une ligne à CHAT_STREAMS.

Contrat d'un stream (ce que le trainer et les rollouts RL attendent) :
  - `next_conv() -> {"segs": [...], "info": {...}, "kind": str}`
  - `.rng` avec getstate/setstate/seed (déterminisme au resume)
  - `grade_conv(conv, texts) -> float` (optionnel : eval décodée)
Le constructeur est uniforme : `Cls(tok, seed=..., **gen)`.

L'import est différé (lazy) : charger le registre ne tire pas datasets/
transformers, et chat_mix peut s'y référencer sans import circulaire.
"""
from __future__ import annotations

from importlib import import_module

# nom de config → (module du paquet, classe)
CHAT_STREAMS: dict[str, tuple[str, str]] = {
    "sota_session": ("sota_session_data", "SotaSessionStream"),
    "tool_session": ("tool_env_data",     "ToolSessionStream"),
    "code_exec":    ("code_exec_data",    "CodeExecStream"),
    "persona":      ("persona_chat_data", "PersonaChatStream"),
    "math_school":  ("math_school_data",  "MathSchoolStream"),
    "chat_mix":     ("chat_mix",          "ChatMixStream"),
}

# rl_disagg nomme ses envs par `kind:` (historique, et les configs RL déjà
# écrites l'utilisent) ; on traduit plutôt que de renommer les clés existantes.
# `code` n'est pas ici : CodeChunkStream a une autre forme de constructeur et
# reste construit sur place dans rl_disagg.
RL_ENV_KIND_TO_STREAM: dict[str, str] = {
    "tool": "tool_session",
    "exec": "code_exec",
    "sota": "sota_session",
}


def _known() -> str:
    return " | ".join(sorted(CHAT_STREAMS))


def chat_stream_class(name: str) -> type:
    """Classe du stream `name` (import différé)."""
    try:
        mod, cls = CHAT_STREAMS[name]
    except KeyError:
        raise ValueError(f"stream inconnu {name!r} (connus : {_known()})") from None
    return getattr(import_module(f".{mod}", __package__), cls)


def build_chat_stream(name: str, tok, *, seed: int, gen: dict | None = None):
    """Instancie le stream `name` avec le contrat uniforme (tok, seed, **gen)."""
    return chat_stream_class(name)(tok, seed=seed, **(gen or {}))


def rl_stream_class(kind: str) -> type:
    """Classe du stream derrière un `kind:` d'env RL (hors `code`)."""
    try:
        name = RL_ENV_KIND_TO_STREAM[kind]
    except KeyError:
        raise ValueError(
            f"env kind {kind!r} sans stream chat (connus : code | "
            f"{' | '.join(sorted(RL_ENV_KIND_TO_STREAM))})") from None
    return chat_stream_class(name)


if __name__ == "__main__":  # self-test : le registre résout, sans construire
    for _n in CHAT_STREAMS:
        assert isinstance(chat_stream_class(_n), type), _n
    for _k in RL_ENV_KIND_TO_STREAM:
        assert isinstance(rl_stream_class(_k), type), _k
    for _bad, _fn in (("nope", chat_stream_class), ("code", rl_stream_class)):
        try:
            _fn(_bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"{_fn.__name__}({_bad!r}) aurait dû lever")
    print(f"streams self-test: OK ({len(CHAT_STREAMS)} streams, "
          f"{len(RL_ENV_KIND_TO_STREAM)} kinds RL)")
