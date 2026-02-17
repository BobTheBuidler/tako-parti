from __future__ import annotations

from .model import EngineId

PUBLIC_ENGINE_ID: EngineId = "robob"
INTERNAL_ENGINE_ID: EngineId = "codex"
ENGINE_DIRECTIVE_IDS: tuple[EngineId, ...] = (PUBLIC_ENGINE_ID,)


def resolve_engine_directive(engine: EngineId | None) -> EngineId | None:
    if engine is None:
        return None
    if engine.lower() == PUBLIC_ENGINE_ID:
        return INTERNAL_ENGINE_ID
    return None
