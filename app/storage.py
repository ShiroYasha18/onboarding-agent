import os

from app.state import OnboardingState

_memory: dict[str, str] = {}


def save_state(session_id: str, state: OnboardingState) -> None:
    payload = state.model_dump_json()
    client = _redis_client()
    if client is None:
        _memory[session_id] = payload
        return
    try:
        client.set(session_id, payload)
    except Exception:
        _memory[session_id] = payload


def load_state(session_id: str) -> OnboardingState:
    client = _redis_client()
    if client is None:
        payload = _memory.get(session_id)
        if not payload:
            raise ValueError("Session not found")
        return OnboardingState.model_validate_json(payload)

    try:
        raw = client.get(session_id)
    except Exception:
        raw = None

    if not raw:
        payload = _memory.get(session_id)
        if not payload:
            raise ValueError("Session not found")
        return OnboardingState.model_validate_json(payload)

    return OnboardingState.model_validate_json(raw)


def _redis_client():
    redis_url = os.getenv("REDIS_URL")
    if not redis_url:
        return None
    try:
        import redis  # type: ignore

        return redis.Redis.from_url(redis_url, decode_responses=True)
    except Exception:
        return None
