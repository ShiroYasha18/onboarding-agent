from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app import engine
from app.graph import app_graph
from app.state import OnboardingState
from app.storage import load_state, save_state

app = FastAPI()

QUESTIONS = engine.get_questions()
QUESTIONS_BY_ID = {q["id"]: q for q in QUESTIONS}
_ALL_STEP_IDS = set(QUESTIONS_BY_ID.keys())

_static_dir = Path(__file__).parent / "static"
if _static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")


def _load_dotenv() -> None:
    env_path = Path(__file__).with_name(".env")
    if not env_path.exists():
        return
    try:
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            if not key or key in os.environ:
                continue
            value = value.strip().strip("'").strip('"')
            if value:
                os.environ[key] = value
    except Exception:
        return


_load_dotenv()

def _load_flow() -> dict[str, Any]:
    flow_path = Path(__file__).with_name("flow.json")
    if not flow_path.exists():
        return {}
    try:
        flow = json.loads(flow_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return flow if isinstance(flow, dict) else {}


FLOW = _load_flow()

def _build_phases() -> list[dict[str, Any]]:
    raw = FLOW.get("phases")
    if not isinstance(raw, list):
        return []
    phases: list[dict[str, Any]] = []
    for p in raw:
        if not isinstance(p, dict):
            continue
        pid = p.get("id")
        if not isinstance(pid, str) or not pid:
            continue
        phases.append(
            {
                "id": pid,
                "label": str(p.get("label") or pid),
                "description": str(p.get("description") or ""),
                "order": int(p.get("order") or 0),
            }
        )
    phases.sort(key=lambda x: (x.get("order", 0), str(x.get("id", ""))))
    return phases


def _build_groups() -> list[dict[str, Any]]:
    raw_questions = FLOW.get("questions")
    if not isinstance(raw_questions, list):
        return []

    groups: list[dict[str, Any]] = []
    for q in raw_questions:
        if not isinstance(q, dict):
            continue
        group_id = q.get("id")
        if not isinstance(group_id, str) or not group_id:
            continue
        phase_id = q.get("phase_id") if isinstance(q.get("phase_id"), str) else None
        group = {
            "id": group_id,
            "label": str(q.get("label") or group_id),
            "order": int(q.get("order") or 0),
            "optional": bool(q.get("optional")),
            "repeatable": bool(q.get("repeatable")),
            "phase_id": phase_id,
            "step_ids": [],
        }

        step_ids: list[str] = []
        if "field" in q and group_id in _ALL_STEP_IDS:
            step_ids = [group_id]
        else:
            fields = q.get("fields")
            if isinstance(fields, dict):
                for field_name in fields.keys():
                    if not isinstance(field_name, str) or not field_name:
                        continue
                    candidate = f"{group_id}.{field_name}"
                    if candidate in _ALL_STEP_IDS:
                        step_ids.append(candidate)
                        continue
                    if group_id == "hotel_contact_persons":
                        primary = f"hotel_contact_persons.primary.{field_name}"
                        if primary in _ALL_STEP_IDS:
                            step_ids.append(primary)

                if group_id == "price_list" and "price_list.items" in _ALL_STEP_IDS:
                    step_ids = ["price_list.items"]

            if group_id == "shipping_address" and "shipping_address.same_as_billing" in _ALL_STEP_IDS:
                step_ids = ["shipping_address.same_as_billing", *step_ids]

            if group_id == "inventory_pool" and "inventory_pool.want_pool" in _ALL_STEP_IDS:
                rest = [sid for sid in _ALL_STEP_IDS if sid.startswith("inventory_pool.") and sid != "inventory_pool.want_pool"]
                rest.sort(key=lambda s: engine.FLOW.index(s) if s in engine.FLOW else 10_000)
                step_ids = ["inventory_pool.want_pool", *rest]

        seen: set[str] = set()
        for sid in step_ids:
            if sid in _ALL_STEP_IDS and sid not in seen:
                seen.add(sid)
                group["step_ids"].append(sid)

        if group["step_ids"]:
            groups.append(group)

    groups.sort(key=lambda g: (g.get("order", 0), str(g.get("id", ""))))
    return groups


PHASES = _build_phases()
GROUPS = _build_groups()

_PHASE_BY_ID = {p["id"]: p for p in PHASES if isinstance(p, dict) and p.get("id")}


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    index_path = _static_dir / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Frontend not built")
    return HTMLResponse(index_path.read_text(encoding="utf-8"))


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/questions")
def questions() -> dict[str, Any]:
    return {"questions": QUESTIONS, "groups": GROUPS, "phases": PHASES}


@app.get("/flow")
def flow() -> dict[str, Any]:
    return {"groups": GROUPS, "phases": PHASES}


def _prompt_for(state: OnboardingState, step_id: str) -> str:
    return engine.prompt_for(state, step_id)

def _format_value(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False)
    except Exception:
        return str(value)


def _review_summary(state: OnboardingState) -> str:
    data = {k: v for k, v in (state.data or {}).items() if isinstance(k, str) and not k.startswith("_")}
    if not data:
        return "Review summary:\n(no details captured)"

    lines: list[str] = ["Review summary:"]
    if GROUPS:
        for g in GROUPS:
            step_ids = g.get("step_ids") or []
            picked = [sid for sid in step_ids if sid in data]
            if not picked:
                continue
            lines.append(str(g.get("label") or g.get("id") or "Details") + ":")
            for sid in picked:
                q = QUESTIONS_BY_ID.get(sid, {})
                label = str(q.get("label") or q.get("prompt") or sid)
                lines.append(f"- {label}: {_format_value(data[sid])}")
    else:
        for sid in sorted(data.keys()):
            q = QUESTIONS_BY_ID.get(sid, {})
            label = str(q.get("label") or q.get("prompt") or sid)
            lines.append(f"- {label}: {_format_value(data[sid])}")

    return "\n".join(lines)

def _phase_intro() -> str:
    if not PHASES:
        return "Hey, I’m LG. Welcome to the onboarding interface."
    lines: list[str] = ["Hey, I’m LG. Welcome to the onboarding interface.", "", "There are 4 phases:"]
    for p in PHASES:
        label = str(p.get("label") or p.get("id") or "")
        desc = str(p.get("description") or "").strip()
        if desc:
            lines.append(f"- {label}: {desc}")
        else:
            lines.append(f"- {label}")
    return "\n".join(lines)


@app.post("/start")
def start(tenant_id: str = "default") -> dict[str, Any]:
    session_id = str(uuid.uuid4())
    first_step = QUESTIONS[0]["id"]

    state = OnboardingState(tenant_id=tenant_id, session_id=session_id, current_step=first_step)
    state = engine.resolve_next_step(state)
    save_state(session_id, state)

    intro = _phase_intro()
    prompt = _prompt_for(state, state.current_step)
    reply = f"{intro}\n\n{prompt}" if intro else prompt

    return {
        "session_id": session_id,
        "current_step": state.current_step,
        "reply": reply,
        "data": state.data,
        "completed_steps": state.completed_steps,
        "completed": state.completed,
    }


class ChatRequest(BaseModel):
    session_id: str
    message: str | None = None
    action: Literal["message", "set"] = "message"
    step_id: str | None = None
    value: Any | None = None


@app.post("/chat")
def chat(req: ChatRequest) -> dict[str, Any]:
    try:
        state = load_state(req.session_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Session not found")

    if req.action == "set":
        if not req.step_id:
            raise HTTPException(status_code=400, detail="step_id is required for set")
        state.last_intent = {
            "intent": "correct_step",
            "step_id": req.step_id,
            "value": {req.step_id: req.value},
            "confidence": 1.0,
        }
        state.intent_source = "ui"
        state.last_user_message = None
    else:
        state.last_user_message = req.message or ""
        state.last_intent = None
        state.intent_source = None

    next_state = app_graph.invoke(state)
    if not isinstance(next_state, OnboardingState):
        next_state = OnboardingState.model_validate(next_state)

    next_state.last_user_message = None
    next_state.last_intent = None
    save_state(req.session_id, next_state)

    if next_state.last_error:
        prompt = "" if next_state.completed else _prompt_for(next_state, next_state.current_step)
        reply = next_state.last_error if not prompt else f"{next_state.last_error}\n{prompt}"
    elif next_state.confirmed:
        reply = "Confirmed."
    elif next_state.completed:
        reply = _review_summary(next_state) + "\n\nReply 'confirm' to finish, or tell me what to change."
    else:
        reply = _prompt_for(next_state, next_state.current_step)

    return {
        "reply": reply,
        "current_step": next_state.current_step,
        "data": next_state.data,
        "completed_steps": next_state.completed_steps,
        "completed": next_state.completed,
        "intent_source": next_state.intent_source,
    }
