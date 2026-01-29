from __future__ import annotations

import json
import os
import re
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
    def load_file(path: Path) -> None:
        if not path.exists():
            return
        try:
            for raw_line in path.read_text(encoding="utf-8").splitlines():
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

    load_file(Path(__file__).with_name(".env"))
    load_file(Path(__file__).resolve().parents[1] / ".env")
    load_file(Path.cwd() / ".env")


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


def _hint_for_step(step_id: str) -> str:
    q = QUESTIONS_BY_ID.get(step_id)
    if not isinstance(q, dict):
        return ""
    qtype = str(q.get("type") or "text")
    options = q.get("options") or []
    optional = bool(q.get("optional"))
    parts: list[str] = []

    if qtype == "enum":
        opts = [str(o) for o in options if o is not None]
        if opts:
            parts.append(f"Options: {' / '.join(opts)}.")
    elif qtype == "phone":
        parts.append("Example: +91 88123413123.")
    elif qtype == "email":
        parts.append("Example: name@company.com.")
    elif qtype == "date":
        parts.append("Use YYYY-MM-DD.")
    elif qtype == "json":
        parts.append("Paste valid JSON.")
    elif qtype == "number":
        parts.append("Send a number.")

    if optional:
        parts.append("You can reply 'skip' to leave it empty.")

    return " ".join(parts).strip()


def _wants_help(text: str) -> bool:
    t = (text or "").strip().lower()
    return bool(re.search(r"\b(help|how|format|example)\b", t))


def _wants_why(text: str) -> bool:
    t = (text or "").strip().lower()
    return bool(re.search(r"\b(why|reason|what\s+for|needed|need\s+this)\b", t))


def _looks_like_question(text: str) -> bool:
    t = (text or "").strip().lower()
    if not t:
        return False
    if "?" in t:
        return True
    return bool(re.search(r"^(what|why|how|where|which|can\s+i|do\s+i|is\s+this|meaning|explain)\b", t))


def _answer_like_for_step(step_id: str, message: str) -> bool:
    q = QUESTIONS_BY_ID.get(step_id) or {}
    qtype = str(q.get("type") or "text")
    text = (message or "").strip()
    if not text:
        return False

    if qtype == "enum":
        options = [str(o) for o in (q.get("options") or []) if o is not None]
        tokens = set(re.findall(r"[a-z0-9]+", text.lower()))
        return any(set(re.findall(r"[a-z0-9]+", opt.lower())).issubset(tokens) for opt in options if opt)

    if qtype == "email":
        return bool(re.search(r"[^\s@,;]+@[^\s@,;]+\.[^\s@,;]+", text))

    if qtype == "phone":
        return bool(re.search(r"\+?\d[\d\s\-\(\)]{6,}\d", text))

    if qtype == "date":
        return bool(re.match(r"^\d{4}-\d{2}-\d{2}$", text))

    if qtype == "number":
        return bool(re.match(r"^[\d.]+$", text))

    if qtype == "json":
        return text.startswith("{") or text.startswith("[")

    return not _looks_like_question(text)


def _pick_step_for_clarification(message: str, current_step: str) -> str:
    text = (message or "").strip().lower()
    if not text:
        return current_step

    msg_tokens = set(re.findall(r"[a-z0-9]+", text))
    if not msg_tokens:
        return current_step

    flow_index = {sid: i for i, sid in enumerate(engine.FLOW)}
    best_step = current_step
    best_score = 0

    for step_id, q in QUESTIONS_BY_ID.items():
        sid_tokens = set(re.findall(r"[a-z0-9]+", step_id.lower()))
        label_tokens = set(re.findall(r"[a-z0-9]+", str(q.get("label") or "").lower()))
        prompt_tokens = set(re.findall(r"[a-z0-9]+", str(q.get("prompt") or "").lower()))

        score = 0
        score += 3 * len(msg_tokens & sid_tokens)
        score += 2 * len(msg_tokens & label_tokens)
        score += 1 * len(msg_tokens & prompt_tokens)
        if score <= 0:
            continue

        if score > best_score:
            best_score = score
            best_step = step_id
        elif score == best_score and flow_index.get(step_id, 10_000_000) < flow_index.get(best_step, 10_000_000):
            best_step = step_id

    return best_step


def _ack_line(applied_labels: list[str]) -> str:
    if not applied_labels:
        return ""
    if len(applied_labels) == 1:
        return f"Cool — noted {applied_labels[0]}."
    return f"Nice, got these: {', '.join(applied_labels)}."


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

    intro = _phase_intro()
    prompt = _prompt_for(state, state.current_step)
    reply = f"{intro}\n\n{prompt}" if intro else prompt
    save_state(session_id, state)

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

    if req.action == "message":
        message = (req.message or "").strip()
        if message and _looks_like_question(message) and not _answer_like_for_step(state.current_step, message):
            target_step = _pick_step_for_clarification(message, state.current_step)
            explanation = engine.explain_step(state, target_step, message)
            prompt = _prompt_for(state, state.current_step)
            if target_step != state.current_step and prompt:
                reply = f"{explanation}\n\nFor now:\n{prompt}"
            else:
                reply = explanation if not prompt else f"{explanation}\n{prompt}"
            save_state(req.session_id, state)
            return {
                "reply": reply,
                "current_step": state.current_step,
                "data": state.data,
                "completed_steps": state.completed_steps,
                "completed": state.completed,
                "intent_source": "clarify",
            }

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

    applied_steps = next_state.data.pop("_last_applied_steps", None)
    applied_labels: list[str] = []
    if isinstance(applied_steps, list):
        for sid in applied_steps:
            if not isinstance(sid, str) or not sid:
                continue
            q = QUESTIONS_BY_ID.get(sid, {})
            label = str(q.get("label") or q.get("prompt") or sid).strip()
            if label and label not in applied_labels:
                applied_labels.append(label)
        if len(applied_labels) > 4:
            applied_labels = applied_labels[:4]

    next_state.last_user_message = None
    next_state.last_intent = None

    if next_state.last_error:
        # If there's an error, let the intelligent generator handle it
        # We pass the error message and the user's input so it can be contextual
        diff = {}  # No updates if error
        reply, invalidated = engine.generate_turn_reply(
            next_state, 
            diff, 
            next_state.current_step if not next_state.completed else None,
            req.message or "",
            error=next_state.last_error
        )
    elif next_state.confirmed:
        reply = "Confirmed."
        invalidated = []
    elif next_state.completed:
        # Use existing summary logic for now, but wrapped in a nice message
        summary = _review_summary(next_state)
        reply = f"{summary}\n\nReply 'confirm' to finish, or tell me what to change."
        invalidated = []
    else:
        # Success case: We have updates (applied_labels) and a next step
        # Construct a diff-like object for the generator
        # Note: applied_labels are just strings, but generate_turn_reply expects a dict for keys
        # We'll map labels back to a dummy dict for display purposes in the prompt
        diff_context = {lbl: "updated" for lbl in applied_labels}
        
        reply, invalidated = engine.generate_turn_reply(
            next_state,
            diff_context,
            next_state.current_step,
            req.message or ""
        )

    # Apply invalidations if any
    if invalidated:
        for field in invalidated:
            # Try to map back to actual keys if possible, or just use what LLM returned
            # The LLM sees the keys in 'diff' (which are labels in one case, but keys in another)
            # Wait, in the success case 'diff_context' uses labels. The LLM might return labels.
            # But in engine.py we pass 'diff' as is in the error case.
            
            # Actually, in the success case (lines 478), 'diff_context' keys are LABELS.
            # The LLM will return these LABELS in 'invalidated_fields'.
            # We need to map labels back to step_ids to remove them from data.
            # Or we can just try to match them.
            
            # Let's try to remove by exact key first
            next_state.data.pop(field, None)
            if field in next_state.completed_steps:
                next_state.completed_steps.remove(field)
                
            # Also try to remove by finding the step_id that matches the label
            for step_id, q in QUESTIONS_BY_ID.items():
                lbl = str(q.get("label") or q.get("prompt") or step_id).strip()
                if lbl == field:
                    next_state.data.pop(step_id, None)
                    if step_id in next_state.completed_steps:
                        next_state.completed_steps.remove(step_id)
                        
            # If we invalidated the current step, we might want to ensure it's not marked as skipped
            if field == next_state.current_step:
                 pass 

    save_state(req.session_id, next_state)

    return {
        "reply": reply,
        "current_step": next_state.current_step,
        "data": next_state.data,
        "completed_steps": next_state.completed_steps,
        "completed": next_state.completed,
        "intent_source": next_state.intent_source,
    }
