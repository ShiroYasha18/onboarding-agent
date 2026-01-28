from __future__ import annotations

import json
import os
import re
from typing import Any

from app.state import OnboardingState

with open("app/questions.json") as f:
    QUESTIONS: list[dict[str, Any]] = sorted(json.load(f)["questions"], key=lambda q: q["order"])

with open("app/dependencies.json") as f:
    DEPENDENCIES: dict[str, Any] = json.load(f)

QUESTIONS_BY_ID: dict[str, dict[str, Any]] = {q["id"]: q for q in QUESTIONS}
FLOW: list[str] = [q["id"] for q in QUESTIONS]


def get_questions() -> list[dict[str, Any]]:
    return QUESTIONS


def get_question(step_id: str) -> dict[str, Any] | None:
    return QUESTIONS_BY_ID.get(step_id)


def _next_prompt_variant(state: OnboardingState, step_id: str) -> int:
    store = state.data.get("_prompt_variant_counter")
    if not isinstance(store, dict):
        store = {}
        state.data["_prompt_variant_counter"] = store
    raw = store.get(step_id)
    n = int(raw) if isinstance(raw, (int, float)) else 0
    n += 1
    store[step_id] = n
    return n


def _get_prompt_history(state: OnboardingState, step_id: str) -> list[str]:
    store = state.data.get("_prompt_history")
    if not isinstance(store, dict):
        return []
    history = store.get(step_id)
    return [h for h in history if isinstance(h, str) and h.strip()] if isinstance(history, list) else []


def _push_prompt_history(state: OnboardingState, step_id: str, prompt: str) -> None:
    store = state.data.get("_prompt_history")
    if not isinstance(store, dict):
        store = {}
        state.data["_prompt_history"] = store
    history = store.get(step_id)
    if not isinstance(history, list):
        history = []
        store[step_id] = history
    text = (prompt or "").strip()
    if not text:
        return
    if history and history[-1] == text:
        return
    history.append(text)
    if len(history) > 6:
        del history[:-6]


def _fallback_prompt(question: dict[str, Any], *, variant: int) -> str:
    step_id = str(question.get("id") or "")
    label = str(question.get("label") or step_id or "this")
    qtype = str(question.get("type") or "text")
    optional = bool(question.get("optional"))
    required = bool(question.get("required"))
    v = abs(int(variant or 0))

    if qtype == "enum":
        opts = [str(o) for o in (question.get("options") or []) if o is not None]
        opts_text = " / ".join(opts) if opts else ""
        templates = [
            f"Quick one — which {label.lower()} should we use: {opts_text}?",
            f"Which {label.lower()} do you want to go with ({opts_text})?",
            f"Pick a {label.lower()} for this setup: {opts_text}.",
            f"To keep things moving — {label.lower()}: {opts_text}?",
        ]
        if opts_text:
            return templates[v % len(templates)]
        return [f"Quick one — what should we use for {label.lower()}?", f"What should {label.lower()} be?"][v % 2]

    if qtype == "phone":
        templates = [
            f"What’s the best phone number for {label.lower()}?",
            f"Can you share the phone number for {label.lower()}?",
            f"Which phone number should I use for {label.lower()}?",
            f"Phone number for {label.lower()}?",
        ]
        base = templates[v % len(templates)]
        suffix = "(Example: +91 88123413123)" if required else "(Example: +91 88123413123, or reply 'skip')"
        return f"{base} {suffix}"

    if qtype == "email":
        templates = [
            f"What’s the email for {label.lower()}?",
            f"Can you share the email for {label.lower()}?",
            f"Which email should I use for {label.lower()}?",
            f"Email for {label.lower()}?",
        ]
        base = templates[v % len(templates)]
        suffix = "(Example: name@company.com)" if required else "(Example: name@company.com, or reply 'skip')"
        return f"{base} {suffix}"

    if qtype == "date":
        templates = [
            f"What date should I put for {label.lower()}?",
            f"What’s the {label.lower()} date?",
            f"Which date should we use for {label.lower()}?",
            f"Date for {label.lower()}?",
        ]
        base = templates[v % len(templates)]
        suffix = "(YYYY-MM-DD)" if required else "(YYYY-MM-DD, or reply 'skip')"
        return f"{base} {suffix}"

    if qtype == "number":
        templates = [
            f"What number should I put for {label.lower()}?",
            f"What’s the {label.lower()} value?",
            f"Give me the number for {label.lower()}.",
            f"{label} — what number should it be?",
        ]
        base = templates[v % len(templates)]
        return base if required else f"{base} (or reply 'skip')"

    if qtype == "json":
        prompt = str(question.get("prompt") or "")
        example = ""
        if "Example:" in prompt:
            example = prompt.split("Example:", 1)[1].strip()
        templates = [
            f"Can you paste {label.lower()} as JSON?",
            f"Send {label.lower()} as JSON for me.",
            f"Share {label.lower()} in JSON format.",
            f"Drop {label.lower()} as JSON whenever you’re ready.",
        ]
        base = templates[v % len(templates)]
        if example:
            base = f"{base} Example: {example}"
        return base if required else f"{base} (or reply 'skip')"

    templates = [
        f"What should I put for {label.lower()}?",
        f"What do you want to set for {label.lower()}?",
        f"Can you share {label.lower()}?",
        f"{label} — what should it be?",
    ]
    base = templates[v % len(templates)]
    if optional and not required:
        return f"{base} (or reply 'skip')"
    return base


def prompt_for(state: OnboardingState, step_id: str) -> str:
    question = get_question(step_id)
    if not question:
        return "Continue."

    variant = _next_prompt_variant(state, step_id)
    base = _fallback_prompt(question, variant=variant)
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        _push_prompt_history(state, step_id, base)
        return base

    try:
        from google import genai  # type: ignore
    except Exception:
        _push_prompt_history(state, step_id, base)
        return base

    filtered_known = {k: v for k, v in (state.data or {}).items() if isinstance(k, str) and not k.startswith("_")}
    history = _get_prompt_history(state, step_id)[-4:]
    payload = {
        "id": question.get("id"),
        "type": question.get("type"),
        "options": question.get("options"),
        "required": question.get("required"),
        "optional": question.get("optional"),
        "label": question.get("label"),
        "prompt": question.get("prompt"),
    }

    client = genai.Client(api_key=api_key)
    model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    llm_prompt = (
        "Write ONE short, friendly, concrete question for the user.\n"
        "Use a natural conversational tone.\n"
        "Make it feel fresh; do not reuse the same phrasing.\n"
        "Avoid these previous variants (if any):\n"
        f"{json.dumps(history, ensure_ascii=False)}\n"
        "Ask only for the field in step_definition.\n"
        "If it is an enum, include the options.\n"
        "No extra commentary.\n\n"
        f"step_definition: {json.dumps(payload, ensure_ascii=False)}\n"
        f"known_data: {json.dumps(filtered_known, ensure_ascii=False)}\n"
        f"variation_index: {variant}\n"
    )
    try:
        text = (client.models.generate_content(model=model, contents=llm_prompt).text or "").strip()
        if not text:
            _push_prompt_history(state, step_id, base)
            return base
        rendered = text.splitlines()[0].strip() if len(text) > 140 else text
        if not rendered:
            _push_prompt_history(state, step_id, base)
            return base
        _push_prompt_history(state, step_id, rendered)
        return rendered
    except Exception:
        _push_prompt_history(state, step_id, base)
        return base


def generate_turn_reply(
    state: OnboardingState,
    diff: dict[str, Any],
    next_step_id: str | None,
    user_message: str,
    error: str | None = None
) -> str:
    question = get_question(next_step_id) if next_step_id else None
    
    # Get dynamic reasoning from the question definition
    reason_hint = ""
    if question:
        reason_hint = question.get("reasoning", "is required to complete the profile")

    # Fallback logic if no LLM key
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        parts = []
        # Acknowledgement
        if not error and diff:
            labels = [k.split(".")[-1].replace("_", " ") for k in diff.keys()]
            if len(labels) <= 2:
                parts.append(f"Got it, noted your {', '.join(labels)}.")
            else:
                parts.append(f"Nice, got these: {', '.join(labels)}.")
        elif error:
            parts.append(f"Use format: {error}" if "format" in error.lower() else error)
        
        # Check for confusion in fallback mode
        is_confused = re.search(r"\b(why|understand|meaning|what\s+is)\b", user_message.lower())
        
        if is_confused:
            parts.append("I'm sorry if that was unclear.")
            # Use the reasoning from the question definition
            if reason_hint:
                 parts.append(f"We need {next_step_id.split('.')[-1].replace('_', ' ')} because it {reason_hint[0].lower() + reason_hint[1:] if reason_hint else ''}")
        
        # Next Question
        if next_step_id:
            parts.append(prompt_for(state, next_step_id))
        else:
            parts.append("All done! Reviewing your details...")
        
        return "\n".join(parts)

    # LLM Logic
    try:
        from google import genai  # type: ignore
        client = genai.Client(api_key=api_key)
        model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        
        question_payload = {}
        if question:
            question_payload = {
                "id": question.get("id"),
                "type": question.get("type"),
                "options": question.get("options"),
                "label": question.get("label"),
                "prompt": question.get("prompt"),
                "reasoning": question.get("reasoning"),
            }

        llm_prompt = (
            "You are a friendly, intelligent onboarding assistant.\n"
            "Your goal is to acknowledge what was just captured (if any), address any errors, and ask the next question naturally.\n"
            "CRITICAL: If the user seems confused, asks 'Why?', or says 'I don't understand', you MUST explain the reasoning behind the current step before asking again.\n\n"
            f"Context:\n"
            f"- User's last message: '{user_message}'\n"
            f"- Fields successfully updated just now: {json.dumps(diff, ensure_ascii=False)}\n"
            f"- Error/Warning from system: {error or 'None'}\n"
            f"- Next required step: {next_step_id or 'None'}\n"
            f"- Next step definition: {json.dumps(question_payload, ensure_ascii=False)}\n"
            f"- Current known data summary: {json.dumps({k: v for k, v in (state.data or {}).items() if not k.startswith('_')}, ensure_ascii=False)}\n"
            f"- Hint for 'Why' we need this step: {reason_hint}\n\n"
            "Instructions:\n"
            "1. If 'Fields successfully updated' is not empty, acknowledge them briefly (e.g. 'Got the email', 'Noted the warehouse').\n"
            "2. If there is an Error, explain it helpfully and ask for correction.\n"
            "3. If the user is confused or asking a question (e.g. 'Why?', 'What is this?', 'I don't understand'):\n"
            "   - First, apologize for any confusion.\n"
            "   - Explain specifically WHY we need this field using the hint provided or your own knowledge.\n"
            "   - Then, gently ask the question again or guide them on what to enter.\n"
            "4. If 'Next required step' is present, ask the question for it. Use the step definition to know what to ask. If it's an enum, mention options naturally.\n"
            "5. If the user's message contained info that wasn't extracted (compare user message vs updated fields), acknowledge it but say we need to focus on the current step first.\n"
            "6. Keep the whole response under 3 sentences if possible.\n"
            "7. Direct the user clearly to the next step."
        )
        
        resp = client.models.generate_content(model=model, contents=llm_prompt)
        text = (resp.text or "").strip()
        if text:
            return text
            
    except Exception:
        pass
        
    # Fallback if LLM fails
    parts = []
    
    # Check for confusion in fallback mode
    is_confused = re.search(r"\b(why|understand|meaning|what\s+is)\b", user_message.lower())
    
    if is_confused:
        parts.append("I'm sorry if that was unclear.")
        # Try to use the canned reason if available
        if reason_hint:
             parts.append(f"We need {next_step_id.split('.')[-1].replace('_', ' ')} because it {reason_hint[0].lower() + reason_hint[1:] if reason_hint else ''}")
    
    if not error and diff:
        parts.append("Got it.")
    if error and not is_confused:
        parts.append(f"Note: {error}")
        
    if next_step_id:
        parts.append(prompt_for(state, next_step_id))
    return "\n".join(parts)


def explain_step(state: OnboardingState, step_id: str, user_question: str) -> str:
    question = get_question(step_id) or {}
    label = str(question.get("label") or step_id)
    qtype = str(question.get("type") or "text")
    options = [str(o) for o in (question.get("options") or []) if o is not None]
    
    reasoning = question.get("reasoning", "")
    
    if reasoning:
        base = reasoning
        if qtype == "enum" and options:
            return f"{base} Options: {' / '.join(options)}."
        if qtype == "date":
            return f"{base} Use YYYY-MM-DD."
        if qtype == "email":
            return f"{base} Example: name@company.com."
        if qtype == "phone":
            return f"{base} Example: +91 88123413123."
        return base

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        if qtype == "enum" and options:
            return f"For {label.lower()}, pick one option: {' / '.join(options)}."
        if qtype == "date":
            return f"For {label.lower()}, send a date like 2026-01-28."
        if qtype == "email":
            return f"For {label.lower()}, send an email like name@company.com."
        if qtype == "phone":
            return f"For {label.lower()}, send a phone number like +91 88123413123."
        if qtype == "json":
            return f"For {label.lower()}, paste valid JSON."
        if qtype == "number":
            return f"For {label.lower()}, send a number."
        return f"For {label.lower()}, just send the value you want to set."

    try:
        from google import genai  # type: ignore
    except Exception:
        return f"For {label.lower()}, just send the value you want to set."

    client = genai.Client(api_key=api_key)
    model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    payload = {
        "id": question.get("id"),
        "type": question.get("type"),
        "options": question.get("options"),
        "required": question.get("required"),
        "optional": question.get("optional"),
        "label": question.get("label"),
        "prompt": question.get("prompt"),
    }
    llm_prompt = (
        "Explain the current onboarding question in 1-2 short sentences.\n"
        "Include a concrete example answer if helpful.\n"
        "Do not ask for other fields.\n"
        "Do not mention internal system details.\n\n"
        f"question: {json.dumps(payload, ensure_ascii=False)}\n"
        f"user_question: {user_question}\n"
    )
    try:
        text = (client.models.generate_content(model=model, contents=llm_prompt).text or "").strip()
        return text or f"For {label.lower()}, just send the value you want to set."
    except Exception:
        return f"For {label.lower()}, just send the value you want to set."


def _extract_text_value(intent: dict[str, Any], fallback: str) -> str:
    raw = intent.get("value")
    if isinstance(raw, dict) and raw:
        raw = next(iter(raw.values()), raw)
    if raw is None:
        raw = fallback
    return str(raw or "").strip()


def _is_confirm_text(text: str) -> bool:
    lowered = (text or "").strip().lower()
    if not lowered:
        return False
    if lowered in {"y", "yes", "ok", "okay", "confirm", "confirmed", "proceed", "submit", "done"}:
        return True
    return bool(re.search(r"\b(confirm|confirmed|proceed|submit)\b", lowered))


def _is_decline_text(text: str) -> bool:
    lowered = (text or "").strip().lower()
    if lowered in {"n", "no", "nope", "not yet", "dont", "don't"}:
        return True
    return bool(re.search(r"\b(no|nope|not\s+yet|don'?t)\b", lowered))


def apply_intent(state: OnboardingState) -> OnboardingState:
    state.last_error = None
    intent = state.last_intent or {}
    intent_type = intent.get("intent")

    if intent_type in {"answer", "correct_step", "go_to_step"} and state.confirmed:
        state.confirmed = False

    if state.current_step == "review" and intent_type in {"answer", "unknown"}:
        text = _extract_text_value(intent, state.last_user_message or "")
        if _is_confirm_text(text):
            state.confirmed = True
            state.completed = True
            state.current_step = "done"
            return state
        state.last_error = (
            "Tell me what you'd like to change (e.g., 'change hotel name to ...' or 'go to billing address'), "
            "or reply 'confirm' to finish."
            if _is_decline_text(text) or text
            else "Reply 'confirm' to finish, or tell me what you'd like to change."
        )
        return state

    value_map = intent.get("value")
    if intent_type in {"answer", "correct_step"} and isinstance(value_map, dict) and value_map:
        if intent_type == "answer":
            ordered_pairs: list[tuple[str, Any]] = []
            for sid, raw in value_map.items():
                if not isinstance(sid, str) or sid not in QUESTIONS_BY_ID:
                    continue
                ordered_pairs.append((sid, raw))

            if ordered_pairs and (len(ordered_pairs) > 1 or ordered_pairs[0][0] != state.current_step):
                ordered_pairs.sort(key=lambda p: FLOW.index(p[0]) if p[0] in FLOW else 10_000)
                for sid, raw in ordered_pairs:
                    state = _apply_correction_value(state, sid, raw)
                    if state.last_error:
                        return state
                return resolve_next_step(state)

            raw = value_map.get(state.current_step)
            if raw is None and len(value_map) == 1:
                raw = next(iter(value_map.values()))
            if raw is None:
                state.last_error = "Could you rephrase that?"
                return state
            return _apply_answer_value(state, state.current_step, raw)

        step_id = intent.get("step_id")
        if isinstance(step_id, str) and step_id:
            raw = value_map.get(step_id)
            if raw is None and len(value_map) == 1:
                raw = next(iter(value_map.values()))
            if raw is None:
                state.last_error = "Could you rephrase that?"
                return state
            state = _apply_correction_value(state, step_id, raw)
            if state.last_error:
                return state
            extras: list[tuple[str, Any]] = []
            for sid, v in value_map.items():
                if sid == step_id:
                    continue
                if not isinstance(sid, str) or sid not in QUESTIONS_BY_ID:
                    continue
                extras.append((sid, v))
            if extras:
                extras.sort(key=lambda p: FLOW.index(p[0]) if p[0] in FLOW else 10_000)
                for sid, v in extras:
                    state = _apply_correction_value(state, sid, v)
                    if state.last_error:
                        return state
            return resolve_next_step(state)

        applied_any = False
        for sid, raw in value_map.items():
            if not isinstance(sid, str) or sid not in QUESTIONS_BY_ID:
                continue
            state = _apply_correction_value(state, sid, raw)
            if state.last_error:
                return state
            applied_any = True
        if not applied_any:
            state.last_error = "Could you rephrase that?"
            return state
        return resolve_next_step(state)

    if intent_type == "answer":
        step_id = state.current_step
        question = get_question(step_id)
        if not question:
            state.last_error = "Unknown step"
            return state

        raw = intent.get("value")
        if isinstance(raw, dict) and raw:
            raw = raw.get(step_id, next(iter(raw.values()), raw))
        ok, normalized, error = _normalize_value(question, raw)
        if not ok:
            state.last_error = error
            return state

        prior = state.data.get(step_id, None)
        if step_id in state.completed_steps and prior != normalized:
            invalidate_downstream(state, step_id)

        _set_value(state, step_id, normalized)
        _apply_special_rules(state, step_id, normalized)
        return resolve_next_step(state)

    if intent_type == "correct_step":
        step_id = intent.get("step_id")
        if not isinstance(step_id, str) or not step_id:
            state.last_error = "Invalid step"
            return state
        question = get_question(step_id)
        if not question:
            state.last_error = "Unknown step"
            return state

        raw = intent.get("value")
        if isinstance(raw, dict) and raw:
            raw = raw.get(step_id, next(iter(raw.values()), raw))
        ok, normalized, error = _normalize_value(question, raw)
        if not ok:
            state.last_error = error
            return state

        prior = state.data.get(step_id, None)
        _set_value(state, step_id, normalized)
        if step_id in state.completed_steps and prior != normalized:
            invalidate_downstream(state, step_id)
        _apply_special_rules(state, step_id, normalized)
        return resolve_next_step(state)

    if intent_type == "go_to_step":
        step_id = intent.get("step_id")
        if not isinstance(step_id, str) or not step_id:
            state.last_error = "Unknown step"
            return state

        target: str | None = None
        if step_id in QUESTIONS_BY_ID:
            target = step_id
        else:
            prefix = step_id.rstrip(".")
            matches = [sid for sid in FLOW if sid == prefix or sid.startswith(prefix + ".")]
            if matches:
                target = matches[0]

        if not target:
            state.last_error = "Unknown step"
            return state

        question = QUESTIONS_BY_ID.get(target)
        if question and not _is_applicable(state, question):
            state.last_error = "That step isn’t applicable based on earlier answers."
            return state

        state.current_step = target
        state.completed = False
        return state

    if intent_type == "unknown" and (state.intent_source or "").strip().lower() == "none":
        state.last_error = "LLM is not configured (set GEMINI_API_KEY)."
        return state

    state.last_error = "Could you rephrase that?"
    return state


def _apply_answer_value(state: OnboardingState, step_id: str, raw: Any) -> OnboardingState:
    question = get_question(step_id)
    if not question:
        state.last_error = "Unknown step"
        return state

    ok, normalized, error = _normalize_value(question, raw)
    if not ok:
        state.last_error = error
        return state

    prior = state.data.get(step_id, None)
    if step_id in state.completed_steps and prior != normalized:
        invalidate_downstream(state, step_id)

    _set_value(state, step_id, normalized)
    _apply_special_rules(state, step_id, normalized)
    return resolve_next_step(state)


def _apply_correction_value(state: OnboardingState, step_id: str, raw: Any) -> OnboardingState:
    question = get_question(step_id)
    if not question:
        state.last_error = "Unknown step"
        return state

    ok, normalized, error = _normalize_value(question, raw)
    if not ok:
        state.current_step = step_id
        state.last_error = error
        return state

    prior = state.data.get(step_id, None)
    _set_value(state, step_id, normalized)
    if step_id in state.completed_steps and prior != normalized:
        invalidate_downstream(state, step_id)
    _apply_special_rules(state, step_id, normalized)
    return state


def invalidate_downstream(state: OnboardingState, step_id: str) -> None:
    dependency_key = _dependency_key(step_id)
    invalidates = DEPENDENCIES.get(dependency_key, {}).get("invalidates", [])
    for dep in invalidates:
        _clear_prefix(state, str(dep))


def resolve_next_step(state: OnboardingState) -> OnboardingState:
    for step_id in FLOW:
        question = QUESTIONS_BY_ID[step_id]
        if not _is_applicable(state, question):
            _mark_skipped(state, question)
            continue
        if step_id not in state.completed_steps:
            state.current_step = step_id
            state.completed = False
            return state

    state.completed = True
    state.current_step = "review"
    return state


def _dependency_key(step_id: str) -> str:
    return step_id.split(".", 1)[0]


def _is_applicable(state: OnboardingState, question: dict[str, Any]) -> bool:
    dep = question.get("depends_on")
    if not isinstance(dep, dict):
        return True
    dep_id = dep.get("id")
    expected = dep.get("equals")
    if not isinstance(dep_id, str):
        return True
    actual = state.data.get(dep_id)
    return str(actual) == str(expected)


def _mark_skipped(state: OnboardingState, question: dict[str, Any]) -> None:
    step_id = question["id"]
    if step_id in state.completed_steps:
        return
    default = question.get("default")
    if default is not None and step_id not in state.data:
        state.data[step_id] = default
    state.completed_steps.append(step_id)


def _set_value(state: OnboardingState, step_id: str, value: Any) -> None:
    state.data[step_id] = value
    last_applied = state.data.get("_last_applied_steps")
    if not isinstance(last_applied, list):
        last_applied = []
        state.data["_last_applied_steps"] = last_applied
    if step_id not in last_applied:
        last_applied.append(step_id)
    if step_id not in state.completed_steps:
        state.completed_steps.append(step_id)


def _clear_prefix(state: OnboardingState, prefix: str) -> None:
    to_remove = [k for k in state.data.keys() if k == prefix or k.startswith(prefix + ".")]
    for k in to_remove:
        state.data.pop(k, None)

    state.completed_steps = [
        s for s in state.completed_steps if not (s == prefix or s.startswith(prefix + "."))
    ]


def _apply_special_rules(state: OnboardingState, step_id: str, value: Any) -> None:
    if step_id == "shipping_address.same_as_billing":
        normalized = str(value).strip().lower()
        if normalized == "yes":
            billing_map = {
                "shipping_address.address_line_1": state.data.get("billing_address.address_line_1"),
                "shipping_address.address_line_2": state.data.get("billing_address.address_line_2"),
                "shipping_address.city": state.data.get("billing_address.city"),
                "shipping_address.state": state.data.get("billing_address.state"),
                "shipping_address.zip_code": state.data.get("billing_address.zip_code"),
                "shipping_address.country": state.data.get("billing_address.country"),
            }
            for k, v in billing_map.items():
                if v is not None:
                    _set_value(state, k, v)
            return
        if normalized == "no":
            for key in list(state.data.keys()):
                if key.startswith("shipping_address.") and key != "shipping_address.same_as_billing":
                    state.data.pop(key, None)
            state.completed_steps = [
                s
                for s in state.completed_steps
                if not (s.startswith("shipping_address.") and s != "shipping_address.same_as_billing")
            ]
            return

    if step_id == "inventory_pool.want_pool":
        normalized = str(value).strip().lower()
        if normalized == "no":
            _clear_prefix(state, "inventory_pool.pool_name")
            _clear_prefix(state, "inventory_pool.description")
            _clear_prefix(state, "inventory_pool.products")


def _normalize_value(question: dict[str, Any], raw: Any) -> tuple[bool, Any, str | None]:
    qtype = str(question.get("type") or "text")
    required = bool(question.get("required"))
    optional = bool(question.get("optional"))
    has_default = "default" in question

    if isinstance(raw, str):
        stripped = raw.strip()
        if stripped.lower() in {"skip", "na", "n/a", "none", "null"}:
            if required and not optional and not has_default:
                return False, None, "This field is required."
            if has_default:
                return True, question.get("default"), None
            return True, None, None
        raw = stripped

    if raw is None:
        if required and not optional and not has_default:
            return False, None, "This field is required."
        if has_default:
            return True, question.get("default"), None
        return True, None, None

    if qtype == "enum":
        options = question.get("options") or []
        options_str = [str(o) for o in options]
        candidate = str(raw).strip()
        candidate_l = candidate.lower()
        candidate_tokens = set(re.findall(r"[a-z0-9]+", candidate_l))
        for opt in options_str:
            opt_l = opt.lower()
            opt_tokens = set(re.findall(r"[a-z0-9]+", opt_l))
            if candidate_l == opt_l:
                return True, opt, None
            if opt_tokens and opt_tokens.issubset(candidate_tokens):
                return True, opt, None
        return False, None, f"Choose one of: {', '.join(options_str)}"

    if qtype == "number":
        if isinstance(raw, (int, float)):
            return True, raw, None
        candidate = str(raw).strip()
        try:
            value = float(candidate) if "." in candidate else int(candidate)
            return True, value, None
        except Exception:
            return False, None, "Enter a valid number."

    if qtype == "email":
        candidate = str(raw).strip()
        if re.match(r"^[^\s@]+@[^\s@]+\.[^\s@]+$", candidate):
            return True, candidate, None
        embedded = re.search(r"([^\s@]+@[^\s@]+\.[^\s@]+)", candidate)
        if embedded:
            return True, embedded.group(1), None
        return False, None, "Enter a valid email."

    if qtype == "phone":
        candidate = str(raw).strip()
        match = re.search(r"(\+?\d[\d\s\-\(\)]{6,}\d)", candidate)
        if not match:
            return False, None, "Enter a valid phone number."
        normalized = re.sub(r"[\s\-\(\)]+", "", match.group(1))
        digits = normalized[1:] if normalized.startswith("+") else normalized
        if not digits.isdigit() or len(digits) < 8 or len(digits) > 15:
            return False, None, "Enter a valid phone number."
        return True, normalized, None

    if qtype == "date":
        candidate = str(raw).strip()
        if re.match(r"^\d{4}-\d{2}-\d{2}$", candidate):
            return True, candidate, None
        return False, None, "Enter a date as YYYY-MM-DD."

    if qtype == "list":
        if isinstance(raw, list):
            return True, raw, None
        candidate = str(raw).strip()
        parts = [p.strip() for p in candidate.split(",")]
        items = [p for p in parts if p]
        if required and not items and not optional:
            return False, None, "Enter at least one item."
        return True, items, None

    if qtype == "json":
        if isinstance(raw, (dict, list)):
            return True, raw, None
        candidate = str(raw).strip()
        try:
            return True, json.loads(candidate), None
        except Exception:
            return False, None, "Enter valid JSON."

    return True, str(raw).strip(), None
