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


def _push_qa_history(state: OnboardingState, step_id: str, question: str, answer: Any) -> None:
    store = state.data.get("_qa_history")
    if not isinstance(store, list):
        store = []
        state.data["_qa_history"] = store
    item = {
        "step_id": str(step_id),
        "question": str(question or "").strip(),
        "answer": answer,
    }
    store.append(item)
    if len(store) > 5:
        del store[:-5]


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
    previous_prompt = history[-1] if history else ""
    last_user_message = (state.last_user_message or "").strip()
    last_intent = state.last_intent or {}
    last_intent_type = str(last_intent.get("intent") or "")
    pending_data = state.pending_data if isinstance(state.pending_data, dict) else None
    qa_store = state.data.get("_qa_history")
    qa_history = qa_store if isinstance(qa_store, list) else []
    recent_qa = qa_history[-5:]
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
    model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
    llm_prompt = (
        "Write ONE short, friendly, conversational question.\n"
        "Keep it natural and contextual: lightly acknowledge what the user just said and smoothly segue into the ask.\n"
        "Do not apologize or over-explain; avoid robotic phrasing.\n"
        "Avoid repeating prior phrasings:\n"
        f"{json.dumps(history, ensure_ascii=False)}\n"
        "Constraints:\n"
        "- Ask ONLY for the field in 'step_definition'.\n"
        "- If it is an enum, include the options inline.\n"
        "- Keep it under ~80 characters if possible.\n\n"
        "Context:\n"
        f"- previous_prompt: {json.dumps(previous_prompt, ensure_ascii=False)}\n"
        f"- last_user_message: {json.dumps(last_user_message, ensure_ascii=False)}\n"
        f"- last_intent_type: {json.dumps(last_intent_type, ensure_ascii=False)}\n"
        f"- pending_data: {json.dumps(pending_data or {}, ensure_ascii=False)}\n"
        f"- step_definition: {json.dumps(payload, ensure_ascii=False)}\n"
        f"- known_data: {json.dumps(filtered_known, ensure_ascii=False)}\n"
        f"- last_qa_pairs: {json.dumps(recent_qa, ensure_ascii=False)}\n"
        f"- variation_index: {variant}\n"
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
    error: str | None = None,
    intent_result: dict[str, Any] | None = None
) -> tuple[str, list[str]]:
    effective_next = next_step_id or state.current_step
    question = get_question(effective_next) if effective_next else None
    
    # Get dynamic reasoning from the question definition
    reason_hint = ""
    if question:
        reason_hint = question.get("reasoning", "is required to complete the profile")

    # Fallback logic if no LLM key
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

    if state.mode == "confirming" and state.pending_value:
        pairs = []
        pending_fields_payload: list[dict[str, Any]] = []
        for k, v in state.pending_value.items():
            q = get_question(k) or {}
            label = str(q.get("label") or k.split(".")[-1].replace("_", " "))
            pairs.append(f"{label}: {v}")
            pending_fields_payload.append(
                {
                    "id": k,
                    "label": label,
                    "value": v,
                    "reasoning": q.get("reasoning"),
                    "type": q.get("type"),
                }
            )
        summary = "; ".join(pairs) if pairs else ""

        if not api_key:
            return (f"I understood {summary}. Is that correct? (yes / no)", []) if summary else (
                "Is this correct? (yes / no)",
                [],
            )

        try:
            from google import genai  # type: ignore
        except Exception:
            return (f"I understood {summary}. Is that correct? (yes / no)", []) if summary else (
                "Is this correct? (yes / no)",
                [],
            )

        client = genai.Client(api_key=api_key)
        model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
        known_data = {
            k: v
            for k, v in (state.data or {}).items()
            if isinstance(k, str) and not k.startswith("_")
        }
        last_intent_type = intent_result.get("intent") if intent_result else None
        last_intent_thought = intent_result.get("thought") if intent_result else None

        confirm_prompt = (
            "You are a friendly, intelligent onboarding assistant.\n"
            "We have parsed the user's last message and inferred some field values, but we want to confirm them before saving.\n"
            "Your job is to:\n"
            "1. Briefly acknowledge what we understood (use the labels and values).\n"
            "2. If the user's message includes a small personal detail or emotion, briefly acknowledge it.\n"
            "3. Ask clearly for confirmation or correction.\n"
            "Keep the reply to 1-2 sentences.\n"
            "Do not mention internal system details.\n\n"
            f"User message: {user_message!r}\n"
            f"Pending fields: {json.dumps(pending_fields_payload, ensure_ascii=False)}\n"
            f"Current known data: {json.dumps(known_data, ensure_ascii=False)}\n"
            f"Intent type: {last_intent_type}\n"
            f"Intent thought: {last_intent_thought or 'None'}\n\n"
            "Output JSON with shape:\n"
            "{ \"reply\": \"string\" }\n"
        )

        try:
            resp = client.models.generate_content(
                model=model,
                contents=confirm_prompt,
                config={"response_mime_type": "application/json"},
            )
            text = (resp.text or "").strip()
            if text:
                try:
                    data = json.loads(text)
                    reply = data.get("reply")
                    if isinstance(reply, str) and reply.strip():
                        return reply, []
                except json.JSONDecodeError:
                    if text:
                        return text, []
        except Exception:
            pass

        return (f"I understood {summary}. Is that correct? (yes / no)", []) if summary else (
            "Is this correct? (yes / no)",
            [],
        )

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
        is_confused = False
        if intent_result and intent_result.get("intent") == "clarification":
            is_confused = True
        
        if is_confused:
            parts.append("I'm sorry if that was unclear.")
            # Use the reasoning from the question definition
            if reason_hint:
                 parts.append(f"We need {next_step_id.split('.')[-1].replace('_', ' ')} because it {reason_hint[0].lower() + reason_hint[1:] if reason_hint else ''}")
        
        # Next Question
        if effective_next:
            parts.append(prompt_for(state, effective_next))
        else:
            parts.append("All done! Reviewing your details...")
        
        return "\n".join(parts), []

    # LLM Logic
    try:
        from google import genai  # type: ignore
        client = genai.Client(api_key=api_key)
        model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
        
        # Calculate upcoming steps (max 3) for grouping
        upcoming_steps_payload = []
        if next_step_id:
            try:
                start_idx = FLOW.index(next_step_id)
                # Take next 3 steps
                candidates = FLOW[start_idx : start_idx + 3]
                for cid in candidates:
                    q_def = QUESTIONS_BY_ID.get(cid)
                    if q_def:
                        upcoming_steps_payload.append({
                            "id": q_def.get("id"),
                            "label": q_def.get("label"),
                            "type": q_def.get("type"),
                            "options": q_def.get("options"),
                            "reasoning": q_def.get("reasoning"),
                        })
            except ValueError:
                if question:
                    upcoming_steps_payload.append({
                        "id": question.get("id"),
                        "label": question.get("label"),
                        "type": question.get("type"),
                        "options": question.get("options"),
                        "reasoning": question.get("reasoning"),
                    })

        intent_thought = intent_result.get("thought") if intent_result else None
        intent_type = intent_result.get("intent") if intent_result else None
        is_clarification = intent_type in {"clarification", "confusion", "skip"} if intent_type else False
        qa_store = state.data.get("_qa_history")
        qa_history = qa_store if isinstance(qa_store, list) else []
        recent_qa = qa_history[-5:]

        phases_list: list[Any] = []
        phase_map: dict[str, Any] = {}
        phases_raw = state.data.get("_phases")
        if isinstance(phases_raw, list):
            phases_list = phases_raw
        step_phase_raw = state.data.get("_step_phase")
        if isinstance(step_phase_raw, dict):
            phase_map = step_phase_raw
        current_phase = None
        if effective_next and isinstance(phase_map, dict):
            cp = phase_map.get(effective_next)
            if isinstance(cp, dict):
                current_phase = cp

        llm_prompt = (
            "You are a friendly, intelligent onboarding assistant.\n"
            "Your goal is to acknowledge what was just captured (if any), optionally react briefly to personal details, address any errors, and ask the next question(s) naturally.\n"
            "CRITICAL: If the user seems confused, asks 'Why?', or says 'I don't understand', you MUST explain the reasoning behind the current step before asking again.\n"
            "CRITICAL: If 'Fields successfully updated' contains values that seem to be QUESTIONS (e.g. 'Should I...', 'Why...') or are clearly wrong/misinterpreted, you MUST identify them to be removed.\n"
            "CRITICAL: Treat any field already present in 'Current known data summary' as completed. Do NOT ask for it again or say you 'missed' it unless the user is clearly changing it or it appears in 'invalidated_fields'.\n"
            "CRITICAL: Treat all context fields (including the hint) as background only. Do not copy any of them word-for-word; rewrite explanations in your own natural, conversational language.\n\n"
            f"Context:\n"
            f"- User's last message: '{user_message}'\n"
            f"- Fields successfully updated just now: {json.dumps(diff, ensure_ascii=False)}\n"
            f"- Error/Warning from system: {error or 'None'}\n"
            f"- Upcoming steps (in order): {json.dumps(upcoming_steps_payload, ensure_ascii=False)}\n"
            f"- Current known data summary: {json.dumps({k: v for k, v in (state.data or {}).items() if not k.startswith('_')}, ensure_ascii=False)}\n"
            f"- Hint for 'Why' we need the current step: {reason_hint}\n"
            f"- Intent analysis thought: {intent_thought or 'None'}\n"
            f"- Is clarification request: {is_clarification}\n"
            f"- Last 5 Q&A pairs: {json.dumps(recent_qa, ensure_ascii=False)}\n"
            f"- Current phase info: {json.dumps(current_phase or {}, ensure_ascii=False)}\n"
            f"- All phases: {json.dumps(phases_list, ensure_ascii=False)}\n\n"
            "Instructions:\n"
            "1. Check 'Fields successfully updated'. If any value looks like a user question (e.g. starts with 'Should I', 'Why', 'What') or is clearly a misinterpretation of the user's intent, add its key to 'invalidated_fields'.\n"
            "2. If 'Fields successfully updated' is valid, acknowledge them briefly (e.g. 'Got the email', 'Noted the warehouse').\n"
            "3. If there is an Error, explain it helpfully and ask for correction.\n"
            "4. If 'Is clarification request' is true OR the user is confused:\n"
            "   - First, apologize for any confusion.\n"
            "   - Explain specifically WHY we need this field, using the hint as context but paraphrasing it in your own words.\n"
            "   - Keep the explanation short, concrete and human, not like documentation.\n"
            "   - Then, gently ask the question again.\n"
            "5. Ask the next required question from 'Upcoming steps'.\n"
            "   - INTELLIGENT GROUPING: You MAY ask for the next 2-3 steps TOGETHER if they are naturally related (e.g. name, email, phone).\n"
            "   - If the next step is complex or unrelated, just ask one.\n"
            "   - If asking multiple, make it clear what you need.\n"
            "6. When starting a new phase or when the phase changes, briefly mention the phase name and what it covers (e.g. 'This is Phase 2: Addresses'). Keep it short.\n"
            "7. If the user's last message includes a small personal detail or emotion (e.g. 'my wife chose that', 'this is my first hotel'), start by briefly acknowledging it in a warm, one-clause comment before moving on.\n"
            "8. Keep the whole response conversational and under 3-4 sentences.\n\n"
            "Output JSON format:\n"
            "{\n"
            "  \"reply\": \"string\",\n"
            "  \"invalidated_fields\": [\"field_key1\", \"field_key2\"]\n"
            "}"
        )
        
        resp = client.models.generate_content(
            model=model, 
            contents=llm_prompt,
            config={"response_mime_type": "application/json"}
        )
        text = (resp.text or "").strip()
        if text:
            try:
                data = json.loads(text)
                return data.get("reply", ""), data.get("invalidated_fields", [])
            except json.JSONDecodeError:
                return text, []
            
    except Exception:
        pass
        
    # Fallback if LLM fails
    parts = []
    
    # Check for confusion in fallback mode
    is_confused = False
    if intent_result and intent_result.get("intent") in {"clarification", "confusion", "skip"}:
        is_confused = True
    
    if is_confused:
        parts.append("I'm sorry if that was unclear.")
        # Try to use the canned reason if available
        if reason_hint:
             parts.append(f"We need {next_step_id.split('.')[-1].replace('_', ' ')} because it {reason_hint[0].lower() + reason_hint[1:] if reason_hint else ''}")
    
    if not error and diff:
        parts.append("Got it.")
    if error and not is_confused:
        parts.append(f"Note: {error}")
        
    if effective_next:
        parts.append(prompt_for(state, effective_next))
    return "\n".join(parts), []


def explain_step(state: OnboardingState, step_id: str, user_question: str) -> str:
    question = get_question(step_id) or {}
    label = str(question.get("label") or step_id)
    qtype = str(question.get("type") or "text")
    options = [str(o) for o in (question.get("options") or []) if o is not None]
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
    model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
    reasoning = question.get("reasoning", "")
    payload = {
        "id": question.get("id"),
        "type": question.get("type"),
        "options": question.get("options"),
        "required": question.get("required"),
        "optional": question.get("optional"),
        "label": question.get("label"),
        "prompt": question.get("prompt"),
        "reasoning": reasoning,
    }
    llm_prompt = (
        "Explain the current onboarding question in 1-2 short sentences.\n"
        "Include a concrete example answer if helpful.\n"
        "Use the 'reasoning' as background context, but do not copy it word-for-word.\n"
        "Paraphrase it in friendly, natural language that feels like a human explanation.\n"
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
    if lowered in {
        "y",
        "yes",
        "yeah",
        "yep",
        "yup",
        "sure",
        "of course",
        "ok",
        "okay",
        "alright",
        "correct",
        "right",
        "sounds good",
        "looks good",
        "confirm",
        "confirmed",
        "proceed",
        "submit",
        "done",
    }:
        return True
    if any(
        phrase in lowered
        for phrase in [
            "yes i think",
            "yeah i think",
            "i think so",
            "that is fine",
            "thats fine",
            "that's fine",
            "that looks good",
            "that is correct",
            "that sounds right",
            "go ahead",
            "go for it",
        ]
    ):
        return True
    return bool(
        re.search(
            r"\b(yes|yeah|yep|yup|ok|okay|alright|sure|of course|correct|confirm|confirmed|proceed|submit|done)\b",
            lowered,
        )
    )


def _is_decline_text(text: str) -> bool:
    lowered = (text or "").strip().lower()
    if lowered in {"n", "no", "nope", "not yet", "dont", "don't"}:
        return True
    return bool(re.search(r"\b(no|nope|not\s+yet|don'?t)\b", lowered))

def _low_confidence(user_text: str, value: Any) -> bool:
    t = (user_text or "").strip().lower()
    if not t and value is None:
        return True
    markers = ["maybe", "i dont know", "i don't know", "not sure", "?"]
    if any(m in t for m in markers):
        return True
    if isinstance(value, str):
        s = value.strip().lower()
        if any(sep in s for sep in ["/", "|"]) or re.search(r"\b(or|aka)\b", s):
            return True
    return False

def apply_intent(state: OnboardingState) -> OnboardingState:
    state.last_error = None
    intent = state.last_intent or {}
    intent_type = intent.get("intent")

    if state.mode == "confirming":
        text = state.last_user_message or ""
        intent_value = intent.get("value")
        pending = state.pending_value if isinstance(state.pending_value, dict) else {}

        if isinstance(intent_value, dict) and intent_value:
            merged = dict(pending)
            merged.update(intent_value)
            state.pending_value = merged
            state.pending_data = merged
            pending = merged

        if intent_type in {"clarification", "confusion"}:
            state.last_error = "If this is not correct, please tell me what to change."
            return state

        if intent_type == "skip" or _is_decline_text(text):
            state.pending_value = None
            state.pending_data = None
            state.mode = "asking"
            state.confirmed = False
            return state

        if _is_confirm_text(text) or intent_type == "answer":
            if not pending:
                state.last_error = "There is nothing to confirm."
                return state
            for step_id, val in pending.items():
                state = _apply_answer_value(state, step_id, val)
            state.pending_value = None
            state.pending_data = None
            state.confirmed = False
            state.mode = "asking"
            return state

        state.last_error = "Please reply yes or no."
        return state
    
    if intent_type in {"correct_step", "correction"}:
        val = intent.get("value")
        if isinstance(val, dict):
            state.pending_value = val
            state.pending_data = val
            state.mode = "confirming"
            state.confirmed = False
            return state

    if intent_type == "go_to_step":
        target = intent.get("step_id")
        if target and target in QUESTIONS_BY_ID:
            state.current_step = target
            state.mode = "asking"
            return state

    if intent_type == "answer":
        val = intent.get("value")
        if isinstance(val, dict) and val:
            if len(val) == 1 and state.current_step in val:
                raw = val[state.current_step]
                q = get_question(state.current_step) or {}
                qtype = str(q.get("type") or "text")
                low = _low_confidence(state.last_user_message or "", raw)
                simple_types = {"phone", "email", "date", "number", "enum"}
                if qtype in simple_types and not low:
                    state = _apply_answer_value(state, state.current_step, raw)
                    return state
            state.pending_value = val
            state.pending_data = val
            state.mode = "confirming"
            state.confirmed = False
            return state

    if intent_type in {"clarification", "confusion", "skip"}:
        step_id = state.current_step
        q = get_question(step_id) or {}
        required = bool(q.get("required"))
        
        if intent_type == "skip" and required:
            state.last_error = "This field is required."
            return state
            
        state.mode = "clarifying"
        return state

    # Special handling for Review step completion
    if state.current_step == "review" and not state.last_error:
        text = state.last_user_message or ""
        # If user says confirm/submit/done in review step
        if _is_confirm_text(text):
             state.confirmed = True
             state.completed = True
             state.current_step = "done"
             return state
        
        # If not confirming, assume they might want to change something
        # If intent is unknown, give guidance
        if intent_type == "unknown":
            state.last_error = (
                "Tell me what you'd like to change (e.g., 'change hotel name to ...' or 'go to billing address'), "
                "or reply 'confirm' to finish."
                if _is_decline_text(text) or text
                else "Reply 'confirm' to finish, or tell me what you'd like to change."
            )

    # Fallback
    if not state.last_error and not intent_type == "unknown":
        state.last_error = "I didn't quite get that."
    
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
    label = str(question.get("label") or question.get("prompt") or step_id)
    _push_qa_history(state, step_id, label, normalized)
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
