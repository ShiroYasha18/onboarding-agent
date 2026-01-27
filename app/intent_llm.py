import json
import os
import re
from typing import Any, Literal

from pydantic import BaseModel
from pydantic.config import ConfigDict


class Intent(BaseModel):
    model_config = ConfigDict(extra="ignore")

    intent: Literal["answer", "correct_step", "go_to_step", "unknown"]
    step_id: str | None = None
    value: dict[str, Any] | None = None
    confidence: float = 0.0
    source: Literal["llm", "heuristic", "ui", "none"] = "none"

PROMPT = """
You are an intent extraction engine for a guided hotel onboarding system.

IMPORTANT RULES:
- You do NOT decide the flow.
- You do NOT choose the next question.
- You do NOT validate business rules.
- You do NOT invent fields or steps.

Your ONLY responsibility is to understand what the user is trying to do
and extract structured information if present.

The system uses a strict step-based workflow.
Each step has predefined fields.
The orchestration layer (LangGraph) decides what happens next.

---

INTENT TYPES YOU CAN RETURN:

1. answer
   - The user is answering the current question.
   - Extract only the fields explicitly mentioned by the user.

2. correct_step
   - The user wants to correct a previously answered step.
   - Identify which step they are correcting.
   - Extract the corrected fields.

3. go_to_step
   - The user explicitly asks to go back to a specific step.
   - Do not extract field values unless clearly stated.

4. unknown
   - The message is unclear, incomplete, or unrelated.

---

STRICT OUTPUT FORMAT:
Return ONLY valid JSON.
Do NOT add explanations.
Do NOT add markdown.
Do NOT add comments.

JSON schema:

{{
  "intent": "answer | correct_step | go_to_step | unknown",
  "step_id": null or "<step_id>",
  "value": null or {{ "<step_id>": "<value>", "...": "..." }},
  "confidence": number between 0 and 1
}}

---

DYNAMIC CONTEXT (INJECTED AT RUNTIME)

CURRENT STEP:
{current_step}

KNOWN STEPS:
{all_step_ids}

KNOWN STEP DEFINITIONS:
{step_definitions}

FIELDS FOR CURRENT STEP:
{current_step_fields}

KNOWN DATA SO FAR:
{known_data}

USER MESSAGE:
"{user_message}"

---

EXTRACTION RULES:

- Extract ONLY fields that belong to the relevant step.
- If the user provides partial information, extract only that part.
- If the user repeats an existing value, return it again.
- If the user gives multiple fields in one sentence, extract all of them.
- If the user provides multiple fields across different steps, return ALL of them in "value" as a map of step_id -> value.
- Do NOT pack multiple fields into one value (e.g. don't put email+phone inside hotel_name).
- If the user message includes an answer for the CURRENT STEP, you MUST include {current_step} in "value".
- If CURRENT STEP is an enum, pick exactly one matching option and do not assign free text.
- If CURRENT STEP is a name/text field and the message also contains phone/email/address, extract only the name for that field.
- If dates are mentioned, normalize them to YYYY-MM-DD if possible.
- If confidence is low, set confidence below 0.6 and use intent "unknown".
"""

def extract_intent(
    message: str,
    *,
    current_step: str,
    all_step_ids: list[str],
    current_step_fields: list[str],
    known_data: dict[str, Any],
    step_definitions: list[dict[str, Any]] | None = None,
) -> Intent:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return Intent(intent="unknown", confidence=0.0, source="none")

    intent = _extract_with_gemini(
        api_key=api_key,
        message=message,
        current_step=current_step,
        all_step_ids=all_step_ids,
        current_step_fields=current_step_fields,
        known_data=known_data,
        step_definitions=step_definitions,
    )
    if intent is None:
        return Intent(intent="unknown", confidence=0.0, source="llm")

    intent.source = "llm"
    return intent


def _extract_with_gemini(
    *,
    api_key: str,
    message: str,
    current_step: str,
    all_step_ids: list[str],
    current_step_fields: list[str],
    known_data: dict[str, Any],
    step_definitions: list[dict[str, Any]] | None,
) -> Intent | None:
    try:
        import google.generativeai as genai  # type: ignore
    except Exception:
        return None

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(os.getenv("GEMINI_MODEL", "gemini-2.5-flash"))

    filtered_known = {k: v for k, v in (known_data or {}).items() if isinstance(k, str) and not k.startswith("_")}
    prompt = PROMPT.format(
        current_step=current_step,
        all_step_ids=json.dumps(all_step_ids, ensure_ascii=False),
        step_definitions=json.dumps(step_definitions or [], ensure_ascii=False),
        current_step_fields=json.dumps(current_step_fields, ensure_ascii=False),
        known_data=json.dumps(filtered_known, ensure_ascii=False),
        user_message=message,
    )
    try:
        resp = (model.generate_content(prompt).text or "").strip()
        parsed = _extract_json_object(resp)
        intent = Intent.model_validate(parsed)
        return intent
    except Exception:
        return None


def _extract_with_heuristics(
    *,
    message: str,
    current_step: str,
    known_data: dict[str, Any],
    steps: list[str],
    step_definitions: list[dict[str, Any]] | None,
) -> Intent:
    text = (message or "").strip()
    if not text:
        return Intent(intent="unknown", confidence=0.0)

    lowered = text.lower()

    mapped = _map_natural_language_edits(
        text=text,
        lowered=lowered,
        current_step=current_step,
        known_data=known_data,
        steps=steps,
        step_definitions=step_definitions,
    )
    if mapped is not None:
        return mapped

    if step_definitions and current_step in steps:
        step_def = _find_step_definition(current_step, step_definitions=step_definitions)
        if step_def is not None:
            m = re.search(r"^(?:my\s+)?(.+?)\s*(?:is|=|:)\s*(.+)$", text, flags=re.I)
            if m:
                field_phrase = (m.group(1) or "").strip()
                raw_value = (m.group(2) or "").strip()
                if raw_value:
                    if _phrase_matches_step(field_phrase, step_id=current_step, step_def=step_def):
                        return Intent(
                            intent="answer",
                            step_id=current_step,
                            value={current_step: _coerce_value(raw_value)},
                            confidence=0.8,
                        )

    go_to_match = re.search(r"\b(go to|goto|open|jump to)\s+([a-z0-9_.]+)\b", lowered)
    if go_to_match:
        step_id = go_to_match.group(2)
        if step_id in steps:
            return Intent(intent="go_to_step", step_id=step_id, confidence=0.7)

    correct_match = re.search(r"\b(correct|change|update)\s+([a-z0-9_.]+)\s+(to|=)\s+(.+)$", lowered)
    if correct_match:
        step_id = correct_match.group(2)
        raw_value = text[text.lower().find(correct_match.group(4).lower()) :]
        if step_id in steps:
            return Intent(intent="correct_step", step_id=step_id, value={step_id: _coerce_value(raw_value)}, confidence=0.75)

    should_be_match = re.search(r"\b([a-z0-9_.]+)\s+should be\s+(.+)$", lowered)
    if should_be_match:
        step_id = should_be_match.group(1)
        raw_value = text[text.lower().find(should_be_match.group(2).lower()) :]
        if step_id in steps:
            return Intent(intent="correct_step", step_id=step_id, value={step_id: _coerce_value(raw_value)}, confidence=0.75)

    return Intent(intent="answer", step_id=current_step, value={current_step: _coerce_value(text)}, confidence=0.75)


def _map_natural_language_edits(
    *,
    text: str,
    lowered: str,
    current_step: str,
    known_data: dict[str, Any],
    steps: list[str],
    step_definitions: list[dict[str, Any]] | None,
) -> Intent | None:
    if step_definitions:
        go_label_match = re.search(r"\b(go back to|go to|goto|open|jump to)\b\s+(?:the\s+)?(.+)$", lowered)
        if go_label_match:
            target = go_label_match.group(2).strip()
            step_id = _match_step_by_phrase(target, step_definitions=step_definitions, allowed_steps=steps)
            if step_id:
                return Intent(intent="go_to_step", step_id=step_id, confidence=0.78)

        edit_match = re.search(r"\b(change|update|correct|set|edit)\b\s+(?:my\s+)?(.+?)\s*(?:to|=)\s*(.+)$", text, flags=re.I)
        if edit_match:
            field_phrase = edit_match.group(2).strip()
            raw_value = edit_match.group(3).strip()
            direct_step = _extract_step_id_from_text(field_phrase, steps)
            step_id = direct_step or _match_step_by_phrase(field_phrase, step_definitions=step_definitions, allowed_steps=steps)
            if step_id:
                return Intent(
                    intent="correct_step",
                    step_id=step_id,
                    value={step_id: _coerce_value(raw_value)},
                    confidence=0.8,
                )

    values: dict[str, Any] = {}

    if step_definitions:
        kv_match = re.search(r"^(?:my\s+)?(.+?)\s*(?:is|=|:)\s*(.+)$", text, flags=re.I)
        if kv_match:
            field_phrase = kv_match.group(1).strip()
            phrase_tokens = _tokens(field_phrase)
            if not phrase_tokens or len(phrase_tokens) > 4:
                kv_match = None
            elif " and " in field_phrase.lower():
                kv_match = None
        if kv_match:
            raw_value = _clip_free_text_value(kv_match.group(2).strip())
            step_id = _match_step_by_phrase(field_phrase, step_definitions=step_definitions, allowed_steps=steps)
            if step_id:
                if step_id != current_step and (_looks_like_edit(lowered) or step_id in (known_data or {})):
                    return Intent(
                        intent="correct_step",
                        step_id=step_id,
                        value={step_id: _coerce_value(raw_value)},
                        confidence=0.72,
                    )
                values.setdefault(step_id, _coerce_value(raw_value))

        if "hotel name" in lowered or "hotel's name" in lowered:
            m = re.search(r"\bhotel(?:'s)?\s+name\s*(?:is|=|:)\s*([^\n\r,;]+)", text, flags=re.I)
            if m:
                step_id = _match_step_by_phrase("hotel name", step_definitions=step_definitions, allowed_steps=steps)
                if step_id:
                    values.setdefault(step_id, _coerce_value(_clip_free_text_value(m.group(1).strip())))

    enum_match = _match_enum_value(
        text=text,
        lowered=lowered,
        current_step=current_step,
        steps=steps,
        step_definitions=step_definitions,
        known_data=known_data,
    )
    if enum_match is not None and isinstance(enum_match.value, dict):
        for k, v in enum_match.value.items():
            if isinstance(k, str):
                values.setdefault(k, v)

    email_match = re.search(r"([^\s@]+@[^\s@]+\.[^\s@]+)", text)
    if email_match:
        step_id = _pick_contact_step(
            kind="email",
            lowered=lowered,
            current_step=current_step,
            known_data=known_data,
            steps=steps,
            step_definitions=step_definitions,
        )
        if step_id is not None:
            values.setdefault(step_id, email_match.group(1))

    phone_match = re.search(r"(\+?\d[\d\s\-\(\)]{6,}\d)", text)
    if phone_match:
        step_id = _pick_contact_step(
            kind="phone",
            lowered=lowered,
            current_step=current_step,
            known_data=known_data,
            steps=steps,
            step_definitions=step_definitions,
        )
        if step_id is not None:
            values.setdefault(step_id, re.sub(r"[\s\-\(\)]+", "", phone_match.group(1)))

    if values:
        return Intent(intent="answer", step_id=current_step, value=values, confidence=0.84)

    return None


def _clip_free_text_value(raw: str) -> str:
    value = (raw or "").strip()
    if not value:
        return value

    lowered = value.lower()
    markers = [
        " and the contact",
        " and contact",
        " contact details",
        " email ",
        " phone ",
        " contact is",
        " contact:",
        " email:",
        " phone:",
    ]
    cut = None
    for m in markers:
        idx = lowered.find(m)
        if idx != -1:
            cut = idx if cut is None else min(cut, idx)
    if cut is not None and cut > 0:
        return value[:cut].strip(" ,.;")

    and_idx = lowered.find(" and ")
    if and_idx != -1:
        tail = lowered[and_idx + 5 :]
        if "contact" in tail or "email" in tail or "phone" in tail or "warehouse" in tail or "@" in tail or re.search(r"\b\d{8,}\b", tail):
            return value[:and_idx].strip(" ,.;")

    return value


def _extract_step_id_from_text(text: str, steps: list[str]) -> str | None:
    candidate = (text or "").strip()
    if not candidate:
        return None
    dot = re.search(r"[a-z0-9_]+(?:\.[a-z0-9_]+)+", candidate, flags=re.I)
    if dot:
        found = dot.group(0)
        if found in steps:
            return found
    return None


def _looks_like_field_reference(lowered: str) -> bool:
    return bool(
        re.search(
            r"\b(warehouse|hotel|billing|shipping|gstin|pan|agreement|price|customer|inventory|reservation|status|email|phone|password|role)\b",
            lowered,
        )
    )


def _tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", (text or "").lower()) if t}

def _find_step_definition(step_id: str, *, step_definitions: list[dict[str, Any]]) -> dict[str, Any] | None:
    for sd in step_definitions:
        if sd.get("id") == step_id:
            return sd
    return None


def _phrase_matches_step(field_phrase: str, *, step_id: str, step_def: dict[str, Any]) -> bool:
    phrase_tokens = _tokens(field_phrase)
    if not phrase_tokens:
        return False

    sid_tokens = _tokens(step_id.replace(".", " ").replace("_", " "))
    label_tokens = _tokens(str(step_def.get("label") or ""))
    prompt_tokens = _tokens(str(step_def.get("prompt") or ""))

    candidates = sid_tokens | label_tokens | prompt_tokens
    overlap = len(phrase_tokens & candidates)
    if overlap == 0:
        return False
    if overlap >= 2:
        return True
    return bool(label_tokens and phrase_tokens.issubset(label_tokens))


def _match_step_by_phrase(
    phrase: str,
    *,
    step_definitions: list[dict[str, Any]],
    allowed_steps: list[str],
) -> str | None:
    phrase_l = (phrase or "").strip().lower()
    if not phrase_l:
        return None
    phrase_tokens = _tokens(phrase_l)
    if not phrase_tokens:
        return None

    best_id: str | None = None
    best_score = 0
    for sd in step_definitions:
        sid = sd.get("id")
        if not isinstance(sid, str) or sid not in allowed_steps:
            continue
        label = str(sd.get("label") or "")
        prompt = str(sd.get("prompt") or "")
        sid_tokens = _tokens(sid.replace(".", " ").replace("_", " "))
        candidate_tokens = sid_tokens | _tokens(label) | _tokens(prompt)
        overlap = len(candidate_tokens & phrase_tokens)
        if overlap == 0:
            continue

        label_l = label.lower()
        score = overlap * 3
        if label_l and label_l in phrase_l:
            score += 4
        if phrase_l in label_l and len(phrase_l) >= 4:
            score += 2

        if score > best_score:
            best_score = score
            best_id = sid

    if best_id is None:
        return None
    if best_score >= 6:
        return best_id
    if best_score >= 4 and len(phrase_tokens) >= 2:
        return best_id
    return None


def _looks_like_edit(lowered: str) -> bool:
    return bool(re.search(r"\b(actually|correct|change|update|edit|fix)\b", lowered))


def _pick_contact_step(
    *,
    kind: Literal["phone", "email"],
    lowered: str,
    current_step: str,
    known_data: dict[str, Any],
    steps: list[str],
    step_definitions: list[dict[str, Any]] | None,
) -> str | None:
    type_map = _types_by_step_id(step_definitions)
    if type_map:
        target_type = "email" if kind == "email" else "phone"
        candidates = [sid for sid, t in type_map.items() if t == target_type and sid in steps]
    else:
        if kind == "email":
            candidates = [sid for sid in steps if sid.endswith(".email") or sid.endswith("_email") or "email" in sid]
        else:
            candidates = [sid for sid in steps if sid.endswith(".phone") or sid.endswith("_phone") or "phone" in sid]

    if not candidates:
        return None

    if current_step in candidates:
        return current_step

    scored: list[tuple[int, str]] = []
    for sid in candidates:
        score = 0
        if sid == current_step:
            score += 100
        if "basic_details" in sid:
            score += 15
        if "contact_persons" in sid and "primary" not in lowered:
            score -= 6
        if "primary" in lowered and "primary" in sid:
            score += 20
        if "customer" in lowered and "customer" in sid:
            score += 20
        if "hotel" in lowered and "hotel" in sid:
            score += 10
        if "contact" in lowered and "contact" in sid:
            score += 12
        score += 3 if kind in sid else 0
        scored.append((score, sid))

    present = [c for c in candidates if c in known_data]
    if len(present) == 1:
        return present[0]

    scored.sort(reverse=True)
    best = scored[0][1] if scored else None
    return best


def _coerce_value(text: str) -> Any:
    candidate = (text or "").strip()
    if not candidate:
        return candidate
    if candidate.startswith("{") or candidate.startswith("["):
        try:
            return json.loads(candidate)
        except Exception:
            return candidate
    return candidate


def _extract_json_object(text: str) -> dict[str, Any]:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found")
    return json.loads(text[start : end + 1])


def _types_by_step_id(step_definitions: list[dict[str, Any]] | None) -> dict[str, str]:
    if not step_definitions:
        return {}
    out: dict[str, str] = {}
    for sd in step_definitions:
        sid = sd.get("id")
        t = sd.get("type")
        if isinstance(sid, str) and isinstance(t, str):
            out[sid] = t
    return out


def _match_enum_value(
    *,
    text: str,
    lowered: str,
    current_step: str,
    steps: list[str],
    step_definitions: list[dict[str, Any]] | None,
    known_data: dict[str, Any],
) -> Intent | None:
    if not step_definitions:
        return None

    stripped = (text or "").strip()
    if not stripped:
        return None

    candidates: list[tuple[int, str, Any]] = []
    for sd in step_definitions:
        sid = sd.get("id")
        if not isinstance(sid, str) or sid not in steps:
            continue
        if sd.get("type") != "enum":
            continue
        options = sd.get("options")
        if not isinstance(options, list) or not options:
            continue

        label = str(sd.get("label") or "")
        sid_l = sid.lower()
        label_l = label.lower()
        has_step_hint = (sid_l and re.search(rf"\\b{re.escape(sid_l)}\\b", lowered)) or (label_l and label_l in lowered)

        for opt in options:
            opt_s = str(opt).strip()
            if not opt_s:
                continue
            opt_l = opt_s.lower()
            matched = False
            if stripped.lower() == opt_l:
                matched = True
            elif len(opt_l) <= 4 and re.search(rf"\\b{re.escape(opt_l)}\\b", lowered):
                matched = True
            elif opt_l in lowered:
                matched = True

            if not matched:
                continue

            score = 0
            if sid == current_step:
                score += 100
            if has_step_hint:
                score += 20
            if sid in known_data:
                score -= 2
            if opt_l == stripped.lower():
                score += 10
            candidates.append((score, sid, opt_s))

    if not candidates:
        return None

    candidates.sort(reverse=True)
    best_score, best_sid, best_value = candidates[0]
    if len(candidates) > 1:
        second_score = candidates[1][0]
        if best_score == second_score and candidates[1][1] != best_sid:
            return None

    intent = "answer" if best_sid == current_step else "correct_step"
    confidence = 0.86 if best_sid == current_step else (0.8 if best_score >= 20 else 0.72)
    return Intent(intent=intent, step_id=best_sid, value={best_sid: best_value}, confidence=confidence)
