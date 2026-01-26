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
    mode = (os.getenv("INTENT_EXTRACTION_MODE") or ("llm_first" if api_key else "hybrid")).strip().lower()

    if mode in {"llm_first", "llm-only", "llm_only"} and api_key:
        intent = _extract_with_gemini(
            api_key=api_key,
            message=message,
            current_step=current_step,
            all_step_ids=all_step_ids,
            current_step_fields=current_step_fields,
            known_data=known_data,
            step_definitions=step_definitions,
        )
        if intent is not None:
            intent.source = "llm"
            if intent.intent != "unknown" or float(intent.confidence or 0.0) >= 0.6:
                if intent.intent == "answer" and isinstance(intent.value, dict) and intent.value:
                    supplement = _extract_with_heuristics(
                        message=message,
                        current_step=current_step,
                        known_data=known_data,
                        steps=all_step_ids,
                        step_definitions=step_definitions,
                    )
                    if supplement.intent == "answer" and isinstance(supplement.value, dict) and supplement.value:
                        merged = dict(intent.value)
                        for k, v in supplement.value.items():
                            if k not in merged:
                                merged[k] = v
                        intent.value = merged
                return intent

    heuristic = _extract_with_heuristics(
        message=message,
        current_step=current_step,
        known_data=known_data,
        steps=all_step_ids,
        step_definitions=step_definitions,
    )
    if heuristic.intent != "answer":
        heuristic.source = "heuristic"
        return heuristic

    if isinstance(heuristic.value, dict) and len(heuristic.value) > 1:
        heuristic.source = "heuristic"
        return heuristic

    if not api_key:
        heuristic.source = "heuristic"
        return heuristic

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
        heuristic.source = "heuristic"
        return heuristic
    if intent.intent == "unknown" and float(intent.confidence or 0.0) < 0.6:
        heuristic.source = "heuristic"
        return heuristic

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

        kv_match = re.search(r"^(?:my\s+)?(.+?)\s*(?:is|=|:)\s*(.+)$", text, flags=re.I)
        if kv_match:
            field_phrase = kv_match.group(1).strip()
            raw_value = kv_match.group(2).strip()
            step_id = _match_step_by_phrase(field_phrase, step_definitions=step_definitions, allowed_steps=steps)
            if step_id and step_id != current_step:
                if _looks_like_edit(lowered) or _looks_like_field_reference(field_phrase.lower()):
                    return Intent(
                        intent="correct_step",
                        step_id=step_id,
                        value={step_id: _coerce_value(raw_value)},
                        confidence=0.72,
                    )

    if "warehouse" in lowered:
        value = None
        if "ypr" in lowered:
            value = "YPR"
        elif "jpn" in lowered or "japan" in lowered or re.search(r"\bjp\b", lowered):
            value = "JPN"

        if value is not None and "warehouse" in steps:
            return Intent(intent="correct_step", step_id="warehouse", value={"warehouse": value}, confidence=0.8)

        if re.search(r"\b(change|update|edit)\b.*\bwarehouse\b", lowered) and "warehouse" in steps:
            return Intent(intent="go_to_step", step_id="warehouse", confidence=0.75)

    if current_step.endswith(".customer_type") and current_step in steps:
        mapped = _map_customer_type(lowered)
        if mapped is not None:
            return Intent(intent="answer", step_id=current_step, value={current_step: mapped}, confidence=0.8)

    email_match = re.search(r"([^\s@]+@[^\s@]+\.[^\s@]+)", text)
    if email_match:
        step_id = _pick_contact_step(
            kind="email", lowered=lowered, current_step=current_step, known_data=known_data, steps=steps
        )
        if step_id is not None:
            is_edit = "email" in lowered or "mail" in lowered or _looks_like_edit(lowered)
            if is_edit:
                return Intent(intent="correct_step", step_id=step_id, value={step_id: email_match.group(1)}, confidence=0.8)

    phone_match = re.search(r"(\+?\d[\d\s\-\(\)]{6,}\d)", text)
    if phone_match:
        step_id = _pick_contact_step(
            kind="phone", lowered=lowered, current_step=current_step, known_data=known_data, steps=steps
        )
        if step_id is not None:
            is_edit = ("phone" in lowered or "mobile" in lowered or "number" in lowered or _looks_like_edit(lowered))
            if is_edit:
                value = re.sub(r"[\s\-\(\)]+", "", phone_match.group(1))
                return Intent(intent="correct_step", step_id=step_id, value={step_id: value}, confidence=0.8)

    return None


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
    *, kind: Literal["phone", "email"], lowered: str, current_step: str, known_data: dict[str, Any], steps: list[str]
) -> str | None:
    if kind == "phone":
        candidates = [
            "hotel_basic_details.contact_phone",
            "hotel_contact_persons.primary.phone",
            "customer_user.phone",
        ]
        hints = [("primary", "hotel_contact_persons.primary.phone"), ("customer", "customer_user.phone"), ("contact", "hotel_basic_details.contact_phone")]
    else:
        candidates = [
            "hotel_basic_details.contact_email",
            "hotel_contact_persons.primary.email",
            "customer_user.email",
        ]
        hints = [("primary", "hotel_contact_persons.primary.email"), ("customer", "customer_user.email"), ("contact", "hotel_basic_details.contact_email")]

    if current_step in candidates and current_step in steps:
        return current_step

    for token, step_id in hints:
        if token in lowered and step_id in steps:
            return step_id

    present = [c for c in candidates if c in known_data and c in steps]
    if len(present) == 1:
        return present[0]

    for c in candidates:
        if c in steps:
            return c

    return None


def _map_customer_type(lowered: str) -> str | None:
    if "hotel" in lowered or "stay" in lowered or "resort" in lowered or "hostel" in lowered:
        return "HOTEL"
    if "hospital" in lowered or "clinic" in lowered or "medical" in lowered:
        return "HOSPITAL"
    if "restaurant" in lowered or "dining" in lowered or "food" in lowered:
        return "RESTAURANT"
    if "corporate" in lowered or "company" in lowered or "business" in lowered:
        return "CORPORATE"
    if "other" in lowered:
        return "OTHER"
    return None


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
