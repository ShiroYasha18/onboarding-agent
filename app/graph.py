import re

from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import InMemorySaver

from app import engine
from app.engine import apply_intent, get_questions
from app.intent_llm import extract_intent
from app.state import OnboardingState

_QUESTIONS = get_questions()
_ALL_STEP_IDS = [str(q.get("id")) for q in _QUESTIONS if q.get("id")]
_STEP_DEFS = [
    {
        "id": str(q.get("id")),
        "label": q.get("label"),
        "prompt": q.get("prompt"),
        "type": q.get("type"),
        "options": q.get("options"),
    }
    for q in _QUESTIONS
    if q.get("id")
]

_QUESTIONS_BY_ID = {str(q.get("id")): q for q in _QUESTIONS if q.get("id")}
_FLOW_INDEX = {step_id: i for i, step_id in enumerate(engine.FLOW)}
_EMAIL_STEP_IDS = [sid for sid, q in _QUESTIONS_BY_ID.items() if str(q.get("type")) == "email"]
_PHONE_STEP_IDS = [sid for sid, q in _QUESTIONS_BY_ID.items() if str(q.get("type")) == "phone"]


def _token_set(message: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", (message or "").lower()))


def _option_matches_tokens(option: str, tokens: set[str]) -> bool:
    opt_tokens = set(re.findall(r"[a-z0-9]+", (option or "").lower()))
    return bool(opt_tokens) and opt_tokens.issubset(tokens)


def _pick_first(step_ids: list[str], *, prefer_prefix: str | None = None) -> str | None:
    if not step_ids:
        return None
    candidates = step_ids
    if prefer_prefix:
        filtered = [sid for sid in step_ids if sid.startswith(prefer_prefix)]
        if filtered:
            candidates = filtered
    return min(candidates, key=lambda sid: _FLOW_INDEX.get(sid, 10_000_000))


def _pick_by_context(state: OnboardingState, step_ids: list[str], text_lower: str) -> str | None:
    if not step_ids:
        return None
    if state.current_step in step_ids:
        return state.current_step

    if "primary" in text_lower:
        picked = _pick_first(step_ids, prefer_prefix="hotel_contact_persons.primary.")
        if picked:
            return picked

    if "customer user" in text_lower or re.search(r"\bcustomer\s+user\b", text_lower):
        picked = _pick_first(step_ids, prefer_prefix="customer_user.")
        if picked:
            return picked

    if "shipping" in text_lower:
        picked = _pick_first(step_ids, prefer_prefix="shipping_address.")
        if picked:
            return picked

    if "billing" in text_lower:
        picked = _pick_first(step_ids, prefer_prefix="billing_address.")
        if picked:
            return picked

    if "agreement" in text_lower:
        picked = _pick_first(step_ids, prefer_prefix="agreement_details.")
        if picked:
            return picked

    if "hotel" in text_lower:
        picked = _pick_first(step_ids, prefer_prefix="hotel_basic_details.")
        if picked:
            return picked

    pending = [sid for sid in step_ids if sid not in (state.completed_steps or [])]
    picked = _pick_first(pending) or _pick_first(step_ids)
    return picked


def _looks_like_question(text: str) -> bool:
    t = (text or "").strip().lower()
    if not t:
        return False
    if "?" in t:
        return True
    return bool(re.search(r"^(what|why|how|where|which|can\s+i|do\s+i|is\s+this|meaning|means|explain)\b", t))


def _extract_hotel_name(message: str) -> str | None:
    text = (message or "").strip()
    if not text:
        return None
    m = re.search(
        r"\bhotel(?:\s+name)?\s*(?:is\s+called|called|is|=|:)\s*([^\n,;]+)",
        text,
        flags=re.IGNORECASE,
    )
    if not m:
        return None
    candidate = m.group(1).strip().strip('"').strip("'")
    candidate = re.sub(r"^(?:is\s+called|called)\s+", "", candidate, flags=re.IGNORECASE).strip()
    candidate = re.split(r"\band\b|\bcontact\b|\bphone\b|\bemail\b", candidate, maxsplit=1, flags=re.IGNORECASE)[0].strip()
    if not candidate:
        return None
    return candidate[:120]


def _extract_email(message: str) -> str | None:
    match = re.search(r"([^\s@,;]+@[^\s@,;]+\.[^\s@,;]+)", message or "")
    return match.group(1).strip() if match else None


def _extract_primary_contact_name(message: str) -> str | None:
    text = (message or "").strip()
    if not text:
        return None
    m = re.search(
        r"\bprimary\s+contact(?:\s+person)?(?:\s+name)?\s*(?:is|=|:)\s*([^\n,;]+)",
        text,
        flags=re.IGNORECASE,
    )
    if not m:
        return None
    candidate = m.group(1).strip().strip('"').strip("'")
    candidate = re.split(r"\band\b|\bphone\b|\bemail\b", candidate, maxsplit=1, flags=re.IGNORECASE)[0].strip()
    return candidate[:120] if candidate else None


def _extract_value_after(pattern: str, message: str) -> str | None:
    text = (message or "").strip()
    if not text:
        return None
    m = re.search(pattern, text, flags=re.IGNORECASE)
    if not m:
        return None
    candidate = (m.group(1) or "").strip().strip('"').strip("'")
    candidate = re.split(r"\band\b|\bphone\b|\bemail\b", candidate, maxsplit=1, flags=re.IGNORECASE)[0].strip()
    return candidate[:140] if candidate else None


def _extract_phone(message: str) -> str | None:
    match = re.search(r"(\+?\d[\d\s\-\(\)]{6,}\d)", message or "")
    if not match:
        return None
    normalized = re.sub(r"[\s\-\(\)]+", "", match.group(1))
    digits = normalized[1:] if normalized.startswith("+") else normalized
    if not digits.isdigit() or len(digits) < 8 or len(digits) > 15:
        return None
    return normalized


def _extract_gstin(message: str) -> str | None:
    # Strict 15-char format: 2 digits, 5 letters, 4 digits, 1 letter, 1 alphanum, Z, 1 alphanum
    strict = re.search(r"\b([0-9]{2}[A-Z]{5}[0-9]{4}[A-Z]{1}[1-9A-Z]{1}Z[0-9A-Z]{1})\b", (message or "").upper())
    if strict:
        return strict.group(1)
        
    m = re.search(r"\b(gstin|gst)\b[^a-z0-9]*([a-z0-9]{15})", (message or "").lower())
    return m.group(2).upper() if m else None


def _extract_pan(message: str) -> str | None:
    m = re.search(r"\bpan\b[^a-z0-9]*([a-z0-9]{8,20})", (message or "").lower())
    return m.group(1).upper() if m else None


def _heuristic_extract_values(state: OnboardingState, message: str) -> dict[str, str]:
    text = (message or "").strip()
    if not text:
        return {}
    text_lower = text.lower()
    tokens = _token_set(text)

    values: dict[str, str] = {}

    hotel_name = _extract_hotel_name(text)
    if hotel_name:
        values["hotel_basic_details.hotel_name"] = hotel_name

    gstin = _extract_gstin(text)
    if gstin:
        values["registration_details.gstin_number"] = gstin

    pan = _extract_pan(text)
    if pan:
        values["registration_details.pan_number"] = pan

    primary_name = _extract_primary_contact_name(text)
    if primary_name:
        values["hotel_contact_persons.primary.name"] = primary_name

    customer_user_name = _extract_value_after(r"\bcustomer\s+user(?:\s+name)?\s*(?:is|=|:)\s*([^\n,;]+)", text)
    if customer_user_name:
        values["customer_user.customer_name"] = customer_user_name

    billing_city = _extract_value_after(r"\bbilling\s+city\s*(?:is|=|:)\s*([^\n,;]+)", text)
    if billing_city:
        values["billing_address.city"] = billing_city

    billing_line_1 = _extract_value_after(r"\bbilling\s+address(?:\s+line\s*(?:1|one)|\s+1)?\s*(?:is|=|:)\s*([^\n;]+)", text)
    if billing_line_1:
        values["billing_address.address_line_1"] = billing_line_1

    billing_line_2 = _extract_value_after(r"\bbilling\s+address(?:\s+line\s*(?:2|two)|\s+2)?\s*(?:is|=|:)\s*([^\n;]+)", text)
    if billing_line_2:
        values["billing_address.address_line_2"] = billing_line_2

    billing_state = _extract_value_after(r"\bbilling\s+state\s*(?:is|=|:)\s*([^\n,;]+)", text)
    if billing_state:
        values["billing_address.state"] = billing_state

    billing_country = _extract_value_after(r"\bbilling\s+country\s*(?:is|=|:)\s*([^\n,;]+)", text)
    if billing_country:
        values["billing_address.country"] = billing_country

    billing_zip = _extract_value_after(r"\bbilling\s+(?:zip|postal)(?:\s+code)?\s*(?:is|=|:)\s*([a-z0-9-]{4,12})", text)
    if billing_zip:
        values["billing_address.zip_code"] = billing_zip

    shipping_city = _extract_value_after(r"\bshipping\s+city\s*(?:is|=|:)\s*([^\n,;]+)", text)
    if shipping_city:
        values["shipping_address.city"] = shipping_city

    shipping_line_1 = _extract_value_after(r"\bshipping\s+address(?:\s+line\s*(?:1|one)|\s+1)?\s*(?:is|=|:)\s*([^\n;]+)", text)
    if shipping_line_1:
        values["shipping_address.address_line_1"] = shipping_line_1

    shipping_line_2 = _extract_value_after(r"\bshipping\s+address(?:\s+line\s*(?:2|two)|\s+2)?\s*(?:is|=|:)\s*([^\n;]+)", text)
    if shipping_line_2:
        values["shipping_address.address_line_2"] = shipping_line_2

    shipping_state = _extract_value_after(r"\bshipping\s+state\s*(?:is|=|:)\s*([^\n,;]+)", text)
    if shipping_state:
        values["shipping_address.state"] = shipping_state

    shipping_country = _extract_value_after(r"\bshipping\s+country\s*(?:is|=|:)\s*([^\n,;]+)", text)
    if shipping_country:
        values["shipping_address.country"] = shipping_country

    shipping_zip = _extract_value_after(r"\bshipping\s+(?:zip|postal)(?:\s+code)?\s*(?:is|=|:)\s*([a-z0-9-]{4,12})", text)
    if shipping_zip:
        values["shipping_address.zip_code"] = shipping_zip

    email = _extract_email(text)
    if email:
        picked = _pick_by_context(state, _EMAIL_STEP_IDS, text_lower)
        if picked:
            values[picked] = email

    phone = _extract_phone(text)
    if phone:
        picked = _pick_by_context(state, _PHONE_STEP_IDS, text_lower)
        if picked:
            values[picked] = phone

    if state.current_step and state.current_step in _QUESTIONS_BY_ID:
        q = _QUESTIONS_BY_ID[state.current_step]
        if str(q.get("type")) == "enum":
            options = [str(o) for o in (q.get("options") or [])]
            matches = [opt for opt in options if _option_matches_tokens(opt, tokens)]
            if len(matches) == 1:
                values[state.current_step] = matches[0]

    if "warehouse" in _QUESTIONS_BY_ID:
        # Only extract warehouse if it's the current step OR explicitly mentioned
        is_warehouse_step = state.current_step == "warehouse"
        explicit_warehouse = "warehouse" in text_lower

        if is_warehouse_step or explicit_warehouse:
            if "ypr" in tokens and "jpn" not in tokens:
                values["warehouse"] = "YPR"
            if "jpn" in tokens and "ypr" not in tokens:
                values["warehouse"] = "JPN"

    if "agreement_details.credit_days" in _QUESTIONS_BY_ID and "credit" in text_lower:
        if "15" in tokens and "30" not in tokens:
            values["agreement_details.credit_days"] = "15"
        if "30" in tokens and "15" not in tokens:
            values["agreement_details.credit_days"] = "30"

    if "agreement_details.billing_cycle" in _QUESTIONS_BY_ID and "billing" in text_lower and "cycle" in text_lower:
        if "15" in tokens and "30" not in tokens:
            values["agreement_details.billing_cycle"] = "15"
        if "30" in tokens and "15" not in tokens:
            values["agreement_details.billing_cycle"] = "30"

    return values


def classify(state: OnboardingState) -> OnboardingState:
    if state.last_intent:
        return state

    message = (state.last_user_message or "").strip()
    if not message:
        state.last_intent = {"intent": "unknown", "confidence": 0.0}
        state.intent_source = "none"
        return state

    intent = extract_intent(
        message,
        current_step=state.current_step,
        all_step_ids=_ALL_STEP_IDS,
        current_step_fields=[state.current_step],
        known_data=state.data,
        step_definitions=_STEP_DEFS,
        pending_data=state.pending_data,
    )
    result = intent.model_dump()
    intent_type = result.get("intent")
    original_type = intent_type
    overridden = False

    if state.current_step and state.current_step in _QUESTIONS_BY_ID:
        q = _QUESTIONS_BY_ID[state.current_step]
        if str(q.get("type")) == "enum":
            options = [str(o) for o in (q.get("options") or []) if o is not None]
            tokens = _token_set(message)
            matches = [opt for opt in options if _option_matches_tokens(opt, tokens)]
            if len(matches) == 1 and intent_type not in {"correction", "go_to_step", "skip"}:
                picked = matches[0]
                result = {
                    "intent": "answer",
                    "step_id": state.current_step,
                    "value": {state.current_step: picked},
                    "confidence": result.get("confidence") or 0.99,
                    "thought": "Message clearly contains a single enum option; treating as answer.",
                    "requires_confirmation": False,
                }
                intent_type = "answer"
                overridden = True

    state.last_intent = result
    if overridden:
        state.intent_source = "heuristic"
    else:
        state.intent_source = getattr(intent, "source", None)
    return state


def mutate(state: OnboardingState) -> OnboardingState:
    return apply_intent(state)


builder = StateGraph(OnboardingState)
builder.add_node("classify", classify)
builder.add_node("mutate", mutate)

builder.set_entry_point("classify")
builder.add_edge("classify", "mutate")
builder.add_edge("mutate", END)

_checkpointer = InMemorySaver()
app_graph = builder.compile(checkpointer=_checkpointer)
