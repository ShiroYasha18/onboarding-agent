import json
import os
import re
from typing import Any, Literal

from pydantic import BaseModel
from pydantic.config import ConfigDict


class Intent(BaseModel):
    model_config = ConfigDict(extra="ignore")

    intent: Literal[
        "answer",
        "correction",
        "clarification",
        "confusion",
        "skip",
        "unknown",
        "go_to_step",
        "correct_step"
    ]
    step_id: str | None = None
    value: dict[str, Any] | None = None
    confidence: float = 0.0
    thought: str | None = None
    requires_confirmation: bool = False
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

INTENT CLASSIFICATION FIRST:
Classify the user's message as ONE of:
- answer: The user is providing values for the current step (and optionally other steps).
- correction: The user is correcting values already provided (identify target step(s)).
- go_to_step: The user wants to jump to a specific step without changing values.
- clarification: The user is asking about the question/field or wants more info.
- confusion: The user explicitly does not know / is unsure / shows doubt.
- skip: The user wants to skip this field (e.g., "skip", "n/a", "none", "null").
- unknown: Unclear, off-topic, or cannot classify.

Then, ONLY IF intent is "answer" or "correction", perform value extraction.

1. answer
   - The user is answering the current question.
   - Extract only the fields explicitly mentioned by the user.

2. correction
   - The user wants to correct a previously answered step/field.
   - Identify which step they are correcting and extract the corrected fields.
   - Look for phrases like "change", "update", "edit", "correct", "instead use", "make it".
   - The CURRENT STEP might be different from the step being corrected.

3. go_to_step
   - The user explicitly asks to go back to a specific step.
   - Do not extract field values unless clearly stated.

4. clarification
   - The user is asking a question about the current step.
   - The user is asking what something means (e.g. "what is this?", "means?", "explain").
   - Do NOT extract the question text as a value.

5. confusion
   - The user explicitly expresses uncertainty or lack of knowledge (e.g. "I don't know", "Not sure", "??").
   - Do NOT extract any value.

6. skip
   - The user wants to skip this field (e.g., "skip", "n/a", "none", "null").
   - Do NOT extract any value.

7. unknown
   - The message is unclear, incomplete, or unrelated.

---

STRICT OUTPUT FORMAT:
Return ONLY valid JSON.
Do NOT add explanations.
Do NOT add markdown.
Do NOT add comments.

JSON schema:

{{
  "intent": "answer | correction | go_to_step | clarification | confusion | skip | unknown",
  "step_id": null or "<step_id>",
  "value": null or {{ "<step_id>": "<value>", "...": "..." }},
  "confidence": number between 0 and 1,
  "thought": "brief reasoning about why you chose this intent",
  "requires_confirmation": boolean
}}

OUTPUT REQUIREMENTS:
- Always return a JSON object matching the schema exactly.
- For intent="answer", set "step_id" to the CURRENT STEP.
- For intent="correction" or "go_to_step", set "step_id" to the target step id.
- For intent="clarification", set "step_id" to the CURRENT STEP (or null).
- For intent="confusion", set "step_id" to the CURRENT STEP (or null).
- For intent="skip", set "step_id" to the CURRENT STEP and "value" to null.
- For intent="unknown", set "step_id": null and "value": null.
- "thought" field is MANDATORY. Explain your reasoning briefly.
- "requires_confirmation" is MANDATORY (default false). Set to true IF:
  - The user expresses uncertainty (e.g. "I think", "maybe", "not sure", "don't know full name").
  - The extracted value seems ambiguous or potentially a typo mixed with other text.
  - The user provides a partial answer but indicates they are missing info.

---

DYNAMIC CONTEXT (INJECTED AT RUNTIME)

CURRENT STEP:
__CURRENT_STEP__

KNOWN STEPS:
__ALL_STEP_IDS__

KNOWN STEP DEFINITIONS:
__STEP_DEFINITIONS__

FIELDS FOR CURRENT STEP:
__CURRENT_STEP_FIELDS__

KNOWN DATA SO FAR:
__KNOWN_DATA__

PENDING CONFIRMATION DATA:
__PENDING_DATA__

USER MESSAGE:
"__USER_MESSAGE__"

---

EXTRACTION RULES:

- Step 1: Determine intent first. If not "answer" or "correction", set "value": null.
- Extract ONLY fields that belong to the relevant step.
- If the user provides partial information, extract only that part.
- If the user repeats an existing value, return it again.
- If the user gives multiple fields in one sentence, extract all of them.
- If the user provides multiple fields across different steps, return ALL of them in "value" as a map of step_id -> value.
- If you can extract at least one value for any known step, return intent="answer" (not "unknown") unless the language clearly indicates a correction.
- For phrases like "change/update/edit/correct X to Y" or "make X Y":
  - Use intent="correction".
  - Set "step_id" to the step that best matches X based on field labels and known_data keys.
  - Put the corrected value(s) in "value" as { "<step_id>": "<Y>" }.
  - This applies even if CURRENT STEP is different (e.g. user is being asked for phone but corrects hotel name).
- Do NOT pack multiple fields into one value (e.g. don't put email+phone inside hotel_name).
- If the user message contains any answer for the CURRENT STEP, you MUST include __CURRENT_STEP__ in "value".
- If CURRENT STEP is enum:
  - Choose exactly one option from the step definition options.
  - Match case-insensitively (e.g. "ypr" -> "YPR").
  - If multiple options appear, return intent="unknown".
  - Never set the enum value to unrelated text.
- If CURRENT STEP is a name/text field:
  - If the user message is a question like "what?", "means?", "explain", or "I don't understand", return intent="clarification".
  - CRITICAL: If the input is a single word (e.g. "means", "what", "why", "que"), treat it as "clarification", NOT "answer".
  - If the message contains specific text that looks like a value, extract it.
  - If the user says "I don't have the full name" but provides a partial name, extract the partial name AND set "requires_confirmation": true.
   - PRECEDENCE: If the user expresses uncertainty (e.g., "I don't know", "not sure") BUT also provides a plausible value for the CURRENT STEP, choose intent="answer", extract the value, and set "requires_confirmation": true.
  - If the user says "I don't know", "Not sure", "I have no idea", "??":
    - Return intent="confusion".
    - Do NOT extract the text as a value.
    - Do NOT set "requires_confirmation": true (because there is no value to confirm).
  - If the user's answer contains a word that matches an enum option from another step (like "YPR"), treat it as part of the text name, NOT as the enum for the other step (unless it is a clear correction command).
- If PENDING CONFIRMATION DATA exists:
  - If the user says "Yes", "Confirm", "OK", "Sure":
    - Return intent="answer".
    - Set "value" to the PENDING CONFIRMATION DATA.
    - Set "requires_confirmation": false.
  - If the user says "No", "Incorrect", "Change it":
    - Return intent="clarification" (to re-ask) OR intent="answer" if they provide a NEW value immediately.
    - If they provide a new value, extract it and set "requires_confirmation": false (unless they are still uncertain).
- If CURRENT STEP is a name/text field and the message also contains phone/email/address, extract only the name portion (not the other data).
- If dates are mentioned, normalize them to YYYY-MM-DD if possible.
- If confidence is low, set confidence below 0.6 but still return intent "answer" if you extracted something.

EXAMPLES (follow the schema exactly):
1)
CURRENT STEP: warehouse (enum: YPR, JPN)
USER MESSAGE: "ypr"
{ "intent": "answer", "step_id": "warehouse", "value": { "warehouse": "YPR" }, "confidence": 0.95, "thought": "User explicitly selected 'ypr' which matches an enum option.", "requires_confirmation": false }

2)
CURRENT STEP: hotel_basic_details.hotel_name (text)
USER MESSAGE: "Hotel name is Marriott and contact is 88123413123 and marriot@gmail.com"
{ "intent": "answer", "step_id": "hotel_basic_details.hotel_name", "value": { "hotel_basic_details.hotel_name": "Marriott" }, "confidence": 0.85, "thought": "User provided hotel name 'Marriott' along with other contact details. Extracting only the hotel name for the current step.", "requires_confirmation": false }

3)
CURRENT STEP: warehouse (enum: YPR, JPN)
USER MESSAGE: "YPR and my hotel name is Marriott"
{ "intent": "answer", "step_id": "warehouse", "value": { "warehouse": "YPR", "hotel_basic_details.hotel_name": "Marriott" }, "confidence": 0.9, "thought": "User provided warehouse selection 'YPR' and also voluntarily provided hotel name.", "requires_confirmation": false }

4)
CURRENT STEP: registration_details.gstin_number (text)
USER MESSAGE: "29ABCDE1234F1Z5"
{ "intent": "answer", "step_id": "registration_details.gstin_number", "value": { "registration_details.gstin_number": "29ABCDE1234F1Z5" }, "confidence": 0.95, "thought": "User provided a valid GSTIN format.", "requires_confirmation": false }

5)
CURRENT STEP: hotel_basic_details.hotel_name (text)
USER MESSAGE: "I dont have the full name .. YPR holdiay inn"
{ "intent": "answer", "step_id": "hotel_basic_details.hotel_name", "value": { "hotel_basic_details.hotel_name": "YPR holdiay inn" }, "confidence": 0.8, "thought": "User provided a partial hotel name and expressed uncertainty. Setting requires_confirmation to true.", "requires_confirmation": true }

6)
CURRENT STEP: hotel_basic_details.hotel_name (text)
USER MESSAGE: "means"
{ "intent": "clarification", "step_id": "hotel_basic_details.hotel_name", "value": null, "confidence": 0.95, "thought": "User is asking for clarification ('means') instead of providing a name.", "requires_confirmation": false }

7)
CURRENT STEP: hotel_basic_details.hotel_name (text)
PENDING CONFIRMATION DATA: { "hotel_basic_details.hotel_name": "YPR holdiay inn" }
USER MESSAGE: "Yes, that's correct"
{ "intent": "answer", "step_id": "hotel_basic_details.hotel_name", "value": { "hotel_basic_details.hotel_name": "YPR holdiay inn" }, "confidence": 0.99, "thought": "User confirmed the pending value.", "requires_confirmation": false }

8)
CURRENT STEP: hotel_basic_details.hotel_name (text)
USER MESSAGE: "I don't know"
{ "intent": "confusion", "step_id": "hotel_basic_details.hotel_name", "value": null, "confidence": 0.9, "thought": "User explicitly said they don't know.", "requires_confirmation": false }

9)
CURRENT STEP: hotel_basic_details.hotel_name (text)
USER MESSAGE: "skip"
{ "intent": "skip", "step_id": "hotel_basic_details.hotel_name", "value": null, "confidence": 0.95, "thought": "User asked to skip the field.", "requires_confirmation": false }

10)
CURRENT STEP: hotel_basic_details.hotel_name (text)
USER MESSAGE: "I don't know... YPR marriot"
{ "intent": "answer", "step_id": "hotel_basic_details.hotel_name", "value": { "hotel_basic_details.hotel_name": "YPR marriot" }, "confidence": 0.75, "thought": "User expressed uncertainty but provided a plausible hotel name. Extracting with confirmation required.", "requires_confirmation": true }

11)
CURRENT STEP: hotel_basic_details.contact_phone (phone)
KNOWN DATA: { "hotel_basic_details.hotel_name": "jpn jpn" }
USER MESSAGE: "oh can you change my hotel name to Marriot"
{ "intent": "correction", "step_id": "hotel_basic_details.hotel_name", "value": { "hotel_basic_details.hotel_name": "Marriot" }, "confidence": 0.85, "thought": "User explicitly asked to change the hotel name, which is a correction to an earlier answer, not a phone number.", "requires_confirmation": false }

12)
CURRENT STEP: hotel_basic_details.contact_phone (phone)
KNOWN DATA: { "hotel_basic_details.hotel_name": "jpn jpn" }
USER MESSAGE: "change hotel name to Marriot and email to info@marriot.com"
{ "intent": "correction", "step_id": "hotel_basic_details.hotel_name", "value": { "hotel_basic_details.hotel_name": "Marriot", "hotel_basic_details.contact_email": "info@marriot.com" }, "confidence": 0.88, "thought": "User is correcting multiple previously provided fields (hotel name and email) while being asked for phone. Treat as a correction for those fields only.", "requires_confirmation": false }
"""

def extract_intent(
    message: str,
    *,
    current_step: str,
    all_step_ids: list[str],
    current_step_fields: list[str],
    known_data: dict[str, Any],
    step_definitions: list[dict[str, Any]] | None = None,
    pending_data: dict[str, Any] | None = None,
) -> Intent:
    t = (message or "").strip().lower()

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return Intent(intent="unknown", confidence=0.0, source="none")

    qtype = "text"
    if step_definitions:
        for s in step_definitions:
            if s.get("id") == current_step:
                qtype = str(s.get("type") or "text")
                break

    intent = _extract_with_gemini(
        api_key=api_key,
        message=message,
        current_step=current_step,
        all_step_ids=all_step_ids,
        current_step_fields=current_step_fields,
        known_data=known_data,
        step_definitions=step_definitions,
        pending_data=pending_data,
    )
    needs_salvage = False
    if intent is None:
        needs_salvage = True
    elif intent.intent == "unknown":
        needs_salvage = True
    elif intent.intent == "answer" and (not intent.value or not isinstance(intent.value, dict)):
        needs_salvage = True

    if needs_salvage:
        candidate = _salvage_candidate(message)
        if candidate and not _candidate_valid_for_type(candidate, qtype):
            candidate = None
        if candidate:
            base_conf = intent.confidence if intent is not None else 0.5
            conf = base_conf if 0.0 < base_conf < 1.0 else 0.5
            if conf > 0.6:
                conf = 0.6
            return Intent(
                intent="answer",
                step_id=current_step,
                value={current_step: candidate},
                confidence=conf,
                thought="Salvaged a plausible value from uncertain text.",
                requires_confirmation=True,
                source="heuristic",
            )

    if intent is None:
        return Intent(intent="unknown", confidence=0.0, source="llm")

    intent.source = "llm"
    return intent


def _extract_with_gemini(
    api_key: str,
    message: str,
    current_step: str,
    all_step_ids: list[str],
    current_step_fields: list[str],
    known_data: dict[str, Any],
    step_definitions: list[dict[str, Any]] | None,
    pending_data: dict[str, Any] | None,
) -> Intent | None:
    try:
        from google import genai  # type: ignore

        client = genai.Client(api_key=api_key)
        model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")

        prompt = PROMPT.replace("__CURRENT_STEP__", current_step)
        prompt = prompt.replace("__ALL_STEP_IDS__", json.dumps(all_step_ids))
        prompt = prompt.replace("__CURRENT_STEP_FIELDS__", json.dumps(current_step_fields))
        prompt = prompt.replace(
            "__KNOWN_DATA__", json.dumps({k: v for k, v in known_data.items() if not k.startswith("_")})
        )
        prompt = prompt.replace("__USER_MESSAGE__", message)
        
        step_defs_str = "[]"
        if step_definitions:
            step_defs_str = json.dumps(
                [s for s in step_definitions if s["id"] == current_step or s["id"] in current_step_fields]
            )
        prompt = prompt.replace("__STEP_DEFINITIONS__", step_defs_str)
        
        pending_str = "None"
        if pending_data:
            pending_str = json.dumps(pending_data)
        prompt = prompt.replace("__PENDING_DATA__", pending_str)

        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config={
                "response_mime_type": "application/json",
            },
        )
        
        text = (response.text or "").strip()
        if not text:
            return None

        data = json.loads(text)
        return Intent(**data)

    except Exception:
        return None


def _extract_json_object(text: str) -> dict[str, Any]:
    stripped = (text or "").strip()
    if not stripped:
        raise ValueError("No JSON object found")
    try:
        parsed = json.loads(stripped)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found")
    return json.loads(stripped[start : end + 1])


def _salvage_candidate(text: str) -> str | None:
    raw = (text or "").strip()
    if not raw:
        return None
    lowered = raw.lower()
    fillers = [
        "i dont know",
        "i don't know",
        "dont know",
        "idk",
        "maybe",
        "not sure",
        "something like",
        "..",
        "...",
        "?",
    ]
    cleaned = lowered
    for f in fillers:
        cleaned = cleaned.replace(f, " ")
    cleaned = " ".join(cleaned.split())
    cleaned = cleaned.strip(" .,!?:;\"'")
    if len(cleaned) <= 3:
        return None
    return cleaned


def _candidate_valid_for_type(text: str, qtype: str) -> bool:
    if not text:
        return False
    q = (qtype or "text").lower()
    if q == "enum":
        return False
    if q == "phone":
        digits = re.sub(r"\D+", "", text)
        return 8 <= len(digits) <= 15
    if q == "email":
        return bool(re.search(r"[^\s@,;]+@[^\s@,;]+\.[^\s@,;]+", text))
    if q == "number":
        stripped = str(text).strip()
        return bool(re.fullmatch(r"[\d.]+", stripped))
    if q == "date":
        stripped = str(text).strip()
        return bool(re.fullmatch(r"\d{4}-\d{2}-\d{2}", stripped))
    return True


def _looks_like_clarification_text(lowered: str) -> bool:
    t = (lowered or "").strip()
    if not t:
        return False
    if re.search(r"\bi\s+don'?t\s+understand\b", t):
        return True
    if "dont understand" in t:
        return True
    if "not sure" in t or "unsure" in t or "no idea" in t:
        return True
    core = t.strip(" ?.!")
    if core in {
        "means",
        "meaning",
        "what does this mean",
        "what does it mean",
        "can you explain",
        "explain",
        "why",
    }:
        return True
    if "what does this mean" in t or "can you explain" in t:
        return True
    return False
