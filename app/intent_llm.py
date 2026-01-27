import json
import os
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

OUTPUT REQUIREMENTS:
- Always return a JSON object matching the schema exactly.
- For intent="answer", set "step_id" to the CURRENT STEP.
- For intent="correct_step" or "go_to_step", set "step_id" to the target step id.
- For intent="unknown", set "step_id": null and "value": null.

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

USER MESSAGE:
"__USER_MESSAGE__"

---

EXTRACTION RULES:

- Extract ONLY fields that belong to the relevant step.
- If the user provides partial information, extract only that part.
- If the user repeats an existing value, return it again.
- If the user gives multiple fields in one sentence, extract all of them.
- If the user provides multiple fields across different steps, return ALL of them in "value" as a map of step_id -> value.
- Do NOT pack multiple fields into one value (e.g. don't put email+phone inside hotel_name).
- If the user message contains any answer for the CURRENT STEP, you MUST include __CURRENT_STEP__ in "value".
- If CURRENT STEP is enum:
  - Choose exactly one option from the step definition options.
  - Match case-insensitively (e.g. "ypr" -> "YPR").
  - If multiple options appear, return intent="unknown".
  - Never set the enum value to unrelated text.
- If CURRENT STEP is a name/text field and the message also contains phone/email/address, extract only the name portion (not the other data).
- If dates are mentioned, normalize them to YYYY-MM-DD if possible.
- If confidence is low, set confidence below 0.6 and use intent "unknown".

EXAMPLES (follow the schema exactly):
1)
CURRENT STEP: warehouse (enum: YPR, JPN)
USER MESSAGE: "ypr"
{ "intent": "answer", "step_id": "warehouse", "value": { "warehouse": "YPR" }, "confidence": 0.95 }

2)
CURRENT STEP: hotel_basic_details.hotel_name (text)
USER MESSAGE: "Hotel name is Marriott and contact is 88123413123 and marriot@gmail.com"
{ "intent": "answer", "step_id": "hotel_basic_details.hotel_name", "value": { "hotel_basic_details.hotel_name": "Marriott" }, "confidence": 0.85 }

3)
CURRENT STEP: warehouse (enum: YPR, JPN)
USER MESSAGE: "YPR and my hotel name is Marriott"
{ "intent": "answer", "step_id": "warehouse", "value": { "warehouse": "YPR", "hotel_basic_details.hotel_name": "Marriott" }, "confidence": 0.9 }
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
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
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
        from google import genai  # type: ignore
        from google.genai import types  # type: ignore
    except Exception:
        return None
    client = genai.Client(api_key=api_key)
    model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

    filtered_known = {k: v for k, v in (known_data or {}).items() if isinstance(k, str) and not k.startswith("_")}
    replacements = {
        "__CURRENT_STEP__": current_step,
        "__ALL_STEP_IDS__": json.dumps(all_step_ids, ensure_ascii=False),
        "__STEP_DEFINITIONS__": json.dumps(step_definitions or [], ensure_ascii=False),
        "__CURRENT_STEP_FIELDS__": json.dumps(current_step_fields, ensure_ascii=False),
        "__KNOWN_DATA__": json.dumps(filtered_known, ensure_ascii=False),
        "__USER_MESSAGE__": message,
    }
    prompt = PROMPT
    for k, v in replacements.items():
        prompt = prompt.replace(k, v)
    try:
        config = types.GenerateContentConfig(
            temperature=0,
            response_mime_type="application/json",
        )
        resp = (client.models.generate_content(model=model, contents=prompt, config=config).text or "").strip()
        parsed = _extract_json_object(resp)
        intent = Intent.model_validate(parsed)
        return intent
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
