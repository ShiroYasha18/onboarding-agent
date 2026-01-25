from langgraph.graph import END, StateGraph

from app.intent_llm import extract_intent
from app.state import OnboardingState
from app.engine import apply_intent, get_questions

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
    )
    state.last_intent = intent.model_dump()
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

app_graph = builder.compile()
