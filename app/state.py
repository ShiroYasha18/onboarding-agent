from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional

class OnboardingState(BaseModel):
    version: str = "1.0"
    flow_name: str = "flash_hotel_onboarding"

    tenant_id: str
    session_id: str

    current_step: str
    data: Dict[str, Any] = Field(default_factory=dict)
    completed_steps: List[str] = Field(default_factory=list)

    last_user_message: Optional[str] = None
    last_intent: Optional[Dict[str, Any]] = None
    intent_source: Optional[str] = None
    last_error: Optional[str] = None
    completed: bool = False
    confirmed: bool = False
    pending_data: Optional[Dict[str, Any]] = None
    pending_value: Optional[Dict[str, Any]] = None
    mode: str = "asking"
