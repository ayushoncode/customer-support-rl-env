from pydantic import BaseModel
from typing import List, Optional
from enum import Enum

class ActionType(str, Enum):
    RESEARCH = "RESEARCH"
    TAG = "TAG"
    DRAFT = "DRAFT"
    SUBMIT = "SUBMIT"

class SupportObservation(BaseModel):
    email: str
    history: List[str]
    difficulty: str
    task_id: str
    order_info: Optional[dict] = None
    policy_snippet: Optional[str] = None
    frustration_meter: float = 0.0
    valid_actions: List[str] = ["RESEARCH", "TAG", "DRAFT", "SUBMIT"]

class SupportAction(BaseModel):
    action_type: ActionType
    reasoning: Optional[str] = None
    reply: Optional[str] = None
    order_id: Optional[str] = None
    tag: Optional[str] = None

class SupportState(BaseModel):
    status: str
    feedback: str
    current_task: Optional[str] = None
    difficulty: Optional[str] = None
    frustration_meter: float = 0.0
    steps_taken: int = 0

class ResetRequest(BaseModel):
    difficulty: str = "easy"

class StepRequest(BaseModel):
    action: SupportAction

class RunEpisodeRequest(BaseModel):
    difficulty: str = "easy"
