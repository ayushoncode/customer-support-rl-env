from pydantic import BaseModel
from typing import List, Optional


class SupportObservation(BaseModel):
    email: str
    history: List[str]
    difficulty: str
    task_id: str


class SupportAction(BaseModel):
    reply: str


class SupportState(BaseModel):
    status: str
    feedback: str
    current_task: Optional[str] = None
    difficulty: Optional[str] = None


class ResetRequest(BaseModel):
    difficulty: str = "easy"


class StepRequest(BaseModel):
    action: SupportAction
