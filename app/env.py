import random
from typing import Tuple, Dict, Any
from app.models import SupportObservation, SupportAction, SupportState


TASKS = {
    "easy": [
        {
            "id": "easy_refund",
            "email": "Hi, my order #4521 is delayed by 2 weeks. I'd like a full refund please.",
            "keywords": {"solution": ["refund", "reimburs"], "empathy": ["sorry", "apologize", "apologi", "understand", "inconvenien"]},
        },
        {
            "id": "easy_tracking",
            "email": "Hello, I placed an order 5 days ago and haven't received any tracking info. Can you help?",
            "keywords": {"solution": ["track", "shipment", "dispatch", "update", "status"], "empathy": ["sorry", "apologize", "apologi", "understand", "inconvenien"]},
        },
    ],
    "medium": [
        {
            "id": "medium_wrong_item",
            "email": "I received the completely wrong item and it arrived damaged. This is totally unacceptable. I need this resolved immediately.",
            "keywords": {"solution": ["replac", "return", "refund", "exchange", "resend"], "empathy": ["sorry", "apologize", "apologi", "understand", "frustrat", "inconvenien"]},
        },
        {
            "id": "medium_billing",
            "email": "I was charged twice for the same order. I want my money back NOW. This is ridiculous.",
            "keywords": {"solution": ["refund", "revers", "reimburse", "charge", "credit"], "empathy": ["sorry", "apologize", "apologi", "understand", "inconvenien"]},
        },
    ],
    "hard": [
        {
            "id": "hard_missing_laptop",
            "email": "I am absolutely furious! My $1200 laptop has been missing for 3 weeks. I have called THREE times and nothing has happened. Fix this NOW or I am calling my lawyer and reporting you to consumer protection!",
            "keywords": {"solution": ["escalat", "investigat", "replac", "refund", "manag", "compensat", "priorit"], "empathy": ["sorry", "apologize", "apologi", "understand", "frustrat", "sincer", "concern"]},
        },
        {
            "id": "hard_fraud",
            "email": "Someone used my account to make unauthorized purchases totaling $800. I've been a customer for 5 years and this is how you treat me? I want this reversed and my account secured immediately or I'm disputing with my bank!",
            "keywords": {"solution": ["secur", "block", "refund", "fraud", "investigat", "escalat", "account"], "empathy": ["sorry", "apologize", "apologi", "understand", "frustrat", "sincer", "concern", "valued"]},
        },
    ],
}


class SupportEnv:
    def __init__(self):
        self.current_task: Dict = {}
        self.difficulty: str = "easy"
        self._observation: SupportObservation = None

    def reset(self, difficulty: str = "easy") -> SupportObservation:
        self.difficulty = difficulty
        task_pool = TASKS.get(difficulty, TASKS["easy"])
        self.current_task = random.choice(task_pool)
        self._observation = SupportObservation(
            email=self.current_task["email"],
            history=[],
            difficulty=difficulty,
            task_id=self.current_task["id"],
        )
        return self._observation

    def step(self, action: SupportAction) -> Tuple[SupportState, float, bool, Dict[str, Any]]:
        reply = action.reply.lower()
        reward = 0.0
        feedback_parts = []

        empathy_words = self.current_task["keywords"]["empathy"]
        empathy_hits = sum(1 for w in empathy_words if w in reply)
        empathy_score = min(empathy_hits / max(len(empathy_words) * 0.4, 1), 1.0) * 0.3
        reward += empathy_score
        if empathy_score > 0:
            feedback_parts.append(f"Good empathy (score: {empathy_score:.2f})")

        solution_words = self.current_task["keywords"]["solution"]
        solution_hits = sum(1 for w in solution_words if w in reply)
        solution_score = min(solution_hits / max(len(solution_words) * 0.4, 1), 1.0) * 0.4
        reward += solution_score
        if solution_score > 0:
            feedback_parts.append(f"Solution addressed (score: {solution_score:.2f})")
        else:
            feedback_parts.append("No clear solution provided")

        word_count = len(reply.split())
        if word_count >= 40:
            completeness_score = 0.2
        elif word_count >= 20:
            completeness_score = 0.1
        else:
            completeness_score = 0.0
        reward += completeness_score
        if completeness_score > 0:
            feedback_parts.append(f"Response length adequate ({word_count} words)")

        personalisation_score = 0.0
        if any(w in reply for w in ["order", "account", "case", "ticket", "ref"]):
            personalisation_score += 0.05
        if any(w in reply for w in ["name", "customer", "you", "your"]):
            personalisation_score += 0.05
        reward += personalisation_score

        reward = round(min(reward, 1.0), 2)
        state = SupportState(
            status="done",
            feedback=" | ".join(feedback_parts) if feedback_parts else "Incomplete response.",
            current_task=self.current_task["id"],
            difficulty=self.difficulty,
        )
        return state, reward, True, {"word_count": word_count, "empathy_hits": empathy_hits}

    def state(self) -> SupportState:
        if not self.current_task:
            return SupportState(status="idle", feedback="No active task. Call reset() first.")
        return SupportState(
            status="active",
            feedback="Task in progress.",
            current_task=self.current_task.get("id"),
            difficulty=self.difficulty,
        )
