import random
from typing import Tuple, Dict, Any
from app.models import SupportObservation, SupportAction, SupportState
from app.database import lookup_order, check_refund_eligible, POLICY
from app.policy import check_hallucination, check_escalation_needed, check_policy_violation

TASKS = {
    "easy": [
        {"id": "easy_refund", "order_id": "ORD-4521", "email": "Hi, my order #ORD-4521 is delayed by 2 weeks. I would like a full refund please.", "keywords": {"solution": ["refund", "reimburs"], "empathy": ["sorry", "apologize", "apologi", "understand", "inconvenien"]}},
        {"id": "easy_tracking", "order_id": None, "email": "Hello, I placed an order 5 days ago and have not received any tracking info. Can you help?", "keywords": {"solution": ["track", "shipment", "dispatch", "update", "status"], "empathy": ["sorry", "apologize", "apologi", "understand", "inconvenien"]}},
    ],
    "medium": [
        {"id": "medium_wrong_item", "order_id": "ORD-3310", "email": "I received the completely wrong item and it arrived damaged. This is totally unacceptable.", "keywords": {"solution": ["replac", "return", "refund", "exchange", "resend"], "empathy": ["sorry", "apologize", "apologi", "understand", "frustrat", "inconvenien"]}},
        {"id": "medium_billing", "order_id": "ORD-6612", "email": "I was charged twice for the same order. I want my money back NOW.", "keywords": {"solution": ["refund", "revers", "reimburse", "charge", "credit"], "empathy": ["sorry", "apologize", "apologi", "understand", "inconvenien"]}},
    ],
    "hard": [
        {"id": "hard_missing_laptop", "order_id": "ORD-9921", "email": "I am absolutely furious! My 1200 dollar laptop has been missing for 3 weeks. Fix this NOW or I am calling my lawyer!", "keywords": {"solution": ["escalat", "investigat", "replac", "refund", "manag", "compensat", "priorit"], "empathy": ["sorry", "apologize", "apologi", "understand", "frustrat", "sincer", "concern"]}},
        {"id": "hard_fraud", "order_id": None, "email": "Someone used my account to make unauthorized purchases totaling 800 dollars. I am disputing with my bank!", "keywords": {"solution": ["secur", "block", "refund", "fraud", "investigat", "escalat", "account"], "empathy": ["sorry", "apologize", "apologi", "understand", "frustrat", "sincer", "concern", "valued"]}},
    ],
}

class SupportEnv:
    def __init__(self):
        self.current_task = {}
        self.difficulty = "easy"
        self.frustration_meter = 0.0
        self.steps_taken = 0
        self.researched = False
        self.tagged = False
        self.drafted = False
        self._observation = None

    def reset(self, difficulty="easy"):
        self.difficulty = difficulty
        self.frustration_meter = 0.0
        self.steps_taken = 0
        self.researched = False
        self.tagged = False
        self.drafted = False
        task_pool = TASKS.get(difficulty, TASKS["easy"])
        self.current_task = random.choice(task_pool)
        order_info = None
        if self.current_task.get("order_id"):
            order_info = lookup_order(self.current_task["order_id"])
        self._observation = SupportObservation(
            email=self.current_task["email"],
            history=[],
            difficulty=difficulty,
            task_id=self.current_task["id"],
            order_info=order_info,
            policy_snippet="Refund window: 30 days. Max refund: 500 dollars. Escalate orders over 200 dollars.",
            frustration_meter=self.frustration_meter,
            valid_actions=["RESEARCH", "TAG", "DRAFT", "SUBMIT"],
        )
        return self._observation

    def step(self, action):
        self.steps_taken += 1
        reward = 0.0
        done = False
        feedback_parts = []

        if action.action_type == "RESEARCH":
            self.researched = True
            reward = 0.1
            feedback_parts.append("Research step completed (+0.1)")

        elif action.action_type == "TAG":
            self.tagged = True
            reward = 0.1
            feedback_parts.append("Tagging step completed (+0.1)")

        elif action.action_type == "DRAFT":
            if not self.researched:
                self.frustration_meter += 10
                reward = -0.05
                feedback_parts.append("Penalty: drafted without researching first")
            else:
                self.drafted = True
                reward = 0.15
                feedback_parts.append("Draft step completed (+0.15)")

        elif action.action_type == "SUBMIT":
            if not self.drafted:
                reward = -0.1
                self.frustration_meter += 15
                feedback_parts.append("Penalty: submitted without drafting")
                done = True
            else:
                reply = (action.reply or "").lower()
                empathy_words = self.current_task["keywords"]["empathy"]
                empathy_hits = sum(1 for w in empathy_words if w in reply)
                empathy_score = min(empathy_hits / max(len(empathy_words) * 0.4, 1), 1.0) * 0.3
                reward += empathy_score
                solution_words = self.current_task["keywords"]["solution"]
                solution_hits = sum(1 for w in solution_words if w in reply)
                solution_score = min(solution_hits / max(len(solution_words) * 0.4, 1), 1.0) * 0.4
                reward += solution_score
                word_count = len(reply.split())
                completeness_score = 0.2 if word_count >= 40 else (0.1 if word_count >= 20 else 0.0)
                reward += completeness_score
                pers = 0.0
                if any(w in reply for w in ["order", "account", "case", "ticket"]):
                    pers += 0.05
                if any(w in reply for w in ["you", "your", "customer"]):
                    pers += 0.05
                reward += pers
                hallucination_penalty = check_hallucination(reply)
                reward -= hallucination_penalty
                policy_penalty = check_policy_violation(reply, self.current_task.get("order_id"))
                reward -= policy_penalty
                if solution_score < 0.2:
                    self.frustration_meter += 20
                if self.frustration_meter >= 100:
                    reward = -1.0
                    feedback_parts.append("CUSTOMER CHURNED")
                feedback_parts.append(f"Empathy: {empathy_score:.2f} | Solution: {solution_score:.2f} | Complete: {completeness_score:.2f}")
                done = True

        reward = round(min(max(reward, -1.0), 1.0), 2)
        state = SupportState(
            status="done" if done else "active",
            feedback=" | ".join(feedback_parts),
            current_task=self.current_task.get("id"),
            difficulty=self.difficulty,
            frustration_meter=self.frustration_meter,
            steps_taken=self.steps_taken,
        )
        return state, reward, done, {"steps_taken": self.steps_taken, "frustration": self.frustration_meter}

    def state(self):
        if not self.current_task:
            return SupportState(status="idle", feedback="No active task.", frustration_meter=0.0, steps_taken=0)
        return SupportState(
            status="active",
            feedback="Task in progress.",
            current_task=self.current_task.get("id"),
            difficulty=self.difficulty,
            frustration_meter=self.frustration_meter,
            steps_taken=self.steps_taken,
        )
