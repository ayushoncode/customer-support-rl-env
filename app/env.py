import random
from app.models import SupportObservation, SupportAction, SupportState
from app.database import lookup_order, check_refund_eligible, POLICY
from app.policy import check_hallucination, check_escalation_needed, check_policy_violation

EMPATHY_REFERENCE_SENTENCE = (
    "I am genuinely sorry this happened and I understand how frustrating this "
    "experience must be for you; I will help resolve it with care and urgency."
)

_EMPATHY_MODEL = None
_EMPATHY_REFERENCE_EMBEDDING = None


def _get_empathy_model():
    global _EMPATHY_MODEL, _EMPATHY_REFERENCE_EMBEDDING
    if _EMPATHY_MODEL is None:
        try:
            from sentence_transformers import SentenceTransformer

            _EMPATHY_MODEL = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
            _EMPATHY_REFERENCE_EMBEDDING = _EMPATHY_MODEL.encode(
                EMPATHY_REFERENCE_SENTENCE,
                normalize_embeddings=True,
            )
        except Exception:
            _EMPATHY_MODEL = False
            _EMPATHY_REFERENCE_EMBEDDING = None
    return _EMPATHY_MODEL


def semantic_empathy_score(reply: str) -> float:
    """
    Scores empathy with semantic similarity to a gold reference sentence.
    Returns an unweighted [0, 1] score; the reward formula applies the 0.30 weight.
    """
    clean_reply = (reply or "").strip()
    if not clean_reply:
        return 0.0

    model = _get_empathy_model()
    if model:
        reply_embedding = model.encode(clean_reply, normalize_embeddings=True)
        cosine_similarity = float(reply_embedding @ _EMPATHY_REFERENCE_EMBEDDING)
        return max(0.0, min(1.0, (cosine_similarity + 1.0) / 2.0))

    # Dependency fallback keeps local smoke tests usable before installing extras.
    reference_terms = {
        "sorry",
        "apologize",
        "understand",
        "frustrating",
        "help",
        "resolve",
        "care",
        "urgency",
    }
    reply_terms = set(clean_reply.lower().replace(".", " ").replace(",", " ").split())
    return min(len(reference_terms & reply_terms) / max(len(reference_terms) * 0.5, 1), 1.0)


TASKS = {
    "easy": [
        {"id": "easy_refund", "order_id": "ORD-4521", "email": "Hi, my order #ORD-4521 is delayed by 2 weeks. I would like a full refund please.", "keywords": {"solution": ["refund", "reimburs"], "empathy": ["sorry", "apologize", "apologi", "understand", "inconvenien"]}},
        {"id": "easy_tracking", "order_id": None, "email": "Hello, I placed an order 5 days ago and have not received any tracking info. Can you help?", "keywords": {"solution": ["track", "shipment", "dispatch", "update", "status"], "empathy": ["sorry", "apologize", "apologi", "understand", "inconvenien"]}},
    ],
    "medium": [
        {"id": "medium_wrong_item", "order_id": "ORD-3310", "email": "I received the completely wrong item and it arrived damaged. This is totally unacceptable.", "keywords": {"solution": ["replac", "return", "refund", "exchange", "resend"], "empathy": ["sorry", "apologize", "apologi", "understand", "frustrat", "inconvenien"]}},
        {"id": "medium_billing", "order_id": "ORD-6612", "email": "I was charged twice for the same order. I want my money back NOW.", "keywords": {"solution": ["refund", "revers", "reimburse", "charge", "credit"], "empathy": ["sorry", "apologize", "apologi", "understand", "inconvenien"]}},
        {
            "id": "medium_multilingual",
            "order_id": "ORD-8842",
            "email": "mera order abhi tak nahi aaya, 2 weeks ho gaye",
            "order_info": {"customer": "Neha Verma", "item": "Running Shoes", "amount": 74.99, "status": "delayed", "days_since_order": 14, "eligible_refund": True},
            "policy_snippet": "Detect mixed-language complaints. Reply in English, acknowledge the delay with extra patience, explain tracking/refund options, and avoid matching the customer's Hinglish unless asked.",
            "difficulty": "medium",
            "expected_team": "support_ops",
            "expected_actions": ["detect_language_mix", "respond_in_english", "show_extra_patience", "check_tracking", "offer_refund_or_replacement_path"],
            "keywords": {"solution": ["track", "delay", "refund", "replacement", "english", "update"], "empathy": ["sorry", "apologize", "apologi", "understand", "patient", "frustrat", "inconvenien"]},
        },
        {
            "id": "medium_social_threat",
            "order_id": "ORD-7155",
            "email": "If this is not resolved in 1 hour I am posting everything on Twitter and Reddit.",
            "order_info": {"customer": "Maya Chen", "item": "Smart Watch", "amount": 249.99, "status": "delayed", "days_since_order": 9, "eligible_refund": True},
            "policy_snippet": "For social media escalation threats, de-escalate calmly, provide a concrete update timeline, avoid defensive language, and flag the case to the PR team.",
            "difficulty": "medium",
            "expected_team": "pr_team",
            "expected_actions": ["de_escalate", "provide_concrete_timeline", "flag_pr_team", "prioritize_case", "avoid_defensive_language"],
            "keywords": {"solution": ["timeline", "hour", "update", "priorit", "escalat", "pr", "social"], "empathy": ["sorry", "apologize", "apologi", "understand", "frustrat", "concern", "urgent"]},
        },
    ],
    "hard": [
        {"id": "hard_missing_laptop", "order_id": "ORD-9921", "email": "I am absolutely furious! My 1200 dollar laptop has been missing for 3 weeks. Fix this NOW or I am calling my lawyer!", "keywords": {"solution": ["escalat", "investigat", "replac", "refund", "manag", "compensat", "priorit"], "empathy": ["sorry", "apologize", "apologi", "understand", "frustrat", "sincer", "concern"]}},
        {"id": "hard_fraud", "order_id": None, "email": "Someone used my account to make unauthorized purchases totaling 800 dollars. I am disputing with my bank!", "keywords": {"solution": ["secur", "block", "refund", "fraud", "investigat", "escalat", "account"], "empathy": ["sorry", "apologize", "apologi", "understand", "frustrat", "sincer", "concern", "valued"]}},
        {
            "id": "hard_serial_abuser",
            "order_id": "ORD-5588",
            "email": "My package is missing again. I want a refund now, same as the other times.",
            "order_info": {"customer": "Derek Stone", "item": "Bluetooth Speaker", "amount": 179.99, "status": "claimed_missing", "days_since_order": 6, "eligible_refund": False, "refund_claims_last_60_days": 4},
            "policy_snippet": "Serial refund abuse policy: if a customer has 4 or more refund claims in 60 days, deny automatic refund, explain the fraud review requirement politely but firmly, and escalate to the fraud team.",
            "difficulty": "hard",
            "expected_team": "fraud_team",
            "expected_actions": ["flag_fraud_pattern", "deny_refund_politely", "escalate_fraud_team", "document_claim_history", "offer_review_timeline"],
            "keywords": {"solution": ["fraud", "pattern", "deny", "review", "escalat", "investigat", "refund"], "empathy": ["sorry", "understand", "concern", "appreciate", "review", "firm", "care"]},
        },
        {
            "id": "hard_vip_sla",
            "order_id": "ORD-1001",
            "email": "I spend over $10,000 with you and my order is still delayed. This is unacceptable for a VIP customer.",
            "order_info": {"customer": "Anika Rao", "item": "Premium Espresso Machine", "amount": 899.99, "status": "delayed", "days_since_order": 12, "eligible_refund": True, "lifetime_value": 12480.00, "vip": True},
            "policy_snippet": "VIP SLA policy: customers with lifetime value over 10000 dollars receive fast-track handling, proactive compensation, elevated tone, and senior support ownership for delayed orders.",
            "difficulty": "hard",
            "expected_team": "vip_support",
            "expected_actions": ["fast_track_order", "offer_proactive_compensation", "use_elevated_tone", "assign_senior_owner", "provide_sla_update"],
            "keywords": {"solution": ["vip", "fast", "priority", "compensat", "senior", "sla", "escalat"], "empathy": ["sorry", "sincerely", "understand", "valued", "appreciate", "frustrat", "priority"]},
        },
    ],
}

# Difficulty multipliers — hard tasks are genuinely harder to score on
DIFFICULTY_MULTIPLIER = {"easy": 1.0, "medium": 0.88, "hard": 0.75}

class SupportEnv:
    def __init__(self):
        self.current_task = {}
        self.difficulty = "easy"
        self.frustration_meter = 0.0
        self.steps_taken = 0
        self.researched = False
        self.tagged = False
        self.drafted = False
        self.qa_retries = 0        # NEW: track QA retries
        self.episode_num = 0       # NEW: track episode for learning signal
        self._observation = None

    def reset(self, difficulty="easy"):
        self.difficulty = difficulty
        self.frustration_meter = 0.0
        self.steps_taken = 0
        self.researched = False
        self.tagged = False
        self.drafted = False
        self.qa_retries = 0
        self.episode_num += 1
        task_pool = TASKS.get(difficulty, TASKS["easy"])
        self.current_task = random.choice(task_pool)
        order_info = self.current_task.get("order_info")
        if order_info is None and self.current_task.get("order_id"):
            order_info = lookup_order(self.current_task["order_id"])
        policy_snippet = self.current_task.get(
            "policy_snippet",
            "Refund window: 30 days. Max refund: 500 dollars. Escalate orders over 200 dollars.",
        )
        self._observation = SupportObservation(
            email=self.current_task["email"],
            history=[],
            difficulty=difficulty,
            task_id=self.current_task["id"],
            order_info=order_info,
            policy_snippet=policy_snippet,
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

                # ── Empathy score ──────────────────────────────────────
                empathy_raw = semantic_empathy_score(action.reply or "")
                empathy_score = empathy_raw * 0.3
                reward += empathy_score

                # ── Solution score ─────────────────────────────────────
                solution_words = self.current_task["keywords"]["solution"]
                solution_hits = sum(1 for w in solution_words if w in reply)
                solution_score = min(solution_hits / max(len(solution_words) * 0.4, 1), 1.0) * 0.4
                reward += solution_score

                # ── Completeness ───────────────────────────────────────
                word_count = len(reply.split())
                completeness_score = 0.2 if word_count >= 40 else (0.1 if word_count >= 20 else 0.0)
                reward += completeness_score

                # ── Personalisation ────────────────────────────────────
                pers = 0.0
                if any(w in reply for w in ["order", "account", "case", "ticket"]):
                    pers += 0.05
                if any(w in reply for w in ["you", "your", "customer"]):
                    pers += 0.05
                reward += pers

                # ── Penalties ──────────────────────────────────────────
                hallucination_penalty = check_hallucination(reply)
                reward -= hallucination_penalty

                policy_penalty = check_policy_violation(reply, self.current_task.get("order_id"))
                reward -= policy_penalty

                # NEW: QA retry penalty — each retry costs 0.05
                if self.qa_retries > 0:
                    retry_penalty = self.qa_retries * 0.05
                    reward -= retry_penalty
                    feedback_parts.append(f"QA retry penalty: -{retry_penalty:.2f} ({self.qa_retries} retries)")

                frustration_delta = 4

                # NEW: difficulty multiplier — hard tasks score lower
                diff_mult = DIFFICULTY_MULTIPLIER.get(self.difficulty, 1.0)
                if diff_mult < 1.0:
                    reward = reward * diff_mult
                    feedback_parts.append(f"Difficulty modifier: x{diff_mult}")

                # ── Frustration ────────────────────────────────────────
                if solution_score < 0.2:
                    frustration_delta += 24
                elif solution_score >= 0.3:
                    frustration_delta -= 6
                if empathy_score < 0.12:
                    frustration_delta += 15
                elif empathy_score >= 0.22:
                    frustration_delta -= 5
                if completeness_score == 0.0:
                    frustration_delta += 12
                elif completeness_score >= 0.2:
                    frustration_delta -= 4
                if hallucination_penalty > 0:
                    frustration_delta += 18
                if policy_penalty > 0:
                    frustration_delta += 20
                if reward >= 0.75:
                    frustration_delta -= 18
                elif reward >= 0.55:
                    frustration_delta -= 10
                elif reward < 0.3:
                    frustration_delta += 14

                self.frustration_meter = max(0.0, min(100.0, self.frustration_meter + frustration_delta))
                feedback_parts.append(f"Frustration shift: {frustration_delta:+.0f}")
                if self.frustration_meter >= 100:
                    reward = -1.0
                    feedback_parts.append("CUSTOMER CHURNED")

                feedback_parts.append(
                    f"Empathy: {empathy_score:.2f} | Solution: {solution_score:.2f} | "
                    f"Complete: {completeness_score:.2f} | Retries: {self.qa_retries}"
                )
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
