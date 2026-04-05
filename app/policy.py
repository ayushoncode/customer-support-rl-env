ESCALATION_KEYWORDS = ["lawyer", "legal", "court", "sue", "consumer protection", "bank dispute", "fraud"]
HALLUCINATION_PATTERNS = ["100% guarantee", "immediately credit", "free upgrade", "compensate you fully"]
DISMISS_PATTERNS = ["ignore", "not our issue", "not our problem", "cannot help"]

def check_hallucination(reply: str, order_id: str = None) -> float:
    reply_lower = reply.lower()
    penalty = 0.0
    for pattern in HALLUCINATION_PATTERNS:
        if pattern in reply_lower:
            penalty += 0.15
    for pattern in DISMISS_PATTERNS:
        if pattern in reply_lower:
            penalty += 0.25
    return min(penalty, 0.5)

def check_escalation_needed(reply: str, task: dict) -> bool:
    email = task.get("email", "").lower()
    return any(kw in email for kw in ESCALATION_KEYWORDS)

def check_policy_violation(reply: str, order_id: str = None) -> float:
    from app.database import check_refund_eligible
    penalty = 0.0
    reply_lower = reply.lower()
    if order_id and "refund" in reply_lower:
        result = check_refund_eligible(order_id)
        if not result["eligible"] and "escalate" not in result:
            penalty += 0.3
    return penalty
