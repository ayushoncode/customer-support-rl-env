"""
QA Agent - Fourth agent in the pipeline.
Reviews the Resolver Agent's draft BEFORE submission.
Catches hallucinations, policy violations, missing empathy.
Can either APPROVE or REQUEST_REVISION.
This prevents costly penalties at SUBMIT time.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.policy import check_hallucination, check_policy_violation

EMPATHY_WORDS = ["sorry", "apologize", "apologi", "understand", "sincer", "concern", "frustrat", "inconvenien"]
SOLUTION_INDICATORS = ["refund", "replac", "escalat", "investigat", "resolv", "credit", "track", "secur", "block"]

def run(draft: str, research_context: dict, triage_result: dict) -> dict:
    """
    Run the QA Agent.
    Returns a review dict with approval status and issues found.
    """
    issues = []
    warnings = []
    order_id = research_context.get("order_id")
    refund_eligible = research_context.get("refund_eligible") or {}
    draft_lower = draft.lower()

    # --- Check 1: Hallucination detection ---
    hallucination_penalty = check_hallucination(draft_lower, order_id)
    if hallucination_penalty > 0:
        issues.append(f"HALLUCINATION DETECTED (penalty would be -{hallucination_penalty:.2f}): "
                      f"Remove phrases like '100% guarantee', 'free upgrade', 'immediately credit'")

    # --- Check 2: Policy violation ---
    policy_penalty = check_policy_violation(draft_lower, order_id)
    if policy_penalty > 0:
        reason = refund_eligible.get("reason", "outside policy")
        issues.append(f"POLICY VIOLATION (penalty would be -{policy_penalty:.2f}): "
                      f"Draft offers refund but customer is ineligible — {reason}")

    # --- Check 3: Empathy check ---
    empathy_hits = sum(1 for w in EMPATHY_WORDS if w in draft_lower)
    if empathy_hits == 0:
        issues.append("NO EMPATHY DETECTED: Add apologetic/empathetic language (sorry, apologize, understand)")
    elif empathy_hits < 2:
        warnings.append("LOW EMPATHY: Consider adding more empathetic language")

    # --- Check 4: Solution check ---
    solution_hits = sum(1 for w in SOLUTION_INDICATORS if w in draft_lower)
    if solution_hits == 0:
        issues.append("NO SOLUTION DETECTED: Draft must include a concrete action (refund, replacement, escalation, etc.)")

    # --- Check 5: Length check ---
    word_count = len(draft.split())
    if word_count < 20:
        issues.append(f"TOO SHORT ({word_count} words): Draft must be at least 40 words for full score")
    elif word_count < 40:
        warnings.append(f"SHORT ({word_count} words): Consider expanding to 40+ words for completeness score")

    # --- Check 6: Escalation check ---
    needs_escalation = research_context.get("escalation_needed", False)
    if needs_escalation and "escalat" not in draft_lower:
        issues.append("MISSING ESCALATION: This case requires escalation but draft doesn't mention it")

    # --- Check 7: Personalization ---
    has_personal = any(w in draft_lower for w in ["your order", "your account", "your case"])
    if not has_personal:
        warnings.append("LOW PERSONALIZATION: Consider referencing the customer's specific order or account")

    approved = len(issues) == 0

    return {
        "approved": approved,
        "issues": issues,
        "warnings": warnings,
        "word_count": word_count,
        "empathy_hits": empathy_hits,
        "solution_hits": solution_hits,
        "estimated_penalty": hallucination_penalty + policy_penalty,
        "qa_summary": (
            f"✅ APPROVED" if approved else f"❌ REJECTED ({len(issues)} issues)"
        ) + f" | Words: {word_count} | Empathy: {empathy_hits} | Solution: {solution_hits}",
    }
