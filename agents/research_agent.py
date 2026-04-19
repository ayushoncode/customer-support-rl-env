"""
Research Agent - Second agent in the pipeline.
Given the triage output and order_id, it:
  - Looks up the order in the database
  - Fetches the relevant policy
  - Prepares a context summary for the Resolver Agent
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.database import lookup_order, check_refund_eligible, POLICY
from app.policy import check_escalation_needed

def run(triage_result: dict, order_id: str = None, email: str = "") -> dict:
    """
    Run the Research Agent.
    Returns a research context dict.
    """
    context = {
        "order_id": order_id,
        "order_info": None,
        "refund_eligible": None,
        "policy_summary": "",
        "escalation_needed": False,
        "research_notes": [],
    }

    # Look up order if we have an order_id
    if order_id:
        order_info = lookup_order(order_id)
        context["order_info"] = order_info

        if "error" not in order_info:
            refund_check = check_refund_eligible(order_id)
            context["refund_eligible"] = refund_check

            notes = []
            notes.append(f"Order {order_id}: {order_info.get('item')} — ${order_info.get('amount')}")
            notes.append(f"Status: {order_info.get('status')} | Days since order: {order_info.get('days_since_order')}")
            notes.append(f"Refund eligible: {refund_check['eligible']} — {refund_check['reason']}")

            if order_info.get("amount", 0) > POLICY["escalation_threshold"]:
                notes.append(f"⚠️ HIGH VALUE ORDER — exceeds ${POLICY['escalation_threshold']} escalation threshold")
                context["escalation_needed"] = True

            if order_info.get("status") in POLICY["replacement_eligible_statuses"]:
                notes.append(f"✅ Eligible for replacement (status: {order_info.get('status')})")

            context["research_notes"] = notes
        else:
            context["research_notes"] = ["Order not found in database"]
    else:
        context["research_notes"] = ["No order ID provided — general inquiry"]

    # Check if escalation is needed based on email content
    task_mock = {"email": email}
    if check_escalation_needed("", task_mock):
        context["escalation_needed"] = True
        context["research_notes"].append("⚠️ ESCALATION TRIGGER: Legal/fraud keywords detected in email")

    # Build policy summary
    policy_lines = [
        f"Refund window: {POLICY['refund_window_days']} days",
        f"Max refund: ${POLICY['max_refund_amount']}",
        f"Escalate orders over: ${POLICY['escalation_threshold']}",
        f"Fraud auto-block: {POLICY['fraud_auto_block']}",
    ]
    context["policy_summary"] = " | ".join(policy_lines)

    return context
