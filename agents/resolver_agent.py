"""
Resolver Agent - Third agent in the pipeline.
Uses triage + research context to draft an empathetic, policy-compliant reply.
This is the agent that actually writes the customer-facing message.
"""

import os
from openai import OpenAI

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN     = os.getenv("HF_TOKEN", "")

FALLBACK_TEMPLATES = {
    "refund": (
        "Dear valued customer, we sincerely apologize for the inconvenience you experienced with your order. "
        "We understand your frustration and have reviewed your account carefully. "
        "We are pleased to confirm that your refund request has been approved and will be processed within 3-5 business days. "
        "You will receive a confirmation email once the refund has been initiated. "
        "We appreciate your patience and value your continued trust in us."
    ),
    "fraud": (
        "Dear valued customer, we sincerely apologize and take your concern about unauthorized account activity very seriously. "
        "We understand how distressing this situation must be for you. "
        "We have escalated your case to our fraud investigation team who will investigate and secure your account immediately. "
        "We will block any further unauthorized transactions and initiate a full refund for the fraudulent charges. "
        "Our team will contact you within 24 hours with a full update. Your security is our top priority."
    ),
    "missing": (
        "Dear valued customer, we sincerely apologize for the distress caused by your missing order. "
        "We understand how frustrating and concerning this situation is, especially given the value of your item. "
        "We have escalated this case to our senior logistics team to urgently investigate and locate your shipment. "
        "We will provide you with a full resolution — either a replacement or a complete refund — within 48 hours. "
        "We deeply appreciate your patience and are committed to making this right for you."
    ),
    "wrong_item": (
        "Dear valued customer, we sincerely apologize for receiving the wrong item. "
        "We understand your frustration and this falls short of our standards. "
        "We will arrange an immediate replacement to be sent to your address at no cost. "
        "A prepaid return label will be emailed to you for the incorrect item. "
        "We appreciate your patience and apologize for this inconvenience."
    ),
    "billing": (
        "Dear valued customer, we sincerely apologize for the billing error on your account. "
        "We have reviewed your account and confirmed the duplicate charge. "
        "A full refund for the extra charge will be processed back to your original payment method within 3-5 business days. "
        "We understand this is frustrating and we apologize for any inconvenience this has caused. "
        "Thank you for bringing this to our attention."
    ),
    "tracking": (
        "Dear valued customer, we apologize for the lack of tracking information on your order. "
        "We understand how important it is to know the status of your shipment. "
        "We have looked into your order and can confirm it has been dispatched. "
        "Your tracking information will be updated within 24 hours and you will receive a notification. "
        "Thank you for your patience and please do not hesitate to contact us if you need further assistance."
    ),
}

def _get_fallback(ticket_type: str) -> str:
    return FALLBACK_TEMPLATES.get(ticket_type, FALLBACK_TEMPLATES["tracking"])

def run(
    email: str,
    triage_result: dict,
    research_context: dict,
    lessons_prompt: str = "",
) -> dict:
    """
    Run the Resolver Agent.
    Returns a dict with the drafted reply and metadata.
    """
    ticket_type = triage_result.get("ticket_type", "tracking")
    needs_escalation = research_context.get("escalation_needed", False)
    order_info = research_context.get("order_info") or {}
    refund_eligible = research_context.get("refund_eligible") or {}
    research_notes = "\n".join(research_context.get("research_notes", []))
    policy_summary = research_context.get("policy_summary", "")

    if not HF_TOKEN:
        return {
            "draft": _get_fallback(ticket_type),
            "source": "fallback_template",
            "ticket_type": ticket_type,
            "needs_escalation": needs_escalation,
        }

    try:
        client = OpenAI(base_url=API_BASE_URL.rstrip("/"), api_key=HF_TOKEN)

        escalation_note = ""
        if needs_escalation:
            escalation_note = "⚠️ This case MUST be escalated to senior team. Mention escalation explicitly in your reply."

        refund_note = ""
        if refund_eligible:
            if refund_eligible.get("eligible"):
                refund_note = "✅ Customer IS eligible for a refund. You may offer it."
            else:
                refund_note = f"❌ Customer is NOT eligible for refund: {refund_eligible.get('reason')}. Do NOT promise a refund."

        system_prompt = f"""{lessons_prompt}
You are a professional, empathetic customer support agent.

RESEARCH CONTEXT:
{research_notes}

POLICY: {policy_summary}

{refund_note}
{escalation_note}

RULES (strictly follow):
1. Be empathetic — use words like: sorry, apologize, understand, sincerely, concern, frustrating
2. Reference the customer's order or account specifically
3. Provide a clear, actionable solution
4. Write at least 50 words
5. Do NOT make promises you cannot keep (no "100% guarantee", no "free upgrade")
6. Do NOT offer refunds if customer is ineligible
7. If escalation needed, explicitly say the case is being escalated to a senior team

Write ONLY the customer reply. No subject line, no meta-commentary.
"""

        response = client.chat.completions.create(
            model=MODEL_NAME,
            max_tokens=350,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Write a reply to this customer email:\n\n{email}"},
            ],
        )

        draft = response.choices[0].message.content or _get_fallback(ticket_type)
        return {
            "draft": draft,
            "source": "llm",
            "ticket_type": ticket_type,
            "needs_escalation": needs_escalation,
        }

    except Exception as e:
        print(f"[ResolverAgent] LLM error: {e} — using fallback template")
        return {
            "draft": _get_fallback(ticket_type),
            "source": "fallback_template",
            "ticket_type": ticket_type,
            "needs_escalation": needs_escalation,
        }
