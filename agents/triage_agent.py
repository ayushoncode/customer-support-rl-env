"""
Triage Agent - First agent in the pipeline.
Reads the raw customer email and classifies:
  - urgency: high / medium / low
  - ticket_type: refund / fraud / billing / missing / tracking
  - recommended_difficulty: easy / medium / hard
  - needs_escalation: bool
"""

import os
import json
from openai import OpenAI

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN     = os.getenv("HF_TOKEN", "")

ESCALATION_KEYWORDS = ["lawyer", "legal", "court", "sue", "fraud", "bank dispute", "disputing"]

def _rule_based_triage(email: str) -> dict:
    """Fallback rule-based triage when LLM is unavailable."""
    email_lower = email.lower()
    needs_escalation = any(kw in email_lower for kw in ESCALATION_KEYWORDS)

    if any(w in email_lower for w in ["fraud", "unauthorized", "stolen", "disputing"]):
        ticket_type = "fraud"
        urgency = "high"
    elif any(w in email_lower for w in ["missing", "lost", "never arrived", "lawyer", "legal"]):
        ticket_type = "missing"
        urgency = "high"
    elif any(w in email_lower for w in ["wrong item", "damaged", "broken", "incorrect"]):
        ticket_type = "wrong_item"
        urgency = "medium"
    elif any(w in email_lower for w in ["charged twice", "double charge", "billing", "overcharged"]):
        ticket_type = "billing"
        urgency = "medium"
    elif any(w in email_lower for w in ["refund", "reimburs", "money back"]):
        ticket_type = "refund"
        urgency = "medium"
    else:
        ticket_type = "tracking"
        urgency = "low"

    if urgency == "high" or needs_escalation:
        difficulty = "hard"
    elif urgency == "medium":
        difficulty = "medium"
    else:
        difficulty = "easy"

    return {
        "urgency": urgency,
        "ticket_type": ticket_type,
        "difficulty": difficulty,
        "needs_escalation": needs_escalation,
        "reasoning": "Rule-based triage (no LLM)",
    }

def run(email: str, lessons_prompt: str = "") -> dict:
    """
    Run the Triage Agent.
    Returns a classification dict.
    """
    if not HF_TOKEN:
        return _rule_based_triage(email)

    try:
        client = OpenAI(base_url=API_BASE_URL.rstrip("/"), api_key=HF_TOKEN)

        system_prompt = f"""{lessons_prompt}
You are a customer support triage specialist. Analyze the customer email and classify it.

Respond ONLY with valid JSON in this exact format:
{{
  "urgency": "high|medium|low",
  "ticket_type": "refund|fraud|billing|missing|tracking|wrong_item",
  "difficulty": "easy|medium|hard",
  "needs_escalation": true|false,
  "reasoning": "one sentence explanation"
}}

Rules:
- high urgency: legal threats, fraud, missing high-value items
- medium urgency: wrong items, billing issues, standard refunds
- low urgency: tracking questions, general info
- needs_escalation: true if email mentions lawyer, fraud, legal, bank dispute
"""

        response = client.chat.completions.create(
            model=MODEL_NAME,
            max_tokens=200,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Classify this customer email:\n\n{email}"},
            ],
        )

        raw = response.choices[0].message.content or ""
        # Strip markdown code fences if present
        raw = raw.replace("```json", "").replace("```", "").strip()
        return json.loads(raw)

    except Exception as e:
        print(f"[TriageAgent] LLM error: {e} — using rule-based fallback")
        return _rule_based_triage(email)
