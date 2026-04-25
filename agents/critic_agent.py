"""
Critic Agent - The self-improvement engine.
Runs AFTER each episode completes.
Analyzes what went wrong or right and writes a plain-English lesson.
Lessons are stored in AgentMemory and injected into future episodes.
"""

import os
from openai import OpenAI

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN     = os.getenv("HF_TOKEN", "")

# Specific lessons per ticket type — shown when score is high
HIGH_SCORE_LESSONS = {
    "refund": "Refund tickets: always state exact timeline ('3-5 business days') and confirm eligibility explicitly — vague promises reduce trust and future scores.",
    "fraud": "Fraud tickets: mention account security lock BEFORE refund — customers need safety assurance first, then financial resolution.",
    "missing": "Missing item tickets: commit to a specific SLA ('48 hours') and name the senior team — vague escalations score lower than concrete ones.",
    "billing": "Billing tickets: explicitly confirm 'I found the duplicate charge' before offering resolution — validation reduces frustration meter.",
    "wrong_item": "Wrong item tickets: offer replacement AND prepaid return label in the same sentence — removing all friction maximises solution score.",
    "tracking": "Tracking tickets: give exact timeframe ('within 24 hours') and mention notification email — specific > vague every time.",
}

# Specific lessons per ticket type — shown when score is low/medium
LOW_SCORE_LESSONS = {
    "refund": "MISSED: Did not confirm refund eligibility or timeline. Next time: check order date against 30-day policy, then state exact processing time.",
    "fraud": "MISSED: Did not mention account security actions. Next time: always say 'your account has been secured' before discussing refund.",
    "missing": "MISSED: No escalation or SLA commitment. Next time: explicitly escalate to senior team and commit to 48-hour resolution.",
    "billing": "MISSED: Did not confirm the duplicate charge was found. Next time: say 'I have confirmed the duplicate charge on your account' first.",
    "wrong_item": "MISSED: Incomplete solution. Next time: offer both replacement AND prepaid return label — half solutions score 0.4 not 0.8.",
    "tracking": "MISSED: Vague response on tracking. Next time: give specific timeframe and confirm dispatch status from the order database.",
}

def _rule_based_lesson(episode_transcript: dict, final_reward: float) -> str:
    """Generate a specific, actionable lesson based on reward and ticket type."""
    qa_result = episode_transcript.get("qa_result", {})
    issues = qa_result.get("issues", [])
    warnings = qa_result.get("warnings", [])
    triage = episode_transcript.get("triage_result", {})
    ticket_type = triage.get("ticket_type", "general")
    qa_retries = episode_transcript.get("qa_retries", 0)
    difficulty = episode_transcript.get("difficulty", "easy")

    lessons = []

    # High score — give specific tip to maintain performance
    if final_reward >= 0.8:
        base = HIGH_SCORE_LESSONS.get(ticket_type, 
            f"Score {final_reward:.2f}: empathy + solution + escalation worked. "
            f"Key: always reference order ID and use customer name for personalisation boost.")
        if qa_retries > 0:
            base += f" Note: {qa_retries} QA retries cost -{qa_retries * 0.05:.2f} — aim for first-pass approval next time."
        return base

    # Medium score
    if final_reward >= 0.5:
        base = LOW_SCORE_LESSONS.get(ticket_type,
            f"Score {final_reward:.2f} on {ticket_type}: partial credit only.")
        extras = []
        for issue in issues[:2]:
            if "HALLUCINATION" in issue:
                extras.append("Remove impossible promises — '100% guarantee' and 'free upgrade' trigger -0.15 penalty each.")
            elif "POLICY VIOLATION" in issue:
                extras.append("Policy violated — always check refund window (30 days) and amount (<$500) before offering refund.")
            elif "EMPATHY" in issue:
                extras.append("Add empathy words: sorry, apologize, understand, sincerely — minimum 2 hits needed for full score.")
            elif "ESCALATION" in issue:
                extras.append("Legal/fraud keywords detected but escalation missing — always mention senior team for these cases.")
            elif "SHORT" in issue:
                extras.append("Reply too short — write 50+ words for full completeness score (0.20).")
        if extras:
            base += " | " + " | ".join(extras[:2])
        if qa_retries > 0:
            base += f" | {qa_retries} QA retries added -{qa_retries * 0.05:.2f} penalty."
        return base

    # Low score
    base = f"LOW SCORE ({final_reward:.2f}) on {difficulty} {ticket_type} ticket. "
    if issues:
        top_issue = issues[0]
        if "POLICY" in top_issue:
            base += "Critical: policy violation detected. Never offer refund without checking eligibility first."
        elif "HALLUCINATION" in top_issue:
            base += "Critical: hallucination penalty triggered. Remove guarantee/credit/upgrade language entirely."
        elif "EMPATHY" in top_issue:
            base += "Critical: zero empathy detected. Start reply with 'I sincerely apologize' — required for any score."
        elif "SOLUTION" in top_issue:
            base += f"Critical: no solution offered for {ticket_type}. Must include: " + {
                "refund": "refund confirmation",
                "fraud": "account security + investigation",
                "missing": "escalation + replacement/refund",
                "billing": "duplicate charge reversal",
                "wrong_item": "replacement + return label",
                "tracking": "tracking update timeline",
            }.get(ticket_type, "concrete action")
        else:
            base += "Review: empathy keywords, solution completeness, and word count (40+ words needed)."
    else:
        base += LOW_SCORE_LESSONS.get(ticket_type, "Improve empathy + solution coverage.")

    return base

def run(
    episode_transcript: dict,
    final_reward: float,
    episode_num: int,
    difficulty: str,
) -> dict:
    """
    Run the Critic Agent.
    Returns a specific, actionable lesson string to store in AgentMemory.
    """
    lesson = ""

    # Add difficulty and retries to transcript for rule-based fallback
    episode_transcript["difficulty"] = difficulty
    episode_transcript["qa_retries"] = episode_transcript.get("qa_retries", 0)

    if HF_TOKEN:
        try:
            client = OpenAI(base_url=API_BASE_URL.rstrip("/"), api_key=HF_TOKEN)

            qa_issues = episode_transcript.get("qa_result", {}).get("issues", [])
            qa_warnings = episode_transcript.get("qa_result", {}).get("warnings", [])
            env_feedback = episode_transcript.get("env_feedback", "")
            ticket_type = episode_transcript.get("triage_result", {}).get("ticket_type", "unknown")
            final_reply = episode_transcript.get("final_reply", "")[:300]
            qa_retries = episode_transcript.get("qa_retries", 0)

            system_prompt = """You are an AI training coach analyzing a customer support agent's performance.
Write ONE specific, actionable lesson (1-2 sentences) to improve next time.
Be concrete — name the ticket type, the specific mistake, and exactly what to do differently.
Never write generic advice like "be more empathetic". 
Example good lesson: "For fraud tickets, always say 'your account has been secured' before mentioning refund — security reassurance scores 0.15 higher than jumping straight to money."
Example bad lesson: "The agent should improve empathy and solution quality."
"""

            user_content = f"""Episode {episode_num} | Difficulty: {difficulty} | Score: {final_reward:.2f}
Ticket Type: {ticket_type}
QA Retries: {qa_retries}
QA Issues: {qa_issues}
QA Warnings: {qa_warnings}
Env Feedback: {env_feedback}
Reply (300 chars): {final_reply}

Write ONE specific lesson:"""

            response = client.chat.completions.create(
                model=MODEL_NAME,
                max_tokens=120,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
            )
            lesson = response.choices[0].message.content or ""
            lesson = lesson.strip().replace("\n", " ")
            if not lesson:
                lesson = _rule_based_lesson(episode_transcript, final_reward)
        except Exception as e:
            print(f"[CriticAgent] LLM error: {e} — using rule-based lesson")
            lesson = _rule_based_lesson(episode_transcript, final_reward)
    else:
        lesson = _rule_based_lesson(episode_transcript, final_reward)

    return {
        "lesson": lesson,
        "episode": episode_num,
        "reward": final_reward,
        "difficulty": difficulty,
    }