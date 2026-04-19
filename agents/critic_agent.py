"""
Critic Agent - The self-improvement engine.
Runs AFTER each episode completes.
Analyzes what went wrong or right and writes a plain-English lesson.
Lessons are stored in AgentMemory and injected into future episodes.
This is what makes the system "self-improving" — the key Round 2 feature.
"""

import os
from openai import OpenAI

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN     = os.getenv("HF_TOKEN", "")

def _rule_based_lesson(episode_transcript: dict, final_reward: float) -> str:
    """Generate a lesson without LLM based on reward and transcript analysis."""
    reward = final_reward
    qa_result = episode_transcript.get("qa_result", {})
    issues = qa_result.get("issues", [])
    warnings = qa_result.get("warnings", [])
    triage = episode_transcript.get("triage_result", {})
    ticket_type = triage.get("ticket_type", "unknown")

    if reward >= 0.8:
        return f"HIGH SCORE ({reward:.2f}) on {ticket_type} ticket. Strategy worked well: empathetic language + clear solution + correct escalation. Replicate this approach."

    lessons = []
    if reward < 0.3:
        lessons.append(f"LOW SCORE ({reward:.2f}) — major improvements needed")
    elif reward < 0.6:
        lessons.append(f"MEDIUM SCORE ({reward:.2f}) — several improvements possible")

    for issue in issues[:2]:  # top 2 issues
        if "HALLUCINATION" in issue:
            lessons.append("Avoid phrases like '100% guarantee' or 'immediately credit' — they trigger hallucination penalties")
        elif "POLICY VIOLATION" in issue:
            lessons.append(f"Check refund eligibility before offering refunds on {ticket_type} tickets")
        elif "EMPATHY" in issue:
            lessons.append("Always include empathetic words: sorry, apologize, understand, sincerely")
        elif "SOLUTION" in issue:
            lessons.append(f"For {ticket_type} tickets, always mention a concrete action: refund, replacement, or escalation")
        elif "ESCALATION" in issue:
            lessons.append("When customer mentions lawyer/fraud/legal — always mention escalation to senior team")
        elif "SHORT" in issue:
            lessons.append("Write longer replies — aim for 50+ words for full completeness score")

    for warning in warnings[:1]:
        if "PERSONALIZATION" in warning:
            lessons.append("Reference the customer's specific order ID or account to boost personalization score")

    if not lessons:
        lessons.append(f"Score {reward:.2f} on {ticket_type} — review empathy, solution completeness, and policy compliance")

    return " | ".join(lessons)

def run(
    episode_transcript: dict,
    final_reward: float,
    episode_num: int,
    difficulty: str,
) -> dict:
    """
    Run the Critic Agent.
    Returns a lesson string to be stored in AgentMemory.

    episode_transcript should contain:
      - email: the original customer email
      - triage_result: output of TriageAgent
      - research_context: output of ResearchAgent
      - qa_result: output of QAAgent
      - final_reply: the submitted reply
      - env_feedback: feedback string from environment
    """
    lesson = ""

    if HF_TOKEN:
        try:
            client = OpenAI(base_url=API_BASE_URL.rstrip("/"), api_key=HF_TOKEN)

            qa_issues = episode_transcript.get("qa_result", {}).get("issues", [])
            qa_warnings = episode_transcript.get("qa_result", {}).get("warnings", [])
            env_feedback = episode_transcript.get("env_feedback", "")
            ticket_type = episode_transcript.get("triage_result", {}).get("ticket_type", "unknown")
            final_reply = episode_transcript.get("final_reply", "")[:300]

            system_prompt = """You are an AI training coach analyzing a customer support agent's performance.
Your job is to write ONE concise lesson (max 2 sentences) that will help the agent improve next time.
Focus on the most impactful mistake or success.
Be specific — mention the ticket type, what went wrong, and what to do instead.
Do NOT write lists. Write a single actionable lesson.
"""

            user_content = f"""
Episode {episode_num} | Difficulty: {difficulty} | Final Score: {final_reward:.2f}
Ticket Type: {ticket_type}

QA Issues Found: {qa_issues}
QA Warnings: {qa_warnings}
Environment Feedback: {env_feedback}
Final Reply (first 300 chars): {final_reply}

Write ONE lesson to improve future performance:
"""
            response = client.chat.completions.create(
                model=MODEL_NAME,
                max_tokens=100,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
            )
            lesson = response.choices[0].message.content or ""
            lesson = lesson.strip().replace("\n", " ")
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
