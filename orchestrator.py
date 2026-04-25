"""
orchestrator.py - The conductor of the multi-agent system.
Coordinates all 6 agents for one complete support ticket episode.
Handles the full pipeline:
  Triage → Research → Resolve → QA → (Escalation if needed) → Submit → Critic

Also interfaces with the SupportEnv (your Round 1 environment) for reward signals.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.env import SupportEnv
from app.models import SupportAction
from memory import AgentMemory

import agents.triage_agent as triage_agent
import agents.research_agent as research_agent
import agents.resolver_agent as resolver_agent
import agents.qa_agent as qa_agent
import agents.escalation_agent as escalation_agent
import agents.critic_agent as critic_agent

MAX_QA_RETRIES = 2  # How many times resolver can retry if QA rejects

def run_episode(
    difficulty: str,
    episode_num: int,
    memory: AgentMemory,
    verbose: bool = True,
) -> dict:
    """
    Run one complete episode using the multi-agent pipeline.
    Returns episode result dict with reward, transcript, and lesson.
    """
    env = SupportEnv()
    obs = env.reset(difficulty=difficulty)
    lessons_prompt = memory.get_lessons_prompt()

    def log(msg):
        if verbose:
            print(msg, flush=True)

    log(f"\n{'='*60}")
    log(f"[EPISODE {episode_num}] Difficulty: {difficulty} | Task: {obs.task_id}")
    log(f"[EMAIL] {obs.email}")
    if lessons_prompt:
        log(f"[MEMORY] Injecting {len(memory.lessons)} lessons into agents")

    total_reward = 0.0
    submit_reward = 0.0
    step_rewards = {
        "research": 0.0,
        "tag": 0.0,
        "draft": 0.0,
        "submit": 0.0,
    }
    episode_transcript = {
        "email": obs.email,
        "task_id": obs.task_id,
        "difficulty": difficulty,
        "order_info": obs.order_info,
        "triage_result": {},
        "research_context": {},
        "qa_result": {},
        "final_reply": "",
        "env_feedback": "",
        "escalation_used": False,
        "qa_retries": 0,
    }

    # ── STEP 1: RESEARCH (env action) ──────────────────────────────
    log("\n[STEP 1] RESEARCH")
    research_action = SupportAction(
        action_type="RESEARCH",
        reasoning="Gathering order and policy information",
        order_id=obs.order_info.get("order_id") if obs.order_info else None,
    )
    state, reward, done, info = env.step(research_action)
    total_reward += reward
    step_rewards["research"] = reward
    log(f"  → Reward: {reward} | Feedback: {state.feedback}")

    # ── STEP 2: TRIAGE AGENT ───────────────────────────────────────
    log("\n[STEP 2] TRIAGE AGENT")
    triage_result = triage_agent.run(obs.email, lessons_prompt=lessons_prompt)
    episode_transcript["triage_result"] = triage_result
    log(f"  → Type: {triage_result.get('ticket_type')} | Urgency: {triage_result.get('urgency')} | Escalate: {triage_result.get('needs_escalation')}")

    # ── STEP 3: TAG (env action) ───────────────────────────────────
    log("\n[STEP 3] TAG")
    tag_action = SupportAction(
        action_type="TAG",
        tag=triage_result.get("ticket_type", "general"),
        reasoning=f"Classified as {triage_result.get('ticket_type')} with {triage_result.get('urgency')} urgency",
    )
    state, reward, done, info = env.step(tag_action)
    total_reward += reward
    step_rewards["tag"] = reward
    log(f"  → Reward: {reward} | Feedback: {state.feedback}")

    # ── STEP 4: RESEARCH AGENT ─────────────────────────────────────
    log("\n[STEP 4] RESEARCH AGENT")
    # Get order_id from the task definition directly
    order_id = env.current_task.get("order_id")
    research_context = research_agent.run(
        triage_result=triage_result,
        order_id=order_id,
        email=obs.email,
    )
    episode_transcript["research_context"] = research_context
    for note in research_context.get("research_notes", []):
        log(f"  → {note}")

    # ── STEP 5: RESOLVER AGENT + QA LOOP ──────────────────────────
    log("\n[STEP 5] RESOLVER AGENT")
    final_draft = ""
    qa_result = {}

    for attempt in range(MAX_QA_RETRIES + 1):
        resolver_output = resolver_agent.run(
            email=obs.email,
            triage_result=triage_result,
            research_context=research_context,
            lessons_prompt=lessons_prompt,
        )
        draft = resolver_output["draft"]

        # QA check
        log(f"\n[STEP 5b] QA AGENT (attempt {attempt + 1})")
        qa_result = qa_agent.run(
            draft=draft,
            research_context=research_context,
            triage_result=triage_result,
        )
        log(f"  → {qa_result['qa_summary']}")
        for issue in qa_result.get("issues", []):
            log(f"    ⚠️  {issue}")
        for warning in qa_result.get("warnings", []):
            log(f"    💡 {warning}")

        if qa_result["approved"]:
            final_draft = draft
            log(f"  → QA APPROVED on attempt {attempt + 1}")
            break
        else:
            log(f"  → QA REJECTED — resolver will retry with issues in context")
            env.qa_retries += 1
            episode_transcript["qa_retries"] = env.qa_retries
            # Inject QA issues into lessons for next resolver attempt
            qa_feedback = "QA FEEDBACK: " + " | ".join(qa_result["issues"])
            lessons_prompt_with_qa = lessons_prompt + f"\n{qa_feedback}\n"
            lessons_prompt = lessons_prompt_with_qa  # update for retry

    if not final_draft:
        final_draft = draft  # use last attempt even if rejected
        log("  → Using last draft despite QA rejection")

    episode_transcript["qa_result"] = qa_result
    episode_transcript["qa_retries"] = env.qa_retries

    # ── STEP 6: ESCALATION AGENT (conditional) ────────────────────
    needs_escalation = (
        research_context.get("escalation_needed", False) or
        triage_result.get("needs_escalation", False)
    )

    if needs_escalation:
        log("\n[STEP 6] ESCALATION AGENT (triggered)")
        escalation_output = escalation_agent.run(
            email=obs.email,
            triage_result=triage_result,
            research_context=research_context,
            draft=final_draft,
            lessons_prompt=lessons_prompt,
        )
        final_draft = escalation_output["enhanced_reply"]
        episode_transcript["escalation_used"] = True
        log(f"  → Reason: {escalation_output['reason']}")
        log(f"  → Action: {escalation_output['action']}")
        log(f"  → Escalation report generated")
    else:
        log("\n[STEP 6] ESCALATION AGENT — skipped (not needed)")

    # ── STEP 7: DRAFT (env action) ─────────────────────────────────
    log("\n[STEP 7] DRAFT (env)")
    draft_action = SupportAction(
        action_type="DRAFT",
        reasoning="Draft reviewed and approved by QA agent",
    )
    state, reward, done, info = env.step(draft_action)
    total_reward += reward
    step_rewards["draft"] = reward
    log(f"  → Reward: {reward} | Feedback: {state.feedback}")

    # ── STEP 8: SUBMIT (env action) ────────────────────────────────
    log("\n[STEP 8] SUBMIT")
    log(f"  → Reply preview: {final_draft[:120]}...")
    submit_action = SupportAction(
        action_type="SUBMIT",
        reply=final_draft,
        reasoning="Submitting QA-approved, policy-compliant reply",
    )
    state, reward, done, info = env.step(submit_action)
    total_reward += reward
    submit_reward = reward
    step_rewards["submit"] = reward
    episode_transcript["final_reply"] = final_draft
    episode_transcript["env_feedback"] = state.feedback
    log(f"  → Submit reward: {reward}")
    log(f"  → Feedback: {state.feedback}")
    log(f"  → Frustration meter: {state.frustration_meter}")

    total_reward = round(min(max(total_reward, -1.0), 1.0), 3)
    log(f"\n[EPISODE {episode_num}] TOTAL REWARD: {total_reward}")

    # ── STEP 9: CRITIC AGENT (self-improvement) ────────────────────
    log("\n[STEP 9] CRITIC AGENT")
    critic_output = critic_agent.run(
        episode_transcript=episode_transcript,
        final_reward=submit_reward,
        episode_num=episode_num,
        difficulty=difficulty,
    )
    lesson = critic_output["lesson"]
    log(f"  → Lesson: {lesson}")

    # Store in memory
    memory.add_lesson(lesson, episode_num, submit_reward, difficulty)
    memory.add_episode(episode_num, total_reward, difficulty, obs.task_id)

    return {
        "episode": episode_num,
        "difficulty": difficulty,
        "task_id": obs.task_id,
        "total_reward": total_reward,
        "submit_reward": submit_reward,
        "step_rewards": step_rewards,
        "lesson": lesson,
        "transcript": episode_transcript,
        "steps": state.steps_taken,
        "frustration": state.frustration_meter,
        "feedback": state.feedback,
    }
