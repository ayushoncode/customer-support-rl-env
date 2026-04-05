import os
import json
import sys
from openai import OpenAI
from app.env import SupportEnv, TASKS
from app.models import SupportAction

API_BASE_URL = os.getenv("API_BASE_URL")
MODEL_NAME   = os.getenv("MODEL_NAME")
HF_TOKEN     = os.getenv("HF_TOKEN")

if not API_BASE_URL or not MODEL_NAME or not HF_TOKEN:
    raise ValueError("Missing required environment variables: API_BASE_URL, MODEL_NAME, HF_TOKEN")

client = OpenAI(base_url=API_BASE_URL.rstrip("/"), api_key=HF_TOKEN)
env    = SupportEnv()

SYSTEM_PROMPT = """You are a professional customer support agent.
Always: acknowledge frustration, offer a concrete resolution, be polite, write at least 40 words."""

def log_start(task, difficulty, model):
    print(json.dumps({"event": "START", "task": task, "difficulty": difficulty, "model": model}), flush=True)

def log_step(step, action, reward, done, error=None):
    entry = {"event": "STEP", "step": step, "action": {"reply": action}, "reward": reward, "done": done}
    if error:
        entry["error"] = error
    print(json.dumps(entry), flush=True)

def log_end(task, score, success):
    print(json.dumps({"event": "END", "task": task, "score": score, "success": success}), flush=True)

def get_llm_reply(email):
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            max_tokens=300,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Respond to this customer email:\n\n{email}"},
            ],
        )
        return response.choices[0].message.content or "Sorry, I will assist you shortly."
    except Exception as exc:
        print(f"[DEBUG] LLM call failed: {exc}", flush=True)
        return "We sincerely apologize for the inconvenience. We will investigate your case immediately and process a full refund or replacement as required."

def run_episode(difficulty):
    obs = env.reset(difficulty=difficulty)
    log_start(task=obs.task_id, difficulty=difficulty, model=MODEL_NAME)
    ai_reply = get_llm_reply(obs.email)
    action = SupportAction(reply=ai_reply)
    state, reward, done, info = env.step(action)
    log_step(step=1, action=ai_reply, reward=reward, done=done)
    log_end(task=obs.task_id, score=reward, success=reward >= 0.5)
    return reward

if __name__ == "__main__":
    all_rewards = []
    for difficulty in ["easy", "medium", "hard"]:
        reward = run_episode(difficulty=difficulty)
        all_rewards.append(reward)
    overall = round(sum(all_rewards) / len(all_rewards), 2)
    print(json.dumps({"event": "SUMMARY", "rewards": all_rewards, "overall_score": overall}), flush=True)
    sys.exit(0)
