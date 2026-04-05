"""
Smoke test — no LLM required.
Runs a perfect agent and a bad agent to verify reward logic.
"""
from app.env import SupportEnv
from app.models import SupportAction, ActionType

def run_perfect_agent(difficulty: str) -> float:
    env = SupportEnv()
    env.reset(difficulty=difficulty)
    env.step(SupportAction(action_type=ActionType.RESEARCH, reasoning="Looking up order"))
    env.step(SupportAction(action_type=ActionType.TAG, tag="refund"))
    env.step(SupportAction(action_type=ActionType.DRAFT, reasoning="Drafting reply"))
    _, reward, _, _ = env.step(SupportAction(
        action_type=ActionType.SUBMIT,
        reply="We sincerely apologize for the inconvenience with your order and account. We will immediately investigate, escalate to our senior team, and process a full refund or replacement within 24 hours. Your satisfaction is our top priority and we are deeply sorry for this experience."
    ))
    return reward

def run_bad_agent(difficulty: str) -> float:
    env = SupportEnv()
    env.reset(difficulty=difficulty)
    _, reward, _, _ = env.step(SupportAction(action_type=ActionType.SUBMIT, reply="ok"))
    return reward

if __name__ == "__main__":
    print("=== SMOKE TEST ===\n")
    for diff in ["easy", "medium", "hard"]:
        perfect = run_perfect_agent(diff)
        bad = run_bad_agent(diff)
        print(f"[{diff.upper()}] Perfect agent: {perfect} | Bad agent: {bad}")
        assert perfect > bad, f"FAIL: perfect agent should score higher than bad agent on {diff}"
        assert 0.0 <= perfect <= 1.0, f"FAIL: reward out of range on {diff}"
    print("\nAll smoke tests PASSED!")
