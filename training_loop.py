"""
training_loop.py - The main training script.
Runs N episodes across all difficulties, tracks reward improvement,
injects critic lessons between episodes, and prints improvement curves.

Usage:
  python training_loop.py                    # 15 episodes, all difficulties
  python training_loop.py --episodes 30      # 30 episodes
  python training_loop.py --difficulty easy  # easy only
  python training_loop.py --reset-memory     # clear lessons and start fresh
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from memory import AgentMemory
from orchestrator import run_episode

DIFFICULTIES = ["easy", "medium", "hard"]

def print_reward_curve(rewards: list, difficulties: list):
    """ASCII reward curve for terminal display."""
    if not rewards:
        return
    print("\n" + "="*60)
    print("📈 REWARD CURVE (self-improvement over episodes)")
    print("="*60)
    max_r = max(rewards) if rewards else 1.0
    min_r = min(rewards) if rewards else 0.0
    bar_width = 40

    for i, (r, d) in enumerate(zip(rewards, difficulties)):
        # Normalize to bar width
        normalized = (r - min_r) / (max_r - min_r + 1e-6)
        bar_len = int(normalized * bar_width)
        bar = "█" * bar_len + "░" * (bar_width - bar_len)
        diff_icon = {"easy": "🟢", "medium": "🟡", "hard": "🔴"}.get(d, "⚪")
        print(f"  Ep {i+1:02d} {diff_icon} [{bar}] {r:+.3f}")

    print("="*60)
    if len(rewards) >= 3:
        first_third = rewards[:len(rewards)//3]
        last_third = rewards[-(len(rewards)//3):]
        early_avg = sum(first_third) / len(first_third)
        late_avg = sum(last_third) / len(last_third)
        improvement = late_avg - early_avg
        trend = "📈 IMPROVING" if improvement > 0.05 else ("📉 DECLINING" if improvement < -0.05 else "➡️  STABLE")
        print(f"  Early avg: {early_avg:.3f} | Recent avg: {late_avg:.3f} | Trend: {trend} ({improvement:+.3f})")
    print("="*60 + "\n")

def print_lessons_summary(memory: AgentMemory):
    """Print all lessons learned so far."""
    if not memory.lessons:
        print("\n[MEMORY] No lessons stored yet.")
        return
    print(f"\n{'='*60}")
    print(f"🧠 LESSONS LEARNED ({len(memory.lessons)} total)")
    print("="*60)
    for i, entry in enumerate(memory.lessons, 1):
        print(f"  {i}. [Ep {entry['episode']} | {entry['reward']:.2f} | {entry['difficulty']}]")
        print(f"     {entry['lesson']}")
    print("="*60 + "\n")

def run_training(
    total_episodes: int = 15,
    difficulty_filter: str = None,
    reset_memory: bool = False,
    verbose: bool = True,
):
    """Main training loop."""
    memory = AgentMemory(max_lessons=10)

    if reset_memory:
        print("[TRAINING] Resetting memory — starting fresh")
        memory.clear()
        memory = AgentMemory(max_lessons=10)

    print(f"\n{'='*60}")
    print(f"🎧 MULTI-AGENT CUSTOMER SUPPORT RL TRAINING")
    print(f"{'='*60}")
    print(f"Episodes: {total_episodes}")
    print(f"Difficulty: {difficulty_filter or 'rotating (easy→medium→hard)'}")
    print(f"Memory: {memory.summary()}")
    print(f"API: {os.getenv('API_BASE_URL', 'NOT SET')} | Model: {os.getenv('MODEL_NAME', 'NOT SET')}")
    print(f"HF Token: {'SET ✅' if os.getenv('HF_TOKEN') else 'NOT SET ❌ (will use fallback templates)'}")
    print("="*60 + "\n")

    all_rewards = []
    all_difficulties = []
    results_by_difficulty = {"easy": [], "medium": [], "hard": []}
    start_time = time.time()

    for episode in range(1, total_episodes + 1):
        # Rotate difficulties if no filter
        if difficulty_filter:
            difficulty = difficulty_filter
        else:
            difficulty = DIFFICULTIES[(episode - 1) % len(DIFFICULTIES)]

        try:
            result = run_episode(
                difficulty=difficulty,
                episode_num=episode,
                memory=memory,
                verbose=verbose,
            )
            reward = result["total_reward"]
            all_rewards.append(reward)
            all_difficulties.append(difficulty)
            results_by_difficulty[difficulty].append(reward)

            print(f"\n[TRAINING] Episode {episode}/{total_episodes} done | "
                  f"Reward: {reward:+.3f} | "
                  f"Lessons: {len(memory.lessons)}")

        except KeyboardInterrupt:
            print("\n[TRAINING] Interrupted by user")
            break
        except Exception as e:
            print(f"\n[TRAINING] Episode {episode} failed: {e}")
            import traceback
            traceback.print_exc()
            all_rewards.append(0.0)
            all_difficulties.append(difficulty)

    # ── FINAL REPORT ───────────────────────────────────────────────
    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"🏁 TRAINING COMPLETE — {len(all_rewards)} episodes in {elapsed:.1f}s")
    print("="*60)

    for diff in DIFFICULTIES:
        rewards = results_by_difficulty[diff]
        if rewards:
            avg = sum(rewards) / len(rewards)
            best = max(rewards)
            print(f"  {diff.upper():8s}: avg={avg:.3f} | best={best:.3f} | episodes={len(rewards)}")

    if all_rewards:
        overall_avg = sum(all_rewards) / len(all_rewards)
        print(f"\n  OVERALL AVG: {overall_avg:.3f}")

    # Print reward curve
    print_reward_curve(all_rewards, all_difficulties)

    # Print lessons
    print_lessons_summary(memory)

    print(f"[MEMORY] Final state: {memory.summary()}")

    return {
        "total_episodes": len(all_rewards),
        "all_rewards": all_rewards,
        "results_by_difficulty": results_by_difficulty,
        "memory_summary": memory.summary(),
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Agent Support RL Training Loop")
    parser.add_argument("--episodes", type=int, default=15, help="Number of training episodes")
    parser.add_argument("--difficulty", type=str, default=None, choices=["easy", "medium", "hard"], help="Fix difficulty (default: rotate)")
    parser.add_argument("--reset-memory", action="store_true", help="Clear stored lessons before training")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-step output")
    args = parser.parse_args()

    run_training(
        total_episodes=args.episodes,
        difficulty_filter=args.difficulty,
        reset_memory=args.reset_memory,
        verbose=not args.quiet,
    )
