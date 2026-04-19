"""
memory.py - Stores lessons learned between episodes.
This is the core of the self-improvement mechanism.
The Critic Agent writes lessons here after each episode.
These lessons are injected into the next episode's system prompt.
"""

import json
import os
from datetime import datetime
from typing import List, Dict, Any

MEMORY_FILE = "agent_memory.json"

class AgentMemory:
    def __init__(self, max_lessons: int = 10):
        self.max_lessons = max_lessons
        self.lessons: List[Dict[str, Any]] = []
        self.episode_history: List[Dict[str, Any]] = []
        self._load()

    def _load(self):
        """Load memory from disk if it exists."""
        if os.path.exists(MEMORY_FILE):
            try:
                with open(MEMORY_FILE, "r") as f:
                    data = json.load(f)
                    self.lessons = data.get("lessons", [])
                    self.episode_history = data.get("episode_history", [])
            except Exception:
                self.lessons = []
                self.episode_history = []

    def _save(self):
        """Persist memory to disk."""
        with open(MEMORY_FILE, "w") as f:
            json.dump({
                "lessons": self.lessons,
                "episode_history": self.episode_history,
            }, f, indent=2)

    def add_lesson(self, lesson: str, episode: int, reward: float, difficulty: str):
        """Add a new lesson from the Critic Agent."""
        entry = {
            "lesson": lesson,
            "episode": episode,
            "reward": reward,
            "difficulty": difficulty,
            "timestamp": datetime.now().isoformat(),
        }
        self.lessons.append(entry)
        # Keep only the most recent lessons
        if len(self.lessons) > self.max_lessons:
            self.lessons = self.lessons[-self.max_lessons:]
        self._save()

    def add_episode(self, episode: int, reward: float, difficulty: str, task_id: str):
        """Record episode result for tracking improvement."""
        self.episode_history.append({
            "episode": episode,
            "reward": reward,
            "difficulty": difficulty,
            "task_id": task_id,
            "timestamp": datetime.now().isoformat(),
        })
        self._save()

    def get_lessons_prompt(self) -> str:
        """
        Returns lessons formatted as a system prompt injection.
        This is what gets prepended to agent prompts to make them self-improve.
        """
        if not self.lessons:
            return ""
        lines = ["=== LESSONS FROM PREVIOUS EPISODES ==="]
        for i, entry in enumerate(self.lessons[-5:], 1):  # last 5 lessons
            lines.append(f"{i}. [Episode {entry['episode']} | Score {entry['reward']:.2f} | {entry['difficulty']}]: {entry['lesson']}")
        lines.append("Apply these lessons to improve your performance.\n")
        return "\n".join(lines)

    def get_reward_trend(self) -> List[float]:
        """Returns list of rewards over time for plotting improvement."""
        return [e["reward"] for e in self.episode_history]

    def clear(self):
        """Reset memory (for fresh training runs)."""
        self.lessons = []
        self.episode_history = []
        if os.path.exists(MEMORY_FILE):
            os.remove(MEMORY_FILE)

    def summary(self) -> str:
        rewards = self.get_reward_trend()
        if not rewards:
            return "No episodes yet."
        avg = sum(rewards) / len(rewards)
        best = max(rewards)
        recent = rewards[-5:] if len(rewards) >= 5 else rewards
        recent_avg = sum(recent) / len(recent)
        return (
            f"Episodes: {len(rewards)} | "
            f"Avg: {avg:.3f} | "
            f"Best: {best:.3f} | "
            f"Recent avg (last 5): {recent_avg:.3f} | "
            f"Lessons stored: {len(self.lessons)}"
        )
