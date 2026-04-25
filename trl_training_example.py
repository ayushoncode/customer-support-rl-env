"""
Minimal HF TRL training example for SupportOps AI.

This script is designed to be Colab-friendly:
1. Point it at a running environment server.
2. Collect full episodes via /run_episode.
3. Keep high-reward trajectories.
4. Fine-tune a small causal LM with TRL SFTTrainer.

It is intentionally simple and meant for hackathon demonstration use.
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path


SYSTEM_PROMPT = """You are a professional, empathetic customer support agent.
Write a policy-aware reply that resolves the customer's issue clearly.
"""


def post_json(url: str, payload: dict) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))


def collect_dataset(base_url: str, episodes: int, min_reward: float) -> list[dict]:
    samples = []
    difficulties = ["easy", "medium", "hard"]

    for idx in range(episodes):
        difficulty = difficulties[idx % len(difficulties)]
        result = post_json(f"{base_url.rstrip('/')}/run_episode", {"difficulty": difficulty})
        reward = float(result.get("submit_reward", 0.0))
        if reward < min_reward:
            continue

        email = result.get("email", "").strip()
        reply = result.get("final_reply", "").strip()
        lesson = result.get("lesson", "").strip()
        task_id = result.get("task_id", "")

        prompt = (
            f"{SYSTEM_PROMPT}\n"
            f"Difficulty: {difficulty}\n"
            f"Task: {task_id}\n"
            f"Customer Email:\n{email}\n\n"
            f"Reply:"
        )
        text = f"{prompt} {reply}"
        samples.append(
            {
                "text": text,
                "prompt": prompt,
                "completion": reply,
                "reward": reward,
                "difficulty": difficulty,
                "task_id": task_id,
                "lesson": lesson,
            }
        )

    return samples


def save_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal TRL training example for SupportOps AI")
    parser.add_argument("--base-url", default="http://localhost:7860", help="Environment server base URL")
    parser.add_argument("--episodes", type=int, default=24, help="Episodes to collect before training")
    parser.add_argument("--min-reward", type=float, default=0.65, help="Keep only trajectories at or above this reward")
    parser.add_argument("--output", default="artifacts/train_data.jsonl", help="Path for saved JSONL dataset")
    parser.add_argument("--model", default="distilgpt2", help="HF model name for a quick demo fine-tune")
    parser.add_argument("--train", action="store_true", help="Actually run SFT training with TRL")
    args = parser.parse_args()

    rows = collect_dataset(args.base_url, args.episodes, args.min_reward)
    if not rows:
        raise SystemExit("No training rows collected. Try increasing episodes or lowering --min-reward.")

    output_path = Path(args.output)
    save_jsonl(rows, output_path)
    print(f"[COLLECT] Saved {len(rows)} rows to {output_path}")

    if not args.train:
        print("[TRAIN] Skipped. Re-run with --train after installing trl, transformers, datasets, accelerate, and torch.")
        return

    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
    from trl import SFTTrainer

    dataset = load_dataset("json", data_files=str(output_path), split="train")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model)
    training_args = TrainingArguments(
        output_dir="artifacts/trl-checkpoints",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        learning_rate=2e-5,
        num_train_epochs=1,
        logging_steps=5,
        save_strategy="no",
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.model.save_pretrained("artifacts/trl-model")
    tokenizer.save_pretrained("artifacts/trl-model")
    print("[TRAIN] Saved model to artifacts/trl-model")


if __name__ == "__main__":
    main()
