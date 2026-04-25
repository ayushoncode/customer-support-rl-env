---
title: Customer Support RL Environment
emoji: 🎧
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 7860
tags:
  - openenv
  - customer-support
  - reinforcement-learning
  - multi-agent
  - fastapi
---

# SupportOps AI

A multi-agent customer support RL environment with a live dashboard, critic memory, and Docker-ready deployment.

This project simulates a realistic support operations workflow:

- triage incoming tickets
- research order and policy context
- draft a customer reply
- run QA before submission
- escalate fraud and legal-risk cases
- score the final response with a deterministic backend
- write a critic lesson that is injected into future episodes

The result is not just a toy text demo. It is a small but credible agent training loop with memory, reward shaping, and an interface that shows the system thinking step by step.

## What It Does

Each episode runs a six-agent pipeline:

1. Triage Agent classifies urgency, ticket type, and escalation need.
2. Research Agent pulls order and policy context.
3. Resolver Agent drafts the customer-facing reply.
4. QA Agent checks empathy, policy, hallucinations, and completeness.
5. Escalation Agent enhances replies for fraud, legal, and high-risk cases.
6. Critic Agent writes a lesson for future episodes.

Those lessons are stored in memory and injected back into later runs, so the system can adapt over time.

## Main Features

- Real backend reward function, not fixed demo scores
- Live dashboard at `/dashboard`
- Full episode endpoint at `/run_episode`
- Critic memory loop via `memory.py`
- Difficulty-aware tasks: `easy`, `medium`, `hard`
- Dynamic frustration meter
- Minimal HF TRL training example in `trl_training_example.py`
- Dockerfile ready for local Docker and Hugging Face Spaces

## API

| Endpoint | Description |
|---|---|
| `GET /` | Basic service metadata |
| `GET /health` | Health check |
| `GET /docs` | Swagger UI |
| `GET /dashboard` | Live frontend dashboard |
| `GET /tasks` | List available hardcoded tasks |
| `GET /state` | Current environment state |
| `POST /reset` | Reset the base environment only |
| `POST /step` | Step through the base environment manually |
| `POST /run_episode` | Run the full multi-agent orchestrator for one episode |

## Dashboard

The dashboard in [dashboard.html](/Users/ayush/support_env/dashboard.html:1) now renders real backend episode results, including:

- the selected customer email
- the actual final reply from the backend pipeline
- step rewards
- submit score as the headline score
- frustration meter movement
- real critic lessons from the Python critic agent
- reward curve and trend indicators

Open it locally at:

```bash
http://localhost:7860/dashboard
```

## Reward Model

The backend environment scores replies using:

- empathy coverage
- solution coverage
- completeness by reply length
- personalization
- hallucination penalties
- policy violation penalties
- QA retry penalties
- difficulty multipliers

Dense step rewards:

- `RESEARCH`: `+0.10`
- `TAG`: `+0.10`
- `DRAFT`: `+0.15` if valid, `-0.05` if done too early
- `SUBMIT`: composite final score, clamped to `[-1.0, 1.0]`

Frustration also changes dynamically based on reply quality, and can force churn at `100`.

## Critic Memory Loop

The critic flow is:

1. An episode finishes in [orchestrator.py](/Users/ayush/support_env/orchestrator.py:27).
2. The backend passes the final transcript and submit reward to [agents/critic_agent.py](/Users/ayush/support_env/agents/critic_agent.py:104).
3. The critic returns a concrete lesson.
4. The lesson is stored by [memory.py](/Users/ayush/support_env/memory.py:42).
5. Recent lessons are injected into future agent prompts.

This is the core self-improvement mechanic in the project.

## Hackathon Theme Fit

This project aligns most strongly with:

- `Theme #1 - Multi-Agent Interactions`
- `Theme #3.1 - World Modeling / Professional Tasks`
- `Theme #4 - Self-Improvement`

Why:

- multiple agents coordinate under partial observability
- the environment models a realistic professional workflow
- the critic writes lessons that influence later behavior

## Judge-Ready Assets

The repo now includes:

- [JUDGE_READY_CHECKLIST.md](/Users/ayush/support_env/JUDGE_READY_CHECKLIST.md:1) for final submission prep
- [trl_training_example.py](/Users/ayush/support_env/trl_training_example.py:1) as a minimal HF TRL-compatible training path

Before final submission, add these links to this README:

- Hugging Face Space URL
- demo video or HF blog link
- reward plot image(s)
- baseline vs trained comparison

## Project Structure

```text
.
├── agents/
│   ├── critic_agent.py
│   ├── escalation_agent.py
│   ├── qa_agent.py
│   ├── research_agent.py
│   ├── resolver_agent.py
│   └── triage_agent.py
├── app/
│   ├── database.py
│   ├── env.py
│   ├── models.py
│   └── policy.py
├── dashboard.html
├── main.py
├── memory.py
├── orchestrator.py
├── training_loop.py
├── inference.py
├── smoke_test.py
├── requirements.txt
├── Dockerfile
└── README.md
```

## Local Setup

### 1. Install

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Run The App

```bash
uvicorn main:app --host 0.0.0.0 --port 7860
```

### 3. Open The Dashboard

```bash
http://localhost:7860/dashboard
```

### 4. Optional: Run Training Loop

```bash
python training_loop.py --episodes 15
```

### 5. Optional: Collect Data Or Run Minimal TRL Example

```bash
python trl_training_example.py --episodes 24 --output artifacts/train_data.jsonl
```

To actually fine-tune in Colab or a GPU environment:

```bash
pip install trl transformers datasets accelerate torch
python trl_training_example.py --episodes 24 --train
```

### 6. Optional: Run Smoke Test

```bash
python smoke_test.py
```

## Environment Variables

| Variable | Description |
|---|---|
| `HF_TOKEN` | Hugging Face token for router-backed model calls |
| `API_BASE_URL` | OpenAI-compatible base URL |
| `MODEL_NAME` | Model name to use for agent calls |

Example:

```bash
export HF_TOKEN=hf_your_token_here
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
```

If `HF_TOKEN` is not set, the system falls back to rule-based or template behavior where available.

## Docker

The current `Dockerfile` already copies the whole repository with `COPY . .`, so it includes:

- `agents/`
- `memory.py`
- `orchestrator.py`
- `training_loop.py`
- `dashboard.html`

Build and run:

```bash
docker build -t supportops-ai .
docker run -p 7860:7860 \
  -e HF_TOKEN=your_token \
  -e API_BASE_URL=https://router.huggingface.co/v1 \
  -e MODEL_NAME=Qwen/Qwen2.5-72B-Instruct \
  supportops-ai
```

## Hugging Face Spaces

This repo is configured for Docker-based Spaces via the front matter at the top of this README and the included `Dockerfile`.

Once the Space secrets are set, the same container flow should work there as well.

## Notes

- The task emails are currently hardcoded in [app/env.py](/Users/ayush/support_env/app/env.py:6).
- The dashboard now uses the real backend critic lesson instead of a frontend-only lesson map.
- The main score shown in the dashboard is the submit reward, not the accumulated step total.

## Git Push Workflow

If you want to publish the current state:

```bash
git add README.md agents app dashboard.html main.py memory.py orchestrator.py training_loop.py
git commit -m "Update multi-agent dashboard and critic memory flow"
git push origin main
```
