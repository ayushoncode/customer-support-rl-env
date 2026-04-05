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
---

# 🎧 Customer Support RL Environment

A real-world OpenEnv environment that simulates a customer support operations center: email triage, empathy-aware response drafting, resolution routing, and escalation handling.

Designed for training and evaluating agentic AI systems on practical support workflows — not games, not toys.

## Why This Is Real-World

Customer support centers handle thousands of emails daily:

- Triage incoming complaints by urgency and type (refund, fraud, missing item, billing error)
- Draft empathetic, actionable replies that resolve the issue and retain the customer
- Escalate high-stakes complaints (legal threats, fraud, missing high-value items) to senior teams
- Personalise responses with references to the customer order, account, and situation
- Avoid template-sounding, dismissive, or incomplete replies that escalate churn risk

Real-world parallel: Zendesk / Freshdesk support queues, e-commerce ops centers, telecom complaint desks.

## Environment API

Full OpenEnv spec compliance with typed models and standard API:

| Endpoint | Description |
|----------|-------------|
| POST /reset | reset(difficulty=...) observation — start a new episode |
| POST /step | step(action) observation, reward, done, info — submit a reply |
| GET /state | state() current typed state — inspect server state |
| GET /health | Health check endpoint — must return 200 |
| GET /tasks | List all 6 tasks with metadata |
| GET /docs | Interactive Swagger UI |

## Key Files

| File | Purpose |
|------|---------|
| app/models.py | SupportObservation, SupportAction, SupportState — typed Pydantic models |
| app/env.py | Environment logic, deterministic graders, reward computation |
| main.py | FastAPI app exposing all OpenEnv endpoints |
| inference.py | Baseline inference script with structured stdout |
| openenv.yaml | OpenEnv manifest |
| Dockerfile | HF Spaces-compatible Docker build |
| requirements.txt | Python dependencies |

## Action Space

| Field | Type | Description |
|-------|------|-------------|
| reply | string | Full support agent reply to the customer email |

## Observation Space

| Field | Description |
|-------|-------------|
| email | Full customer complaint email text |
| history | Prior conversation turns (multi-turn ready) |
| difficulty | Current difficulty level: easy / medium / hard |
| task_id | Unique identifier for the active task |

## Tasks and Graders

Three deterministic difficulty levels, 2 tasks each (6 total):

### Easy (2 tasks)

Clear, polite complaints with a single obvious resolution path. Max steps: 1

| Task ID | Scenario |
|---------|----------|
| easy_refund | Customer requests refund for 2-week delayed order |
| easy_tracking | Customer asking for missing shipment tracking info |

### Medium (2 tasks)

Frustrated customers with compound issues. Max steps: 1

| Task ID | Scenario |
|---------|----------|
| medium_wrong_item | Wrong item received and it arrived damaged |
| medium_billing | Customer double-charged for the same order |

### Hard (2 tasks)

Furious customers with legal threats and high-value losses. Max steps: 1

| Task ID | Scenario |
|---------|----------|
| hard_missing_laptop | $1200 laptop missing for 3 weeks, threatening legal action |
| hard_fraud | $800 unauthorized account charges, threatening bank dispute |

## Per-Reply Grader (0.0 - 1.0)

| Component | Weight | Criteria |
|-----------|--------|----------|
| Empathy | 30% | sorry, apologize, understand, frustrat, sincer, concern |
| Correct Solution | 40% | refund, escalat, investigat, replac, secur, fraud, return |
| Completeness | 20% | >= 40 words = full, >= 20 words = half, < 20 = 0 |
| Personalisation | 10% | References order/account + uses you/your/customer |

## Reward Function

reward = empathy(0.3) + solution(0.4) + completeness(0.2) + personalisation(0.1)

| Reply Quality | Expected Reward |
|--------------|-----------------|
| Empty / gibberish | 0.00 |
| Short apology only | ~0.15 |
| Apology + correct resolution | ~0.60 |
| Full professional reply 40+ words | ~0.85 - 1.00 |

## Setup

### 1. Install

    python3.11 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt

### 2. Run Locally

    uvicorn main:app --host 0.0.0.0 --port 7860

### 3. Run Baseline Inference

    export API_BASE_URL=https://router.huggingface.co/v1
    export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
    export HF_TOKEN=hf_your_token_here
    python inference.py

## Docker

    docker build -t support-env:latest .
    docker run -p 7860:7860 -e API_BASE_URL=https://router.huggingface.co/v1 -e MODEL_NAME=Qwen/Qwen2.5-72B-Instruct -e HF_TOKEN=hf_your_token_here support-env:latest

## Expected Baseline Scores

| Task | Expected Score Range | Steps |
|------|---------------------|-------|
| Easy | 0.75 - 0.95 | 1 |
| Medium | 0.65 - 0.85 | 1 |
| Hard | 0.55 - 0.80 | 1 |

## Non-Functional Requirements

- Inference runtime target: under 20 minutes on CPU-only machine
- Designed for low resource footprint (2 vCPU / 8 GB sufficient)
- Deterministic graders — no random score variance
- Scores reproducible with fixed model endpoint

## Project Structure

    .
    ├── app/
    │   ├── __init__.py
    │   ├── env.py
    │   └── models.py
    ├── main.py
    ├── inference.py
    ├── openenv.yaml
    ├── requirements.txt
    ├── Dockerfile
    └── README.md

## Required Environment Variables

| Variable | Description |
|----------|-------------|
| API_BASE_URL | OpenAI-compatible LLM API base URL |
| MODEL_NAME | Model identifier string |
| HF_TOKEN | Hugging Face / API authentication token |

---

Built for the OpenEnv x Scaler Hackathon — Round 1
