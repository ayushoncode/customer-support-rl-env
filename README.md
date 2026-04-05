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

A real-world OpenEnv environment that simulates a customer support operations center: multi-step ticket triage, policy-aware resolution, escalation handling, and fraud detection.

Designed for training and evaluating agentic AI systems on practical support workflows — not games, not toys.

## Why This Is Real-World

Customer support centers handle thousands of emails daily:

- Triage incoming complaints by urgency and type (refund, fraud, missing item, billing error)
- Enforce company policy — refunds only within 30 days, max 500 dollars, escalate high-value orders
- Draft empathetic, actionable replies that resolve the issue and retain the customer
- Escalate high-stakes complaints (legal threats, fraud, missing high-value items) to senior teams
- Track a Frustration Meter — if the agent handles the ticket poorly, the customer churns

Real-world parallel: Zendesk / Freshdesk support queues, e-commerce ops centers, telecom complaint desks.

## Environment API

Full OpenEnv spec compliance with typed models and standard API:

| Endpoint | Description |
|----------|-------------|
| POST /reset | reset(difficulty=...) — start a new episode |
| POST /step | step(action) — submit one action, get reward |
| GET /state | state() — inspect current environment state |
| GET /health | Health check — must return 200 |
| GET /tasks | List all 6 tasks with metadata |
| GET /docs | Interactive Swagger UI |

## Key Files

| File | Purpose |
|------|---------|
| app/models.py | SupportObservation, SupportAction, SupportState — typed Pydantic models |
| app/env.py | Environment logic, multi-step graders, frustration meter |
| app/database.py | Mock order database with 5 realistic orders |
| app/policy.py | Refund policy engine, hallucination detector, escalation rules |
| main.py | FastAPI app exposing all OpenEnv endpoints |
| inference.py | Baseline inference script with structured stdout |
| smoke_test.py | Offline validation — no LLM required |
| openenv.yaml | OpenEnv manifest |
| Dockerfile | HF Spaces-compatible Docker build |

## Advanced Action Space

The agent must complete a 4-step professional workflow per ticket.
Skipping steps triggers penalties. This proves agentic behavior, not just text generation.

| Field | Type | Description |
|-------|------|-------------|
| action_type | enum | RESEARCH, TAG, DRAFT, SUBMIT |
| reasoning | string | Chain-of-thought before acting (optional) |
| reply | string | Final reply sent to customer (SUBMIT only) |
| order_id | string | Order to look up (RESEARCH step) |
| tag | string | Ticket category: refund, fraud, billing, escalation |

Required workflow per ticket:

RESEARCH -> TAG -> DRAFT -> SUBMIT

Submitting without completing all steps triggers a penalty and increases the frustration meter.

## Observation Space

| Field | Description |
|-------|-------------|
| email | Full customer complaint email text |
| history | Prior conversation turns (multi-turn ready) |
| difficulty | easy / medium / hard |
| task_id | Unique identifier for the active task |
| order_info | Mock order data: item, amount, status, days since order |
| policy_snippet | Refund and escalation policy the agent must follow |
| frustration_meter | 0-100 — hits 100 = customer churned, episode ends |
| valid_actions | List of currently allowed action types |

## Internal State (SupportState)

| Field | Description |
|-------|-------------|
| status | active or done |
| feedback | Per-step reward breakdown for debugging |
| current_task | Active task ID |
| difficulty | Current difficulty |
| frustration_meter | Running frustration score 0-100 |
| steps_taken | Total steps used in this episode |

## Tasks and Graders

Three deterministic difficulty levels, 2 tasks each (6 total):

### Easy (2 tasks)

Clear, polite complaints. Single resolution path. Max steps: 10. Budget: unlimited.

| Task ID | Scenario | Order |
|---------|----------|-------|
| easy_refund | Refund request for delayed order | ORD-4521 |
| easy_tracking | Missing tracking information | None |

### Medium (2 tasks)

Frustrated customers, compound issues, policy constraints.

| Task ID | Scenario | Order |
|---------|----------|-------|
| medium_wrong_item | Wrong and damaged item received | ORD-3310 |
| medium_billing | Double charge on same order | ORD-6612 |

### Hard (2 tasks)

Furious customers, legal threats, fraud detection, high-value losses.

| Task ID | Scenario | Order |
|---------|----------|-------|
| hard_missing_laptop | $1200 laptop missing, threatening legal action | ORD-9921 |
| hard_fraud | $800 unauthorized charges, threatening bank dispute | None |

## Reward Function (Mathematical)

Step rewards provide dense, shaped learning signal across the full episode:
R = (0.4 x Solution) + (0.3 x Empathy) + (0.2 x Completeness) + (0.1 x Personalisation) - (H x Hallucination) - (P x PolicyViolation)

Where:
- Solution (0.4): Did the agent address the specific task resolution (refund, escalation, fraud block)?
- Empathy (0.3): Keyword coverage — sorry, apologize, understand, concern, sincer, frustrat
- Completeness (0.2): >= 40 words = full, >= 20 words = half, < 20 = 0
- Personalisation (0.1): References order/account + uses you/your/customer
- Hallucination penalty (H): -0.15 per detected hallucination pattern
- Policy violation (P): -0.3 for offering refund outside policy window

Per-step rewards by action type:

| Action | Reward | Condition |
|--------|--------|-----------|
| RESEARCH | +0.10 | Always |
| TAG | +0.10 | Always |
| DRAFT | +0.15 | Only if RESEARCH was done first |
| DRAFT | -0.05 | If RESEARCH was skipped |
| SUBMIT | Full composite score | Only if DRAFT was done |
| SUBMIT | -0.10 | If DRAFT was skipped |

Frustration Meter penalties:

| Event | Frustration Increase |
|-------|---------------------|
| DRAFT without RESEARCH | +10 |
| SUBMIT without DRAFT | +15 |
| Poor solution score (<0.2) | +20 |
| Frustration hits 100 | Episode ends, reward = -1.0 |

## Anti-Gaming Protections

| Protection | Description |
|-----------|-------------|
| Workflow enforcement | Cannot SUBMIT without RESEARCH + DRAFT |
| Hallucination detection | Detects impossible promises and dismiss language |
| Policy enforcement | Penalises refund offers that violate the 30-day policy |
| Frustration meter | Poor handling degrades score and can end the episode |
| Churn condition | Frustration >= 100 = customer churned = reward -1.0 |

## Smoke Test (No LLM Required)
```bash
python smoke_test.py
```

Expected output:
=== SMOKE TEST ===
[EASY]   Perfect agent: 1.0 | Bad agent: -0.1
[MEDIUM] Perfect agent: 1.0 | Bad agent: -0.1
[HARD]   Perfect agent: 1.0 | Bad agent: -0.1
All smoke tests PASSED!

This proves the reward function is deterministic and correct before any LLM is involved.

## Setup

### 1. Install
```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Run Locally
```bash
uvicorn main:app --host 0.0.0.0 --port 7860
```

### 3. Run Smoke Test
```bash
python smoke_test.py
```

### 4. Run Baseline Inference
```bash
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
export HF_TOKEN=hf_your_token_here
python inference.py
```

## Docker
```bash
docker build -t support-env:latest .
docker run -p 7860:7860 -e API_BASE_URL=https://router.huggingface.co/v1 -e MODEL_NAME=Qwen/Qwen2.5-72B-Instruct -e HF_TOKEN=hf_your_token_here support-env:latest
```

## Baseline Inference Script

The mandatory script at inference.py:

- Uses OpenAI client via HF router
- Environment variables: API_BASE_URL, MODEL_NAME, HF_TOKEN
- Runs full 4-step episode (RESEARCH, TAG, DRAFT, SUBMIT) per difficulty
- LLM generates the final reply at SUBMIT step
- Deterministic fallback when model output is malformed
- Structured stdout: [START], [STEP], [END]

### Expected Baseline Scores

| Task | Expected Score | Steps |
|------|---------------|-------|
| Easy | 0.75 - 0.95 | 4 |
| Medium | 0.65 - 0.85 | 4 |
| Hard | 0.55 - 0.80 | 4 |

## Non-Functional Requirements

- Inference runtime: under 20 minutes on CPU-only machine
- Resource footprint: 2 vCPU / 8 GB RAM sufficient
- No heavy NLP libraries (no spaCy, no transformers) — pure keyword graders
- Deterministic graders — scores reproducible with fixed model endpoint
- All rewards clamped to [-1.0, 1.0]

## Required Environment Variables

| Variable | Description |
|----------|-------------|
| API_BASE_URL | OpenAI-compatible LLM API base URL |
| MODEL_NAME | Model identifier string |
| HF_TOKEN | Hugging Face API token (set as Secret in HF Space settings) |

## Project Structure
.
├── app/
│   ├── init.py
│   ├── env.py               # Multi-step environment logic + graders
│   ├── models.py            # Pydantic Action/Observation/State models
│   ├── database.py          # Mock order database
│   └── policy.py            # Refund policy + hallucination detector
├── main.py                  # FastAPI app — OpenEnv endpoints
├── inference.py             # Mandatory baseline inference script
├── smoke_test.py            # Offline deterministic validation
├── openenv.yaml             # OpenEnv manifest
├── requirements.txt         # Python dependencies
├── Dockerfile               # HF Spaces-compatible Docker build
└── README.md                # This file

---

Built for the OpenEnv x Scaler Hackathon — Round 1
