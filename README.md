# Customer Support RL Environment

An OpenEnv-compliant RL environment where an AI agent learns to handle customer support emails.

## Setup
```bash
pip install -r requirements.txt
```

## Run Server
```bash
API_BASE_URL=... MODEL_NAME=... HF_TOKEN=... uvicorn main:app --host 0.0.0.0 --port 7860
```

## Run Inference
```bash
API_BASE_URL=... MODEL_NAME=... HF_TOKEN=... python inference.py
```

## Endpoints
- GET  /health
- POST /reset
- POST /step
- GET  /state
- GET  /tasks

## Reward (0.0 - 1.0)
- Empathy: 0.3
- Correct solution: 0.4
- Completeness: 0.2
- Personalisation: 0.1
