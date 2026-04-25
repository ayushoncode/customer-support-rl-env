from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import json
import os
import re
import subprocess
import sys
from pydantic import BaseModel

from app.env import SupportEnv, TASKS
from app.models import ResetRequest, StepRequest, SupportAction, RunEpisodeRequest
from memory import AgentMemory
from orchestrator import run_episode

app = FastAPI(
    title="Customer Support RL Environment",
    description="An OpenEnv-compliant environment for training agents to handle customer support emails.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

env = SupportEnv()
memory = AgentMemory(max_lessons=10)
GRPO_REWARDS_FILE = "grpo_rewards.json"


class RunGRPORequest(BaseModel):
    episodes: int = 20
    group_size: int = 4
    mock: bool = False


@app.get("/dashboard")
def dashboard():
    return FileResponse("dashboard.html")

@app.get("/")
def root():
    return {
        "message": "Customer Support RL Environment is Running",
        "health": "/health",
        "docs": "/docs",
        "tasks": "/tasks",
        "dashboard": "/dashboard",
    }

@app.get("/health")
def health():
    return {"status": "ok", "env": "CustomerSupportEnv", "version": "1.0.0"}

@app.post("/reset")
def reset(request: ResetRequest = None):
    difficulty = request.difficulty if request else "easy"
    if difficulty not in TASKS:
        raise HTTPException(status_code=400, detail=f"Invalid difficulty. Choose from: {list(TASKS.keys())}")
    obs = env.reset(difficulty=difficulty)
    return obs.model_dump()

@app.post("/step")
def step(request: StepRequest):
    if not env.current_task:
        raise HTTPException(status_code=400, detail="Environment not initialized. Call /reset first.")
    state, reward, done, info = env.step(request.action)
    return {
        "state": state.model_dump(),
        "reward": reward,
        "done": done,
        "info": info,
    }

@app.post("/run_episode")
def run_full_episode(request: RunEpisodeRequest = None):
    difficulty = request.difficulty if request else "easy"
    if difficulty not in TASKS:
        raise HTTPException(status_code=400, detail=f"Invalid difficulty. Choose from: {list(TASKS.keys())}")

    episode_num = len(memory.episode_history) + 1
    result = run_episode(
        difficulty=difficulty,
        episode_num=episode_num,
        memory=memory,
        verbose=False,
    )

    transcript = result["transcript"]
    return {
        "episode": result["episode"],
        "difficulty": result["difficulty"],
        "task_id": result["task_id"],
        "email": transcript.get("email", ""),
        "order_info": transcript.get("order_info"),
        "triage_result": transcript.get("triage_result", {}),
        "qa_result": transcript.get("qa_result", {}),
        "final_reply": transcript.get("final_reply", ""),
        "lesson": result["lesson"],
        "submit_reward": result["submit_reward"],
        "total_reward": result["total_reward"],
        "step_rewards": result["step_rewards"],
        "feedback": result["feedback"],
        "frustration": result["frustration"],
        "qa_retries": transcript.get("qa_retries", 0),
        "escalation_used": transcript.get("escalation_used", False),
        "escalation_reason": transcript.get("escalation_reason", ""),
        "escalation_action": transcript.get("escalation_action", ""),
        "resolver_source": transcript.get("resolver_source", ""),
        "resolver_error": transcript.get("resolver_error", ""),
        "memory_size": len(memory.lessons),
    }

@app.get("/state")
def state():
    return env.state().model_dump()

@app.get("/tasks")
def list_tasks():
    all_tasks = []
    for difficulty, task_list in TASKS.items():
        for task in task_list:
            all_tasks.append({
                "id": task["id"],
                "difficulty": difficulty,
                "email": task["email"],
            })
    return {"tasks": all_tasks, "total": len(all_tasks)}


@app.get("/grpo_rewards")
def get_grpo_rewards():
    if not os.path.exists(GRPO_REWARDS_FILE):
        return JSONResponse(
            status_code=200,
            content={
                "project": "SupportOps AI",
                "algorithm": "GRPO",
                "episodes_completed": 0,
                "rewards": [],
                "history": [],
            },
        )
    try:
        with open(GRPO_REWARDS_FILE, "r", encoding="utf-8") as f:
            payload = json.load(f)
        return payload
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read {GRPO_REWARDS_FILE}: {e}")


@app.post("/run_grpo")
def run_grpo(request: RunGRPORequest = RunGRPORequest()):
    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "grpo_train.py")
    if not os.path.exists(script_path):
        raise HTTPException(status_code=404, detail="grpo_train.py not found")

    cmd = [
        sys.executable,
        script_path,
        "--episodes",
        str(request.episodes),
        "--group-size",
        str(request.group_size),
    ]
    if request.mock:
        cmd.append("--mock")

    def sse_stream():
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except Exception as e:
            yield f"event: error\ndata: Failed to start GRPO subprocess: {e}\n\n"
            return

        episode_re = re.compile(r"\[GRPO\]\s+Episode\s+(\d+)\s*/\s*(\d+)", re.IGNORECASE)
        total_hint = max(1, request.episodes)

        try:
            assert proc.stdout is not None
            for raw_line in proc.stdout:
                line = raw_line.rstrip("\n")
                safe_line = line.replace("\r", " ")
                yield f"event: log\ndata: {safe_line}\n\n"

                m = episode_re.search(line)
                if m:
                    done = int(m.group(1))
                    total = int(m.group(2)) if m.group(2).isdigit() else total_hint
                    progress_payload = json.dumps({
                        "completed": done,
                        "total": total,
                        "percent": round((done / max(1, total)) * 100, 2),
                    })
                    yield f"event: progress\ndata: {progress_payload}\n\n"

            return_code = proc.wait()
            if return_code == 0:
                yield "event: done\ndata: GRPO training completed\n\n"
            else:
                yield f"event: error\ndata: GRPO training exited with code {return_code}\n\n"
        except Exception as e:
            yield f"event: error\ndata: Streaming failed: {e}\n\n"
        finally:
            if proc.poll() is None:
                proc.kill()

    return StreamingResponse(sse_stream(), media_type="text/event-stream")


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=7860, reload=False)
