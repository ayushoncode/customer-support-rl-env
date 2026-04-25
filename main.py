from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

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

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=7860, reload=False)
