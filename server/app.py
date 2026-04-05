from app.env import SupportEnv, TASKS
from app.models import ResetRequest, StepRequest, SupportAction
from fastapi import FastAPI, HTTPException

app = FastAPI(title="Customer Support RL Environment")
env = SupportEnv()

@app.get("/")
def root():
    return {"message": "Customer Support RL Environment is Running", "docs": "/docs"}

@app.get("/health")
def health():
    return {"status": "ok", "env": "CustomerSupportEnv", "version": "1.0.0"}

@app.post("/reset")
def reset(request: ResetRequest = None):
    difficulty = request.difficulty if request else "easy"
    if difficulty not in TASKS:
        raise HTTPException(status_code=400, detail=f"Invalid difficulty")
    obs = env.reset(difficulty=difficulty)
    return obs.model_dump()

@app.post("/step")
def step(request: StepRequest):
    if not env.current_task:
        raise HTTPException(status_code=400, detail="Call /reset first")
    state, reward, done, info = env.step(request.action)
    return {"observation": state.model_dump(), "reward": reward, "done": done, "info": info}

@app.get("/state")
def state():
    return env.state().model_dump()

@app.get("/tasks")
def list_tasks():
    all_tasks = []
    for difficulty, task_list in TASKS.items():
        for task in task_list:
            all_tasks.append({"id": task["id"], "difficulty": difficulty, "email": task["email"]})
    return {"tasks": all_tasks, "total": len(all_tasks)}
