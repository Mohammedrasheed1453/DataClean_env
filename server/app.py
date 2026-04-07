"""app.py — FastAPI server for the Data Preparation Pipeline Agent."""

from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from models import Action, ResetResponse, StepResponse, EnvironmentState
from server.environment import DataPrepEnv

app = FastAPI(
    title="Data Preparation Pipeline Agent — OpenEnv",
    description="Full 4-phase data prep environment: EDA → Cleaning → Feature Engineering → Validation",
    version="2.0.0",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

env = DataPrepEnv()

class ResetRequest(BaseModel):
    task_id: Optional[int] = 1

@app.get("/health")
def health():
    return {"status": "ok", "version": "2.0.0"}

@app.post("/reset", response_model=ResetResponse)
def reset(request: ResetRequest = ResetRequest()):
    try:
        task_id = request.task_id if request.task_id in (1, 2, 3) else 1
        return env.reset(task_id=task_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

@app.post("/step", response_model=StepResponse)
def step(action: Action):
    try:
        return env.step(action)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

@app.get("/state", response_model=EnvironmentState)
def state():
    try:
        return env.state()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()