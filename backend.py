from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from orchestrator import run as orchestrator_run

app = FastAPI(title="ChemMind Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    question: str
    material_system: Optional[str] = None
    simulation_software: Optional[str] = None


class ChatResponse(BaseModel):
    response: str
    agent_used: str
    intent: str


@app.get("/")
async def root():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    result = orchestrator_run(
        question=request.question,
        material_system=request.material_system,
        simulation_software=request.simulation_software,
    )
    return ChatResponse(**result)


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
