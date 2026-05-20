import os
import threading
from typing import Optional

import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from supabase import create_client, Client
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

MODEL_ID = "unsloth/Qwen3-4B-bnb-4bit"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
NGROK_AUTH_TOKEN = os.getenv("NGROK_AUTH_TOKEN")


class AppState:
    hf_model = None
    hf_tokenizer = None
    _embedder = None
    _supabase_client: Client = None


state = AppState()


def _load_model():
    print("Loading Qwen3 4B with 4bit quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
    )
    state.hf_tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    state.hf_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    state.hf_model.eval()
    print("Model loaded.")


def _load_embedder():
    print("Loading embedding model...")
    state._embedder = SentenceTransformer(EMBEDDING_MODEL)
    print("Embedder loaded.")


def _connect_supabase():
    print("Connecting to Supabase...")
    state._supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
    print("Supabase connected.")


def _setup_ngrok():
    try:
        from pyngrok import ngrok
        if NGROK_AUTH_TOKEN:
            ngrok.set_auth_token(NGROK_AUTH_TOKEN)
        public_url = ngrok.connect(8000)
        print(f"ngrok tunnel open: {public_url}")
    except Exception as e:
        print(f"ngrok setup skipped: {e}")


class ChatRequest(BaseModel):
    question: str
    material_system: Optional[str] = None
    simulation_software: Optional[str] = None


class ChatResponse(BaseModel):
    response: str
    agent_used: str
    intent: str


app = FastAPI(title="ChemMind Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    from orchestrator import run as orchestrator_run
    result = orchestrator_run(
        question=request.question,
        material_system=request.material_system,
        simulation_software=request.simulation_software,
    )
    return ChatResponse(**result)


@app.get("/health")
async def health():
    return {"status": "ok"}


def startup():
    _load_model()
    _load_embedder()
    _connect_supabase()
    _setup_ngrok()


if __name__ == "__main__":
    import uvicorn
    thread = threading.Thread(target=startup, daemon=True)
    thread.start()
    uvicorn.run(app, host="0.0.0.0", port=8000)
