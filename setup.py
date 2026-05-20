import os
import subprocess
from typing import Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TextStreamer
from sentence_transformers import SentenceTransformer
from supabase import create_client, Client


print("Installing bitsandbytes...")
subprocess.check_call(["pip", "install", "-q", "-U", "bitsandbytes"])


MODEL_ID = "unsloth/Qwen3-4B-bnb-4bit"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
MAX_SEQ_LENGTH = 4096

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")


print("Loading Qwen3 4B with 4bit quantization...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16,
)
model.eval()

print("Loading embedding model...")
embed_model = SentenceTransformer(EMBEDDING_MODEL)

print("Connecting to Supabase...")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


def _embed(text: str) -> list[float]:
    return embed_model.encode(text).tolist()


def retrieve_memories(
    query: str,
    top_k: int = 5,
    material_system: Optional[str] = None,
    simulation_software: Optional[str] = None,
    memory_type: Optional[str] = None,
    min_novelty: float = 0.0,
) -> list[dict]:
    embedding = _embed(query)

    result = supabase.rpc(
        "match_memories",
        {
            "query_embedding": embedding,
            "match_count": top_k,
            "min_novelty": min_novelty,
            "filter_material_system": material_system,
            "filter_simulation_software": simulation_software,
            "filter_memory_type": memory_type,
        },
    ).execute()
    return result.data


def store_memory(
    text: str,
    material_system: Optional[str] = None,
    simulation_software: Optional[str] = None,
    memory_type: str = "general",
) -> str:
    import numpy as np
    from datetime import datetime

    embedding = _embed(text)

    result = supabase.table("memories").select("embedding").execute()
    if result.data:
        stored = [r["embedding"] for r in result.data]
        query_vec = np.array(embedding)
        stored_matrix = np.array(stored)
        norms = np.linalg.norm(stored_matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1e-8
        cosine_sim = (stored_matrix @ query_vec) / (norms.squeeze() * np.linalg.norm(query_vec))
        novelty = round(1.0 - cosine_sim.max(), 4)
    else:
        novelty = 1.0

    row = {
        "content": text,
        "embedding": embedding,
        "novelty_score": novelty,
        "material_system": material_system,
        "simulation_software": simulation_software,
        "memory_type": memory_type,
        "created_at": datetime.utcnow().isoformat(),
    }

    result = supabase.table("memories").insert(row).execute()
    return result.data[0]["id"]


def generate(
    prompt: str,
    max_new_tokens: int = 2048,
    temperature: float = 0.7,
    top_p: float = 0.9,
    repetition_penalty: float = 1.1,
    stream: bool = False,
) -> str:
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer(text, return_tensors="pt").to("cuda")

    if stream:
        streamer = TextStreamer(tokenizer, skip_prompt=True)
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            streamer=streamer,
        )
    else:
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    assistant_marker = "assistant"
    if assistant_marker in response:
        response = response.split(assistant_marker)[-1].strip()
    return response


def ask(
    question: str,
    material_system: Optional[str] = None,
    simulation_software: Optional[str] = None,
    max_new_tokens: int = 2048,
    temperature: float = 0.7,
) -> str:
    SYSTEM_PROMPT = (
        "You are ChemMind, an expert AI assistant for computational chemistry PhD students. "
        "Answer questions accurately using the provided context from previous research memories."
    )

    memories = retrieve_memories(
        query=question,
        top_k=5,
        material_system=material_system,
        simulation_software=simulation_software,
    )

    context_parts = []
    for i, mem in enumerate(memories, 1):
        content = mem.get("content", "")
        material = mem.get("material_system", "N/A")
        software = mem.get("simulation_software", "N/A")
        similarity = mem.get("similarity", 0)
        context_parts.append(
            f"[Memory {i}] (similarity: {similarity:.3f}, material: {material}, software: {software})\n{content}"
        )

    context = "\n\n".join(context_parts) if context_parts else "No relevant memories found."

    prompt = (
        "/no_think\n\n"
        f"{SYSTEM_PROMPT}\n\n"
        "## Relevant Research Memories\n\n"
        f"{context}\n\n"
        "## Question\n\n"
        f"{question}\n\n"
        "Provide a detailed, technically accurate answer for a computational chemistry researcher."
    )

    response = generate(
        prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )

    store_memory(
        text=f"Q: {question}\nA: {response}",
        material_system=material_system,
        simulation_software=simulation_software,
        memory_type="qa_pair",
    )

    return response


print("ChemMind ready!")
