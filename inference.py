from typing import Optional

from hippocampus import retrieve_memories, store_memory
from model import generate


SYSTEM_PROMPT = """You are ChemMind, an expert AI assistant for computational chemistry PhD students.
Answer questions accurately using the provided context from previous research memories.
If the context does not contain relevant information, use your general chemistry knowledge but note that the answer is not from stored memories."""


def _build_prompt(question: str, memories: list[dict]) -> str:
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

    prompt = f"/no_think\n\n{SYSTEM_PROMPT}

## Relevant Research Memories

{context}

## Question

{question}

Provide a detailed, technically accurate answer for a computational chemistry researcher."

    return prompt


def ask(
    question: str,
    material_system: Optional[str] = None,
    simulation_software: Optional[str] = None,
    max_new_tokens: int = 2048,
    temperature: float = 0.7,
) -> str:
    memories = retrieve_memories(
        query=question,
        top_k=5,
        material_system=material_system,
        simulation_software=simulation_software,
    )

    prompt = _build_prompt(question, memories)
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
