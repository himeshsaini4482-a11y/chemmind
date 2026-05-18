from typing import Optional, TypedDict

from langgraph.graph import StateGraph, END

from inference import ask
from literature_agent import LiteratureAgent
from code_agent import CodeAgent


class AgentState(TypedDict):
    question: str
    material_system: Optional[str]
    simulation_software: Optional[str]
    intent: str
    response: str
    agent_used: str


INTENT_KEYWORDS = {
    "simulation": ["simulation", "vasp", "lammps", "gaussian", "dft", "md", "molecular dynamics", "geometry optimization", "band structure", "dos", "phonon"],
    "literature": ["literature", "paper", "research", "survey", "review", "arxiv", "publication", "cite", "reference"],
    "code": ["code", "script", "python", "ase", "pymatgen", "plot", "parse", "automate", "write a script", "generate code"],
}


def classify_intent(state: AgentState) -> AgentState:
    question_lower = state["question"].lower()

    scores = {intent: 0 for intent in INTENT_KEYWORDS}
    for intent, keywords in INTENT_KEYWORDS.items():
        for kw in keywords:
            if kw in question_lower:
                scores[intent] += 1

    max_score = max(scores.values())
    if max_score == 0:
        state["intent"] = "general"
    else:
        state["intent"] = max(scores, key=scores.get)

    return state


def route_simulation(state: AgentState) -> AgentState:
    response = ask(
        question=state["question"],
        material_system=state["material_system"],
        simulation_software=state["simulation_software"],
    )
    state["response"] = response
    state["agent_used"] = "simulation"
    return state


def route_literature(state: AgentState) -> AgentState:
    agent = LiteratureAgent()
    papers = agent.search(
        topic=state["question"],
        material_system=state["material_system"],
        simulation_software=state["simulation_software"],
    )

    if papers:
        paper_summaries = []
        for p in papers:
            paper_summaries.append(
                f"- {p.get('title', 'N/A')} ({p.get('year', 'N/A')})\n"
                f"  Authors: {p.get('authors', 'N/A')}\n"
                f"  Abstract: {p.get('abstract', '')}"
            )
        context = "\n\n".join(paper_summaries)
        state["response"] = f"Found {len(papers)} relevant papers:\n\n{context}"
    else:
        state["response"] = "No relevant papers found for this topic."

    state["agent_used"] = "literature"
    return state


def route_code(state: AgentState) -> AgentState:
    agent = CodeAgent()
    code = agent.write(
        task=state["question"],
        material_system=state["material_system"],
        simulation_software=state["simulation_software"],
    )
    state["response"] = code
    state["agent_used"] = "code"
    return state


def route_general(state: AgentState) -> AgentState:
    response = ask(
        question=state["question"],
        material_system=state["material_system"],
        simulation_software=state["simulation_software"],
    )
    state["response"] = response
    state["agent_used"] = "general"
    return state


def route_decision(state: AgentState) -> str:
    intent = state["intent"]
    if intent == "simulation":
        return "simulation"
    elif intent == "literature":
        return "literature"
    elif intent == "code":
        return "code"
    else:
        return "general"


def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("classify", classify_intent)
    graph.add_node("simulation", route_simulation)
    graph.add_node("literature", route_literature)
    graph.add_node("code", route_code)
    graph.add_node("general", route_general)

    graph.set_entry_point("classify")

    graph.add_conditional_edges(
        "classify",
        route_decision,
        {
            "simulation": "simulation",
            "literature": "literature",
            "code": "code",
            "general": "general",
        },
    )

    graph.add_edge("simulation", END)
    graph.add_edge("literature", END)
    graph.add_edge("code", END)
    graph.add_edge("general", END)

    return graph.compile()


_app = None


def get_app():
    global _app
    if _app is None:
        _app = build_graph()
    return _app


def run(
    question: str,
    material_system: Optional[str] = None,
    simulation_software: Optional[str] = None,
) -> dict:
    app = get_app()

    initial_state = {
        "question": question,
        "material_system": material_system,
        "simulation_software": simulation_software,
        "intent": "",
        "response": "",
        "agent_used": "",
    }

    result = app.invoke(initial_state)
    return {
        "response": result["response"],
        "agent_used": result["agent_used"],
        "intent": result["intent"],
    }
