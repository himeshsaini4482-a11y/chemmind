from typing import Optional

from hippocampus import store_memory
from inference import ask


class SimulationAgent:
    EXPERTISE = (
        "You are an expert in computational chemistry simulation software: "
        "VASP, LAMMPS, Quantum ESPRESSO, and GROMACS. "
        "You provide specific parameter recommendations, input file configurations, "
        "convergence criteria, and best practices for accurate simulations."
    )

    PARAM_INSTRUCTIONS = (
        "Always give specific parameter recommendations including: "
        "energy cutoffs, k-point grids, convergence thresholds, "
        "timestep values, force field selections, and ensemble settings. "
        "Include warnings about common pitfalls and how to verify results."
    )

    def solve(
        self,
        question: str,
        material_system: Optional[str] = None,
        simulation_software: Optional[str] = None,
    ) -> str:
        prompt = (
            f"{self.EXPERTISE}\n\n"
            f"Question: {question}\n\n"
            f"{self.PARAM_INSTRUCTIONS}"
        )

        response = ask(
            question=prompt,
            material_system=material_system,
            simulation_software=simulation_software,
        )

        store_memory(
            text=f"Q: {question}\nA: {response}",
            material_system=material_system,
            simulation_software=simulation_software,
            memory_type="simulation",
        )

        return response
