from typing import Optional

from hippocampus import store_memory
from inference import ask


class CodeAgent:
    EXPERTISE = (
        "You are an expert in computational chemistry Python libraries: "
        "ASE (Atomic Simulation Environment), pymatgen, VASP output parsing, "
        "LAMMPS post-processing, numpy, scipy, and matplotlib. "
    )

    CODE_INSTRUCTIONS = (
        "Return ONLY complete, runnable Python code. "
        "Add detailed comments explaining each step. "
        "Include all necessary imports. "
        "Do not include markdown code fences or explanations outside the code. "
        "Ensure the code handles file I/O errors gracefully where applicable."
    )

    def write(
        self,
        task: str,
        material_system: Optional[str] = None,
        simulation_software: Optional[str] = None,
    ) -> str:
        prompt = (
            f"{self.EXPERTISE}\n\n"
            f"Task: {task}\n\n"
            f"{self.CODE_INSTRUCTIONS}"
        )

        code = ask(
            question=prompt,
            material_system=material_system,
            simulation_software=simulation_software,
            max_new_tokens=4096,
            temperature=0.3,
        )

        store_memory(
            text=f"Task: {task}\n\nCode:\n{code}",
            material_system=material_system,
            simulation_software=simulation_software,
            memory_type="generated_code",
        )

        return code
