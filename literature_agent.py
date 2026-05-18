from typing import Optional
import requests

from hippocampus import store_memory


SEMANTIC_SCHOLAR_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
FIELDS = "title,abstract,year,authors,citationCount,externalIds"


class LiteratureAgent:
    def search(
        self,
        topic: str,
        limit: int = 5,
        material_system: Optional[str] = None,
        simulation_software: Optional[str] = None,
    ) -> list[dict]:
        params = {
            "query": topic,
            "fields": FIELDS,
            "limit": 50,
            "sort": "citationCount:desc",
        }

        response = requests.get(SEMANTIC_SCHOLAR_URL, params=params)
        response.raise_for_status()
        data = response.json()

        papers = data.get("data", [])
        papers = [p for p in papers if p.get("abstract")]
        papers = papers[:limit]

        for paper in papers:
            authors = ", ".join(
                a.get("name", "") for a in paper.get("authors", []) if a.get("name")
            )
            doi = paper.get("externalIds", {}).get("DOI", "N/A")
            summary = (
                f"Title: {paper.get('title', 'N/A')}\n"
                f"Authors: {authors}\n"
                f"Year: {paper.get('year', 'N/A')}\n"
                f"Citations: {paper.get('citationCount', 0)}\n"
                f"DOI: {doi}\n"
                f"Abstract: {paper.get('abstract', '')}"
            )

            store_memory(
                text=summary,
                material_system=material_system,
                simulation_software=simulation_software,
                memory_type="literature",
            )

        return papers
