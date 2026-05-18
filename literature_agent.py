from typing import Optional
import requests
import xml.etree.ElementTree as ET

from hippocampus import store_memory


ARXIV_URL = "http://export.arxiv.org/api/query"
ATOM_NS = "{http://www.w3.org/2005/Atom}"


class LiteratureAgent:
    def search(
        self,
        topic: str,
        limit: int = 5,
        material_system: Optional[str] = None,
        simulation_software: Optional[str] = None,
    ) -> list[dict]:
        params = {
            "search_query": f"all:{topic}",
            "max_results": 10,
            "sortBy": "relevance",
            "sortOrder": "descending",
        }

        response = requests.get(ARXIV_URL, params=params)
        response.raise_for_status()

        root = ET.fromstring(response.content)
        entries = root.findall(f"{ATOM_NS}entry")

        papers = []
        for entry in entries[:limit]:
            title = entry.find(f"{ATOM_NS}title").text.strip()
            abstract = entry.find(f"{ATOM_NS}summary").text.strip()[:300]
            published = entry.find(f"{ATOM_NS}published").text
            year = published[:4] if published else "N/A"

            author_elements = entry.findall(f"{ATOM_NS}author")
            authors = []
            for a in author_elements[:3]:
                name_elem = a.find(f"{ATOM_NS}name")
                if name_elem is not None and name_elem.text:
                    authors.append(name_elem.text.strip())
            author_str = ", ".join(authors) if authors else "N/A"

            summary = (
                f"Title: {title}\n"
                f"Authors: {author_str}\n"
                f"Year: {year}\n"
                f"Abstract: {abstract}"
            )

            store_memory(
                text=summary,
                material_system=material_system,
                simulation_software=simulation_software,
                memory_type="literature",
            )

            papers.append({
                "title": title,
                "authors": author_str,
                "year": year,
                "abstract": abstract,
            })

        return papers
