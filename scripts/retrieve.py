from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAG_DIR = PROJECT_ROOT / "data" / "rag"

MONTH_MAP = {
    "jan": "01", "january": "01",
    "feb": "02", "february": "02",
    "mar": "03", "march": "03",
    "apr": "04", "april": "04",
    "may": "05",
    "jun": "06", "june": "06",
    "jul": "07", "july": "07",
    "aug": "08", "august": "08",
    "sep": "09", "september": "09",
    "oct": "10", "october": "10",
    "nov": "11", "november": "11",
    "dec": "12", "december": "12",
}


@dataclass
class RagDocument:
    doc_id: str
    text: str


def _extract_date_pattern(query: str) -> str | None:
    """Extract formatted date patterns like '-07-10' or '2024-07-10' from queries."""
    query_lower = query.lower()

    # 1. Match YYYY-MM-DD
    iso_match = re.search(r"\b\d{4}-\d{2}-\d{2}\b", query_lower)
    if iso_match:
        return iso_match.group(0)

    months_regex = "|".join(MONTH_MAP.keys())
    
    # 2. Match "10 july", "10th july", "10th of july"
    pat_day_first = rf"\b(\d{{1,2}})(?:st|nd|rd|th)?(?:\s+of)?\s+({months_regex})\b"
    m1 = re.search(pat_day_first, query_lower)
    if m1:
        day = int(m1.group(1))
        month = MONTH_MAP[m1.group(2)]
        return f"-{month}-{day:02d}"

    # 3. Match "july 10", "july 10th"
    pat_month_first = rf"\b({months_regex})\s+(\d{{1,2}})(?:st|nd|rd|th)?\b"
    m2 = re.search(pat_month_first, query_lower)
    if m2:
        day = int(m2.group(2))
        month = MONTH_MAP[m2.group(1)]
        return f"-{month}-{day:02d}"

    return None


class RagRetriever:
    """TF-IDF retriever over daily plan documents with explicit date matching."""

    def __init__(self, rag_dir: Path = RAG_DIR) -> None:
        self.rag_dir = rag_dir
        self.days_dir = rag_dir / "days"
        self.summary_path = rag_dir / "monthly_summary.txt"

        self.documents: list[RagDocument] = self._load_documents()
        if not self.documents:
            raise FileNotFoundError(
                f"No RAG documents found under {self.rag_dir}."
            )

        self._vectorizer = TfidfVectorizer(stop_words="english")
        self._matrix = self._vectorizer.fit_transform(
            [doc.text for doc in self.documents]
        )

    def _load_documents(self) -> list[RagDocument]:
        docs: list[RagDocument] = []

        if self.summary_path.exists():
            docs.append(
                RagDocument(
                    doc_id="monthly_summary",
                    text=self.summary_path.read_text(encoding="utf-8"),
                )
            )

        if self.days_dir.exists():
            for path in sorted(self.days_dir.glob("*.txt")):
                docs.append(
                    RagDocument(
                        doc_id=path.stem, text=path.read_text(encoding="utf-8")
                    )
                )

        return docs

    def retrieve(self, query: str, top_k: int = 5) -> list[RagDocument]:
        selected: list[RagDocument] = []
        selected_ids: set[str] = set()

        # Always include monthly summary
        summary_doc = next(
            (d for d in self.documents if d.doc_id == "monthly_summary"), None
        )
        if summary_doc is not None:
            selected.append(summary_doc)
            selected_ids.add(summary_doc.doc_id)

        day_budget = top_k

        # Direct Date Matching: Force inclusion if date is detected
        date_pattern = _extract_date_pattern(query)
        if date_pattern:
            for doc in self.documents:
                if date_pattern in doc.doc_id and doc.doc_id not in selected_ids:
                    selected.append(doc)
                    selected_ids.add(doc.doc_id)
                    day_budget -= 1
                    break

        # TF-IDF Ranking for remaining slot budget
        if day_budget > 0:
            query_vec = self._vectorizer.transform([query])
            scores = cosine_similarity(query_vec, self._matrix).flatten()
            ranked_indices = scores.argsort()[::-1]

            for idx in ranked_indices:
                doc = self.documents[idx]
                if doc.doc_id in selected_ids:
                    continue
                if scores[idx] <= 0.0:
                    break
                selected.append(doc)
                selected_ids.add(doc.doc_id)
                day_budget -= 1
                if day_budget <= 0:
                    break

        return selected