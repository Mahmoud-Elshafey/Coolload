"""
rag_agent.py

Purpose
-------
Answers a building manager's natural-language questions about the
HVAC optimization plan by retrieving relevant day/summary documents
(rag_retriever.py) and passing them as context to a Groq-hosted LLM.

Design decisions
-----------------
1. API key is read from GROQ_API_KEY in .env at the project root.
   load_dotenv() is called once, explicitly, before anything reads
   the env var -- the previous version defined a _load_api_key()
   helper that did this correctly but never actually called it, so
   GROQ_API_KEY silently came from the ambient shell environment (or
   was None) instead of the .env file.

2. Model: "openai/gpt-oss-120b". Groq has deprecated its earlier
   Llama chat models (llama-3.3-70b-versatile, llama-3.1-8b-instant)
   and now points general-purpose/reasoning workloads at the GPT-OSS
   family. 120B is used here (rather than the smaller 20B) because
   manager questions often require synthesizing across several
   retrieved day-briefs (e.g. "which days saved the most and why"),
   and Groq's inference speed makes the larger model's latency a
   non-issue for a chat-style demo. If a demo ever needs to shave
   latency further, swap MODEL to "openai/gpt-oss-20b" -- no other
   code changes needed.

3. Groq's SDK is OpenAI-compatible, NOT the Anthropic Messages API.
   That means: `client.chat.completions.create(model=..., messages=[
   {"role": "system", ...}, {"role": "user", ...}])`, and the reply
   is read from `response.choices[0].message.content` -- not
   `client.messages.create(system=..., ...)` / `response.content`
   blocks, which is Anthropic's shape and does not exist on a Groq
   client. (The previous version of this file mixed the two and
   would raise AttributeError on the first real question.)

4. The system prompt explicitly instructs the model to answer only
   from the provided context and to say so when the answer isn't
   present there, rather than inventing numbers -- important for a
   tool a manager will use to reason about real energy costs.

5. This module exposes a plain `RagAgent.ask(question)` method plus a
   small CLI loop, so it can be called directly from a script or
   notebook now and wired into a FastAPI endpoint later (the "AI
   Agent" -> "FastAPI" pipeline stages) without changing this logic.

Setup
-----
    pip install groq python-dotenv scikit-learn

.env at the project root must contain:
    GROQ_API_KEY=gsk_...

Usage
-----
    python scripts/rag_agent.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq

SCRIPTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPTS_DIR.parents[0]
sys.path.insert(0, str(SCRIPTS_DIR))

from retrieve import RagDocument, RagRetriever  # noqa: E402

MODEL = "openai/gpt-oss-120b"
MAX_TOKENS = 1500
TOP_K_DAYS = 2

# Groq's free/on-demand tier caps openai/gpt-oss-120b at 8,000 tokens
# PER REQUEST-MINUTE (prompt + reserved completion combined). Each
# retrieved day document contains a full 24-row hourly table (~700+
# tokens), so TOP_K_DAYS=5 (the original setting) plus the monthly
# summary could already exceed 8,000 tokens before the model writes a
# single word of its answer -- that's the literal cause of the 413
# "Request too large" error. TOP_K_DAYS=2 keeps normal questions well
# under budget; MAX_CONTEXT_CHARS_PER_DAY_DOC below is a second,
# independent safety net so a single oversized day document (or a
# future larger dataset) can't blow the budget even if top_k changes.
MAX_CONTEXT_CHARS_PER_DAY_DOC = 900
# Change MAX_TOKENS from 700 to 1500 to prevent truncation


SYSTEM_PROMPT = (
    "You are CoolLoad AI's HVAC optimization assistant for building managers.\n\n"
    "Strict Rules:\n"
    "- Answer ONLY using the provided context documents.\n"
    "- Focus strictly on high-level managerial metrics: Daily Cost ($), Peak Load (kW), Peak Hour, and Comfort Penalty.\n"
    "- DO NOT generate 24-hour tables or hour-by-hour breakdowns unless the manager explicitly asks for 'all hours', 'every hour', or 'full table'.\n"
    "- Summarize daily operational plans in concise bullet points covering only key events (e.g., pre-cooling phase, peak shaving window, and setpoint changes).\n"
    "- If context is missing to answer the query, state so explicitly."
)


def _load_api_key() -> str:
    """Load .env from the project root and return GROQ_API_KEY.

    This is the single place .env gets loaded -- called once from
    RagAgent.__init__ before anything reads the environment, so the
    key is never silently pulled from an unrelated ambient env var.
    """
    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY not found. Add it to the .env file at the "
            "project root, e.g.:\n  GROQ_API_KEY=gsk_..."
        )
    return api_key


def _build_context_block(documents: list[RagDocument]) -> str:
    """Concatenate retrieved documents into a single context string.

    Day documents (everything except "monthly_summary") are truncated
    to MAX_CONTEXT_CHARS_PER_DAY_DOC. This is a hard safety net on
    token budget: it caps context size regardless of TOP_K_DAYS or how
    verbose export_rag_data.py's hourly table gets as the dataset
    grows, so a single question can't silently exceed Groq's per-
    request token limit again. The summary document is left intact --
    it's already short and many questions depend on its exact figures.
    """
    parts = []
    for doc in documents:
        text = doc.text
        if doc.doc_id != "monthly_summary" and len(text) > MAX_CONTEXT_CHARS_PER_DAY_DOC:
            text = (
                text[:MAX_CONTEXT_CHARS_PER_DAY_DOC]
                + "\n[...hourly detail truncated for length...]"
            )
        parts.append(f"--- Document: {doc.doc_id} ---\n{text}")
    return "\n\n".join(parts)


class RagAgent:
    """Retrieval-augmented question answering over the optimizer output."""

    def __init__(self, top_k: int = TOP_K_DAYS) -> None:
        api_key = _load_api_key()
        self.client = Groq(api_key=api_key)
        self.retriever = RagRetriever()
        self.top_k = top_k

    def ask(self, question: str) -> str:
        retrieved = self.retriever.retrieve(question, top_k=self.top_k)
        print(f"\n[Debug] Context documents retrieved: {[doc.doc_id for doc in retrieved]}")
        
        context = _build_context_block(retrieved)

        user_message = (
            f"Context documents:\n\n{context}\n\n"
            f"Manager's question: {question}"
        )

        response = self.client.chat.completions.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
        )

        return response.choices[0].message.content or ""


def main() -> None:
    print("CoolLoad AI — Manager Q&A (type 'exit' to quit)\n")
    agent = RagAgent()

    while True:
        question = input("Ask about the HVAC plan: ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue
        try:
            answer = agent.ask(question)
        except Exception as exc:  # surface API/config errors clearly, don't crash the loop
            print(f"\n[Error] {exc}\n")
            continue
        print(f"\n{answer}\n")


if __name__ == "__main__":
    main()