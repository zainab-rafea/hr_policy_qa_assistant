"""RAG chains: retrieve → answer → flag legal review."""


from __future__ import annotations

import json
import os
import re
from typing import Any, Literal

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from src.ingest import load_or_build_vectorstore
from src.legal_flag import heuristic_escalation
from src.schemas import HRPolicyAnswer

Provider = Literal["ollama", "gemini", "anthropic"]

SYSTEM_PROMPT = """You are an internal HR Policy Q&A assistant for employees.

Answer ONLY from the policy context. Do not invent rules.

Return ONLY one JSON object with these exact keys (no markdown, no extra text):
{{
  "answer": "short employee-facing answer",
  "source_policy_section": "Policy Name — Section X: Title",
  "confidence_level": "High",
  "escalation_needed": "N",
  "escalation_reason": null
}}

Rules for the fields:
- confidence_level must be exactly "High", "Medium", or "Low"
- escalation_needed must be exactly "Y" or "N"
- Use "Y" for lawsuits, termination, discrimination, harassment, retaliation,
  cross-border/tax/legal risk, or if policy is unclear and risky; else "N"
- If escalation_needed is "N", set escalation_reason to null

Example:
{{
  "answer": "Employees are entitled to 14 paid sick days per year.",
  "source_policy_section": "Leave Policy — Section 2: Sick Leave",
  "confidence_level": "High",
  "escalation_needed": "N",
  "escalation_reason": null
}}

Policy context:
{context}
"""

HUMAN_PROMPT = "Employee question: {question}\n\nRespond with the JSON object only."

DEFAULT_OLLAMA_MODEL = "tinyllama"
DEFAULT_GEMINI_MODEL = "gemini-2.0-flash"
DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-4-20250514"


def format_docs(docs: list[Document]) -> str:
    blocks: list[str] = []
    for i, doc in enumerate(docs, start=1):
        policy = doc.metadata.get("policy_name") or doc.metadata.get("policy") or "Policy"
        section = doc.metadata.get("section_label") or doc.metadata.get("section") or "General"
        source = doc.metadata.get("source", "")
        blocks.append(
            f"[Chunk {i}] {policy} — {section}\n"
            f"File: {source}\n"
            f"{doc.page_content.strip()}"
        )
    return "\n\n---\n\n".join(blocks) if blocks else "No relevant policy sections retrieved."


def source_snippets(docs: list[Document]) -> list[str]:
    snippets: list[str] = []
    for doc in docs:
        policy = doc.metadata.get("policy_name") or "Policy"
        section = doc.metadata.get("section_label") or "General"
        preview = " ".join(doc.page_content.split())[:160]
        snippets.append(f"{policy} — {section}: {preview}...")
    return snippets


def merge_escalation(result: HRPolicyAnswer, question: str) -> HRPolicyAnswer:
    """OR LLM escalation with keyword heuristics (safer for legal risk)."""
    needs, reason = heuristic_escalation(question)
    if needs and result.escalation_needed == "N":
        return result.model_copy(
            update={
                "escalation_needed": "Y",
                "escalation_reason": reason,
            }
        )
    if needs and result.escalation_needed == "Y" and not result.escalation_reason:
        return result.model_copy(update={"escalation_reason": reason})
    return result


def _extract_json(text: str) -> dict[str, Any]:
    """Pull the first JSON object from a model response."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def _coerce_answer(data: dict[str, Any], *, question: str, context: str, docs: list[Document]) -> HRPolicyAnswer:
    """Normalize weak/small-model JSON into HRPolicyAnswer."""
    # Tiny models sometimes echo a JSON Schema instead of values.
    if "properties" in data and "answer" not in data:
        raise ValueError("Model returned a schema instead of an answer payload")

    payload = {
        "answer": data.get("answer") or data.get("Answer"),
        "source_policy_section": data.get("source_policy_section")
        or data.get("source")
        or data.get("Source Policy Section"),
        "confidence_level": data.get("confidence_level") or data.get("confidence") or "Medium",
        "escalation_needed": data.get("escalation_needed") or data.get("escalation") or "N",
        "escalation_reason": data.get("escalation_reason"),
    }

    conf = str(payload["confidence_level"]).strip().title()
    if conf not in {"High", "Medium", "Low"}:
        conf = "Medium"
    payload["confidence_level"] = conf

    esc = str(payload["escalation_needed"]).strip().upper()
    payload["escalation_needed"] = "Y" if esc in {"Y", "YES", "TRUE", "1"} else "N"
    if payload["escalation_needed"] == "N":
        payload["escalation_reason"] = None

    if not payload["source_policy_section"] and docs:
        first = docs[0]
        policy = first.metadata.get("policy_name") or "Policy"
        section = first.metadata.get("section_label") or "General"
        payload["source_policy_section"] = f"{policy} — {section}"

    if not payload["answer"]:
        # Last-resort extractive fallback so the demo still runs on tiny models.
        snippet = " ".join((docs[0].page_content if docs else context).split())[:400]
        payload["answer"] = (
            f"Based on the retrieved policy text: {snippet}"
            if snippet
            else f"I could not find a clear policy answer for: {question}"
        )
        payload["confidence_level"] = "Low"

    return HRPolicyAnswer.model_validate(payload)


def _fallback_from_docs(question: str, docs: list[Document], context: str) -> HRPolicyAnswer:
    if docs:
        first = docs[0]
        policy = first.metadata.get("policy_name") or "Policy"
        section = first.metadata.get("section_label") or "General"
        snippet = " ".join(first.page_content.split())[:400]
        source = f"{policy} — {section}"
    else:
        snippet = "No relevant policy sections were retrieved."
        source = "N/A"
    needs, reason = heuristic_escalation(question)
    return HRPolicyAnswer(
        answer=f"Based on the retrieved policy text: {snippet}",
        source_policy_section=source,
        confidence_level="Low",
        escalation_needed="Y" if needs else "N",
        escalation_reason=reason if needs else None,
    )


def get_llm(provider: Provider, model: str, temperature: float = 0.0):
    if provider == "ollama":
        from src.llm_clients import ChatOllamaHTTP

        return ChatOllamaHTTP(model=model, temperature=temperature, format="json")
    if provider == "gemini":
        from src.llm_clients import ChatGeminiHTTP

        return ChatGeminiHTTP(model=model, temperature=temperature)
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=model, temperature=temperature)
    raise ValueError(f"Unsupported provider: {provider}")


def resolve_provider_and_model(
    provider: str | None = None,
    model: str | None = None,
) -> tuple[Provider, str]:
    chosen: Provider = (provider or os.getenv("LLM_PROVIDER", "ollama")).lower()  # type: ignore[assignment]
    if chosen not in {"ollama", "gemini", "anthropic"}:
        raise ValueError("LLM_PROVIDER must be 'ollama', 'gemini', or 'anthropic'")

    if model:
        return chosen, model
    if chosen == "ollama":
        return chosen, os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)
    if chosen == "gemini":
        return chosen, os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)
    return chosen, os.getenv("ANTHROPIC_MODEL", DEFAULT_ANTHROPIC_MODEL)


def build_qa_chain(
    *,
    provider: str | None = None,
    model: str | None = None,
    k: int = 4,
    rebuild_index: bool = False,
    temperature: float = 0.0,
):
    """Build LCEL chain: retrieve relevant policy → structured answer → flag."""
    provider_name, model_name = resolve_provider_and_model(provider, model)
    vectorstore = load_or_build_vectorstore(rebuild=rebuild_index)
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    llm = get_llm(provider_name, model_name, temperature=temperature)

    if provider_name == "anthropic":
        answer_runnable = llm.with_structured_output(HRPolicyAnswer)
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_PROMPT),
                ("human", HUMAN_PROMPT),
            ]
        )
    else:
        answer_runnable = llm
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_PROMPT),
                ("human", HUMAN_PROMPT),
            ]
        )

    def retrieve_step(payload: dict[str, Any]) -> dict[str, Any]:
        question = payload["question"]
        docs = retriever.invoke(question)
        return {
            "question": question,
            "docs": docs,
            "context": format_docs(docs),
            "source_snippets": source_snippets(docs),
        }

    def answer_step(payload: dict[str, Any]) -> dict[str, Any]:
        try:
            raw = (prompt | answer_runnable).invoke(
                {"question": payload["question"], "context": payload["context"]}
            )
            if isinstance(raw, HRPolicyAnswer):
                answer = raw
            elif isinstance(raw, dict):
                answer = _coerce_answer(
                    raw,
                    question=payload["question"],
                    context=payload["context"],
                    docs=payload["docs"],
                )
            else:
                content = getattr(raw, "content", raw)
                if isinstance(content, list):
                    content = "".join(
                        part.get("text", "") if isinstance(part, dict) else str(part)
                        for part in content
                    )
                answer = _coerce_answer(
                    _extract_json(str(content)),
                    question=payload["question"],
                    context=payload["context"],
                    docs=payload["docs"],
                )
        except Exception:
            answer = _fallback_from_docs(
                payload["question"], payload["docs"], payload["context"]
            )

        answer = merge_escalation(answer, payload["question"])
        return {
            "answer": answer,
            "retrieved_snippets": payload["source_snippets"],
            "question": payload["question"],
            "provider": provider_name,
            "model": model_name,
        }

    chain = (
        RunnablePassthrough.assign(question=lambda x: x["question"])
        | RunnableLambda(retrieve_step)
        | RunnableLambda(answer_step)
    )
    return chain


def ask_policy_question(question: str, **chain_kwargs) -> dict[str, Any]:
    """Convenience helper used by the CLI."""
    chain = build_qa_chain(**chain_kwargs)
    return chain.invoke({"question": question})
