#!/usr/bin/env python3
"""CLI for the AI HR Policy Q&A Assistant."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Ensure project root is on sys.path when run as `python app.py`
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.chains import (
    DEFAULT_ANTHROPIC_MODEL,
    DEFAULT_GEMINI_MODEL,
    DEFAULT_OLLAMA_MODEL,
    ask_policy_question,
    build_qa_chain,
    resolve_provider_and_model,
)
from src.ingest import build_vectorstore
from src.schemas import HRPolicyAnswer


def print_result(result: dict) -> None:
    answer: HRPolicyAnswer = result["answer"]
    # Windows consoles often can't render em dashes; normalize for display.
    def _safe(text: str) -> str:
        return text.replace("—", "-").replace("•", "-")

    print("\n" + "=" * 60)
    print("HR POLICY ANSWER")
    print("=" * 60)
    if result.get("provider"):
        print(f"\nProvider / Model:     {result['provider']} / {result.get('model')}")
    print(f"\nAnswer:\n  {_safe(answer.answer)}")
    print(f"\nSource Policy Section:\n  {_safe(answer.source_policy_section)}")
    print(f"\nConfidence Level:     {answer.confidence_level}")
    print(f"Escalation Needed:    {answer.escalation_needed}")
    if answer.escalation_reason:
        print(f"Escalation Reason:    {_safe(answer.escalation_reason)}")
    if result.get("retrieved_snippets"):
        print("\nRetrieved snippets:")
        for snip in result["retrieved_snippets"]:
            print(f"  - {_safe(snip)}")
    print("=" * 60 + "\n")


def print_json(result: dict) -> None:
    answer: HRPolicyAnswer = result["answer"]
    payload = {
        "answer": answer.answer,
        "source_policy_section": answer.source_policy_section,
        "confidence_level": answer.confidence_level,
        "escalation_needed": answer.escalation_needed,
        "escalation_reason": answer.escalation_reason,
        "retrieved_snippets": result.get("retrieved_snippets", []),
        "provider": result.get("provider"),
        "model": result.get("model"),
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def interactive_loop(chain) -> None:
    print("AI HR Policy Q&A Assistant")
    print("Ask about leave, remote work, code of conduct, or local labor law.")
    print("Type 'quit' or 'exit' to leave.\n")
    while True:
        try:
            question = input("Employee question> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break
        if not question:
            continue
        if question.lower() in {"quit", "exit", "q"}:
            print("Goodbye.")
            break
        try:
            result = chain.invoke({"question": question})
            print_result(result)
        except Exception as exc:  # noqa: BLE001 — surface API/config errors to user
            print(f"\nError: {exc}\n", file=sys.stderr)


def main() -> int:
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="AI HR Policy Q&A Assistant (RAG + structured output)"
    )
    parser.add_argument("-q", "--question", help="Single employee question (non-interactive)")
    parser.add_argument(
        "--rebuild-index",
        action="store_true",
        help="Rebuild the FAISS index from policies/",
    )
    parser.add_argument(
        "--index-only",
        action="store_true",
        help="Only build/persist the vector store, then exit",
    )
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    parser.add_argument(
        "--provider",
        choices=["ollama", "gemini", "anthropic"],
        default=os.getenv("LLM_PROVIDER", "ollama"),
        help="LLM backend (default: ollama — free/local)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            f"Model id (gemini default: {DEFAULT_GEMINI_MODEL}; "
            f"ollama default: {DEFAULT_OLLAMA_MODEL}; "
            f"anthropic default: {DEFAULT_ANTHROPIC_MODEL})"
        ),
    )
    parser.add_argument("-k", type=int, default=4, help="Number of chunks to retrieve")
    args = parser.parse_args()

    if args.index_only:
        build_vectorstore()
        print("Vector store built and saved under ./vectorstore")
        return 0

    provider, model = resolve_provider_and_model(args.provider, args.model)

    if provider == "anthropic" and not os.getenv("ANTHROPIC_API_KEY"):
        print(
            "Missing ANTHROPIC_API_KEY for --provider anthropic.\n"
            "Use a free option instead:\n"
            "  python app.py --provider gemini\n"
            "  python app.py --provider ollama",
            file=sys.stderr,
        )
        return 1

    if provider == "gemini" and not (
        os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    ):
        print(
            "Missing GOOGLE_API_KEY.\n"
            "1) Get a free key: https://aistudio.google.com/apikey\n"
            "2) Put it in .env as GOOGLE_API_KEY=...\n"
            "Or use local Ollama (no key): python app.py --provider ollama",
            file=sys.stderr,
        )
        return 1

    chain_kwargs = {
        "provider": provider,
        "model": model,
        "k": args.k,
        "rebuild_index": args.rebuild_index,
    }

    label = {
        "ollama": "free/local",
        "gemini": "free cloud",
        "anthropic": "paid cloud",
    }.get(provider, provider)
    print(f"Using {label} backend: {provider} / {model}")

    if args.question:
        result = ask_policy_question(args.question, **chain_kwargs)
        if args.json:
            print_json(result)
        else:
            print_result(result)
        return 0

    chain = build_qa_chain(**chain_kwargs)
    interactive_loop(chain)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

