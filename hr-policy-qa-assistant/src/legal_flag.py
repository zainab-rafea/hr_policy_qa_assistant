"""Heuristics that push borderline questions toward Legal review."""

from __future__ import annotations

import re

# (human label, pattern) — labels are used in escalation reasons.
LEGAL_REVIEW_RULES: list[tuple[str, re.Pattern[str]]] = [
    (label, re.compile(pattern, re.IGNORECASE))
    for label, pattern in [
        ("lawsuit", r"\blawsuit\b"),
        ("litigation", r"\blitigation\b"),
        ("attorney", r"\battorney\b"),
        ("lawyer", r"\blawyer\b"),
        ("sue", r"\bsue\b"),
        ("legal action", r"\blegal action\b"),
        ("wrongful termination", r"\bwrongful (termination|dismissal)\b"),
        ("firing", r"\bfir(e|ed|ing)\b"),
        ("termination", r"\bterminat(e|ed|ion)\b"),
        ("discrimination", r"\bdiscriminat"),
        ("harassment", r"\bharass"),
        ("retaliation", r"\bretaliat"),
        ("whistleblowing", r"\bwhistleblow"),
        ("cross-border work", r"\bcross[- ]border\b"),
        ("another country", r"\banother country\b"),
        ("tax implications", r"\btax (implication|liability|residence)\b"),
        ("permanent establishment", r"\bpermanent establishment\b"),
        ("unlawful", r"\bunlawful\b"),
        ("illegal", r"\billegal\b"),
        ("statutory rights", r"\bstatutory\b"),
        ("labor authority", r"\blabor (board|authority|court)\b"),
        ("severance", r"\bseverance\b"),
        ("constructive dismissal", r"\bconstructive dismissal\b"),
    ]
]


def heuristic_escalation(question: str) -> tuple[bool, str | None]:
    """Return (needs_escalation, reason) from keyword heuristics alone."""
    hits = [label for label, pattern in LEGAL_REVIEW_RULES if pattern.search(question)]
    if not hits:
        return False, None
    return True, (
        "Question involves potential legal risk or statutory issues "
        f"(matched: {', '.join(hits[:3])}). Route to HR Legal."
    )
