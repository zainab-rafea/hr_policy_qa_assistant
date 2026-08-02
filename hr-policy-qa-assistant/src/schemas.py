"""Pydantic schemas for structured HR Q&A output."""

from typing import Literal

from pydantic import BaseModel, Field


class HRPolicyAnswer(BaseModel):
    """Structured response returned to the employee / HR UI."""

    answer: str = Field(
        description="Clear, employee-facing answer grounded in retrieved policy text."
    )
    source_policy_section: str = Field(
        description=(
            "Cited source(s), e.g. 'Leave Policy — Section 2: Sick Leave'. "
            "List multiple sections if needed, separated by '; '."
        )
    )
    confidence_level: Literal["High", "Medium", "Low"] = Field(
        description=(
            "High = policy clearly answers the question; "
            "Medium = partial match or inference required; "
            "Low = weak/no coverage in retrieved sources."
        )
    )
    escalation_needed: Literal["Y", "N"] = Field(
        description=(
            "Y if Legal/HR review is required (legal risk, dispute, "
            "policy gap with legal implications, cross-border/tax, etc.); "
            "otherwise N."
        )
    )
    escalation_reason: str | None = Field(
        default=None,
        description="Short reason when escalation_needed is Y; null otherwise.",
    )


class RetrievalResult(BaseModel):
    """Intermediate retrieve step output."""

    question: str
    context: str
    source_snippets: list[str]
