"""Task-specific deterministic fallback graders."""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel, Field

from nl_reward_env.models import TaskDefinition


class DeterministicGrade(BaseModel):
    """Fallback grading payload."""

    score: float = Field(..., ge=0.0, le=1.0)
    rubric_scores: dict[str, float] = Field(default_factory=dict)
    penalties: list[str] = Field(default_factory=list)
    feedback: str = Field(default="")


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _contains_any(text: str, phrases: list[str]) -> bool:
    lowered = _normalize(text)
    return any(phrase.lower() in lowered for phrase in phrases)


def _clip(value: float) -> float:
    return max(0.0, min(1.0, round(value, 4)))


def _parse_json_response(text: str) -> dict[str, Any] | None:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return None
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None


class BaseDeterministicGrader:
    def grade(self, task: TaskDefinition, response: str) -> DeterministicGrade:
        raise NotImplementedError


class CustomerSupportGrader(BaseDeterministicGrader):
    def grade(self, task: TaskDefinition, response: str) -> DeterministicGrade:
        targets = task.deterministic_targets
        text = _normalize(response)
        penalties: list[str] = []

        empathy = 1.0 if _contains_any(text, targets["empathy_keywords"]) else 0.15
        next_steps = 1.0 if _contains_any(text, targets["next_step_keywords"]) else 0.2
        honesty = 1.0 if _contains_any(text, targets["honesty_keywords"]) else 0.25
        brevity = 1.0 if 60 <= len(response.split()) <= 190 else 0.65

        if _contains_any(text, targets["banned_keywords"]):
            penalties.append("Made a prohibited promise or used dismissive language.")

        if "refund" in text and "5" not in text and "business day" not in text:
            penalties.append("Mentioned refunds without clarifying the policy.")

        score = (
            0.28 * empathy
            + 0.28 * next_steps
            + 0.28 * honesty
            + 0.16 * brevity
            - 0.18 * len(penalties)
        )
        return DeterministicGrade(
            score=_clip(score),
            rubric_scores={
                "empathy": empathy,
                "next_steps": next_steps,
                "policy_accuracy": honesty,
                "conciseness": brevity,
            },
            penalties=penalties,
            feedback=(
                "Strong responses acknowledge frustration, explain policy honestly, "
                "and offer a concrete trace or follow-up."
            ),
        )


class EmailTriageGrader(BaseDeterministicGrader):
    def grade(self, task: TaskDefinition, response: str) -> DeterministicGrade:
        targets = task.deterministic_targets
        parsed = _parse_json_response(response)
        penalties: list[str] = []

        structure = 1.0 if isinstance(parsed, dict) else 0.05
        priority = 0.0
        category = 0.0
        assignee = 0.0
        safe_ops = 0.0

        if parsed:
            priority = (
                1.0
                if str(parsed.get("priority", "")).lower() == targets["expected_priority"]
                else 0.1
            )
            category_value = str(parsed.get("category", "")).lower()
            category = 1.0 if targets["expected_category"] in category_value else 0.15
            assignee_value = str(parsed.get("assignee", "")).lower()
            assignee = 1.0 if targets["expected_assignee"] in assignee_value else 0.2
            response_bits = " ".join(
                str(parsed.get(key, "")) for key in ("justification", "response_draft")
            )
            safe_ops = (
                1.0
                if _contains_any(response_bits, targets["safe_response_keywords"])
                else 0.15
            )

        lowered = _normalize(response)
        if _contains_any(lowered, targets["banned_keywords"]):
            penalties.append("Recommended an unsafe routing or action.")
        if structure < 1.0:
            penalties.append("Output was not valid JSON.")

        score = (
            0.18 * structure
            + 0.22 * priority
            + 0.2 * category
            + 0.2 * assignee
            + 0.2 * safe_ops
            - 0.18 * len(penalties)
        )
        return DeterministicGrade(
            score=_clip(score),
            rubric_scores={
                "json_structure": structure,
                "priority": priority,
                "category": category,
                "assignee": assignee,
                "safe_response": safe_ops,
            },
            penalties=penalties,
            feedback=(
                "High-scoring triage marks this as high-priority security fraud, routes "
                "it to security, and blocks payment pending trusted verification."
            ),
        )


class CodeReviewGrader(BaseDeterministicGrader):
    def grade(self, task: TaskDefinition, response: str) -> DeterministicGrade:
        targets = task.deterministic_targets
        text = _normalize(response)
        penalties: list[str] = []

        risk = 1.0 if _contains_any(text, targets["risk_keywords"]) else 0.0
        line_ref = 1.0 if _contains_any(text, targets["line_keywords"]) else 0.2
        remediation = 1.0 if _contains_any(text, targets["fix_keywords"]) else 0.1
        severity = 1.0 if any(token in text for token in ("high", "critical", "security")) else 0.25

        if risk == 0.0:
            penalties.append("Missed the authorization or privilege escalation issue.")
        if any(token in text for token in ("nit", "formatting", "style")) and risk < 1.0:
            penalties.append("Focused on style instead of the main behavioral risk.")

        score = (
            0.35 * risk
            + 0.2 * line_ref
            + 0.25 * remediation
            + 0.2 * severity
            - 0.2 * len(penalties)
        )
        return DeterministicGrade(
            score=_clip(score),
            rubric_scores={
                "risk_detection": risk,
                "line_reference": line_ref,
                "remediation": remediation,
                "severity": severity,
            },
            penalties=penalties,
            feedback=(
                "The key issue is that an authenticated user can set arbitrary roles on "
                "lines 5-6. A strong review explains the impact and proposes a server-side guard."
            ),
        )


class DeterministicGraderRegistry:
    """Registry for deterministic fallback graders."""

    _graders: dict[str, BaseDeterministicGrader] = {
        "customer_support": CustomerSupportGrader(),
        "email_triage": EmailTriageGrader(),
        "code_review": CodeReviewGrader(),
    }

    @classmethod
    def grade(cls, task: TaskDefinition, response: str) -> DeterministicGrade:
        return cls._graders[task.grader_name].grade(task, response)
