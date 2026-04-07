"""Pydantic models used by the OpenEnv environment."""

from __future__ import annotations

from typing import Any, Literal

from openenv.core.env_server.types import Action, Observation, State
from pydantic import BaseModel, Field


Difficulty = Literal["easy", "medium", "hard"]


class RewardModel(BaseModel):
    """Typed reward payload produced by the RewardInterpreter."""

    total: float = Field(..., ge=0.0, le=1.0)
    blended_score: float = Field(..., ge=0.0, le=1.0)
    fallback_score: float = Field(..., ge=0.0, le=1.0)
    llm_score: float | None = Field(default=None, ge=0.0, le=1.0)
    used_fallback: bool = Field(default=False)
    improvement_bonus: float = Field(default=0.0)
    penalties_applied: float = Field(default=0.0)
    rubric_scores: dict[str, float] = Field(default_factory=dict)
    penalties: list[str] = Field(default_factory=list)
    feedback: str = Field(default="")
    judge_summary: str = Field(default="")


class InteractionRecord(BaseModel):
    """Single interaction inside an episode."""

    step: int = Field(..., ge=1)
    action: str = Field(default="")
    reward: float = Field(..., ge=0.0, le=1.0)
    feedback: str = Field(default="")


class TaskDefinition(BaseModel):
    """Serializable task specification."""

    task_id: str
    name: str
    difficulty: Difficulty
    instruction: str
    scenario: str
    response_format: str
    grader_name: str
    max_steps: int = Field(default=3, ge=1)
    success_threshold: float = Field(default=0.85, ge=0.0, le=1.0)
    deterministic_targets: dict[str, Any] = Field(default_factory=dict)


class NaturalLanguageRewardAction(Action):
    """Action model accepted by the environment."""

    response: str = Field(..., description="Agent-generated output to be judged.")


class NaturalLanguageRewardObservation(Observation):
    """Observation presented to the agent."""

    benchmark: str = Field(default="natural_language_reward_definition_env")
    task_id: str = Field(default="")
    task_name: str = Field(default="")
    difficulty: Difficulty = Field(default="easy")
    instruction: str = Field(default="")
    scenario: str = Field(default="")
    response_format: str = Field(default="")
    last_feedback: str = Field(default="")
    last_submission: str = Field(default="")
    messages_remaining: int = Field(default=0, ge=0)
    reward_details: RewardModel | None = Field(default=None)


class NaturalLanguageRewardState(State):
    """Environment state exposed through `state()`."""

    benchmark: str = Field(default="natural_language_reward_definition_env")
    task_id: str | None = Field(default=None)
    task_name: str | None = Field(default=None)
    difficulty: Difficulty | None = Field(default=None)
    reward_instruction: str = Field(default="")
    scenario: str = Field(default="")
    response_format: str = Field(default="")
    max_steps: int = Field(default=0, ge=0)
    best_reward: float = Field(default=0.0, ge=0.0, le=1.0)
    last_reward: float = Field(default=0.0, ge=0.0, le=1.0)
    done: bool = Field(default=False)
    last_feedback: str = Field(default="")
    last_submission: str = Field(default="")
    best_submission: str = Field(default="")
    last_reward_details: RewardModel | None = Field(default=None)
    history: list[InteractionRecord] = Field(default_factory=list)
