"""LLM-based reward interpreter with deterministic fallback."""

from __future__ import annotations

import json
from typing import Any

from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError

from nl_reward_env.config import RuntimeConfig, load_runtime_config
from nl_reward_env.graders import DeterministicGraderRegistry
from nl_reward_env.models import NaturalLanguageRewardState, RewardModel, TaskDefinition


class LLMJudgeOutput(BaseModel):
    """Structured JSON returned by the LLM judge."""

    score: float = Field(..., ge=0.0, le=1.0)
    rubric_scores: dict[str, float] = Field(default_factory=dict)
    penalties: list[str] = Field(default_factory=list)
    strengths: list[str] = Field(default_factory=list)
    concerns: list[str] = Field(default_factory=list)
    summary: str = Field(default="")


class RewardInterpreter:
    """Interpret plain-English reward instructions using an LLM judge."""

    def __init__(self, config: RuntimeConfig | None = None):
        self.config = config or load_runtime_config()
        self._client = (
            OpenAI(
                base_url=self.config.api_base_url,
                api_key=self.config.api_key,
            )
            if self.config.api_key and self.config.model_name
            else None
        )

    def interpret(
        self,
        task: TaskDefinition,
        agent_output: str,
        state: NaturalLanguageRewardState,
    ) -> RewardModel:
        """Return a dense reward in `[0, 1]`."""
        deterministic = DeterministicGraderRegistry.grade(task, agent_output)
        llm_judgment = self._judge_with_llm(
            task,
            agent_output,
            state,
            deterministic.model_dump(),
        )

        if llm_judgment is None:
            blended = deterministic.score
            llm_score = None
            judge_summary = deterministic.feedback
            used_fallback = True
            merged_penalties = list(deterministic.penalties)
            rubric_scores = dict(deterministic.rubric_scores)
        else:
            blended = 0.7 * llm_judgment.score + 0.3 * deterministic.score
            llm_score = llm_judgment.score
            judge_summary = llm_judgment.summary
            used_fallback = False
            merged_penalties = list(
                dict.fromkeys(deterministic.penalties + llm_judgment.penalties)
            )
            rubric_scores = dict(deterministic.rubric_scores)
            for key, value in llm_judgment.rubric_scores.items():
                rubric_scores[f"llm_{key}"] = max(0.0, min(1.0, float(value)))

        previous_best = state.best_reward
        improvement_bonus = max(0.0, blended - previous_best) * 0.12
        penalties_applied = 0.0

        if not agent_output.strip():
            merged_penalties.append("Submitted an empty response.")
            penalties_applied += 0.25

        if state.last_submission and agent_output.strip() == state.last_submission.strip():
            merged_penalties.append("Repeated the previous answer without improvement.")
            penalties_applied += 0.1

        penalties_applied += 0.03 * max(0, state.step_count - 1)
        penalties_applied += 0.04 * len(deterministic.penalties)

        total = self._clip(blended + improvement_bonus - penalties_applied)
        feedback = self._build_feedback(
            deterministic.feedback,
            judge_summary,
            merged_penalties,
        )

        return RewardModel(
            total=total,
            blended_score=self._clip(blended),
            fallback_score=deterministic.score,
            llm_score=None if llm_score is None else self._clip(llm_score),
            used_fallback=used_fallback,
            improvement_bonus=round(improvement_bonus, 4),
            penalties_applied=round(penalties_applied, 4),
            rubric_scores=rubric_scores,
            penalties=merged_penalties,
            feedback=feedback,
            judge_summary=judge_summary,
        )

    def _judge_with_llm(
        self,
        task: TaskDefinition,
        agent_output: str,
        state: NaturalLanguageRewardState,
        deterministic_payload: dict[str, Any],
    ) -> LLMJudgeOutput | None:
        if self._client is None or not self.config.model_name:
            return None

        state_payload = {
            "task_id": state.task_id,
            "task_name": state.task_name,
            "step_count": state.step_count,
            "best_reward": state.best_reward,
            "last_feedback": state.last_feedback,
            "history": [entry.model_dump() for entry in state.history[-2:]],
        }
        user_payload = {
            "reward_instruction": task.instruction,
            "scenario": task.scenario,
            "response_format": task.response_format,
            "agent_output": agent_output,
            "state": state_payload,
            "deterministic_fallback": deterministic_payload,
        }

        try:
            completion = self._client.chat.completions.create(
                model=self.config.model_name,
                temperature=self.config.judge_temperature,
                response_format={"type": "json_object"},
                max_completion_tokens=600,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a strict reinforcement-learning reward judge. "
                            "Read the natural-language reward instruction and score the "
                            "agent output from 0 to 1. Return JSON only with keys: "
                            "score, rubric_scores, penalties, strengths, concerns, summary."
                        ),
                    },
                    {
                        "role": "user",
                        "content": json.dumps(user_payload, ensure_ascii=True, indent=2),
                    },
                ],
            )
            text = (completion.choices[0].message.content or "").strip()
            payload = json.loads(text)
            return LLMJudgeOutput.model_validate(payload)
        except (ValidationError, ValueError, KeyError, TypeError, json.JSONDecodeError, Exception):
            return None

    @staticmethod
    def _clip(value: float) -> float:
        return max(0.0, min(1.0, round(float(value), 4)))

    @staticmethod
    def _build_feedback(
        deterministic_feedback: str,
        llm_summary: str,
        penalties: list[str],
    ) -> str:
        parts = [part for part in (llm_summary, deterministic_feedback) if part]
        if penalties:
            parts.append("Penalties: " + "; ".join(penalties[:3]))
        return " ".join(parts).strip()
