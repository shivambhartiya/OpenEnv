"""OpenEnv client for the Natural Language Reward Definition Environment."""

from __future__ import annotations

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

from nl_reward_env.models import (
    NaturalLanguageRewardAction,
    NaturalLanguageRewardObservation,
    NaturalLanguageRewardState,
    RewardModel,
)


class NaturalLanguageRewardEnv(
    EnvClient[
        NaturalLanguageRewardAction,
        NaturalLanguageRewardObservation,
        NaturalLanguageRewardState,
    ]
):
    """WebSocket client for the environment server."""

    def _step_payload(self, action: NaturalLanguageRewardAction) -> Dict:
        return {
            "response": action.response,
            "metadata": action.metadata,
        }

    def _parse_result(self, payload: Dict) -> StepResult[NaturalLanguageRewardObservation]:
        obs_data = payload.get("observation", {})
        reward_details_payload = obs_data.get("reward_details")
        reward_details = (
            RewardModel.model_validate(reward_details_payload)
            if reward_details_payload
            else None
        )

        observation = NaturalLanguageRewardObservation(
            benchmark=obs_data.get("benchmark", "natural_language_reward_definition_env"),
            task_id=obs_data.get("task_id", ""),
            task_name=obs_data.get("task_name", ""),
            difficulty=obs_data.get("difficulty", "easy"),
            instruction=obs_data.get("instruction", ""),
            scenario=obs_data.get("scenario", ""),
            response_format=obs_data.get("response_format", ""),
            last_feedback=obs_data.get("last_feedback", ""),
            last_submission=obs_data.get("last_submission", ""),
            messages_remaining=obs_data.get("messages_remaining", 0),
            reward_details=reward_details,
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> NaturalLanguageRewardState:
        return NaturalLanguageRewardState.model_validate(payload)
