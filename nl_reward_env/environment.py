"""In-process environment implementation."""

from __future__ import annotations

from uuid import uuid4

from openenv.core.env_server.interfaces import Environment

from nl_reward_env.config import load_runtime_config
from nl_reward_env.models import (
    InteractionRecord,
    NaturalLanguageRewardAction,
    NaturalLanguageRewardObservation,
    NaturalLanguageRewardState,
)
from nl_reward_env.reward_interpreter import RewardInterpreter
from nl_reward_env.tasks import get_task


class NaturalLanguageRewardEnvironment(
    Environment[
        NaturalLanguageRewardAction,
        NaturalLanguageRewardObservation,
        NaturalLanguageRewardState,
    ]
):
    """OpenEnv-compatible environment whose reward is defined in plain English."""

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self):
        super().__init__()
        self._runtime = load_runtime_config()
        self._reward_interpreter = RewardInterpreter(self._runtime)
        self._current_task = get_task(self._runtime.default_task_id)
        self._state = self._blank_state()

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        task_id: str | None = None,
        **kwargs,
    ) -> NaturalLanguageRewardObservation:
        del seed, kwargs
        self._current_task = get_task(task_id or self._runtime.default_task_id)
        self._state = self._blank_state(episode_id=episode_id)
        self._state.task_id = self._current_task.task_id
        self._state.task_name = self._current_task.name
        self._state.difficulty = self._current_task.difficulty
        self._state.reward_instruction = self._current_task.instruction
        self._state.scenario = self._current_task.scenario
        self._state.response_format = self._current_task.response_format
        self._state.max_steps = self._current_task.max_steps
        self._state.last_feedback = (
            "No submission yet. Aim for a strong first draft that follows the task format."
        )
        return self._observation()

    def step(
        self,
        action: NaturalLanguageRewardAction,
        timeout_s: float | None = None,
        **kwargs,
    ) -> NaturalLanguageRewardObservation:
        del timeout_s, kwargs
        self._state.step_count += 1

        reward = self._reward_interpreter.interpret(
            task=self._current_task,
            agent_output=action.response,
            state=self._state,
        )

        self._state.last_reward = reward.total
        self._state.last_reward_details = reward
        self._state.last_submission = action.response
        self._state.last_feedback = reward.feedback
        if reward.total >= self._state.best_reward:
            self._state.best_reward = reward.total
            self._state.best_submission = action.response

        self._state.done = (
            self._state.step_count >= self._current_task.max_steps
            or reward.total >= self._current_task.success_threshold
        )
        self._state.history.append(
            InteractionRecord(
                step=self._state.step_count,
                action=action.response,
                reward=reward.total,
                feedback=reward.feedback,
            )
        )

        return self._observation(reward=reward)

    @property
    def state(self) -> NaturalLanguageRewardState:
        return self._state

    def get_metadata(self):
        metadata = super().get_metadata()
        metadata.name = "Natural Language Reward Definition Environment"
        metadata.description = (
            "RL environment with plain-English reward definitions interpreted by an "
            "LLM judge plus deterministic fallback graders."
        )
        metadata.version = "1.0.0"
        return metadata

    def _blank_state(self, episode_id: str | None = None) -> NaturalLanguageRewardState:
        return NaturalLanguageRewardState(
            episode_id=episode_id or str(uuid4()),
            benchmark=self._runtime.benchmark_name,
            step_count=0,
        )

    def _observation(
        self,
        reward=None,
    ) -> NaturalLanguageRewardObservation:
        messages_remaining = max(0, self._current_task.max_steps - self._state.step_count)
        return NaturalLanguageRewardObservation(
            benchmark=self._runtime.benchmark_name,
            task_id=self._current_task.task_id,
            task_name=self._current_task.name,
            difficulty=self._current_task.difficulty,
            instruction=self._current_task.instruction,
            scenario=self._current_task.scenario,
            response_format=self._current_task.response_format,
            last_feedback=self._state.last_feedback,
            last_submission=self._state.last_submission,
            messages_remaining=messages_remaining,
            reward_details=reward,
            reward=0.0 if reward is None else reward.total,
            done=self._state.done,
            metadata={
                "best_reward": self._state.best_reward,
                "best_submission": self._state.best_submission,
                "step_count": self._state.step_count,
            },
        )
