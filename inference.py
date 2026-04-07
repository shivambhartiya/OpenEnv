"""Baseline inference script with strict stdout logging."""

from __future__ import annotations

import asyncio
import os
from typing import List, Optional

from nl_reward_env import BaselineAgent
from nl_reward_env.environment import NaturalLanguageRewardEnvironment
from nl_reward_env.models import NaturalLanguageRewardAction
from nl_reward_env.tasks import list_tasks

MODEL_NAME = os.getenv("MODEL_NAME") or "fallback-baseline"
BENCHMARK = os.getenv(
    "NLRDE_BENCHMARK", "natural_language_reward_definition_env"
)
MAX_STEPS_OVERRIDE = int(os.getenv("NLRDE_MAX_STEPS", "0"))
SUCCESS_SCORE_THRESHOLD = 0.75


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(
    step: int,
    action: str,
    reward: float,
    done: bool,
    error: Optional[str],
) -> None:
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={str(done).lower()} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{reward:.2f}" for reward in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


async def run_task(task_id: str) -> None:
    env = NaturalLanguageRewardEnvironment()
    agent = BaselineAgent()
    rewards: List[float] = []
    steps_taken = 0
    success = False
    score = 0.0

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = env.reset(task_id=task_id)
        max_steps = MAX_STEPS_OVERRIDE or env.state.max_steps

        for step in range(1, max_steps + 1):
            if result.done:
                break

            action_text = agent.act(result, step=step)
            result = env.step(NaturalLanguageRewardAction(response=action_text))
            reward = float(result.reward or 0.0)
            rewards.append(reward)
            steps_taken = step

            log_step(
                step=step,
                action=action_text.replace("\n", "\\n"),
                reward=reward,
                done=result.done,
                error=None,
            )

            if result.done:
                break

        score = env.state.best_reward
        success = score >= SUCCESS_SCORE_THRESHOLD
    except Exception as exc:
        log_step(
            step=max(1, steps_taken + 1),
            action="error",
            reward=0.0,
            done=True,
            error=str(exc).replace("\n", " "),
        )
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


async def main() -> None:
    task_filter = os.getenv("NLRDE_TASK")
    task_ids = [task_filter] if task_filter else [task.task_id for task in list_tasks()]
    for task_id in task_ids:
        await run_task(task_id)


if __name__ == "__main__":
    asyncio.run(main())
