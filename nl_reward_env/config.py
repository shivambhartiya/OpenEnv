"""Configuration helpers for the environment and inference stack."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(slots=True)
class RuntimeConfig:
    api_base_url: str | None
    model_name: str | None
    api_key: str | None
    benchmark_name: str = "natural_language_reward_definition_env"
    default_task_id: str = "customer_support_response"
    default_temperature: float = 0.2
    judge_temperature: float = 0.0


def load_runtime_config() -> RuntimeConfig:
    """Load runtime settings from environment variables."""
    api_key = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
    api_base_url = os.getenv("API_BASE_URL")
    model_name = os.getenv("MODEL_NAME")
    return RuntimeConfig(
        api_base_url=api_base_url,
        model_name=model_name,
        api_key=api_key,
        benchmark_name=os.getenv(
            "NLRDE_BENCHMARK", "natural_language_reward_definition_env"
        ),
        default_task_id=os.getenv("NLRDE_TASK", "customer_support_response"),
        default_temperature=float(os.getenv("NLRDE_TEMPERATURE", "0.2")),
        judge_temperature=float(os.getenv("NLRDE_JUDGE_TEMPERATURE", "0.0")),
    )
