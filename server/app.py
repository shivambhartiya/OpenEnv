"""FastAPI app exposing the Natural Language Reward Definition Environment."""

from __future__ import annotations

import uvicorn
from openenv.core.env_server import create_app

from nl_reward_env.models import (
    NaturalLanguageRewardAction,
    NaturalLanguageRewardObservation,
)
from server.natural_language_reward_environment import NaturalLanguageRewardEnvironment


app = create_app(
    NaturalLanguageRewardEnvironment,
    NaturalLanguageRewardAction,
    NaturalLanguageRewardObservation,
    env_name="natural_language_reward_definition_env",
    max_concurrent_envs=8,
)


def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
