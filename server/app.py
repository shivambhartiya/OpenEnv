"""FastAPI app exposing the Natural Language Reward Definition Environment."""

from __future__ import annotations

import argparse
import os
import socket
import uvicorn
from openenv.core.env_server import create_app

from nl_reward_env.models import (
    NaturalLanguageRewardAction,
    NaturalLanguageRewardObservation,
)
from server.natural_language_reward_environment import NaturalLanguageRewardEnvironment


DEFAULT_HOST = os.getenv("HOST", "0.0.0.0")
DEFAULT_PORT = int(os.getenv("PORT", "8000"))
DEFAULT_WORKERS = int(os.getenv("WORKERS", "2"))
DEFAULT_MAX_CONCURRENT_ENVS = int(os.getenv("MAX_CONCURRENT_ENVS", "100"))


app = create_app(
    NaturalLanguageRewardEnvironment,
    NaturalLanguageRewardAction,
    NaturalLanguageRewardObservation,
    env_name="natural_language_reward_definition_env",
    max_concurrent_envs=DEFAULT_MAX_CONCURRENT_ENVS,
)


def _is_port_available(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return sock.connect_ex((host, port)) != 0


def _next_available_port(host: str, start_port: int, attempts: int = 20) -> int:
    for port in range(start_port, start_port + attempts):
        if _is_port_available(host, port):
            return port
    raise RuntimeError(
        f"No free port found in range {start_port}-{start_port + attempts - 1}."
    )


def _parse_server_args(
    default_host: str, default_port: int, default_workers: int
) -> tuple[str, int, int, bool]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--host", default=default_host)
    parser.add_argument("--port", type=int)
    parser.add_argument("--workers", type=int)
    args, _ = parser.parse_known_args()

    env_port = os.getenv("PORT")
    explicit_port = args.port is not None or env_port is not None
    port = args.port if args.port is not None else int(env_port or default_port)
    workers = (
        args.workers if args.workers is not None else int(os.getenv("WORKERS", default_workers))
    )
    return args.host, port, workers, explicit_port


def main(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> None:
    resolved_host, resolved_port, resolved_workers, explicit_port = _parse_server_args(
        host, port, DEFAULT_WORKERS
    )

    if not _is_port_available("127.0.0.1", resolved_port):
        if explicit_port:
            raise OSError(
                f"Port {resolved_port} is already in use. Choose a different port, "
                f"for example: uv run server --port {resolved_port + 1}"
            )
        fallback_port = _next_available_port("127.0.0.1", resolved_port + 1)
        print(
            f"Port {resolved_port} is busy, starting server on port {fallback_port} instead.",
            flush=True,
        )
        resolved_port = fallback_port

    uvicorn.run(
        "server.app:app",
        host=resolved_host,
        port=resolved_port,
        workers=max(1, resolved_workers),
    )


if __name__ == "__main__":
    main()
