# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Entry point for the ``dd-web`` command.

Serves the agent over HTTP and WebSocket so the browser UI can drive it.
Configuration comes from the same environment variables as ``dd-agent``; only
the listening address is specific to the gateway.
"""

from __future__ import annotations

import argparse
import logging
import os

DEFAULT_PORT = 8000


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="dd-web",
        description=(
            "Web gateway for the drug design agent.  Reads the same "
            "REINVENT4_AGENT_* environment variables as dd-agent."
        ),
    )
    parser.add_argument(
        "--host",
        default=os.environ.get("REINVENT4_WEB_HOST", "0.0.0.0"),  # noqa: S104 - container service
        help="Address to listen on (default: 0.0.0.0, or REINVENT4_WEB_HOST)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("REINVENT4_WEB_PORT") or DEFAULT_PORT),
        help=f"Port to listen on (default: {DEFAULT_PORT}, or REINVENT4_WEB_PORT)",
    )
    parser.add_argument(
        "--log-level",
        default=os.environ.get("REINVENT4_WEB_LOG_LEVEL", "info"),
        choices=["critical", "error", "warning", "info", "debug"],
        help="Log level (default: info, or REINVENT4_WEB_LOG_LEVEL)",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )

    import uvicorn
    from reinvent4_agent.web.server import create_app

    uvicorn.run(create_app(), host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":
    main()
