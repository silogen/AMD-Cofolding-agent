# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Entry point for the ``dd-agent`` command.

Launches the interactive terminal chatbot that connects to the Drug Discovery
MCP server and uses an LLM for reasoning about drug design tasks.
Supports both Anthropic and OpenAI-compatible backends.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from typing import TYPE_CHECKING, Any

from reinvent4_agent import __version__

if TYPE_CHECKING:
    from reinvent4_agent.agent import Agent
    from reinvent4_agent.coordinator import Coordinator


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="dd-agent",
        description=(
            "Interactive drug design agent powered by an LLM and REINVENT4. "
            "Connects to the Drug Discovery MCP server for molecular scoring, "
            "component discovery, and multi-property optimization."
        ),
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    # -- LLM backend ---------------------------------------------------
    parser.add_argument(
        "--backend",
        choices=["anthropic", "openai"],
        default=os.environ.get("REINVENT4_AGENT_BACKEND", "anthropic"),
        help=(
            "LLM backend: 'anthropic' (Anthropic Messages API) or "
            "'openai' (OpenAI-compatible Chat Completions API) "
            "(default: anthropic, or REINVENT4_AGENT_BACKEND env var)"
        ),
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("REINVENT4_AGENT_MODEL", "claude-sonnet-4"),
        help=(
            "Model name as the endpoint knows it "
            "(default: claude-sonnet-4, or REINVENT4_AGENT_MODEL env var)"
        ),
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("REINVENT4_AGENT_BASE_URL"),
        help=(
            "Base URL for the LLM API endpoint "
            "(default: REINVENT4_AGENT_BASE_URL env var; "
            "omit for the SDK default)"
        ),
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("REINVENT4_AGENT_API_KEY"),
        help=(
            "API key for the LLM endpoint "
            "(default: REINVENT4_AGENT_API_KEY env var, or the SDK-specific "
            "env var ANTHROPIC_API_KEY / OPENAI_API_KEY)"
        ),
    )
    parser.add_argument(
        "--gateway-key",
        default=os.environ.get("LLM_GATEWAY_KEY"),
        help=(
            "Gateway subscription key (Ocp-Apim-Subscription-Key header) "
            "(default: LLM_GATEWAY_KEY env var)"
        ),
    )
    parser.add_argument(
        "--api-version",
        default=os.environ.get("REINVENT4_AGENT_API_VERSION"),
        help=(
            "Azure api-version for the OpenAI backend (e.g. "
            "2024-12-01-preview).  With a gateway key the Azure route "
            "<base-url>/openai/deployments/<model> is used and this "
            "defaults to 2024-12-01-preview; override it if a deployment "
            "needs a newer version. "
            "(default: REINVENT4_AGENT_API_VERSION env var)"
        ),
    )
    parser.add_argument(
        "--no-verify-ssl",
        action="store_true",
        default=os.environ.get("REINVENT4_AGENT_NO_VERIFY_SSL", "").lower()
        in ("1", "true", "yes"),
        help="Disable TLS certificate verification",
    )

    # -- Thinking / reasoning ------------------------------------------
    parser.add_argument(
        "--thinking-budget",
        type=int,
        default=(
            int(b)
            if (b := os.environ.get("REINVENT4_AGENT_THINKING_BUDGET"))
            else None
        ),
        help=(
            "Token budget for extended thinking (Anthropic only). "
            "Omit to disable thinking. "
            "(default: REINVENT4_AGENT_THINKING_BUDGET env var)"
        ),
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=["low", "medium", "high"],
        default=os.environ.get("REINVENT4_AGENT_REASONING_EFFORT") or None,
        help=(
            "Reasoning effort for o-series models (OpenAI only).  When set, "
            "--max-tokens is sent as max_completion_tokens. "
            "(default: REINVENT4_AGENT_REASONING_EFFORT env var)"
        ),
    )

    # -- Generation params ---------------------------------------------
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=int(os.environ.get("REINVENT4_AGENT_MAX_TOKENS") or 4096),
        help=(
            "Maximum tokens per LLM response (default: 4096, or "
            "REINVENT4_AGENT_MAX_TOKENS env var).  On reasoning models "
            "(gpt-5, o-series) this cap also covers the hidden reasoning "
            "tokens, so it is raised to a workable floor automatically; "
            "set it explicitly if replies come back truncated."
        ),
    )

    parser.add_argument(
        "--deny-tools",
        default=os.environ.get("REINVENT4_AGENT_DENY_TOOLS") or "",
        help=(
            "Comma-separated tool names to withhold from this agent "
            "(default: REINVENT4_AGENT_DENY_TOOLS env var).  A deny list "
            "rather than an allow list, so a tool added later is available "
            "by default.  Used where the deployment's write access and the "
            "agent's authority have to agree -- the conductor can create a "
            "base TTP but not rewrite one a campaign is already measuring."
        ),
    )

    # -- MCP server ----------------------------------------------------
    parser.add_argument(
        "--server-command",
        default=os.environ.get("REINVENT4_MCP_COMMAND", "dd-mcp"),
        help=(
            "Command to launch the MCP server "
            "(default: dd-mcp, or REINVENT4_MCP_COMMAND env var)"
        ),
    )
    parser.add_argument(
        "--server-args",
        nargs="*",
        default=None,
        help="Additional arguments to pass to the MCP server command",
    )

    # -- Multi-agent (Driver + Reviewer) -------------------------------
    parser.add_argument(
        "--multi-agent",
        dest="multi_agent",
        action="store_true",
        default=os.environ.get("REINVENT4_AGENT_MULTI_AGENT", "true").lower()
        in ("1", "true", "yes"),
        help=(
            "Run an independent read-only Reviewer alongside the Driver, "
            "enabling the /review command and the background Reviewer monitor "
            "(default: on, or REINVENT4_AGENT_MULTI_AGENT env var)"
        ),
    )
    parser.add_argument(
        "--no-multi-agent",
        dest="multi_agent",
        action="store_false",
        help="Run the Driver agent only (disable the Reviewer).",
    )

    # -- Co-scientist (two-agent) --------------------------------------
    parser.add_argument(
        "--goal",
        default=None,
        help=(
            "Design goal for the two-agent co-scientist run (dd-coscientist). "
            "If omitted, you are prompted for it interactively."
        ),
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=int(os.environ.get("REINVENT4_AGENT_MAX_TURNS", "8")),
        help=(
            "Global cap on Driver+Reviewer turns for a co-scientist run "
            "(default: 8, or REINVENT4_AGENT_MAX_TURNS env var)"
        ),
    )
    parser.add_argument(
        "--token-budget",
        type=int,
        default=(
            int(b)
            if (b := os.environ.get("REINVENT4_AGENT_TOKEN_BUDGET"))
            else None
        ),
        help=(
            "Global token budget across both agents for a co-scientist run "
            "(default: unlimited, or REINVENT4_AGENT_TOKEN_BUDGET env var)"
        ),
    )

    # -- Logging -------------------------------------------------------
    parser.add_argument(
        "--log-dir",
        default=os.environ.get("REINVENT4_AGENT_LOG_DIR"),
        help=(
            "Directory for session log files.  Each session writes a "
            "timestamped .log (human-readable) and .jsonl (structured "
            "transcript).  (default: REINVENT4_AGENT_LOG_DIR env var; "
            "omit for stderr-only logging)"
        ),
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose (DEBUG) logging on stderr",
    )

    # -- Monitoring ---------------------------------------------------
    parser.add_argument(
        "--monitor-interval",
        type=float,
        default=float(
            os.environ.get("REINVENT4_AGENT_MONITOR_INTERVAL", "300")
        ),
        help=(
            "Seconds between background job-alert polls "
            "(default: 120, or REINVENT4_AGENT_MONITOR_INTERVAL env var)"
        ),
    )
    parser.add_argument(
        "--no-monitor",
        action="store_true",
        default=os.environ.get("REINVENT4_AGENT_NO_MONITOR", "").lower()
        in ("1", "true", "yes"),
        help="Disable autonomous background job monitoring",
    )

    return parser.parse_args(argv)


def denied_tools(spec: str) -> set[str]:
    """Tool names to withhold, from a comma-separated list."""

    return {name.strip() for name in spec.split(",") if name.strip()}


def permitted_tools(
    tools: list[dict[str, Any]], spec: str
) -> set[str] | None:
    """The allowed set for ``Agent``, or None when nothing is withheld.

    Computed from the live tool list minus the denied names, matching how the
    supervisor restricts its Driver: a deny list means a tool added later is
    available by default and only the exclusions have to be maintained.
    """

    denied = denied_tools(spec)
    if not denied:
        return None
    return {tool["name"] for tool in tools} - denied


async def run_agent(args: argparse.Namespace) -> None:
    """Main async entry point: connect to MCP server, run chat loop."""
    from reinvent4_agent.agent import Agent
    from reinvent4_agent.backends import create_backend
    from reinvent4_agent.display import (
        RichEventHandler,
        console,
        print_error,
        print_status,
        print_welcome,
    )
    from reinvent4_agent.coordinator import REVIEWER_READ_ONLY_TOOLS
    from reinvent4_agent.mcp_client import MCPClient
    from reinvent4_agent.system_prompt import (
        REVIEWER_SYSTEM_PROMPT,
        SYSTEM_PROMPT,
    )

    api_key = _resolve_api_key(args)

    print_welcome()

    endpoint_label = args.base_url or "default"
    if args.backend == "openai":
        thinking_label = args.reasoning_effort or "off"
    else:
        thinking_label = (
            f"{args.thinking_budget} tokens"
            if args.thinking_budget
            else "off"
        )
    print_status(
        f"Backend: {args.backend} | Model: {args.model} | "
        f"Endpoint: {endpoint_label} | Thinking: {thinking_label}"
    )
    if args.log_dir:
        print_status(f"Logs: {args.log_dir}")
    print_status(
        f"Connecting to MCP server: {args.server_command} "
        f"{' '.join(args.server_args or [])}".strip()
    )

    try:
        async with MCPClient(
            server_command=args.server_command,
            server_args=args.server_args,
        ) as mcp:
            tools = await mcp.list_tools()
            print_status(
                f"Connected. {len(tools)} tool(s) available: "
                + ", ".join(t["name"] for t in tools)
            )
            console.print()

            backend = create_backend(
                args.backend,
                model=args.model,
                base_url=args.base_url,
                api_key=api_key,
                gateway_key=args.gateway_key,
                api_version=args.api_version,
                verify_ssl=not args.no_verify_ssl,
                thinking_budget=args.thinking_budget,
                reasoning_effort=args.reasoning_effort,
                system_prompt=SYSTEM_PROMPT,
            )
            # The backend may normalise the URL (e.g. add the gateway's
            # OpenAI route), so report what will actually be called.
            resolved = getattr(backend, "base_url", None)
            if resolved and resolved != args.base_url:
                print_status(f"Resolved endpoint: {resolved}")

            allowed = permitted_tools(tools, args.deny_tools)
            if allowed is not None:
                print_status(
                    "Withheld from this agent: "
                    + ", ".join(sorted(denied_tools(args.deny_tools)))
                )

            agent = Agent(
                mcp,
                backend,
                max_tokens=args.max_tokens,
                log_dir=args.log_dir,
                event_handler=RichEventHandler(),
                allowed_tools=allowed,
            )

            # A read-only Reviewer, summonable on demand via /review. It shares
            # the same MCP client but keeps an independent conversation and is
            # scoped to read-only tools so it can never mutate the campaign.
            # Only created in multi-agent mode; otherwise the Driver runs solo.
            reviewer: Agent | None = None
            if args.multi_agent:
                reviewer = Agent(
                    mcp,
                    create_backend(
                        args.backend,
                        model=args.model,
                        base_url=args.base_url,
                        api_key=api_key,
                        gateway_key=args.gateway_key,
                        api_version=args.api_version,
                        verify_ssl=not args.no_verify_ssl,
                        thinking_budget=args.thinking_budget,
                        reasoning_effort=args.reasoning_effort,
                        system_prompt=REVIEWER_SYSTEM_PROMPT,
                    ),
                    max_tokens=args.max_tokens,
                    log_dir=args.log_dir,
                    event_handler=RichEventHandler(agent_label="reviewer"),
                    name="reviewer",
                    system_prompt=REVIEWER_SYSTEM_PROMPT,
                    allowed_tools=set(REVIEWER_READ_ONLY_TOOLS),
                )
                print_status(
                    "Multi-agent mode: Driver + read-only Reviewer "
                    "(/review enabled)."
                )
            else:
                print_status("Single-agent mode: Driver only (Reviewer off).")

            await _chat_loop(agent, reviewer, args)

    except KeyboardInterrupt:
        console.print("\n[dim]Session ended.[/dim]")
    except ConnectionError as exc:
        print_error(f"Failed to connect to MCP server: {exc}")
        sys.exit(1)
    except Exception as exc:
        print_error(f"Unexpected error: {exc}")
        console.print_exception()
        sys.exit(1)


async def _chat_loop(
    agent: Agent, reviewer: Agent | None, args: argparse.Namespace
) -> None:
    """Run the interactive read-eval-print loop with background monitoring.

    When ``reviewer`` is ``None`` (single-agent mode) the /review command and
    the background Reviewer monitor are disabled; the Driver's own job-alert
    monitor still runs.
    """
    from reinvent4_agent.coordinator import Coordinator
    from reinvent4_agent.display import (
        console,
        print_agent_message,
        print_error,
        print_status,
    )

    prompt_async = _make_prompt_reader()

    # Context the periodic Reviewer monitor references as the "goal"; updated
    # to the latest substantive user request so its assessment stays relevant.
    monitor_ctx = {
        "goal": "the user's active drug-design campaign in this session"
    }
    reviewer_monitor: asyncio.Task[None] | None = None

    if not args.no_monitor:
        await agent.start_monitoring(poll_interval=args.monitor_interval)
        if reviewer is not None:
            coordinator = Coordinator(agent, reviewer)
            reviewer_monitor = asyncio.create_task(
                _reviewer_monitor_loop(
                    coordinator,
                    monitor_ctx,
                    args.monitor_interval,
                    agent._chat_lock,
                )
            )
            print_status(
                "Reviewer monitor active: every "
                f"{args.monitor_interval:.0f}s the Reviewer will assess any "
                "running campaign and the Driver will act on its verdict."
            )

    try:
        while True:
            try:
                user_input = await prompt_async()
            except (EOFError, KeyboardInterrupt):
                console.print("\n[dim]Session ended.[/dim]")
                break

            if not user_input:
                continue

            stripped = user_input.strip()
            cmd = stripped.lower()
            if cmd in ("exit", "quit", "/exit", "/quit"):
                console.print("[dim]Goodbye.[/dim]")
                break
            if cmd in ("/reset", "/clear"):
                agent.reset_history()
                if reviewer is not None:
                    reviewer.reset_history()
                console.print("[dim]Conversation history cleared.[/dim]\n")
                continue
            if cmd in ("/help", "/?"):
                _print_help(multi_agent=reviewer is not None)
                continue
            if cmd == "/review" or cmd.startswith("/review "):
                if reviewer is None:
                    print_status(
                        "/review is unavailable in single-agent mode. "
                        "Enable multi-agent mode (REINVENT4_AGENT_MULTI_AGENT=true "
                        "or --multi-agent) to use the Reviewer."
                    )
                    continue
                # Optional job id argument; preserve original case.
                parts = stripped.split(maxsplit=1)
                job_id = parts[1].strip() if len(parts) > 1 else None
                await _run_manual_review(agent, reviewer, job_id)
                continue

            monitor_ctx["goal"] = stripped
            try:
                if agent._tool_in_flight:
                    console.print(
                        "[dim yellow]⏳ Agent is waiting for a long-running tool call "
                        "(e.g. Boltz cofolding or hotspot ensemble). "
                        "Your message will be processed as soon as it returns.[/dim yellow]"
                    )
                async with agent._chat_lock:
                    response = await agent.chat(user_input)
                print_agent_message(response)
            except ConnectionError as exc:
                print_error(str(exc))
            except Exception as exc:
                print_error(f"Agent error: {exc}")
                console.print("[dim]  Run with --verbose for the full traceback.[/dim]")
    finally:
        await agent.stop_monitoring()
        if reviewer_monitor is not None:
            reviewer_monitor.cancel()
            try:
                await reviewer_monitor
            except asyncio.CancelledError:
                pass


async def _reviewer_monitor_loop(
    coordinator: "Coordinator",
    monitor_ctx: dict[str, str],
    interval: float,
    lock: asyncio.Lock,
) -> None:
    """Background task: every ``interval`` seconds, if an optimization campaign
    is running, have the Reviewer assess it and the Driver act on the verdict.

    Runs the whole cycle under the Driver's chat lock so it never interleaves
    with the user's turns or the alert monitor on the shared MCP session.
    """
    import logging

    logger = logging.getLogger(__name__)
    events = _RichCoordinatorEvents()
    cycle = 0

    while True:
        try:
            await asyncio.sleep(interval)
            async with lock:
                job_id = await coordinator._running_optimization_job()
                if job_id is None:
                    continue
                cycle += 1
                await coordinator.run_monitor_cycle(
                    monitor_ctx.get("goal", ""),
                    job_id,
                    cycle,
                    event_sink=events,
                )
        except asyncio.CancelledError:
            break
        except Exception:
            logger.error("Reviewer monitor cycle failed", exc_info=True)
            await asyncio.sleep(30)


async def _run_manual_review(
    agent: Agent, reviewer: Agent, job_id: str | None
) -> None:
    """Summon the read-only Reviewer to critique the current candidates.

    Runs under the Driver's chat lock so it never overlaps with the Driver's
    own turns or its background monitor (both share one MCP session).
    """
    from reinvent4_agent.display import (
        console,
        print_agent_message,
        print_error,
        print_status,
    )

    print_status(
        "Invoking Reviewer"
        + (f" on job {job_id}" if job_id else " on the latest candidates")
        + "..."
    )
    prompt = _build_manual_review_prompt(job_id)
    try:
        async with agent._chat_lock:
            response = await reviewer.chat(prompt)
        print_agent_message(response, agent_name="reviewer")
    except ConnectionError as exc:
        print_error(str(exc))
    except Exception as exc:
        print_error(f"Reviewer error: {exc}")
        console.print("[dim]  Run with --verbose for the full traceback.[/dim]")


def _build_manual_review_prompt(job_id: str | None) -> str:
    """Build the prompt for an on-demand Reviewer critique."""
    lines = [
        "[MANUAL REVIEW REQUESTED] The user has asked for an independent, "
        "read-only review of the current drug-design candidates.",
        "",
    ]
    if job_id:
        lines.append(
            f"Focus on optimization job '{job_id}'. Pull the real scored "
            f"molecules with get_optimization_status(job_id='{job_id}') and "
            f"get_top_candidates(job_id='{job_id}')."
        )
    else:
        lines.append(
            "Use list_jobs to find the most recent optimization job, then "
            "get_top_candidates / read_results to pull its scored molecules. "
            "If no job or results exist yet, say so plainly."
        )
    lines += [
        "",
        "Apply the Boltz2 confidence gate and PoseBusters thresholds, "
        "red-team medchem liabilities, rank the survivors, and return your "
        "verdict as the JSON handoff block.",
    ]
    return "\n".join(lines)


def _make_prompt_reader():
    """Build an async callable that reads one user message from the terminal.

    Uses ``prompt_toolkit`` when available so that bracketed-paste mode keeps
    multi-line pasted text as a single message (a plain ``input()`` returns at
    the first newline, splitting a paste into many spurious messages).
    Falls back to ``input()`` in a thread when ``prompt_toolkit`` is missing
    or stdin is not a TTY (e.g. piped input).
    """
    use_ptk = sys.stdin.isatty()
    if use_ptk:
        try:
            from prompt_toolkit import PromptSession
            from prompt_toolkit.formatted_text import HTML
        except ImportError:
            use_ptk = False

    if use_ptk:
        session: PromptSession[str] = PromptSession()
        prompt_fragment = HTML("<ansigreen><b>You></b></ansigreen> ")

        async def _prompt() -> str:
            return await session.prompt_async(prompt_fragment)

        return _prompt

    from reinvent4_agent.display import console

    async def _prompt_fallback() -> str:
        console.print("[user]You>[/user] ", end="")
        return await asyncio.get_event_loop().run_in_executor(None, input)

    return _prompt_fallback


def _print_help(multi_agent: bool = True) -> None:
    """Print available slash commands."""
    from reinvent4_agent.display import console

    review_line = (
        "  /review [job_id]  Summon the read-only Reviewer to critique the "
        "latest (or given) candidates\n"
        if multi_agent
        else ""
    )
    reset_hint = "both agents" if multi_agent else "the agent"
    console.print(
        "\n[bold]Commands:[/bold]\n"
        "  /help, /?       Show this help\n"
        f"{review_line}"
        f"  /reset, /clear  Clear conversation history ({reset_hint})\n"
        "  /exit, /quit    Exit the agent\n"
    )


def _configure_logging(args: argparse.Namespace) -> str | None:
    """Set up logging to stderr and optionally to a session log file.

    Returns the log file path if file logging was configured, else None.
    """
    from datetime import datetime

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        "%(asctime)s [%(name)s] %(levelname)s: %(message)s"
    )

    stderr_level = logging.DEBUG if args.verbose else logging.WARNING
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(stderr_level)
    stderr_handler.setFormatter(fmt)
    root_logger.addHandler(stderr_handler)

    log_file_path = None
    if args.log_dir:
        from pathlib import Path

        log_dir = Path(args.log_dir).expanduser()
        log_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file_path = str(log_dir / f"session_{timestamp}.log")

        file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(fmt)
        root_logger.addHandler(file_handler)

    return log_file_path


def _resolve_api_key(args: argparse.Namespace) -> str | None:
    """Resolve the LLM API key from args/env, exiting on failure."""
    from reinvent4_agent.display import print_error

    api_key = args.api_key
    if not api_key and not args.gateway_key:
        fallback_var = (
            "ANTHROPIC_API_KEY"
            if args.backend == "anthropic"
            else "OPENAI_API_KEY"
        )
        api_key = os.environ.get(fallback_var)
        if not api_key:
            print_error(
                f"No API key found. Set {fallback_var}, "
                "REINVENT4_AGENT_API_KEY, --api-key, or --gateway-key."
            )
            sys.exit(1)
    if args.gateway_key and not api_key:
        api_key = "dummy"
    return api_key


async def run_coscientist(args: argparse.Namespace) -> None:
    """Run a two-agent Driver <-> Reviewer co-scientist session."""
    from reinvent4_agent.agent import Agent
    from reinvent4_agent.backends import create_backend
    from reinvent4_agent.coordinator import (
        REVIEWER_READ_ONLY_TOOLS,
        Coordinator,
    )
    from reinvent4_agent.display import (
        RichEventHandler,
        console,
        print_error,
        print_status,
        print_welcome,
    )
    from reinvent4_agent.mcp_client import MCPClient
    from reinvent4_agent.system_prompt import (
        DRIVER_SYSTEM_PROMPT,
        REVIEWER_SYSTEM_PROMPT,
    )

    api_key = _resolve_api_key(args)

    goal = args.goal
    if not goal:
        prompt_async = _make_prompt_reader()
        console.print(
            "[bold]Co-scientist mode[/bold] — describe the design goal "
            "for the Driver + Reviewer team:"
        )
        try:
            goal = await prompt_async()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Session ended.[/dim]")
            return
    if not goal or not goal.strip():
        print_error("No goal provided; nothing to do.")
        return

    print_welcome()
    print_status(
        f"Backend: {args.backend} | Model: {args.model} | "
        f"Co-scientist: Driver + Reviewer | "
        f"Budget: {args.max_turns} turns"
        + (f", {args.token_budget:,} tokens" if args.token_budget else "")
    )
    print_status(
        f"Connecting to MCP server: {args.server_command} "
        f"{' '.join(args.server_args or [])}".strip()
    )

    def _make_backend(system_prompt: str):
        return create_backend(
            args.backend,
            model=args.model,
            base_url=args.base_url,
            api_key=api_key,
            gateway_key=args.gateway_key,
            api_version=args.api_version,
            verify_ssl=not args.no_verify_ssl,
            thinking_budget=args.thinking_budget,
            reasoning_effort=args.reasoning_effort,
            system_prompt=system_prompt,
        )

    try:
        async with MCPClient(
            server_command=args.server_command,
            server_args=args.server_args,
        ) as mcp:
            tools = await mcp.list_tools()
            reviewer_tools = sorted(
                t["name"]
                for t in tools
                if t["name"] in REVIEWER_READ_ONLY_TOOLS
            )
            print_status(
                f"Connected. Driver has {len(tools)} tool(s); "
                f"Reviewer restricted to {len(reviewer_tools)} read-only "
                f"tool(s): {', '.join(reviewer_tools)}"
            )
            console.print()

            driver = Agent(
                mcp,
                _make_backend(DRIVER_SYSTEM_PROMPT),
                max_tokens=args.max_tokens,
                log_dir=args.log_dir,
                event_handler=RichEventHandler(agent_label="driver"),
                name="driver",
                system_prompt=DRIVER_SYSTEM_PROMPT,
                allowed_tools=permitted_tools(tools, args.deny_tools),
            )
            reviewer = Agent(
                mcp,
                _make_backend(REVIEWER_SYSTEM_PROMPT),
                max_tokens=args.max_tokens,
                log_dir=args.log_dir,
                event_handler=RichEventHandler(agent_label="reviewer"),
                name="reviewer",
                system_prompt=REVIEWER_SYSTEM_PROMPT,
                allowed_tools=set(REVIEWER_READ_ONLY_TOOLS),
            )

            coordinator = Coordinator(
                driver,
                reviewer,
                max_turns=args.max_turns,
                token_budget=args.token_budget,
            )

            events = _RichCoordinatorEvents()
            result = await coordinator.run(goal.strip(), event_sink=events)
            _print_coscientist_summary(result)

            if not args.no_monitor:
                job_id = (
                    result.last_driver.job_id if result.last_driver else None
                )
                print_status(
                    "Entering campaign monitoring: the Reviewer will assess "
                    f"the run every {args.monitor_interval:.0f}s and the "
                    "Driver will act on its verdict. Press Ctrl-C to stop."
                )
                monitor_result = await coordinator.monitor_campaign(
                    goal.strip(),
                    job_id=job_id,
                    interval=args.monitor_interval,
                    event_sink=events,
                )
                _print_monitor_summary(monitor_result)

    except KeyboardInterrupt:
        console.print("\n[dim]Session ended.[/dim]")
    except ConnectionError as exc:
        print_error(f"Failed to connect to MCP server: {exc}")
        sys.exit(1)
    except Exception as exc:
        print_error(f"Unexpected error: {exc}")
        console.print_exception()
        sys.exit(1)


class _RichCoordinatorEvents:
    """Render coordinator events (turn banners, messages, budget) via Rich."""

    def on_turn_start(self, agent_name: str, turn: int) -> None:
        from reinvent4_agent.display import print_agent_turn_banner

        print_agent_turn_banner(agent_name, turn)

    def on_agent_message(self, agent_name: str, text: str) -> None:
        from reinvent4_agent.display import print_agent_message

        print_agent_message(text, agent_name=agent_name)

    def on_budget(
        self, turn: int, max_turns: int, total_tokens: int, budget: int | None
    ) -> None:
        from reinvent4_agent.display import print_budget_status

        print_budget_status(turn, max_turns, total_tokens, budget)

    def on_monitor_cycle_start(self, cycle: int, job_id: str) -> None:
        from reinvent4_agent.display import print_monitor_cycle_banner

        print_monitor_cycle_banner(cycle, job_id)

    def on_monitor_end(self, job_id: str, state: str) -> None:
        from reinvent4_agent.display import print_monitor_end

        print_monitor_end(job_id, state)


def _print_coscientist_summary(result: Any) -> None:
    """Print the final verdict banner for a co-scientist run."""
    from reinvent4_agent.display import console

    status_styles = {
        "approved": "score_good",
        "rejected": "error",
        "budget_exhausted": "score_ok",
        "incomplete": "dim",
    }
    style = status_styles.get(result.status, "dim")
    console.print()
    console.print(
        f"[{style}]Co-scientist finished: {result.status}[/{style}] "
        f"[dim]({result.turns} turns, {result.total_tokens:,} tokens)[/dim]"
    )


def _print_monitor_summary(result: Any) -> None:
    """Print the final banner for a campaign-monitoring session."""
    from reinvent4_agent.display import console

    if result.status == "no_active_campaign":
        console.print(
            "[dim]No running optimization campaign to monitor.[/dim]"
        )
        return
    console.print()
    console.print(
        f"[coordinator]Monitoring finished: {result.status}[/coordinator] "
        f"[dim](job {result.job_id}, {result.cycles} cycle(s), "
        f"{result.total_tokens:,} tokens)[/dim]"
    )


def main() -> None:
    """CLI entry point (single-agent interactive chat)."""
    args = parse_args()

    log_file = _configure_logging(args)
    if log_file:
        logging.getLogger(__name__).info("Session log: %s", log_file)

    asyncio.run(run_agent(args))


def coscientist_main() -> None:
    """CLI entry point for the two-agent co-scientist runner."""
    args = parse_args()

    log_file = _configure_logging(args)
    if log_file:
        logging.getLogger(__name__).info("Session log: %s", log_file)

    asyncio.run(run_coscientist(args))


if __name__ == "__main__":
    main()
