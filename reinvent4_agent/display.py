"""Rich terminal formatting for the Drug Discovery Agent.

Provides formatted output for score tables, molecule results, progress
indicators, and conversation display using the Rich library.
"""

from __future__ import annotations

import json
from typing import Any

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

_THEME = Theme(
    {
        "agent": "bold cyan",
        "user": "bold green",
        "tool_name": "bold yellow",
        "tool_result": "dim",
        "error": "bold red",
        "score_good": "green",
        "score_ok": "yellow",
        "score_bad": "red",
        "driver": "bold cyan",
        "reviewer": "bold magenta",
        "coordinator": "bold blue",
    }
)

console = Console(theme=_THEME)

# Per-agent display styling for the co-scientist run. Keyed by agent name.
_AGENT_STYLES: dict[str, str] = {
    "driver": "driver",
    "reviewer": "reviewer",
    "coordinator": "coordinator",
}


def _agent_style(name: str) -> str:
    """Return the Rich style registered for an agent name (default cyan)."""
    return _AGENT_STYLES.get(name.lower(), "agent")


def print_welcome() -> None:
    """Print the welcome banner."""
    console.print(
        Panel(
            "[bold cyan]Drug Discovery Agent[/bold cyan]\n"
            "Interactive multi-property molecular optimization\n\n"
            "[dim]Type your request to begin. The agent can score molecules,\n"
            "discover available components, and run optimization campaigns.\n"
            'Type "exit" or "quit" to end the session.[/dim]',
            border_style="cyan",
            padding=(1, 2),
        )
    )
    console.print()


def print_user_prompt() -> None:
    """Print the user input prompt indicator."""
    console.print("[user]You>[/user] ", end="")


def print_agent_message(text: str, agent_name: str | None = None) -> None:
    """Render agent response as markdown, optionally attributed to an agent."""
    console.print()
    if agent_name:
        style = _agent_style(agent_name)
        console.print(f"[{style}]{agent_name.capitalize()}:[/{style}]")
    md = Markdown(text)
    console.print(md)
    console.print()


def print_agent_turn_banner(agent_name: str, turn: int, detail: str = "") -> None:
    """Print a labeled banner announcing which agent is now speaking."""
    style = _agent_style(agent_name)
    title = f"{agent_name.upper()} · turn {turn}"
    if detail:
        title += f" · {detail}"
    console.print()
    console.print(
        Panel(
            title,
            border_style=style,
            padding=(0, 1),
        )
    )


def print_budget_status(
    turn: int,
    max_turns: int,
    total_tokens: int,
    max_tokens: int | None,
) -> None:
    """Show the running turn/token budget for the co-scientist loop."""
    parts = [f"turns {turn}/{max_turns}"]
    if max_tokens is not None:
        parts.append(f"tokens {total_tokens:,}/{max_tokens:,}")
    else:
        parts.append(f"tokens {total_tokens:,}")
    console.print(f"[coordinator][budget] {' | '.join(parts)}[/coordinator]")


def print_monitor_cycle_banner(cycle: int, job_id: str) -> None:
    """Announce a periodic campaign-monitoring cycle."""
    console.print()
    console.print(
        Panel(
            f"CAMPAIGN MONITOR · cycle {cycle} · job {job_id}",
            border_style="coordinator",
            padding=(0, 1),
        )
    )


def print_monitor_end(job_id: str, state: str) -> None:
    """Report that campaign monitoring has stopped."""
    console.print(
        f"[coordinator]Campaign monitoring ended for job {job_id} "
        f"(state: {state}).[/coordinator]"
    )


def print_tool_call(tool_name: str, arguments: dict[str, Any]) -> None:
    """Display a tool call being made."""
    args_summary = _summarize_args(arguments)
    console.print(
        f"  [tool_name]> {tool_name}[/tool_name]"
        f"({args_summary})",
        highlight=False,
    )


def print_tool_result_brief(tool_name: str, result: str) -> None:
    """Display a brief summary of tool results."""
    try:
        data = json.loads(result)
        summary = _summarize_tool_result(tool_name, data)
    except (json.JSONDecodeError, TypeError):
        text = result[:200]
        if len(result) > 200:
            text += "..."
        summary = text

    console.print(f"  [tool_result]{summary}[/tool_result]")


def print_thinking(text: str) -> None:
    """Display LLM thinking/reasoning content in a collapsible panel."""
    truncated = text
    max_display = 500
    if len(text) > max_display:
        truncated = text[:max_display] + f"\n... ({len(text) - max_display} more chars)"
    console.print(
        Panel(
            truncated,
            title="[dim]thinking[/dim]",
            border_style="dim",
            padding=(0, 1),
        )
    )


def print_error(message: str) -> None:
    """Display an error message."""
    console.print(f"[error]Error: {message}[/error]")


def print_status(message: str) -> None:
    """Display a status/info message."""
    console.print(f"[dim]{message}[/dim]")


def print_monitor_alert(alert: dict[str, Any]) -> None:
    """Render a monitor alert as a Rich panel."""
    severity = alert.get("severity", "medium")
    style_map = {
        "critical": "bold red",
        "high": "red",
        "medium": "yellow",
        "low": "dim",
    }
    style = style_map.get(severity, "yellow")
    console.print(Panel(
        alert.get("message", "Unknown alert"),
        title=f"[{style}]Monitor Alert: {alert.get('type', '?')}[/{style}]",
        subtitle=f"Job {alert.get('job_id', '?')}",
        border_style=style,
    ))


def render_score_table(result: dict[str, Any]) -> None:
    """Render a molecule scoring result as a rich table."""
    molecules = result.get("molecules", [])
    if not molecules:
        console.print("[dim]No molecules to display.[/dim]")
        return

    all_components = set()
    for mol in molecules:
        all_components.update(mol.get("components", {}).keys())
    component_names = sorted(all_components)

    # Many components as columns overflow the terminal (every cell shrinks to
    # "..."). Transpose: components as ROWS, molecules as compact M1..Mn columns,
    # with a SMILES legend underneath. Keep the wide-by-molecule layout for the
    # small case where it reads fine.
    if len(component_names) > 6:
        _render_score_table_transposed(molecules, component_names, result)
        return

    table = Table(
        title="Molecule Scores",
        show_lines=True,
        title_style="bold",
    )
    table.add_column("SMILES", style="bold", max_width=40, no_wrap=True)
    table.add_column("Total", justify="right", style="bold")
    for name in component_names:
        table.add_column(name, justify="right", max_width=12)

    for mol in molecules:
        total = mol.get("total_score")
        total_str = _score_text(total) if total is not None else "-"

        row = [
            _truncate(mol.get("smiles", "?"), 40),
            total_str,
        ]
        for name in component_names:
            comp = mol.get("components", {}).get(name, {})
            transformed = comp.get("transformed")
            raw = comp.get("raw")
            if transformed is not None and raw is not None:
                cell = f"{_format_score(transformed)} ({_format_raw(raw)})"
            elif transformed is not None:
                cell = _format_score(transformed)
            else:
                cell = "-"
            row.append(cell)

        table.add_row(*row)

    summary = result.get("summary", {})
    if summary:
        mean = summary.get("mean_total_score")
        best = summary.get("best_total_score")
        n_valid = summary.get("n_valid", 0)
        n_total = summary.get("n_molecules", 0)
        caption = f"{n_valid}/{n_total} valid"
        if mean is not None:
            caption += f" | mean: {mean:.3f}"
        if best is not None:
            caption += f" | best: {best:.3f}"
        table.caption = caption

    console.print(table)
    console.print()


def _render_score_table_transposed(
    molecules: list[dict[str, Any]],
    component_names: list[str],
    result: dict[str, Any],
) -> None:
    """Readable layout for many-component TTPs: components as rows, molecules as columns."""
    table = Table(title="Molecule Scores", title_style="bold")
    table.add_column("Component", style="bold", no_wrap=True)
    for i, _ in enumerate(molecules, 1):
        table.add_column(f"M{i}", justify="right")

    total_row: list[str] = ["[bold]Total[/bold]"]
    for mol in molecules:
        t = mol.get("total_score")
        total_row.append(_format_score(t) if t is not None else "-")
    table.add_row(*total_row)

    for name in component_names:
        row = [name]
        for mol in molecules:
            comp = mol.get("components", {}).get(name, {})
            transformed = comp.get("transformed")
            raw = comp.get("raw")
            if transformed is not None and raw is not None:
                row.append(f"{_format_score(transformed)} ({_format_raw(raw)})")
            elif transformed is not None:
                row.append(_format_score(transformed))
            else:
                row.append("-")
        table.add_row(*row)

    console.print(table)

    # SMILES legend (columns are Mi to keep the table narrow)
    for i, mol in enumerate(molecules, 1):
        console.print(f"[dim]M{i}[/dim]  {mol.get('smiles', '?')}")

    summary = result.get("summary", {})
    if summary:
        mean = summary.get("mean_total_score")
        best = summary.get("best_total_score")
        n_valid = summary.get("n_valid", 0)
        n_total = summary.get("n_molecules", 0)
        caption = f"{n_valid}/{n_total} valid"
        if mean is not None:
            caption += f" | mean: {mean:.3f}"
        if best is not None:
            caption += f" | best: {best:.3f}"
        console.print(f"[dim]{caption}[/dim]")
    console.print()


# ---------------------------------------------------------------------------
# Event handler
# ---------------------------------------------------------------------------


class RichEventHandler:
    """AgentEventHandler that renders events using Rich terminal output.

    ``agent_label`` (e.g. ``"driver"`` / ``"reviewer"``) prefixes tool calls
    and thinking so a co-scientist run makes clear which agent is acting.
    """

    def __init__(self, agent_label: str | None = None) -> None:
        self._label = agent_label

    def _prefix(self) -> str:
        if not self._label:
            return ""
        style = _agent_style(self._label)
        return f"[{style}]{self._label}[/{style}] "

    def on_thinking(self, text: str) -> None:
        print_thinking(text)

    def on_tool_call(self, name: str, arguments: dict[str, Any]) -> None:
        if self._label:
            args_summary = _summarize_args(arguments)
            style = _agent_style(self._label)
            console.print(
                f"  [{style}]{self._label}[/{style}] "
                f"[tool_name]> {name}[/tool_name]({args_summary})",
                highlight=False,
            )
        else:
            print_tool_call(name, arguments)

    def on_tool_result(self, name: str, result: str, is_error: bool) -> None:
        if is_error:
            print_error(result)
        else:
            print_tool_result_brief(name, result)
            self._maybe_render_score_table(name, result)

    def on_error(self, message: str) -> None:
        print_error(message)

    def on_monitor_alert(self, alert: dict[str, Any]) -> None:
        print_monitor_alert(alert)

    def on_monitor_response(self, text: str) -> None:
        print_agent_message(text)

    @staticmethod
    def _maybe_render_score_table(tool_name: str, result_text: str) -> None:
        """Render a score table if the tool result contains molecule scores."""
        if tool_name != "score_molecules":
            return
        try:
            data = json.loads(result_text)
            if isinstance(data, dict) and "molecules" in data:
                render_score_table(data)
        except (json.JSONDecodeError, TypeError):
            pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _score_text(score: float | None) -> Text | str:
    """Return a colored Rich Text for display in tables."""
    if score is None:
        return "-"
    text = Text(f"{score:.3f}")
    if score >= 0.7:
        text.stylize("score_good")
    elif score >= 0.4:
        text.stylize("score_ok")
    else:
        text.stylize("score_bad")
    return text


def _format_score(val: float | None) -> str:
    if val is None:
        return "-"
    return f"{val:.3f}"


def _format_raw(val: float | int | None) -> str:
    if val is None:
        return "-"
    if isinstance(val, float):
        if abs(val) >= 100:
            return f"{val:.1f}"
        return f"{val:.3f}"
    return str(val)


def _truncate(s: str, max_len: int) -> str:
    if len(s) <= max_len:
        return s
    return s[: max_len - 3] + "..."


def _summarize_args(args: dict[str, Any]) -> str:
    """Create a short summary of tool call arguments."""
    parts = []
    for key, val in args.items():
        if isinstance(val, list):
            if val and all(isinstance(item, dict) for item in val):
                keys = sorted({k for item in val for k in item})
                parts.append(f"{key}=[{len(val)} items keys={keys}]")
            else:
                parts.append(f"{key}=[{len(val)} items]")
        elif isinstance(val, str) and len(val) > 60:
            parts.append(f"{key}='{val[:57]}...'")
        else:
            parts.append(f"{key}={val!r}")
    summary = ", ".join(parts)
    if len(summary) > 120:
        return summary[:117] + "..."
    return summary


def _summarize_tool_result(tool_name: str, data: Any) -> str:
    """Create a brief textual summary of tool result data."""
    if isinstance(data, dict):
        # Truthiness, not key presence: a task row carries an ``error`` column
        # that is null while nothing is wrong, so testing presence reported a
        # successful enqueue as "Error: None". A conductor told its enqueue had
        # failed went on to open three more campaigns.
        if data.get("error"):
            return f"Error: {data['error']}"
        if "molecules" in data:
            mols = data["molecules"]
            summary = data.get("summary", {})
            n = len(mols)
            mean = summary.get("mean_total_score")
            suffix = f", mean={mean:.3f}" if mean is not None else ""
            return f"{n} molecule(s) scored{suffix}"
        if isinstance(data, list):
            return f"{len(data)} items returned"
    if isinstance(data, list):
        return f"{len(data)} items returned"
    return f"Result received ({type(data).__name__})"
