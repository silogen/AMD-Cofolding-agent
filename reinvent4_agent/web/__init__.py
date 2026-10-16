# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Web gateway that puts the drug design agent behind HTTP and WebSocket.

The agent itself is transport-agnostic: it reports what it does through the
``AgentEventHandler`` protocol (see ``reinvent4_agent.agent``).  The terminal
client supplies a Rich handler; this package supplies a handler that forwards
the same events to a browser over a WebSocket.  No agent logic is duplicated.

Importing this package must not pull in FastAPI: ``artifacts`` is pure Python
and is imported by the tests and by the event handler, which both run in
contexts where the web dependencies may be absent.  Import ``create_app`` from
``reinvent4_agent.web.server`` directly.
"""

__all__: list[str] = []
