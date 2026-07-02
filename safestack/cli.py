"""SafeStack command-line interface (Typer).

The config-driven `run` command is wired in together with the runner module (next
increment of Phase 0). This stub keeps the `safestack` console entry point importable.
"""

from __future__ import annotations

import typer

app = typer.Typer(
    help="SafeStack: defense-in-depth LLM safety evaluation.",
    no_args_is_help=True,
)
