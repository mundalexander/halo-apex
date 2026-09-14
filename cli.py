#!/usr/bin/env python3
"""Halo Apex CLI — production entry point.

Usage (inside the project venv):

    python cli.py health
    python cli.py index <pfad>
    python cli.py task "<auftrag>" --file <datei>

Safety: every task runs on its own git branch (halo/<task-slug>), the
current branch is never patched directly. Review the result with
"git diff <branch>...halo/<slug>" and merge if it looks good.
"""

import json
from pathlib import Path

import click

from runner import TaskRunner


@click.group()
@click.version_option(version="1.1.0", prog_name="halo-apex")
def cli():
    """Halo Apex — local AI dev engine (Blade + Vector, Stream optional)."""


@cli.command()
def health():
    """Healthcheck all active Halo subsystems."""
    runner = TaskRunner()
    report = runner.health()
    click.echo(json.dumps(report, indent=2, default=str))
    if not report.get("healthy"):
        raise SystemExit(1)


@cli.command()
@click.argument("path", type=click.Path(exists=True))
def index(path):
    """Index a .py file or a whole directory tree into the vector store."""
    runner = TaskRunner()
    count = runner.index_path(Path(path))
    click.echo(f"Indexed {count} file(s) into {runner.persist_dir}")


@cli.command()
@click.argument("task")
@click.option("--file", "file_path", required=True, type=click.Path(exists=True))
def task(task, file_path):
    """Run TASK on FILE with the full safety ring (branch, gate, rollback)."""
    runner = TaskRunner()
    try:
        result = runner.run_task(task, Path(file_path))
    except RuntimeError as exc:
        click.echo(f"[FAILED] {exc}")
        raise SystemExit(1)
    click.echo("[SUCCESS]")
    click.echo(f"  branch : {result['branch']}")
    click.echo(f"  commit : {result['commit']}")
    click.echo(f"  file   : {result['file']}")
    click.echo(f"  {result['note']}")
    click.echo(result["diffstat"])


if __name__ == "__main__":
    cli()