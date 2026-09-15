#!/usr/bin/env python3
"""Halo Apex Benchmark — misst den Performance-Zugewinn der Pipeline.

Erstellt ein frisches, aufopferbares Git-Repo mit drei repräsentativen
Python-Dateien und führt darüber eine Standard-Task-Suite mit dem
produktiven TaskRunner (inkl. Safety-Ring) aus. Gemessen wird:

- Indexierungs- und Komponenten-Latenzen (Embedding, Retrieval)
- End-to-End-Zeit pro Task (Index, LLM, Apply, Gate, Commit)
- Versuche bis Erfolg (First-Try-Rate)
- Kontext-Effizienz: Hybrid-Prompt vs. naiver Volldump des Repos

Der Vektor-Index der Benchmark läuft isoliert in einem temporären
Verzeichnis — der produktive Index unter ~/.local/share/halo_apex
bleibt unangetastet.

Ausführen:  ./venv/bin/python bench.py
"""

from __future__ import annotations

import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# --- Isolierten Benchmark-Index setzen, BEVOR die Config geladen wird ----
_BENCH_ROOT = Path(tempfile.mkdtemp(prefix="halo_bench_"))
os.environ["HALO_VECTOR_PERSIST_DIR"] = str(_BENCH_ROOT / "chroma")

from config import halo_settings  # noqa: E402
from halo_blade import OpenClawOrchestrator  # noqa: E402
from runner import TaskRunner  # noqa: E402

BENCH_FILES = {
    "bench_shapes.py": '''"""Geometry helpers for the Halo Apex benchmark suite."""


import math


def circle_area(radius):
    if radius < 0:
        raise ValueError("radius must be non-negative")
    return math.pi * radius ** 2


def rectangle_perimeter(width, height):
    return 2 * (width + height)


def triangle_hypotenuse(a, b):
    return math.sqrt(a * a + b * b)


def distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
''',
    "bench_text.py": '''"""Text utilities for the Halo Apex benchmark suite."""


def word_count(text):
    return len(text.split())


def capitalize_words(text):
    return " ".join(w.capitalize() for w in text.split())


def truncate(text, max_length):
    if len(text) <= max_length:
        return text
    return text[: max_length - 3] + "..."


def slug(text):
    lowered = text.lower().replace(" ", "-")
    return "".join(c for c in lowered if c.isalnum() or c == "-")
''',
    "bench_stats.py": '''"""Small statistics helpers for the Halo Apex benchmark suite."""


def mean(values):
    if not values:
        raise ValueError("values must not be empty")
    return sum(values) / len(values)


def variance(values):
    avg = mean(values)
    return sum((v - avg) ** 2 for v in values) / len(values)


def stddev(values):
    return variance(values) ** 0.5


def median(values):
    ordered = sorted(values)
    n = len(ordered)
    if n == 0:
        raise ValueError("values must not be empty")
    mid = n // 2
    if n % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2
''',
}

TASKS = [
    (
        "Add type hints to all function arguments and return types.",
        "bench_shapes.py",
    ),
    ("Add docstrings to all functions.", "bench_text.py"),
    (
        "Add input validation raising ValueError for invalid inputs to all functions.",
        "bench_stats.py",
    ),
]


def setup_bench_repo() -> Path:
    """Create a fresh sacrificial git repo with the benchmark files."""
    repo = _BENCH_ROOT / "repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, capture_output=True, check=True)
    for name, content in BENCH_FILES.items():
        (repo / name).write_text(content, encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=repo, capture_output=True, check=True)
    subprocess.run(
        ["git", "commit", "-m", "bench: initial files"],
        cwd=repo,
        capture_output=True,
        check=True,
    )
    return repo


def _timed(fn, *args, **kwargs):
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    return result, time.perf_counter() - t0


def main() -> int:
    print("Halo Apex Benchmark — Performance-Zugewinn-Messung", flush=True)
    print(f"  bench root : {_BENCH_ROOT}", flush=True)
    print(
        f"  llm        : {halo_settings.halo_blade_llm_model or 'auto'} "
        f"@ {halo_settings.halo_blade_llm_url}",
        flush=True,
    )

    repo = setup_bench_repo()
    runner = TaskRunner()

    # --- [1] Indexierung ----------------------------------------------
    _, t_index = _timed(runner.index_path, repo)
    print(f"\n[1] Indexierung ({len(BENCH_FILES)} Dateien): {t_index:.1f}s", flush=True)

    # --- [2] Embedding-Latenz -----------------------------------------
    lats = []
    for i in range(3):
        _, t = _timed(runner.embedding_service.embed_text, f"benchmark ping {i}")
        lats.append(t)
    emb_avg = statistics.mean(lats)
    print(f"[2] Embedding-Latenz (Ø aus 3): {emb_avg * 1000:.0f} ms", flush=True)

    # --- [3] Retrieval-Latenz -----------------------------------------
    lats = []
    for q in ("geometry calculations", "text processing", "statistics helpers"):
        _, t = _timed(runner.vector_engine.query, q, 3)
        lats.append(t)
    ret_avg = statistics.mean(lats)
    print(f"[3] Retrieval-Latenz (Ø aus 3): {ret_avg * 1000:.0f} ms", flush=True)

    # --- [4] Kontext-Effizienz ----------------------------------------
    orchestrator = OpenClawOrchestrator(
        llm_base_url=halo_settings.halo_blade_llm_url,
        llm_model=halo_settings.halo_blade_llm_model,
        vector_engine=runner.vector_engine,
    )
    task_text, target_name = TASKS[0]
    target = repo / target_name
    halo_ctx = orchestrator.build_context(
        task_text, str(target), target_dir=str(repo)
    )
    per_file = []
    for name in BENCH_FILES:
        lines = (repo / name).read_text(encoding="utf-8").splitlines()
        numbered = "\n".join(f"{i} | {line}" for i, line in enumerate(lines, 1))
        per_file.append(len(numbered))
    naive_total = sum(per_file)
    naive_50 = statistics.mean(per_file) * 50
    print(
        f"[4] Hybrid-Prompt (real, konstant): {len(halo_ctx):,} Zeichen — "
        f"unabhängig von der Repo-Größe (Top-3-Retrieval-Cap)",
        flush=True,
    )
    print(
        f"    Naiver Volldump: {naive_total:,} Zeichen bei {len(BENCH_FILES)} "
        f"Dateien, ~{naive_50:,.0f} bei 50 Dateien (wächst linear)",
        flush=True,
    )

    # --- [5] Task-Suite über den produktiven Pfad ----------------------
    print("\n[5] Task-Suite (produktiver TaskRunner mit Safety-Ring):", flush=True)
    results = []
    for task_text, target_name in TASKS:
        print(f"\n  >>> {task_text}", flush=True)
        print(f"      target: {target_name}", flush=True)
        try:
            result = runner.run_task(task_text, repo / target_name)
            results.append(
                {
                    "task": task_text,
                    "file": target_name,
                    "ok": True,
                    "attempts": result["attempts"],
                    "duration": result["duration_seconds"],
                }
            )
            print(
                f"      OK — Versuch {result['attempts']}, "
                f"{result['duration_seconds']}s end-to-end",
                flush=True,
            )
        except RuntimeError as exc:
            results.append(
                {
                    "task": task_text,
                    "file": target_name,
                    "ok": False,
                    "error": str(exc)[:300],
                }
            )
            print(f"      FEHLGESCHLAGEN: {str(exc)[:200]}", flush=True)

    # --- [6] Zusammenfassung -------------------------------------------
    ok = [r for r in results if r["ok"]]
    print("\n" + "=" * 60, flush=True)
    print("HALO APEX BENCHMARK REPORT", flush=True)
    print("=" * 60, flush=True)
    if ok:
        durations = [r["duration"] for r in ok]
        attempts = [r["attempts"] for r in ok]
        print(f"Tasks erfolgreich     : {len(ok)}/{len(results)}")
        print(
            f"First-Try-Erfolge     : "
            f"{sum(1 for a in attempts if a == 1)}/{len(ok)}"
        )
        print(f"Ø End-to-End pro Task : {statistics.mean(durations):.1f}s")
        print(f"Ø Versuche pro Task   : {statistics.mean(attempts):.1f}")
    print(f"Indexierung           : {t_index:.1f}s ({len(BENCH_FILES)} Dateien)")
    print(f"Embedding-Latenz      : {emb_avg * 1000:.0f} ms")
    print(f"Retrieval-Latenz      : {ret_avg * 1000:.0f} ms")
    print(f"Bench-Repo           : {_BENCH_ROOT} (bleibt für Inspektion liegen)")

    return 0 if len(ok) == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())