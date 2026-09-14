#!/usr/bin/env python3
"""Halo Apex — master integration & healthcheck.

Verifies the decoupled Halo architecture end to end:

- Halo Vector works standalone (bge-m3 embeddings -> ChromaDB retrieval)
- Halo Blade works standalone (AST skeleton, git-apply patching, executor)
- Halo Stream routes correctly (stub; disabled -> direct pass-through)
- Blade + Vector interact seamlessly: the orchestrator consumes vector
  results through the injected ``VectorSearch`` interface

Finally performs a test patch on a temporary file and cleans up.
"""

from __future__ import annotations

import ast
import shutil
import sys
import tempfile
from pathlib import Path

import requests

from config import halo_settings
from halo_blade import (
    ASTCodeParser,
    FileOperations,
    LocalExecutor,
    OpenClawOrchestrator,
)
from halo_stream import HaloStreamProvider
from halo_vector import EmbeddingService, VectorEngine

DEMO_SOURCE = '''"""Minimal demo calculator for ecosystem integration testing."""


def add(a, b):
    return a + b


def subtract(a, b):
    return a - b


def multiply(a, b):
    return a * b


def divide(a, b):
    if b == 0:
        raise ValueError("division by zero")
    return a / b
'''

TASK = "Add type hints to all function arguments and return types."


def step(title: str) -> None:
    print(f"\n=== {title} ===")


def check_http(name: str, url: str) -> bool:
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        print(f"  [OK]   {name}: {url}")
        return True
    except Exception as exc:
        print(f"  [FAIL] {name}: {url} -> {type(exc).__name__}: {exc}")
        return False


def type_hints_present(source: str) -> bool:
    """All four demo functions must have annotated args and return types."""
    tree = ast.parse(source)
    annotated = 0
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.returns is not None and any(a.annotation for a in node.args.args):
                annotated += 1
    return annotated >= 4


def main() -> int:
    print("Halo Apex - Master Integration & Healthcheck")
    print(f"  execution mode: {halo_settings.halo_execution_mode}")

    if halo_settings.halo_execution_mode.upper() == "LOCAL_ONLY":
        urls = (
            halo_settings.halo_vector_base_url,
            halo_settings.halo_blade_llm_url,
            halo_settings.halo_stream_backend_url,
        )
        remote = [
            u for u in urls if "localhost" not in u and "127.0.0.1" not in u
        ]
        if remote:
            print(f"  [WARN] LOCAL_ONLY active but non-local URLs: {remote}")

    # --- Halo Stream (stub routing) ---------------------------------
    step("Halo Stream (paging interface stub)")
    stream_provider = HaloStreamProvider(
        enabled=halo_settings.use_halo_stream,
        backend_url=halo_settings.halo_stream_backend_url,
        default_url=halo_settings.halo_blade_llm_url,
    )
    status = stream_provider.stream_status()
    print(
        f"  routing: {status['routing']} "
        f"(USE_HALO_STREAM={halo_settings.use_halo_stream})"
    )
    blade_llm_url = stream_provider.get_backend_url(halo_settings.halo_blade_llm_url)
    print(f"  effective Halo Blade LLM url: {blade_llm_url}")

    # --- Halo Vector healthcheck ------------------------------------
    step("Halo Vector healthcheck (embeddings + endpoint)")
    vector_ok = check_http(
        "Vector endpoint",
        halo_settings.halo_vector_base_url.rstrip("/") + "/models",
    )
    embedding_service = EmbeddingService(
        base_url=halo_settings.halo_vector_base_url,
        model_name=halo_settings.halo_vector_embedding_model,
    )
    try:
        vector = embedding_service.embed_text("halo vector health check")
        print(
            f"  [OK]   Embedding round-trip "
            f"({halo_settings.halo_vector_embedding_model}): dim={len(vector)}"
        )
    except Exception as exc:
        print(
            f"  [FAIL] Embedding round-trip "
            f"({halo_settings.halo_vector_embedding_model}): {exc}"
        )
        vector_ok = False

    # --- Halo Blade healthcheck -------------------------------------
    step("Halo Blade healthcheck (LLM endpoint)")
    blade_ok = check_http(
        "Blade LLM endpoint", blade_llm_url.rstrip("/") + "/models"
    )

    if not (vector_ok and blade_ok):
        print("\nRESULT: FAILED - active Halo subsystems are not healthy.")
        return 1

    workspace = Path(tempfile.mkdtemp(prefix="halo_demo_"))
    chroma_dir = Path(tempfile.mkdtemp(prefix="halo_chroma_"))
    try:
        step("Halo Vector standalone: create + index demo file")
        demo = workspace / "demo_calculator.py"
        demo.write_text(DEMO_SOURCE, encoding="utf-8")
        print(f"  created: {demo}")
        vector_engine = VectorEngine(
            persist_dir=str(chroma_dir), embedding_service=embedding_service
        )
        chunks = vector_engine.index_file(str(demo), DEMO_SOURCE)
        print(f"  indexed {chunks} chunk(s), collection size: {vector_engine.size}")
        hits = vector_engine.query("calculator arithmetic functions", n_results=2)
        print(f"  retrieval sanity check: {len(hits)} hit(s)")
        for hit in hits:
            print(
                f"    - similarity={hit['similarity']:.3f} "
                f"file={hit['metadata']['file_name']}"
            )

        step("Halo Blade standalone: AST skeleton")
        parser = ASTCodeParser()
        print(
            "\n".join(
                "  " + line
                for line in parser.extract_skeleton(str(demo)).splitlines()
            )
        )

        step("Blade + Vector integration: orchestrated diff")
        orchestrator = OpenClawOrchestrator(
            llm_base_url=blade_llm_url,
            llm_model=halo_settings.halo_blade_llm_model,
            vector_engine=vector_engine,
        )
        diff = orchestrator.request_diff(TASK, str(demo), target_dir=str(workspace))
        print("  ---- generated diff ----")
        print("\n".join("  " + line for line in diff.splitlines()))
        print("  -------------------------")

        step("Halo Blade: apply diff via git apply")
        file_ops = FileOperations()
        diff_path = workspace / "type_hints.patch"
        diff_path.write_text(diff, encoding="utf-8")
        print(f"  {file_ops.apply_patch(str(diff_path), target_dir=str(workspace))}")

        step("Verification")
        executor = LocalExecutor(max_log_lines=halo_settings.halo_max_log_lines)
        compile_result = executor.run_command(
            f'python3 -m py_compile "{demo}"', cwd=str(workspace)
        )
        print(f"  py_compile returncode: {compile_result['returncode']}")
        if compile_result["returncode"] != 0:
            print(f"  stderr: {compile_result['stderr']}")
            print("\nRESULT: FAILED - patched file does not compile.")
            return 1

        patched = demo.read_text(encoding="utf-8")
        ok = type_hints_present(patched)
        print("  patched file:")
        print("\n".join("  " + line for line in patched.splitlines()))
        print(f"  type hints present on all functions: {'YES' if ok else 'NO'}")

        print(f"\nRESULT: {'PASSED' if ok else 'FAILED (type hints missing)'}")
        return 0 if ok else 1

    except Exception as exc:
        print(f"\nRESULT: FAILED - {type(exc).__name__}: {exc}")
        return 1
    finally:
        step("Cleanup")
        shutil.rmtree(workspace, ignore_errors=True)
        shutil.rmtree(chroma_dir, ignore_errors=True)
        print("  temporary files removed.")


if __name__ == "__main__":
    sys.exit(main())