"""Lightweight subprocess sandbox helpers for verifier and agent execution."""

from __future__ import annotations

from dataclasses import dataclass
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Iterable


@dataclass
class SandboxResult:
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool = False


def _sandbox_env(extra_env: dict[str, str] | None = None) -> dict[str, str]:
    env = {
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PATH": os.environ.get("PATH", ""),
        "SYSTEMROOT": os.environ.get("SYSTEMROOT", ""),
        "TEMP": os.environ.get("TEMP", ""),
        "TMP": os.environ.get("TMP", ""),
    }
    if extra_env:
        env.update(extra_env)
    return env


def _text_or_empty(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def run_python_code(
    code: str,
    *,
    timeout_s: float = 3.0,
    cwd: str | Path | None = None,
    extra_env: dict[str, str] | None = None,
) -> SandboxResult:
    with tempfile.TemporaryDirectory() as td:
        root = Path(cwd) if cwd is not None else Path(td)
        root.mkdir(parents=True, exist_ok=True)
        script = root / "sandbox_target.py"
        script.write_text(code, encoding="utf-8")
        try:
            result = subprocess.run(
                [sys.executable, str(script)],
                cwd=root,
                capture_output=True,
                text=True,
                timeout=timeout_s,
                check=False,
                env=_sandbox_env(extra_env),
            )
        except subprocess.TimeoutExpired as exc:
            return SandboxResult(
                returncode=-1,
                stdout=_text_or_empty(exc.stdout),
                stderr=_text_or_empty(exc.stderr),
                timed_out=True,
            )
        return SandboxResult(
            returncode=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
            timed_out=False,
        )


def run_python_module(
    module: str,
    args: Iterable[str],
    *,
    timeout_s: float = 5.0,
    cwd: str | Path | None = None,
    extra_env: dict[str, str] | None = None,
) -> SandboxResult:
    try:
        result = subprocess.run(
            [sys.executable, "-m", module, *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
            env=_sandbox_env(extra_env),
        )
    except subprocess.TimeoutExpired as exc:
        return SandboxResult(
            returncode=-1,
            stdout=_text_or_empty(exc.stdout),
            stderr=_text_or_empty(exc.stderr),
            timed_out=True,
        )
    return SandboxResult(
        returncode=result.returncode,
        stdout=result.stdout,
        stderr=result.stderr,
        timed_out=False,
    )
