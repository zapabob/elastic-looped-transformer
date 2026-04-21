"""Runtime configuration for the dense MILSPEC-style agent scaffold."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class SandboxConfig:
    timeout_s: float = 3.0


@dataclass
class AuditConfig:
    path: str = "runs/milspec_agent/audit.jsonl"


@dataclass
class SbomConfig:
    enabled: bool = True
    path: str = "runs/milspec_agent/sbom.json"


@dataclass
class AgentRuntimeConfig:
    L: int = 4
    temperature: float = 0.0
    top_k: int = 1
    require_determinism: bool = True
    sandbox: SandboxConfig = field(default_factory=SandboxConfig)
    audit: AuditConfig = field(default_factory=AuditConfig)
    sbom: SbomConfig = field(default_factory=SbomConfig)


def load_agent_runtime_config(path: str | Path) -> AgentRuntimeConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    sandbox_raw = raw.pop("sandbox", {}) or {}
    audit_raw = raw.pop("audit", {}) or {}
    sbom_raw = raw.pop("sbom", {}) or {}
    return AgentRuntimeConfig(
        sandbox=SandboxConfig(**sandbox_raw),
        audit=AuditConfig(**audit_raw),
        sbom=SbomConfig(**sbom_raw),
        **raw,
    )
