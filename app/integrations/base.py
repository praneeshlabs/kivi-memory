"""Shared shape for Kivi's app connections.

Every integration is credential-gated and reports its own readiness, so the UI
can show what is connected without the caller knowing each service's details.
These are personal-use credentials (tokens and app passwords) rather than a
distributable OAuth app: one user, their own accounts, secrets in .env.
"""
import os
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class IntegrationStatus:
    key: str
    name: str
    direction: str          # "read" (learns into memory) or "write" (acts outward)
    summary: str
    connected: bool
    missing: list[str] = field(default_factory=list)
    setup_url: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "name": self.name,
            "direction": self.direction,
            "summary": self.summary,
            "connected": self.connected,
            "missing": self.missing,
            "setup_url": self.setup_url,
        }


def env(name: str) -> Optional[str]:
    value = os.getenv(name, "").strip()
    return value or None


def missing_vars(*names: str) -> list[str]:
    return [name for name in names if not env(name)]


class IntegrationError(RuntimeError):
    """Raised when a configured integration fails at the remote end."""
