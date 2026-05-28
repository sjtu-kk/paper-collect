# Derived from AutoResearchClaw (MIT License). Copyright (c) 2026 Aiming Lab.

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProviderBlocked(Exception):
    source_name: str
    status: str
    message: str = ""

    def __str__(self) -> str:
        if self.message:
            return f"{self.source_name}:{self.status}: {self.message}"
        return f"{self.source_name}:{self.status}"
