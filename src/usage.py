"""Per-query usage logging: token counts + wall-clock latency -> JSONL.

Opt-in via the USAGE_LOG env var (empty = disabled). Every answered question
appends one JSON line, so cost/latency trends can be analyzed later with plain
jq / pandas. Refusals ("I don't know") are logged too — latency was spent on
retrieval even when the answer was refused.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class UsageRecord:
    timestamp: str  # UTC ISO-8601
    question: str  # truncated to 200 chars by the caller
    model: str
    grounded: bool  # False when we refused for lack of evidence
    num_sources: int
    prompt_tokens: int | None  # None when the provider did not report usage
    completion_tokens: int | None
    total_tokens: int | None
    latency_ms: float  # wall clock: question received -> answer ready


def append_usage(log_path: str, record: UsageRecord) -> None:
    """Append one record as a single JSON line; parent dirs are created."""
    path = Path(log_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
