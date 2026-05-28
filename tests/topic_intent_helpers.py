from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def write_confirmed_topic_intent(
    run_root: Path,
    *,
    topic: str,
    description: str = "",
    intended_meaning: list[str] | None = None,
    research_purpose: list[str] | None = None,
    included_concepts: list[str] | None = None,
    excluded_meanings: list[str] | None = None,
    uncertainties: list[str] | None = None,
    human_notes: str = "confirmed by test",
    extra_fields: dict[str, Any] | None = None,
) -> Path:
    intent = {
        "artifact_version": "topic_search_topic_intent.v1",
        "status": "confirmed",
        "topic": topic,
        "description": description,
        "intended_meaning": intended_meaning
        or ["AI agent systems that autonomously search and synthesize research literature"],
        "research_purpose": research_purpose or ["Learn how others automate scholarly information collection"],
        "included_concepts": included_concepts or ["autonomous research agents", "automated literature discovery"],
        "excluded_meanings": excluded_meanings or ["automotive research", "self-driving vehicles"],
        "uncertainties": uncertainties or [],
        "human_notes": human_notes,
    }
    if extra_fields:
        intent.update(extra_fields)
    stage01_dir = run_root / "stage-01"
    stage01_dir.mkdir(parents=True, exist_ok=True)
    intent_path = stage01_dir / "topic_intent.md"
    intent_path.write_text(
        yaml.safe_dump(intent, sort_keys=False, allow_unicode=False),
        encoding="utf-8",
    )
    return intent_path


def install_stage01_autoresume(monkeypatch: Any, *, cli_module: Any, topic_search_module: Any) -> None:
    real_cli_run = cli_module.run_topic_search
    real_topic_run = topic_search_module.run_topic_search

    def wrap(run_func: Any) -> Any:
        def run_with_confirmed_intent(**kwargs: Any) -> Any:
            topic = str(kwargs.get("topic") or "").strip()
            if not kwargs.get("resume") and topic:
                generated_at = topic_search_module._utc_timestamp()
                run_id = kwargs.get("run_id") or topic_search_module._default_run_id(topic, generated_at)
                stable_run_id = topic_search_module._safe_run_id(run_id)
                kwargs["run_id"] = stable_run_id
                output_root = Path(kwargs.get("output_root") or Path("runtime_data") / "topic_search_runs")
                write_confirmed_topic_intent(
                    output_root / stable_run_id,
                    topic=topic,
                    description=str(kwargs.get("description") or ""),
                )
                kwargs["resume"] = True
            return run_func(**kwargs)

        return run_with_confirmed_intent

    monkeypatch.setattr(cli_module, "run_topic_search", wrap(real_cli_run), raising=False)
    monkeypatch.setattr(topic_search_module, "run_topic_search", wrap(real_topic_run), raising=False)
