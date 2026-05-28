from __future__ import annotations

import json
import os
import re
import urllib.request
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from paper_collect.literature import filter_verified_bibtex, papers_to_bibtex, search_papers_multi_query, verify_citations
from paper_collect import provider_registry

REFERENCE_PLAN_LABELS = ("openreview", "google_scholar")
DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
DEFAULT_OPENAI_TIMEOUT_SECONDS = 120.0
MIN_STAGE03_STRATEGIES = 3
MIN_STAGE03_QUERIES = 8


@dataclass(frozen=True)
class TopicSearchRunResult:
    run_id: str
    run_root: Path
    run_meta_path: Path
    stage02_dir: Path
    stage03_dir: Path


def run_topic_search(
    *,
    topic: str | None,
    description: str = "",
    run_id: str | None = None,
    output_root: Path | str = Path("runtime_data") / "topic_search_runs",
    max_results_per_query: int = 40,
    year_min: int = 2020,
    s2_api_key: str = "",
    inter_query_delay: float = 1.5,
    inter_verify_delay: float = 1.0,
    verification_enabled: bool = True,
    plan_only: bool = False,
    allow_deterministic_fallback: bool = False,
    resume: bool = False,
) -> TopicSearchRunResult:
    topic = (topic or "").strip()
    description = description.strip()
    if not resume and not topic:
        raise ValueError("topic must not be empty")
    if resume and not run_id:
        raise ValueError("resume requires --run-id")
    if max_results_per_query < 1:
        raise ValueError("max_results_per_query must be positive")

    generated_at = _utc_timestamp()
    stable_run_id = _safe_run_id(run_id or _default_run_id(topic, generated_at))
    run_root = Path(output_root) / stable_run_id
    stage01_dir = run_root / "stage-01"
    topic_intent = collect_stage01(
        stage01_dir=stage01_dir,
        topic=topic,
        description=description,
        generated_at=generated_at,
        resume=resume,
    )
    topic = str(topic_intent.get("topic") or topic).strip()
    description = str(topic_intent.get("description") or description).strip()
    if topic_intent.get("status") != "confirmed":
        run_meta = _build_stage01_blocked_run_meta(
            stable_run_id=stable_run_id,
            generated_at=generated_at,
            topic=topic,
            description=description,
            max_results_per_query=max_results_per_query,
            year_min=year_min,
            s2_api_key=s2_api_key,
            inter_query_delay=inter_query_delay,
            inter_verify_delay=inter_verify_delay,
        )
        run_meta_path = run_root / "run_meta.json"
        _write_json(run_meta_path, run_meta)
        return TopicSearchRunResult(stable_run_id, run_root, run_meta_path, run_root / "stage-02", run_root / "stage-03")

    stage02_dir = run_root / "stage-02"
    stage03_dir = run_root / "stage-03"
    topic_profile = collect_stage02(
        stage02_dir=stage02_dir,
        topic=topic,
        topic_intent=topic_intent,
        generated_at=generated_at,
        allow_deterministic_fallback=allow_deterministic_fallback,
    )
    topic_profile_generation = {
        "mode": topic_profile.get("generation", {}).get("mode", "deterministic_fallback"),
        "fallback_reason": topic_profile.get("generation", {}).get("fallback_reason", ""),
    }

    stage03_result = collect_stage03(
        stage03_dir=stage03_dir,
        topic=topic,
        topic_profile=topic_profile,
        max_results_per_query=max_results_per_query,
        year_min=year_min,
        generated_at=generated_at,
        allow_deterministic_fallback=allow_deterministic_fallback,
    )
    queries = stage03_result["queries"]

    if plan_only:
        run_meta = _build_run_meta(
            stable_run_id=stable_run_id,
            generated_at=generated_at,
            topic=topic,
            description=description,
            max_results_per_query=max_results_per_query,
            year_min=year_min,
            s2_api_key=s2_api_key,
            inter_query_delay=inter_query_delay,
            inter_verify_delay=inter_verify_delay,
            stage04_status="disabled",
            stage05_status="disabled",
            verification_enabled=False,
            topic_profile_generation=topic_profile_generation,
            query_planning=stage03_result["query_planning"],
            include_stage04_artifacts=False,
            include_stage05_artifacts=False,
        )
        run_meta_path = run_root / "run_meta.json"
        _write_json(run_meta_path, run_meta)
        return TopicSearchRunResult(stable_run_id, run_root, run_meta_path, stage02_dir, stage03_dir)

    stage04_dir = run_root / "stage-04"
    stage04_result = collect_stage04(
        stage04_dir=stage04_dir,
        queries=queries,
        search_intents=stage03_result["search_intents"],
        max_results_per_query=max_results_per_query,
        year_min=year_min,
        s2_api_key=s2_api_key,
        inter_query_delay=inter_query_delay,
        generated_at=generated_at,
    )
    stage05_dir = run_root / "stage-05"
    if verification_enabled:
        stage05_result = collect_stage05(
            stage05_dir=stage05_dir,
            references_bib=str(stage04_result["references_bib"]),
            s2_api_key=s2_api_key,
            inter_verify_delay=inter_verify_delay,
            generated_at=generated_at,
            upstream_stage_status=str(stage04_result["stage_status"]),
        )
    else:
        stage05_result = {
            "stage_status": "disabled",
            "verification_report": {
                "artifact_version": "topic_search_verification_report.v1",
                "stage": "stage-05",
                "generated_at": generated_at,
                "status": "disabled",
                "include_suspicious": True,
                "summary": {
                    "total": 0,
                    "verified": 0,
                    "suspicious": 0,
                    "hallucinated": 0,
                    "skipped": 0,
                    "integrity_score": 1.0,
                },
                "results": [],
            },
            "references_verified_bib": "",
        }

    run_meta = _build_run_meta(
        stable_run_id=stable_run_id,
        generated_at=generated_at,
        topic=topic,
        description=description,
        max_results_per_query=max_results_per_query,
        year_min=year_min,
        s2_api_key=s2_api_key,
        inter_query_delay=inter_query_delay,
        inter_verify_delay=inter_verify_delay,
        stage04_status=stage04_result["stage_status"],
        stage05_status=stage05_result["stage_status"],
        verification_enabled=verification_enabled,
        topic_profile_generation=topic_profile_generation,
        query_planning=stage03_result["query_planning"],
        include_stage04_artifacts=True,
        include_stage05_artifacts=verification_enabled,
    )
    run_meta_path = run_root / "run_meta.json"
    _write_json(run_meta_path, run_meta)
    return TopicSearchRunResult(stable_run_id, run_root, run_meta_path, stage02_dir, stage03_dir)


def _build_run_meta(
    *,
    stable_run_id: str,
    generated_at: str,
    topic: str,
    description: str,
    max_results_per_query: int,
    year_min: int,
    s2_api_key: str,
    inter_query_delay: float,
    inter_verify_delay: float,
    stage04_status: str,
    stage05_status: str,
    verification_enabled: bool,
    topic_profile_generation: dict[str, Any],
    query_planning: dict[str, Any],
    include_stage04_artifacts: bool,
    include_stage05_artifacts: bool,
) -> dict[str, Any]:
    return {
        "artifact_version": "topic_search_run.v1",
        "run_id": stable_run_id,
        "generated_at": generated_at,
        "request_type": "topic",
        "request": {
            "topic": topic,
            "description": description,
            "max_results_per_query": max_results_per_query,
            "year_min": year_min,
            "s2_api_key_provided": bool(s2_api_key),
            "inter_query_delay": inter_query_delay,
            "inter_verify_delay": inter_verify_delay,
        },
        "status": "complete" if stage04_status == "complete" and stage05_status in {"complete", "disabled"} else stage04_status,
        "cache": {
            "enabled": False,
            "can_substitute_provider_results": False,
        },
        "stage_statuses": {
            "stage-01": "confirmed",
            "stage-02": "complete",
            "stage-03": "complete",
            "stage-04": stage04_status,
            "stage-05": stage05_status,
        },
        "stage_artifacts": {
            "stage-01": {
                "topic_intent_draft_md": "stage-01/topic_intent_draft.md",
                "topic_intent_md": "stage-01/topic_intent.md",
            },
            "stage-02": {
                "topic_profile_md": "stage-02/topic_profile.md",
            },
            "stage-03": {
                "search_plan_yaml": "stage-03/search_plan.yaml",
                "sources_json": "stage-03/sources.json",
                "search_intents_json": "stage-03/search_intents.json",
                "queries_json": "stage-03/queries.json",
            },
            "stage-04": {} if not include_stage04_artifacts else {
                "candidates_jsonl": "stage-04/candidates.jsonl",
                "references_bib": "stage-04/references.bib",
                "search_meta_json": "stage-04/search_meta.json",
                "match_ledger_jsonl": "stage-04/match_ledger.jsonl",
            },
            "stage-05": {} if not include_stage05_artifacts else {
                "verification_report_json": "stage-05/verification_report.json",
                "references_verified_bib": "stage-05/references_verified.bib",
            },
        },
        "verification": {
            "enabled": verification_enabled,
            "status": stage05_status,
        },
        "topic_profile_generation": topic_profile_generation,
        "query_planning": query_planning,
        "out_of_scope_guardrails": _guardrails(),
    }


def _build_stage01_blocked_run_meta(
    *,
    stable_run_id: str,
    generated_at: str,
    topic: str,
    description: str,
    max_results_per_query: int,
    year_min: int,
    s2_api_key: str,
    inter_query_delay: float,
    inter_verify_delay: float,
) -> dict[str, Any]:
    return {
        "artifact_version": "topic_search_run.v1",
        "run_id": stable_run_id,
        "generated_at": generated_at,
        "request_type": "topic",
        "status": "needs_human_topic_intent",
        "request": {
            "topic": topic,
            "description": description,
            "max_results_per_query": max_results_per_query,
            "year_min": year_min,
            "s2_api_key_provided": bool(s2_api_key),
            "inter_query_delay": inter_query_delay,
            "inter_verify_delay": inter_verify_delay,
        },
        "cache": {
            "enabled": False,
            "can_substitute_provider_results": False,
        },
        "stage_statuses": {
            "stage-01": "needs_human_topic_intent",
            "stage-02": "blocked",
            "stage-03": "blocked",
            "stage-04": "blocked",
            "stage-05": "blocked",
        },
        "stage_artifacts": {
            "stage-01": {
                "topic_intent_draft_md": "stage-01/topic_intent_draft.md",
                "topic_intent_md": "stage-01/topic_intent.md",
            },
            "stage-02": {},
            "stage-03": {},
            "stage-04": {},
            "stage-05": {},
        },
        "verification": {
            "enabled": False,
            "status": "blocked",
        },
        "topic_intent": {
            "status": "draft",
            "requires_human_confirmation": True,
        },
        "out_of_scope_guardrails": _guardrails(),
    }


def collect_stage01(
    *,
    stage01_dir: Path,
    topic: str,
    description: str,
    generated_at: str,
    resume: bool,
) -> dict[str, Any]:
    confirmed_path = stage01_dir / "topic_intent.md"
    draft_path = stage01_dir / "topic_intent_draft.md"
    if resume:
        if not confirmed_path.is_file():
            raise ValueError("confirmed topic intent is required before resume can enter Stage02")
        confirmed = _read_yaml_mapping(confirmed_path)
        _validate_resume_request(confirmed, topic=topic, description=description)
        _validate_topic_intent_contract(confirmed)
        if not draft_path.is_file():
            _write_yaml(draft_path, _draft_topic_intent_from_formal(confirmed, generated_at=generated_at))
        return confirmed

    if not topic:
        raise ValueError("topic must not be empty")
    if confirmed_path.is_file():
        confirmed = _read_yaml_mapping(confirmed_path)
        _validate_resume_request(confirmed, topic=topic, description=description)
        _validate_topic_intent_contract(confirmed)
        if not draft_path.is_file():
            _write_yaml(draft_path, _draft_topic_intent_from_formal(confirmed, generated_at=generated_at))
        return confirmed

    draft = _build_llm_stage01_topic_intent_draft(
        topic=topic,
        description=description,
        generated_at=generated_at,
    ) or _build_topic_intent_draft(topic=topic, description=description, generated_at=generated_at)
    _write_yaml(draft_path, draft)
    _write_yaml(confirmed_path, draft)
    return draft


def _build_llm_stage01_topic_intent_draft(
    *,
    topic: str,
    description: str,
    generated_at: str,
) -> dict[str, Any] | None:
    config, _missing_reason = _openai_env_config()
    if config is None:
        return None
    try:
        payload = _request_openai_compatible_topic_intent_draft(config, topic, description)
    except Exception:  # noqa: BLE001
        return None
    return _validate_llm_topic_intent_draft(payload, topic=topic, description=description, generated_at=generated_at)


def _request_openai_compatible_topic_intent_draft(
    config: dict[str, str],
    topic: str,
    description: str,
) -> dict[str, Any]:
    endpoint = f"{config['base_url']}/chat/completions"
    prompt = (
        "Create a Stage 01-lite topic intent draft as JSON. Return only a JSON object with "
        "topic_intent_draft containing intended_meaning, research_purpose, included_concepts, "
        "excluded_meanings, uncertainties, and human_notes. This is an editable template for a "
        "human to review in topic_intent.md and confirm; it is not a report. Do not include "
        "Stage 02 or Stage 03 fields such as sub_questions, priorities, strategies, or queries."
    )
    body = {
        "model": config["model"],
        "messages": [
            {"role": "system", "content": "You draft editable literature-search intent templates."},
            {
                "role": "user",
                "content": f"{prompt}\nTopic: {topic}\nDescription: {description}",
            },
        ],
        "temperature": 0.1,
        "response_format": {"type": "json_object"},
    }
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {config['api_key']}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    timeout = float(os.environ.get("OPENAI_TIMEOUT_SECONDS", str(DEFAULT_OPENAI_TIMEOUT_SECONDS)))
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = json.loads(response.read().decode("utf-8"))
    content = raw.get("choices", [{}])[0].get("message", {}).get("content")
    if isinstance(content, dict):
        return content
    if not isinstance(content, str) or not content.strip():
        return {}
    try:
        loaded = json.loads(content)
    except json.JSONDecodeError:
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _validate_llm_topic_intent_draft(
    payload: dict[str, Any],
    *,
    topic: str,
    description: str,
    generated_at: str,
) -> dict[str, Any] | None:
    raw_draft = payload.get("topic_intent_draft")
    if not isinstance(raw_draft, dict):
        direct_keys = {
            "intended_meaning",
            "research_purpose",
            "included_concepts",
            "excluded_meanings",
            "uncertainties",
            "human_notes",
        }
        raw_draft = payload if any(key in payload for key in direct_keys) else None
    if not isinstance(raw_draft, dict):
        return None

    intended_meaning = _sanitize_stage01_string_list(raw_draft.get("intended_meaning"))
    if not intended_meaning:
        return None
    research_purpose = _sanitize_stage01_string_list(raw_draft.get("research_purpose"))
    included_concepts = _sanitize_stage01_string_list(raw_draft.get("included_concepts"))
    excluded_meanings = _sanitize_stage01_string_list(raw_draft.get("excluded_meanings"))
    uncertainties = _sanitize_stage01_string_list(raw_draft.get("uncertainties"))
    human_notes = _safe_text(str(raw_draft.get("human_notes") or ""), limit=400)
    if "human" not in human_notes.casefold():
        human_notes = (
            f"{human_notes} Human: edit topic_intent.md, delete wrong candidates, and change "
            "status to confirmed before resume."
        ).strip()
    return {
        "artifact_version": "topic_search_topic_intent.v1",
        "status": "draft",
        "topic": topic,
        "description": description,
        "generated_at": generated_at,
        "intended_meaning": intended_meaning,
        "research_purpose": research_purpose or _draft_research_purpose(description),
        "included_concepts": included_concepts or intended_meaning,
        "excluded_meanings": excluded_meanings,
        "uncertainties": uncertainties,
        "human_notes": human_notes,
    }


def _build_topic_intent_draft(*, topic: str, description: str, generated_at: str) -> dict[str, Any]:
    intended_meaning = _draft_intended_meanings(topic, description)
    return {
        "artifact_version": "topic_search_topic_intent.v1",
        "status": "draft",
        "topic": topic,
        "description": description,
        "generated_at": generated_at,
        "intended_meaning": intended_meaning,
        "research_purpose": _draft_research_purpose(description),
        "included_concepts": intended_meaning,
        "excluded_meanings": [],
        "uncertainties": ["Review whether the intended meaning is too broad, too narrow, or ambiguous."],
        "human_notes": "Human: edit topic_intent.md, delete wrong candidates, and change status to confirmed before resume.",
    }


def _sanitize_stage01_string_list(raw_values: Any) -> list[str]:
    if isinstance(raw_values, str):
        value = _safe_text(raw_values, limit=800)
        return [value] if value else []
    if not isinstance(raw_values, list):
        return []
    values: list[str] = []
    for raw_value in raw_values:
        if not isinstance(raw_value, str):
            continue
        value = _safe_text(raw_value, limit=800)
        if value and value.casefold() not in {existing.casefold() for existing in values}:
            values.append(value)
    return values


def _draft_topic_intent_from_formal(formal: dict[str, Any], *, generated_at: str) -> dict[str, Any]:
    return {
        **formal,
        "status": "draft",
        "generated_at": generated_at,
        "human_notes": "Draft reconstructed from topic_intent.md for review history.",
    }


def _draft_intended_meanings(topic: str, description: str) -> list[str]:
    values = []
    if description:
        values.append(description)
    values.append(topic)
    return _merge_string_lists([_safe_text(value, limit=200) for value in values if value.strip()])


def _draft_research_purpose(description: str) -> list[str]:
    if description:
        return [_safe_text(description, limit=200)]
    return ["Clarify the literature search intent before query planning."]


def _validate_resume_request(confirmed: dict[str, Any], *, topic: str, description: str) -> None:
    existing_topic = str(confirmed.get("topic") or "").strip()
    existing_description = str(confirmed.get("description") or "").strip()
    if topic and topic != existing_topic:
        raise ValueError("resume topic does not match existing run topic")
    if description and description != existing_description:
        raise ValueError("resume description does not match existing run description")


def _validate_confirmed_topic_intent(intent: dict[str, Any]) -> None:
    _validate_topic_intent_contract(intent)
    if str(intent.get("status") or "").strip() != "confirmed":
        raise ValueError("topic_intent.md must have status: confirmed")


def _validate_topic_intent_contract(intent: dict[str, Any]) -> None:
    if intent.get("artifact_version") != "topic_search_topic_intent.v1":
        raise ValueError("topic_intent.md has unsupported artifact_version")
    status = str(intent.get("status") or "").strip()
    if status not in {"draft", "confirmed"}:
        raise ValueError("topic_intent.md must have status: draft or confirmed")
    if not str(intent.get("topic") or "").strip():
        raise ValueError("topic_intent.md must include topic")
    if not _sanitize_string_list(intent.get("intended_meaning")):
        raise ValueError("topic_intent.md must include non-empty intended_meaning")
    forbidden = {"sub_questions", "priorities", "strategies", "queries"}
    present = sorted(key for key in forbidden if key in intent)
    if present:
        raise ValueError(f"topic_intent.md contains Stage02/03 fields: {', '.join(present)}")


def _read_yaml_mapping(path: Path) -> dict[str, Any]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"{path.name} must be a YAML mapping")
    return loaded


def collect_stage02(
    *,
    stage02_dir: Path,
    topic: str,
    topic_intent: dict[str, Any] | None = None,
    generated_at: str,
    allow_deterministic_fallback: bool = False,
) -> dict[str, Any]:
    topic_intent = topic_intent or _minimal_confirmed_topic_intent(topic)
    llm_profile, fallback_reason = _build_llm_stage02_profile(topic, topic_intent, generated_at)
    if llm_profile is not None:
        llm_profile = {
            **llm_profile,
            "generation": {
                "mode": "llm",
                "fallback_reason": "",
                "llm_required": True,
            },
        }
        _write_yaml(stage02_dir / "topic_profile.md", llm_profile)
        return llm_profile

    intended_meaning = _sanitize_string_list(topic_intent.get("intended_meaning"))
    included_concepts = _sanitize_string_list(topic_intent.get("included_concepts"))
    research_purpose = _sanitize_string_list(topic_intent.get("research_purpose"))
    excluded_interpretations = _sanitize_string_list(topic_intent.get("excluded_meanings"))
    uncertainties = _sanitize_string_list(topic_intent.get("uncertainties"))
    primary_interpretation = intended_meaning[0] if intended_meaning else ""
    scope = _merge_string_lists(included_concepts, intended_meaning)
    disambiguation_terms = _merge_string_lists(intended_meaning, included_concepts)
    negative_terms = excluded_interpretations
    quality_warnings = []
    if not intended_meaning:
        quality_warnings.append("sparse_confirmed_intent_missing_intended_meaning")
    if not included_concepts:
        quality_warnings.append("sparse_confirmed_intent_missing_included_concepts")
    profile = {
        "artifact_version": "topic_search_topic_profile.v1",
        "stage": "stage-02",
        "topic": topic,
        "generated_at": generated_at,
        "generation": {
            "mode": "deterministic_fallback",
            "fallback_reason": fallback_reason,
            "llm_required": False,
        },
        "topic_goal": _safe_text("; ".join(intended_meaning or [topic]), limit=600),
        "scope": scope,
        "boundary_notes": _boundary_notes(primary_interpretation, excluded_interpretations),
        "primary_interpretation": primary_interpretation,
        "excluded_interpretations": excluded_interpretations,
        "disambiguation_hints": disambiguation_terms,
        "negative_terms": negative_terms,
        "sub_questions": research_purpose,
        "priorities": [],
        "review_hints": uncertainties,
        "quality_warnings": quality_warnings,
    }
    _write_yaml(stage02_dir / "topic_profile.md", profile)
    return profile


def _minimal_confirmed_topic_intent(topic: str) -> dict[str, Any]:
    return {
        "artifact_version": "topic_search_topic_intent.v1",
        "status": "confirmed",
        "topic": topic,
        "description": "",
        "intended_meaning": [topic],
        "research_purpose": [],
        "included_concepts": [topic],
        "excluded_meanings": [],
        "uncertainties": [],
        "human_notes": "",
    }


def _build_llm_stage02_profile(
    topic: str,
    topic_intent: dict[str, Any],
    generated_at: str,
) -> tuple[dict[str, Any] | None, str]:
    config, missing_reason = _openai_env_config()
    if config is None:
        return None, missing_reason
    try:
        payload = _request_openai_compatible_topic_profile(config, topic, topic_intent)
    except Exception:  # noqa: BLE001
        return None, "llm_call_failed"
    profile = _validate_llm_topic_profile(payload, topic=topic, topic_intent=topic_intent, generated_at=generated_at)
    if profile is None:
        return None, "invalid_llm_topic_profile"
    return profile, ""


def _request_openai_compatible_topic_profile(
    config: dict[str, str],
    topic: str,
    topic_intent: dict[str, Any],
) -> dict[str, Any]:
    endpoint = f"{config['base_url']}/chat/completions"
    prompt = (
        "Create a Stage 02-lite topic profile as JSON. Return only a JSON object with "
        "topic_profile containing topic_goal, scope, boundary_notes, disambiguation_hints, "
        "negative_terms, sub_questions, priorities, review_hints, primary_interpretation, "
        "excluded_interpretations, and quality_warnings. For short ambiguous topics, identify "
        "the intended research meaning and excluded meanings from the confirmed Stage 01 intent. "
        "Do not include provider selection, "
        "provider requests, URLs, candidates, BibTeX, citations, or bibliographic truth."
    )
    body = {
        "model": config["model"],
        "messages": [
            {"role": "system", "content": "You decompose research topics without choosing providers or papers."},
            {
                "role": "user",
                "content": (
                    f"{prompt}\nTopic: {topic}\n"
                    f"Confirmed topic intent: {json.dumps(topic_intent, ensure_ascii=True)}"
                ),
            },
        ],
        "temperature": 0.1,
        "response_format": {"type": "json_object"},
    }
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {config['api_key']}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    timeout = float(os.environ.get("OPENAI_TIMEOUT_SECONDS", str(DEFAULT_OPENAI_TIMEOUT_SECONDS)))
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = json.loads(response.read().decode("utf-8"))
    content = raw.get("choices", [{}])[0].get("message", {}).get("content")
    if isinstance(content, dict):
        return content
    if not isinstance(content, str) or not content.strip():
        return {}
    try:
        loaded = json.loads(content)
    except json.JSONDecodeError:
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _validate_llm_topic_profile(
    payload: dict[str, Any],
    *,
    topic: str,
    topic_intent: dict[str, Any] | None = None,
    generated_at: str,
) -> dict[str, Any] | None:
    raw_profile = payload.get("topic_profile")
    if not isinstance(raw_profile, dict):
        direct_keys = {
            "topic_goal",
            "scope",
            "boundary_notes",
            "disambiguation_hints",
            "negative_terms",
            "sub_questions",
            "priorities",
            "review_hints",
        }
        raw_profile = payload if any(key in payload for key in direct_keys) else None
    if not isinstance(raw_profile, dict):
        return None

    topic_intent = topic_intent or _minimal_confirmed_topic_intent(topic)
    intent_meanings = _sanitize_string_list(topic_intent.get("intended_meaning"))
    intent_included = _sanitize_string_list(topic_intent.get("included_concepts"))
    intent_excluded = _sanitize_string_list(topic_intent.get("excluded_meanings"))
    intent_positive_terms = _merge_string_lists(intent_meanings, intent_included)
    topic_goal, topic_goal_warnings = _safe_topic_goal(str(raw_profile.get("topic_goal") or "; ".join(intent_meanings) or topic))
    negative_terms = _merge_string_lists(
        intent_excluded,
        _sanitize_string_list(raw_profile.get("negative_terms")),
    )
    disambiguation_hints = _merge_string_lists(
        intent_meanings,
        intent_included,
        _sanitize_string_list(raw_profile.get("disambiguation_hints")),
    )
    primary_interpretation = _safe_text(
        str(raw_profile.get("primary_interpretation") or (intent_meanings[0] if intent_meanings else "")),
        limit=240,
    )
    quality_warnings = _merge_string_lists(
        _sanitize_string_list(raw_profile.get("quality_warnings")),
        topic_goal_warnings,
    )
    raw_excluded_interpretations = _sanitize_string_list(raw_profile.get("excluded_interpretations"))
    primary_repaired_from_intent = False
    if intent_meanings and _text_conflicts_with_boundary(primary_interpretation, intent_excluded):
        primary_interpretation = _safe_text(intent_meanings[0], limit=240)
        primary_repaired_from_intent = True
        quality_warnings.append("primary_interpretation_repaired_from_confirmed_intent")
    if primary_repaired_from_intent:
        excluded_interpretations = intent_excluded
        if raw_excluded_interpretations:
            quality_warnings.append("excluded_interpretation_repaired_from_confirmed_intent")
    else:
        kept_raw_excluded = []
        removed_positive_excluded = False
        for value in raw_excluded_interpretations:
            if _text_conflicts_with_boundary(value, intent_positive_terms):
                removed_positive_excluded = True
                continue
            kept_raw_excluded.append(value)
        if removed_positive_excluded:
            quality_warnings.append("excluded_interpretation_repaired_from_confirmed_intent")
        excluded_interpretations = _merge_string_lists(intent_excluded, kept_raw_excluded)
    scope = _sanitize_string_list(raw_profile.get("scope"))
    if not scope:
        scope = _merge_string_lists(intent_included, intent_meanings)
    raw_boundary_notes = _sanitize_string_list(raw_profile.get("boundary_notes"))
    boundary_notes = _merge_string_lists(
        raw_boundary_notes,
        _boundary_notes(primary_interpretation, excluded_interpretations),
    )
    sub_questions, removed_excluded_sub_question = _filter_boundary_conflicting_strings(
        _sanitize_string_list(raw_profile.get("sub_questions")),
        negative_terms,
        excluded_interpretations,
    )
    if not scope:
        quality_warnings.append("missing_scope")
    if not boundary_notes and not disambiguation_hints:
        quality_warnings.append("missing_boundary_or_disambiguation")
    if not primary_interpretation:
        quality_warnings.append("missing_primary_interpretation")
    if removed_excluded_sub_question:
        quality_warnings.append("excluded_interpretation_removed_from_sub_questions")

    profile = {
        "artifact_version": "topic_search_topic_profile.v1",
        "stage": "stage-02",
        "topic": topic,
        "generated_at": generated_at,
        "generation": {
            "mode": "llm",
            "fallback_reason": "",
            "llm_required": True,
        },
        "topic_goal": topic_goal,
        "scope": scope,
        "boundary_notes": boundary_notes,
        "primary_interpretation": primary_interpretation,
        "excluded_interpretations": excluded_interpretations,
        "disambiguation_hints": disambiguation_hints,
        "negative_terms": negative_terms,
        "sub_questions": sub_questions,
        "priorities": _sanitize_string_list(raw_profile.get("priorities")),
        "review_hints": _sanitize_string_list(raw_profile.get("review_hints")),
        "quality_warnings": _merge_string_lists(quality_warnings),
    }
    if not any(
        profile[key]
        for key in (
            "scope",
            "boundary_notes",
            "disambiguation_hints",
            "negative_terms",
            "sub_questions",
            "priorities",
            "review_hints",
        )
    ) and profile["topic_goal"] == topic:
        return None
    return profile


def collect_stage03(
    *,
    stage03_dir: Path,
    topic: str,
    topic_profile: dict[str, Any],
    max_results_per_query: int,
    year_min: int,
    generated_at: str,
    allow_deterministic_fallback: bool = False,
) -> dict[str, Any]:
    stage03_dir.mkdir(parents=True, exist_ok=True)
    plan, fallback_reason = _build_llm_stage03_plan(
        topic=topic,
        topic_profile=topic_profile,
        max_results_per_query=max_results_per_query,
        year_min=year_min,
        generated_at=generated_at,
    )
    if plan is None:
        plan, queries = _build_deterministic_stage03_plan(
            topic=topic,
            topic_profile=topic_profile,
            max_results_per_query=max_results_per_query,
            year_min=year_min,
            generated_at=generated_at,
            fallback_reason=fallback_reason,
        )
        mode = "deterministic_fallback"
    else:
        queries = _extract_plan_queries(plan)
        mode = "llm"
        fallback_reason = ""

    plan["topic_profile_ref"] = "../stage-02/topic_profile.md"
    plan["topic_profile"] = _stage03_topic_profile_summary(topic_profile)
    sources = _build_stage03_sources(topic, generated_at, plan)
    search_intents = _build_stage03_search_intents(
        topic=topic,
        topic_profile=topic_profile,
        plan=plan,
        queries=queries,
        generated_at=generated_at,
    )
    queries_artifact = {
        "artifact_version": "topic_search_queries.v1",
        "stage": "stage-03",
        "topic": topic,
        "queries": queries,
        "year_min": int(plan.get("filters", {}).get("min_year", year_min)) if isinstance(plan.get("filters"), dict) else year_min,
        "max_results_per_query": max_results_per_query,
        "query_generation": {
            "mode": mode,
            "fallback_reason": fallback_reason,
        },
        "quality_warnings": _merge_string_lists(_sanitize_string_list(plan.get("quality_warnings"))),
    }

    _write_yaml(stage03_dir / "search_plan.yaml", plan)
    _write_json(stage03_dir / "sources.json", sources)
    _write_json(stage03_dir / "search_intents.json", search_intents)
    _write_json(stage03_dir / "queries.json", queries_artifact)

    return {
        "queries": queries,
        "search_intents": search_intents,
        "query_planning": {
            "mode": mode,
            "fallback_reason": fallback_reason,
            "openai_compatible_env_present": _openai_env_config()[0] is not None,
        },
    }


def _openai_env_config() -> tuple[dict[str, str] | None, str]:
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    model = os.environ.get("OPENAI_MODEL", "").strip()
    if not api_key or not model:
        return None, "missing_openai_env"
    return {
        "api_key": api_key,
        "base_url": (os.environ.get("OPENAI_BASE_URL", "").strip() or DEFAULT_OPENAI_BASE_URL).rstrip("/"),
        "model": model,
    }, ""


def _build_llm_stage03_plan(
    *,
    topic: str,
    topic_profile: dict[str, Any],
    max_results_per_query: int,
    year_min: int,
    generated_at: str,
) -> tuple[dict[str, Any] | None, str]:
    config, missing_reason = _openai_env_config()
    if config is None:
        return None, missing_reason
    try:
        payload = _request_openai_compatible_plan(config, topic, topic_profile, year_min)
    except Exception:  # noqa: BLE001
        return None, "llm_call_failed"
    plan = _validate_llm_plan(
        payload,
        topic=topic,
        topic_profile=topic_profile,
        max_results_per_query=max_results_per_query,
        year_min=year_min,
        generated_at=generated_at,
    )
    if plan is None:
        return None, "invalid_llm_plan"
    return plan, ""


def _request_openai_compatible_plan(
    config: dict[str, str],
    topic: str,
    topic_profile: dict[str, Any],
    year_min: int,
) -> dict[str, Any]:
    endpoint = f"{config['base_url']}/chat/completions"
    profile_summary = _stage03_topic_profile_summary(topic_profile)
    prompt = (
        "Create a Stage 03 literature-search query plan as JSON. "
        "Return only a JSON object with search_strategies and filters. "
        "Each search strategy must include name, intent_type, 1-5 short keyword queries, "
        "preferred_sources chosen only from openalex, semantic_scholar, arxiv, crossref, dblp, doaj, pubmed, optional "
        "negative_terms, and optional disambiguation. Use the topic profile as the semantic "
        "boundary for scope, disambiguation, negative terms, sub-questions, priorities, and "
        "review hints. Do not emit provider URLs, executable requests, or bibliographic citations."
    )
    body = {
        "model": config["model"],
        "messages": [
            {"role": "system", "content": "You design auditable literature retrieval strategies."},
            {
                "role": "user",
                "content": (
                    f"{prompt}\nTopic: {topic}\nMinimum year: {year_min}\n"
                    f"Topic profile: {json.dumps(profile_summary, ensure_ascii=True)}"
                ),
            },
        ],
        "temperature": 0.1,
        "response_format": {"type": "json_object"},
    }
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {config['api_key']}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    timeout = float(os.environ.get("OPENAI_TIMEOUT_SECONDS", str(DEFAULT_OPENAI_TIMEOUT_SECONDS)))
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = json.loads(response.read().decode("utf-8"))
    content = raw.get("choices", [{}])[0].get("message", {}).get("content")
    if isinstance(content, dict):
        return content
    if not isinstance(content, str) or not content.strip():
        return {}
    try:
        loaded = json.loads(content)
    except json.JSONDecodeError:
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _validate_llm_plan(
    payload: dict[str, Any],
    *,
    topic: str,
    topic_profile: dict[str, Any],
    max_results_per_query: int,
    year_min: int,
    generated_at: str,
) -> dict[str, Any] | None:
    raw_plan: dict[str, Any] = payload
    raw_yaml = payload.get("search_plan_yaml")
    if isinstance(raw_yaml, str) and raw_yaml.strip().startswith("{"):
        try:
            maybe_plan = json.loads(raw_yaml)
        except json.JSONDecodeError:
            maybe_plan = None
        if isinstance(maybe_plan, dict):
            raw_plan = maybe_plan

    raw_strategies = raw_plan.get("search_strategies")
    if not isinstance(raw_strategies, list) or not raw_strategies:
        return None

    profile_negative_terms = _stage03_boundary_terms(topic, topic_profile)
    profile_disambiguation_hints = _sanitize_string_list(topic_profile.get("disambiguation_hints"))
    quality_warnings = _merge_string_lists(_sanitize_string_list(topic_profile.get("quality_warnings")))
    strategies: list[dict[str, Any]] = []
    seen_queries: set[str] = set()
    for index, item in enumerate(raw_strategies, start=1):
        if not isinstance(item, dict):
            continue
        negative_terms = _sanitize_string_list(item.get("negative_terms"))
        combined_negative_terms = _merge_string_lists(profile_negative_terms, negative_terms)
        if _strategy_conflicts_with_boundary(item, combined_negative_terms):
            quality_warnings.append("boundary_conflict_strategy_dropped")
            continue
        queries = _sanitize_queries(_raw_strategy_queries(item), combined_negative_terms, seen_queries)
        if not queries:
            quality_warnings.append("boundary_conflict_strategy_dropped")
            continue
        strategy: dict[str, Any] = {
            "name": _safe_strategy_name(str(item.get("name") or f"strategy_{index}")),
            "intent_type": _safe_text(str(item.get("intent_type") or "topic_decomposition"), limit=64),
            "queries": queries,
            "sources": list(_active_search_provider_ids()),
            "preferred_sources": _sanitize_preferred_sources(item.get("preferred_sources")),
            "max_results_per_query": max_results_per_query,
        }
        if combined_negative_terms:
            strategy["negative_terms"] = combined_negative_terms
        disambiguation = _safe_text(str(item.get("disambiguation") or ""), limit=240)
        if disambiguation:
            strategy["disambiguation"] = disambiguation
        elif profile_disambiguation_hints:
            strategy["disambiguation"] = "; ".join(profile_disambiguation_hints[:3])
        strategies.append(strategy)

    if not strategies:
        return None

    strategies, lower_bound_warnings = _apply_stage03_lower_bounds(
        strategies,
        topic=topic,
        topic_profile=topic_profile,
        max_results_per_query=max_results_per_query,
        seen_queries=seen_queries,
    )
    quality_warnings = _merge_string_lists(quality_warnings, lower_bound_warnings)

    filters = raw_plan.get("filters") if isinstance(raw_plan.get("filters"), dict) else {}
    try:
        min_year = int(filters.get("min_year", year_min))
    except (TypeError, ValueError):
        min_year = year_min
    return {
        "artifact_version": "topic_search_search_plan.v1",
        "stage": "stage-03",
        "topic": topic,
        "generated": generated_at,
        "query_generation": {
            "implementation_source": "OpenAI-compatible Stage 03 planner",
            "mode": "llm",
            "llm_required": True,
            "fallback_reason": "",
            "reference_plan_labels": list(REFERENCE_PLAN_LABELS),
            "reference_labels_are_active_stage04_providers": False,
            "quality_warnings": _merge_string_lists(quality_warnings),
        },
        "search_strategies": strategies,
        "filters": {"min_year": min_year},
        "quality_warnings": _merge_string_lists(quality_warnings),
        "out_of_scope_guardrails": _guardrails(),
    }


def _build_deterministic_stage03_plan(
    *,
    topic: str,
    topic_profile: dict[str, Any],
    max_results_per_query: int,
    year_min: int,
    generated_at: str,
    fallback_reason: str,
) -> tuple[dict[str, Any], list[str]]:
    profile_negative_terms = _sanitize_string_list(topic_profile.get("negative_terms"))
    profile_disambiguation_terms = _sanitize_string_list(topic_profile.get("disambiguation_hints"))
    negative_terms = _merge_string_lists(profile_negative_terms)
    disambiguation_terms = _merge_string_lists(profile_disambiguation_terms)
    queries = _build_stage03_queries(topic)
    quality_warnings = _merge_string_lists(_sanitize_string_list(topic_profile.get("quality_warnings")))
    seen_queries = {query.casefold() for query in queries}
    base_strategy = {
        "name": "keyword_core",
        "queries": queries[:5],
        "sources": list(_active_search_provider_ids()),
        "max_results_per_query": max_results_per_query,
        "negative_terms": negative_terms,
        "disambiguation": disambiguation_terms,
    }
    breadth_strategy = {
        "name": "breadth_variants",
        "queries": queries[5:] or queries[:3],
        "sources": list(_active_search_provider_ids()),
        "max_results_per_query": max_results_per_query,
        "negative_terms": negative_terms,
        "disambiguation": disambiguation_terms,
    }
    strategies, lower_bound_warnings = _apply_stage03_lower_bounds(
        [base_strategy, breadth_strategy],
        topic=topic,
        topic_profile=topic_profile,
        max_results_per_query=max_results_per_query,
        seen_queries=seen_queries,
    )
    quality_warnings = _merge_string_lists(quality_warnings, lower_bound_warnings)
    plan = {
        "artifact_version": "topic_search_search_plan.v1",
        "stage": "stage-03",
        "topic": topic,
        "generated": generated_at,
        "query_generation": {
            "implementation_source": "AutoResearchClaw-derived fallback query planning",
            "mode": "deterministic_fallback",
            "llm_required": False,
            "fallback_reason": fallback_reason,
            "reference_plan_labels": list(REFERENCE_PLAN_LABELS),
            "reference_labels_are_active_stage04_providers": False,
            "quality_warnings": _merge_string_lists(quality_warnings),
        },
        "negative_terms": negative_terms,
        "disambiguation": disambiguation_terms,
        "search_strategies": strategies,
        "filters": {"min_year": year_min},
        "quality_warnings": _merge_string_lists(quality_warnings),
        "out_of_scope_guardrails": _guardrails(),
    }
    return plan, _extract_plan_queries(plan)


def _merge_string_lists(*lists: list[str]) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for values in lists:
        for value in values:
            key = value.casefold()
            if key in seen:
                continue
            seen.add(key)
            merged.append(value)
    return merged


def _stage03_topic_profile_summary(topic_profile: dict[str, Any]) -> dict[str, Any]:
    return {
        "topic_goal": _safe_text(str(topic_profile.get("topic_goal") or topic_profile.get("topic") or ""), limit=240),
        "scope": _sanitize_string_list(topic_profile.get("scope")),
        "boundary_notes": _sanitize_string_list(topic_profile.get("boundary_notes")),
        "disambiguation_hints": _sanitize_string_list(topic_profile.get("disambiguation_hints")),
        "negative_terms": _sanitize_string_list(topic_profile.get("negative_terms")),
        "sub_questions": _sanitize_string_list(topic_profile.get("sub_questions")),
        "priorities": _sanitize_string_list(topic_profile.get("priorities")),
        "review_hints": _sanitize_string_list(topic_profile.get("review_hints")),
    }


def _build_stage03_sources(topic: str, generated_at: str, plan: dict[str, Any]) -> dict[str, Any]:
    active_provider_ids = list(_active_search_provider_ids())
    reference_plan_labels = list(REFERENCE_PLAN_LABELS)
    preferred = {
        source
        for strategy in plan.get("search_strategies", [])
        if isinstance(strategy, dict)
        for source in strategy.get("preferred_sources", [])
        if source in active_provider_ids
    }
    return {
        "artifact_version": "topic_search_sources.v1",
        "stage": "stage-03",
        "generated": generated_at,
        "count": len(active_provider_ids) + len(reference_plan_labels),
        "active_provider_ids": active_provider_ids,
        "reference_plan_labels": reference_plan_labels,
        "provider_set_selection": {
            "active_provider_ids": active_provider_ids,
            "preferred_provider_ids": sorted(preferred),
            "reference_plan_labels": reference_plan_labels,
            "rejected_provider_ids": [],
            "selection_source": "registry-bounded_stage03_plan",
        },
        "provider_registry_snapshot": provider_registry.provider_registry_snapshot(generated_at=generated_at),
        "sources": [
            {
                "source_name": source_name,
                "stage04_provider": True,
                "status": "preferred" if source_name in preferred else "active",
                "query": topic,
            }
            for source_name in active_provider_ids
        ]
        + [
            {
                "source_name": source_name,
                "stage04_provider": False,
                "status": "reference_only",
                "query": topic,
            }
            for source_name in reference_plan_labels
        ],
    }


def _build_stage03_search_intents(
    *,
    topic: str,
    topic_profile: dict[str, Any],
    plan: dict[str, Any],
    queries: list[str],
    generated_at: str,
) -> dict[str, Any]:
    year_min = _stage03_year_min(plan)
    profile_negative_terms = _sanitize_string_list(topic_profile.get("negative_terms"))
    profile_disambiguation_hints = _sanitize_string_list(topic_profile.get("disambiguation_hints"))
    intent_rows: list[dict[str, Any]] = []
    intent_index = 1
    for strategy_index, strategy in enumerate(plan.get("search_strategies", []), start=1):
        if not isinstance(strategy, dict):
            continue
        strategy_ref = f"search_strategies[{strategy_index - 1}]"
        strategy_name = _safe_text(str(strategy.get("name") or f"strategy_{strategy_index}"), limit=64)
        strategy_negative_terms = _sanitize_string_list(strategy.get("negative_terms"))
        strategy_disambiguation = _sanitize_string_list([strategy.get("disambiguation")] if isinstance(strategy.get("disambiguation"), str) else [])
        for query in strategy.get("queries", []):
            if not isinstance(query, str):
                continue
            query_text = _safe_text(query, limit=160)
            if not query_text or query_text not in queries and query_text.casefold() not in {item.casefold() for item in queries}:
                continue
            filters = {"year_min": year_min} if year_min is not None else {}
            review_hints = _merge_string_lists(
                [f"strategy: {strategy_name}"],
                [f"topic_profile: {topic_profile.get('topic_goal') or topic}"],
                [f"negative_terms: {', '.join(strategy_negative_terms or profile_negative_terms)}"] if (strategy_negative_terms or profile_negative_terms) else [],
                [f"disambiguation: {hint}" for hint in (strategy_disambiguation or profile_disambiguation_hints)],
            )
            intent: dict[str, Any] = {
                "intent_id": f"intent_{intent_index:02d}",
                "intent_type": _safe_text(str(strategy.get("intent_type") or "topic_decomposition"), limit=64),
                "query_text": query_text,
                "filters": filters,
                "source_strategy_ref": strategy_ref,
                "rationale": _build_intent_rationale(topic, strategy_name, query_text),
            }
            if review_hints:
                intent["review_hints"] = review_hints
            intent_rows.append(intent)
            intent_index += 1
    return {
        "artifact_version": "topic_search_search_intents.v1",
        "stage": "stage-03",
        "topic": topic,
        "topic_profile_ref": "../stage-02/topic_profile.md",
        "generated": generated_at,
        "query_generation": plan.get("query_generation", {}),
        "filters": {"year_min": year_min} if year_min is not None else {},
        "intents": intent_rows,
        "quality_warnings": _merge_string_lists(_sanitize_string_list(plan.get("quality_warnings"))),
        "out_of_scope_guardrails": _guardrails(),
    }


def _extract_plan_queries(plan: dict[str, Any]) -> list[str]:
    seen: set[str] = set()
    queries: list[str] = []
    for strategy in plan.get("search_strategies", []):
        if not isinstance(strategy, dict):
            continue
        for query in strategy.get("queries", []):
            if not isinstance(query, str):
                continue
            key = query.casefold()
            if key not in seen:
                seen.add(key)
                queries.append(query)
    return queries


def _extract_search_intent_queries(search_intents: dict[str, Any] | None) -> list[str]:
    if not isinstance(search_intents, dict):
        return []
    seen: set[str] = set()
    queries: list[str] = []
    for intent in search_intents.get("intents", []):
        if not isinstance(intent, dict):
            continue
        query_text = intent.get("query_text")
        if not isinstance(query_text, str):
            continue
        normalized = _safe_text(query_text, limit=160)
        if not normalized:
            continue
        key = normalized.casefold()
        if key in seen:
            continue
        seen.add(key)
        queries.append(normalized)
    return queries


def _stage03_year_min(plan: dict[str, Any]) -> int | None:
    filters = plan.get("filters") if isinstance(plan.get("filters"), dict) else {}
    if not isinstance(filters, dict):
        return None
    try:
        return int(filters.get("min_year"))
    except (TypeError, ValueError):
        return None


def _build_intent_rationale(topic: str, strategy_name: str, query_text: str) -> str:
    rationale = f"Derived from {strategy_name} for topic '{topic}' using query '{query_text}'."
    return _safe_text(rationale, limit=240)


def _sanitize_queries(raw_queries: Any, negative_terms: list[str], seen_queries: set[str]) -> list[str]:
    if not isinstance(raw_queries, list):
        return []
    queries: list[str] = []
    for raw_query in raw_queries:
        if isinstance(raw_query, dict):
            raw_values = [value for value in raw_query.values() if isinstance(value, str)]
        elif isinstance(raw_query, str):
            raw_values = [raw_query]
        else:
            raw_values = []
        for value in raw_values:
            query = _safe_query(value, negative_terms)
            if not query:
                continue
            key = query.casefold()
            if key in seen_queries:
                continue
            seen_queries.add(key)
            queries.append(query)
    return queries


def _raw_strategy_queries(strategy: dict[str, Any]) -> Any:
    raw_queries = strategy.get("queries")
    if isinstance(raw_queries, list):
        return raw_queries
    return strategy.get("keyword_queries")


def _safe_query(value: str, negative_terms: list[str]) -> str:
    query = _safe_text(value, limit=160)
    if not query or "http://" in query.lower() or "https://" in query.lower():
        return ""
    query_folded = query.casefold()
    for term in negative_terms:
        if term.casefold() in query_folded:
            return ""
    if len(query) > 80:
        terms = _extract_terms(query)
        query = " ".join(terms[:8]) or query[:80]
    return query[:80].strip()


def _sanitize_preferred_sources(raw_sources: Any) -> list[str]:
    if not isinstance(raw_sources, list):
        return []
    sources: list[str] = []
    active_provider_ids = set(_active_search_provider_ids())
    for raw_source in raw_sources:
        if not isinstance(raw_source, str):
            continue
        source_name = re.sub(r"[^a-z0-9_]+", "_", raw_source.strip().casefold().replace("-", "_").replace(" ", "_")).strip("_")
        if source_name in active_provider_ids and source_name not in sources:
            sources.append(source_name)
    return sources


def _sanitize_string_list(raw_values: Any) -> list[str]:
    if not isinstance(raw_values, list):
        return []
    values: list[str] = []
    for raw_value in raw_values:
        if not isinstance(raw_value, str):
            continue
        value = _safe_text(raw_value, limit=200)
        if value and value.casefold() not in {existing.casefold() for existing in values}:
            values.append(value)
    return values


def _safe_strategy_name(value: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip()).strip("_")
    return name[:64] or "query_strategy"


def _safe_text(value: str, *, limit: int) -> str:
    text = " ".join(value.split()).strip().strip('"').strip("'")
    return text[:limit]


def collect_stage04(
    *,
    stage04_dir: Path,
    queries: list[str] | None,
    search_intents: dict[str, Any] | None,
    max_results_per_query: int,
    year_min: int,
    s2_api_key: str,
    inter_query_delay: float,
    generated_at: str,
) -> dict[str, Any]:
    stage04_dir.mkdir(parents=True, exist_ok=True)
    search_intent_queries = _extract_search_intent_queries(search_intents)
    stage04_queries = search_intent_queries or list(queries or [])
    query_source = "search_intents" if search_intent_queries else "queries"
    search_intents_contract_ref = "stage-03/search_intents.json" if search_intent_queries else ""
    active_provider_ids = _active_search_provider_ids()
    search_result = search_papers_multi_query(
        stage04_queries,
        limit_per_query=max_results_per_query,
        sources=active_provider_ids,
        year_min=year_min,
        s2_api_key=s2_api_key,
        inter_query_delay=inter_query_delay,
    )
    provider_executions = _build_provider_executions(
        provider_ids=active_provider_ids,
        search_result=search_result,
        query_source=query_source,
        search_intents_contract_ref=search_intents_contract_ref,
        search_intents=search_intents,
        queries=stage04_queries,
        year_min=year_min,
        max_results_per_query=max_results_per_query,
    )
    provider_execution_tasks = _build_provider_execution_tasks(
        provider_ids=active_provider_ids,
        search_result=search_result,
        query_source=query_source,
        search_intents_contract_ref=search_intents_contract_ref,
        search_intents=search_intents,
        queries=stage04_queries,
        year_min=year_min,
        max_results_per_query=max_results_per_query,
    )
    provider_surface_audit = _build_provider_surface_audit(
        provider_ids=active_provider_ids,
        provider_executions=provider_executions,
        provider_execution_tasks=provider_execution_tasks,
    )
    provider_task_ids = {
        str(task.get("provider_id")): [
            str(candidate.get("provider_execution_id"))
            for candidate in provider_execution_tasks
            if candidate.get("provider_id") == task.get("provider_id")
        ]
        for task in provider_execution_tasks
    }
    for execution in provider_executions:
        execution["provider_execution_task_ids"] = provider_task_ids.get(str(execution.get("provider_id")), [])
    match_ledger = _build_match_ledger(search_result, provider_execution_tasks)
    if not match_ledger:
        match_ledger = _build_fallback_match_ledger(search_result, provider_execution_tasks)
    primary_hit_by_candidate = _primary_hit_by_candidate(search_result, match_ledger)

    candidates = []
    for paper in search_result.papers:
        candidate = paper.to_dict()
        candidate["source_name"] = paper.source
        provenance = _build_candidate_match_provenance(paper, primary_hit_by_candidate)
        candidate["match_provenance"] = provenance
        candidates.append(candidate)

    references_bib = papers_to_bibtex(search_result.papers)
    search_meta = search_result.to_search_meta()
    search_meta.update(
        {
            "artifact_version": "topic_search_stage04_search_meta.v1",
            "stage": "stage-04",
            "generated_at": generated_at,
            "year_min": year_min,
            "max_results_per_query": max_results_per_query,
            "cache": {
                "enabled": False,
                "substituted_provider_results": False,
            },
            "placeholder_or_llm_generated_candidates": False,
            "search_intents_contract_ref": search_intents_contract_ref,
            "query_source": query_source,
            "provider_executions": provider_executions,
            "provider_execution_tasks": provider_execution_tasks,
            "provider_surface_audit": provider_surface_audit,
            "match_ledger_ref": "stage-04/match_ledger.jsonl",
            "match_ledger_summary": dict(Counter(str(row.get("hit_type") or "unknown") for row in match_ledger)),
        }
    )

    _write_jsonl(stage04_dir / "candidates.jsonl", candidates)
    _write_jsonl(stage04_dir / "match_ledger.jsonl", match_ledger)
    _write_text(stage04_dir / "references.bib", references_bib)
    _write_json(stage04_dir / "search_meta.json", search_meta)

    return {
        "stage_status": "complete" if search_result.status == "completed" else "blocked_or_failed",
        "candidates": candidates,
        "references_bib": references_bib,
        "search_meta": search_meta,
    }


def _build_provider_executions(
    *,
    provider_ids: tuple[str, ...],
    search_result: Any,
    query_source: str,
    search_intents_contract_ref: str,
    search_intents: dict[str, Any] | None,
    queries: list[str],
    year_min: int,
    max_results_per_query: int,
) -> list[dict[str, Any]]:
    status_by_provider = {
        str(status.get("source_name")): status
        for status in search_result.provider_statuses
        if isinstance(status, dict)
    }
    query_count_by_provider = _provider_query_counts(search_result, provider_ids)
    intent_count = len(search_intents.get("intents", [])) if isinstance(search_intents, dict) else 0
    executions: list[dict[str, Any]] = []
    for provider_id in provider_ids:
        provider = provider_registry.provider_entry(provider_id)
        provider_status = status_by_provider.get(provider_id, {})
        warnings = _sanitize_string_list(provider_status.get("warnings"))
        status = _structured_provider_status(str(provider_status.get("status") or "skipped"))
        status_compiled_query_summary = provider_status.get("compiled_query_summary") if isinstance(provider_status.get("compiled_query_summary"), dict) else {}
        query_count = query_count_by_provider.get(provider_id, int(provider_status.get("queries_attempted", 0) or len(queries)))
        if query_count == 0:
            query_count = int(provider_status.get("queries_attempted", 0) or len(queries))
        compiled_query_summary = {
            "query_source": query_source,
            "query_count": query_count,
            "intent_count": intent_count,
            "max_results_per_query": max_results_per_query,
            "sample_queries": queries[:3],
        }
        compiled_query_summary.update(status_compiled_query_summary)
        execution = {
            "provider_execution_id": _provider_execution_id(provider_id),
            "provider_id": provider.provider_id,
            "provider_name": provider.display_name,
            "intended_surface": provider.supported_search_surfaces[0],
            "executed_surface": _executed_surface(provider.provider_id, provider.supported_search_surfaces),
            "query_source": query_source,
            "search_intents_contract_ref": search_intents_contract_ref,
            "compiled_query_summary": compiled_query_summary,
            "applied_filters": {"year_min": year_min},
            "status": status,
            "provider_status": str(provider_status.get("status") or "skipped"),
            "returned_count": int(provider_status.get("returned_count", 0) or 0),
            "warnings": warnings,
        }
        if provider_status.get("rate_control"):
            execution["rate_control"] = provider_status["rate_control"]
        executions.append(execution)
    return executions


def _provider_query_counts(search_result: Any, provider_ids: tuple[str, ...]) -> dict[str, int]:
    counts = {provider_id: 0 for provider_id in provider_ids}
    for row in getattr(search_result, "provider_result_rows", []):
        if not isinstance(row, dict):
            continue
        provider_id = str(row.get("source_name") or "")
        if provider_id in counts:
            counts[provider_id] += 1
    return counts


def _build_provider_execution_tasks(
    *,
    provider_ids: tuple[str, ...],
    search_result: Any,
    query_source: str,
    search_intents_contract_ref: str,
    search_intents: dict[str, Any] | None,
    queries: list[str],
    year_min: int,
    max_results_per_query: int,
) -> list[dict[str, Any]]:
    intent_by_query = _intent_by_query(search_intents)
    rows = _stage04_provider_rows(search_result, provider_ids, queries)
    tasks: list[dict[str, Any]] = []
    provider_query_seen: dict[tuple[str, str], int] = {}
    for row_index, row in enumerate(rows, start=1):
        provider_id = str(row.get("source_name") or "")
        if provider_id not in provider_ids:
            continue
        provider = provider_registry.provider_entry(provider_id)
        query_text = _safe_text(str(row.get("query_text") or ""), limit=160)
        query_key = query_text.casefold()
        occurrence_key = (provider_id, query_key)
        provider_query_seen[occurrence_key] = provider_query_seen.get(occurrence_key, 0) + 1
        intent = intent_by_query.get(query_key, {})
        intent_id = str(intent.get("intent_id") or f"query_{provider_query_seen[occurrence_key]:02d}")
        source_strategy_ref = str(intent.get("source_strategy_ref") or "")
        warnings = _sanitize_string_list(row.get("warnings"))
        status = _structured_provider_status(str(row.get("status") or "skipped"))
        returned_count = int(row.get("returned_count", 0) or 0)
        papers = row.get("papers") if isinstance(row.get("papers"), list) else []
        row_compiled_query_summary = row.get("compiled_query_summary") if isinstance(row.get("compiled_query_summary"), dict) else {}
        compiled_query_summary = {
            "query_source": query_source,
            "query_text": query_text,
            "max_results_per_query": max_results_per_query,
        }
        compiled_query_summary.update(row_compiled_query_summary)
        task = {
            "provider_execution_id": _provider_task_execution_id(provider_id, intent_id, provider_query_seen[occurrence_key]),
            "provider_id": provider.provider_id,
            "provider_name": provider.display_name,
            "intent_id": intent_id,
            "query_text": query_text,
            "source_strategy_ref": source_strategy_ref,
            "intended_surface": provider.supported_search_surfaces[0],
            "executed_surface": _executed_surface(provider.provider_id, provider.supported_search_surfaces),
            "query_source": query_source,
            "search_intents_contract_ref": search_intents_contract_ref,
            "compiled_query_summary": compiled_query_summary,
            "applied_filters": {"year_min": year_min},
            "status": status,
            "provider_status": str(row.get("status") or "skipped"),
            "returned_count": returned_count,
            "normalized_count": len(papers),
            "warnings": warnings,
            "provider_result_row_index": row_index,
        }
        if row.get("rate_control"):
            task["rate_control"] = row["rate_control"]
        if row.get("matched_surface"):
            task["matched_surface"] = row.get("matched_surface")
        source_authority = _source_authority_evidence_from_papers(papers)
        if provider_id == "openalex" and source_authority["normalized_count"]:
            task["source_authority_evidence"] = source_authority
        bibliographic_authority = _bibliographic_authority_evidence_from_papers(papers)
        if provider_id == "crossref" and bibliographic_authority["normalized_count"]:
            task["bibliographic_authority_evidence"] = bibliographic_authority
        cs_bibliography = _cs_bibliography_evidence_from_papers(papers)
        if provider_id == "dblp" and cs_bibliography["normalized_count"]:
            task["cs_bibliography_evidence"] = cs_bibliography
        oa_metadata = _oa_metadata_evidence_from_papers(papers)
        if provider_id == "doaj" and oa_metadata["normalized_count"]:
            task["oa_metadata_evidence"] = oa_metadata
        biomedical_corpus = _biomedical_corpus_evidence_from_papers(papers)
        if provider_id == "pubmed" and biomedical_corpus["normalized_count"]:
            task["biomedical_corpus_evidence"] = biomedical_corpus
        tasks.append(task)
    return tasks


def _build_provider_surface_audit(
    *,
    provider_ids: tuple[str, ...],
    provider_executions: list[dict[str, Any]],
    provider_execution_tasks: list[dict[str, Any]],
) -> dict[str, Any]:
    execution_by_provider = {
        str(execution.get("provider_id") or ""): execution
        for execution in provider_executions
        if isinstance(execution, dict)
    }
    tasks_by_provider: dict[str, list[dict[str, Any]]] = {provider_id: [] for provider_id in provider_ids}
    for task in provider_execution_tasks:
        provider_id = str(task.get("provider_id") or "")
        if provider_id in tasks_by_provider:
            tasks_by_provider[provider_id].append(task)

    providers: list[dict[str, Any]] = []
    for provider_id in provider_ids:
        provider = provider_registry.provider_entry(provider_id)
        execution = execution_by_provider.get(provider_id, {})
        tasks = tasks_by_provider.get(provider_id, [])
        executed_surface = str(execution.get("executed_surface") or _executed_surface(provider_id, provider.supported_search_surfaces))
        coverage_judgment, surface_gap_state = _provider_surface_judgment(provider_id, execution, tasks)
        providers.append(
            {
                "provider_id": provider.provider_id,
                "provider_name": provider.display_name,
                "supported_search_surfaces": list(provider.supported_search_surfaces),
                "executed_surface": executed_surface,
                "coverage_judgment": coverage_judgment,
                "surface_gap_state": surface_gap_state,
                "status": str(execution.get("status") or "skipped"),
                "future_scope_surface_gaps": _future_scope_surface_gaps(provider_id, provider.supported_search_surfaces, executed_surface),
            }
        )
    return {
        "artifact_version": "topic_search_provider_surface_audit.v1",
        "providers": providers,
        "notes": [
            "supported_search_surfaces is registry/static capability.",
            "executed_surface is the actual Stage04 provider task surface recorded at runtime.",
            "candidate match_provenance remains a minimal trace pointer and does not claim provider-internal hit fields.",
        ],
    }


def _executed_surface(provider_id: str, supported_search_surfaces: tuple[str, ...]) -> str:
    if provider_id == "pubmed":
        return "esearch+esummary"
    if provider_id == "openalex":
        return "works.search"
    if provider_id == "semantic_scholar":
        return "paper.search"
    if provider_id == "arxiv":
        return "query_all"
    return supported_search_surfaces[0] if supported_search_surfaces else "unknown"


def _provider_surface_judgment(
    provider_id: str,
    execution: dict[str, Any],
    tasks: list[dict[str, Any]],
) -> tuple[str, str]:
    status = str(execution.get("status") or "")
    if status == "rate_limited" and provider_id in {"semantic_scholar", "arxiv"}:
        return "broad_main_surface_rate_limited", "rate_limited_provider_local_backoff"
    if provider_id == "dblp" and _tasks_use_provider_compaction(tasks):
        return "needs_provider_local_compaction", "provider_local_compaction_applied"
    if provider_id in {"dblp", "doaj", "pubmed"}:
        return "accepted_corpus_limited_main_surface", "corpus_limited"
    return "accepted_broad_main_surface", "none"


def _tasks_use_provider_compaction(tasks: list[dict[str, Any]]) -> bool:
    for task in tasks:
        summary = task.get("compiled_query_summary")
        if isinstance(summary, dict) and summary.get("rewrite_reason") == "dblp_keyword_compaction":
            return True
    return False


def _future_scope_surface_gaps(provider_id: str, supported_search_surfaces: tuple[str, ...], executed_surface: str) -> list[str]:
    if provider_id == "openalex" and "primary_location.source" in supported_search_surfaces:
        return ["source-authority enrichment is visible separately from broad works.search execution"]
    if provider_id == "semantic_scholar" and "title.search" in supported_search_surfaces:
        return ["title-specific execution remains future scope"]
    if provider_id == "arxiv" and "id_list" in supported_search_surfaces:
        return ["id_list lookup remains future scope for known arXiv IDs"]
    if provider_id == "pubmed":
        return ["PMC/full-text and Europe PMC remain future scope"]
    if executed_surface == "unknown":
        return ["executed surface is unknown"]
    return []


def _stage04_provider_rows(search_result: Any, provider_ids: tuple[str, ...], queries: list[str]) -> list[dict[str, Any]]:
    raw_rows = [
        row
        for row in getattr(search_result, "provider_result_rows", [])
        if isinstance(row, dict) and str(row.get("source_name") or "") in provider_ids
    ]
    status_by_provider = {
        str(status.get("source_name")): status
        for status in getattr(search_result, "provider_statuses", [])
        if isinstance(status, dict)
    }
    rows = list(raw_rows)
    existing_pairs = {
        (str(row.get("source_name") or ""), str(row.get("query_text") or "").casefold())
        for row in rows
    }
    for provider_id in provider_ids:
        status = status_by_provider.get(provider_id, {})
        attempted = int(status.get("queries_attempted", 0) or 1)
        selected_queries = queries[: max(1, min(attempted, len(queries) or 1))]
        if not selected_queries:
            selected_queries = [""]
        for query_text in selected_queries:
            pair = (provider_id, query_text.casefold())
            if pair in existing_pairs:
                continue
            row = {
                "source_name": provider_id,
                "query_text": query_text,
                "status": str(status.get("status") or "skipped"),
                "returned_count": int(status.get("returned_count", 0) or 0) if attempted <= 1 else 0,
                "warnings": list(status.get("warnings", [])) if isinstance(status.get("warnings"), list) else [],
                "papers": [],
            }
            if isinstance(status.get("compiled_query_summary"), dict):
                row["compiled_query_summary"] = status["compiled_query_summary"]
            if status.get("rate_control"):
                row["rate_control"] = status["rate_control"]
            rows.append(row)
            existing_pairs.add(pair)
    return rows


def _intent_by_query(search_intents: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    if not isinstance(search_intents, dict):
        return {}
    by_query: dict[str, dict[str, Any]] = {}
    for intent in search_intents.get("intents", []):
        if not isinstance(intent, dict):
            continue
        query_text = _safe_text(str(intent.get("query_text") or ""), limit=160)
        if query_text:
            by_query.setdefault(query_text.casefold(), intent)
    return by_query


def _build_match_ledger(search_result: Any, provider_execution_tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    task_by_row_index = {
        int(task.get("provider_result_row_index", 0)): task
        for task in provider_execution_tasks
        if isinstance(task.get("provider_result_row_index"), int)
    }
    final_by_key = {_paper_identity_key(paper): paper for paper in search_result.papers if _paper_identity_key(paper)}
    primary_seen: set[str] = set()
    rows = [
        row
        for row in getattr(search_result, "provider_result_rows", [])
        if isinstance(row, dict)
    ]
    ledger: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows, start=1):
        task = task_by_row_index.get(row_index)
        if not task:
            continue
        papers = row.get("papers") if isinstance(row.get("papers"), list) else []
        for provider_result_index, raw_paper in enumerate(papers, start=1):
            paper_row = _paper_row_dict(raw_paper)
            if not paper_row:
                continue
            candidate_key = _paper_identity_key_from_dict(paper_row)
            retained = final_by_key.get(candidate_key)
            retained_paper_id = str(getattr(retained, "paper_id", "")) if retained is not None else ""
            exact_retained_hit = bool(
                retained is not None
                and str(paper_row.get("paper_id") or "") == retained_paper_id
                and str(paper_row.get("source") or row.get("source_name") or "") == str(getattr(retained, "source", ""))
            )
            if retained is None:
                hit_type = "raw_hit_not_retained"
            elif exact_retained_hit and candidate_key not in primary_seen:
                hit_type = "primary"
                primary_seen.add(candidate_key)
            elif exact_retained_hit:
                hit_type = "additional_hit"
            else:
                hit_type = "dedup_loser"
            ledger.append(
                {
                    "candidate_key": candidate_key,
                    "hit_type": hit_type,
                    "provider_execution_id": task["provider_execution_id"],
                    "provider_id": task["provider_id"],
                    "intent_id": task.get("intent_id", ""),
                    "query_text": task.get("query_text", ""),
                    "source_strategy_ref": task.get("source_strategy_ref", ""),
                    "provider_result_index": provider_result_index,
                    "paper_id": str(paper_row.get("paper_id") or ""),
                    "retained_paper_id": retained_paper_id,
                    "title": _safe_text(str(paper_row.get("title") or ""), limit=240),
                    "doi": str(paper_row.get("doi") or ""),
                    "arxiv_id": str(paper_row.get("arxiv_id") or ""),
                    **_source_authority_ledger_field(paper_row),
                    **_bibliographic_authority_ledger_field(paper_row),
                    **_cs_bibliography_ledger_field(paper_row),
                    **_oa_metadata_ledger_field(paper_row),
                    **_biomedical_corpus_ledger_field(paper_row),
                }
            )
    return ledger


def _build_fallback_match_ledger(search_result: Any, provider_execution_tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    first_task_by_provider: dict[str, dict[str, Any]] = {}
    for task in provider_execution_tasks:
        provider_id = str(task.get("provider_id") or "")
        first_task_by_provider.setdefault(provider_id, task)
    next_index_by_provider: dict[str, int] = {}
    ledger: list[dict[str, Any]] = []
    for paper in search_result.papers:
        provider_id = str(getattr(paper, "source", ""))
        task = first_task_by_provider.get(provider_id)
        if not task:
            continue
        next_index = next_index_by_provider.get(provider_id, 0) + 1
        next_index_by_provider[provider_id] = next_index
        ledger.append(
            {
                "candidate_key": _paper_identity_key(paper),
                "hit_type": "primary",
                "provider_execution_id": task["provider_execution_id"],
                "provider_id": provider_id,
                "intent_id": task.get("intent_id", ""),
                "query_text": task.get("query_text", ""),
                "source_strategy_ref": task.get("source_strategy_ref", ""),
                "provider_result_index": next_index,
                "paper_id": str(getattr(paper, "paper_id", "")),
                "retained_paper_id": str(getattr(paper, "paper_id", "")),
                "title": _safe_text(str(getattr(paper, "title", "")), limit=240),
                "doi": str(getattr(paper, "doi", "")),
                "arxiv_id": str(getattr(paper, "arxiv_id", "")),
            }
        )
    return ledger


def _primary_hit_by_candidate(search_result: Any, match_ledger: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    primary_by_key: dict[str, dict[str, Any]] = {}
    for row in match_ledger:
        if row.get("hit_type") == "primary":
            primary_by_key.setdefault(str(row.get("candidate_key") or ""), row)
    for paper in search_result.papers:
        key = _paper_identity_key(paper)
        if key and key not in primary_by_key:
            for row in match_ledger:
                if row.get("candidate_key") == key:
                    primary_by_key[key] = row
                    break
    return primary_by_key


def _build_candidate_match_provenance(paper: Any, primary_hit_by_candidate: dict[str, dict[str, Any]]) -> dict[str, Any]:
    hit = primary_hit_by_candidate.get(_paper_identity_key(paper), {})
    provenance: dict[str, Any] = {
        "provider_execution_id": str(hit.get("provider_execution_id") or _provider_execution_id(str(paper.source))),
        "provider_result_index": int(hit.get("provider_result_index", 1) or 1),
    }
    native_identifiers: dict[str, str] = {}
    if getattr(paper, "doi", ""):
        native_identifiers["doi"] = str(paper.doi)
    if getattr(paper, "arxiv_id", ""):
        native_identifiers["arxiv_id"] = str(paper.arxiv_id)
    if native_identifiers:
        provenance["provider_native_identifiers"] = native_identifiers
    return provenance


def _source_authority_evidence_from_papers(papers: list[Any]) -> dict[str, Any]:
    sources: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for raw_paper in papers:
        paper_row = _paper_row_dict(raw_paper)
        source_metadata = paper_row.get("source_metadata") if isinstance(paper_row.get("source_metadata"), dict) else {}
        summary = _source_authority_summary(source_metadata)
        if not summary:
            continue
        key = (summary.get("source_id", ""), summary.get("display_name", ""))
        if key in seen:
            continue
        seen.add(key)
        sources.append(summary)
    return {
        "evidence_kind": "openalex_source_authority",
        "normalized_count": len(sources),
        "sources": sources,
    }


def _source_authority_summary(source_metadata: dict[str, Any]) -> dict[str, str]:
    if source_metadata.get("evidence_kind") != "openalex_source_authority":
        return {}
    summary = {
        "source_id": str(source_metadata.get("source_id") or ""),
        "display_name": str(source_metadata.get("display_name") or ""),
        "issn_l": str(source_metadata.get("issn_l") or ""),
        "type": str(source_metadata.get("type") or ""),
    }
    return summary if any(summary.values()) else {}


def _source_authority_ledger_field(paper_row: dict[str, Any]) -> dict[str, dict[str, str]]:
    source_metadata = paper_row.get("source_metadata") if isinstance(paper_row.get("source_metadata"), dict) else {}
    summary = _source_authority_summary(source_metadata)
    return {"source_authority_evidence": summary} if summary else {}


def _bibliographic_authority_evidence_from_papers(papers: list[Any]) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for raw_paper in papers:
        paper_row = _paper_row_dict(raw_paper)
        source_metadata = paper_row.get("source_metadata") if isinstance(paper_row.get("source_metadata"), dict) else {}
        summary = _bibliographic_authority_summary(source_metadata)
        if not summary:
            continue
        key = (str(summary.get("crossref_id", "")), str(summary.get("container_title", "")))
        if key in seen:
            continue
        seen.add(key)
        records.append(summary)
    return {
        "evidence_kind": "crossref_bibliographic_authority",
        "normalized_count": len(records),
        "records": records,
    }


def _bibliographic_authority_summary(source_metadata: dict[str, Any]) -> dict[str, Any]:
    if source_metadata.get("evidence_kind") != "crossref_bibliographic_authority":
        return {}
    summary: dict[str, Any] = {
        "crossref_id": str(source_metadata.get("crossref_id") or ""),
        "container_title": str(source_metadata.get("container_title") or ""),
        "issn": [str(item) for item in source_metadata.get("issn", []) if str(item)]
        if isinstance(source_metadata.get("issn"), list)
        else [],
        "publisher": str(source_metadata.get("publisher") or ""),
        "type": str(source_metadata.get("type") or ""),
    }
    return summary if any(value for value in summary.values()) else {}


def _bibliographic_authority_ledger_field(paper_row: dict[str, Any]) -> dict[str, dict[str, Any]]:
    source_metadata = paper_row.get("source_metadata") if isinstance(paper_row.get("source_metadata"), dict) else {}
    summary = _bibliographic_authority_summary(source_metadata)
    return {"bibliographic_authority_evidence": summary} if summary else {}


def _cs_bibliography_evidence_from_papers(papers: list[Any]) -> dict[str, Any]:
    records: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for raw_paper in papers:
        paper_row = _paper_row_dict(raw_paper)
        source_metadata = paper_row.get("source_metadata") if isinstance(paper_row.get("source_metadata"), dict) else {}
        summary = _cs_bibliography_summary(source_metadata)
        if not summary:
            continue
        key = (summary.get("dblp_key", ""), summary.get("venue", ""))
        if key in seen:
            continue
        seen.add(key)
        records.append(summary)
    return {
        "evidence_kind": "dblp_cs_bibliographic",
        "normalized_count": len(records),
        "records": records,
    }


def _cs_bibliography_summary(source_metadata: dict[str, Any]) -> dict[str, str]:
    if source_metadata.get("evidence_kind") != "dblp_cs_bibliographic":
        return {}
    summary = {
        "dblp_key": str(source_metadata.get("dblp_key") or ""),
        "venue": str(source_metadata.get("venue") or ""),
        "publication_type": str(source_metadata.get("publication_type") or ""),
        "ee": str(source_metadata.get("ee") or ""),
    }
    return summary if any(summary.values()) else {}


def _cs_bibliography_ledger_field(paper_row: dict[str, Any]) -> dict[str, dict[str, str]]:
    source_metadata = paper_row.get("source_metadata") if isinstance(paper_row.get("source_metadata"), dict) else {}
    summary = _cs_bibliography_summary(source_metadata)
    return {"cs_bibliography_evidence": summary} if summary else {}


def _oa_metadata_evidence_from_papers(papers: list[Any]) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for raw_paper in papers:
        paper_row = _paper_row_dict(raw_paper)
        source_metadata = paper_row.get("source_metadata") if isinstance(paper_row.get("source_metadata"), dict) else {}
        summary = _oa_metadata_summary(source_metadata)
        if not summary:
            continue
        key = (str(summary.get("doaj_id", "")), str(summary.get("journal_title", "")))
        if key in seen:
            continue
        seen.add(key)
        records.append(summary)
    return {
        "evidence_kind": "doaj_oa_metadata",
        "normalized_count": len(records),
        "records": records,
    }


def _oa_metadata_summary(source_metadata: dict[str, Any]) -> dict[str, Any]:
    if source_metadata.get("evidence_kind") != "doaj_oa_metadata":
        return {}
    summary: dict[str, Any] = {
        "doaj_id": str(source_metadata.get("doaj_id") or ""),
        "journal_title": str(source_metadata.get("journal_title") or ""),
        "issn": [str(item) for item in source_metadata.get("issn", []) if str(item)]
        if isinstance(source_metadata.get("issn"), list)
        else [],
        "publisher": str(source_metadata.get("publisher") or ""),
        "license": [str(item) for item in source_metadata.get("license", []) if str(item)]
        if isinstance(source_metadata.get("license"), list)
        else [],
        "language": [str(item) for item in source_metadata.get("language", []) if str(item)]
        if isinstance(source_metadata.get("language"), list)
        else [],
        "link_url": str(source_metadata.get("link_url") or ""),
        "link_type": str(source_metadata.get("link_type") or ""),
    }
    return summary if any(value for value in summary.values()) else {}


def _oa_metadata_ledger_field(paper_row: dict[str, Any]) -> dict[str, dict[str, Any]]:
    source_metadata = paper_row.get("source_metadata") if isinstance(paper_row.get("source_metadata"), dict) else {}
    summary = _oa_metadata_summary(source_metadata)
    return {"oa_metadata_evidence": summary} if summary else {}


def _biomedical_corpus_evidence_from_papers(papers: list[Any]) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for raw_paper in papers:
        paper_row = _paper_row_dict(raw_paper)
        source_metadata = paper_row.get("source_metadata") if isinstance(paper_row.get("source_metadata"), dict) else {}
        summary = _biomedical_corpus_summary(source_metadata)
        if not summary:
            continue
        key = (str(summary.get("pmid", "")), str(summary.get("journal", "")))
        if key in seen:
            continue
        seen.add(key)
        records.append(summary)
    return {
        "evidence_kind": "pubmed_biomedical_corpus",
        "normalized_count": len(records),
        "records": records,
    }


def _biomedical_corpus_summary(source_metadata: dict[str, Any]) -> dict[str, Any]:
    if source_metadata.get("evidence_kind") != "pubmed_biomedical_corpus":
        return {}
    summary: dict[str, Any] = {
        "pmid": str(source_metadata.get("pmid") or ""),
        "journal": str(source_metadata.get("journal") or ""),
        "source": str(source_metadata.get("source") or ""),
        "publication_types": [str(item) for item in source_metadata.get("publication_types", []) if str(item)]
        if isinstance(source_metadata.get("publication_types"), list)
        else [],
        "language": [str(item) for item in source_metadata.get("language", []) if str(item)]
        if isinstance(source_metadata.get("language"), list)
        else [],
        "has_abstract": bool(source_metadata.get("has_abstract")),
    }
    return summary if any(value for value in summary.values()) else {}


def _biomedical_corpus_ledger_field(paper_row: dict[str, Any]) -> dict[str, dict[str, Any]]:
    source_metadata = paper_row.get("source_metadata") if isinstance(paper_row.get("source_metadata"), dict) else {}
    summary = _biomedical_corpus_summary(source_metadata)
    return {"biomedical_corpus_evidence": summary} if summary else {}


def _paper_row_dict(raw_paper: Any) -> dict[str, Any]:
    if isinstance(raw_paper, dict):
        return raw_paper
    if hasattr(raw_paper, "to_dict"):
        maybe = raw_paper.to_dict()
        return maybe if isinstance(maybe, dict) else {}
    return {}


def _paper_identity_key(paper: Any) -> str:
    return _paper_identity_key_from_values(
        doi=str(getattr(paper, "doi", "") or ""),
        arxiv_id=str(getattr(paper, "arxiv_id", "") or ""),
        title=str(getattr(paper, "title", "") or ""),
        paper_id=str(getattr(paper, "paper_id", "") or ""),
    )


def _paper_identity_key_from_dict(paper: dict[str, Any]) -> str:
    return _paper_identity_key_from_values(
        doi=str(paper.get("doi") or ""),
        arxiv_id=str(paper.get("arxiv_id") or ""),
        title=str(paper.get("title") or ""),
        paper_id=str(paper.get("paper_id") or ""),
    )


def _paper_identity_key_from_values(*, doi: str, arxiv_id: str, title: str, paper_id: str) -> str:
    doi_key = doi.strip().casefold()
    if doi_key:
        return f"doi:{doi_key}"
    arxiv_key = arxiv_id.strip().casefold()
    if arxiv_key:
        return f"arxiv:{arxiv_key}"
    title_key = " ".join(re.sub(r"[^a-z0-9\s]", "", title.casefold()).split())
    if title_key:
        return f"title:{title_key}"
    return f"paper_id:{paper_id.strip().casefold()}"


def _provider_execution_id(provider_id: str) -> str:
    safe_provider_id = re.sub(r"[^a-z0-9_]+", "_", provider_id.strip().casefold().replace("-", "_")).strip("_")
    return f"provider_execution_{safe_provider_id or 'unknown'}"


def _provider_task_execution_id(provider_id: str, intent_id: str, occurrence: int) -> str:
    safe_provider_id = re.sub(r"[^a-z0-9_]+", "_", provider_id.strip().casefold().replace("-", "_")).strip("_")
    safe_intent_id = re.sub(r"[^a-z0-9_]+", "_", intent_id.strip().casefold().replace("-", "_")).strip("_")
    return f"provider_execution_{safe_provider_id or 'unknown'}_{safe_intent_id or 'query'}_{occurrence:02d}"


def _structured_provider_status(status: str) -> str:
    normalized = status.strip().casefold()
    status_map = {
        "completed": "completed",
        "skipped": "skipped",
        "unsupported_query": "unsupported",
        "unsupported": "unsupported",
        "blocked": "blocked",
        "provider_error": "blocked",
        "rate_limited": "rate_limited",
        "completed_no_results": "completed_no_results",
        "missing_api_key": "auth",
        "auth": "auth",
        "access": "access",
        "network_failed": "network",
        "network": "network",
        "timeout": "timeout",
        "live_client_not_configured": "needs_live_probe",
        "needs_live_probe": "needs_live_probe",
        "black_box_limitation": "blocked",
    }
    return status_map.get(normalized, "blocked")


def collect_stage05(
    *,
    stage05_dir: Path,
    references_bib: str,
    s2_api_key: str,
    inter_verify_delay: float,
    generated_at: str,
    upstream_stage_status: str,
) -> dict[str, Any]:
    stage05_dir.mkdir(parents=True, exist_ok=True)
    if references_bib.strip():
        report = verify_citations(
            references_bib,
            s2_api_key=s2_api_key,
            inter_verify_delay=inter_verify_delay,
        )
        report_payload = report.to_dict()
        stage_status = "complete"
        references_verified_bib = filter_verified_bibtex(
            references_bib,
            report,
            include_suspicious=True,
        )
    else:
        report_payload = {
            "summary": {
                "total": 0,
                "verified": 0,
                "suspicious": 0,
                "hallucinated": 0,
                "skipped": 0,
                "integrity_score": 1.0,
            },
            "results": [],
        }
        stage_status = "blocked_or_failed" if upstream_stage_status == "blocked_or_failed" else "complete"
        references_verified_bib = ""

    verification_report = {
        "artifact_version": "topic_search_verification_report.v1",
        "stage": "stage-05",
        "generated_at": generated_at,
        "status": stage_status,
        "include_suspicious": True,
        "summary": report_payload["summary"],
        "results": report_payload["results"],
    }
    _write_json(stage05_dir / "verification_report.json", verification_report)
    _write_text(stage05_dir / "references_verified.bib", references_verified_bib)

    return {
        "stage_status": stage_status,
        "verification_report": verification_report,
        "references_verified_bib": references_verified_bib,
    }


def _safe_topic_goal(value: str) -> tuple[str, list[str]]:
    warnings: list[str] = []
    text = " ".join(value.split()).strip().strip('"').strip("'")
    if len(text) > 600:
        shortened = text[:600].rstrip()
        boundary = max(shortened.rfind("."), shortened.rfind(";"), shortened.rfind(":"))
        if boundary >= 120:
            text = shortened[: boundary + 1].strip()
        else:
            space = shortened.rfind(" ")
            text = shortened[:space].strip() if space > 120 else shortened
        warnings.append("topic_goal_truncated_to_complete_boundary")
    repaired = _repair_dangling_topic_goal(text)
    if repaired != text:
        text = repaired
        warnings.append("dangling_topic_goal_repaired")
    return text or value[:240].strip() or "topic", warnings


def _repair_dangling_topic_goal(text: str) -> str:
    lowered = text.casefold()
    dangling_suffixes = (
        " of",
        " of a",
        " of an",
        " of the",
        " while",
        " while a",
        " while an",
        " while the",
        " with",
        " to",
        " for",
        " as",
    )
    if not lowered.endswith(dangling_suffixes):
        return text
    for marker in (" while ", " although ", " whereas "):
        marker_index = lowered.rfind(marker)
        if marker_index > 80:
            return text[:marker_index].rstrip(" ,;:")
    sentence_boundary = max(text.rfind("."), text.rfind(";"), text.rfind(":"))
    if sentence_boundary > 80:
        return text[: sentence_boundary + 1].strip()
    words = text.split()
    while words and " ".join(words).casefold().endswith(dangling_suffixes):
        words.pop()
    return " ".join(words).strip() or text


def _filter_boundary_conflicting_strings(
    values: list[str],
    negative_terms: list[str],
    excluded_interpretations: list[str],
) -> tuple[list[str], bool]:
    kept: list[str] = []
    removed = False
    boundary_terms = _merge_string_lists(negative_terms, excluded_interpretations)
    for value in values:
        if _text_conflicts_with_boundary(value, boundary_terms):
            removed = True
            continue
        kept.append(value)
    return kept, removed


def _boundary_notes(primary_interpretation: str, excluded_interpretations: list[str]) -> list[str]:
    notes: list[str] = []
    if primary_interpretation:
        notes.append(f"Primary interpretation: {primary_interpretation}.")
    if excluded_interpretations:
        notes.append(f"Excluded interpretations: {', '.join(excluded_interpretations)}.")
    return notes


def _stage03_boundary_terms(topic: str, topic_profile: dict[str, Any]) -> list[str]:
    return _merge_string_lists(
        _sanitize_string_list(topic_profile.get("negative_terms")),
        _sanitize_string_list(topic_profile.get("excluded_interpretations")),
        _boundary_terms_from_notes(_sanitize_string_list(topic_profile.get("boundary_notes"))),
    )


def _strategy_conflicts_with_boundary(strategy: dict[str, Any], boundary_terms: list[str]) -> bool:
    raw_parts: list[str] = []
    for key in ("name", "intent_type", "rationale"):
        value = strategy.get(key)
        if isinstance(value, str):
            raw_parts.append(value)
    return _text_conflicts_with_boundary(" ".join(raw_parts), boundary_terms)


def _text_conflicts_with_boundary(text: str, boundary_terms: list[str]) -> bool:
    folded = text.casefold()
    text_tokens = _extract_terms(folded)
    for term in boundary_terms:
        term_folded = term.casefold().strip()
        if not term_folded:
            continue
        if term_folded in folded:
            return True
        term_tokens = _extract_terms(term_folded)
        if term_tokens and _text_tokens_cover_boundary_term(text_tokens, term_tokens):
            return True
    return False


def _boundary_terms_from_notes(boundary_notes: list[str]) -> list[str]:
    terms: list[str] = []
    markers = (
        "excluded interpretations:",
        "excluded interpretation:",
        "exclude ",
        "excluding ",
        "not ",
    )
    for note in boundary_notes:
        folded = note.casefold()
        extracted = ""
        for marker in markers:
            marker_index = folded.find(marker)
            if marker_index >= 0:
                extracted = note[marker_index + len(marker) :]
                break
        if not extracted:
            continue
        pieces = re.split(r"[,;/]|\band\b|\bor\b", extracted)
        for piece in pieces:
            cleaned = piece.strip(" .:;()[]{}")
            if cleaned:
                terms.append(cleaned)
    return _merge_string_lists(terms)


def _text_tokens_cover_boundary_term(text_tokens: list[str], term_tokens: list[str]) -> bool:
    return all(any(_boundary_token_matches(text_token, term_token) for text_token in text_tokens) for term_token in term_tokens)


def _boundary_token_matches(text_token: str, term_token: str) -> bool:
    if text_token == term_token:
        return True
    if len(text_token) > 3 and len(term_token) > 3 and text_token.rstrip("s") == term_token.rstrip("s"):
        return True
    if term_token.endswith("ies") and text_token == f"{term_token[:-3]}y":
        return True
    if text_token.endswith("ies") and term_token == f"{text_token[:-3]}y":
        return True
    return False


def _build_stage03_queries(topic: str) -> list[str]:
    terms = _extract_terms(topic)
    if not terms:
        return [topic[:60]]
    primary = " ".join(terms[:6])
    short = " ".join(terms[:4])
    shifted = " ".join(terms[1:5]) if len(terms) > 4 else short
    candidates = [
        primary,
        f"{short} benchmark",
        f"{short} survey",
        shifted,
        f"{short} recent advances",
        f"{' '.join(terms[:3])} comparison",
        f"{' '.join(terms[:3])} state of the art",
    ]
    seen: set[str] = set()
    queries: list[str] = []
    for candidate in candidates:
        normalized = " ".join(candidate.split()).strip()
        key = normalized.casefold()
        if normalized and key not in seen:
            seen.add(key)
            queries.append(normalized[:80])
    return queries


def _apply_stage03_lower_bounds(
    strategies: list[dict[str, Any]],
    *,
    topic: str,
    topic_profile: dict[str, Any],
    max_results_per_query: int,
    seen_queries: set[str],
) -> tuple[list[dict[str, Any]], list[str]]:
    warnings: list[str] = []
    normalized = [_normalize_strategy_queries(strategy) for strategy in strategies]
    normalized = [strategy for strategy in normalized if strategy.get("queries")]
    while len(normalized) < MIN_STAGE03_STRATEGIES:
        strategy_index = len(normalized) + 1
        supplemental = _supplemental_stage03_queries(topic, topic_profile, seen_queries)
        if not supplemental:
            break
        normalized.append(
            {
                "name": f"minimum_diversity_{strategy_index}",
                "intent_type": "minimum_diversity_padding",
                "queries": supplemental[:3],
                "sources": list(_active_search_provider_ids()),
                "preferred_sources": [],
                "max_results_per_query": max_results_per_query,
            }
        )
        warnings.append("minimum_strategy_diversity_padded")
    total_queries = sum(len(strategy.get("queries", [])) for strategy in normalized)
    while normalized and total_queries < MIN_STAGE03_QUERIES:
        supplemental = _supplemental_stage03_queries(topic, topic_profile, seen_queries)
        if not supplemental:
            break
        target = min(normalized, key=lambda item: len(item.get("queries", [])))
        target_queries = target.setdefault("queries", [])
        if isinstance(target_queries, list):
            target_queries.append(supplemental[0])
            total_queries += 1
            warnings.append("minimum_query_diversity_padded")
        else:
            break
    for strategy in normalized:
        strategy["queries"] = [query for query in strategy.get("queries", [])[:5] if isinstance(query, str)]
    return normalized, _merge_string_lists(warnings)


def _normalize_strategy_queries(strategy: dict[str, Any]) -> dict[str, Any]:
    next_strategy = {**strategy}
    queries: list[str] = []
    seen: set[str] = set()
    for query in strategy.get("queries", []):
        if not isinstance(query, str):
            continue
        key = query.casefold()
        if key in seen:
            continue
        seen.add(key)
        queries.append(query)
    next_strategy["queries"] = queries[:5]
    return next_strategy


def _supplemental_stage03_queries(
    topic: str,
    topic_profile: dict[str, Any],
    seen_queries: set[str],
) -> list[str]:
    seeds = _merge_string_lists(
        [topic],
        _sanitize_string_list(topic_profile.get("scope")),
        _sanitize_string_list(topic_profile.get("disambiguation_hints")),
        _sanitize_string_list(topic_profile.get("sub_questions")),
        _sanitize_string_list(topic_profile.get("review_hints")),
    )
    suffixes = ("survey", "benchmark", "systems", "methods", "recent advances", "evaluation", "literature")
    supplemental: list[str] = []
    negative_terms = _stage03_boundary_terms(topic, topic_profile)
    for seed in seeds:
        terms = " ".join(_extract_terms(seed)[:5]).strip()
        if not terms:
            continue
        for suffix in suffixes:
            candidate = _safe_query(f"{terms} {suffix}", negative_terms)
            key = candidate.casefold()
            if candidate and key not in seen_queries:
                seen_queries.add(key)
                supplemental.append(candidate)
                if len(supplemental) >= 3:
                    return supplemental
    return supplemental


def _extract_terms(value: str) -> list[str]:
    stop_words = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "in",
        "is",
        "of",
        "on",
        "or",
        "the",
        "to",
        "using",
        "via",
        "with",
    }
    return [
        word
        for word in re.split(r"[^A-Za-z0-9]+", value)
        if len(word) > 1 and word.casefold() not in stop_words
    ]


def _guardrails() -> dict[str, bool]:
    return {
        "accepts_exact_title": False,
        "downloads_pdf_or_full_text": False,
        "uses_browser_or_deep_web": False,
        "uses_paid_or_institutional_library": False,
        "writes_paper_or_screening_output": False,
        "triggers_paper_ingest_canonical": False,
        "uses_placeholder_or_llm_generated_candidates": False,
    }


def _default_run_id(topic: str, generated_at: str) -> str:
    slug = _safe_run_id(topic)[:48] or "topic-search"
    compact = generated_at.replace("-", "").replace(":", "")
    return f"{slug}-{compact}"


def _safe_run_id(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip()).strip("-")
    return slug[:96] or "topic-search"


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(_yaml_lines(payload, 0)) + "\n", encoding="utf-8")


def _yaml_lines(value: Any, indent: int) -> list[str]:
    spaces = " " * indent
    if isinstance(value, dict):
        lines: list[str] = []
        for key, item in value.items():
            if isinstance(item, (dict, list)) and item:
                lines.append(f"{spaces}{key}:")
                lines.extend(_yaml_lines(item, indent + 2))
            elif item == []:
                lines.append(f"{spaces}{key}: []")
            elif item == {}:
                lines.append(f"{spaces}{key}: {{}}")
            else:
                lines.append(f"{spaces}{key}: {_yaml_scalar(item)}")
        return lines
    if isinstance(value, list):
        lines = []
        for item in value:
            if isinstance(item, (dict, list)):
                lines.append(f"{spaces}-")
                lines.extend(_yaml_lines(item, indent + 2))
            else:
                lines.append(f"{spaces}- {_yaml_scalar(item)}")
        return lines
    return [f"{spaces}{_yaml_scalar(value)}"]


def _yaml_scalar(value: Any) -> str:
    if value is True:
        return "true"
    if value is False:
        return "false"
    if value is None:
        return "null"
    if isinstance(value, (int, float)):
        return str(value)
    return json.dumps(str(value), ensure_ascii=False)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n" for row in rows)
    path.write_text(payload, encoding="utf-8")


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def _active_search_provider_ids() -> tuple[str, ...]:
    return provider_registry.active_provider_ids()
