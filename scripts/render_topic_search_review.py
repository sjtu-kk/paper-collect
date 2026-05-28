from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import Any


STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "by",
    "for",
    "from",
    "in",
    "into",
    "is",
    "of",
    "on",
    "or",
    "the",
    "to",
    "with",
}


STATUS_ORDER = {
    "hallucinated": 0,
    "suspicious": 1,
    "disabled": 2,
    "unknown": 2,
    "verified": 3,
    "skipped": 4,
}


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                rows.append(json.loads(text))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
    return rows


def _read_optional_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return _read_jsonl(path)


def _count_bib_entries(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.lstrip().startswith("@"):
                count += 1
    return count


def _tokens(text: str) -> list[str]:
    return [
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if len(token) > 1 and token not in STOP_WORDS
    ]


def _authors_preview(authors: Any) -> str:
    if not isinstance(authors, list):
        return ""
    names: list[str] = []
    for author in authors[:4]:
        if isinstance(author, dict):
            name = str(author.get("name", "")).strip()
        else:
            name = str(author).strip()
        if name:
            names.append(name)
    if len(authors) > 4:
        names.append(f"+{len(authors) - 4}")
    return ", ".join(names)


def _strip_html(text: str) -> str:
    return re.sub(r"<[^>]+>", "", text).replace("&amp;", "&")


def _identity(candidate: dict[str, Any]) -> str:
    doi = str(candidate.get("doi") or "").strip()
    arxiv_id = str(candidate.get("arxiv_id") or "").strip()
    if doi and arxiv_id:
        return f"doi:{doi} | arxiv:{arxiv_id}"
    if doi:
        return f"doi:{doi}"
    if arxiv_id:
        return f"arxiv:{arxiv_id}"
    paper_id = str(candidate.get("paper_id") or "").strip()
    return paper_id


def _lexical_hits(candidate: dict[str, Any], topic_terms: list[str]) -> dict[str, Any]:
    haystack = " ".join(
        [
            str(candidate.get("title") or ""),
            str(candidate.get("abstract") or ""),
            str(candidate.get("venue") or ""),
        ]
    ).lower()
    title = str(candidate.get("title") or "").lower()

    hits: list[str] = []
    title_hits: list[str] = []
    for term in topic_terms:
        pattern = rf"\b{re.escape(term)}\w*\b" if term == "auto" else rf"\b{re.escape(term)}\b"
        if re.search(pattern, haystack):
            hits.append(term)
        if re.search(pattern, title):
            title_hits.append(term)
    return {
        "hit_terms": hits,
        "title_hit_terms": title_hits,
        "hit_count": len(hits),
        "title_hit_count": len(title_hits),
    }


def _make_review_rows(
    candidates: list[dict[str, Any]],
    verification: dict[str, Any],
    topic: str,
) -> list[dict[str, Any]]:
    verification_status = str(verification.get("status") or "unknown")
    verification_by_key = {
        str(row.get("cite_key")): row for row in verification.get("results", [])
    }
    topic_terms = _tokens(topic)
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        cite_key = str(candidate.get("cite_key") or "")
        verified = verification_by_key.get(cite_key, {})
        status = str(verified.get("status") or ("disabled" if verification_status == "disabled" else "unknown"))
        lexical = _lexical_hits(candidate, topic_terms)
        citation_count = candidate.get("citation_count")
        try:
            citation_count_int = int(citation_count or 0)
        except (TypeError, ValueError):
            citation_count_int = 0

        flags: list[str] = []
        if status in {"hallucinated", "suspicious"}:
            flags.append("verify-review")
        if lexical["hit_count"] == 0:
            flags.append("low-topic-hint")
        if not str(candidate.get("abstract") or "").strip():
            flags.append("missing-abstract")
        if not str(candidate.get("doi") or "").strip() and not str(candidate.get("arxiv_id") or "").strip():
            flags.append("missing-strong-id")
        if citation_count_int >= 1000 and lexical["title_hit_count"] == 0:
            flags.append("high-citation-off-topic-risk")

        rows.append(
            {
                "cite_key": cite_key,
                "title": _strip_html(str(candidate.get("title") or verified.get("title") or "")),
                "authors": _authors_preview(candidate.get("authors")),
                "year": candidate.get("year") or "",
                "venue": str(candidate.get("venue") or ""),
                "source": str(candidate.get("source_name") or candidate.get("source") or ""),
                "paper_id": str(candidate.get("paper_id") or ""),
                "citation_count": citation_count_int,
                "doi": str(candidate.get("doi") or ""),
                "arxiv_id": str(candidate.get("arxiv_id") or ""),
                "identity": _identity(candidate),
                "url": str(candidate.get("url") or ""),
                "abstract": str(candidate.get("abstract") or ""),
                "status": status,
                "confidence": verified.get("confidence", ""),
                "method": str(verified.get("method") or ""),
                "details": str(verified.get("details") or ""),
                "topic_hits": lexical["hit_terms"],
                "title_topic_hits": lexical["title_hit_terms"],
                "topic_hit_count": lexical["hit_count"],
                "title_topic_hit_count": lexical["title_hit_count"],
                "flags": flags,
                "sort_weight": STATUS_ORDER.get(status, 2),
            }
        )
    rows.sort(
        key=lambda row: (
            row["sort_weight"],
            0 if "high-citation-off-topic-risk" in row["flags"] else 1,
            -int(row["citation_count"] or 0),
            -int(row["year"] or 0) if str(row["year"]).isdigit() else 0,
            row["title"].lower(),
        )
    )
    return rows


def _count_by(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(Counter(str(row.get(key) or "unknown") for row in rows))


def _provider_registry_snapshot(run_meta: dict[str, Any], run_root: Path) -> dict[str, Any]:
    stage03_artifacts = run_meta.get("stage_artifacts", {}).get("stage-03", {}) if isinstance(run_meta.get("stage_artifacts"), dict) else {}
    snapshot_ref = stage03_artifacts.get("sources_json", "stage-03/sources.json")
    snapshot_path = run_root / str(snapshot_ref)
    if snapshot_path.exists():
        snapshot = _read_json(snapshot_path)
    else:
        snapshot = {}
    registry_snapshot = snapshot.get("provider_registry_snapshot")
    if isinstance(registry_snapshot, dict):
        return registry_snapshot
    return {
        "artifact_version": "paper_collect_provider_registry_snapshot.v1",
        "generated_at": str(run_meta.get("generated_at") or ""),
        "active_provider_ids": [],
        "provider_registry_entries": [],
    }


def _provider_executions(search_meta: dict[str, Any]) -> list[dict[str, Any]]:
    executions = search_meta.get("provider_executions")
    if isinstance(executions, list):
        return [execution for execution in executions if isinstance(execution, dict)]
    return []


def _provider_execution_tasks(search_meta: dict[str, Any]) -> list[dict[str, Any]]:
    tasks = search_meta.get("provider_execution_tasks")
    if isinstance(tasks, list):
        return [task for task in tasks if isinstance(task, dict)]
    return []


def _disabled_search_meta(run_meta: dict[str, Any], generated_at: str) -> dict[str, Any]:
    stage_statuses = run_meta.get("stage_statuses", {}) if isinstance(run_meta.get("stage_statuses"), dict) else {}
    status = str(stage_statuses.get("stage-04") or "disabled")
    request = run_meta.get("request", {}) if isinstance(run_meta.get("request"), dict) else {}
    return {
        "artifact_version": "topic_search_stage04_search_meta.v1",
        "stage": "stage-04",
        "generated_at": generated_at,
        "status": status,
        "real_search": False,
        "queries_used": [],
        "year_min": request.get("year_min"),
        "max_results_per_query": request.get("max_results_per_query"),
        "total_candidates": 0,
        "bibtex_entries": 0,
        "provider_statuses": [],
        "provider_executions": [],
        "provider_execution_tasks": [],
        "provider_surface_audit": {
            "artifact_version": "topic_search_provider_surface_audit.v1",
            "providers": [],
            "notes": [],
        },
        "cache": {
            "enabled": False,
            "substituted_provider_results": False,
        },
        "placeholder_or_llm_generated_candidates": False,
        "query_source": "disabled",
        "search_intents_contract_ref": "",
    }


def _artifact_ref(path: Path, run_root: Path) -> str:
    return str(path.relative_to(run_root)).replace("\\", "/")


def _evidence_sections(
    *,
    run_root: Path,
    run_meta: dict[str, Any],
    search_meta_path: Path,
    candidates_path: Path,
    match_ledger_path: Path,
    references_path: Path,
    verification_path: Path,
    verified_bib_path: Path,
    verification_status: str,
) -> dict[str, Any]:
    search_artifacts = [
        path
        for path in (search_meta_path, candidates_path, match_ledger_path, references_path)
        if path.exists()
    ]
    verification_artifacts = [
        path
        for path in (verification_path, verified_bib_path)
        if path.exists()
    ]
    stage_statuses = run_meta.get("stage_statuses", {}) if isinstance(run_meta.get("stage_statuses"), dict) else {}
    search_status = "present" if search_artifacts else str(stage_statuses.get("stage-04") or "missing")
    verification_section_status = (
        "present"
        if verification_artifacts
        else verification_status
        if verification_status != "unknown"
        else str(stage_statuses.get("stage-05") or "missing")
    )
    return {
        "search": {
            "status": search_status,
            "stage": "stage-04",
            "artifact_refs": [_artifact_ref(path, run_root) for path in search_artifacts],
            "evidence_kind": "search_evidence",
        },
        "verification": {
            "status": verification_section_status,
            "stage": "stage-05",
            "artifact_refs": [_artifact_ref(path, run_root) for path in verification_artifacts],
            "evidence_kind": "verification_evidence",
        },
    }


def _gap_states(
    *,
    registry_payload: list[dict[str, Any]],
    provider_executions: list[dict[str, Any]],
    search_status: str,
    verification_status: str,
) -> dict[str, Any]:
    provider_execution_statuses = dict(Counter(str(execution.get("status") or "unknown") for execution in provider_executions))
    unsupported_providers = [
        entry["provider_id"]
        for entry in registry_payload
        if entry.get("registry_state") in {"unsupported", "deferred", "needs_live_probe"}
    ]
    unsupported_filters = {
        str(entry["provider_id"]): entry["supported_filters"]
        for entry in registry_payload
        if entry.get("registry_state") != "active"
    }
    black_box_gaps: list[str] = []
    if search_status == "disabled":
        black_box_gaps.append("search_disabled")
    if verification_status == "disabled":
        black_box_gaps.append("verification_disabled")
    if unsupported_providers:
        black_box_gaps.append("unsupported_or_deferred_providers")
    if provider_execution_statuses:
        black_box_gaps.extend(
            status
            for status in ("rate_limited", "auth", "access", "network", "timeout", "blocked")
            if status in provider_execution_statuses
        )
    return {
        "provider_execution_statuses": provider_execution_statuses,
        "unsupported_providers": unsupported_providers,
        "unsupported_filters": unsupported_filters,
        "black_box_gaps": black_box_gaps,
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


def _provider_source_authority(
    *,
    candidates: list[dict[str, Any]],
    provider_statuses: list[Any],
) -> dict[str, Any]:
    status_by_provider = {
        str(status.get("source_name")): status
        for status in provider_statuses
        if isinstance(status, dict)
    }
    openalex_status = status_by_provider.get("openalex", {})
    sources: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for candidate in candidates:
        if str(candidate.get("source_name") or candidate.get("source") or "") != "openalex":
            continue
        source_metadata = candidate.get("source_metadata") if isinstance(candidate.get("source_metadata"), dict) else {}
        summary = _source_authority_summary(source_metadata)
        if not summary:
            continue
        key = (summary.get("source_id", ""), summary.get("display_name", ""))
        if key in seen:
            continue
        seen.add(key)
        sources.append(summary)
    crossref_status = status_by_provider.get("crossref", {})
    records: list[dict[str, Any]] = []
    seen_records: set[tuple[str, str]] = set()
    for candidate in candidates:
        if str(candidate.get("source_name") or candidate.get("source") or "") != "crossref":
            continue
        source_metadata = candidate.get("source_metadata") if isinstance(candidate.get("source_metadata"), dict) else {}
        summary = _bibliographic_authority_summary(source_metadata)
        if not summary:
            continue
        key = (str(summary.get("crossref_id", "")), str(summary.get("container_title", "")))
        if key in seen_records:
            continue
        seen_records.add(key)
        records.append(summary)
    dblp_status = status_by_provider.get("dblp", {})
    dblp_records: list[dict[str, str]] = []
    seen_dblp: set[tuple[str, str]] = set()
    for candidate in candidates:
        if str(candidate.get("source_name") or candidate.get("source") or "") != "dblp":
            continue
        source_metadata = candidate.get("source_metadata") if isinstance(candidate.get("source_metadata"), dict) else {}
        summary = _cs_bibliography_summary(source_metadata)
        if not summary:
            continue
        key = (summary.get("dblp_key", ""), summary.get("venue", ""))
        if key in seen_dblp:
            continue
        seen_dblp.add(key)
        dblp_records.append(summary)
    doaj_status = status_by_provider.get("doaj", {})
    doaj_records: list[dict[str, Any]] = []
    seen_doaj: set[tuple[str, str]] = set()
    for candidate in candidates:
        if str(candidate.get("source_name") or candidate.get("source") or "") != "doaj":
            continue
        source_metadata = candidate.get("source_metadata") if isinstance(candidate.get("source_metadata"), dict) else {}
        summary = _oa_metadata_summary(source_metadata)
        if not summary:
            continue
        key = (str(summary.get("doaj_id", "")), str(summary.get("journal_title", "")))
        if key in seen_doaj:
            continue
        seen_doaj.add(key)
        doaj_records.append(summary)
    pubmed_status = status_by_provider.get("pubmed", {})
    pubmed_records: list[dict[str, Any]] = []
    seen_pubmed: set[tuple[str, str]] = set()
    for candidate in candidates:
        if str(candidate.get("source_name") or candidate.get("source") or "") != "pubmed":
            continue
        source_metadata = candidate.get("source_metadata") if isinstance(candidate.get("source_metadata"), dict) else {}
        summary = _biomedical_corpus_summary(source_metadata)
        if not summary:
            continue
        key = (str(summary.get("pmid", "")), str(summary.get("journal", "")))
        if key in seen_pubmed:
            continue
        seen_pubmed.add(key)
        pubmed_records.append(summary)
    return {
        "openalex": {
            "provider_id": "openalex",
            "target_effect": "source_authority_visibility",
            "boundary": "source metadata is authority/review evidence, not relevance ranking or full-text reachability",
            "returned_count": int(openalex_status.get("returned_count", 0) or 0) if openalex_status else 0,
            "normalized_count": len(sources),
            "sources": sources,
        },
        "crossref": {
            "provider_id": "crossref",
            "target_effect": "journal_proceedings_bibliographic_authority",
            "boundary": "container-title, ISSN, and publisher metadata are authority/review evidence, not full-text reachability",
            "returned_count": int(crossref_status.get("returned_count", 0) or 0) if crossref_status else 0,
            "normalized_count": len(records),
            "records": records,
        },
        "dblp": {
            "provider_id": "dblp",
            "target_effect": "cs_conference_bibliography_coverage",
            "boundary": "DBLP records are CS bibliography/conference metadata, not all-discipline authority or full-text reachability",
            "returned_count": int(dblp_status.get("returned_count", 0) or 0) if dblp_status else 0,
            "normalized_count": len(dblp_records),
            "records": dblp_records,
        },
        "doaj": {
            "provider_id": "doaj",
            "target_effect": "oa_journal_article_metadata_coverage",
            "boundary": "DOAJ records are OA journal/article metadata, not proof of acquired full text or PDF reachability",
            "returned_count": int(doaj_status.get("returned_count", 0) or 0) if doaj_status else 0,
            "normalized_count": len(doaj_records),
            "records": doaj_records,
        },
        "pubmed": {
            "provider_id": "pubmed",
            "target_effect": "biomedical_corpus_coverage",
            "boundary": "PubMed records are biomedical corpus metadata, not all-discipline venue authority or full-text reachability",
            "returned_count": int(pubmed_status.get("returned_count", 0) or 0) if pubmed_status else 0,
            "normalized_count": len(pubmed_records),
            "records": pubmed_records,
        },
    }


def _build_payload(run_root: Path) -> dict[str, Any]:
    run_meta = _read_json(run_root / "run_meta.json")
    queries = _read_json(run_root / "stage-03" / "queries.json")
    search_meta_path = run_root / "stage-04" / "search_meta.json"
    candidates_path = run_root / "stage-04" / "candidates.jsonl"
    match_ledger_path = run_root / "stage-04" / "match_ledger.jsonl"
    references_path = run_root / "stage-04" / "references.bib"
    search_meta = _read_json(search_meta_path) if search_meta_path.exists() else _disabled_search_meta(run_meta, str(run_meta.get("generated_at") or ""))
    candidates = _read_optional_jsonl(candidates_path)
    verification_path = run_root / "stage-05" / "verification_report.json"
    verified_bib_path = run_root / "stage-05" / "references_verified.bib"
    verification_status = str(run_meta.get("verification", {}).get("status") or "unknown")
    if verification_path.exists():
        verification = _read_json(verification_path)
        verified_bib_count = _count_bib_entries(verified_bib_path)
    else:
        verification = {
            "artifact_version": "topic_search_verification_report.v1",
            "generated_at": run_meta.get("generated_at", ""),
            "stage": "stage-05",
            "status": verification_status if verification_status != "unknown" else "disabled",
            "include_suspicious": False,
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
        verified_bib_count = 0

    topic = str(run_meta.get("request", {}).get("topic") or queries.get("topic") or "")
    rows = _make_review_rows(candidates, verification, topic)
    status_counts = _count_by(rows, "status")
    source_counts = _count_by(rows, "source")
    method_counts = _count_by(rows, "method")
    flag_counts = Counter(flag for row in rows for flag in row["flags"])
    provider_statuses = search_meta.get("provider_statuses", [])
    provider_returned_total = sum(
        int(provider.get("returned_count") or 0)
        for provider in provider_statuses
        if isinstance(provider, dict)
    )

    problem_rows = [
        row
        for row in rows
        if row["status"] in {"hallucinated", "suspicious"}
        or "high-citation-off-topic-risk" in row["flags"]
    ][:40]
    registry_snapshot = _provider_registry_snapshot(run_meta, run_root)
    registry_payload = list(registry_snapshot.get("provider_registry_entries", []))
    provider_executions = _provider_executions(search_meta)
    provider_execution_tasks = _provider_execution_tasks(search_meta)
    provider_surface_audit = search_meta.get("provider_surface_audit")
    if not isinstance(provider_surface_audit, dict):
        provider_surface_audit = {
            "artifact_version": "topic_search_provider_surface_audit.v1",
            "providers": [],
            "notes": [],
        }

    return {
        "run_root": str(run_root),
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "run_meta": run_meta,
        "queries": queries,
        "search_meta": search_meta,
        "provider_registry_snapshot": registry_snapshot,
        "provider_registry": registry_payload,
        "provider_executions": provider_executions,
        "provider_execution_tasks": provider_execution_tasks,
        "provider_surface_audit": provider_surface_audit,
        "evidence_sections": _evidence_sections(
            run_root=run_root,
            run_meta=run_meta,
            search_meta_path=search_meta_path,
            candidates_path=candidates_path,
            match_ledger_path=match_ledger_path,
            references_path=references_path,
            verification_path=verification_path,
            verified_bib_path=verified_bib_path,
            verification_status=str(verification.get("status") or "unknown"),
        ),
        "verification": {
            "artifact_version": verification.get("artifact_version"),
            "generated_at": verification.get("generated_at"),
            "status": verification.get("status"),
            "include_suspicious": verification.get("include_suspicious"),
            "summary": verification.get("summary", {}),
        },
        "review": {
            "rows": rows,
            "topic_terms": _tokens(topic),
            "status_counts": status_counts,
            "source_counts": source_counts,
            "method_counts": method_counts,
            "flag_counts": dict(flag_counts),
            "provider_returned_total": provider_returned_total,
            "verified_bib_count": verified_bib_count,
            "problem_rows": problem_rows,
            "provider_source_authority": _provider_source_authority(
                candidates=candidates,
                provider_statuses=provider_statuses,
            ),
            "provider_surface_audit": provider_surface_audit,
            "gap_states": _gap_states(
                registry_payload=registry_payload,
                provider_executions=provider_executions,
                search_status=str(search_meta.get("status") or "unknown"),
                verification_status=str(verification.get("status") or "unknown"),
            ),
        },
    }


def _authority_evidence_html(payload: dict[str, Any]) -> str:
    authority = payload.get("review", {}).get("provider_source_authority", {})
    cards: list[str] = []
    if isinstance(authority, dict):
        crossref = authority.get("crossref", {})
        if isinstance(crossref, dict) and (crossref.get("returned_count") or crossref.get("normalized_count")):
            records = crossref.get("records", [])
            record_cards = []
            if isinstance(records, list):
                for record in records[:6]:
                    if not isinstance(record, dict):
                        continue
                    record_cards.append(
                        "<div class=\"authority-card\">"
                        f"<h3>{escape(str(record.get('container_title') or record.get('crossref_id') or 'Crossref authority record'))}</h3>"
                        "<dl>"
                        f"<dt>DOI</dt><dd>{escape(str(record.get('crossref_id') or ''))}</dd>"
                        f"<dt>Publisher</dt><dd>{escape(str(record.get('publisher') or ''))}</dd>"
                        f"<dt>ISSN</dt><dd>{escape(', '.join(str(item) for item in record.get('issn', []) if str(item)) if isinstance(record.get('issn'), list) else '')}</dd>"
                        f"<dt>Type</dt><dd>{escape(str(record.get('type') or ''))}</dd>"
                        "</dl>"
                        "</div>"
                    )
            record_cards_html = "".join(record_cards) if record_cards else '<div class="empty">No Crossref authority records visible.</div>'
            cards.append(
                "<div class=\"panel section\">"
                "<div class=\"section-head\">"
                "<h2>Crossref authority evidence</h2>"
                f"<span>{int(crossref.get('returned_count', 0) or 0)} returned / {int(crossref.get('normalized_count', 0) or 0)} review-visible records</span>"
                "</div>"
                f"<div class=\"note-block\">{escape(str(crossref.get('boundary') or 'Crossref authority evidence is not full-text reachability.'))}</div>"
                f"<div class=\"authority-grid\" style=\"margin-top: 12px;\">{record_cards_html}</div>"
                "</div>"
            )
        openalex = authority.get("openalex", {})
        if isinstance(openalex, dict) and (openalex.get("returned_count") or openalex.get("normalized_count")):
            sources = openalex.get("sources", [])
            source_cards = []
            if isinstance(sources, list):
                for source in sources[:6]:
                    if not isinstance(source, dict):
                        continue
                    source_cards.append(
                        "<div class=\"authority-card\">"
                        f"<h3>{escape(str(source.get('display_name') or source.get('source_id') or 'OpenAlex source'))}</h3>"
                        "<dl>"
                        f"<dt>Source ID</dt><dd>{escape(str(source.get('source_id') or ''))}</dd>"
                        f"<dt>ISSN-L</dt><dd>{escape(str(source.get('issn_l') or ''))}</dd>"
                        f"<dt>Type</dt><dd>{escape(str(source.get('type') or ''))}</dd>"
                        "</dl>"
                        "</div>"
                    )
            source_cards_html = "".join(source_cards) if source_cards else '<div class="empty">No OpenAlex source authority records visible.</div>'
            cards.append(
                "<div class=\"panel section\">"
                "<div class=\"section-head\">"
                "<h2>OpenAlex source authority evidence</h2>"
                f"<span>{int(openalex.get('returned_count', 0) or 0)} returned / {int(openalex.get('normalized_count', 0) or 0)} review-visible sources</span>"
                "</div>"
                f"<div class=\"note-block\">{escape(str(openalex.get('boundary') or 'OpenAlex source evidence is not relevance ranking.'))}</div>"
                f"<div class=\"authority-grid\" style=\"margin-top: 12px;\">{source_cards_html}</div>"
                "</div>"
            )
        dblp = authority.get("dblp", {})
        if isinstance(dblp, dict) and (dblp.get("returned_count") or dblp.get("normalized_count")):
            records = dblp.get("records", [])
            record_cards = []
            if isinstance(records, list):
                for record in records[:6]:
                    if not isinstance(record, dict):
                        continue
                    record_cards.append(
                        "<div class=\"authority-card\">"
                        f"<h3>{escape(str(record.get('venue') or record.get('dblp_key') or 'DBLP record'))}</h3>"
                        "<dl>"
                        f"<dt>DBLP key</dt><dd>{escape(str(record.get('dblp_key') or ''))}</dd>"
                        f"<dt>Venue</dt><dd>{escape(str(record.get('venue') or ''))}</dd>"
                        f"<dt>Type</dt><dd>{escape(str(record.get('publication_type') or ''))}</dd>"
                        f"<dt>EE</dt><dd>{escape(str(record.get('ee') or ''))}</dd>"
                        "</dl>"
                        "</div>"
                    )
            record_cards_html = "".join(record_cards) if record_cards else '<div class="empty">No DBLP CS bibliography records visible.</div>'
            cards.append(
                "<div class=\"panel section\">"
                "<div class=\"section-head\">"
                "<h2>DBLP CS bibliography evidence</h2>"
                f"<span>{int(dblp.get('returned_count', 0) or 0)} returned / {int(dblp.get('normalized_count', 0) or 0)} review-visible records</span>"
                "</div>"
                f"<div class=\"note-block\">{escape(str(dblp.get('boundary') or 'DBLP records are CS bibliography evidence only.'))}</div>"
                f"<div class=\"authority-grid\" style=\"margin-top: 12px;\">{record_cards_html}</div>"
                "</div>"
            )
        doaj = authority.get("doaj", {})
        if isinstance(doaj, dict) and (doaj.get("returned_count") or doaj.get("normalized_count")):
            records = doaj.get("records", [])
            record_cards = []
            if isinstance(records, list):
                for record in records[:6]:
                    if not isinstance(record, dict):
                        continue
                    record_cards.append(
                        "<div class=\"authority-card\">"
                        f"<h3>{escape(str(record.get('journal_title') or record.get('doaj_id') or 'DOAJ record'))}</h3>"
                        "<dl>"
                        f"<dt>DOAJ ID</dt><dd>{escape(str(record.get('doaj_id') or ''))}</dd>"
                        f"<dt>ISSN</dt><dd>{escape(', '.join(str(item) for item in record.get('issn', []) if str(item)) if isinstance(record.get('issn'), list) else '')}</dd>"
                        f"<dt>Publisher</dt><dd>{escape(str(record.get('publisher') or ''))}</dd>"
                        f"<dt>License</dt><dd>{escape(', '.join(str(item) for item in record.get('license', []) if str(item)) if isinstance(record.get('license'), list) else '')}</dd>"
                        f"<dt>Language</dt><dd>{escape(', '.join(str(item) for item in record.get('language', []) if str(item)) if isinstance(record.get('language'), list) else '')}</dd>"
                        f"<dt>Link</dt><dd>{escape(str(record.get('link_url') or ''))}</dd>"
                        "</dl>"
                        "</div>"
                    )
            record_cards_html = "".join(record_cards) if record_cards else '<div class="empty">No DOAJ OA metadata records visible.</div>'
            cards.append(
                "<div class=\"panel section\">"
                "<div class=\"section-head\">"
                "<h2>DOAJ OA metadata evidence</h2>"
                f"<span>{int(doaj.get('returned_count', 0) or 0)} returned / {int(doaj.get('normalized_count', 0) or 0)} review-visible records</span>"
                "</div>"
                f"<div class=\"note-block\">{escape(str(doaj.get('boundary') or 'DOAJ metadata is not proof of acquired full text or PDF reachability.'))}</div>"
                f"<div class=\"authority-grid\" style=\"margin-top: 12px;\">{record_cards_html}</div>"
                "</div>"
            )
        pubmed = authority.get("pubmed", {})
        if isinstance(pubmed, dict) and (pubmed.get("returned_count") or pubmed.get("normalized_count")):
            records = pubmed.get("records", [])
            record_cards = []
            if isinstance(records, list):
                for record in records[:6]:
                    if not isinstance(record, dict):
                        continue
                    record_cards.append(
                        "<div class=\"authority-card\">"
                        f"<h3>{escape(str(record.get('journal') or record.get('pmid') or 'PubMed record'))}</h3>"
                        "<dl>"
                        f"<dt>PMID</dt><dd>{escape(str(record.get('pmid') or ''))}</dd>"
                        f"<dt>Journal</dt><dd>{escape(str(record.get('journal') or ''))}</dd>"
                        f"<dt>Source</dt><dd>{escape(str(record.get('source') or ''))}</dd>"
                        f"<dt>Type</dt><dd>{escape(', '.join(str(item) for item in record.get('publication_types', []) if str(item)) if isinstance(record.get('publication_types'), list) else '')}</dd>"
                        f"<dt>Language</dt><dd>{escape(', '.join(str(item) for item in record.get('language', []) if str(item)) if isinstance(record.get('language'), list) else '')}</dd>"
                        f"<dt>Abstract</dt><dd>{escape('present' if record.get('has_abstract') else 'not indicated')}</dd>"
                        "</dl>"
                        "</div>"
                    )
            record_cards_html = "".join(record_cards) if record_cards else '<div class="empty">No PubMed biomedical records visible.</div>'
            cards.append(
                "<div class=\"panel section\">"
                "<div class=\"section-head\">"
                "<h2>PubMed biomedical corpus evidence</h2>"
                f"<span>{int(pubmed.get('returned_count', 0) or 0)} returned / {int(pubmed.get('normalized_count', 0) or 0)} review-visible records</span>"
                "</div>"
                f"<div class=\"note-block\">{escape(str(pubmed.get('boundary') or 'PubMed records are biomedical corpus metadata only.'))}</div>"
                f"<div class=\"authority-grid\" style=\"margin-top: 12px;\">{record_cards_html}</div>"
                "</div>"
            )
    if not cards:
        return ""
    return "<section class=\"grid two-col\" style=\"margin-top: 18px;\">" + "".join(cards) + "</section>"


def _provider_surface_audit_html(payload: dict[str, Any]) -> str:
    audit = payload.get("provider_surface_audit")
    if not isinstance(audit, dict):
        return ""
    providers = audit.get("providers")
    if not isinstance(providers, list) or not providers:
        return ""
    rows: list[str] = []
    for provider in providers:
        if not isinstance(provider, dict):
            continue
        supported = provider.get("supported_search_surfaces", [])
        supported_text = ", ".join(str(item) for item in supported if str(item)) if isinstance(supported, list) else str(supported or "")
        gaps = provider.get("future_scope_surface_gaps", [])
        gap_text = "; ".join(str(item) for item in gaps if str(item)) if isinstance(gaps, list) else str(gaps or "")
        rows.append(
            "<tr>"
            f"<td>{escape(str(provider.get('provider_id') or ''))}</td>"
            f"<td>{escape(str(provider.get('executed_surface') or ''))}</td>"
            f"<td>{escape(supported_text)}</td>"
            f"<td>{escape(str(provider.get('coverage_judgment') or ''))}</td>"
            f"<td>{escape(str(provider.get('surface_gap_state') or ''))}</td>"
            f"<td>{escape(gap_text)}</td>"
            "</tr>"
        )
    if not rows:
        return ""
    return (
        "<section class=\"grid\" style=\"margin-top: 18px;\">"
        "<div class=\"panel section\">"
        "<div class=\"section-head\">"
        "<h2>Provider surface audit</h2>"
        "<span>supported_search_surfaces are registry/static; executed surface is the Stage04 runtime task surface</span>"
        "</div>"
        "<table class=\"mini-table\">"
        "<thead><tr>"
        "<th>Provider</th><th>executed surface</th><th>supported_search_surfaces</th><th>Judgment</th><th>Gap state</th><th>Future gaps</th>"
        "</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody>"
        "</table>"
        "<div class=\"note-block\" style=\"margin-top: 12px;\">Candidate match_provenance is a minimal trace pointer; this audit must not infer provider-internal hit fields.</div>"
        "</div>"
        "</section>"
    )


def _json_script(payload: dict[str, Any]) -> str:
    text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return text.replace("</", "<\\/")


def _render_html(payload: dict[str, Any]) -> str:
    title = f"Topic Search 复核 - {payload['run_meta'].get('run_id', 'run')}"
    data = _json_script(payload)
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(title)}</title>
  <style>
    :root {{
      --ink: #1f2528;
      --muted: #667176;
      --line: #d7dedc;
      --panel: #fbfcf8;
      --panel-strong: #f1f5eb;
      --paper: #f7f4ec;
      --green: #2f6f5e;
      --green-soft: #dce9df;
      --amber: #9a6a00;
      --amber-soft: #f1e2bb;
      --red: #9a3c33;
      --red-soft: #efd7d1;
      --blue: #315f7d;
      --blue-soft: #d9e5ec;
      --shadow: 0 12px 28px rgba(36, 46, 44, 0.08);
    }}

    * {{
      box-sizing: border-box;
    }}

    body {{
      margin: 0;
      color: var(--ink);
      background:
        linear-gradient(90deg, rgba(47, 111, 94, 0.08) 0 1px, transparent 1px 100%),
        linear-gradient(0deg, rgba(47, 111, 94, 0.06) 0 1px, transparent 1px 100%),
        var(--paper);
      background-size: 28px 28px;
      font-family: "Aptos", "Segoe UI", Tahoma, sans-serif;
      line-height: 1.45;
    }}

    button,
    input,
    select {{
      font: inherit;
    }}

    .shell {{
      width: min(1540px, calc(100vw - 40px));
      margin: 0 auto;
      padding: 24px 0 38px;
    }}

    header {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 20px;
      align-items: end;
      padding: 22px 0 18px;
      border-bottom: 2px solid var(--ink);
    }}

    .eyebrow {{
      margin: 0 0 8px;
      color: var(--green);
      font-family: "Cascadia Mono", Consolas, monospace;
      font-size: 12px;
      letter-spacing: 0;
      text-transform: uppercase;
    }}

    h1 {{
      margin: 0;
      max-width: 980px;
      font-family: Georgia, "Times New Roman", serif;
      font-size: clamp(32px, 4.2vw, 64px);
      line-height: 0.95;
      letter-spacing: 0;
    }}

    .header-meta {{
      min-width: 310px;
      display: grid;
      gap: 6px;
      padding: 12px 14px;
      border: 1px solid var(--line);
      background: rgba(251, 252, 248, 0.82);
    }}

    .meta-line {{
      display: flex;
      justify-content: space-between;
      gap: 16px;
      color: var(--muted);
      font-size: 13px;
    }}

    .meta-line strong {{
      color: var(--ink);
      font-weight: 650;
      text-align: right;
    }}

    .grid {{
      display: grid;
      gap: 16px;
      margin-top: 18px;
    }}

    .kpis {{
      grid-template-columns: repeat(7, minmax(132px, 1fr));
    }}

    .panel {{
      background: rgba(251, 252, 248, 0.94);
      border: 1px solid var(--line);
      box-shadow: var(--shadow);
    }}

    .kpi {{
      min-height: 108px;
      padding: 14px;
      display: flex;
      flex-direction: column;
      justify-content: space-between;
      border-top: 4px solid var(--green);
    }}

    .kpi.warn {{
      border-top-color: var(--amber);
    }}

    .kpi.bad {{
      border-top-color: var(--red);
    }}

    .kpi .label {{
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
      text-transform: uppercase;
    }}

    .kpi .value {{
      font-family: Georgia, "Times New Roman", serif;
      font-size: 34px;
      line-height: 1;
    }}

    .kpi .note {{
      color: var(--muted);
      font-size: 12px;
      overflow-wrap: anywhere;
    }}

    .two-col {{
      grid-template-columns: minmax(0, 1.1fr) minmax(380px, 0.9fr);
    }}

    .section {{
      padding: 16px;
    }}

    .section-head {{
      display: flex;
      align-items: baseline;
      justify-content: space-between;
      gap: 16px;
      margin-bottom: 12px;
      border-bottom: 1px solid var(--line);
      padding-bottom: 10px;
    }}

    h2 {{
      margin: 0;
      font-size: 17px;
      letter-spacing: 0;
    }}

    .section-head span {{
      color: var(--muted);
      font-size: 12px;
    }}

    .pipeline {{
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 12px;
    }}

    .stage {{
      padding: 14px;
      min-height: 132px;
      border: 1px solid var(--line);
      background: var(--panel-strong);
    }}

    .stage .tag {{
      display: inline-flex;
      height: 24px;
      align-items: center;
      padding: 0 8px;
      border: 1px solid var(--green);
      color: var(--green);
      font-size: 12px;
      font-weight: 700;
      text-transform: uppercase;
    }}

    .stage h3 {{
      margin: 10px 0 6px;
      font-size: 16px;
    }}

    .stage p {{
      margin: 0;
      color: var(--muted);
      font-size: 13px;
    }}

    .query-list {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 12px;
    }}

    .chip {{
      max-width: 100%;
      min-height: 28px;
      display: inline-flex;
      align-items: center;
      padding: 4px 8px;
      border: 1px solid var(--line);
      background: #fffdf7;
      color: var(--ink);
      font-size: 12px;
      overflow-wrap: anywhere;
    }}

    .provider-list {{
      display: grid;
      gap: 10px;
    }}

    .provider {{
      padding: 12px;
      border: 1px solid var(--line);
      background: #fffdf7;
    }}

    .provider-top {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 12px;
      align-items: start;
    }}

    .provider h3 {{
      margin: 0 0 4px;
      font-size: 15px;
    }}

    .provider .count {{
      font-family: Georgia, "Times New Roman", serif;
      font-size: 28px;
      line-height: 1;
      color: var(--green);
    }}

    .bar {{
      margin-top: 9px;
      height: 9px;
      background: #e7e1d5;
      overflow: hidden;
    }}

    .bar > span {{
      display: block;
      height: 100%;
      background: linear-gradient(90deg, var(--green), var(--blue));
    }}

    .warnings {{
      margin: 10px 0 0;
      padding: 0;
      display: grid;
      gap: 6px;
      list-style: none;
    }}

    .warnings li {{
      padding: 8px;
      background: var(--amber-soft);
      color: #4d3400;
      font-family: "Cascadia Mono", Consolas, monospace;
      font-size: 11px;
      overflow-wrap: anywhere;
    }}

    .split {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 16px;
    }}

    .mini-table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }}

    .mini-table th,
    .mini-table td {{
      border-bottom: 1px solid var(--line);
      padding: 7px 0;
      text-align: left;
      vertical-align: top;
    }}

    .mini-table th {{
      color: var(--muted);
      font-size: 11px;
      text-transform: uppercase;
    }}

    .controls {{
      display: grid;
      grid-template-columns: minmax(280px, 1fr) 150px 160px 150px 160px auto;
      gap: 10px;
      align-items: center;
      padding: 14px 16px;
      position: sticky;
      top: 0;
      z-index: 5;
      background: rgba(247, 244, 236, 0.96);
      backdrop-filter: blur(8px);
      border: 1px solid var(--line);
      box-shadow: var(--shadow);
    }}

    .control {{
      min-height: 38px;
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 0;
      background: #fffdf7;
      color: var(--ink);
      padding: 0 10px;
    }}

    .toggle {{
      min-height: 38px;
      border: 1px solid var(--line);
      background: #fffdf7;
      color: var(--ink);
      padding: 0 10px;
      cursor: pointer;
      font-weight: 700;
    }}

    .toggle.active {{
      border-color: var(--green);
      background: var(--green);
      color: #fff;
    }}

    .filter-help {{
      grid-column: 1 / -1;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.45;
    }}

    .table-wrap {{
      margin-top: 12px;
      border: 1px solid var(--line);
      background: rgba(251, 252, 248, 0.96);
      overflow: auto;
      max-height: 76vh;
      box-shadow: var(--shadow);
    }}

    table.results {{
      width: 100%;
      min-width: 1362px;
      border-collapse: collapse;
      table-layout: fixed;
      font-size: 13px;
    }}

    .results th {{
      position: sticky;
      top: 0;
      z-index: 2;
      background: var(--ink);
      color: #fff;
      text-align: left;
      padding: 9px 10px;
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0;
      border-right: 1px solid rgba(255, 255, 255, 0.16);
    }}

    .results td {{
      padding: 10px;
      border-bottom: 1px solid var(--line);
      border-right: 1px solid #edf1ee;
      vertical-align: top;
      background: rgba(255, 253, 247, 0.82);
      overflow-wrap: anywhere;
    }}

    .results tr:hover td {{
      background: #f3f7ed;
    }}

    .col-status {{ width: 126px; }}
    .col-title {{ width: 300px; }}
    .col-source {{ width: 100px; }}
    .col-year {{ width: 64px; }}
    .col-cites {{ width: 78px; }}
    .col-method {{ width: 120px; }}
    .col-topic {{ width: 100px; }}
    .col-id {{ width: 170px; }}
    .col-details {{ width: 240px; }}
    .col-link {{ width: 64px; }}

    .status {{
      display: inline-flex;
      min-height: 24px;
      align-items: center;
      padding: 3px 7px;
      border: 1px solid var(--line);
      font-size: 12px;
      font-weight: 800;
      text-transform: uppercase;
      white-space: nowrap;
    }}

    .status.verified {{
      color: var(--green);
      background: var(--green-soft);
      border-color: #aac8b6;
    }}

    .status.suspicious {{
      color: var(--amber);
      background: var(--amber-soft);
      border-color: #d9bd76;
    }}

    .status.hallucinated {{
      color: var(--red);
      background: var(--red-soft);
      border-color: #d5aaa3;
    }}

    .status.unknown,
    .status.skipped,
    .status.disabled {{
      color: var(--blue);
      background: var(--blue-soft);
      border-color: #aec7d5;
    }}

    .title-cell strong {{
      display: block;
      font-size: 14px;
      line-height: 1.32;
      margin-bottom: 5px;
    }}

    .subtle {{
      color: var(--muted);
      font-size: 12px;
    }}

    .flag-list {{
      display: flex;
      flex-wrap: wrap;
      gap: 5px;
      margin-top: 7px;
    }}

    .flag {{
      display: inline-flex;
      min-height: 21px;
      align-items: center;
      padding: 2px 6px;
      background: #ece7dc;
      color: #4f5657;
      font-size: 11px;
      border: 1px solid #ddd4c7;
    }}

    .flag.strong {{
      background: var(--red-soft);
      border-color: #d5aaa3;
      color: var(--red);
    }}

    .link {{
      color: var(--blue);
      font-weight: 700;
      text-decoration: none;
      border-bottom: 1px solid currentColor;
    }}

    .empty {{
      padding: 24px;
      color: var(--muted);
      text-align: center;
    }}

    .note-block {{
      padding: 12px;
      border-left: 4px solid var(--green);
      background: var(--green-soft);
      color: #29453a;
      font-size: 13px;
    }}

    .authority-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
      gap: 12px;
    }}

    .authority-card {{
      border: 1px solid var(--line);
      background: #fffdf7;
      padding: 12px;
    }}

    .authority-card h3 {{
      margin: 0 0 6px;
      font-size: 16px;
    }}

    .authority-card dl {{
      display: grid;
      grid-template-columns: 110px 1fr;
      gap: 6px 10px;
      margin: 10px 0 0;
      font-size: 13px;
    }}

    .authority-card dt {{
      color: var(--muted);
      font-weight: 700;
    }}

    .authority-card dd {{
      margin: 0;
      word-break: break-word;
    }}

    .problem-list {{
      display: grid;
      gap: 8px;
      max-height: 320px;
      overflow: auto;
    }}

    .problem {{
      padding: 10px;
      border: 1px solid var(--line);
      background: #fffdf7;
    }}

    .problem strong {{
      display: block;
      margin-bottom: 4px;
    }}

    @media (max-width: 1180px) {{
      .shell {{
        width: min(100vw - 24px, 980px);
      }}

      header,
      .two-col,
      .pipeline,
      .split {{
        grid-template-columns: 1fr;
      }}

      .kpis {{
        grid-template-columns: repeat(2, minmax(0, 1fr));
      }}

      .controls {{
        position: static;
        grid-template-columns: 1fr 1fr;
      }}
    }}

    @media (max-width: 680px) {{
      .shell {{
        width: min(100vw - 18px, 560px);
        padding-top: 12px;
      }}

      h1 {{
        font-size: 34px;
      }}

      .header-meta,
      .controls {{
        min-width: 0;
        grid-template-columns: 1fr;
      }}

      .kpis {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <div class="shell">
    <header>
      <div>
        <p class="eyebrow">paper-collect / topic-search-runtime / 人工复核页</p>
        <h1 id="pageTitle">Topic Search 复核</h1>
      </div>
      <div class="header-meta" id="headerMeta"></div>
    </header>

    <section class="grid kpis" id="kpis"></section>

    <section class="grid two-col">
      <div class="panel section">
        <div class="section-head">
          <h2>处理链路</h2>
          <span>验证论文真实性，不等于相关性排序</span>
        </div>
        <div class="pipeline" id="pipeline"></div>
        <div class="query-list" id="queryList"></div>
      </div>
      <div class="panel section">
        <div class="section-head">
          <h2>来源返回</h2>
          <span id="providerTotal"></span>
        </div>
        <div class="provider-list" id="providers"></div>
      </div>
    </section>

    {_provider_surface_audit_html(payload)}

    {_authority_evidence_html(payload)}

    <section class="grid two-col">
      <div class="panel section">
        <div class="section-head">
          <h2>验证拆分</h2>
          <span>按 cite_key 合并候选与验证结果</span>
        </div>
        <div class="split">
          <table class="mini-table" id="statusBreakdown"></table>
          <table class="mini-table" id="methodBreakdown"></table>
        </div>
      </div>
      <div class="panel section">
        <div class="section-head">
          <h2>优先复核队列</h2>
          <span>默认先看最可能有问题的结果</span>
        </div>
        <div class="note-block">
          这里不是新的检索阶段，也不是最终相关性排序。它把疑似未证实、待复核、以及“高引用但主题命中弱”的条目提前，方便先人工判断这批结果到底偏在哪里。
        </div>
        <div class="problem-list" id="problemList" style="margin-top: 12px;"></div>
      </div>
    </section>

    <section style="margin-top: 18px;">
      <div class="controls">
        <input class="control" id="searchBox" type="search" placeholder="搜索标题、作者、期刊、DOI、验证细节">
        <select class="control" id="statusFilter"></select>
        <select class="control" id="sourceFilter"></select>
        <select class="control" id="methodFilter"></select>
        <select class="control" id="sortMode">
          <option value="problem">问题优先</option>
          <option value="citations">引用数从高到低</option>
          <option value="year">年份从新到旧</option>
          <option value="topic">主题命中从高到低</option>
          <option value="title">标题 A-Z</option>
        </select>
        <button class="toggle" id="flagToggle" type="button">只看标记项</button>
        <div class="filter-help">筛选器来自本次 topic-search 输出：状态来自 verification_report，候选来源来自 candidates.jsonl，验证方法 method 来自 Stage 05。标记项是本页为了人工 review 额外计算的辅助标签。</div>
      </div>
      <div class="table-wrap">
        <table class="results">
          <thead>
            <tr>
              <th class="col-status">状态</th>
              <th class="col-title">论文</th>
              <th class="col-source">候选来源</th>
              <th class="col-year">年份</th>
              <th class="col-cites">引用</th>
              <th class="col-method">验证方法</th>
              <th class="col-topic">主题命中</th>
              <th class="col-id">标识符</th>
              <th class="col-details">验证细节</th>
              <th class="col-link">链接</th>
            </tr>
          </thead>
          <tbody id="resultBody"></tbody>
        </table>
        <div class="empty" id="emptyState" hidden>当前筛选条件下没有结果。</div>
      </div>
    </section>
  </div>

  <script id="review-data" type="application/json">{data}</script>
  <script>
    const payload = JSON.parse(document.getElementById("review-data").textContent);
    const rows = payload.review.rows;
    const $ = (id) => document.getElementById(id);
    const STATUS_LABELS = {{
      verified: "已验证",
      suspicious: "待复核",
      disabled: "已禁用",
      hallucinated: "疑似未证实",
      skipped: "跳过",
      unknown: "未知",
    }};
    const METHOD_LABELS = {{
      doi: "DOI 确认",
      title_search: "题名搜索 title_search",
      arxiv_id: "arXiv ID 确认",
      openalex: "OpenAlex 确认",
      unknown: "未知方法",
    }};
    const FLAG_LABELS = {{
      "verify-review": "需人工核查验证",
      "low-topic-hint": "主题命中弱",
      "missing-abstract": "缺摘要",
      "missing-strong-id": "缺 DOI/arXiv",
      "high-citation-off-topic-risk": "高引用偏题风险",
    }};

    function fmt(value) {{
      if (value === null || value === undefined || value === "") return "-";
      if (typeof value === "number") return value.toLocaleString();
      return String(value);
    }}

    function labelStatus(value) {{
      return STATUS_LABELS[value] || value || "未知";
    }}

    function labelMethod(value) {{
      return METHOD_LABELS[value] || value || "未知方法";
    }}

    function labelFlag(value) {{
      return FLAG_LABELS[value] || value;
    }}

    function textNode(value) {{
      return document.createTextNode(fmt(value));
    }}

    function make(tag, className, text) {{
      const el = document.createElement(tag);
      if (className) el.className = className;
      if (text !== undefined) el.appendChild(textNode(text));
      return el;
    }}

    function countMapToRows(map, label, kind) {{
      const table = make("table");
      table.innerHTML = `<thead><tr><th>${{label}}</th><th>数量</th></tr></thead>`;
      const tbody = document.createElement("tbody");
      Object.entries(map).sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0])).forEach(([key, count]) => {{
        const tr = document.createElement("tr");
        const display = kind === "status" ? labelStatus(key) : kind === "method" ? labelMethod(key) : key || "unknown";
        tr.append(make("td", "", display));
        tr.append(make("td", "", count));
        tbody.append(tr);
      }});
      table.append(tbody);
      return table.innerHTML;
    }}

    function renderHeader() {{
      const run = payload.run_meta;
      const request = run.request || {{}};
      $("pageTitle").textContent = `${{request.topic || "topic"}}`;
      const meta = [
        ["Run", run.run_id],
        ["请求类型", run.request_type],
        ["运行生成时间", run.generated_at],
        ["报告生成时间", payload.generated_at],
        ["Run 目录", payload.run_root],
      ];
      $("headerMeta").replaceChildren(...meta.map(([k, v]) => {{
        const line = make("div", "meta-line");
        line.append(make("span", "", k));
        line.append(make("strong", "", v));
        return line;
      }}));
    }}

    function renderKpis() {{
      const summary = payload.verification.summary || {{}};
      const search = payload.search_meta || {{}};
      const cards = [
        ["候选论文", search.total_candidates, `${{payload.review.provider_returned_total}} 条 provider 原始返回`, ""],
        ["已验证", summary.verified, "真实性已确认", ""],
        ["待复核", summary.suspicious, "保留进入 bibliography", "warn"],
        ["疑似未证实", summary.hallucinated, "未进入 verified BibTeX", "bad"],
        ["完整性分", summary.integrity_score, "已验证 / 非跳过", ""],
        ["保留 BibTeX", payload.review.verified_bib_count, "verified + suspicious", ""],
        ["查询数", payload.queries.queries.length, `每个最多 ${{payload.queries.max_results_per_query}} 条`, ""],
      ];
      $("kpis").replaceChildren(...cards.map(([label, value, note, tone]) => {{
        const card = make("div", `panel kpi ${{tone}}`);
        card.append(make("div", "label", label));
        card.append(make("div", "value", value));
        card.append(make("div", "note", note));
        return card;
      }}));
    }}

    function renderPipeline() {{
      const stage05Desc = payload.verification.status === "disabled"
        ? "verification disabled"
        : `${{payload.verification.summary.verified}} verified，${{payload.verification.summary.suspicious}} suspicious，${{payload.verification.summary.hallucinated}} hallucinated`;
      const stages = [
        ["stage-03", "Topic 转 query", `从 topic 生成 ${{payload.queries.queries.length}} 个 query：${{payload.queries.topic}}`],
        ["stage-04", "实时 provider 检索", `${{payload.search_meta.total_candidates}} 条去重候选，并写出 BibTeX`],
        ["stage-05", "引文验证 Citation verification", stage05Desc],
      ];
      $("pipeline").replaceChildren(...stages.map(([tag, title, desc]) => {{
        const stage = make("div", "stage");
        stage.append(make("span", "tag", tag));
        stage.append(make("h3", "", title));
        stage.append(make("p", "", desc));
        return stage;
      }}));
      $("queryList").replaceChildren(...payload.queries.queries.map((query) => make("span", "chip", query)));
    }}

    function renderProviders() {{
      const providers = payload.search_meta.provider_statuses || [];
      const maxReturned = Math.max(1, ...providers.map((p) => Number(p.returned_count || 0)));
      $("providerTotal").textContent = `${{payload.review.provider_returned_total}} 条原始返回，去重后进入候选`;
      $("providers").replaceChildren(...providers.map((provider) => {{
        const panel = make("div", "provider");
        const top = make("div", "provider-top");
        const title = make("div");
        title.append(make("h3", "", provider.source_name));
        title.append(make("div", "subtle", `${{provider.status}} / 尝试 ${{provider.queries_attempted}} 个 query`));
        top.append(title);
        top.append(make("div", "count", provider.returned_count));
        panel.append(top);
        const bar = make("div", "bar");
        const fill = make("span");
        fill.style.width = `${{Math.max(4, Number(provider.returned_count || 0) / maxReturned * 100)}}%`;
        bar.append(fill);
        panel.append(bar);
        if (provider.warnings && provider.warnings.length) {{
          const list = make("ul", "warnings");
          provider.warnings.forEach((warning) => list.append(make("li", "", warning)));
          panel.append(list);
        }}
        return panel;
      }}));
    }}

    function renderBreakdowns() {{
      $("statusBreakdown").innerHTML = countMapToRows(payload.review.status_counts, "状态", "status");
      $("methodBreakdown").innerHTML = countMapToRows(payload.review.method_counts, "验证方法", "method");
    }}

    function renderProblemList() {{
      const problems = payload.review.problem_rows || [];
      $("problemList").replaceChildren(...problems.map((row) => {{
        const item = make("div", "problem");
        item.append(make("strong", "", row.title));
        item.append(make("div", "subtle", `${{labelStatus(row.status)}} / ${{row.source}} / ${{row.year || "-"}} / 引用 ${{fmt(row.citation_count)}}`));
        item.append(make("div", "flag-list"));
        const flags = item.querySelector(".flag-list");
        row.flags.forEach((flag) => flags.append(make("span", `flag ${{flag.includes("risk") || flag.includes("verify") ? "strong" : ""}}`, labelFlag(flag))));
        return item;
      }}));
    }}

    function fillSelect(id, label, values, kind) {{
      const select = $(id);
      select.replaceChildren();
      select.append(new Option(label, ""));
      [...values].sort().forEach((value) => {{
        const optionLabel = kind === "status" ? labelStatus(value) : kind === "method" ? labelMethod(value) : value || "unknown";
        select.append(new Option(optionLabel, value));
      }});
    }}

    function initControls() {{
      fillSelect("statusFilter", "全部状态", new Set(rows.map((row) => row.status)), "status");
      fillSelect("sourceFilter", "全部候选来源", new Set(rows.map((row) => row.source)), "source");
      fillSelect("methodFilter", "全部验证方法", new Set(rows.map((row) => row.method)), "method");
      ["searchBox", "statusFilter", "sourceFilter", "methodFilter", "sortMode"].forEach((id) => {{
        $(id).addEventListener("input", renderRows);
      }});
      $("flagToggle").addEventListener("click", () => {{
        $("flagToggle").classList.toggle("active");
        renderRows();
      }});
    }}

    function rowText(row) {{
      return [
        row.title,
        row.authors,
        row.venue,
        row.source,
        row.doi,
        row.arxiv_id,
        row.details,
        row.cite_key,
      ].join(" ").toLowerCase();
    }}

    function filteredRows() {{
      const query = $("searchBox").value.trim().toLowerCase();
      const status = $("statusFilter").value;
      const source = $("sourceFilter").value;
      const method = $("methodFilter").value;
      const flaggedOnly = $("flagToggle").classList.contains("active");
      const sortMode = $("sortMode").value;
      const result = rows.filter((row) => {{
        if (query && !rowText(row).includes(query)) return false;
        if (status && row.status !== status) return false;
        if (source && row.source !== source) return false;
        if (method && row.method !== method) return false;
        if (flaggedOnly && (!row.flags || row.flags.length === 0)) return false;
        return true;
      }});
      result.sort((a, b) => {{
        if (sortMode === "citations") return b.citation_count - a.citation_count || a.title.localeCompare(b.title);
        if (sortMode === "year") return Number(b.year || 0) - Number(a.year || 0) || b.citation_count - a.citation_count;
        if (sortMode === "topic") return b.topic_hit_count - a.topic_hit_count || b.title_topic_hit_count - a.title_topic_hit_count || b.citation_count - a.citation_count;
        if (sortMode === "title") return a.title.localeCompare(b.title);
        return a.sort_weight - b.sort_weight || Number(b.flags.includes("high-citation-off-topic-risk")) - Number(a.flags.includes("high-citation-off-topic-risk")) || b.citation_count - a.citation_count;
      }});
      return result;
    }}

    function td(className, child) {{
      const cell = make("td", className || "");
      if (child instanceof Node) cell.append(child);
      else cell.append(textNode(child));
      return cell;
    }}

    function renderRows() {{
      const body = $("resultBody");
      const visible = filteredRows();
      const fragment = document.createDocumentFragment();
      visible.forEach((row) => {{
        const tr = document.createElement("tr");
        const status = make("span", `status ${{row.status}}`, labelStatus(row.status));
        tr.append(td("", status));

        const title = make("div", "title-cell");
        title.append(make("strong", "", row.title));
        title.append(make("div", "subtle", row.authors || row.cite_key));
        if (row.flags.length) {{
          const flags = make("div", "flag-list");
          row.flags.forEach((flag) => flags.append(make("span", `flag ${{flag.includes("risk") || flag.includes("verify") ? "strong" : ""}}`, labelFlag(flag))));
          title.append(flags);
        }}
        tr.append(td("", title));
        tr.append(td("", row.source));
        tr.append(td("", row.year));
        tr.append(td("", row.citation_count));
        tr.append(td("", `${{labelMethod(row.method)}} / ${{row.confidence || "-"}}`));
        tr.append(td("", row.topic_hits.length ? row.topic_hits.join(", ") : "无"));
        tr.append(td("", row.identity || "-"));
        tr.append(td("", row.details || "-"));
        const linkCell = make("div");
        if (row.url) {{
          const a = make("a", "link", "打开");
          a.href = row.url;
          a.target = "_blank";
          a.rel = "noreferrer";
          linkCell.append(a);
        }} else {{
          linkCell.append(textNode("-"));
        }}
        tr.append(td("", linkCell));
        fragment.append(tr);
      }});
      body.replaceChildren(fragment);
      $("emptyState").hidden = visible.length !== 0;
    }}

    renderHeader();
    renderKpis();
    renderPipeline();
    renderProviders();
    renderBreakdowns();
    renderProblemList();
    initControls();
    renderRows();
  </script>
</body>
</html>
"""


def render(run_root: Path, output: Path | None = None) -> Path:
    run_root = run_root.resolve()
    output = (output or (run_root / "review.html")).resolve()
    payload = _build_payload(run_root)
    output.write_text(_render_html(payload), encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Render an offline topic-search review HTML.")
    parser.add_argument("run_root", type=Path, help="Path to topic_search_runs/<run_id>")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output HTML path. Defaults to <run_root>/review.html.",
    )
    args = parser.parse_args()
    output = render(args.run_root, args.output)
    print(output)


if __name__ == "__main__":
    main()
