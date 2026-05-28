# Derived from AutoResearchClaw (MIT License). Copyright (c) 2026 Aiming Lab.
"""Unified literature search with deduplication."""

from __future__ import annotations

import json
import re
import time
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from paper_collect.literature.arxiv_client import search_arxiv
from paper_collect.literature.crossref_client import search_crossref
from paper_collect.literature.dblp_client import dblp_query_summary, search_dblp
from paper_collect.literature.doaj_client import search_doaj
from paper_collect.literature.errors import ProviderBlocked
from paper_collect.literature.models import Author, Paper
from paper_collect.literature.openalex_client import search_openalex
from paper_collect.literature.pubmed_client import search_pubmed
from paper_collect.literature.semantic_scholar import search_semantic_scholar
from paper_collect import provider_registry


@dataclass(frozen=True)
class SearchBatchResult:
    queries_used: list[str]
    papers: list[Paper]
    provider_statuses: list[dict[str, Any]]
    status: str
    provider_result_rows: list[dict[str, Any]] = field(default_factory=list)
    real_search: bool = True

    @property
    def total_candidates(self) -> int:
        return len(self.papers)

    @property
    def bibtex_entries(self) -> int:
        return len(self.papers)

    def to_search_meta(self) -> dict[str, Any]:
        return {
            "real_search": self.real_search,
            "queries_used": list(self.queries_used),
            "provider_statuses": list(self.provider_statuses),
            "total_candidates": self.total_candidates,
            "bibtex_entries": self.bibtex_entries,
            "status": self.status,
        }


def _papers_to_dicts(papers: list[Paper]) -> list[dict[str, object]]:
    return [asdict(paper) for paper in papers]


def _as_int(value: object, default: int = 0) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return default
    return default


def _dicts_to_papers(dicts: list[dict[str, object]]) -> list[Paper]:
    papers: list[Paper] = []
    for item in dicts:
        try:
            authors_raw = item.get("authors", ())
            if not isinstance(authors_raw, list):
                authors_raw = []
            authors = tuple(
                Author(
                    name=str(cast_item.get("name", "")),
                    affiliation=str(cast_item.get("affiliation", "")),
                )
                for cast_item in authors_raw
                if isinstance(cast_item, dict)
            )
            papers.append(
                Paper(
                    paper_id=str(item["paper_id"]),
                    title=str(item["title"]),
                    authors=authors,
                    year=_as_int(item.get("year", 0), 0),
                    abstract=str(item.get("abstract", "")),
                    venue=str(item.get("venue", "")),
                    citation_count=_as_int(item.get("citation_count", 0), 0),
                    doi=str(item.get("doi", "")),
                    arxiv_id=str(item.get("arxiv_id", "")),
                    url=str(item.get("url", "")),
                    source=str(item.get("source", "")),
                    source_metadata=item.get("source_metadata") if isinstance(item.get("source_metadata"), dict) else {},
                )
            )
        except (KeyError, TypeError, ValueError):
            continue
    return papers


def search_papers(
    query: str,
    *,
    limit: int = 20,
    sources: Sequence[str] | None = None,
    year_min: int = 0,
    deduplicate: bool = True,
    s2_api_key: str = "",
) -> SearchBatchResult:
    if sources is None:
        sources = provider_registry.active_provider_ids()
    all_papers: list[Paper] = []
    source_statuses: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    for source_name in sources:
        result = _search_source(
            source_name,
            query,
            limit=limit,
            year_min=year_min,
            s2_api_key=s2_api_key,
        )
        all_papers.extend(result["papers"])
        source_statuses.append(result["status"])
        row = {
            "source_name": source_name,
            "query_text": query,
            "status": result["status"]["status"],
            "returned_count": len(result["papers"]),
            "warnings": list(result["status"].get("warnings", [])),
            "papers": _papers_to_dicts(result["papers"]),
        }
        if result.get("compiled_query_summary"):
            row["compiled_query_summary"] = result["compiled_query_summary"]
        if result["status"].get("rate_control"):
            row["rate_control"] = result["status"]["rate_control"]
        source_rows.append(row)

    papers = _deduplicate(all_papers) if deduplicate else list(all_papers)
    overall_status = "completed" if any(item["status"] in {"completed", "completed_no_results"} for item in source_statuses) else "blocked_or_failed"
    return SearchBatchResult(
        queries_used=[query],
        papers=sorted(papers, key=lambda paper: (paper.citation_count, paper.year), reverse=True),
        provider_statuses=source_statuses,
        provider_result_rows=source_rows,
        status=overall_status,
        real_search=True,
    )


def search_papers_multi_query(
    queries: list[str],
    *,
    limit_per_query: int = 20,
    sources: Sequence[str] | None = None,
    year_min: int = 0,
    s2_api_key: str = "",
    inter_query_delay: float = 1.5,
) -> SearchBatchResult:
    if sources is None:
        sources = provider_registry.active_provider_ids()
    all_papers: list[Paper] = []
    merged_statuses: dict[str, dict[str, Any]] = {}
    queries_used: list[str] = []
    provider_result_rows: list[dict[str, Any]] = []
    for index, query in enumerate(queries):
        if index > 0 and inter_query_delay > 0:
            time.sleep(inter_query_delay)
        queries_used.append(query)
        result = search_papers(
            query,
            limit=limit_per_query,
            sources=sources,
            year_min=year_min,
            deduplicate=False,
            s2_api_key=s2_api_key,
        )
        all_papers.extend(result.papers)
        provider_result_rows.extend(result.provider_result_rows)
        for source_status in result.provider_statuses:
            source_name = str(source_status["source_name"])
            merged = merged_statuses.setdefault(
                source_name,
                {
                    "source_name": source_name,
                    "status": source_status["status"],
                    "returned_count": 0,
                    "queries_attempted": 0,
                    "warnings": [],
                },
            )
            merged["queries_attempted"] += 1
            merged["returned_count"] += int(source_status.get("returned_count", 0))
            merged["warnings"] = sorted(set(merged["warnings"]) | set(source_status.get("warnings", [])))
            if source_status.get("rate_control"):
                merged["rate_control"] = source_status["rate_control"]
            merged["status"] = _merge_status(str(merged["status"]), str(source_status["status"]))

    papers = _deduplicate(all_papers)
    papers.sort(key=lambda paper: (paper.citation_count, paper.year), reverse=True)
    provider_statuses = [merged_statuses[source_name] for source_name in sources if source_name in merged_statuses]
    overall_status = "completed" if any(status["status"] in {"completed", "completed_no_results"} for status in provider_statuses) else "blocked_or_failed"
    return SearchBatchResult(
        queries_used=queries_used,
        papers=papers,
        provider_statuses=provider_statuses,
        provider_result_rows=provider_result_rows,
        status=overall_status,
        real_search=True,
    )


def papers_to_bibtex(papers: Sequence[Paper]) -> str:
    entries = [paper.to_bibtex() for paper in papers]
    return "\n\n".join(entries) + ("\n" if entries else "")


def _search_source(
    source_name: str,
    query: str,
    *,
    limit: int,
    year_min: int,
    s2_api_key: str,
) -> dict[str, Any]:
    try:
        compiled_query_summary: dict[str, Any] = {}
        if source_name == "openalex":
            papers = search_openalex(query, limit=limit, year_min=year_min)
        elif source_name == "semantic_scholar":
            papers = search_semantic_scholar(query, limit=limit, year_min=year_min, api_key=s2_api_key)
        elif source_name == "arxiv":
            papers = search_arxiv(query, limit=limit, year_min=year_min)
        elif source_name == "crossref":
            papers = search_crossref(query, limit=limit, year_min=year_min)
        elif source_name == "dblp":
            compiled_query_summary = dblp_query_summary(query)
            papers = search_dblp(query, limit=limit, year_min=year_min)
        elif source_name == "doaj":
            papers = search_doaj(query, limit=limit, year_min=year_min)
        elif source_name == "pubmed":
            papers = search_pubmed(query, limit=limit, year_min=year_min)
        else:
            raise ProviderBlocked(source_name, "unsupported_query", f"unsupported source {source_name}")
        status_value = "completed_no_results" if source_name == "dblp" and not papers else "completed"
        response = {
            "papers": papers,
            "status": {
                "source_name": source_name,
                "status": status_value,
                "returned_count": len(papers),
                "warnings": [],
            },
        }
        if compiled_query_summary:
            response["compiled_query_summary"] = compiled_query_summary
        return response
    except ProviderBlocked as exc:
        status = {
            "source_name": source_name,
            "status": exc.status,
            "returned_count": 0,
            "warnings": [exc.message] if exc.message else [exc.status],
        }
        if exc.status == "rate_limited" and source_name in {"arxiv", "semantic_scholar"}:
            status["rate_control"] = _provider_local_rate_control()
        return {
            "papers": [],
            "status": status,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "papers": [],
            "status": {
                "source_name": source_name,
                "status": "provider_error",
                "returned_count": 0,
                "warnings": [f"{type(exc).__name__}: {exc}"],
            },
        }


def _deduplicate(papers: list[Paper]) -> list[Paper]:
    seen_doi: dict[str, int] = {}
    seen_arxiv: dict[str, int] = {}
    seen_title: dict[str, int] = {}
    result: list[Paper] = []

    def update_indices(paper: Paper, index: int) -> None:
        if paper.doi:
            seen_doi[paper.doi.lower().strip()] = index
        if paper.arxiv_id:
            seen_arxiv[paper.arxiv_id.strip()] = index
        normalized_title = _normalize_title(paper.title)
        if normalized_title:
            seen_title[normalized_title] = index

    def replace_at(old: Paper, new: Paper, index: int) -> None:
        if old.doi:
            old_doi = old.doi.lower().strip()
            if old_doi != new.doi.lower().strip() and seen_doi.get(old_doi) == index:
                del seen_doi[old_doi]
        if old.arxiv_id:
            old_arxiv = old.arxiv_id.strip()
            if old_arxiv != new.arxiv_id.strip() and seen_arxiv.get(old_arxiv) == index:
                del seen_arxiv[old_arxiv]
        old_title = _normalize_title(old.title)
        new_title = _normalize_title(new.title)
        if old_title and old_title != new_title and seen_title.get(old_title) == index:
            del seen_title[old_title]
        result[index] = new
        update_indices(new, index)

    for paper in papers:
        is_duplicate = False
        doi_key = paper.doi.lower().strip()
        if doi_key and doi_key in seen_doi:
            index = seen_doi[doi_key]
            if _paper_rank(paper) > _paper_rank(result[index]):
                replace_at(result[index], paper, index)
            is_duplicate = True
        arxiv_key = paper.arxiv_id.strip()
        if not is_duplicate and arxiv_key and arxiv_key in seen_arxiv:
            index = seen_arxiv[arxiv_key]
            if _paper_rank(paper) > _paper_rank(result[index]):
                replace_at(result[index], paper, index)
            is_duplicate = True
        title_key = _normalize_title(paper.title)
        if not is_duplicate and title_key and title_key in seen_title:
            index = seen_title[title_key]
            if _paper_rank(paper) > _paper_rank(result[index]):
                replace_at(result[index], paper, index)
            is_duplicate = True
        if is_duplicate:
            continue
        index = len(result)
        update_indices(paper, index)
        result.append(paper)

    return result


def _paper_rank(paper: Paper) -> tuple[int, int]:
    return paper.citation_count, paper.year


def _normalize_title(title: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9\s]", "", title.lower()).split())


def _merge_status(current: str, incoming: str) -> str:
    priority = {
        "completed": 4,
        "completed_no_results": 4,
        "rate_limited": 3,
        "missing_api_key": 3,
        "live_client_not_configured": 3,
        "network_failed": 2,
        "timeout": 2,
        "provider_error": 2,
        "unsupported_query": 1,
        "blocked": 1,
    }
    if priority.get(incoming, 0) > priority.get(current, 0):
        return incoming
    if current == "completed" or incoming == "completed":
        return "completed"
    return current


def _provider_local_rate_control() -> dict[str, Any]:
    return {
        "provider_local": True,
        "strategy": "throttle_backoff_circuit_breaker",
        "circuit_breaker": "open_for_provider",
    }
