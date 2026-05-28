# Derived from AutoResearchClaw (MIT License). Copyright (c) 2026 Aiming Lab.
"""arXiv API client powered by the official ``arxiv`` library."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from paper_collect.literature.errors import ProviderBlocked
from paper_collect.literature.models import Author, Paper

try:
    import arxiv  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - exercised in dependency-missing environments
    arxiv = None  # type: ignore[assignment]

_MAX_PAGE_SIZE = 100
_DELAY_SECONDS = 3.1
_NUM_RETRIES = 3


def search_arxiv(
    query: str,
    *,
    limit: int = 20,
    sort_by: str = "relevance",
    year_min: int = 0,
) -> list[Paper]:
    if arxiv is None:
        raise ProviderBlocked("arxiv", "live_client_not_configured", "arxiv library not installed")

    sort_criterion = arxiv.SortCriterion.Relevance if sort_by == "relevance" else arxiv.SortCriterion.SubmittedDate
    search_query = query if query.startswith("ti:") else f'all:"{query}"'
    requested_limit = max(1, limit)
    search = arxiv.Search(query=search_query, max_results=requested_limit, sort_by=sort_criterion)
    client = arxiv.Client(
        page_size=min(requested_limit, _MAX_PAGE_SIZE),
        delay_seconds=_DELAY_SECONDS,
        num_retries=_NUM_RETRIES,
    )

    papers: list[Paper] = []
    try:
        for result in client.results(search):
            try:
                published_year = result.published.year if isinstance(result.published, datetime) else 0
                if year_min > 0 and published_year and published_year < year_min:
                    continue
                papers.append(_convert_result(result))
            except Exception:  # noqa: BLE001
                continue
    except Exception as exc:  # noqa: BLE001
        message = str(exc)
        if "429" in message or "rate" in message.casefold():
            raise ProviderBlocked("arxiv", "rate_limited", "arXiv rate limited") from exc
        if "timeout" in message.casefold():
            raise ProviderBlocked("arxiv", "timeout", "arXiv request timed out") from exc
        raise ProviderBlocked("arxiv", "provider_error", message) from exc
    return papers


def _convert_result(result: Any) -> Paper:
    title = str(getattr(result, "title", "")).strip()
    authors = tuple(Author(name=str(author.name).strip()) for author in getattr(result, "authors", []) if str(author.name).strip())
    paper_id = str(getattr(result, "entry_id", "") or getattr(result, "get_short_id", lambda: "")())
    source_url = str(getattr(result, "entry_id", "") or "")
    doi = _clean_doi(str(getattr(result, "doi", "") or ""))
    arxiv_id = _extract_arxiv_id(source_url)
    venue = str(getattr(result, "journal_ref", "") or "arXiv")
    published = getattr(result, "published", None)
    year = published.year if isinstance(published, datetime) else 0
    return Paper(
        paper_id=paper_id or arxiv_id or title,
        title=title,
        authors=authors,
        year=year,
        abstract=str(getattr(result, "summary", "") or ""),
        venue=venue,
        citation_count=0,
        doi=doi,
        arxiv_id=arxiv_id,
        url=source_url,
        source="arxiv",
    )


def _extract_arxiv_id(source_url: str) -> str:
    raw_identifier = source_url.rstrip("/").rsplit("/", 1)[-1]
    if "v" in raw_identifier:
        stem, version = raw_identifier.rsplit("v", 1)
        if version.isdigit():
            return stem
    return raw_identifier


def _clean_doi(value: str) -> str:
    doi = value.strip()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        if doi.casefold().startswith(prefix):
            return doi[len(prefix) :]
    return doi
