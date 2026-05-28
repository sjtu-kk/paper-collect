# Derived from AutoResearchClaw (MIT License). Copyright (c) 2026 Aiming Lab.
"""Semantic Scholar API client."""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from paper_collect.literature.errors import ProviderBlocked
from paper_collect.literature.models import Author, Paper

_BASE_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
_FIELDS = "paperId,title,abstract,year,venue,citationCount,authors,externalIds,url"
_MAX_PER_REQUEST = 100
_TIMEOUT_SEC = 30


def search_semantic_scholar(
    query: str,
    *,
    limit: int = 20,
    year_min: int = 0,
    api_key: str = "",
) -> list[Paper]:
    limit = max(1, min(limit, _MAX_PER_REQUEST))
    params: dict[str, str] = {
        "query": query,
        "limit": str(limit),
        "fields": _FIELDS,
    }
    if year_min > 0:
        params["year"] = f"{year_min}-"
    url = f"{_BASE_URL}?{urllib.parse.urlencode(params)}"
    headers = {
        "User-Agent": "paper-collect/0.1; semantic scholar search",
        "Accept": "application/json",
    }
    if api_key:
        headers["x-api-key"] = api_key
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT_SEC) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        if exc.code == 429:
            raise ProviderBlocked("semantic_scholar", "rate_limited", "Semantic Scholar rate limited") from exc
        if exc.code in {401, 403} and not api_key:
            raise ProviderBlocked("semantic_scholar", "missing_api_key", "Semantic Scholar API key missing") from exc
        raise ProviderBlocked("semantic_scholar", "provider_error", f"HTTP {exc.code}") from exc
    except urllib.error.URLError as exc:
        raise ProviderBlocked("semantic_scholar", "network_failed", str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise ProviderBlocked("semantic_scholar", "provider_error", str(exc)) from exc

    if not isinstance(payload, dict):
        raise ProviderBlocked("semantic_scholar", "provider_error", "non-object JSON payload")
    records = payload.get("data", [])
    if not isinstance(records, list):
        return []
    papers: list[Paper] = []
    for item in records:
        try:
            papers.append(_parse_s2_paper(item))
        except Exception:  # noqa: BLE001
            continue
    return papers


def _parse_s2_paper(item: dict[str, Any]) -> Paper:
    external_ids = item.get("externalIds") or {}
    authors = []
    for author in item.get("authors", []) or []:
        name = str(author.get("name") or "").strip()
        if name:
            authors.append(Author(name=name))
    doi = _clean_doi(str(external_ids.get("DOI", "") or ""))
    arxiv_id = str(external_ids.get("ArXiv", "") or "")
    paper_id = str(item.get("paperId") or doi or arxiv_id or item.get("title") or "")
    return Paper(
        paper_id=paper_id,
        title=str(item.get("title") or ""),
        authors=tuple(authors),
        year=int(item.get("year") or 0),
        abstract=str(item.get("abstract") or ""),
        venue=str(item.get("venue") or ""),
        citation_count=int(item.get("citationCount") or 0),
        doi=doi,
        arxiv_id=arxiv_id,
        url=str(item.get("url") or ""),
        source="semantic_scholar",
    )


def _clean_doi(value: str) -> str:
    doi = value.strip()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        if doi.casefold().startswith(prefix):
            return doi[len(prefix) :]
    return doi
